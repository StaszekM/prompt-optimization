from __future__ import annotations

import json
import os
import subprocess
import uuid
from contextlib import AbstractContextManager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Self

import wandb
import yaml

PIPELINE_META_DIR = Path(".pipeline-meta")
SIDE_CAR_SUFFIX = ".json"
ENV_WANDB_ENABLED = "DVC_WANDB_ENABLED"
ENV_PIPELINE_RUN_ID = "DVC_WANDB_PIPELINE_RUN_ID"
ENV_GROUP = "DVC_WANDB_GROUP"
ENV_PARENT_ACTIVE = "DVC_WANDB_PARENT_ACTIVE"
ATTEMPTS_DIR = PIPELINE_META_DIR / "attempts"


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def json_safe(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, list | tuple | set):
        return [json_safe(v) for v in value]
    return str(value)


def load_params() -> dict[str, Any]:
    import dvc.api

    params = dvc.api.params_show("./params.yaml")
    if isinstance(params, dict):
        return params
    raise TypeError("Params is not a dict")


def _dvc_path(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, dict) and value:
        return str(next(iter(value)))
    return None


def _read_dvc_yaml() -> dict[str, Any]:
    dvc_yaml_path = Path("dvc.yaml")
    if not dvc_yaml_path.exists():
        return {}
    loaded = yaml.safe_load(dvc_yaml_path.read_text())
    return loaded if isinstance(loaded, dict) else {}


def read_dvc_stage_metadata() -> dict[str, dict[str, Any]]:
    metadata: dict[str, dict[str, Any]] = {}
    stages = _read_dvc_yaml().get("stages", {})
    if not isinstance(stages, dict):
        return metadata

    for stage_name, stage_config in stages.items():
        if not isinstance(stage_config, dict):
            continue

        outs = stage_config.get("outs", [])
        if not isinstance(outs, list):
            outs = []

        sidecar_path = None
        for output in outs:
            path = _dvc_path(output)
            if (
                path
                and Path(path).parent == PIPELINE_META_DIR
                and path.endswith(SIDE_CAR_SUFFIX)
            ):
                sidecar_path = path
                break

        if sidecar_path:
            metadata[str(stage_name)] = {
                "deps": json_safe(stage_config.get("deps", [])),
                "params": json_safe(stage_config.get("params", [])),
                "outs": json_safe(outs),
                "sidecar": sidecar_path,
            }
    return metadata


def _env_or_param(env_name: str, value: Any) -> Any:
    return os.environ.get(env_name, value)


def _parse_enabled(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    normalized = str(value or "auto").strip().lower()
    if normalized in {"1", "yes", "true", "on"}:
        return "true"
    if normalized in {"0", "no", "false", "off"}:
        return "false"
    if normalized == "auto":
        return "auto"
    raise ValueError("wandb.enabled must be one of: true, false, auto")


@dataclass(frozen=True)
class WandbSettings:
    enabled: str
    project: str | None
    entity: str | None
    mode: str
    tags: list[str]

    @property
    def required(self) -> bool:
        return self.enabled == "true"

    @property
    def disabled(self) -> bool:
        return self.enabled == "false" or self.mode == "disabled"


def get_wandb_settings(params: dict[str, Any] | None = None) -> WandbSettings:
    params = params or load_params()
    wandb_params = params.get("wandb", {})
    if not isinstance(wandb_params, dict):
        wandb_params = {}

    project = _env_or_param("WANDB_PROJECT", wandb_params.get("project"))
    entity = _env_or_param("WANDB_ENTITY", wandb_params.get("entity"))
    tags = wandb_params.get("tags", ["dvc"])
    if isinstance(tags, str):
        tags = [tags]

    return WandbSettings(
        enabled=_parse_enabled(
            _env_or_param(ENV_WANDB_ENABLED, wandb_params.get("enabled", "auto"))
        ),
        project=str(project) if project else None,
        entity=str(entity) if entity else None,
        mode=str(_env_or_param("WANDB_MODE", wandb_params.get("mode", "online"))),
        tags=[str(tag) for tag in tags],
    )


class TrackingRun(AbstractContextManager["TrackingRun"]):
    def __init__(
        self,
        *,
        run_id: str,
        name: str,
        group: str,
        job_type: str,
        config: dict[str, Any] | None = None,
        manage_wandb_run: bool = True,
    ) -> None:
        self.settings = get_wandb_settings()
        self.local_run_id = run_id
        self.name = name
        self.group = group
        self.job_type = job_type
        self.config = config or {}
        self.manage_wandb_run = manage_wandb_run
        self._run: wandb.Run | None = None
        self._wandb_run_id: str | None = None
        self._started_at = utc_now()

    def __enter__(self) -> Self:
        if not self.manage_wandb_run or self.settings.disabled:
            return self

        try:
            import wandb

            self._run = wandb.init(**self.wandb_init_kwargs())
        except Exception:
            if self.settings.required:
                raise
            self._run = None

        if self._run is not None:
            self._wandb_run_id = getattr(self._run, "id", self.local_run_id)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        if self._run is not None:
            try:
                self._run.finish(exit_code=1 if exc_type else 0)
            except TypeError:
                self._run.finish()
        return False

    @property
    def active(self) -> bool:
        return self._run is not None

    @property
    def wandb_run_id(self) -> str | None:
        return self._wandb_run_id

    def wandb_init_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "id": self.local_run_id,
            "name": self.name,
            "group": self.group,
            "job_type": self.job_type,
            "mode": self.settings.mode,
            "config": json_safe(self.config),
            "tags": self.settings.tags,
            "resume": "allow",
        }
        if self.settings.project:
            kwargs["project"] = self.settings.project
        if self.settings.entity:
            kwargs["entity"] = self.settings.entity
        return kwargs

    def can_external_wandb_start(self) -> bool:
        if self.settings.disabled:
            return False
        if (
            self.settings.enabled == "auto"
            and not os.environ.get(ENV_PARENT_ACTIVE)
            and not os.environ.get("WANDB_API_KEY")
        ):
            return False
        try:
            import wandb  # noqa: F401

            return True
        except Exception:
            if self.settings.required:
                raise
            return False

    def log_metrics(self, metrics: dict[str, Any], step: int | None = None) -> None:
        if self._run is None:
            return
        self._run.log(json_safe(metrics), step=step)

    def set_summary(self, values: dict[str, Any]) -> None:
        if self._run is None:
            return
        for key, value in json_safe(values).items():
            self._run.summary[key] = value


class TrackedStage(AbstractContextManager["TrackedStage"]):
    def __init__(
        self,
        stage_name: str,
        *,
        deps: list[Any] | None = None,
        params: dict[str, Any] | None = None,
        outs: list[Any] | None = None,
        manage_wandb_run: bool = True,
    ) -> None:
        stage_metadata = read_dvc_stage_metadata()
        if stage_name not in stage_metadata:
            raise ValueError(
                f"Stage {stage_name!r} does not declare a "
                ".pipeline-meta/*.json out in dvc.yaml"
            )

        self.stage_name = stage_name
        self.deps = deps if deps is not None else stage_metadata[stage_name]["deps"]
        self.params = params or {}
        self.param_declarations = stage_metadata[stage_name]["params"]
        self.outs = outs if outs is not None else stage_metadata[stage_name]["outs"]
        self.sidecar_path = Path(stage_metadata[stage_name]["sidecar"])
        self.pipeline_run_id = os.environ.get(ENV_PIPELINE_RUN_ID)
        self.group = (
            os.environ.get(ENV_GROUP)
            or self.pipeline_run_id
            or f"stage-{uuid.uuid4().hex[:8]}"
        )
        self.local_attempt_id = f"{stage_name}-{uuid.uuid4().hex[:10]}"
        self.tracking = TrackingRun(
            run_id=self.local_attempt_id,
            name=stage_name,
            group=self.group,
            job_type=stage_name,
            config={
                "stage": stage_name,
                "pipeline_run_id": self.pipeline_run_id,
                "deps": self.deps,
                "params": self.params,
                "outs": self.outs,
            },
            manage_wandb_run=manage_wandb_run,
        )

    def __enter__(self) -> Self:
        self.tracking.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        if exc_type is not None:
            self._write_attempt(status="failed", error=str(exc_val))
        self.tracking.__exit__(exc_type, exc_val, exc_tb)
        return False

    @property
    def active(self) -> bool:
        return self.tracking.active

    @property
    def attempt_run_id(self) -> str | None:
        if not self.tracking.manage_wandb_run and self.can_external_wandb_start():
            return self.local_attempt_id
        return self.tracking.wandb_run_id

    def log_metrics(self, metrics: dict[str, Any], step: int | None = None) -> None:
        self.tracking.log_metrics(metrics, step=step)

    def set_summary(self, values: dict[str, Any]) -> None:
        self.tracking.set_summary(values)

    def wandb_init_kwargs(self) -> dict[str, Any]:
        return self.tracking.wandb_init_kwargs()

    def can_external_wandb_start(self) -> bool:
        return self.tracking.can_external_wandb_start()

    def mark_succeeded(self, metrics: dict[str, Any] | None = None) -> None:
        if metrics:
            self.log_metrics(metrics)
            self.set_summary(metrics)
        self._write_sidecar(status="succeeded")

    def _payload(self, status: str, error: str | None = None) -> dict[str, Any]:
        return json_safe(
            {
                "stage": self.stage_name,
                "status": status,
                "producer_run_id": (
                    self.attempt_run_id if status == "succeeded" else None
                ),
                "attempt_run_id": self.attempt_run_id,
                "local_attempt_id": self.local_attempt_id,
                "pipeline_run_id": self.pipeline_run_id,
                "group": self.group,
                "wandb_active": self.active,
                "deps": self.deps,
                "params": self.params,
                "param_declarations": self.param_declarations,
                "outs": self.outs,
                "sidecar": str(self.sidecar_path),
                "timestamp": utc_now(),
                "error": error,
            }
        )

    def _write_sidecar(self, status: str) -> None:
        self.sidecar_path.parent.mkdir(parents=True, exist_ok=True)
        self.sidecar_path.write_text(
            json.dumps(self._payload(status), indent=2, sort_keys=True) + "\n"
        )

    def _write_attempt(self, status: str, error: str | None = None) -> None:
        pipeline_id = self.pipeline_run_id or "standalone"
        attempt_path = ATTEMPTS_DIR / pipeline_id / f"{self.stage_name}.json"
        attempt_path.parent.mkdir(parents=True, exist_ok=True)
        attempt_path.write_text(
            json.dumps(self._payload(status, error=error), indent=2, sort_keys=True)
            + "\n"
        )


def tracked_stage(
    stage_name: str,
    *,
    deps: list[Any] | None = None,
    params: dict[str, Any] | None = None,
    outs: list[Any] | None = None,
    manage_wandb_run: bool = True,
) -> TrackedStage:
    return TrackedStage(
        stage_name,
        deps=deps,
        params=params,
        outs=outs,
        manage_wandb_run=manage_wandb_run,
    )


def pipeline_tracking_run(command: list[str]) -> TrackingRun:
    pipeline_id = f"pipeline-{uuid.uuid4().hex[:10]}"
    exp_name = os.environ.get("DVC_EXP_NAME")
    return TrackingRun(
        run_id=pipeline_id,
        name=exp_name or pipeline_id,
        group=exp_name or pipeline_id,
        job_type="pipeline",
        config={
            "command": command,
            "dvc_exp_name": exp_name,
            "git_commit": git_commit(),
        },
    )


def git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def read_stage_payloads(pipeline_id: str) -> dict[str, dict[str, Any]]:
    stage_metadata = read_dvc_stage_metadata()
    attempts_dir = ATTEMPTS_DIR / pipeline_id
    attempts: dict[str, dict[str, Any]] = {}
    if attempts_dir.exists():
        for path in attempts_dir.glob("*.json"):
            payload = json.loads(path.read_text())
            attempts[payload["stage"]] = payload

    payloads: dict[str, dict[str, Any]] = {}
    for stage, metadata in stage_metadata.items():
        path = Path(metadata["sidecar"])
        sidecar_payload: dict[str, Any] | None = None
        if path.exists():
            sidecar_payload = json.loads(path.read_text())

        attempt = attempts.get(stage)
        if attempt and attempt.get("status") == "failed":
            payloads[stage] = {**attempt, "resolution": "failed"}
        elif sidecar_payload and sidecar_payload.get("pipeline_run_id") == pipeline_id:
            payloads[stage] = {**sidecar_payload, "resolution": "succeeded"}
        elif sidecar_payload:
            payloads[stage] = {**sidecar_payload, "resolution": "reused"}
        else:
            payloads[stage] = {
                "stage": stage,
                "status": "skipped",
                "resolution": "skipped",
                "producer_run_id": None,
                "attempt_run_id": None,
                "pipeline_run_id": pipeline_id,
                "deps": metadata["deps"],
                "param_declarations": metadata["params"],
                "outs": metadata["outs"],
                "sidecar": metadata["sidecar"],
                "timestamp": utc_now(),
            }
    return payloads


def summarize_stage_payloads(payloads: dict[str, dict[str, Any]]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for payload in payloads.values():
        resolution = str(payload.get("resolution", payload.get("status", "unknown")))
        counts[resolution] = counts.get(resolution, 0) + 1
    return {
        "pipeline/stage_count": len(payloads),
        **{f"pipeline/stages_{key}": value for key, value in counts.items()},
    }
