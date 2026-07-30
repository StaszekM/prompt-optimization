# W&B Instrumentation for Koda Reformulation

This subproject uses W&B as optional observability around a DVC-owned pipeline.
DVC remains the source of truth for execution, caching, dependencies, params,
models, CSVs, and generated files. W&B records what happened, but it should not
be required for reproducibility unless you explicitly opt into required tracking.

## Mental Model

- Run the pipeline from `tasks/koda_reformulation`.
- Use `run_pipeline_with_wandb.py` as the canonical entrypoint for tracked DVC
  runs.
- The wrapper creates one parent W&B pipeline run when W&B is active.
- Python stage modules use `tracked_stage(...)` internally.
- A stage creates a W&B stage run only when DVC actually executes that stage.
- If DVC restores a stage from cache, no new stage run is created; the parent
  run reads the restored sidecar and classifies the stage as `reused`.
- If W&B is disabled or unavailable in `auto` mode, DVC still runs normally.

## Setup

Install dependencies from the repository root after the W&B migration:

```bash
uv sync
```

Activate the virtualenv before running local commands:

```bash
source .venv/bin/activate
```

For online W&B tracking, authenticate once:

```bash
wandb login
```

or set an API key in the environment:

```bash
export WANDB_API_KEY=...
```

The default W&B configuration lives in `params.yaml`:

```yaml
wandb:
  enabled: auto
  project: koda-reformulation
  entity: null
  mode: online
  tags:
    - dvc
    - koda_reformulation
```

## User-Facing Environment Variables

`DVC_WANDB_ENABLED` controls whether this instrumentation is active:

- `false`: never initialize W&B; all tracking calls become no-ops.
- `auto`: try W&B, but fall back to no-op if credentials, network, or W&B setup
  are unavailable.
- `true`: require W&B; initialization errors fail the run early.

W&B SDK variables can override `params.yaml`:

- `WANDB_PROJECT`: W&B project name.
- `WANDB_ENTITY`: W&B team or user entity.
- `WANDB_MODE`: W&B mode, usually `online`, `offline`, or `disabled`.
- `WANDB_API_KEY`: API key for online tracking.

Optional noise and overhead controls:

- `WANDB_CONSOLE=off`: reduce console capture.
- `WANDB_SILENT=true`: reduce W&B SDK output.
- `WANDB_DISABLE_CODE=true`: skip W&B code saving/diff metadata.

## Internal Environment Variables

These are set by `run_pipeline_with_wandb.py`; do not set them manually for
normal use:

- `DVC_WANDB_PIPELINE_RUN_ID`: parent pipeline run id passed to stages.
- `DVC_WANDB_GROUP`: W&B group shared by parent and stage runs.
- `DVC_WANDB_PARENT_ACTIVE`: tells stages that the parent W&B run initialized.

## Running the Pipeline

Always change into the subproject first:

```bash
cd tasks/koda_reformulation
```

Cheap offline smoke test:

```bash
DVC_WANDB_ENABLED=true WANDB_MODE=offline PYTHONPATH=../.. \
  python -m run_pipeline_with_wandb -- \
  dvc repro create_examples
```

Full online experiment run:

```bash
DVC_WANDB_ENABLED=true PYTHONPATH=../.. \
  python -m run_pipeline_with_wandb -- \
  dvc exp run -f
```

Tracked reproduction instead of an experiment:

```bash
DVC_WANDB_ENABLED=true PYTHONPATH=../.. \
  python -m run_pipeline_with_wandb -- \
  dvc repro
```

Run with W&B fully disabled:

```bash
DVC_WANDB_ENABLED=false PYTHONPATH=../.. \
  python -m run_pipeline_with_wandb -- \
  dvc repro create_examples
```

Sync offline runs later:

```bash
wandb sync wandb/offline-run-*
```

## Adding Tracking to a New DVC Stage

The DVC stage name and `tracked_stage(...)` name must match.

1. Add `wandb_tracking.py` to the stage `deps` in `dvc.yaml`.
2. Add a sidecar output under `.pipeline-meta/` to the stage `outs`.
3. Wrap the stage body with `tracked_stage(...)`.
4. Call `tracking.mark_succeeded(...)` only after DVC-owned outputs are written.

Declared DVC params are discovered automatically from `dvc.yaml` with the stage
name. Pass `extra_params` only for runtime context that DVC does not track, such
as a CLI flag that selects a variant.

Example `dvc.yaml` stage:

```yaml
stages:
  my_stage:
    cmd: PYTHONPATH=../.. python -m my_stage
    deps:
      - my_stage.py
      - wandb_tracking.py
    params:
      - my_stage.some_param
    outs:
      - out/my_stage_output.json
      - .pipeline-meta/my_stage.json
```

Example Python stage:

```python
import dvc.api

from wandb_tracking import tracked_stage


def main() -> None:
    params = dvc.api.params_show("./params.yaml")

    with tracked_stage(
        "my_stage",
        extra_params={"runtime_flag": "example"},
    ) as tracking:
        result = run_computation(params["my_stage"])
        write_dvc_owned_output(result, "out/my_stage_output.json")

        tracking.mark_succeeded(
            {
                "my_stage/example_count": len(result),
                "my_stage/score": result.score,
            }
        )
```

For libraries that manage their own W&B run, let the library initialize W&B and
only use this instrumentation to provide consistent run metadata:

```python
with tracked_stage(
    "run_optimization",
    extra_params={"runtime_flag": "example"},
    manage_wandb_run=False,
) as tracking:
    use_wandb = tracking.can_external_wandb_start()

    optimizer = SomeLibraryOptimizer(
        use_wandb=use_wandb,
        wandb_init_kwargs=tracking.wandb_init_kwargs() if use_wandb else None,
    )
```

## Sidecars and Stage Resolution

Successful stage metadata is written to:

```text
.pipeline-meta/<stage>.json
```

Failed attempt metadata is written to:

```text
.pipeline-meta/attempts/<pipeline_run_id>/<stage>.json
```

The success sidecars are declared as DVC outputs. That means DVC cache restores
the previous producer metadata when a stage is reused.

The parent pipeline run classifies stages as:

- `succeeded`: the stage executed in this pipeline run and wrote a fresh sidecar.
- `reused`: DVC restored or kept a sidecar from a previous producer run.
- `failed`: the stage started, failed, and wrote an attempt sidecar.
- `skipped`: no sidecar was available for that stage in this run.

Sidecars include DVC-native fields so the metadata maps back to `dvc.yaml`:

- `stage`
- `status`
- `producer_run_id`
- `attempt_run_id`
- `local_attempt_id`
- `pipeline_run_id`
- `group`
- `wandb_active`
- `deps`
- `params`: DVC-declared parameter values discovered for the stage.
- `extra_params`
- `param_declarations`
- `params_error`
- `outs`
- `sidecar`
- `timestamp`
- `error`

## Performance Notes

For tiny stages, W&B run initialization and finalization can dominate runtime,
especially in online mode. For quick checks, prefer one of:

```bash
export WANDB_MODE=offline
export WANDB_CONSOLE=off
export WANDB_SILENT=true
export WANDB_DISABLE_CODE=true
```

or disable W&B entirely:

```bash
export DVC_WANDB_ENABLED=false
```

The current integration logs metrics and summaries only. DVC-owned files should
stay DVC-owned; do not use W&B artifacts for pipeline outputs unless the
ownership model is intentionally changed.

## Troubleshooting

If `DVC_WANDB_ENABLED=auto` and W&B cannot initialize, the wrapper disables W&B
for the child DVC process and the pipeline should continue.

If `DVC_WANDB_ENABLED=true`, missing credentials, import failures, or W&B
initialization errors should fail the run. Use this mode when tracking is a hard
requirement.

If DVC reports `.pipeline-meta/*.json` files as deleted, that usually means the
sidecars have not been produced or restored yet. Run the relevant DVC stage.

Run commands from `tasks/koda_reformulation`. Do not rely on `dvc --cd` from the
repository root for this subproject.

## W&B SDK Notes

The instrumentation uses `wandb.init(...)` fields such as `project`, `entity`,
`id`, `name`, `tags`, `config`, `group`, `job_type`, `mode`, and `resume`.

Useful W&B modes are:

- `online`: send data to W&B immediately.
- `offline`: write local W&B runs for later sync.
- `disabled`: make W&B calls no-op at the SDK level.

Useful W&B environment variables include `WANDB_PROJECT`, `WANDB_ENTITY`,
`WANDB_MODE`, `WANDB_API_KEY`, `WANDB_RUN_ID`, `WANDB_RESUME`,
`WANDB_CONSOLE`, `WANDB_SILENT`, and `WANDB_DISABLE_CODE`.
