from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Sequence

from wandb_tracking import (
    pipeline_tracking_run,
    read_stage_payloads,
    summarize_stage_payloads,
)


def _split_command(argv: Sequence[str]) -> list[str]:
    if "--" in argv:
        separator = argv.index("--")
        command = list(argv[separator + 1 :])
    else:
        command = list(argv)

    if not command:
        return ["dvc", "exp", "run", "-f"]
    return command


def main(argv: Sequence[str] | None = None) -> int:  # pyright: ignore[reportReturnType]
    command = _split_command(argv or sys.argv[1:])

    with pipeline_tracking_run(command) as tracking:
        pipeline_id = tracking.wandb_run_id or tracking.local_run_id
        env = os.environ.copy()
        env["DVC_WANDB_PIPELINE_RUN_ID"] = pipeline_id
        env["DVC_WANDB_GROUP"] = tracking.group

        if tracking.active:
            env["DVC_WANDB_PARENT_ACTIVE"] = "1"
        elif tracking.settings.enabled == "auto":
            env["DVC_WANDB_ENABLED"] = "false"

        completed = subprocess.run(command, env=env)  # noqa: PLW1510
        stage_payloads = read_stage_payloads(pipeline_id)
        summary = {
            "pipeline/dvc_exit_code": completed.returncode,
            **summarize_stage_payloads(stage_payloads),
        }
        tracking.log_metrics(summary)
        tracking.set_summary(
            {
                **summary,
                "pipeline/stages": stage_payloads,
            }
        )
        return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
