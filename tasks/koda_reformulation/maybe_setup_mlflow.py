import os

import dvc.api
import mlflow


def maybe_setup_mlflow():
    exp_name = os.environ.get("DVC_EXP_NAME")
    if exp_name:
        params = dvc.api.params_show("./params.yaml")
        mlflow.set_tracking_uri(
            f"http://{params['mlflow_tracking_host']}:{params['mlflow_tracking_port']}"
        )
        # Enable autologging with all features
        mlflow.dspy.autolog(  # type: ignore
            log_compiles=True,  # Track optimization process
            log_evals=True,  # Track evaluation results
            log_traces_from_compile=True,  # Track program traces during optimization
        )
        mlflow.set_experiment(f"dvc-exp-{exp_name}")
