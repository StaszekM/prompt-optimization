import os
import pickle
from typing import Literal

import dspy
import mlflow

from src.clarin_lm import create_clarin_lm
from tasks.koda_reformulation.metrics.aggregated_metric import AggregatedMetric
from tasks.reformulation.reformulators.reformulator import VanillaReformulator

exp_name = os.environ.get("DVC_EXP_NAME")
if exp_name:
    mlflow.set_tracking_uri("sqlite:///out/mlflow.db")
    # Enable autologging with all features
    mlflow.dspy.autolog(  # type: ignore
        log_compiles=True,  # Track optimization process
        log_evals=True,  # Track evaluation results
        log_traces_from_compile=True,  # Track program traces during optimization
    )
    mlflow.set_experiment(f"dvc-exp-{exp_name}")

evaluation_variant = Literal["before", "after"]


def main(variant: evaluation_variant):
    examples: list[dspy.Example]
    examples_location = os.path.join(
        os.path.dirname(__file__), "out/reformulation_eval.pkl"
    )
    eval_location = os.path.join(
        os.path.dirname(__file__), f"out/eval_results_{variant}.csv"
    )

    with open(examples_location, "rb") as f:
        examples = pickle.load(f)

    metric = AggregatedMetric(
        train_location="./data/convos.jsonl",
        val_location="./data/rag_paraphrase_eval.jsonl",
    )

    evaluate = dspy.Evaluate(
        devset=examples,
        metric=metric,
        display_progress=True,
        save_as_csv=eval_location,
        provide_traceback=True,
    )

    reformulator = VanillaReformulator()
    if variant == "after":
        reformulator.load("./out/reform_bot_optimized.json")

    model = create_clarin_lm(model_name="gemma-4-31b-it")
    with dspy.context(lm=model, provide_traceback=True):
        evaluate(reformulator, callback_metadata={"metric_key": f"eval-{variant}"})


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["before", "after"], required=True)
    args = parser.parse_args()
    main(variant=args.variant)
