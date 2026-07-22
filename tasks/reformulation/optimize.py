import os
import pickle

import dspy
import mlflow
from dspy.primitives import Example

from src.clarin_lm import create_clarin_lm
from tasks.reformulation.evaluators.word_coverage_score import word_coverage_metric
from tasks.reformulation.reformulators.reformulator import VanillaReformulator

exp_name = os.environ.get("DVC_EXP_NAME")
if exp_name:
    mlflow.set_tracking_uri("sqlite///out/mlflow.db")
    # Enable autologging with all features
    mlflow.dspy.autolog(  # type: ignore
        log_compiles=True,  # Track optimization process
        log_evals=True,  # Track evaluation results
        log_traces_from_compile=True,  # Track program traces during optimization
    )
    mlflow.set_experiment(f"dvc-exp-{exp_name}")


def list_splitter(list_to_split, ratio):
    elements = len(list_to_split)
    middle = int(elements * ratio)
    return [list_to_split[:middle], list_to_split[middle:]]


def main():
    backbone_lm = create_clarin_lm(model_name="gemma-4-31b-it")
    dspy.configure(lm=backbone_lm)

    reflection_lm = create_clarin_lm(model_name="gpt-4o")

    optimizer = dspy.GEPA(
        metric=word_coverage_metric,
        reflection_lm=reflection_lm,
        auto="light",
        num_threads=2,
        log_dir="./out/gepa_logs",
    )

    bot = VanillaReformulator()

    with open("out/reformulation_examples.pkl", "rb") as f:
        examples: list[Example] = pickle.load(f)

    optimized_bot = optimizer.compile(bot, trainset=examples)

    optimized_bot.save("./out/reform_bot_optimized.json")


if __name__ == "__main__":
    main()
