import os
import pickle

import dspy
import dvc
import dvc.api
import mlflow
from dspy.primitives import Example

from src.create_lm_from_config import create_lm_from_config
from tasks.koda_reformulation.evaluate import AggregatedMetric
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


def list_splitter(list_to_split, ratio):
    elements = len(list_to_split)
    middle = int(elements * ratio)
    return [list_to_split[:middle], list_to_split[middle:]]


def main():
    params = dvc.api.params_show("./params.yaml")
    backbone_lm = create_lm_from_config(params["generator_llm"])
    dspy.configure(lm=backbone_lm)

    reflection_lm = create_lm_from_config(params["reflection_llm"])

    metric = AggregatedMetric(
        train_location="./data/convos.jsonl",
        val_location="./data/rag_paraphrase_eval.jsonl",
    )

    optimizer = dspy.GEPA(
        metric=metric,
        reflection_lm=reflection_lm,
        num_threads=2,
        log_dir="./out/gepa_logs",
        **params["gepa_config"],
    )

    bot = VanillaReformulator()

    with open("out/reformulation_examples.pkl", "rb") as f:
        examples: list[Example] = pickle.load(f)

    with open("out/reformulation_eval.pkl", "rb") as f:
        eval_examples: list[Example] = pickle.load(f)

    optimized_bot = optimizer.compile(bot, trainset=examples, valset=eval_examples)

    optimized_bot.save("./out/reform_bot_optimized.json")


if __name__ == "__main__":
    main()
