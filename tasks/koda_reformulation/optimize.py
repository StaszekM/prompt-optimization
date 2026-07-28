import pickle

import dspy
import dvc
import dvc.api
from dspy.primitives import Example

from src.create_lm_from_config import create_lm_from_config
from tasks.koda_reformulation.evaluate import AggregatedMetric
from tasks.koda_reformulation.maybe_setup_mlflow import maybe_setup_mlflow
from tasks.reformulation.reformulators.reformulator import VanillaReformulator

maybe_setup_mlflow()


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
        num_threads=params["gepa_num_threads"],
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
