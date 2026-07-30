import pickle

import dspy
from dspy.primitives import Example

from lm.clarin_lm import create_clarin_lm
from tasks.reformulation.evaluators.word_coverage_score import word_coverage_metric
from tasks.reformulation.reformulators.reformulator import VanillaReformulator


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
