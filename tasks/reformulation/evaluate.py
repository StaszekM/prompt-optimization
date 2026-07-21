import os
import pickle

import dspy
from evaluators.word_coverage_score import word_coverage_metric

from src.clarin_lm import create_clarin_lm
from tasks.reformulation.reformulators.reformulator import VanillaReformulator

if __name__ == "__main__":
    examples: list[dspy.Example]
    examples_location = os.path.join(
        os.path.dirname(__file__), "out/reformulation_examples.pkl"
    )
    eval_location = os.path.join(os.path.dirname(__file__), "out/eval_results.csv")

    with open(examples_location, "rb") as f:
        examples = pickle.load(f)

    evaluate = dspy.Evaluate(
        devset=examples,
        metric=word_coverage_metric,
        display_progress=True,
        save_as_csv=eval_location,
        provide_traceback=True,
    )

    reformulator = VanillaReformulator()

    model = create_clarin_lm(model_name="wcss-gpt-oss-20b")
    with dspy.context(lm=model, provide_traceback=True):
        score = evaluate(reformulator)

    print(model.inspect_history(n=1))
