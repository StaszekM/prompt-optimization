import os
import pickle
from typing import Literal

import dspy
import dvc
import dvc.api
from wandb_tracking import tracked_stage

from src.create_lm_from_config import create_lm_from_config
from tasks.koda_reformulation.metrics.aggregated_metric import AggregatedMetric
from tasks.reformulation.reformulators.reformulator import VanillaReformulator

evaluation_variant = Literal["before", "after"]


def main(variant: evaluation_variant):
    params = dvc.api.params_show("./params.yaml")
    examples: list[dspy.Example]
    examples_location = os.path.join(
        os.path.dirname(__file__), "out/reformulation_eval.pkl"
    )
    eval_location = os.path.join(
        os.path.dirname(__file__), f"out/eval_results_{variant}.csv"
    )

    model_config = params["generator_llm"]
    stage_name = "evaluate" if variant == "before" else "evaluate-after"
    with tracked_stage(
        stage_name,
        params={
            "variant": variant,
            "generator_llm": model_config,
        },
    ) as tracking:
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

        print(f"Using model: {model_config}")
        model = create_lm_from_config(model_config)
        with dspy.context(lm=model, provide_traceback=True):
            result = evaluate(
                reformulator, callback_metadata={"metric_key": f"eval-{variant}"}
            )

        metrics = {
            f"evaluation/{variant}/score": result.score,
            f"evaluation/{variant}/example_count": len(examples),
        }
        tracking.mark_succeeded(metrics)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["before", "after"], required=True)
    args = parser.parse_args()
    main(variant=args.variant)
