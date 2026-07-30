import os
import pickle

import dspy
import dvc
import dvc.api
from dspy.primitives import Example
from wandb_tracking import tracked_stage

from src.create_lm_from_config import create_lm_from_config
from tasks.koda_reformulation.evaluate import AggregatedMetric
from tasks.reformulation.reformulators.reformulator import VanillaReformulator


def list_splitter(list_to_split, ratio):
    elements = len(list_to_split)
    middle = int(elements * ratio)
    return [list_to_split[:middle], list_to_split[middle:]]


def main():
    params = dvc.api.params_show("./params.yaml")
    tracking_config = {
        "generator_llm": params["generator_llm"],
        "reflection_llm": params["reflection_llm"],
        "gepa_config": params["gepa_config"],
        "gepa_num_threads": params["gepa_num_threads"],
    }
    with tracked_stage(
        "run_optimization",
        params=tracking_config,
        manage_wandb_run=False,
    ) as tracking:
        backbone_lm = create_lm_from_config(params["generator_llm"])
        dspy.configure(lm=backbone_lm)

        reflection_lm = create_lm_from_config(params["reflection_llm"])

        metric = AggregatedMetric(
            train_location="./data/convos.jsonl",
            val_location="./data/rag_paraphrase_eval.jsonl",
        )

        use_gepa_wandb = tracking.can_external_wandb_start()
        optimizer = dspy.GEPA(
            metric=metric,
            reflection_lm=reflection_lm,
            num_threads=params["gepa_num_threads"],
            use_wandb=use_gepa_wandb,
            wandb_api_key=os.environ.get("WANDB_API_KEY"),
            wandb_init_kwargs=tracking.wandb_init_kwargs() if use_gepa_wandb else None,
            use_mlflow=False,
            **params["gepa_config"],
        )

        bot = VanillaReformulator()

        with open("out/reformulation_examples.pkl", "rb") as f:
            examples: list[Example] = pickle.load(f)

        with open("out/reformulation_eval.pkl", "rb") as f:
            eval_examples: list[Example] = pickle.load(f)

        optimized_bot = optimizer.compile(bot, trainset=examples, valset=eval_examples)
        optimized_bot.save("./out/reform_bot_optimized.json")
        tracking.mark_succeeded(
            {
                "optimization/train_examples": len(examples),
                "optimization/eval_examples": len(eval_examples),
            }
        )


if __name__ == "__main__":
    main()
