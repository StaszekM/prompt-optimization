import ast
import os
import pickle

import dvc
import dvc.api
import pandas as pd
from sklearn.model_selection import train_test_split
from wandb_tracking import tracked_stage


def _normalize_role(role: str) -> str:
    mapper = {"customer": "user", "ai_agent": "assistant"}

    normalized_value = mapper.get(role)
    if not normalized_value:
        raise ValueError("Unknown role:", role)

    return normalized_value


def to_examples(reader):
    examples: list[Example] = []
    for idx, row in reader:
        message_history = [
            {"role": _normalize_role(h["role"]), "content": h["content"]}
            for h in row["history"]
        ]
        examples.append(
            Example(
                message_history=message_history,
                last_user_message=row["user_question"],
                reformulated_message=row["reformulated question"],
            ).with_inputs("message_history", "last_user_message")
        )

    return examples


if __name__ == "__main__":
    from dspy.primitives.example import Example

    params = dvc.api.params_show("./params.yaml")["examples_creator"]

    input_location_train = os.path.join(
        os.path.dirname(__file__), "data", "convos.jsonl"
    )
    input_location_eval = os.path.join(
        os.path.dirname(__file__), "data", "rag_paraphrase_eval.jsonl"
    )
    output_location_train = os.path.join(
        os.path.dirname(__file__), "out", "reformulation_examples.pkl"
    )
    output_location_eval = os.path.join(
        os.path.dirname(__file__), "out", "reformulation_eval.pkl"
    )

    with tracked_stage(
        "create_examples",
        params=params,
    ) as tracking:
        df_train = pd.read_json(input_location_train, lines=True)
        df_eval = (
            pd.read_json(input_location_eval, lines=True)
            .assign(history=lambda df: df["history"].apply(ast.literal_eval))
            .rename(columns={"gold_paraphrase": "reformulated question"})
        )

        # df eval is so large that it would be beneficial to shrink it and extend training set
        df_eval_subsample_training, df_eval_subsample_test = train_test_split(
            df_eval, test_size=params["test_size"], random_state=params["random_state"]
        )

        train_reader = list(df_train.iterrows()) + list(
            df_eval_subsample_training.iterrows()
        )
        examples = to_examples(train_reader)

        val_reader = list(df_eval_subsample_test.iterrows())
        examples_val = to_examples(val_reader)

        with open(output_location_train, "wb") as output_file:
            pickle.dump(examples, output_file)

        with open(output_location_eval, "wb") as output_file:
            pickle.dump(examples_val, output_file)

        metrics = {
            "examples/train_count": len(train_reader),
            "examples/eval_count": len(val_reader),
        }
        tracking.mark_succeeded(metrics)
        print(
            f"Created {len(train_reader)} train examples and {len(val_reader)} val examples."
        )
