import ast
import os
import pickle

import pandas as pd
from sklearn.model_selection import train_test_split


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

    eval_examples: list[Example] = []

    df_train = pd.read_json(input_location_train, lines=True)
    df_eval = (
        pd.read_json(input_location_eval, lines=True)
        .assign(history=lambda df: df["history"].apply(ast.literal_eval))
        .rename(columns={"gold_paraphrase": "reformulated question"})
    )

    # df eval is so large that it would be beneficial to shrink it and extend training set
    df_eval_subsample_training, df_eval_subsample_test = train_test_split(
        df_eval, test_size=0.3, random_state=34
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
