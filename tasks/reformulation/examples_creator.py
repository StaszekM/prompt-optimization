import os
import pickle

import pandas as pd

if __name__ == "__main__":
    from dspy.primitives.example import Example

    input_location = os.path.join(
        os.path.dirname(__file__), "data", "reformulation_dataset.csv"
    )
    output_location = os.path.join(
        os.path.dirname(__file__), "out", "reformulation_examples.pkl"
    )

    examples: list[Example] = []

    df = pd.read_csv(input_location)
    reader = list(df.iterrows())
    for idx, row in reader:
        message_history = [
            {"role": "user", "content": row["query"]},
            {"role": "assistant", "content": row["answer"]},
        ]
        examples.append(
            Example(
                message_history=message_history,
                last_user_message=row["query_to_reformulate"],
                reformulated_message=row["query_after_reformulation"],
                requires_reformulation=row["query_type"] == "reformulate",
            ).with_inputs("message_history", "last_user_message")
        )

    with open(output_location, "wb") as output_file:
        pickle.dump(examples, output_file)
