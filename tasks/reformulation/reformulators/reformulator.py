import dspy

from .Reformulator import Reformulator


class VanillaReformulator(dspy.Module):
    def __init__(self, callbacks=None):
        self._reformulating_fn = dspy.Predict(Reformulator)

    def forward(
        self, message_history: list[dict], last_user_message: str
    ) -> dspy.Prediction:
        message_history_str = "\n\n".join(
            [
                f"Role: {message['role']}\nContent: {message['content']}"
                for message in message_history
            ]
        )
        return self._reformulating_fn(
            message_history=message_history_str, last_user_message=last_user_message
        )
