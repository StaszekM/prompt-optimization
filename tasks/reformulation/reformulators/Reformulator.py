import dspy


class Reformulator(dspy.Signature):
    """
    Given the conversation history, try to create a reformulation of last user message in order to land with a full-context, standalone question that can be used for semantic retrieval.
    """

    message_history: str = dspy.InputField(desc="User-assistant conversation so far")
    last_user_message: str = dspy.InputField()
    reformulated_question = dspy.OutputField(
        desc="Reformulated question, or rewritten last user message if reformulation is unnecessary"
    )
