import dspy


class Reformulator(dspy.Signature):
    """
    Korzystając z historii rozmowy, sparafrazuj ostatnie pytanie użytkownika by otrzymać pełne pytanie, zawierające kompletny kontekst, które może zostać użyte do wyszukiwania semantycznego.
    """

    message_history: str = dspy.InputField(
        desc="Dotychczasowa rozmowa użytkownika z asystentem"
    )
    last_user_message: str = dspy.InputField()
    reformulated_question = dspy.OutputField(
        desc="Pytanie sparafrazowane, lub pytanie przepisane bez zmian, jeśli nie było konieczności parafrazy",
    )
