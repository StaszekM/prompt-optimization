import os

import dspy


def create_clarin_lm(
    model_name: str, temperature=0.5, max_tokens=8000, cache=False, num_retries=5
) -> dspy.LM:
    return dspy.LM(
        f"openai/{model_name}",
        num_retries=num_retries,
        api_key=os.environ["CLARIN_API_KEY"],
        api_base="https://services.clarin-pl.eu/api/v1/oapi",
        model_type="chat",
        temperature=temperature,
        max_tokens=max_tokens,
        cache=cache,
    )
