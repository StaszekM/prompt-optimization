import dspy


def create_local_lm(
    model_name: str,
    api_base: str,
    temperature=0.5,
    max_tokens=8000,
    cache=False,
    num_retries=5,
) -> dspy.LM:
    return dspy.LM(
        f"hosted_vllm/{model_name}",
        num_retries=num_retries,
        api_base=api_base,
        model_type="chat",
        temperature=temperature,
        max_tokens=max_tokens,
        cache=cache,
    )
