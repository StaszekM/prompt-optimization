from importlib import import_module
from typing import Any

import dspy


def create_lm_from_config(config: dict[Any, Any]) -> dspy.LM:
    kind = config["kind"]

    if not isinstance(kind, str) or not kind.isidentifier():
        raise ValueError(f"Invalid LM kind: {kind!r}")

    module_name = f"{kind}_lm"
    factory_name = f"create_{kind}_lm"
    package_prefix = f"{__package__}.lm" if __package__ else "lm"

    try:
        module = import_module(f"{package_prefix}.{module_name}")
    except ModuleNotFoundError as exc:
        raise ValueError(f"Unsupported LM kind: {kind!r}") from exc

    loaded_factory_fun = getattr(module, factory_name, None)
    if not callable(loaded_factory_fun):
        raise TypeError(
            f"LM module {module.__name__!r} does not define callable {factory_name!r}"
        )

    return loaded_factory_fun(  # type: ignore
        **config.get("params", {}),
        **config.get("untracked_params", {}),
    )
