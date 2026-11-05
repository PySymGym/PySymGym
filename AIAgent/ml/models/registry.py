"""The model registry: the single selection path for model classes.

Every model-selection site (the training entrypoint, the ONNX export CLI,
the coverage-validation preparator) resolves a model through this registry
by name; no dotted import paths or FQN reconstruction elsewhere. A model
joins the registry by conforming to the ``Model`` forward contract
(``ml.protocols``) — the registry tests pin that.
"""

import torch

from ml.models.NorthernPenguin.model import (
    StateModelEncoder as NorthernPenguinStateModelEncoder,
)

# Registry name -> model class.
MODELS: dict[str, type[torch.nn.Module]] = {
    "NorthernPenguin": NorthernPenguinStateModelEncoder,
}


def _known_models() -> str:
    """The registered names, comma-joined (for error messages)."""
    return ", ".join(sorted(MODELS))


def get_model_class(name: str) -> type[torch.nn.Module]:
    """The model class registered under ``name``.

    Args:
        name: The registry name of the model.

    Returns:
        The registered model class.

    Raises:
        ValueError: If ``name`` is not registered; the message lists the
            known names so a typo is self-diagnosing.
    """
    try:
        return MODELS[name]
    except KeyError:
        raise ValueError(
            f"Unknown model name {name!r}; known models: {_known_models()}"
        ) from None


def get_model_name(model: torch.nn.Module | type[torch.nn.Module]) -> str:
    """The registry name of a model instance or class.

    Matches by exact class identity: a subclass or an unregistered
    ``nn.Module`` is rejected, never mislabelled.

    Args:
        model: A model instance or class.

    Returns:
        The registry name of the model's class.

    Raises:
        ValueError: If the model's class is not registered; the message
            lists the known names.
    """
    model_class = model if isinstance(model, type) else type(model)
    for name, model_cls in MODELS.items():
        if model_class is model_cls:
            return name
    raise ValueError(
        f"Model {model_class!r} is not registered; known models: {_known_models()}"
    )


def model_names() -> list[str]:
    """The registered model names, sorted (stable iteration order)."""
    return sorted(MODELS)
