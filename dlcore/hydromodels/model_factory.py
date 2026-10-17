"""Create one of the 15 supported models without a module-level registry."""

from __future__ import annotations

from dlcore.hydromodels.cardformer import CARDformerRunoffModel
from dlcore.hydromodels.crossformer import CrossformerRunoffModel
from dlcore.hydromodels.dlinear import DLinearRunoffModel
from dlcore.hydromodels.etsformer import ETSformerRunoffModel
from dlcore.hydromodels.informer import InformerRunoffModel
from dlcore.hydromodels.itransformer import ITransformerRunoffModel
from dlcore.hydromodels.model_config import ModelConfig
from dlcore.hydromodels.pyraformer import PyraformerRunoffModel
from dlcore.hydromodels.recurrent import RecurrentRunoffModel
from dlcore.hydromodels.reformer import ReformerRunoffModel
from dlcore.hydromodels.tcn import TCNRunoffModel
from dlcore.hydromodels.transformer import TransformerRunoffModel


def available_model_names() -> tuple[str, ...]:
    return (
        "dlinear",
        "itransformer",
        "lstm",
        "informer",
        "reformer",
        "transformer",
        "etsformer",
        "pyraformer",
        "crossformer",
        "cardformer",
        "bilstm",
        "gru",
        "bigru",
        "rnn",
        "tcn",
    )


def create_model(config: ModelConfig):
    config.validate()
    name = config.model_name.lower()
    constructors = {
        "dlinear": lambda: DLinearRunoffModel(config),
        "itransformer": lambda: ITransformerRunoffModel(config),
        "lstm": lambda: RecurrentRunoffModel(config, "lstm", False),
        "informer": lambda: InformerRunoffModel(config),
        "reformer": lambda: ReformerRunoffModel(config),
        "transformer": lambda: TransformerRunoffModel(config),
        "etsformer": lambda: ETSformerRunoffModel(config),
        "pyraformer": lambda: PyraformerRunoffModel(config),
        "crossformer": lambda: CrossformerRunoffModel(config),
        "cardformer": lambda: CARDformerRunoffModel(config),
        "bilstm": lambda: RecurrentRunoffModel(config, "lstm", True),
        "gru": lambda: RecurrentRunoffModel(config, "gru", False),
        "bigru": lambda: RecurrentRunoffModel(config, "gru", True),
        "rnn": lambda: RecurrentRunoffModel(config, "rnn", False),
        "tcn": lambda: TCNRunoffModel(config),
    }
    if name not in constructors:
        raise ValueError(
            f"Unknown model '{config.model_name}'. Available: {', '.join(constructors)}"
        )
    return constructors[name]()


def count_trainable_parameters(model) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))
