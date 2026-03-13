"""
Model registry.

Usage:
    from models import build_model, SUPPORTED_MODELS

    model = build_model("lstm", config, input_dim=5)
"""

from models.rnn  import RNNModel,   build_rnn
from models.lstm import LSTMModel,  build_lstm
from models.gru  import GRUModel,   build_gru
from models.lnn  import LiquidNN,   build_lnn

# ── Registry ──────────────────────────────────────────────────────────────────

SUPPORTED_MODELS = ["rnn", "lstm", "gru", "lnn"]

_BUILDERS = {
    "rnn":  build_rnn,
    "lstm": build_lstm,
    "gru":  build_gru,
    "lnn":  build_lnn,
}

_CLASSES = {
    "rnn":  RNNModel,
    "lstm": LSTMModel,
    "gru":  GRUModel,
    "lnn":  LiquidNN,
}


def build_model(arch: str, config: dict, input_dim: int):
    """
    Build a model by architecture name.

    Args:
        arch:      one of 'rnn', 'lstm', 'gru', 'lnn'
        config:    full config dict (from config.yaml)
        input_dim: number of input features (determined by data pipeline)

    Returns:
        nn.Module with interface (batch, seq_len, input_dim) -> (batch, forecast_horizon)

    Raises:
        ValueError: if arch is not in SUPPORTED_MODELS
    """
    arch = arch.lower()
    if arch not in _BUILDERS:
        raise ValueError(
            f"Unknown architecture '{arch}'. "
            f"Choose from: {SUPPORTED_MODELS}"
        )
    return _BUILDERS[arch](config, input_dim)


def get_model_class(arch: str):
    """Return the model class for a given architecture name."""
    arch = arch.lower()
    if arch not in _CLASSES:
        raise ValueError(f"Unknown architecture '{arch}'.")
    return _CLASSES[arch]
