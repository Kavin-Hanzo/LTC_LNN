# models/__init__.py
# Model registry — single entry point for all architectures.
#
# Usage anywhere in the pipeline:
#   from models import build_model, SUPPORTED_MODELS
#   model = build_model(cfg_model, n_features=7, horizon=5)
#
# Adding a new architecture:
#   1. Create models/myarch.py with MyModel(BaseModel) + build_myarch()
#   2. Import and register below — nothing else changes

from models.base import BaseModel
from models.rnn  import RNNModel,  build_rnn
from models.gru  import GRUModel,  build_gru
from models.lstm import LSTMModel, build_lstm
from models.lnn  import LiquidNN,  build_lnn

# ── Registry ──────────────────────────────────────────────────────

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


def build_model(cfg_model, n_features: int, horizon: int) -> BaseModel:
    """
    Build a model by reading cfg_model.arch.

    Args:
        cfg_model  : ModelConfig dataclass from config.py
        n_features : number of raw indicator features (e.g. 7)
        horizon    : MIMO output steps (e.g. 5 for short-term)

    Returns:
        BaseModel subclass ready for training

    Raises:
        ValueError: if cfg_model.arch is not in SUPPORTED_MODELS
    """
    arch = cfg_model.arch.lower()
    if arch not in _BUILDERS:
        raise ValueError(
            f"Unknown arch '{arch}'. Choose from: {SUPPORTED_MODELS}"
        )
    model = _BUILDERS[arch](cfg_model, n_features, horizon)
    print(f"  [Model] {model}")
    return model


def get_model_class(arch: str):
    """Return the class (not an instance) for a given arch name."""
    arch = arch.lower()
    if arch not in _CLASSES:
        raise ValueError(f"Unknown arch '{arch}'. Choose from: {SUPPORTED_MODELS}")
    return _CLASSES[arch]
