"""
Smoke test — verifies all 4 models produce correct output shapes.
Run from project root: python models/test_models.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import yaml
from models import build_model, SUPPORTED_MODELS

# ── Load config ───────────────────────────────────────────────────────────────
with open("config.yaml") as f:
    config = yaml.safe_load(f)

# ── Test params ───────────────────────────────────────────────────────────────
BATCH       = 8
SEQ_LEN     = config["data"]["window_size"]        # 30
INPUT_DIM   = len(config["data"]["features"])      # 5
HORIZON     = config["training"]["forecast_horizon"]  # 30

print(f"\nInput  shape: (batch={BATCH}, seq_len={SEQ_LEN}, input_dim={INPUT_DIM})")
print(f"Expected out: (batch={BATCH}, horizon={HORIZON})\n")
print("=" * 55)

# ── Run each model ─────────────────────────────────────────────────────────────
all_passed = True

for arch in SUPPORTED_MODELS:
    try:
        model = build_model(arch, config, input_dim=INPUT_DIM)
        model.eval()

        x = torch.randn(BATCH, SEQ_LEN, INPUT_DIM)
        with torch.no_grad():
            out = model(x)

        assert out.shape == (BATCH, HORIZON), \
            f"Shape mismatch: expected ({BATCH}, {HORIZON}), got {out.shape}"

        summary = model.model_summary()
        print(f"  [{arch.upper():4s}]  output={tuple(out.shape)}  "
              f"params={summary['trainable_params']:,}  ✓")

    except Exception as e:
        print(f"  [{arch.upper():4s}]  FAILED — {e}")
        all_passed = False

print("=" * 55)
print(f"\n{'All models passed ✓' if all_passed else 'Some models FAILED ✗'}\n")
