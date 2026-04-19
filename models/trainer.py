# models/trainer.py
# Training loop: Adam + ReduceLROnPlateau + EarlyStopping + gradient clipping.
# Saves the best checkpoint automatically.

import os
import time
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.dataset import collate
from models.base import BaseModel


class EarlyStopping:
    def __init__(self, patience: int = 15):
        self.patience   = patience
        self.best_loss  = np.inf
        self.counter    = 0
        self.best_state = None

    def step(self, val_loss: float, model: nn.Module) -> bool:
        if val_loss < self.best_loss - 1e-6:
            self.best_loss  = val_loss
            self.counter    = 0
            self.best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
        return self.counter >= self.patience

    def restore(self, model: nn.Module):
        if self.best_state:
            model.load_state_dict(self.best_state)


def _one_epoch(model: BaseModel, loader, optimizer, criterion, device, train: bool) -> float:
    model.train(train)
    total, n = 0.0, 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in loader:
            x   = batch["x"].to(device)
            idn = batch["identity"].to(device) if model.use_identity else None
            y   = batch["y"].to(device)
            if train:
                optimizer.zero_grad()
            pred = model(x, idn)
            loss = criterion(pred, y)
            if train:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            total += loss.item()
            n     += 1
    return total / max(n, 1)


def train(model:           BaseModel,
          train_ds,
          val_ds,
          cfg_training,
          device:          torch.device,
          checkpoint_path: Optional[str] = None) -> Dict[str, List[float]]:
    """
    Train model and return loss history dict.
    """
    tr_loader = DataLoader(train_ds,
                           batch_size=cfg_training.batch_size,
                           shuffle=True, collate_fn=collate, num_workers=0)
    va_loader = DataLoader(val_ds,
                           batch_size=cfg_training.batch_size * 2,
                           shuffle=False, collate_fn=collate, num_workers=0)

    model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=cfg_training.lr,
                                 weight_decay=cfg_training.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5)
    stopper   = EarlyStopping(patience=cfg_training.patience)
    history   = {"train_loss": [], "val_loss": []}

    print(f"\n  [Train] device={device}  epochs={cfg_training.epochs}  "
          f"patience={cfg_training.patience}")
    print(f"  {'Ep':>5}  {'Train':>10}  {'Val':>10}  {'LR':>9}")
    print("  " + "─" * 40)

    t0 = time.time()
    for ep in range(1, cfg_training.epochs + 1):
        tr = _one_epoch(model, tr_loader, optimizer, criterion, device, True)
        va = _one_epoch(model, va_loader, optimizer, criterion, device, False)
        scheduler.step(va)
        history["train_loss"].append(tr)
        history["val_loss"].append(va)

        lr_now = optimizer.param_groups[0]["lr"]
        if ep % 10 == 0 or ep == 1:
            print(f"  {ep:>5}  {tr:>10.6f}  {va:>10.6f}  {lr_now:>9.2e}")

        if stopper.step(va, model):
            print(f"  Early stop @ ep {ep}  best_val={stopper.best_loss:.6f}")
            break

    stopper.restore(model)
    print(f"  Done in {time.time()-t0:.1f}s  best_val={stopper.best_loss:.6f}\n")

    if checkpoint_path:
        os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)
        # torch.save({"state": model.state_dict(), "history": history,
        #             "best_val": stopper.best_loss}, checkpoint_path)
        torch.save({"model": model, "history": history, "best_val": stopper.best_loss}, checkpoint_path)
        print(f"  [Saved] {checkpoint_path}")

    return history


# def load_checkpoint(model: BaseModel, path: str) -> dict:
#     ckpt = torch.load(path, map_location="cpu")
#     model.load_state_dict(ckpt["state"])
#     return ckpt

def load_checkpoint(path: str) -> tuple:  # Return model and ckpt dict
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = ckpt["model"]
    return model, ckpt  # Or adjust return type as needed
