"""
pipeline/trainer.py

Trainer class:
  - Train loop  (MSE loss, Adam optimizer, gradient clipping)
  - Val loop    (tracked every epoch)
  - ReduceLROnPlateau scheduler
  - Early stopping
  - Best checkpoint  ->  artifacts/{arch}/model.pth
  - Training log     ->  artifacts/{arch}/training_log.csv
"""

from __future__ import annotations

import csv
import os
import time

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader


class Trainer:

    def __init__(
        self,
        model:          nn.Module,
        train_loader:   DataLoader,
        val_loader:     DataLoader,
        config:         dict,
        arch:           str,
        device:         torch.device,
    ):
        self.model        = model.to(device)
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.config       = config
        self.arch         = arch
        self.device       = device

        # hyperparams
        train_cfg      = config["training"]
        self.epochs    = train_cfg["epochs"]
        self.patience  = train_cfg["early_stopping_patience"]
        self.lr        = train_cfg["lr"]

        # output paths
        self.out_dir         = os.path.join(config["artifacts"]["base_dir"], arch)
        os.makedirs(self.out_dir, exist_ok=True)
        self.checkpoint_path = os.path.join(self.out_dir, "model.pth")
        self.log_path        = os.path.join(self.out_dir, "training_log.csv")

        # optimizer / loss / scheduler
        self.criterion = nn.MSELoss()
        self.optimizer = Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5
        )

        # state
        self.best_val_loss     = float("inf")
        self.epochs_no_improve = 0
        self.history           = []

    # ── epoch passes ──────────────────────────────────────────────────────────

    def _train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0
        for X, y in self.train_loader:
            X, y = X.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            pred = self.model(X)
            loss = self.criterion(pred, y)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            total_loss += loss.item() * X.size(0)
        return total_loss / len(self.train_loader.dataset)

    def _val_epoch(self) -> float:
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for X, y in self.val_loader:
                X, y = X.to(self.device), y.to(self.device)
                pred = self.model(X)
                loss = self.criterion(pred, y)
                total_loss += loss.item() * X.size(0)
        return total_loss / len(self.val_loader.dataset)

    # ── main run ──────────────────────────────────────────────────────────────

    def run(self) -> dict:
        """
        Full training run with early stopping.

        Returns:
            dict: arch, best_val_loss, epochs_trained, checkpoint_path, log_path
        """
        print(f"\n[Trainer]  arch={self.arch.upper()}  device={self.device}  "
              f"epochs={self.epochs}  lr={self.lr}")
        print(f"  output -> {self.out_dir}")
        print("-" * 60)

        # init CSV log
        log_fields = ["epoch", "train_loss", "val_loss", "lr", "elapsed_s"]
        with open(self.log_path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=log_fields).writeheader()

        t0 = time.time()

        for epoch in range(1, self.epochs + 1):
            train_loss = self._train_epoch()
            val_loss   = self._val_epoch()
            current_lr = self.optimizer.param_groups[0]["lr"]
            elapsed    = round(time.time() - t0, 1)

            self.scheduler.step(val_loss)

            row = {
                "epoch":      epoch,
                "train_loss": round(train_loss, 6),
                "val_loss":   round(val_loss,   6),
                "lr":         current_lr,
                "elapsed_s":  elapsed,
            }
            self.history.append(row)
            with open(self.log_path, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=log_fields).writerow(row)

            if epoch % 5 == 0 or epoch == 1:
                print(f"  epoch {epoch:>4d}/{self.epochs}  "
                      f"train={train_loss:.5f}  val={val_loss:.5f}  "
                      f"lr={current_lr:.6f}")

            # checkpoint on improvement
            if val_loss < self.best_val_loss:
                self.best_val_loss     = val_loss
                self.epochs_no_improve = 0
                self._save_checkpoint(epoch)
            else:
                self.epochs_no_improve += 1

            # early stopping
            if self.epochs_no_improve >= self.patience:
                print(f"\n  Early stopping at epoch {epoch} "
                      f"(no improvement for {self.patience} epochs)")
                break

        total_time = round(time.time() - t0, 1)
        print(f"\n  Done.  best_val_loss={self.best_val_loss:.6f}  time={total_time}s")
        print(f"  Checkpoint -> {self.checkpoint_path}")

        return {
            "arch":            self.arch,
            "best_val_loss":   self.best_val_loss,
            "epochs_trained":  len(self.history),
            "checkpoint_path": self.checkpoint_path,
            "log_path":        self.log_path,
        }

    # ── checkpoint helpers ────────────────────────────────────────────────────

    def _save_checkpoint(self, epoch: int):
        torch.save(
            {
                "epoch":      epoch,
                "arch":       self.arch,
                "state_dict": self.model.state_dict(),
                "val_loss":   self.best_val_loss,
                "config":     self.config,
            },
            self.checkpoint_path,
        )

    @staticmethod
    def load_checkpoint(
        path: str,
        model: nn.Module,
        device: torch.device,
    ) -> dict:
        """Load checkpoint into model in-place. Returns full checkpoint dict."""
        ckpt = torch.load(path, map_location=device)
        model.load_state_dict(ckpt["state_dict"])
        return ckpt
