# quick_test_model_pl.py
import argparse
import os
import torch
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from config import CFG
from quick_test_data_pl import StockDataModule
from models import build_model
from models.base import BaseModel
import torchmetrics

# ── LIGHTNING MODULE ─────────────────────────────────────────────────────────

class StockForecastingModule(L.LightningModule):
    """
    Wraps the core LNN/LSTM/RNN models for Lightning training.
    Handles loss, optimization, and metric synchronization across GPUs.
    """
    def __init__(self, model: BaseModel, lr: float = 1e-3):
        super().__init__()
        self.model = model
        self.lr = lr
        self.criterion = torch.nn.MSELoss()

        # Metrics for return-space (synchronized across GPUs)
        self.train_mse = torchmetrics.MeanSquaredError()
        self.val_mse = torchmetrics.MeanSquaredError()
        self.val_mae = torchmetrics.MeanAbsoluteError()

    def forward(self, x, identity=None):
        return self.model(x, identity)

    def training_step(self, batch, batch_idx):
        x, idn, y_true = batch["x"], batch.get("identity"), batch["y"]
        y_pred = self(x, idn)
        
        loss = self.criterion(y_pred, y_true)
        
        # Log metrics
        self.train_mse(y_pred, y_true)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, idn, y_true = batch["x"], batch.get("identity"), batch["y"]
        y_pred = self(x, idn)
        
        loss = self.criterion(y_pred, y_true)
        self.val_mse(y_pred, y_true)
        self.val_mae(y_pred, y_true)
        
        self.log_dict({
            "val_loss": loss,
            "val_mse": self.val_mse,
            "val_mae": self.val_mae
        }, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch"
            }
        }

# ── MAIN EXECUTION ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Lightning Stock Model Trainer")
    parser.add_argument("--arch", choices=["lnn", "lstm", "rnn"], default="lnn")
    parser.add_argument("--variant", choices=["baseline", "stock2vec"], default="stock2vec")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--fast_dev_run", action="store_true", help="Run 1 batch to test code.")
    args = parser.parse_args()

    # 1. Initialize Data
    dm = StockDataModule(batch_size=args.batch_size)
    dm.prepare_data()
    dm.setup()

    # 2. Build Model Architecture
    use_identity = (args.variant == "stock2vec")
    
    # We pull dimensions directly from the processed DataModule
    core_model = build_model(
        arch_name=args.arch,
        cfg=CFG,
        n_features=len(dm.train_ds.samples[0]['x'][0]), # Dynamically get input dim
        horizon=CFG.model.output_dim,
        use_identity=use_identity,
        identity_dim=dm.identity_dim if use_identity else 0
    )

    model_module = StockForecastingModule(model=core_model, lr=CFG.training.lr)

    # 3. Configure Fast/Parallel Trainer
    # '16-mixed' precision and 'ddp' strategy provide the fastest training path
    trainer = L.Trainer(
        accelerator="auto",
        devices="auto",
        strategy="auto", # Set to "ddp" for multi-GPU training
        precision="16-mixed", 
        max_epochs=args.epochs,
        fast_dev_run=args.fast_dev_run,
        gradient_clip_val=1.0,
        logger=CSVLogger("logs", name=f"{args.arch}_{args.variant}"),
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=10),
            ModelCheckpoint(monitor="val_loss", filename="best_model", save_top_k=1)
        ]
    )

    # 4. Train
    print(f"\n[Trainer] Starting {args.arch.upper()} ({args.variant}) training...")
    trainer.fit(model_module, datamodule=dm)

    # 5. Final Test Evaluation
    print("\n[Trainer] Evaluating on Test Set...")
    trainer.test(model_module, datamodule=dm)

if __name__ == "__main__":
    main()
