import torch
import torchmetrics
from torchmetrics.regression import MeanSquaredError, MeanAbsoluteError, R2Score
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
from Utils.helpers import calculate_price_deviation

class Evaluator:
    def __init__(self, model, config, dirs):
        """
        model: Trained PyTorch model
        config: Standardized YAML config
        dirs: Dictionary of paths (from utils.setup_directories)
        """
        self.model = model
        self.config = config
        self.device = torch.device(config['system']['device'])
        self.dirs = dirs
        self.ticker = config['data']['ticker']
        self.arch = config['model']['architecture']
        self.horizon = config['data']['prediction_horizon']
        
        # Initialize GPU-based metrics (kept on device for speed)
        self.mse_metric = MeanSquaredError().to(self.device)
        self.mae_metric = MeanAbsoluteError().to(self.device)
        self.r2_metric = R2Score().to(self.device)

    def evaluate(self, test_loader):
        print(f"📊 Running GPU-accelerated evaluation (Horizon: {self.horizon})...")
        self.model.eval()
        
        all_preds = []
        all_actuals = []

        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                # 1. Move to Device & Ensure Type Consistency
                X_batch = X_batch.to(self.device, dtype=torch.float32)
                y_batch = y_batch.to(self.device, dtype=torch.float32)
                
                # 2. Forward Pass & Shape Flattening
                # model output is [batch, 1], we need [batch]
                preds = self.model(X_batch).view(-1)
                targets = y_batch.view(-1)

                # 3. Update Metrics on GPU (Batch-wise accumulation)
                self.mse_metric.update(preds, targets)
                self.mae_metric.update(preds, targets)
                self.r2_metric.update(preds, targets)

                # Store for plotting later
                all_preds.append(preds)
                all_actuals.append(targets)

        # 4. Final Computation (Calculated on GPU, results returned as scalars)
        mse_val = self.mse_metric.compute().item()
        mae_val = self.mae_metric.compute().item()
        r2_val = self.r2_metric.compute().item()
        
        # 5. Custom Metric: sMAPE (torch-native for GPU efficiency)
        preds_tensor = torch.cat(all_preds)
        actuals_tensor = torch.cat(all_actuals)

        price_metrics = calculate_price_deviation(
            actuals_tensor, 
            preds_tensor, 
            self.scaler
        )

        print(f"💵 Average Price Deviation: ${price_metrics['avg_usd_error']:.2f}")
        metrics["Avg_USD_Error"] = price_metrics['avg_usd_error']

        smape_val = self._calculate_torch_smape(preds_tensor, actuals_tensor).item()

        metrics = {
            "Model": self.arch,
            "Ticker": self.ticker,
            "Horizon": self.horizon,
            "MSE": mse_val,
            "RMSE": np.sqrt(mse_val),
            "MAE": mae_val,
            "R2": r2_val,
            "sMAPE": smape_val
        }

        # 6. Save Artifacts (Requires conversion to CPU/Numpy for Matplotlib/Pandas)
        self._save_results(metrics, actuals_tensor.cpu().numpy(), preds_tensor.cpu().numpy())
        
        # Reset metric states for potential future runs
        self.mse_metric.reset()
        self.mae_metric.reset()
        self.r2_metric.reset()
        
        return metrics

    def _calculate_torch_smape(self, preds, target):
        """Calculates Symmetric Mean Absolute Percentage Error on GPU."""
        # sMAPE = (100 / n) * sum(|F_t - A_t| / ((|A_t| + |F_t|) / 2))
        numerator = torch.abs(preds - target)
        denominator = (torch.abs(target) + torch.abs(preds)) / 2.0
        # Prevent division by zero with a tiny epsilon
        return torch.mean(numerator / (denominator + 1e-8)) * 100

    def _save_results(self, metrics, y_true, y_pred):
        # 1. Save Metrics Table
        df = pd.DataFrame([metrics])
        csv_path = os.path.join(self.dirs['results'], f"{self.ticker}_{self.arch}_h{self.horizon}_metrics.csv")
        df.to_csv(csv_path, index=False)
        print(f"✅ CSV Saved: {csv_path}")

        # 2. Save Time-Series Overlay Plot
        plt.figure(figsize=(14, 7))
        
        # Plot only the last 150 points for visual clarity in long sequences
        plot_window = min(150, len(y_true))
        
        plt.plot(y_true[-plot_window:], label="Actual Close (Scaled)", color='#1f77b4', linewidth=2, alpha=0.8)
        plt.plot(y_pred[-plot_window:], label=f"Predicted Close (h={self.horizon})", color='#d62728', linestyle='--', linewidth=2)
        
        plt.title(f"Forecasting Performance: {self.ticker} | {self.arch} (Horizon: {self.horizon})", fontsize=14)
        plt.xlabel("Validation Time Steps", fontsize=12)
        plt.ylabel("Normalized Price", fontsize=12)
        plt.grid(True, which='both', linestyle='--', alpha=0.5)
        plt.legend(loc='best')
        
        plot_path = os.path.join(self.dirs['figures'], f"{self.ticker}_{self.arch}_h{self.horizon}_overlay.png")
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"✅ Plot Saved: {plot_path}")