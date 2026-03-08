import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os

class Evaluator:
    def __init__(self, model, config, dirs):
        self.model = model
        self.config = config
        self.device = torch.device(config['system']['device'])
        # Extract the first ticker from the list for single-model testing
        self.ticker = config['data_config']['tickers'][0]
        self.arch = config['model_config']['architecture']
        self.horizon = config['data_config']['prediction_horizon']
        self.dirs = dirs

    def evaluate(self, val_loader):
        print("📊 Running Evaluation...")
        self.model.eval()
        preds, actuals = [], []
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(self.device)
                out = self.model(X_batch).squeeze()
                
                # Handle single-item batch scalar edge cases
                if out.ndim == 0:
                    out = out.unsqueeze(0)
                    
                preds.append(out.cpu().numpy())
                actuals.append(y_batch.numpy())
                
        preds = np.concatenate(preds)
        actuals = np.concatenate(actuals)
        
        metrics = self._calculate_metrics(actuals, preds)
        self._save_results(metrics, actuals, preds)
        return metrics

    def _calculate_metrics(self, y_true, y_pred):
        # Standardize shapes to avoid broadcasting errors on GPU
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()

        # sMAPE is better for comparing Next-Day vs Next-Month
        smape = 100/len(y_true) * np.sum(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))

        return {
            "Horizon": self.horizon,
            "MSE": mean_squared_error(y_true, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
            "MAE": mean_absolute_error(y_true, y_pred),
            "R2": r2_score(y_true, y_pred),
            "sMAPE": smape # Crucial for your Results Table
        }

    def _save_results(self, metrics, y_true, y_pred):
        # 1. Save Metrics to CSV
        df = pd.DataFrame([metrics])
        csv_path = os.path.join(self.dirs['results'], f"{self.ticker}_{self.arch}_metrics.csv")
        df.to_csv(csv_path, index=False)
        print(f"✅ Metrics saved to {csv_path}")

        plt.figure(figsize=(12, 6))

        # We shift the prediction plot to align with the actual day it was aiming for
        plt.plot(y_true[-100:], label="Actual Price", color='blue', alpha=0.6)
        plt.plot(y_pred[-100:], label=f"Predicted (h={self.horizon})", color='red', linestyle='--')

        plt.title(f"{self.arch} Forecasting: {self.ticker} (Horizon: {self.horizon} steps)")
        plt.ylabel("Scaled Price")
        plt.xlabel("Time Steps (Validation Set)")
        plt.legend()
        plot_path = os.path.join(self.dirs['figures'], f"{self.ticker}_{self.arch}_overlay.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"✅ Plot saved to {plot_path}")