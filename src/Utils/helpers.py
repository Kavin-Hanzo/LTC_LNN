import os
import yaml
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from boruta import BorutaPy

def load_config(config_path="Utils/model_config.yaml"):
    """Loads the YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def setup_directories(base_dir):
    """Creates necessary folders for saving artifacts."""
    dirs = {
        "models": os.path.join(base_dir, "models"),
        "figures": os.path.join(base_dir, "figures"),
        "results": os.path.join(base_dir, "results")
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    return dirs

def apply_boruta(X, y):
    """
    Applies Boruta Feature Selection.
    X: DataFrame of features
    y: Series/Array of the target variable
    """
    print("🔍 Running Boruta Feature Selection...")
    rf = RandomForestRegressor(n_jobs=-1, max_depth=5)
    boruta_selector = BorutaPy(rf, n_estimators='auto', verbose=0, random_state=42)
    
    boruta_selector.fit(X.values, y)
    
    selected_features = X.columns[boruta_selector.support_].tolist()
    print(f"✅ Boruta selected {len(selected_features)} out of {X.shape[1]} features.")
    return selected_features

def calculate_price_deviation(y_true_norm, y_pred_norm, scaler, target_idx=-1):
    import torch
    """
    Converts normalized errors back to actual price deviations (USD).
    
    Args:
        y_true_norm (np.array/torch.Tensor): The actual scaled values.
        y_pred_norm (np.array/torch.Tensor): The predicted scaled values.
        scaler (sklearn.preprocessing.MinMaxScaler): The scaler used in dataloading.
        target_idx (int): The index of the target column in the original dataset 
                          (usually the last column).
    """
    # 1. Ensure inputs are numpy arrays for the scaler
    if torch.is_tensor(y_true_norm):
        y_true_norm = y_true_norm.cpu().detach().numpy()
    if torch.is_tensor(y_pred_norm):
        y_pred_norm = y_pred_norm.cpu().detach().numpy()
        
    y_true_norm = y_true_norm.flatten().reshape(-1, 1)
    y_pred_norm = y_pred_norm.flatten().reshape(-1, 1)

    # 2. To inverse transform, we need a dummy matrix of the same width 
    # as the original features if the scaler was fit on multiple columns.
    dummy_true = np.zeros((len(y_true_norm), scaler.n_features_in_))
    dummy_pred = np.zeros((len(y_pred_norm), scaler.n_features_in_))
    
    # Place our predictions in the target column index
    dummy_true[:, target_idx] = y_true_norm[:, 0]
    dummy_pred[:, target_idx] = y_pred_norm[:, 0]

    # 3. Inverse transform back to original price scale
    y_true_actual = scaler.inverse_transform(dummy_true)[:, target_idx]
    y_pred_actual = scaler.inverse_transform(dummy_pred)[:, target_idx]

    # 4. Calculate Absolute Deviations
    deviations = np.abs(y_true_actual - y_pred_actual)
    avg_price_error = np.mean(deviations)
    max_price_error = np.max(deviations)
    
    return {
        "avg_usd_error": avg_price_error,
        "max_usd_error": max_price_error,
        "true_prices": y_true_actual,
        "pred_prices": y_pred_actual
    }