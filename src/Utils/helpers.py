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
    
    boruta_selector.fit(X.values, y.values)
    
    selected_features = X.columns[boruta_selector.support_].tolist()
    print(f"✅ Boruta selected {len(selected_features)} out of {X.shape[1]} features.")
    return selected_features