import os
import torch
import numpy as np
import random

# Import from your modular python files
from Utils.helpers import load_config, setup_directories
from dataloader import TimeSeriesDataModule
from build import create_model
from training import Trainer
from testing import Evaluator

def set_seed(seed=42):
    """Locks all random number generators for reproducible research."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def main():
    # 1. Load Standardized Configuration
    config_path = "Utils/model_config.yaml"
    print(f"📄 Loading configuration from {config_path}...")
    config = load_config(config_path)
    
    # 2. Setup System & Seed
    set_seed(config['system'].get('random_seed', 42))
    device = torch.device(config['system']['device'] if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Active Device: {device}")

    # 3. Setup Directories
    dirs = setup_directories(config['logging']['save_dir'])
    print(f"📁 Artifacts will be saved to: {config['logging']['save_dir']}")

    # 4. Initialize Data Pipeline
    print("\n" + "="*40)
    print(" PHASE 1: DATA PREPARATION ")
    print("="*40)
    data_module = TimeSeriesDataModule(config)
    # train_loader, val_loader, scaler, input_dim = data_module.prepare_data()
    train_loader, val_loader, scaler, input_dim, seq_len = data_module.prepare_data()
    print(f"✅ Dynamic Setup: Lookback={seq_len}, Horizon={config['data']['prediction_horizon']}")
    print(f"✅ Data prepared. Input features after Boruta/Selection: {input_dim}")

    # 5. Build Model
    print("\n" + "="*40)
    print(f" PHASE 2: MODEL BUILD ({config['model']['architecture']}) ")
    print("="*40)
    model = create_model(config, input_dim)
    
    # Optional: Print Model Summary if torchinfo is installed
    try:
        from torchinfo import summary
        batch_size = config['training']['batch_size']
        seq_len = config['data']['sequence_length']
        summary(model, input_size=(batch_size, seq_len, input_dim))
    except ImportError:
        print("ℹ️ Install 'torchinfo' for a detailed layer summary.")

    # 6. Training Phase
    print("\n" + "="*40)
    print(" PHASE 3: TRAINING ")
    print("="*40)
    trainer = Trainer(model, config)
    trained_model, history = trainer.fit(train_loader, val_loader)

    # 7. Evaluation Phase
    print("\n" + "="*40)
    print(" PHASE 4: EVALUATION & LOGGING ")
    print("="*40)
    evaluator = Evaluator(trained_model, config, dirs)
    metrics = evaluator.evaluate(val_loader)

    # Print Final Metrics Summary
    print("\n📊 Final Test Metrics:")
    for metric, value in metrics.items():
        if isinstance(value, float):
            print(f"   - {metric}: {value:.4f}")
        else:
            print(f"   - {metric}: {value}")

    # 8. Save Model Weights
    model_save_path = os.path.join(dirs['models'], f"{config['model']['architecture']}_best.pth")
    torch.save(trained_model.state_dict(), model_save_path)
    print(f"\n💾 Model weights saved to {model_save_path}")
    print("🚀 Single model test complete!")

if __name__ == "__main__":
    main()