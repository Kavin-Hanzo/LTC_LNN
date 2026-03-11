import os
import torch
import pandas as pd
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import json
from datetime import datetime

# Modular Imports
from Utils.helpers import load_config, setup_directories
from dataloader import TimeSeriesDataModule
from build import create_model
from torch_training import Trainer
from torchmetric_testing import Evaluator

def plot_training_history(history, save_path, title):
    # Use the 'plt.subplots' context manager for better memory handling
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Loss Plot
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Val Loss')
    ax1.set_title(f'Loss: {title}')
    ax1.legend()
    
    # R2 Plot
    if 'val_r2' in history:
        ax2.plot(history['val_r2'], label='Val R2', color='green')
        ax2.set_title(f'R2 Score: {title}')
        ax2.legend()
        
    plt.tight_layout()
    plt.savefig(save_path)
    
    # CRITICAL: Clear and close to free memory and prevent Tkinter errors
    plt.cla()   # Clear axis
    plt.clf()   # Clear figure
    plt.close(fig)

def main():
    # 1. Load Master Configuration
    config = load_config("./Utils/config.yaml")
    base_save_dir = config['logging']['save_dir']
    
    # Define architectures to test (Iterates through all possible types)
    architectures = ["RNN", "LSTM", "GRU", "LNN"]
    
    all_results = []
    
    # 2. Nested Experiment Loops
    for ticker in config['data_config']['tickers']:
        for data_set in config['data_config']['historical_sets']:
            for arch in architectures:
                
                exp_name = f"{ticker}_{data_set['name']}_{arch}"
                print(f"\n🚀 STARTING EXPERIMENT: {exp_name}")
                
                # 3. Construct Run-Specific Configuration
                # This resolves the mismatch between config.yaml and module expectations
                run_config = {
                    'system': config['system'],
                    'data': {
                        'ticker': ticker,
                        'start_date': data_set['start'],
                        'end_date': data_set['end'],
                        'interval': data_set['interval'],
                        'train_split': config['data_config']['train_split_ratio'],
                        'prediction_horizon': config['data_config']['prediction_horizon'],
                        'target_column': config['data_config']['target_column'],
                        'use_boruta': config['data_config']['feature_selection']['use_boruta']
                    },
                    'model': {
                        'architecture': arch,
                        'hidden_size': config['model']['hidden_size'],
                        'num_layers': config['model']['num_layers'],
                        'dropout_rate': config['model']['dropout_rate'],
                        'ltc_params': config['model']['ltc_params']
                    },
                    'training': config['model']['training'],
                    'logging': config['logging']
                }

                # 4. Setup Experiment Directory
                exp_dir = os.path.join(base_save_dir, ticker, data_set['name'], arch)
                dirs = setup_directories(exp_dir)

                try:
                    # 5. Data Pipeline
                    dm = TimeSeriesDataModule(run_config)
                    train_loader, val_loader, scaler, in_dim, seq_len = dm.prepare_data()

                    # 6. Build and Train
                    model = create_model(run_config, in_dim)
                    trainer = Trainer(model, run_config)
                    trained_model, history = trainer.fit(train_loader, val_loader)

                    # 7. Plot History
                    history_path = os.path.join(dirs['figures'], "training_history.png")
                    plot_training_history(history, history_path, exp_name)

                    # 8. Evaluate
                    evaluator = Evaluator(trained_model, run_config, dirs, scaler)
                    metrics = evaluator.evaluate(val_loader)
                    
                    # Add metadata to metrics for the master CSV
                    metrics['Dataset'] = data_set['name']
                    metrics['Timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M")
                    all_results.append(metrics)

                    # 9. Save Model
                    torch.save(trained_model.state_dict(), os.path.join(dirs['models'], "best_model.pth"))
                    
                except Exception as e:
                    print(f"❌ Experiment {exp_name} failed: {e}")
                    continue

    # 10. Save Master Results Table
    master_df = pd.DataFrame(all_results)
    master_path = os.path.join(base_save_dir, "master_results_summary.csv")
    master_df.to_csv(master_path, index=False)
    print(f"\n✅ ALL EXPERIMENTS COMPLETE. Summary saved to: {master_path}")

if __name__ == "__main__":
    main()