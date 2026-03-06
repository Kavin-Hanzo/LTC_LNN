# main.py
from Utils.helpers import load_config, setup_directories
from dataloader import TimeSeriesDataModule
from build import create_model
from training import Trainer
from testing import Evaluator
import torch

# 1. Setup
config = load_config("Utils/model_config.yaml")
dirs = setup_directories(config['logging']['save_dir'])

# 2. Data
data_module = TimeSeriesDataModule(config)
train_loader, val_loader, scaler, input_dim = data_module.prepare_data()

# 3. Build Model
model = create_model(config, input_dim)

# 4. Train
trainer = Trainer(model, config)
trained_model, history = trainer.fit(train_loader, val_loader)

# 5. Evaluate
evaluator = Evaluator(trained_model, config, dirs)
metrics = evaluator.evaluate(val_loader)

# 6. Save Model
torch.save(trained_model.state_dict(), f"{dirs['models']}/{config['model_specs']['architecture']}_best.pth")