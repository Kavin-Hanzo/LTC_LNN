import torch
import torch.nn as nn
import torch.optim as optim
from torchmetrics.regression import R2Score, MeanAbsoluteError
import copy

class Trainer:
    def __init__(self, model, config):
        self.model = model
        # Mapped to model_config.yaml keys perfectly
        self.cfg = config['training']
        self.device = torch.device(config['system']['device'])
        
        # Setup Loss and Optimizer (Using 'loss_function' key from model_config.yaml)
        self.criterion = getattr(nn, self.cfg['loss_function'])()
        self.optimizer = getattr(optim, self.cfg['optimizer'])(
            self.model.parameters(), 
            lr=self.cfg['learning_rate']
        )
        
        self.val_r2_metric = R2Score().to(self.device)
        self.val_mae_metric = MeanAbsoluteError().to(self.device)
        
        self.best_val_r2 = -float('inf')
        self.best_model_state = None

    def fit(self, train_loader, val_loader):
        print(f"🚀 Training starting on {self.device} (High-Res Tracking Enabled)")
        
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_r2': [],
            'val_mae': []
        }
        
        for epoch in range(self.cfg['epochs']):
            self.model.train()
            running_train_loss = 0.0
            
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device, dtype=torch.float32)
                y_batch = y_batch.to(self.device, dtype=torch.float32)
                
                self.optimizer.zero_grad()
                
                preds = self.model(X_batch).view(-1)
                loss = self.criterion(preds, y_batch.view(-1))
                loss.backward()
                
                # Gradient Clipping using the explicit key from model_config.yaml
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg['grad_clip'])
                
                self.optimizer.step()
                running_train_loss += loss.item()

            self.model.eval()
            running_val_loss = 0.0
            
            with torch.no_grad():
                for X_val, y_val in val_loader:
                    X_val = X_val.to(self.device, dtype=torch.float32)
                    y_val = y_val.to(self.device, dtype=torch.float32)
                    
                    v_preds = self.model(X_val).view(-1)
                    v_loss = self.criterion(v_preds, y_val.view(-1))
                    
                    running_val_loss += v_loss.item()
                    
                    self.val_r2_metric.update(v_preds, y_val.view(-1))
                    self.val_mae_metric.update(v_preds, y_val.view(-1))

            avg_train_loss = running_train_loss / len(train_loader)
            avg_val_loss = running_val_loss / len(val_loader)
            
            epoch_r2 = self.val_r2_metric.compute().item()
            epoch_mae = self.val_mae_metric.compute().item()
            
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(avg_val_loss)
            history['val_r2'].append(epoch_r2)
            history['val_mae'].append(epoch_mae)
            
            if epoch_r2 > self.best_val_r2:
                self.best_val_r2 = epoch_r2
                self.best_model_state = copy.deepcopy(self.model.state_dict())
            
            self.val_r2_metric.reset()
            self.val_mae_metric.reset()

            if (epoch + 1) % 5 == 0:
                print(f"Epoch [{epoch+1}/{self.cfg['epochs']}] | "
                      f"Train Loss: {avg_train_loss:.5f} | "
                      f"Val R2: {epoch_r2:.4f} | "
                      f"Val MAE: {epoch_mae:.5f}")

        if self.best_model_state:
            print(f"✅ Training Complete. Loading Best Model (R2: {self.best_val_r2:.4f})")
            self.model.load_state_dict(self.best_model_state)
            
        return self.model, history