import torch
import torch.nn as nn
import torch.optim as optim
from torchmetrics.regression import R2Score, MeanAbsoluteError
import copy

class Trainer:
    def __init__(self, model, config):
        """
        model: PyTorch model instance
        config: Unified YAML config dictionary
        """
        self.model = model
        self.cfg = config['training']
        self.device = torch.device(config['system']['device'])
        
        # Loss & Optimizer
        self.criterion = getattr(nn, self.cfg['loss_function'])()
        self.optimizer = getattr(optim, self.cfg['optimizer'])(
            self.model.parameters(), 
            lr=self.cfg['learning_rate']
        )
        
        # High-Resolution GPU Metrics
        # These stay on the GPU to avoid CPU-bottlenecks every epoch
        self.val_r2_metric = R2Score().to(self.device)
        self.val_mae_metric = MeanAbsoluteError().to(self.device)
        
        # Best model tracking
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
            # --- TRAINING PHASE ---
            self.model.train()
            running_train_loss = 0.0
            
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device, dtype=torch.float32)
                y_batch = y_batch.to(self.device, dtype=torch.float32)
                
                self.optimizer.zero_grad()
                
                # Forward pass - use .view(-1) for direct horizon mapping
                preds = self.model(X_batch).view(-1)
                loss = self.criterion(preds, y_batch.view(-1))
                
                # Backward pass
                loss.backward()
                
                # Gradient Clipping (Essential for LNN/RNN stability)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg['grad_clip'])
                
                self.optimizer.step()
                running_train_loss += loss.item()

            # --- VALIDATION PHASE (High-Resolution) ---
            self.model.eval()
            running_val_loss = 0.0
            
            with torch.no_grad():
                for X_val, y_val in val_loader:
                    X_val = X_val.to(self.device, dtype=torch.float32)
                    y_val = y_val.to(self.device, dtype=torch.float32)
                    
                    v_preds = self.model(X_val).view(-1)
                    v_loss = self.criterion(v_preds, y_val.view(-1))
                    
                    running_val_loss += v_loss.item()
                    
                    # Update metrics on GPU
                    self.val_r2_metric.update(v_preds, y_val.view(-1))
                    self.val_mae_metric.update(v_preds, y_val.view(-1))

            # --- EPOCH LOGGING & BEST MODEL CHECK ---
            avg_train_loss = running_train_loss / len(train_loader)
            avg_val_loss = running_val_loss / len(val_loader)
            
            # Compute final metric values for this epoch
            epoch_r2 = self.val_r2_metric.compute().item()
            epoch_mae = self.val_mae_metric.compute().item()
            
            # Store in history
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(avg_val_loss)
            history['val_r2'].append(epoch_r2)
            history['val_mae'].append(epoch_mae)
            
            # Track the 'Best' model based on R2 score
            if epoch_r2 > self.best_val_r2:
                self.best_val_r2 = epoch_r2
                # Deep copy the weights so we don't just save a reference
                self.best_model_state = copy.deepcopy(self.model.state_dict())
            
            # Reset metrics for next epoch
            self.val_r2_metric.reset()
            self.val_mae_metric.reset()

            if (epoch + 1) % 5 == 0:
                print(f"Epoch [{epoch+1}/{self.cfg['epochs']}] | "
                      f"Train Loss: {avg_train_loss:.5f} | "
                      f"Val R2: {epoch_r2:.4f} | "
                      f"Val MAE: {epoch_mae:.5f}")

        # Load the best weights before returning
        if self.best_model_state:
            print(f"✅ Training Complete. Loading Best Model (R2: {self.best_val_r2:.4f})")
            self.model.load_state_dict(self.best_model_state)
            
        return self.model, history