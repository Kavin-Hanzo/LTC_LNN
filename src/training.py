import torch
import torch.nn as nn
import torch.optim as optim

class Trainer:
    def __init__(self, model, config):
        self.model = model
        # Mapped to the nested training config in config.yaml
        self.cfg = config['model_config']['training']
        self.device = torch.device(config['system']['device'])
        
        # Setup Loss and Optimizer (using 'criterion' key from config.yaml)
        self.criterion = getattr(nn, self.cfg['criterion'])()
        self.optimizer = getattr(optim, self.cfg['optimizer'])(
            self.model.parameters(), 
            lr=self.cfg['learning_rate'],
            weight_decay=self.cfg.get('weight_decay', 0)
        )
        
    def fit(self, train_loader, val_loader):
        print(f"🚀 Starting Training on {self.device}...")
        history = {'train_loss': [], 'val_loss': []}
        
        # safely handle grad_clip if missing in config.yaml
        grad_clip = self.cfg.get('grad_clip', 1.0)
        
        for epoch in range(self.cfg['epochs']):
            self.model.train()
            train_loss = 0.0
            
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                
                self.optimizer.zero_grad()
                predictions = self.model(X_batch) 
                
                loss = self.criterion(predictions.view(-1), y_batch.view(-1))
                loss.backward()
                
                # Gradient Clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
                self.optimizer.step()
                
                train_loss += loss.item()
                
            # Validation Phase
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for X_val, y_val in val_loader:
                    X_val, y_val = X_val.to(self.device), y_val.to(self.device)
                    val_preds = self.model(X_val).squeeze()
                    val_loss += self.criterion(val_preds, y_val).item()
                    
            t_loss_avg = train_loss / len(train_loader)
            v_loss_avg = val_loss / len(val_loader)
            history['train_loss'].append(t_loss_avg)
            history['val_loss'].append(v_loss_avg)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{self.cfg['epochs']}] | Train Loss: {t_loss_avg:.5f} | Val Loss: {v_loss_avg:.5f}")
                
        return self.model, history