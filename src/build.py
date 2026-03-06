import torch
import torch.nn as nn

class StandardRNN(nn.Module):
    def __init__(self, input_dim, hidden_size, num_layers, output_dim, dropout):
        super(StandardRNN, self).__init__()
        self.rnn = nn.RNN(input_dim, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, output_dim)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :]) # Take last time step

class StandardLSTM(nn.Module):
    def __init__(self, input_dim, hidden_size, num_layers, output_dim, dropout):
        super(StandardLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

class StandardGRU(nn.Module):
    def __init__(self, input_dim, hidden_size, num_layers, output_dim, dropout):
        super(StandardGRU, self).__init__()
        # GRU typically requires fewer parameters than LSTM
        self.gru = nn.GRU(
            input_dim, 
            hidden_size, 
            num_layers, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_size, output_dim)

    def forward(self, x):
        # x shape: [batch, seq_len, features]
        # GRU returns (output, hidden). We take 'output' for the many-to-one mapping.
        out, _ = self.gru(x)
        
        # Horizon Alignment: Extract only the last time step [t + h]
        # out[:, -1, :] shape: [batch, hidden_size]
        out = self.fc(out[:, -1, :]) 
        
        return out # Final shape: [batch, 1]

class LTCCell(nn.Module):
    def __init__(self, input_size, hidden_size, tau=1.0, dt=0.1):
        super(LTCCell, self).__init__()
        self.tau = tau
        self.dt = dt
        self.input_map = nn.Linear(input_size, hidden_size)
        self.recurrent_map = nn.Linear(hidden_size, hidden_size)
        self.tanh = nn.Tanh()

    def forward(self, x, h):
        v = self.tanh(self.input_map(x) + self.recurrent_map(h))
        dh = (-h / self.tau) + v
        return h + self.dt * dh

class LiquidNN(nn.Module):
    def __init__(self, input_dim, hidden_size, output_dim, ltc_params, device):
        super(LiquidNN, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.unfolds = ltc_params.get('ode_unfolds', 6)
        self.cell = LTCCell(input_dim, hidden_size, tau=ltc_params['tau_constant'])
        self.fc = nn.Linear(hidden_size, output_dim)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        h = torch.zeros(batch_size, self.hidden_size).to(self.device)
        for t in range(seq_len):
            for _ in range(self.unfolds):
                h = self.cell(x[:, t, :], h)
        return self.fc(h).squeeze(-1) # Output shape: (Batch)

def create_model(config, input_dim):
    """Factory function to instantiate the correct model."""
    specs = config['model_specs']
    device = torch.device(config['system']['device'])
    arch = specs['architecture'].upper()
    
    # Common parameters
    h_size = specs['hidden_size']
    layers = specs['num_layers']
    drop = specs['dropout_rate']
    
    if arch == "RNN":
        model = StandardRNN(input_dim, h_size, layers, 1, drop)
    elif arch == "LSTM":
        model = StandardLSTM(input_dim, h_size, layers, 1, drop)
    elif arch == "GRU":
        model = StandardGRU(input_dim, h_size, layers, 1, drop)
    elif arch == "LNN":
        model = LiquidNN(input_dim, h_size, 1, specs['ltc_params'], device)
    else:
        raise ValueError(f"Architecture {arch} not supported.")
        
    return model.to(device)