import torch
import torch.nn as nn


class GCVNetwork(nn.Module):
    def __init__(self, input_dim) -> None:
        super().__init__()
        
        self.model = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1),
        )

    def forward(self, x):
        return self.model(x)
    
    def predict(self, x):
        with torch.no_grad():
            x = torch.from_numpy(x).float()
            return self.model(x)
