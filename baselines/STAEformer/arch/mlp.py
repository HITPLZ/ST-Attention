import torch.nn as nn
import torch

class MultiLayerPerceptron(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.2) -> None:
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=hidden_dim, bias=True),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim, bias=True),
        )

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        hidden = self.fc(input_data)  # MLP
        hidden = hidden + input_data  # residual
        return hidden
