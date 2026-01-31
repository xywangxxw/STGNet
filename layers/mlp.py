import torch
import torch.nn as nn

from utils import init_weights


class MLP(nn.Module):
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 output_dim: int) -> None:
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )
        self.apply(init_weights)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.mlp(input)
