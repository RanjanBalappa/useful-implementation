import torch
import torch.nn as nn
import torch.nn.functional as F


class CosineLinear(nn.Module):
    def __init__(self, input_dim: int, out_dim: int):
        self.weight = nn.Parameter((out_dim, input_dim)) / torch.sqrt(input_dim)

    def forward(self, x: torch.Tensor):
        return F.linear(F.normalize(x, p=2, dim=-1), F.normalize(self.weight, p=2, dim=-1))