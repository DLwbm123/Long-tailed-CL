
import torch
import torch.nn as nn


class StrategyController(nn.Module):
    def __init__(self, num_classes, state_dim=5, hidden_dim=32):
        super().__init__()
        self.gating_logits = nn.Parameter(torch.zeros(num_classes, 1))

    def forward(self):
        g_min = 0.1
        g_max = 0.9
        g_sigmoid = torch.sigmoid(self.gating_logits)
        g_smoothed = g_sigmoid * (g_max - g_min) + g_min
        return g_smoothed

    @torch.no_grad()
    def set_gating_logits(self, new_logits):
        self.gating_logits.data = new_logits