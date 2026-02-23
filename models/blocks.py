import torch
import torch.nn as nn


class CNNBackbone(nn.Module):
    def __init__(self, channels, dropout=0.0):
        super().__init__()

        layers = []
        in_ch = 1

        for ch in channels:
            layers += [
                nn.Conv1d(in_ch, ch, kernel_size=3, padding=1),
                nn.BatchNorm1d(ch),
                nn.ReLU(),
                nn.Dropout(dropout)
            ]
            in_ch = ch

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class TemporalAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Linear(dim, 1)

    def forward(self, x):
        # x: (B, T, C)
        w = torch.softmax(self.fc(x), dim=1)
        return (x * w).sum(dim=1)