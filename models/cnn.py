import torch.nn as nn
from models.blocks import CNNBackbone


class CNN(nn.Module):
    def __init__(self, cnn_channels, dropout=0.0, n_class=2):
        super().__init__()

        self.cnn = CNNBackbone(cnn_channels, dropout)
        self.pool = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Sequential(
            nn.Linear(cnn_channels[-1], cnn_channels[-1]//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(cnn_channels[-1]//2, n_class)
        )

    def forward(self, x):
        x = self.cnn(x)
        x = self.pool(x).squeeze(-1)
        return self.fc(x)