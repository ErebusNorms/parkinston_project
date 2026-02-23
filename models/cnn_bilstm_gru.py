import torch.nn as nn
from models.blocks import CNNBackbone

class CNN_BiLSTM_GRU(nn.Module):
    def __init__(self, cnn_channels, hidden, n_class=2):
        super().__init__()
        self.cnn = CNNBackbone(cnn_channels)
        self.bilstm = nn.LSTM(
            cnn_channels[-1],
            hidden,
            batch_first=True,
            bidirectional=True
        )
        self.gru = nn.GRU(hidden * 2, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, n_class)

    def forward(self, x):
        x = self.cnn(x).permute(0, 2, 1)
        x, _ = self.bilstm(x)
        x, _ = self.gru(x)
        return self.fc(x.mean(dim=1))