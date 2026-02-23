import torch.nn as nn
from models.blocks import CNNBackbone, TemporalAttention

class CNN_GRU_BiLSTM_Att(nn.Module):
    def __init__(self, cnn_channels, hidden, n_class=2):
        super().__init__()
        self.cnn = CNNBackbone(cnn_channels)
        self.gru = nn.GRU(cnn_channels[-1], hidden, batch_first=True)
        self.bilstm = nn.LSTM(
            hidden,
            hidden,
            batch_first=True,
            bidirectional=True
        )
        self.att = TemporalAttention(hidden * 2)
        self.fc = nn.Linear(hidden * 2, n_class)

    def forward(self, x):
        x = self.cnn(x).permute(0, 2, 1)
        x, _ = self.gru(x)
        x, _ = self.bilstm(x)
        x = self.att(x)
        return self.fc(x)