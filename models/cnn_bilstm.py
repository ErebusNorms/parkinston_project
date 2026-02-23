import torch.nn as nn
from models.blocks import CNNBackbone

class CNN_BiLSTM(nn.Module):
    def __init__(self,
                 cnn_channels,
                 hidden,
                 rnn_layers=1,
                 dropout=0.0,
                 n_class=2):

        super().__init__()

        self.cnn = CNNBackbone(cnn_channels, dropout)

        self.rnn = nn.LSTM(
            cnn_channels[-1],
            hidden,
            num_layers=rnn_layers,
            dropout=dropout if rnn_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=True
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden*2, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_class)
        )

    def forward(self, x):
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        x, _ = self.rnn(x)
        x = x.mean(dim=1)
        return self.fc(x)