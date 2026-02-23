import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self,
                 hidden,
                 rnn_layers=1,
                 dropout=0.0,
                 n_class=2):

        super().__init__()

        self.rnn = nn.LSTM(
            input_size=1,
            hidden_size=hidden,
            num_layers=rnn_layers,
            dropout=dropout if rnn_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=True
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_class)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)   # (B, T, 1)
        x, _ = self.rnn(x)
        x = x.mean(dim=1)
        return self.fc(x)