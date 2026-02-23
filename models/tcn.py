import torch.nn as nn


class TCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dilation, dropout=0.0):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 3,
                      padding=dilation,
                      dilation=dilation),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class TCN(nn.Module):
    def __init__(self,
                 hidden,
                 dropout=0.0,
                 n_class=2):

        super().__init__()

        self.tcn = nn.Sequential(
            TCNBlock(1, hidden, 1, dropout),
            TCNBlock(hidden, hidden, 2, dropout),
            TCNBlock(hidden, hidden, 4, dropout),
            TCNBlock(hidden, hidden, 8, dropout),
        )

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Sequential(
            nn.Linear(hidden, hidden//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden//2, n_class)
        )

    def forward(self, x):
        x = self.tcn(x)
        x = self.pool(x).squeeze(-1)
        return self.fc(x)