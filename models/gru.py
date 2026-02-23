import torch.nn as nn

class GRU(nn.Module):
    def __init__(self, hidden, n_class=2):
        super().__init__()
        self.rnn = nn.GRU(1, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, n_class)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x, _ = self.rnn(x)
        return self.fc(x.mean(dim=1))