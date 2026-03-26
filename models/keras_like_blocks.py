import torch.nn as nn
from models.norms import get_norm

class KerasRNNStack(nn.Module):
    def __init__(self,
                 rnn_type,
                 input_size,
                 hidden,
                 num_layers,
                 dropout=0.0,
                 norm="none"):

        super().__init__()

        self.layers = nn.ModuleList()
        self.rnn_type = rnn_type
        self.hidden = hidden

        for i in range(num_layers):

            in_size = input_size if i == 0 else self._get_output_dim()

            if rnn_type == "lstm":
                rnn = nn.LSTM(in_size, hidden, batch_first=True)

            elif rnn_type == "gru":
                rnn = nn.GRU(in_size, hidden, batch_first=True)

            elif rnn_type == "bilstm":
                rnn = nn.LSTM(
                    in_size,
                    hidden,
                    batch_first=True,
                    bidirectional=True
                )

            else:
                raise ValueError("Unknown rnn_type")

            self.layers.append(rnn)

        self.dropout = nn.Dropout(dropout)

        # 🔥 norm ở output cuối
        self.norm = get_norm(norm, self._get_output_dim(), dim="fc")

    def _get_output_dim(self):
        if self.rnn_type == "bilstm":
            return self.hidden * 2
        return self.hidden

    def forward(self, x):

        for i, rnn in enumerate(self.layers):

            x, _ = rnn(x)

            if i != len(self.layers) - 1:
                x = self.dropout(x)

        x = x[:, -1]   # (B, hidden)

        x = self.norm(x)

        return x
    
class KerasCNN(nn.Module):
    def __init__(self, channels, use_global_pool=True, norm="none"):
        super().__init__()

        layers = []
        in_ch = 1

        for ch in channels:
            layers += [
                nn.Conv1d(in_ch, ch, 3, padding=1),
                get_norm(norm, ch, dim="1d"),
                nn.ReLU(),
                nn.MaxPool1d(2)
            ]
            in_ch = ch

        if use_global_pool:
            layers.append(nn.AdaptiveAvgPool1d(1))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = self.net(x)

        if x.dim() == 3:
            x = x.squeeze(-1)

        return x