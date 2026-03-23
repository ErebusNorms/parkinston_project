import torch.nn as nn


class KerasRNNStack(nn.Module):
    def __init__(self,
                 rnn_type,
                 input_size,
                 hidden,
                 num_layers,
                 dropout=0.0):

        super().__init__()

        self.layers = nn.ModuleList()

        for i in range(num_layers):

            in_size = input_size if i == 0 else hidden

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
                hidden = hidden * 2

            else:
                raise ValueError("Unknown rnn_type")

            self.layers.append(rnn)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        for i, rnn in enumerate(self.layers):

            x, _ = rnn(x)

            # Keras behavior:
            if i != len(self.layers) - 1:
                x = self.dropout(x)
            else:
                x = x[:, -1]

        return x
    
class KerasCNN(nn.Module):
    def __init__(self, channels, use_global_pool=True):
        super().__init__()

        layers = []
        in_ch = 1

        for ch in channels:
            layers += [
                nn.Conv1d(in_ch, ch, 3, padding=1),
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