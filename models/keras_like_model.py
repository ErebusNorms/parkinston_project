import torch.nn as nn
from models.keras_like_blocks import KerasRNNStack, KerasCNN

class KerasLikeModel(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        model_type = cfg["name"].lower()

        self.model_type = model_type

        # ================= CNN =================
        if model_type in ["cnn", "cnn_lstm", "cnn_gru", "cnn_bilstm"]:
            self.cnn = KerasCNN(
                cfg["cnn_channels"],
                cfg["use_global_pool"]
            )
            cnn_out = cfg["cnn_channels"][-1]
        else:
            self.cnn = None
            cnn_out = 1

        # ================= RNN =================
        if model_type == "lstm":
            self.rnn = KerasRNNStack("lstm", 1,
                                    cfg["rnn_hidden"],
                                    cfg["rnn_layers"],
                                    cfg["dropout"])
            fc_in = cfg["rnn_hidden"]

        elif model_type == "gru":
            self.rnn = KerasRNNStack("gru", 1,
                                    cfg["rnn_hidden"],
                                    cfg["rnn_layers"],
                                    cfg["dropout"])
            fc_in = cfg["rnn_hidden"]

        elif model_type == "bilstm":
            self.rnn = KerasRNNStack("bilstm", 1,
                                    cfg["rnn_hidden"],
                                    cfg["rnn_layers"],
                                    cfg["dropout"])
            fc_in = cfg["rnn_hidden"] * 2

        elif model_type == "cnn_lstm":
            self.rnn = KerasRNNStack("lstm", cnn_out,
                                    cfg["rnn_hidden"],
                                    cfg["rnn_layers"],
                                    cfg["dropout"])
            fc_in = cfg["rnn_hidden"]

        elif model_type == "cnn_gru":
            self.rnn = KerasRNNStack("gru", cnn_out,
                                    cfg["rnn_hidden"],
                                    cfg["rnn_layers"],
                                    cfg["dropout"])
            fc_in = cfg["rnn_hidden"]

        elif model_type == "cnn_bilstm":
            self.rnn = KerasRNNStack("bilstm", cnn_out,
                                    cfg["rnn_hidden"],
                                    cfg["rnn_layers"],
                                    cfg["dropout"])
            fc_in = cfg["rnn_hidden"] * 2

        else:
            self.rnn = None
            fc_in = cnn_out

        # ================= HEAD =================
        self.fc = nn.Sequential(
            nn.Linear(fc_in, cfg["dense_hidden"]),
            nn.ReLU(),
            nn.Dropout(cfg["dropout"]),
            nn.Linear(cfg["dense_hidden"], 2)
        )

    def forward(self, x):

        # x: (B,1,T)

        if self.cnn is not None:
            x = self.cnn(x)

            if x.dim() == 2:
                x = x.unsqueeze(1)

        else:
            x = x.permute(0, 2, 1)

        if self.rnn is not None:
            x = self.rnn(x)

        x = self.fc(x)

        if x.dim() == 3:
            x = x.squeeze(1)

        return x