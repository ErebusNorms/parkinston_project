import torch.nn as nn
from models.keras_like_blocks import KerasRNNStack, KerasCNN
from models.norms import get_norm

class KerasLikeModel(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        model_type = cfg["name"].lower()
        self.model_type = model_type

        self.use_cnn = model_type in [
            "cnn", "cnn_lstm", "cnn_gru", "cnn_bilstm"
        ]

        self.use_rnn = model_type in [
            "lstm", "gru", "bilstm",
            "cnn_lstm", "cnn_gru", "cnn_bilstm"
        ]

        # ================= CNN =================
        if self.use_cnn:
            self.cnn = KerasCNN(
                cfg["cnn_channels"],
                cfg["use_global_pool"],
                norm=cfg["norm"]
            )
            cnn_out = cfg["cnn_channels"][-1]
        else:
            self.cnn = None
            cnn_out = 1

        # ================= RNN =================
        if self.use_rnn:

            if "bilstm" in model_type:
                rnn_type = "bilstm"
                rnn_out = cfg["rnn_hidden"] * 2
            elif "gru" in model_type:
                rnn_type = "gru"
                rnn_out = cfg["rnn_hidden"]
            else:
                rnn_type = "lstm"
                rnn_out = cfg["rnn_hidden"]

            input_size = cnn_out if self.use_cnn else 1

            self.rnn = KerasRNNStack(
                rnn_type,
                input_size,
                cfg["rnn_hidden"],
                cfg["rnn_layers"],
                cfg["dropout"],
                norm=cfg["norm"]   # 🔥 thêm dòng này
            )

            fc_in = rnn_out

        else:
            self.rnn = None
            fc_in = cnn_out

        # ================= HEAD =================
        self.fc = nn.Sequential(
            nn.Linear(fc_in, cfg["dense_hidden"]),
            get_norm(cfg["norm"], cfg["dense_hidden"], dim="fc"),
            nn.ReLU(),
            nn.Dropout(cfg["dropout"]),
            nn.Linear(cfg["dense_hidden"], 2)
        )

    def forward(self, x):

        # x: (B,1,T)

        # ===== CNN =====
        if self.use_cnn:
            x = self.cnn(x)      # (B,C)

        # ===== reshape cho RNN =====
        if self.use_rnn:

            if self.use_cnn:
                # CNN output (B, C) → (B, 1, C)
                if x.dim() == 2:
                    x = x.unsqueeze(1)

            else:
                # raw signal (B,1,T) → (B,T,1)
                x = x.permute(0, 2, 1)

            x = self.rnn(x)

        # ===== FC =====
        x = self.fc(x)

        return x