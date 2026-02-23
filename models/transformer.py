import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() *
            (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class TransformerClassifier(nn.Module):
    def __init__(self,
                 hidden,
                 num_layers=3,
                 nhead=4,
                 dropout=0.2,
                 n_class=2):

        super().__init__()

        self.embedding = nn.Linear(1, hidden)

        self.pos_encoder = PositionalEncoding(hidden)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=nhead,
            dim_feedforward=hidden * 4,
            dropout=dropout,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, n_class)
        )

    def forward(self, x):
        # x: (B,1,T)
        x = x.permute(0, 2, 1)      # (B,T,1)
        x = self.embedding(x)      # (B,T,H)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        x = x.mean(dim=1)          # Global average pooling
        return self.fc(x)