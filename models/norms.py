
import torch
import torch.nn as nn


# =========================
# PERMUTE (for LayerNorm in CNN)
# =========================
class Permute(nn.Module):
    def __init__(self, *dims):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(*self.dims)


# =========================
# SWITCH NORM 1D
# =========================
class SwitchNorm1D(nn.Module):
    def __init__(self, num_features):
        super().__init__()

        self.bn = nn.BatchNorm1d(num_features)
        self.inorm = nn.InstanceNorm1d(num_features)

        # LayerNorm dùng trên (B,T,C)
        self.ln = nn.LayerNorm(num_features)

        # learnable weights
        self.w = nn.Parameter(torch.ones(3))

    def forward(self, x):
        # x: (B, C, T)

        bn = self.bn(x)
        inn = self.inorm(x)

        # LayerNorm cần transpose
        ln = self.ln(x.transpose(1, 2)).transpose(1, 2)

        w = torch.softmax(self.w, dim=0)

        return w[0] * bn + w[1] * ln + w[2] * inn


# =========================
# GET NORM FUNCTION
# =========================
def get_norm(norm_type, num_features, dim="conv"):
    """
    dim:
        - "conv": input (B,C,T)
        - "fc": input (B,C)
        - "rnn": input (B,T,C)
    """

    if norm_type == "none":
        return nn.Identity()

    # ================= CNN =================
    if dim == "conv":

        if norm_type == "batch":
            return nn.BatchNorm1d(num_features)

        if norm_type == "group":
            return nn.GroupNorm(min(8, num_features), num_features)

        if norm_type == "instance":
            return nn.InstanceNorm1d(num_features)

        if norm_type == "layer":
            # FIX: cần permute (B,C,T) → (B,T,C)
            return nn.Sequential(
                Permute(0, 2, 1),
                nn.LayerNorm(num_features),
                Permute(0, 2, 1)
            )

        if norm_type == "switch":
            return SwitchNorm1D(num_features)

        if norm_type == "auto":
            # BEST CHOICE cho CNN EEG
            return nn.GroupNorm(min(8, num_features), num_features)

    # ================= RNN =================
    elif dim == "rnn":

        if norm_type == "layer":
            return nn.LayerNorm(num_features)

        if norm_type == "batch":
            return nn.BatchNorm1d(num_features)

        if norm_type == "group":
            return nn.GroupNorm(1, num_features)

        if norm_type == "instance":
            return nn.InstanceNorm1d(num_features)

        if norm_type == "switch":
            return nn.LayerNorm(num_features)  # fallback

        if norm_type == "auto":
            return nn.LayerNorm(num_features)

    # ================= FC =================
    elif dim == "fc":

        if norm_type == "layer":
            return nn.LayerNorm(num_features)

        if norm_type == "batch":
            return nn.BatchNorm1d(num_features)

        if norm_type == "group":
            return nn.GroupNorm(1, num_features)

        if norm_type == "instance":
            return nn.InstanceNorm1d(num_features)

        if norm_type == "switch":
            return nn.LayerNorm(num_features)

        if norm_type == "auto":
            return nn.LayerNorm(num_features)

    raise ValueError(f"Unknown norm: {norm_type}")