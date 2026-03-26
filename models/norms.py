import torch
import torch.nn as nn


class SwitchNorm1D(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.bn = nn.BatchNorm1d(num_features)
        self.ln = nn.LayerNorm(num_features)
        self.inorm = nn.InstanceNorm1d(num_features)

        self.w = nn.Parameter(torch.ones(3))

    def forward(self, x):
        # x: (B,C,T)
        bn = self.bn(x)
        ln = self.ln(x.transpose(1, 2)).transpose(1, 2)
        inn = self.inorm(x)

        w = torch.softmax(self.w, dim=0)

        return w[0]*bn + w[1]*ln + w[2]*inn


def get_norm(norm_type, num_features, dim="1d"):

    if norm_type == "none":
        return nn.Identity()

    if norm_type == "batch":
        return nn.BatchNorm1d(num_features)

    if norm_type == "layer":
        return nn.LayerNorm(num_features)

    if norm_type == "group":
        return nn.GroupNorm(min(8, num_features), num_features)

    if norm_type == "instance":
        return nn.InstanceNorm1d(num_features)

    if norm_type == "switch":
        return SwitchNorm1D(num_features)

    raise ValueError(f"Unknown norm: {norm_type}")