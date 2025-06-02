#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Encoder / Decoder モジュール
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

# ---------------- Encoder -----------------
class GEncoder(nn.Module):
    """GCN 2-layer -> mean / logvar"""
    def __init__(self, in_dim, hid=40, z_dim=20):
        super().__init__()
        self.gcn1 = GCNConv(in_dim, hid)
        self.gcn2 = GCNConv(hid,  z_dim)
        self.mu    = nn.Linear(z_dim, z_dim)
        self.logvar = nn.Linear(z_dim, z_dim)

    def forward(self, x, edge_index):
        h = F.relu(self.gcn1(x, edge_index))
        h = self.gcn2(h, edge_index)
        return self.mu(h), self.logvar(h)

# ---------------- Decoder -----------------
class EdgeMLP(nn.Module):
    """
    2 つの潜在ベクトル (z_i, z_j) を結合 → MLP → **logit**  
    *Sigmoid をかけずに「生のロジット」を返す*  
    （BCEWithLogitsLoss で安全に AMP 可能）
    """
    def __init__(self, z_dim, hid=128):
        super().__init__()
        self.fc1 = nn.Linear(2 * z_dim, hid)
        self.fc2 = nn.Linear(hid, 1)

    def forward(self, z):
        n = z.size(0)
        zi = z.unsqueeze(1).expand(n, n, -1)   # (n,n,z)
        zj = z.unsqueeze(0).expand(n, n, -1)   # (n,n,z)
        pair = torch.cat([zi, zj], dim=-1)     # (n,n,2z)
        h = F.relu(self.fc1(pair))
        logits = self.fc2(h).squeeze(-1)       # (n,n)
        return logits
