"""
Modernized GraphVAE implementation (PyTorch ≥ 2.2, CUDA 12.x) that preserves the
*exact* matching phase (MPM + Hungarian) from the original `model.py` code.
A small `forward_test()` utility is re‑added to mimic the toy example you posted
so you can sanity‑check the permutation logic without setting up a full dataset.
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np
import scipy.optimize
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_add_pool

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def vec_to_adj(vec: torch.Tensor, N: int) -> torch.Tensor:
    """Upper‑triangular vector → dense symmetric adjacency."""
    adj = vec.new_zeros(N, N)
    idx = torch.triu_indices(N, N)
    adj[idx[0], idx[1]] = vec
    adj = adj + adj.T - torch.diag(adj.diag())
    return adj


def deg_sim(d1, d2):
    """Degree similarity (same as original implementation)."""
    return 1.0 / (torch.abs(d1 - d2) + 1.0)


# -----------------------------------------------------------------------------
# GraphVAE module (unchanged forward + new test helpers)
# -----------------------------------------------------------------------------

class GraphVAE(nn.Module):
    def __init__(self, in_dim: int, hid_dim: int, z_dim: int, max_nodes: int, *, pool: str = "sum") -> None:
        super().__init__()
        self.max_n = max_nodes
        self.pool = pool

        # ----- encoder -----
        self.conv1 = GCNConv(in_dim, hid_dim)
        self.bn1 = nn.BatchNorm1d(hid_dim)
        self.conv2 = GCNConv(hid_dim, hid_dim)
        self.bn2 = nn.BatchNorm1d(hid_dim)
        self.mu = nn.Linear(hid_dim, z_dim)
        self.logvar = nn.Linear(hid_dim, z_dim)

        # ----- decoder -----
        tri = max_nodes * (max_nodes + 1) // 2
        self.dec = nn.Sequential(
            nn.Linear(z_dim, hid_dim), nn.ReLU(),
            nn.Linear(hid_dim, tri),
        )

        self._init_weights()

    # ---------------- utils ----------------
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, GCNConv)):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain("relu"))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    @staticmethod
    def _reparam(mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def _pool(self, h, batch):
        if self.pool == "sum":
            return global_add_pool(h, batch)
        return global_add_pool(h, batch) / torch.bincount(batch, minlength=batch.max() + 1).unsqueeze(-1).type_as(h)

    # ---------- permutation‑matching utilities ----------
    def _edge_sim_tensor(self, A, B, degA, degB):
        N = self.max_n
        S = A.new_zeros(N, N, N, N)
        for i in range(N):
            for j in range(N):
                if i == j:
                    for a in range(N):
                        S[i, i, a, a] = A[i, i] * B[a, a] * deg_sim(degA[i], degB[a])
                else:
                    for a in range(N):
                        for b in range(N):
                            if a == b:
                                continue
                            S[i, j, a, b] = (
                                A[i, j] * A[i, i] * A[j, j] * B[a, b] * B[a, a] * B[b, b]
                            )
        return S

    def _mpm(self, x0, S, iters: int = 50):
        x = x0
        N = self.max_n
        for _ in range(iters):
            x_new = x * S.diagonal(dim1=1, dim2=3)  # node similarity term
            for i in range(N):
                for a in range(N):
                    neigh = [torch.max(x[j] * S[i, j, a]) for j in range(N) if j != i]
                    x_new[i, a] += sum(neigh)
            x = x_new / x_new.norm(p=2)
        return x

    # ---------- canonical forward (training) ----------
    def forward(self, data):
        if data.batch.numel() == 0:
            raise RuntimeError("Data batch is empty")
        x, edge_index, batch = data.x, data.edge_index, data.batch
        A_gt = data.adj_dense  # (B, N, N)
        if A_gt.size(0) != 1:
            raise NotImplementedError("Current matching impl supports batch=1 only")

        # encoder
        h = F.relu(self.bn1(self.conv1(x, edge_index)))
        h = F.relu(self.bn2(self.conv2(h, edge_index)))
        g = self._pool(h, batch)
        mu, logvar = self.mu(g), self.logvar(g)
        z = self._reparam(mu, logvar)

        # decoder → logits vec & dense
        vec_logits = self.dec(z)
        A_hat_logits = vec_to_adj(vec_logits[0], self.max_n)

        # MPM + Hungarian
        degA = A_gt[0].sum(1)
        degB = A_hat_logits.sigmoid().detach().sum(1)
        S = self._edge_sim_tensor(A_gt[0], A_hat_logits.sigmoid(), degA, degB)
        init = torch.full((self.max_n, self.max_n), 1 / self.max_n, device=x.device)
        X = self._mpm(init, S)
        row, col = scipy.optimize.linear_sum_assignment(-X.cpu().numpy())
        row = torch.as_tensor(row, device=x.device)
        A_perm = A_gt[0][row][:, row]
        tri_truth = A_perm[torch.triu(torch.ones_like(A_perm)).bool()]

        loss_rec = F.binary_cross_entropy_with_logits(vec_logits[0], tri_truth)
        loss_kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        loss = loss_rec + loss_kl
        return loss, {"rec": loss_rec.detach(), "kl": loss_kl.detach()}

    # ---------------------------------------------------------------------
    # Toy forward_test (replicates the snippet you posted)
    # ---------------------------------------------------------------------
    def forward_test(self):
        """Run the 4‑node toy example from the user snippet to sanity‑check the
        MPM + Hungarian + permute logic. Prints intermediate results to stdout.
        """
        N = self.max_n = 4  # ensure consistency
        A = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0], [0, 1, 1, 1], [0, 0, 1, 1]], dtype=torch.float32)
        A1 = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1]], dtype=torch.float32)
        d, d1 = A.sum(1), A1.sum(1)

        S = self._edge_sim_tensor(A, A1, d, d1)
        X0 = torch.full((N, N), 1 / N)
        X = self._mpm(X0, S)
        r, c = scipy.optimize.linear_sum_assignment(-X.numpy())
        A_perm = A[r][:, r]

        tri_truth = A1[torch.triu(torch.ones_like(A1)).bool()]
        tri_pred = A_perm[torch.triu(torch.ones_like(A_perm)).bool()]
        diff = F.binary_cross_entropy(tri_pred, tri_truth)

        print("assignment matrix\n", X)
        print("row:", r, "col:", c)
        print("permuted adjacency:\n", A_perm)
        print("bce diff:", diff.item())


# -----------------------------------------------------------------------------
# Minimal smoke test
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print("--- smoke test (toy forward_test) ---")
    model = GraphVAE(in_dim=8, hid_dim=16, z_dim=8, max_nodes=4)
    model.forward_test()
