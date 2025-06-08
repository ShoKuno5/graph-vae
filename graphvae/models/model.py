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
from torch_geometric.nn import GCNConv, global_add_pool, global_mean_pool
from torch_geometric.data import Batch

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

"""def vec_to_adj(vec: torch.Tensor, N: int) -> torch.Tensor:
    # Upper‑triangular vector → dense symmetric adjacency.
    adj = vec.new_zeros(N, N)
    idx = torch.triu_indices(N, N)
    adj[idx[0], idx[1]] = vec
    adj = adj + adj.T - torch.diag(adj.diag())
    return adj"""

def vec_to_adj(vec: torch.Tensor, N: int) -> torch.Tensor:
    """
    上三角（対角なし）ベクトル → 対称隣接行列 (N,N)
    """
    adj = vec.new_zeros(N, N)
    idx = torch.triu_indices(N, N, offset=1)       # ← 対角を飛ばす
    adj[idx[0], idx[1]] = vec
    # adj = adj + adj.T                              # 対称にする
    adj = adj + adj.T
    idx = torch.arange(N, device=adj.device)
    adj[idx, idx] = 1.0                     # ★ 生成側も対角=1
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
        #tri = max_nodes * (max_nodes + 1) // 2
        off_diag = max_nodes * (max_nodes - 1) // 2
        self.dec = nn.Sequential(
            nn.Linear(z_dim, hid_dim), nn.ReLU(),
            nn.Linear(hid_dim, off_diag),
            #nn.Tanh()  
        )

        self._init_weights()

    # ---------------- utils ----------------
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain("relu"))
                nn.init.zeros_(m.bias)
            elif isinstance(m, GCNConv):
                nn.init.xavier_uniform_(m.lin.weight,
                                        gain=nn.init.calculate_gain("relu"))
                if m.lin.bias is not None:
                    nn.init.zeros_(m.lin.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)


    @staticmethod
    def _reparam(mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def _pool_add(self, h, batch):
        if self.pool == "sum":
            return global_add_pool(h, batch)
        return global_add_pool(h, batch) / torch.bincount(batch, minlength=batch.max() + 1).unsqueeze(-1).type_as(h)
    
    def _pool(self, h, batch):
        """Graph-level readout.
        h : (num_nodes, hidden_dim)
        batch : (num_nodes,) -- each node’s graph-index
        """
        return global_mean_pool(h, batch)   # ← ここだけで OK

    # ---------- permutation‑matching utilities ----------
    def edge_sim_tensor_loop(self, A, B, degA, degB):
        A = A.float();  B = B.float()
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
    
    def _edge_sim_tensor(self, A, B, degA, degB):
        # ---- 型を float に統一 ---------------------------------------------
        A = A.float()
        B = B.float()
        N = self.max_n

        # ---- ノード類似度 ---------------------------------------------------
        node_sim = deg_sim(degA.view(-1, 1), degB.view(1, -1))   # (N,N)

        # ---- エッジ類似度（ブロードキャストで N⁴ 要素を一気に計算） ----------
        #AA = A.unsqueeze(1).unsqueeze(3)     # (N,1,N,1)
        #BB = B.unsqueeze(0).unsqueeze(2)     # (1,N,1,N)
        #S  = (AA * BB)                       # (N,N,N,N)  float32
        
        # ---- エッジ類似度（正しい i-j, a-b 対応 & 対角係数込み） ---------------
        A_pair = A.unsqueeze(2).unsqueeze(3)      # (N,N,1,1) → A[i,j]
        B_pair = B.unsqueeze(0).unsqueeze(1)      # (1,1,N,N) → B[a,b]

        Ai = A.diag().view(N, 1, 1, 1)            # A[i,i]
        Aj = A.diag().view(1, N, 1, 1)            # A[j,j]
        Ba = B.diag().view(1, 1, N, 1)            # B[a,a]
        Bb = B.diag().view(1, 1, 1, N)            # B[b,b]

        S = A_pair * Aj * Ai * B_pair * Ba * Bb   # (N,N,N,N)        

        # ---- “片側だけ対角” の項目を 0 にする (i==j XOR a==b) ---------------
        eye = torch.eye(N, dtype=torch.bool, device=A.device)
        i_eq_j = eye.unsqueeze(2).unsqueeze(3)   # (N,N,1,1)
        a_eq_b = eye.unsqueeze(0).unsqueeze(1)   # (1,1,N,N)
        S[i_eq_j ^ a_eq_b] = 0                  # xor マスクで一括 0 埋め

        # ---- 対角ブロック S[i,i,a,a] ← A[i,i]·B[a,a]·deg_sim --------------
        Ai_diag = A.diag().view(N, 1)          # (N,1)
        Ba_diag = B.diag().view(1, N)          # (1,N)
        node_block = Ai_diag * Ba_diag * node_sim   # (N,N)

        idx = torch.arange(N, device=A.device)
        ii  = idx[:, None]                     # broadcast → dims (i,i)
        aa  = idx[None, :]                     # broadcast → dims (a,a)
        S[ii, ii, aa, aa] = node_block        # S[i,i,a,a] に一括代入
     
        return S


    def _mpm_loop(self, X0, S, iters: int = 50):
        """Max-pool-matching (MPM) producing an (N, N) assignment matrix."""
        X = X0.clone()                            # (N, N)
        N = self.max_n
        for _ in range(iters):
            X_new = torch.zeros_like(X)

            # --- node-similarity項 ---
            for i in range(N):
                for a in range(N):
                    X_new[i, a] = X[i, a] * S[i, i, a, a]

            # --- edge-similarity項 ---
            for i in range(N):
                for a in range(N):
                    edge_sum = 0.0
                    for j in range(N):
                        if j == i:               # i ≠ j
                            continue
                        # 各 j について b 方向の max を取る
                        max_b = torch.max(X[j] * S[i, j, a])
                        edge_sum += max_b
                    X_new[i, a] += edge_sum

            # ℓ2 正規化
            X = X_new / X_new.norm(p=2)

        return X                                  # (N, N)

    def _mpm(self, X0: torch.Tensor, S: torch.Tensor, iters: int = 50, tol: float = 1e-4, early_stop: bool = True):
        X  = X0.clone()                    # (N,N)
        N  = X.size(0)
        idx = torch.arange(N, device=X.device)

        # S[i,i,a,a] → (N,N)
        node_sim = S[idx[:, None], idx[:, None], idx[None, :], idx[None, :]]

        mask = (~torch.eye(N, dtype=torch.bool, device=X.device)).unsqueeze(-1)  # (N,N,1)

        for _ in range(iters):
            X_prev  = X
            node_term = X * node_sim                               # (N,N)

            XS_max = (S * X.unsqueeze(0).unsqueeze(2)).amax(-1)    # (N,N,N)
            edge_term = (XS_max * mask).sum(dim=1)                 # (N,N)

            X = (node_term + edge_term)
            X = X / X.norm(p=2)                                    # ℓ2 正規化

            # if (X - X_prev).abs().max() < 1e-4:                    # 早期収束
            if early_stop and (X - X_prev).abs().max() < tol:      # 早期収束
                break

        return X



    # ---------- canonical forward (training) ----------
    
    def forward(self, data):
        # ❶ Data / Batch をリスト化 -------------------------------------------------
        graphs = data.to_data_list() if isinstance(data, Batch) else [data]

        loss_rec_all, loss_kl_all, max_logit_all = [], [], []  # ❸ loss のリスト

        # ❷ 1 グラフずつ処理 --------------------------------------------------------
        for g in graphs:
            # --- Batch が無い単一グラフにも対応 -------------------------------------
            if getattr(g, "batch", None) is None:
                g.batch = torch.zeros(g.num_nodes,
                                    dtype=torch.long,
                                    device=g.x.device)

            x, edge_index, batch = g.x, g.edge_index, g.batch
            A_gt = g.adj_dense.squeeze(0)                # (N,N)

            # ---------- encoder -----------------------------------------------------
            h = F.relu(self.bn1(self.conv1(x, edge_index)))
            h = F.relu(self.bn2(self.conv2(h, edge_index)))
            g_ = self._pool(h, batch)
            mu, logvar = self.mu(g_), self.logvar(g_)
            logvar = logvar.clamp(min=-4.0, max=4.0)  # より広い範囲に変更

            # ↓ ここを追加 -------------------------------------------------------
            self._latest_stats = (mu.detach(), logvar.detach())  # ★
            # -------------------------------------------------------------------
            z = self._reparam(mu, logvar)

            # ---------- decoder -----------------------------------------------------
            vec_logits   = self.dec(z)                    # (U,)
            max_logit_i  = vec_logits.detach().abs().max()   # ★ 追加
            A_hat_logits = vec_to_adj(vec_logits, self.max_n)

            # ---------- MPM + Hungarian --------------------------------------------
            degA = A_gt.sum(1)
            degB = A_hat_logits.sigmoid().detach().sum(1)
            S    = self._edge_sim_tensor(A_gt,
                                        A_hat_logits.sigmoid(),
                                        degA, degB)

            init = torch.full((self.max_n, self.max_n),
                            1 / self.max_n,
                            device=x.device)
            X = self._mpm(init, S)

            with torch.no_grad():
                cost_np = (-X.detach().cpu()).numpy()
                cost_np = np.nan_to_num(cost_np, nan=1e6,
                                        posinf=1e6, neginf=-1e6)
                row_np, col_np = scipy.optimize.linear_sum_assignment(cost_np)

            row = torch.as_tensor(row_np, device=X.device)
            col = torch.as_tensor(col_np, device=X.device)
            perm = torch.empty_like(col); perm[col] = row

            # ---------- permute both GT and prediction ------------------------------
            A_gt_perm        = A_gt[perm][:, perm]
            # A_hat_perm_logits = A_hat_logits[perm][:, perm]
            A_hat_perm_logits = A_hat_logits          # ← そのまま使う

            #idx = torch.triu_indices(self.max_n,
            #                        self.max_n,
            #                        offset=1,
            #                        device=x.device)
            #tri_truth = A_gt_perm[idx[0], idx[1]]              # (U,)
            #tri_pred  = A_hat_perm_logits[idx[0], idx[1]]      # (U,)
            
            R = int(g.num_real_nodes) 
            idx = torch.triu_indices(self.max_n,
                             self.max_n,
                             offset=1,
                             device=x.device)
            valid = (idx[0] < R) & (idx[1] < R)            # 👈 マスク
            tri_truth = A_gt_perm[idx[0][valid], idx[1][valid]]
            tri_pred  = A_hat_perm_logits[idx[0][valid], idx[1][valid]]

            # ---------- loss --------------------------------------------------------
            # loss_rec_i = F.binary_cross_entropy_with_logits(tri_pred, tri_truth)
            # ① rho を「実ノードだけ」の上三角から計算 -----------------------★
            num_pos = tri_truth.sum()                          # 正例 (=1) 本数
            R = int(g.num_real_nodes)
            total_possible = R * (R - 1) / 2                   # 完全グラフの上三角
            rho = num_pos / (total_possible + 1e-8)            # 0 除けの ε
            
            # ② pos_weight = (1-ρ)/ρ でクラス不均衡を補正 --------------------★
            pos_weight = (1 - rho) / (rho + 1e-8)              # Tensor 型
            pos_weight = pos_weight.clamp(max = 20)
            
            # ③ BCEWithLogits に渡す -----------------------------------------★
            loss_rec_i = F.binary_cross_entropy_with_logits(
                tri_pred, tri_truth,
                pos_weight=pos_weight
            )

            # （KL はそのまま）
            loss_kl_i = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss_rec_all.append(loss_rec_i)
            loss_kl_all.append(loss_kl_i)
            max_logit_all.append(max_logit_i)                # ★ 追加

        # ❸ 勾配をまとめる ----------------------------------------------------------
        loss_rec = torch.stack(loss_rec_all).mean()
        loss_kl  = torch.stack(loss_kl_all).mean()
        max_logit   = torch.stack(max_logit_all).max()       # ★ 追加
        loss     = loss_rec + loss_kl

        return {"rec": loss_rec, "kl": loss_kl, "max_logit": max_logit, "logvar_max": logvar.max()}

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
        perm = torch.argsort(torch.as_tensor(c))
        A_perm = A[perm][:, perm]

        tri_truth = A1[torch.triu(torch.ones_like(A1)).bool()]
        tri_pred = A_perm[torch.triu(torch.ones_like(A_perm)).bool()]
        diff = F.binary_cross_entropy(tri_pred, tri_truth)

        print("assignment matrix\n", X)
        print("row:", r, "col:", c)
        print("permuted adjacency:\n", A_perm)
        print("bce diff:", diff.item())

# -----------------------------------------------------------------------------
# Test edge similarity tensor functions
# -----------------------------------------------------------------------------

def test_edge_sim_equivalence(N: int = 8):
    """
    Test equivalence of edge_sim_tensor_loop and _edge_sim_tensor.
    """
    # グラフ行列生成
    A = torch.randint(0, 2, (N, N), dtype=torch.float32).triu(1)
    A = A + A.T; A.fill_diagonal_(1)
    B = torch.randint(0, 2, (N, N), dtype=torch.float32).triu(1)
    B = B + B.T; B.fill_diagonal_(1)
    dA, dB = A.sum(1), B.sum(1)

    # メソッド呼び出しにはダミーモデルを利用
    model = GraphVAE(in_dim=1, hid_dim=1, z_dim=1, max_nodes=N)
    S_loop = model.edge_sim_tensor_loop(A, B, dA, dB)
    S_fast = model._edge_sim_tensor(A, B, dA, dB)
    assert torch.allclose(S_loop, S_fast)

    eq = torch.allclose(S_loop, S_fast)
    print(f"edge sim equivalence (N={N}): {eq}")
    return eq


# ---------------------------------------------------------------------
# MPM-specific unit tests
# ---------------------------------------------------------------------
def _rand_sym_adj(N, p=0.3):
    """Random symmetric (0/1) adjacency with self-loops (=1)."""
    A = (torch.rand(N, N) < p).float().triu(1)
    A = A + A.T
    A.fill_diagonal_(1.0)
    return A


def test_mpm_equivalence(N: int = 8, iters: int = 20, tol: float = 1e-4):
    """
    _mpm_loop と _mpm が (ほぼ) 同じ出力になるか判定。
    """
    torch.manual_seed(0)
    model = GraphVAE(in_dim=1, hid_dim=1, z_dim=1, max_nodes=N)

    # ランダム A/B で S を作る
    A, B = _rand_sym_adj(N), _rand_sym_adj(N)
    dA, dB = A.sum(1), B.sum(1)
    S  = model._edge_sim_tensor(A, B, dA, dB)

    X0 = torch.rand(N, N)
    X0 /= X0.norm()

    out_loop = model._mpm_loop(X0, S, iters=iters)
    # out_fast = model._mpm(X0, S, iters=iters)
    out_fast = model._mpm(X0, S, iters=iters, early_stop=False)

    assert torch.allclose(out_loop, out_fast, atol=tol), \
        f"MPM mismatch: {torch.max((out_loop-out_fast).abs())}"
    print(f"[✓] MPM equivalence   N={N}, iters={iters}, max |Δ|={torch.max((out_loop-out_fast).abs()):.2e}")


def test_mpm_gradients(N: int = 8, iters: int = 10):
    """
    _mpm が autograd に対応しているかチェック。
    (_mpm_loop はテスト対象外：速度的に不要)
    """
    torch.manual_seed(1)
    model = GraphVAE(in_dim=1, hid_dim=1, z_dim=1, max_nodes=N)

    S  = torch.rand(N, N, N, N, requires_grad=False)
    X0 = torch.rand(N, N, requires_grad=True)
    loss = model._mpm(X0, S, iters=iters).sum()
    loss.backward()

    assert torch.isfinite(X0.grad).all(), "Gradient contains inf / nan"
    print(f"[✓] MPM gradients     N={N}, iters={iters}")


def test_mpm_speed(N: int = 16, iters: int = 50):
    """
    粗いベンチ：_mpm が _mpm_loop より速いことだけ確認。
    """
    import time
    torch.manual_seed(2)
    model = GraphVAE(in_dim=1, hid_dim=1, z_dim=1, max_nodes=N)
    S  = torch.rand(N, N, N, N)
    X0 = torch.rand(N, N)
    X0 /= X0.norm()

    t0 = time.perf_counter()
    model._mpm_loop(X0, S, iters=iters)
    t_loop = time.perf_counter() - t0

    t0 = time.perf_counter()
    model._mpm(X0, S, iters=iters)
    t_fast = time.perf_counter() - t0

    speedup = t_loop / t_fast
    assert speedup > 5, f"_mpm speedup too small: {speedup:.1f}×"
    print(f"[✓] MPM speed         N={N}, iters={iters}, {speedup:.1f}× faster")

# -----------------------------------------------------------------------------
# Minimal smoke test
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print("--- smoke test (toy forward_test) ---")
    model = GraphVAE(in_dim=8, hid_dim=16, z_dim=8, max_nodes=4)
    model.forward_test()
    # 追加：edge similarity テスト関数の呼び出し
    test_edge_sim_equivalence()
    
    print("\n--- MPM unit-tests ---")
    test_mpm_equivalence()
    test_mpm_gradients()
    test_mpm_speed()
    print("----------------------\n")