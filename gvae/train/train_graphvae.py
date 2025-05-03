#!/usr/bin/env python
"""
Single-GPU / CPU 兼用版 ―― AMP 対応安全版
dict でも Namespace でも呼び出せる train() を提供
"""
import os, argparse, torch, torch.nn as nn, torch.nn.functional as F
from types import SimpleNamespace
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from gvae.models.graphvae_core import GEncoder, EdgeMLP
from scipy.optimize import linear_sum_assignment

# ---------- Hungarian で真値隣接を並び替え -------------------------
def permute_adj(A_true, A_prob):
    eps = 1e-9
    cost = -(A_true * torch.log(A_prob + eps) +
             (1 - A_true) * torch.log(1 - A_prob + eps))
    r, c = linear_sum_assignment(cost.detach().cpu().numpy())
    P = torch.zeros_like(A_true); P[r, c] = 1.0
    return P.T @ A_true @ P
# -------------------------------------------------------------------

class GraphVAE(nn.Module):
    def __init__(self, in_dim, hid=64, z_dim=32):
        super().__init__()
        self.enc = GEncoder(in_dim, hid, z_dim)
        self.dec = EdgeMLP(z_dim)

    def forward(self, data):
        mu, logvar = self.enc(data.x, data.edge_index)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        logits = self.dec(z)                 # (N,N) ロジット
        return logits, mu, logvar


# ───────────────────────────────────────────────────────────────
# 内部ループ ―― ロジックは元の train() をそのまま移植
# ───────────────────────────────────────────────────────────────
def _train_inner(args):
    # -------- device 判定 ----------
    device = (torch.device("cuda" if torch.cuda.is_available() else "cpu")
              if args.device == "auto" else torch.device(args.device))
    print("Device =", device)

    # -------- dataset / loader -----
    ds = QM9(root=args.data_root).shuffle()[: args.n_graph]
    loader = DataLoader(ds,
                        batch_size=1,
                        shuffle=True,
                        pin_memory=(device.type == "cuda"))

    # -------- model / opt ----------
    model  = GraphVAE(ds.num_features).to(device)
    opt    = torch.optim.Adam(model.parameters(), lr=1e-3)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")

    os.makedirs("runs", exist_ok=True)

    for ep in range(args.epochs):
        L = R = K = 0.0
        # 線形アニーリング
        neg_w = args.neg_w_start - (args.neg_w_start - args.neg_w_end) * ep / args.epochs
        l1    = args.l1_start  - (args.l1_start  - args.l1_end ) * ep / args.epochs
        kl_w  = min(1.0, ep / args.kl_w_warmup) if args.anneal else 1.0

        for data in loader:
            data = data.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=args.amp and device.type == "cuda"):
                logits, mu, logvar = model(data)
                prob = torch.sigmoid(logits.detach())      # 0-1 確率 (Hungarian 用)

                # 真値隣接行列
                A_true = torch.zeros_like(prob)
                A_true[data.edge_index[0], data.edge_index[1]] = 1
                A_perm = permute_adj(A_true, prob)

                # BCEWithLogits は AMP に安全
                bce_all = F.binary_cross_entropy_with_logits(
                              logits, A_perm, reduction="none")
                recon = ((1 - A_perm) * neg_w * bce_all + A_perm * bce_all).mean()

                kl   = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                loss = recon + kl_w * kl + l1 * logits.abs().mean()

            # backward (AMP)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            opt.zero_grad(set_to_none=True)

            L += loss.item(); R += recon.item(); K += kl.item()

        print(f"Ep {ep:02d} | Loss {L/len(loader):.3f} "
              f"Re {R/len(loader):.3f} KL {K/len(loader):.3f} "
              f"(neg_w={neg_w:.1f}, l1={l1:.1e}, klw={kl_w:.2f})")

    torch.save(model.state_dict(), "runs/graphvae_gpu_amp.pt")
    print("✔ Saved to runs/graphvae_gpu_amp.pt")


# ───────────────────────────────────────────────────────────────
# 外部 API  ―― dict または Namespace を受け付ける
# ───────────────────────────────────────────────────────────────
def train(cfg):
    """
    Parameters
    ----------
    cfg : dict  または  argparse.Namespace / SimpleNamespace
        • dict の場合 → SimpleNamespace へ変換して _train_inner に渡す
        • Namespace の場合 → そのまま流用
    """
    # YAML が {"trainer": {...}} の形なら深い辞書を一段下げる
    if isinstance(cfg, dict) and "trainer" in cfg:
        cfg = cfg["trainer"]

    # dict → Namespace に変換
    if isinstance(cfg, dict):
        cfg = SimpleNamespace(**cfg)

    # 必須パラメータが dict に無かった場合のデフォルト
    defaults = dict(
        epochs=50, n_graph=1000, data_root="data/QM9", device="auto",
        amp=False, neg_w_start=40.0, neg_w_end=5.0,
        l1_start=1e-3, l1_end=5e-5, kl_w_warmup=30, anneal=False,
    )
    for k, v in defaults.items():
        if not hasattr(cfg, k):
            setattr(cfg, k, v)

    _train_inner(cfg)


# ───────────────────────────────────────────────────────────────
# 旧 CLI インターフェース（後方互換）
# ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    pa = argparse.ArgumentParser()
    pa.add_argument("--epochs",     type=int,   default=50)
    pa.add_argument("--n_graph",    type=int,   default=1000)
    pa.add_argument("--data_root",  type=str,   default="data/QM9")
    pa.add_argument("--device",     choices=["auto", "cpu", "cuda"], default="auto")
    pa.add_argument("--amp",        action="store_true")

    # Dynamic weights
    pa.add_argument("--neg_w_start", type=float, default=40.0)
    pa.add_argument("--neg_w_end",   type=float, default=5.0)
    pa.add_argument("--l1_start",    type=float, default=1e-3)
    pa.add_argument("--l1_end",      type=float, default=5e-5)
    pa.add_argument("--kl_w_warmup", type=int,   default=30)
    pa.add_argument("--anneal",      action="store_true")

    _train_inner(pa.parse_args())
