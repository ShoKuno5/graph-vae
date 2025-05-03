#!/usr/bin/env python
"""
1-ノード N-GPU (spawn) 版  ――  ロジット ⇒ BCEWithLogitsLoss に統一
"""
import os, argparse, random, numpy as np, torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from gvae.models.graphvae_core import GEncoder, EdgeMLP
from scipy.optimize import linear_sum_assignment

# ---------------- Hungarian (確率で計算) ----------------
def permute_adj(A_true, prob, eps=1e-9):
    # prob ∈ [0,1] でないと log が発散するので clamp
    P = torch.clamp(prob, eps, 1 - eps)
    cost = -(A_true * torch.log(P) + (1 - A_true) * torch.log(1 - P))
    r, c = linear_sum_assignment(cost.detach().cpu().numpy())
    M = torch.zeros_like(A_true); M[r, c] = 1.0
    return M.T @ A_true @ M
# --------------------------------------------------------

class GraphVAE(torch.nn.Module):
    """Encoder → z → EdgeMLP → **logit**"""
    def __init__(self, in_dim, hid=64, z_dim=32):
        super().__init__()
        self.enc = GEncoder(in_dim, hid, z_dim)
        self.dec = EdgeMLP(z_dim)
    def forward(self, data):
        mu, logvar = self.enc(data.x, data.edge_index)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        logits = self.dec(z)              # (N,N)
        return logits, mu, logvar

def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def main(rank, world_size, args):
    setup(rank, world_size)
    torch.manual_seed(42 + rank)

    ds = QM9(root=args.data_root)[: args.n_graph]
    sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=True)
    loader  = DataLoader(ds, batch_size=1, sampler=sampler, pin_memory=True)

    model = GraphVAE(ds.num_features).to(rank)
    model = DDP(model, device_ids=[rank])
    opt   = torch.optim.Adam(model.parameters(), lr=1e-3)

    for ep in range(args.epochs):
        sampler.set_epoch(ep)
        tot_loss = tot_recon = tot_kl = 0.0

        if rank == 0:
            print(f"Epoch {ep:02d}")

        for data in loader:
            data = data.to(rank, non_blocking=True)
            logits, mu, logvar = model(data)
            prob = torch.sigmoid(logits.detach())         # 0-1 確率

            A_true = torch.zeros_like(prob)
            A_true[data.edge_index[0], data.edge_index[1]] = 1
            A_perm = permute_adj(A_true, prob)            # permuted GT

            bce_all = F.binary_cross_entropy_with_logits(logits, A_perm, reduction="none")
            recon   = bce_all.mean()
            kl      = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss    = recon + kl

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            tot_loss += loss.item(); tot_recon += recon.item(); tot_kl += kl.item()

        totals = torch.tensor([tot_loss, tot_recon, tot_kl], device=rank)
        dist.all_reduce(totals)                      # 平均表示用
        if rank == 0:
            n = len(loader) * world_size
            print(f"  Loss {totals[0]/n:.3f}  Re {totals[1]/n:.3f}  KL {totals[2]/n:.3f}")

    if rank == 0:
        torch.save(model.module.state_dict(), "runs/graphvae_ddp_amp.pt")
        print("✔ Saved runs/graphvae_ddp_amp.pt")
    cleanup()

if __name__ == "__main__":
    pa = argparse.ArgumentParser()
    pa.add_argument("--epochs", type=int, default=80)
    pa.add_argument("--n_graph", type=int, default=3000)
    pa.add_argument("--data_root", default="/dataset/QM9")
    args = pa.parse_args()

    world = torch.cuda.device_count()         # (=8 on short-a node)
    torch.multiprocessing.spawn(main, nprocs=world, args=(world, args))
