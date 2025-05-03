#!/usr/bin/env python
"""
1-node N-GPU Distributed Data Parallel version
• keeps the original spawn-based logic
• **adds dict-API + YAML Config support**  so you can call it from torchrun
"""

import os, argparse, yaml, torch, torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler
from types import SimpleNamespace
import os
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from gvae.models.graphvae_core import GEncoder, EdgeMLP
from scipy.optimize import linear_sum_assignment

# ── util ────────────────────────────────────────────────────────────
def permute_adj(A_true, prob, eps=1e-9):
    P = torch.clamp(prob, eps, 1 - eps)
    cost = -(A_true * torch.log(P) + (1 - A_true) * torch.log(1 - P))
    r, c = linear_sum_assignment(cost.detach().cpu().numpy())
    M = torch.zeros_like(A_true); M[r, c] = 1.0
    return M.T @ A_true @ M

class GraphVAE(torch.nn.Module):
    def __init__(self, in_dim, hid=64, z_dim=32):
        super().__init__()
        self.enc = GEncoder(in_dim, hid, z_dim)
        self.dec = EdgeMLP(z_dim)
    def forward(self, data):
        mu, logvar = self.enc(data.x, data.edge_index)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        return self.dec(z), mu, logvar                            # logits

# ── DDP core  ───────────────────────────────────────────────────────
def _ddp_worker(rank, world_size, args):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    ds = QM9(root=args.data_root)[: args.n_graph]
    sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=True)
    loader  = DataLoader(ds, batch_size=1, sampler=sampler, pin_memory=True)

    model = GraphVAE(ds.num_features).to(rank)
    model = DDP(model, device_ids=[rank])
    opt   = torch.optim.Adam(model.parameters(), lr=1e-3)

    for ep in range(args.epochs):
        sampler.set_epoch(ep)
        loss_sum = recon_sum = kl_sum = 0.0

        if rank == 0:
            print(f"Epoch {ep:02d}")

        for data in loader:
            data = data.to(rank, non_blocking=True)
            logits, mu, logvar = model(data)
            prob = torch.sigmoid(logits.detach())

            A_true = torch.zeros_like(prob)
            A_true[data.edge_index[0], data.edge_index[1]] = 1
            A_perm = permute_adj(A_true, prob)

            bce = F.binary_cross_entropy_with_logits(logits, A_perm, reduction="none").mean()
            kl  = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss = bce + kl

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            loss_sum += loss.item(); recon_sum += bce.item(); kl_sum += kl.item()

        totals = torch.tensor([loss_sum, recon_sum, kl_sum], device=rank)
        dist.all_reduce(totals)
        if rank == 0:
            n = len(loader) * world_size
            print(f"  Loss {totals[0]/n:.3f}  Re {totals[1]/n:.3f}  KL {totals[2]/n:.3f}")

    if rank == 0:
        os.makedirs(args.log_dir, exist_ok=True)
        torch.save(model.module.state_dict(), f"{args.log_dir}/graphvae_ddp_amp.pt")
        print("✔ Saved", f"{args.log_dir}/graphvae_ddp_amp.pt")
    dist.destroy_process_group()

# ── public API : dict も Namespace も受け付ける ────────────────────
def train(cfg):
    """cfg : dict  or argparse/​SimpleNamespace"""
    if isinstance(cfg, dict):
        if "trainer" in cfg:          # hydra-like nesting
            cfg = cfg["trainer"]
        cfg = SimpleNamespace(**cfg)

    # default values if missing
    defaults = dict(
        epochs=80, n_graph=3000,
        data_root="/dataset/QM9", log_dir="runs",
    )
    for k, v in defaults.items():
        if not hasattr(cfg, k):
            setattr(cfg, k, v)

        # torchrun で来たか？（RANK, WORLD_SIZE が環境に入る）
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank  = int(os.environ["RANK"])
        world = int(os.environ["WORLD_SIZE"])
        _ddp_worker(rank, world, cfg)          # そのまま実行
    else:
        world = torch.cuda.device_count()      # 単独 python 実行 → spawn
        torch.multiprocessing.spawn(_ddp_worker,
                                    nprocs=world,
                                    args=(world, cfg))


# ── CLI / torchrun entry ────────────────────────────────────────────
if __name__ == "__main__":
    pa = argparse.ArgumentParser()
    pa.add_argument("--config", help="yaml config (overrides CLI)")
    pa.add_argument("--epochs", type=int, default=80)
    pa.add_argument("--n_graph", type=int, default=3000)
    pa.add_argument("--data_root", default="/dataset/QM9")
    pa.add_argument("--log_dir", default="runs")
    cli_args = pa.parse_args()

    if cli_args.config:
        with open(cli_args.config) as f:
            cfg = yaml.safe_load(f)
        train(cfg)                         # dict path
    else:
        train(cli_args)                    # Namespace path
