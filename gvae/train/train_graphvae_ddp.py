#!/usr/bin/env python
"""
1-node N-GPU DDP 版
  • dict / CLI 両対応
  • torchrun でも単体 python でも動く
  • rank-0 が TensorBoard + wandb を出力
"""

import os, argparse, yaml, torch, torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from types import SimpleNamespace
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
        return self.dec(z), mu, logvar   # logits

# ── DDP worker ──────────────────────────────────────────────────────
def _ddp_worker(rank: int, world: int, args):
    dist.init_process_group("nccl", rank=rank, world_size=world)
    torch.cuda.set_device(rank)

    ds = QM9(root=args.data_root)[: args.n_graph]
    sampler = DistributedSampler(ds, num_replicas=world, rank=rank, shuffle=True)
    loader  = DataLoader(ds, batch_size=1, sampler=sampler, pin_memory=True)

    model = GraphVAE(ds.num_features).to(rank)
    model = DDP(model, device_ids=[rank])
    opt   = torch.optim.Adam(model.parameters(), lr=1e-3)

    # ─ logging (rank-0 only) ─
    tb = SummaryWriter(os.path.join(args.log_dir, "tb")) if rank == 0 else None
    use_wandb = rank == 0 and os.getenv("WANDB_DISABLED", "false").lower() not in ("true", "1")
    if use_wandb:
        import wandb
        mode = "offline" if os.getenv("WANDB_MODE", "offline") == "offline" else "online"
        wandb.init(project="graph-vae", name="ddp-run", mode=mode, dir=args.log_dir,
                   config={k: getattr(args, k) for k in vars(args)})

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
            n = len(loader) * world
            L, R, K = totals / n
            print(f"  Loss {L:.3f}  Re {R:.3f}  KL {K:.3f}")
            tb.add_scalars("loss", {"total": L, "recon": R, "kl": K}, ep)
            if use_wandb:
                wandb.log({"loss/total": L, "loss/recon": R, "loss/kl": K, "epoch": ep})

    if rank == 0:
        save_dir = os.path.abspath(args.log_dir)
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, "graphvae_ddp_amp.pt")
        torch.save(model.module.state_dict(), path)
        print("✔ Saved", path)
        tb.close()
        if use_wandb:
            wandb.save(path)

    dist.destroy_process_group()

# ── public train() : dict / Namespace OK ────────────────────────────
def train(cfg):
    if isinstance(cfg, dict):
        cfg = cfg.get("trainer", cfg)          # hydra nesting OK
        cfg = SimpleNamespace(**cfg)

    defaults = dict(
        epochs=80, n_graph=3000,
        data_root="/dataset/QM9",
        log_dir="runs"
    )
    for k, v in defaults.items():
        if not hasattr(cfg, k):
            setattr(cfg, k, v)

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        _ddp_worker(int(os.environ["RANK"]),
                    int(os.environ["WORLD_SIZE"]), cfg)
    else:
        world = torch.cuda.device_count()
        torch.multiprocessing.spawn(_ddp_worker,
                                    nprocs=world,
                                    args=(world, cfg))

# ── CLI / torchrun entry ────────────────────────────────────────────
if __name__ == "__main__":
    pa = argparse.ArgumentParser()
    pa.add_argument("--config", help="yaml file (overrides CLI)")
    pa.add_argument("--epochs", type=int, default=80)
    pa.add_argument("--n_graph", type=int, default=3000)
    pa.add_argument("--data_root", default="/dataset/QM9")
    pa.add_argument("--log_dir",  default="runs")
    cli = pa.parse_args()

    if cli.config:
        with open(cli.config) as f:
            cfg = yaml.safe_load(f)
        # allow CLI args to override YAML values
        tr = cfg.get("trainer", cfg)
        tr["epochs"] = cli.epochs
        tr["n_graph"] = cli.n_graph
        tr["data_root"] = cli.data_root
        tr["log_dir"] = cli.log_dir
        cfg["trainer"] = tr
        train(cfg)
    else:
        train(cli)
