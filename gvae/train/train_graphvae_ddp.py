#!/usr/bin/env python
"""
1-node N-GPU DDP
train_graphvae_stable.py と同一アルゴリズム（logits 対応）
"""

import os, random, argparse, yaml, torch, numpy as np, torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from types import SimpleNamespace
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from gvae.models.graphvae_core import GEncoder, EdgeMLP
from scipy.optimize import linear_sum_assignment

# ─────────── stable と同じハイパーパラメータ ───────────
SEED     = 42
EPOCHS   = 80
NEG_W0, NEG_W1 = 40., 1.
L1_0,  L1_1    = 1e-3, 0.0
KL_WARM        = 60
MINDEG_W       = 0.25
# -----------------------------------------------------

def _set_seed():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.use_deterministic_algorithms(True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

# ── util ──────────────────────────────────────────────
def permute_adj(A_true, prob, eps=1e-9):
    P    = torch.clamp(prob, eps, 1 - eps)
    cost = -(A_true * torch.log(P) + (1 - A_true) * torch.log(1 - P))
    r, c = linear_sum_assignment(cost.detach().cpu().numpy())
    M    = torch.zeros_like(A_true); M[r, c] = 1.0
    return M.T @ A_true @ M

class GraphVAE(torch.nn.Module):
    def __init__(self, in_dim, hid=64, z_dim=32):
        super().__init__()
        self.enc = GEncoder(in_dim, hid, z_dim)
        self.dec = EdgeMLP(z_dim)  # logits
    def forward(self, data):
        mu, logvar = self.enc(data.x, data.edge_index)
        z   = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        logits = self.dec(z)              # logits (unbounded)
        prob   = torch.sigmoid(logits)    # ✎ 追加 — 確率も返す
        return logits, prob, mu, logvar

# ── DDP worker ────────────────────────────────────────
def _ddp_worker(rank: int, world: int, args):
    _set_seed()
    dist.init_process_group("nccl", rank=rank, world_size=world)
    torch.cuda.set_device(rank)

    ds       = QM9(root=args.data_root)[: args.n_graph]
    sampler  = DistributedSampler(ds, num_replicas=world, rank=rank, shuffle=True)
    loader   = DataLoader(ds, batch_size=1, sampler=sampler, pin_memory=True)

    model = GraphVAE(ds.num_features).to(rank)
    model = DDP(model, device_ids=[rank])
    opt   = torch.optim.Adam(model.parameters(), lr=1e-3)

    tb  = SummaryWriter(os.path.join(args.log_dir, "tb")) if rank == 0 else None
    use_wandb = rank == 0 and os.getenv("WANDB_DISABLED", "false").lower() not in ("true", "1")
    if use_wandb:
        import wandb
        wandb.init(project="graph-vae", name="ddp-stable", mode="offline",
                   dir=args.log_dir, config={k: getattr(args, k) for k in vars(args)})

    for ep in range(args.epochs):
        sampler.set_epoch(ep)

        neg_w = NEG_W0 - (NEG_W0 - NEG_W1) * ep / EPOCHS
        l1    = L1_0  - (L1_0  - L1_1 ) * ep / EPOCHS
        kl_w  = min(1.0, ep / KL_WARM)

        loss_sum = recon_sum = kl_sum = 0.0
        if rank == 0:
            print(f"Epoch {ep:02d}  (neg_w={neg_w:.1f}, l1={l1:.0e}, kl_w={kl_w:.2f})")

        for data in loader:
            data = data.to(rank, non_blocking=True)
            logits, prob, mu, logvar = model(data)

            # ground-truth adjacency
            A_true = torch.zeros_like(prob)
            A_true[data.edge_index[0], data.edge_index[1]] = 1
            A_perm = permute_adj(A_true, prob)

            # -------- loss --------
            bce = F.binary_cross_entropy_with_logits(logits, A_perm, reduction="none")  # ✎ 修正
            recon = ((1 - A_perm) * neg_w * bce + A_perm * bce).mean()
            kl   = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            deg  = prob.sum(-1)
            loss = recon + kl_w * kl + l1 * prob.sum() + MINDEG_W * torch.relu(1 - deg).mean()

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            loss_sum  += loss.item()
            recon_sum += recon.item()
            kl_sum    += kl.item()

        totals = torch.tensor([loss_sum, recon_sum, kl_sum], device=rank)
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
        if rank == 0:
            n = len(loader) * world
            L, R, K = totals / n
            print(f"  Loss {L:.3f} | Re {R:.3f} | KL {K:.3f}")
            tb.add_scalars("loss", {"total": L, "recon": R, "kl": K}, ep)
            if use_wandb:
                wandb.log({"loss/total": L, "loss/recon": R, "loss/kl": K,
                           "neg_w": neg_w, "l1": l1, "kl_w": kl_w, "epoch": ep})

    if rank == 0:
        path = os.path.join(os.path.abspath(args.log_dir), "graphvae_ddp_stable.pt")
        torch.save(model.module.state_dict(), path)
        print("✔ Saved", path)
        tb.close()
        if use_wandb:
            wandb.save(path)

    dist.destroy_process_group()

# ── public train() ───────────────────────────────────
def train(cfg):
    if isinstance(cfg, dict):
        cfg = cfg.get("trainer", cfg)
        cfg = SimpleNamespace(**cfg)

    defaults = dict(
        epochs   = EPOCHS,
        n_graph  = 1000,
        data_root= "/dataset/QM9",
        log_dir  = "runs/EXP",
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

# ── CLI entry ─────────────────────────────────────────
if __name__ == "__main__":
    pa = argparse.ArgumentParser()
    pa.add_argument("--config")
    pa.add_argument("--epochs",   type=int, default=EPOCHS)
    pa.add_argument("--n_graph",  type=int, default=1000)
    pa.add_argument("--data_root", default="/dataset/QM9")
    pa.add_argument("--log_dir",   default="runs")
    cli = pa.parse_args()

    if cli.config:
        with open(cli.config) as f:
            cfg = yaml.safe_load(f)
        tr = cfg.get("trainer", cfg)
        tr.update(vars(cli))
        cfg["trainer"] = tr
        train(cfg)
    else:
        train(cli)
