#!/usr/bin/env python
"""
PyG‑style, multi‑GPU DDP training script for GraphVAE.

* Replaces the original `otehon_train.py` data pipeline with torch‑geometric `Data` objects.
* Re‑uses the DDP/TensorBoard/W&B logging pattern from `my_train.py`.
* Works with the modernised `GraphVAE` in `model.py` (expects `data.adj_dense`).

Usage (single node, all GPUs):
    python -m torch.distributed.run --nproc_per_node=4 train_graphvae_ddp.py \
           --dataset grid --epochs 200 --log_dir runs/exp1

When running under SLURM/PJM, set the usual env vars (RANK, WORLD_SIZE, LOCAL_RANK) and
launch the script once per process.
"""

from __future__ import annotations

import argparse, os, random, math, yaml, time, itertools, warnings
from types import SimpleNamespace
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DistributedSampler

import networkx as nx
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import from_networkx, to_undirected
from torch_geometric.datasets import TUDataset

from graphvae.models.model import GraphVAE  # ← your modernised implementation

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
SEED = 42

def _set_seed():
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.use_deterministic_algorithms(False)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

# ---------------------------------------------------------------------------
# Dataset helpers —> turn any NetworkX graph into a fixed‑size PyG `Data`
# ---------------------------------------------------------------------------
class GraphDataset(torch.utils.data.Dataset):
    """Wraps a list of NetworkX graphs, padding to `max_nodes` and adding
    `adj_dense` so `GraphVAE.forward()` can compute its permutation loss."""

    def __init__(self, graphs, max_nodes: int, feature_type: str = "id") -> None:
        self.graphs = graphs
        self.max_nodes = max_nodes
        self.feature_type = feature_type

    # -------- helper: create node feature matrix --------
    def _node_features(self, g: nx.Graph) -> torch.Tensor:
        N = g.number_of_nodes()
        if self.feature_type == "id":
            x = torch.eye(self.max_nodes)
            return x[:N]  # (N, max_nodes)
        elif self.feature_type == "deg":
            deg = torch.tensor([d for _, d in g.degree()], dtype=torch.float32)
            x = torch.zeros(self.max_nodes, 1)
            x[:N, 0] = deg
            return x  # (max_nodes, 1)
        else:
            raise ValueError(f"Unknown feature_type: {self.feature_type}")

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        g = self.graphs[idx]
        # ensure deterministic node ordering 0..N-1
        mapping = {n: i for i, n in enumerate(g.nodes())}
        g = nx.relabel_nodes(g, mapping)

        x = self._node_features(g)
        data = from_networkx(g)
        data.edge_index = to_undirected(data.edge_index, num_nodes=g.number_of_nodes())
        data.x = x
        data.num_nodes = self.max_nodes  # pad virtual nodes implicitly

        # dense adjacency for loss
        adj = torch.zeros(self.max_nodes, self.max_nodes)
        ei = data.edge_index
        adj[ei[0], ei[1]] = 1
        adj = torch.maximum(adj, adj.T)  # undirected
        data.adj_dense = adj
        
        # 追加 : 実ノード数を保存（損失マスク用）
        data.num_real_nodes = torch.tensor(g.number_of_nodes(), dtype=torch.long)
        return data

# ---------------------------------------------------------------------------
# Graph generators matching the original baseline choices
# ---------------------------------------------------------------------------

def load_tu_dataset(name: str, root: str = "/tmp"):
    """DDP 環境でも安全に TUDataset をロードするユーティリティ."""
    rank = int(os.environ.get("RANK", 0))

    if rank == 0:
        td = TUDataset(root=root, name=name)   # ↓ rank0 だけ DL
        if dist.is_initialized():
            dist.barrier()                     # rank1+ を解放
    else:
        if dist.is_initialized():
            dist.barrier()                     # rank0 が終わるまで待機
        td = TUDataset(root=root, name=name)   # すでに raw があるので即ロード
    return td

def build_graph_list(dataset: str, max_nodes: int | None):
    if dataset == "grid":
        graphs = [nx.grid_2d_graph(i, j) 
                  for i, j in itertools.product(range(2, 4), repeat=2)]

    elif dataset == "enzymes":
        td = load_tu_dataset("ENZYMES", root="/tmp/TUD")   # ← ここだけ変更
        from torch_geometric.utils import to_networkx
        graphs = [to_networkx(g, to_undirected=True) for g in td]
    else:
        raise ValueError("dataset must be 'grid' or 'enzymes'")

    # 最大ノード数を超えるグラフは除外
    if max_nodes is None:
        max_nodes = max(g.number_of_nodes() for g in graphs)
    graphs = [g for g in graphs if g.number_of_nodes() <= max_nodes]

    return graphs, max_nodes


# ---------------------------------------------------------------------------
# Debug helpers
# ---------------------------------------------------------------------------
class DebugState:
    """Keeps step-wise stats and pushes them to TB/W&B."""
    def __init__(self, writer, wandb_run):
        self.tb = writer
        self.wb = wandb_run
        self.step = 0

    def log(self, tag, value):
        if self.tb:
            self.tb.add_scalar(tag, value, self.step)
        if self.wb:
            import wandb
            wandb.log({tag: value, "step": self.step})

    def next(self):
        self.step += 1


def finite_or_raise(t, name):
    if not torch.isfinite(t).all():
        raise FloatingPointError(f"{name} contains inf / nan")
    
# rank / debug フラグに関係なく dbg を持たせる
class _NoDbg:
    step = 0
    def log(self, *_, **__): pass
    def next(self): self.step += 1

# ---------------------------------------------------------------------------
# DDP worker
# ---------------------------------------------------------------------------

def _ddp_worker(rank: int, world: int, args: argparse.Namespace):
    _set_seed()
    t0 = time.perf_counter()
    dist.init_process_group("nccl", rank=rank, world_size=world)
    torch.cuda.set_device(rank)
    if rank == 0:
        print(f"[rank0] init_process_group ({time.perf_counter()-t0:.2f}s)", flush=True)

    t0 = time.perf_counter()
    graphs, max_nodes = build_graph_list(args.dataset, args.max_nodes)
    if rank == 0:
        print(f"[rank0] graph list ready ({time.perf_counter()-t0:.2f}s)", flush=True)
        
    t0 = time.perf_counter()
    ds = GraphDataset(graphs, max_nodes, args.feature_type)
    sampler = DistributedSampler(ds, num_replicas=world, rank=rank, shuffle=True)
    dl = DataLoader(ds, batch_size=1, sampler=sampler, pin_memory=True)
    if rank == 0:
        print(f"[rank0] DataLoader ready ({time.perf_counter()-t0:.2f}s)", flush=True)

    model = GraphVAE(in_dim=ds[0].x.size(-1), hid_dim=64, z_dim=32, max_nodes=max_nodes).to(rank)
    model = DDP(model, device_ids=[rank], find_unused_parameters=False)
    # opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
   
    
    # --- logger setup ----------------------------------------------------
    if rank == 0 and args.use_wandb and os.getenv("WANDB_DISABLED", "false").lower() not in ("true", "1"):
        import wandb
        use_wandb = True
        wb_run = wandb.init(project="graphvae",
                            name=args.run_name or f"{args.dataset}-ddp",
                            dir=args.log_dir,
                            config=vars(args))
    else:
        use_wandb = False
        wb_run = None

    tb = SummaryWriter(Path(args.log_dir, "tb")) if rank == 0 else None

    # --- Debug setup ----------------------------------------------------
    beta0, beta_final = 0.1, 1.0          # KL weight annealing
    warmup_steps = 100                    # LR warm-up
    clip_max = 5.0                        # grad clip
    
    dbg = (DebugState(tb, wb_run)                 # ← ここを wb_run に
       if (rank == 0 and args.debug) else _NoDbg())


    for ep in range(args.epochs):
        sampler.set_epoch(ep)
        model.train()
        loss_acc = 0.0
        n_batches = 0

        if rank == 0:
            print(f"\n[rank0] === Epoch {ep+1}/{args.epochs} ===", flush=True)        

        if rank == 0:
            t_fetch = time.perf_counter()

        for batch_idx, data in enumerate(dl):
            if batch_idx == 0 and rank == 0:
                print(f"[rank0]   first batch fetched "
                    f"({time.perf_counter()-t_fetch:.2f}s)", flush=True)

            # ---- forward ----
            if rank == 0:
                torch.cuda.synchronize(rank)
                t_fwd = time.perf_counter()

            data = data.to(rank, non_blocking=True)
            # loss, _logs = model(data)
            # ---- forward ----
            loss, logs = model(data)      # logs = {"rec":.., "kl":..}
            
            # β-KL annealing
            beta = beta0 + (beta_final-beta0) * min(1.0, dbg.step / (len(dl)*5))
            loss = logs["rec"] + beta * logs["kl"]
            
            # finite check
            if rank == 0:
                finite_or_raise(loss, "loss")
                finite_or_raise(logs["rec"], "reconstruction loss")
                finite_or_raise(logs["kl"], "KL divergence")

            # LR warm-up
            lr_scale = min(1.0, (dbg.step+1) / warmup_steps)
            for pg in opt.param_groups:
                pg["lr"] = args.lr * lr_scale
            
            if rank == 0:
                torch.cuda.synchronize(rank)
                print(f"[rank0]   forward  {time.perf_counter()-t_fwd:.2f}s",
                    flush=True)

            # ---- backward ----
            if rank == 0:
                torch.cuda.synchronize(rank)
                t_bwd = time.perf_counter()

            opt.zero_grad(set_to_none=True)
            loss.backward()
            # grad clip
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max)
            opt.step()
            
            # ---- debug stats (rank0 only) ------------------------------
            if rank == 0:
                # max|logit| は Decoder 出力の爆跳検知
                max_logit = logs.get("max_logit", torch.tensor(0.)).item()
                grad_norm = torch.sqrt(sum(
                    (p.grad.detach().norm() ** 2)
                    for p in model.parameters() if p.grad is not None)).item()
                param_norm = torch.sqrt(sum(
                    (p.detach().norm() ** 2)
                    for p in model.parameters())).item()

                dbg.log("train/rec",  logs["rec"].item())
                dbg.log("train/kl",   logs["kl"].item())
                dbg.log("train/loss", loss.item())
                dbg.log("debug/max_logit", max_logit)
                dbg.log("debug/grad_norm", grad_norm)
                dbg.log("debug/param_norm", param_norm)
                dbg.next()                

            if rank == 0:
                torch.cuda.synchronize(rank)
                print(f"[rank0]   backward {time.perf_counter()-t_bwd:.2f}s",
                    flush=True)

            loss_acc += loss.item()
            n_batches += 1

        if rank == 0:
            avg_loss = loss_acc / n_batches
            print(f"[rank0] epoch {ep+1} done  avg_loss={avg_loss:.4f}", flush=True)
            if tb:
                tb.add_scalar("loss/rec+kl", avg_loss, ep)
            if use_wandb:
                wandb.log({"loss": avg_loss, "epoch": ep})    


    # save ckpt once training is done
    if rank == 0:
        ckpt = Path(args.log_dir, "graphvae_ddp.pt")
        torch.save(model.module.state_dict(), ckpt)
        print("✔ saved", ckpt)
        if use_wandb:
            wandb.save(str(ckpt))
            wandb.finish()
        if tb:
            tb.close()

    dist.destroy_process_group()

# ---------------------------------------------------------------------------
# Public train() – convenient entry point for both CLI and YAML configs
# ---------------------------------------------------------------------------

def train(cfg):
    if isinstance(cfg, dict):  # allow YAML dict
        cfg = cfg.get("trainer", cfg)
        cfg = SimpleNamespace(**cfg)

    defaults = dict(
        dataset="grid",
        feature_type="id",
        lr=1e-3,
        epochs=200,
        max_nodes=None,  # autodetect
        log_dir="runs",
        use_wandb=True,
        run_name=None,
    )
    for k, v in defaults.items():
        if not hasattr(cfg, k):
            setattr(cfg, k, v)

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        _ddp_worker(int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"]), cfg)
    else:
        world = torch.cuda.device_count()
        torch.multiprocessing.spawn(_ddp_worker, nprocs=world, args=(world, cfg))

# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    pa = argparse.ArgumentParser()
    pa.add_argument("--dataset", choices=["grid", "enzymes"], default="grid")
    pa.add_argument("--feature_type", choices=["id", "deg"], default="id")
    pa.add_argument("--lr", type=float, default=1e-3)
    pa.add_argument("--epochs", type=int, default=200)
    pa.add_argument("--max_nodes", type=int, default=None)
    pa.add_argument("--log_dir", default="runs")
    pa.add_argument("--config", help="YAML config with a 'trainer:' section")
    pa.add_argument("--no_wandb", action="store_true", help="disable Weights & Biases logging")
    pa.add_argument("--run_name", help="custom W&B run name")
    pa.add_argument("--debug", action="store_true", help="enable extra fin-check / TB stats")
    args = pa.parse_args()
    args.use_wandb = not args.no_wandb

    if args.config:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        train(cfg)
    else:
        train(args)

