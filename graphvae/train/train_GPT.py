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

from model import GraphVAE  # ← your modernised implementation
from utils import compute_dataset_rho

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
SEED = 42

def _set_seed():
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.use_deterministic_algorithms(True)
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
        data.num_real_nodes = g.number_of_nodes()   # ★ ここを追加
        return data

# ---------------------------------------------------------------------------
# Graph generators matching the original baseline choices
# ---------------------------------------------------------------------------


def build_graph_list(dataset: str, max_nodes: int | None):
    if dataset == "grid":
        graphs = [nx.grid_2d_graph(i, j) for i, j in itertools.product(range(2, 4), repeat=2)]
    elif dataset == "enzymes":
        td = TUDataset(root="/tmp", name="ENZYMES")
        graphs = [g.to_networkx() for g in td]
    else:
        raise ValueError("dataset must be 'grid' or 'enzymes'")

    if max_nodes is None:
        max_nodes = max(g.number_of_nodes() for g in graphs)
    # remove too‑large graphs like original script
    graphs = [g for g in graphs if g.number_of_nodes() <= max_nodes]
    return graphs, max_nodes

# ---------------------------------------------------------------------------
# DDP worker
# ---------------------------------------------------------------------------

def _ddp_worker(rank: int, world: int, args: argparse.Namespace):
    _set_seed()
    dist.init_process_group("nccl", rank=rank, world_size=world)
    torch.cuda.set_device(rank)

    graphs, max_nodes = build_graph_list(args.dataset, args.max_nodes)
    # ds = GraphDataset(graphs, max_nodes, args.feature_type)
    # sampler = DistributedSampler(ds, num_replicas=world, rank=rank, shuffle=True)
    # dl = DataLoader(ds, batch_size=1, sampler=sampler, pin_memory=True)
    
      
    # 例: 90 % / 10 % ランダム split
    if rank == 0:   
        random.shuffle(graphs)
    obj = [graphs]                      # ★ リストで包む
    dist.broadcast_object_list(obj, src=0)
    graphs = obj[0]                     # ★ 取り出す
    split = int(0.9 * len(graphs))
    g_train, g_val = graphs[:split], graphs[split:]

    ds_train = GraphDataset(g_train, max_nodes, args.feature_type)
    ds_val   = GraphDataset(g_val,   max_nodes, args.feature_type)
    
    sampler_tr = DistributedSampler(ds_train, world, rank, shuffle=True)
    sampler_va = DistributedSampler(ds_val,   world, rank, shuffle=False)

    dl_tr = DataLoader(ds_train, batch_size=1, sampler=sampler_tr, pin_memory=True)
    dl_va = DataLoader(ds_val,   batch_size=1, sampler=sampler_va, pin_memory=True)
    
    if rank == 0:
        rho = compute_dataset_rho(ds_train)
        rho_tensor = torch.tensor([rho], dtype=torch.float32, device="cuda")
    else:
        rho_tensor = torch.zeros(1, dtype=torch.float32, device="cuda")

    dist.broadcast(rho_tensor, src=0)
    rho = float(rho_tensor.item())        # Python float

    if rank == 0:
        print(f"[INFO] dataset rho = {rho:.4f} (pos_weight={(1-rho)/rho:.2f})")
        
    # ------------------------------------------------------------------
    # 3. GraphVAE を作成 ― ここで ρ を渡す
    # ------------------------------------------------------------------        

    model = GraphVAE(
        in_dim = ds_train[0].x.size(-1),
        hid_dim = 64,
        z_dim = 32,
        max_nodes = max_nodes,
        pool = "sum",
        rho_dataset = rho,                # ★ 固定 pos_weight
    ).to(rank)
    model = DDP(model, device_ids=[rank], find_unused_parameters=False)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    tb = SummaryWriter(Path(args.log_dir, "tb")) if rank == 0 else None
    use_wandb = rank == 0 and args.use_wandb and os.getenv("WANDB_DISABLED", "false").lower() not in ("true", "1")
    if use_wandb:
        import wandb
        wandb.init(project="graphvae", name=args.run_name or f"{args.dataset}-ddp",
                   dir=args.log_dir, config=vars(args))

    for ep in range(args.epochs):
        # sampler.set_epoch(ep)
        sampler_tr.set_epoch(ep)
        model.train()
        # loss_acc = 0.0
        loss_tr = 0.0     
        
        for data in dl_tr:
            data = data.to(rank)
            loss, logs = model(data)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            loss_tr += loss.item()
            
        # ---- validation ----
        model.eval(); loss_va = 0.0
        with torch.no_grad():
            for data in dl_va:
                data = data.to(rank)
                loss, logs = model(data)
                loss_va += loss.item()  

        # ---- distributed reduce + logging ----
        #tot = torch.tensor([loss_acc], device=rank)
        #dist.all_reduce(tot, op=dist.ReduceOp.SUM)
        
        tot_tr = torch.tensor([loss_tr], device=rank)
        
        tot_va = torch.tensor([loss_va], device=rank)
        dist.all_reduce(tot_va, op=dist.ReduceOp.SUM)      
        dist.all_reduce(tot_tr, op=dist.ReduceOp.SUM)
        #if rank == 0:
        #    mean_loss = tot.item() / len(dl) / world
        #    print(f"Epoch {ep:03d}: loss {mean_loss:.4f}")
        #    if tb:
        #        tb.add_scalar("loss/train", mean_loss, ep)
        #    if use_wandb:
        #        wandb.log({"loss/train": mean_loss, "epoch": ep})
        if rank == 0:
            mean_tr = tot_tr.item()   / len(dl_tr) / world
            mean_va = tot_va.item()/ len(dl_va) / world
            print(f"Ep {ep:03d} | train {mean_tr:.4f} | val {mean_va:.4f}")

            if tb:
                tb.add_scalar("loss/train", mean_tr, ep)
                tb.add_scalar("loss/val",   mean_va, ep)
            if use_wandb:
                wandb.log({"loss/train":mean_tr, "loss/val":mean_va, "epoch":ep})

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
    args = pa.parse_args()
    args.use_wandb = not args.no_wandb

    if args.config:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        train(cfg)
    else:
        train(args)
