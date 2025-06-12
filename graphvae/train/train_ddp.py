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

from datetime import datetime
from pathlib import Path
import yaml

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
        # data.num_nodes = x.size(0)  # pad virtual nodes implicitly
        # 「本当に存在するノード数」だけ知らせる
        data.num_nodes = g.number_of_nodes()

        # dense adjacency for loss
        adj = torch.zeros(self.max_nodes, self.max_nodes)
        ei = data.edge_index
        adj[ei[0], ei[1]] = 1
        #adj = torch.maximum(adj, adj.T)  # undirected
        adj = torch.maximum(adj, adj.T)
        # ---- ここを追加: 実ノードに自己ループを入れる ----
        N = g.number_of_nodes()
        idx = torch.arange(N, device=adj.device)
        adj[idx, idx] = 1.0
        # --------------------------------------------
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

# ---------------------------------------------------------------------------
# DDP worker with train / val split
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# DDP worker – per-graph ρ 計算バージョン
# ---------------------------------------------------------------------------
def _ddp_worker(rank: int, world: int, args):
    print("DEBUG rank", rank, type(args.lr), args.lr, flush=True)
    _set_seed()
    dist.init_process_group("nccl", rank=rank, world_size=world)
    torch.cuda.set_device(rank)

    # 1. グラフ読み込み → rank-0 で 90/10 split を決めて broadcast
    graphs, max_nodes = build_graph_list(args.dataset, args.max_nodes)
    if rank == 0:
        random.shuffle(graphs)
    obj = [graphs]
    dist.broadcast_object_list(obj, src=0)
    graphs = obj[0]

    split = int(0.9 * len(graphs))
    g_train, g_val = graphs[:split], graphs[split:]

    ds_tr = GraphDataset(g_train, max_nodes=max_nodes, feature_type=args.feature_type)
    ds_va = GraphDataset(g_val,   max_nodes=max_nodes, feature_type=args.feature_type)

    smp_tr = DistributedSampler(ds_tr, world, rank, shuffle=True)
    smp_va = DistributedSampler(ds_va, world, rank, shuffle=False)

    dl_tr  = DataLoader(ds_tr, batch_size=args.batch_size, sampler=smp_tr, pin_memory=True)
    dl_va  = DataLoader(ds_va, batch_size=args.batch_size, sampler=smp_va, pin_memory=True)

    # 2. モデル（ρ固定引数なし）
    model = GraphVAE(
        in_dim    = args.in_dim,  # ← YAML で指定した in_dim
        hid_dim   = args.hid_dim,
        z_dim     = args.z_dim,
        max_nodes = max_nodes,
        pool      = args.pool,          # ρ は forward 内で都度算出
    ).to(rank)
    # ---------- ★ ここを追加 ----------------------------------------------------
    # DataLoader *全体* を使って rank0 だけ ρ を測る
    if rank == 0:
        full_loader = DataLoader(
            GraphDataset(g_train, max_nodes=max_nodes,
                        feature_type=args.feature_type),
            batch_size=args.batch_size, shuffle=False
        )
        model.precompute_dataset_rho(full_loader)   # ← 必須呼び出し
    # ほかの rank が来るのを待つ
    if dist.is_initialized():
        dist.barrier()
    # ---------------------------------------------------------------------------

    # その後で DDP ラップすれば OK
    model = DDP(model, device_ids=[rank], find_unused_parameters=False)
    opt   = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # 3. ロガー
    use_wandb = rank == 0 and args.use_wandb and os.getenv("WANDB_DISABLED","0") not in ("true","1")
    if use_wandb:
        import wandb
        wb = wandb.init(project="graphvae",
                        name=args.run_name or f"{args.dataset}-ddp",
                        dir=args.log_dir, config=vars(args))
    tb = SummaryWriter(Path(args.log_dir, "tb")) if rank == 0 else None

    # 4. β–KL & LR warm-up 設定
    """beta0, beta_final = 0.0, 0.5
    warm_epochs       = 40
    clip_max          = 5.0
    """
    
    steps_per_epoch   = len(dl_tr)
    anneal_steps      = args.warm_epochs * steps_per_epoch
    global_step       = 0

    # ループ外で一度だけ
    if rank == 0:
        dbg = DebugState(tb, wb) if args.debug else _NoDbg()
    else:
        dbg = _NoDbg()
        
    # --------------------------- Training loop ----------------------------
    for ep in range(args.epochs):

        # ---- TRAIN ------------------------------------------------------
        smp_tr.set_epoch(ep)
        model.train()
        sum_loss_tr = 0.0
        for data in dl_tr:
            data = data.to(rank, non_blocking=True)
            logs = model(data)                     # rec, kl は内部で ρ 計算済み

            # ─── 以前と同じ β-VAE 損失計算 ─────
            beta = args.beta0 + (args.beta_final -args.beta0) * min(1.0, global_step / anneal_steps)
            loss = logs["rec"] + beta * logs["kl"]
            
            # LR warm-up
            lr_scale = min(1.0, global_step / anneal_steps)
            for pg in opt.param_groups:
                pg["lr"] = args.lr * lr_scale

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_max)
            
            # 10 step ごとにだけノルムを測定
            if args.debug and rank == 0 and dbg.step % 10 == 0:
                with torch.no_grad():
                    param_norm = torch.nn.utils.parameters_to_vector(
                                    model.parameters()
                                ).norm().item()
                    grad_norm  = torch.nn.utils.parameters_to_vector(
                                    [p.grad for p in model.parameters() if p.grad is not None]
                                ).norm().item()

                dbg.log("debug/param_norm", param_norm)
                dbg.log("debug/grad_norm",  grad_norm)

            # 残りの指標は毎ステップでも軽いのでそのまま
            dbg.log("train/rec",  logs["rec"].item())
            dbg.log("train/kl",   logs["kl"].item())
            dbg.log("train/loss", loss.item())
            dbg.log("debug/max_logit",  logs.get("max_logit", 0.).item())
            dbg.log("debug/logvar_max", logs.get("logvar_max", 0.).item())

            dbg.next()          # ← 最後に step++ する
            
            opt.step()

            sum_loss_tr += loss.item()
            global_step += 1

        # all-reduce train loss
        tot_tr = torch.tensor([sum_loss_tr], device=rank)
        dist.all_reduce(tot_tr, op=dist.ReduceOp.SUM)
        mean_tr = tot_tr.item() / len(dl_tr) / world

        # ---- VALIDATION -------------------------------------------------
        model.eval(); sum_loss_va = 0.0
        with torch.no_grad():
            for data in dl_va:
                data = data.to(rank, non_blocking=True)
                logs = model(data)
                beta = args.beta_final
                sum_loss_va += (logs["rec"] + beta * logs["kl"]).item()

        tot_va = torch.tensor([sum_loss_va], device=rank)
        dist.all_reduce(tot_va, op=dist.ReduceOp.SUM)
        mean_va = tot_va.item() / len(dl_va) / world

        # ---- LOG (rank-0) ----------------------------------------------
        if rank == 0:
            print(f"Ep {ep:03d} | train {mean_tr:.4f} | val {mean_va:.4f}")
            if tb:
                tb.add_scalars("loss", {"train": mean_tr, "val": mean_va}, ep)
            if use_wandb:
                wandb.log({"loss/train": mean_tr, "loss/val": mean_va, "epoch": ep})

    # save
    if rank == 0:
        ckpt = Path(args.log_dir, "graphvae_ddp.pt")
        torch.save(model.module.state_dict(), ckpt)
        print("✔ saved", ckpt)
        if use_wandb:
            wandb.save(str(ckpt)); wandb.finish()
        if tb:
            tb.close()

    dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Public train() – convenient entry point for both CLI and YAML configs
# ---------------------------------------------------------------------------

import os, yaml
from pathlib import Path
from types import SimpleNamespace

def train(cfg):
    # -------- 1. dict → SimpleNamespace --------
    if isinstance(cfg, dict):
        cfg = SimpleNamespace(**cfg.get("trainer", cfg))

    # -------- 2. defaults を穴埋め --------
    defaults = dict(
        dataset="grid", feature_type="deg",
        lr=1e-4, epochs=200, max_nodes=None,
        log_dir="runs", use_wandb=True, run_name=None,
        batch_size=64, weight_decay=1e-4,
        beta0=0.0, beta_final=0.5, warm_epochs=40, clip_max=5.0,
        hid_dim=128, z_dim=64, pool="sum",
    )
    for k, v in defaults.items():
        if not hasattr(cfg, k):
            setattr(cfg, k, v)

    # -------- 3. 数値フィールドを float 化 --------
    for key in ("lr", "weight_decay", "beta0", "beta_final", "clip_max"):
        val = getattr(cfg, key)
        if isinstance(val, str):
            try:
                setattr(cfg, key, float(val))
            except ValueError:
                raise ValueError(f"{key} should be numeric, got {val!r}")

    if cfg.log_dir in ("runs", "./runs"):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        cfg.log_dir = str(Path("runs") / ts)
    Path(cfg.log_dir).mkdir(parents=True, exist_ok=True)
    
    # --- 3. データセットを 1 グラフだけ読み込んで in_dim を決める ---
    graphs, max_nodes_auto = build_graph_list(cfg.dataset, cfg.max_nodes)

    ds_tmp = GraphDataset([graphs[0]],
                        feature_type=cfg.feature_type,
                        max_nodes=cfg.max_nodes or max_nodes_auto)   # ★ 追加

    cfg.in_dim = ds_tmp[0].x.size(-1)

    # -------- 4. params.yaml を rank0 だけ保存 --------
    is_rank0 = os.environ.get("RANK", "0") == "0"
    if is_rank0:
        with open(Path(cfg.log_dir) / "params.yaml", "w") as f:
            yaml.safe_dump(vars(cfg), f, sort_keys=False)
            print("✔ saved params to", f.name)

    # -------- 5. DDP 起動 --------
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
    pa.add_argument("--dataset", choices=["grid", "enzymes"], default=None)
    pa.add_argument("--feature_type", choices=["id", "deg"], default=None)
    #pa.add_argument("--lr", type=float, default=1e-4)
    #pa.add_argument("--epochs", type=int, default=200)
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
            ycfg = yaml.safe_load(f).get("trainer", {})
        # CLI → dict（None は除外）
        cli = {k: v for k, v in vars(args).items()
            if k not in ("config", "no_wandb") and v is not None}
        # CLI が YAML を上書き
        merged = {**ycfg, **cli}
        train(SimpleNamespace(**merged))
    else:
        train(args)