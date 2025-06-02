#!/usr/bin/env python
"""eval_stable_sweep.py — GraphVAE 評価スクリプト（α / p_min 並列スイープ & 度数キャップ付き）

固定パラメータは下記のリストで定義（CLI で上書き可）：

```python
ALPHAS = [3.0, 3.2, 3.4]
P_MINS = [0.06, 0.08]
```

特徴
------
* **ProcessPoolExecutor** で α×p_min の組を並列評価
* 生成時に **各頂点の次数が `k_max` (=6) を超えないようキャップ**
* 10 000 サンプル × 5 回平均で Validity / Uniqueness / Degree‑MMD を計算
* `--out` を指定すると全結果を 1 ファイルへ追記保存
"""

from __future__ import annotations
import argparse, inspect, itertools, os, random, textwrap
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from statistics import mean, pstdev

import torch, numpy as np
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader

from . import metrics as M
import gvae.models.graphvae_core as core
print("★ loaded:", inspect.getfile(core))

###############################################################################
# スイープ対象（デフォルト）
###############################################################################
ALPHAS = [2.8, 3.0, 3.2, 3.4, 3.6, 3.8, 4.0]  # エッジ確率スケール
P_MINS  = [0.05, 0.06, 0.07, 0.08, 0.09, 0.10]  # 最小確率の底上げ
K_MAX   = 6  # 度数キャップ

###############################################################################
# 再現性設定
###############################################################################
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

###############################################################################
# モデル定義
###############################################################################
class GVAE(torch.nn.Module):
    def __init__(self, in_dim: int, hid: int = 64, z: int = 32):
        super().__init__()
        # local import to keep picklable
        from gvae.models.graphvae_core import GEncoder, EdgeMLP
        self.enc = GEncoder(in_dim, hid, z)
        self.dec = EdgeMLP(z)

    def encode(self, x, edge_index):
        mu, _ = self.enc(x, edge_index)
        return mu

###############################################################################
# サンプリング（度数キャップ付）
###############################################################################
@torch.no_grad()
def sample(model: GVAE, loader: DataLoader, k: int, *, alpha: float, p_min: float, k_max: int = K_MAX):
    """k 個のグラフを生成して (ref, gen) を返す。

    * alpha : エッジ確率スケール
    * p_min : 最小確率の底上げ
    * k_max : 1 頂点あたりの最大次数
    """
    ref, gen = [], []
    for data in loader:
        if len(gen) >= k:
            break

        # --- 潜在変数 & エッジ確率 ---
        z = model.encode(data.x, data.edge_index)
        p = model.dec(z).clamp_min(p_min)

        # --- ベルヌーイサンプリング & 対称化 ---
        m = (torch.rand_like(p) < p * alpha)
        m = torch.triu(m, 1)
        m = m | m.T  # 対称マスク

        # --- 度数キャップ（各頂点 <= k_max） ---
        #   単純ランダムに余剰エッジを間引く。反復は稀なので O(n^2) でも許容。
        deg = m.sum(-1)
        while (deg > k_max).any():
            v = (deg > k_max).nonzero(as_tuple=False)[0, 0]
            idx = m[v].nonzero(as_tuple=False).flatten()
            # 余剰エッジ数
            excess = int(deg[v] - k_max)
            drop = idx[torch.randperm(idx.numel())[:excess]]
            m[v, drop] = False
            m[drop, v] = False
            deg[v] -= excess
            deg[drop] -= 1  # 各隣接頂点の次数も 1 減

        # --- 上三角だけ取り出して edge_index 生成 ---
        m_ut = torch.triu(m, 1)
        edges = torch.cat([m_ut.nonzero(), m_ut.nonzero()[:, [1, 0]]]).t()

        gen.append(M.to_nx(edges, data.num_nodes))
        ref.append(M.to_nx(data.edge_index, data.num_nodes))
    return ref, gen

###############################################################################
# 評価ルーチン
###############################################################################

def evaluate(ckpt: str, *, alpha: float, p_min: float, ds_size: int = 3000,
             n_samples: int = 10_000, repeats: int = 5, k_max: int = K_MAX) -> str:
    qm9_root = os.getenv("QM9_ROOT", "data/QM9")
    ds = QM9(root=qm9_root)[:ds_size]
    loader = DataLoader(ds, batch_size=1, shuffle=False)

    model = GVAE(ds.num_features)
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.eval()

    valid, uniq_iso, mmd = [], [], []
    invalid_tags_all = Counter()

    for _ in range(repeats):
        ref, gen = sample(model, loader, n_samples, alpha=alpha, p_min=p_min, k_max=k_max)
        valid.append(M.validity(gen))
        uniq_iso.append(M.uniqueness_iso(gen))
        mmd.append(M.degree_mmd(ref, gen))
        invalid_tags_all.update(
            M.why_invalid(g, k_max=k_max) for g in gen if not M.is_valid(g, k_max)
        )

    lines = [
        f"Validity        : {mean(valid):.3f} ± {pstdev(valid):.3f}",
        f"Uniqueness (iso): {mean(uniq_iso):.3f} ± {pstdev(uniq_iso):.3f}",
        f"Degree-MMD      : {mean(mmd):.3f} ± {pstdev(mmd):.3f}",
        "Invalid breakdown (aggregated over all repeats):",
    ]
    total_inv = sum(invalid_tags_all.values()) or 1
    for tag, cnt in invalid_tags_all.items():
        lines.append(f"  {tag:<12}: {cnt:>6}  ({cnt/total_inv:.2%})")
    return "\n".join(lines) + "\n"

###############################################################################
# 並列実行関数
###############################################################################

def _eval_job(args):
    ckpt, alpha, p_min = args
    header = f"=== α={alpha}, p_min={p_min} ===\n"
    body = evaluate(ckpt, alpha=alpha, p_min=p_min)
    return header + body

###############################################################################
# CLI
###############################################################################

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="path to model checkpoint")
    ap.add_argument("--alpha", help="override ALPHAS list, e.g. 3.0,3.4")
    ap.add_argument("--p_min", help="override P_MINS list, e.g. 0.06,0.08")
    ap.add_argument("--max_workers", type=int, default=os.cpu_count(), help="parallel workers")
    ap.add_argument("--out", help="file to append results")
    args = ap.parse_args()

    alphas = [float(a) for a in (args.alpha.split(",") if args.alpha else ALPHAS)]
    p_mins = [float(p) for p in (args.p_min.split(",") if args.p_min else P_MINS)]

    grid = [(args.ckpt, a, p) for a, p in itertools.product(alphas, p_mins)]

    out_f = open(args.out, "a") if args.out else None

    with ProcessPoolExecutor(max_workers=args.max_workers) as ex:
        futs = {ex.submit(_eval_job, job): job for job in grid}
        for fut in as_completed(futs):
            txt = fut.result()
            print(txt, end="\n")
            if out_f:
                out_f.write(txt + "\n")

    if out_f:
        out_f.close()

if __name__ == "__main__":
    main()