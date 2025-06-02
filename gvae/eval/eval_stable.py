#!/usr/bin/env python
"""eval_stable.py — GraphVAE 評価スクリプト（改訂版）

* 固定 SEED
* 10 000 サンプル × 5 回平均
* Degree‑MMD = 1‑D ワッサーシュタイン距離
* Invalid グラフの原因内訳を表示（disconnected / selfloop / deg>k / other）
"""

import inspect, gvae.models.graphvae_core as core
print("★ loaded:", inspect.getfile(core))

import os, random, argparse
from collections import Counter
from statistics import mean, pstdev

import torch, numpy as np
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader

from . import metrics as M
from gvae.models.graphvae_core import GEncoder, EdgeMLP


###############################################################################
# 再現性設定
###############################################################################
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

###############################################################################
# モデル定義（最低限のラッパー）
###############################################################################
class GVAE(torch.nn.Module):
    def __init__(self, in_dim: int, hid: int = 64, z: int = 32):
        super().__init__()
        self.enc = GEncoder(in_dim, hid, z)
        self.dec = EdgeMLP(z)

    def encode(self, x, edge_index):
        # エンコーダから μ のみを取得（対数分散は無視）
        mu, _ = self.enc(x, edge_index)
        return mu

###############################################################################
# サンプリング関数
###############################################################################
@torch.no_grad()
def sample(model: GVAE,
           loader: DataLoader,
           k: int,
           alpha: float = 2.5,
           p_min: float = 0.06):
    """k 個のグラフを生成し (ref, gen) を返す。

    * **alpha** : 生成エッジ確率のスケール。大きくすると密になる。
    * **p_min** : 0 に近い確率を底上げして disconnected を防ぐ。
    """
    ref, gen = [], []
    for data in loader:
        if len(gen) >= k:
            break
        z = model.encode(data.x, data.edge_index)
        p = model.dec(z).clamp_min(p_min)          # ← 底上げ
        mask = torch.rand_like(p) < p * alpha
        mask = torch.triu(mask, diagonal=1)        # 上三角のみ
        edges = torch.cat(
            [mask.nonzero(), mask.nonzero()[:, [1, 0]]]
        ).t()                                      # 対称化
        gen.append(M.to_nx(edges, data.num_nodes))
        ref.append(M.to_nx(data.edge_index, data.num_nodes))
    return ref, gen

###############################################################################
# メイン処理
###############################################################################

def evaluate(ckpt: str,
             ds_size: int = 3000,
             n_samples: int = 10_000,
             repeats: int = 5,
             k_max: int = 6):
    """モデルを評価してスコアと invalid breakdown を返す"""

    # ------------- データセット & モデル ------------------
    qm9_root = os.getenv("QM9_ROOT", "data/QM9")
    ds = QM9(root=qm9_root)[:ds_size]
    loader = DataLoader(ds, batch_size=1, shuffle=False)

    model = GVAE(ds.num_features)
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.eval()
    # ------------------------------------------------------

    valid, uniq_iso, mmd = [], [], []
    invalid_tags_all = Counter()

    for _ in range(repeats):
        ref, gen = sample(model, loader, n_samples)
        valid.append(M.validity(gen))
        uniq_iso.append(M.uniqueness_iso(gen))
        mmd.append(M.degree_mmd(ref, gen))
        invalid_tags_all.update(M.why_invalid(g, k_max=k_max) for g in gen if not M.is_valid(g, k_max))

    # --- 統計値 ---
    result_lines = [
        f"Validity        : {mean(valid):.3f} ± {pstdev(valid):.3f}",
        f"Uniqueness (iso): {mean(uniq_iso):.3f} ± {pstdev(uniq_iso):.3f}",
        f"Degree-MMD      : {mean(mmd):.3f} ± {pstdev(mmd):.3f}",
    ]

    # --- invalid breakdown ---
    total_invalid = sum(invalid_tags_all.values()) or 1  # div0 guard
    result_lines.append("Invalid breakdown (aggregated over all repeats):")
    for tag, cnt in invalid_tags_all.items():
        result_lines.append(f"  {tag:<12}: {cnt:>6}  ({cnt/total_invalid:.2%})")

    return "\n".join(result_lines) + "\n"

###############################################################################
# CLI エントリポイント
###############################################################################

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/workspace/runs/graphvae_ddp_amp.pt",
                    help="model checkpoint path")
    ap.add_argument("--out", help="file to save metrics", default=None)
    args = ap.parse_args()

    txt = evaluate(args.ckpt)
    print(txt, end="")
    if args.out:
        with open(args.out, "w") as f:
            f.write(txt)

if __name__ == "__main__":
    main()
