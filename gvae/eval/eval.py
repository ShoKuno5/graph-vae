# eval.py
# -------------------------------------------------
import argparse, torch, networkx as nx, matplotlib.pyplot as plt
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
import metrics as M
from gvae.models.baseline_vgae import Encoder                       # VGAE 用

# ==== GraphVAE 用エンコーダ & MLP デコーダ =========
from gvae.models.graphvae_core import GEncoder, EdgeMLP
# ---------------------------------------------------

def build_model(name: str, in_dim: int, device):
    if name == "vgae":
        from torch_geometric.nn import VGAE
        return VGAE(Encoder(in_dim, 64, 32)).to(device)

    elif name == "graphvae":
        class GraphVAEWrap(torch.nn.Module):
            """Encoder + EdgeMLP decoder (学習時と同構造)"""
            def __init__(self, in_dim, hid=64, z_dim=32):
                super().__init__()
                self.enc = GEncoder(in_dim, hid, z_dim)
                self.dec = EdgeMLP(z_dim)
            def encode(self, x, edge_index):
                mu, _ = self.enc(x, edge_index)
                return mu
        return GraphVAEWrap(in_dim).to(device)

    else:
        raise ValueError(f"unknown model: {name}")

# ---------- グラフ生成 ----------------------------------------
@torch.no_grad()
def sample_graphs(model, loader, k: int, device):
    ref, gen = [], []
    for data in loader:
        if len(gen) >= k:
            break
        data = data.to(device)

        z = model.encode(data.x, data.edge_index)

        # --- デコーダをモデルに合わせて選択 ---
        if hasattr(model, "dec"):               # GraphVAE
            adj_p = model.dec(z)
        else:                                   # VGAE
            adj_p = torch.sigmoid(z @ z.t())

        # Bernoulli 抽出でエッジサンプリング
        edge_idx = (torch.rand_like(adj_p) < adj_p).nonzero(as_tuple=False).t()

        gen.append(M.to_nx(edge_idx, data.num_nodes))
        ref.append(M.to_nx(data.edge_index, data.num_nodes))
    return ref, gen
# --------------------------------------------------------------

def main(args):
    device  = torch.device("cpu")
    dataset = QM9(root="data/QM9")[: args.sample * 2]
    loader  = DataLoader(dataset, batch_size=1, shuffle=False)

    model = build_model(args.model, dataset.num_features, device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))  # strict=True OK
    model.eval()

    refs, gens = sample_graphs(model, loader, args.sample, device)

    print(f"Validity   : {M.validity(gens):.3f}")
    print(f"Uniqueness : {M.uniqueness(gens):.3f}")
    print(f"Degree-MMD : {M.degree_mmd(refs, gens):.3f}")

    if args.vis:
        nx.draw(gens[0], with_labels=True)
        plt.show()

# ------------------- CLI -------------------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["vgae", "graphvae"], default="vgae")
    p.add_argument("--ckpt", required=True, help="checkpoint .pt path")
    p.add_argument("--sample", type=int, default=300, help="#graphs to eval")
    p.add_argument("--vis", action="store_true", help="show one generated graph")
    main(p.parse_args())
