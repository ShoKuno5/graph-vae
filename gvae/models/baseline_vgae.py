# baseline_vgae.py
import argparse, torch, torch.nn.functional as F
from torch_geometric.nn import GCNConv, VGAE
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader

class Encoder(torch.nn.Module):
    def __init__(self, in_dim, hid_dim, z_dim):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hid_dim)
        self.conv_mu = GCNConv(hid_dim, z_dim)
        self.conv_logvar = GCNConv(hid_dim, z_dim)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        return self.conv_mu(x, edge_index), self.conv_logvar(x, edge_index)

def train(model, loader, device):
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(args.epochs):
        loss_all = 0
        for data in loader:
            data = data.to(device)
            opt.zero_grad()

            z = model.encode(data.x, data.edge_index)           # ← 修正
            loss  = model.recon_loss(z, data.edge_index)
            loss += model.kl_loss() / data.num_nodes            # ← 修正

            loss.backward()
            opt.step()
            loss_all += loss.item()

        print(f"Epoch {epoch:02d}  Loss {loss_all/len(loader):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="QM9")
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = QM9(root="data/QM9").shuffle()[:1000]       # 小スケールで素早く
    loader = DataLoader(ds, batch_size=32, shuffle=True)

    model = VGAE(Encoder(ds.num_features, 64, 32)).to(device)
    train(model, loader, device)
    
    import os
    os.makedirs("runs", exist_ok=True)
    torch.save(model.state_dict(), "runs/vgae_epoch10.pt")
    print("✔ Saved to runs/vgae_epoch10.pt")
