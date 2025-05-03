#!/usr/bin/env python
# train_graphvae_stable.py  – 1000 graph / 80 epoch / 固定SEED
import os, random, torch, numpy as np, torch.nn.functional as F
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from gvae.models.graphvae_core import GEncoder, EdgeMLP
from scipy.optimize import linear_sum_assignment

# reproducible
SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
torch.use_deterministic_algorithms(True)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

def permute_adj(A_true, A_pred):
    eps=1e-9
    cost=-(A_true*torch.log(A_pred+eps)+(1-A_true)*torch.log(1-A_pred+eps))
    r,c=linear_sum_assignment(cost.detach().cpu().numpy())
    P=torch.zeros_like(A_true); P[r,c]=1
    return P.T@A_true@P

class GraphVAE(torch.nn.Module):
    def __init__(self, in_dim, hid=64, z=32):
        super().__init__()
        self.enc=GEncoder(in_dim,hid,z); self.dec=EdgeMLP(z)
    def forward(self,d):
        mu,logvar=self.enc(d.x,d.edge_index)
        z=mu+torch.randn_like(mu)*torch.exp(0.5*logvar)
        return self.dec(z),mu,logvar

# ---------------- hyper -----------------
EPOCHS=80
NEG_W0, NEG_W1=40., 1.   # ★最終罰則を弱めた
L1_0,  L1_1 =1e-3,0.0    # ★最後は L1=0
KL_WARM=60
MINDEG_W=0.25            # ★孤立ノード罰則を強め
# ----------------------------------------

ds = QM9(root="/dataset/QM9")[:1000]   # root は processed の 1 つ上の QM9
loader=DataLoader(ds,batch_size=1,shuffle=True)
model=GraphVAE(ds.num_features).cpu()
opt=torch.optim.Adam(model.parameters(),lr=1e-3)

for ep in range(EPOCHS):
    neg_w=NEG_W0-(NEG_W0-NEG_W1)*ep/EPOCHS
    l1   =L1_0 -(L1_0 -L1_1 )*ep/EPOCHS
    kl_w=min(1.0,ep/KL_WARM)
    L=R=K=0
    for d in loader:
        A_hat,mu,logvar=model(d)
        A_true=torch.zeros_like(A_hat)
        A_true[d.edge_index[0],d.edge_index[1]]=1
        A_perm=permute_adj(A_true,A_hat)
        bce=F.binary_cross_entropy(A_hat,A_perm,reduction='none')
        recon=((1-A_perm)*neg_w*bce + A_perm*bce).mean()
        kl=-0.5*torch.mean(1+logvar-mu.pow(2)-logvar.exp())
        deg=A_hat.sum(-1)
        loss=recon+kl_w*kl+l1*A_hat.sum()+MINDEG_W*torch.relu(1-deg).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        L+=loss.item(); R+=recon.item(); K+=kl.item()
    print(f"Ep{ep:02d} | Loss {L/len(loader):.3f} Re {R/len(loader):.3f} "
          f"KL {K/len(loader):.3f} (neg_w={neg_w:.1f}, l1={l1:.0e})")

os.makedirs("runs",exist_ok=True)
torch.save(model.state_dict(),"runs/graphvae_stable.pt")
print("✔ saved runs/graphvae_stable.pt")
