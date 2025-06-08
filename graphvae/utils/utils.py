import torch

def compute_dataset_rho(dataset):
    num_pos, num_tot = 0, 0
    for g in dataset:
        R = g.num_real_nodes
        idx = torch.triu_indices(R, R, offset=1)
        num_pos += g.adj_dense[idx[0], idx[1]].sum().item()  # ★ 2-次元
        num_tot += R * (R - 1) / 2
    return num_pos / num_tot            # >> ρ
