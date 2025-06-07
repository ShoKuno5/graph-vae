# utils_perm.py
import torch
from scipy.optimize import linear_sum_assignment

def permute_adj(adj_true: torch.Tensor, adj_pred: torch.Tensor):
    """
    Hungarian assignment で真値隣接行列を並び替える
    adj_* : (N,N) 0/1 or 0-1 値 Tensor
    Returns A_perm (Tensor)
    """
    eps = 1e-9
    cost = -(adj_true * torch.log(adj_pred + eps) +
             (1 - adj_true) * torch.log(1 - adj_pred + eps))

    row, col = linear_sum_assignment(cost.detach().cpu().numpy())
    P = torch.zeros_like(adj_true)
    P[row, col] = 1.0
    return P.T @ adj_true @ P
