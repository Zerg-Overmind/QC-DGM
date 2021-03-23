import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.hungarian import hungarian

class FMLoss(nn.Module):
    """
    Cross entropy loss between two permutations.
    """
    def __init__(self):
        super(FMLoss, self).__init__()

    def forward(self, pred_perm, gt_perm, pred_ns, gt_ns, G1_gt, G2_gt, H1_gt, H2_gt):
        batch_num = pred_perm.shape[0]

        pred_perm = pred_perm.to(dtype=torch.float32)
        
        assert torch.all((pred_perm >= 0) * (pred_perm <= 1))
        assert torch.all((gt_perm >= 0) * (gt_perm <= 1))
    
        loss_p = torch.tensor(0.).to(pred_perm.device)         
        loss_n = torch.tensor(0.).to(pred_perm.device)
        n_sum = torch.tensor(0.).to(pred_perm.device)

 
        alpha = 2
        beta = 0.1
        for b in range(batch_num):
            ## false positive
            loss_p += torch.sum(torch.mul(pred_perm[b, :pred_ns[b], :gt_ns[b]], (1 - gt_perm[b, :pred_ns[b], :gt_ns[b]])))
            ## false negative
            loss_n += torch.sum(torch.mul(gt_perm[b, :pred_ns[b], :gt_ns[b]], (1 - pred_perm[b, :pred_ns[b], :gt_ns[b]])))

            n_sum += pred_ns[b].to(n_sum.dtype).to(pred_perm.device)
        return torch.exp(alpha * loss_n / n_sum) - 1 + torch.exp(beta * loss_p / n_sum) - 1
           # a more general setting
           # alpha = 2
           # beta = 2
           # return (torch.exp(alpha * loss_n / n_sum) + torch.exp(beta * loss_p / n_sum))/4  
