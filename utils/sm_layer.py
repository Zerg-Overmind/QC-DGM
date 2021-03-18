import torch
import torch.nn as nn


class sm(nn.Module):
    """
    Input: permutation or doubly stochastic matrix s
           ///point set on source image P_src
           ///point set on target image P_tgt
           ground truth number of effective points in source image ns_gt
    Output: softmax matrix s
    """
    def __init__(self, alpha=200, pixel_thresh=None):
        super(sm, self).__init__()
        self.alpha = alpha
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, s, nrow_gt, ncol_gt=None):
        ret_s = torch.zeros_like(s)
        # filter dummy nodes
        for b, n in enumerate(nrow_gt):
            if ncol_gt is None:
                ret_s[b, 0:n, :] = \
                    self.softmax(self.alpha * s[b, 0:n, :])
            else:
                ret_s[b, 0:n, 0:ncol_gt[b]] =\
                    self.softmax(self.alpha * s[b, 0:n, 0:ncol_gt[b]])

        return ret_s
