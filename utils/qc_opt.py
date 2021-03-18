import torch
from utils.hungarian import hungarian
def qc_opt(A_src, A_tgt, s_pred, X, lb):

    lap_solver = hungarian
    for niter in range(10):
      perm_src = torch.bmm(torch.bmm(X.transpose(1,2), A_src), X)
      perm_tgt = torch.bmm(torch.bmm(X, A_tgt), X.transpose(1,2))
      P = (A_src - perm_tgt)
      AP = torch.mul(A_src.cuda(), P.cuda())
      V = -2 * AP.cuda()
      V_XB = torch.bmm(torch.bmm(V.transpose(1,2), X), A_tgt)
      VX_B = torch.bmm(torch.bmm(V, X), A_tgt.transpose(1,2))
      N = -s_pred
      G = lb*(V_XB + VX_B) + (1-lb)*N
      S = lap_solver(-G)
      lam = 2/(niter+2)
      Xnew = X + lam*(S - X)
      X = Xnew
    return X
