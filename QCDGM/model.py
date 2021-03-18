import torch
import torch.nn as nn
import numpy as np
from torch import Tensor
from utils.sh_layer import sh
from utils.sm_layer import sm
from utils.feature_align import feature_align
from QCDGM.gconv import Gconv_layer
from QCDGM.affinity_layer import Affinity
from utils.hungarian import hungarian
import torch.nn.functional as F
from utils.config import cfg
from utils.hungarian import hungarian
import utils.backbone
CNN = eval('utils.backbone.{}'.format(cfg.BACKBONE))

def kronecker_torch(t1: Tensor, t2: Tensor):
    batch_num = t1.shape[0]
    t1dim1, t1dim2 = t1.shape[1], t1.shape[2]
    t2dim1, t2dim2 = t2.shape[1], t2.shape[2]
    t1 = t1.view(batch_num, -1, 1)
    t2 = t2.view(batch_num, 1, -1)
    tt = torch.bmm(t1, t2)
    tt = tt.view(batch_num, t1dim1, t1dim2, t2dim1, t2dim2)
    tt = tt.permute([0, 1, 3, 2, 4])
    tt = tt.reshape(batch_num, t1dim1 * t2dim1, t1dim2 * t2dim2)
    return tt

class Net(CNN):
    def __init__(self):
        super(Net, self).__init__()
        self.sh_layer = sh(max_iter=cfg.QCDGM.BS_ITER_NUM, epsilon=cfg.QCDGM.BS_EPSILON)
        self.sm_layer = sm(alpha=cfg.QCDGM.SCALE_ALPHA)
        self.l2norm = nn.LocalResponseNorm(cfg.QCDGM.FEATURE_CHANNEL * 2, alpha=cfg.QCDGM.FEATURE_CHANNEL * 2, beta=0.5, k=0)
        self.gnn_layer = cfg.QCDGM.GNN_LAYER
        self.softmax = nn.Softmax(dim=-1) 

        for i in range(self.gnn_layer):
            if i == 0:
                gnn_layer = Gconv_layer(cfg.QCDGM.FEATURE_CHANNEL * 2 + 2, cfg.QCDGM.GNN_FEAT + 2)
                affinity_layer = Affinity(cfg.QCDGM.GNN_FEAT + 2)
            else:
                gnn_layer = Gconv_layer(cfg.QCDGM.GNN_FEAT * 2 + 4, cfg.QCDGM.GNN_FEAT + 2)
                affinity_layer = Affinity(cfg.QCDGM.GNN_FEAT + 2)
            self.add_module('gnn_layer_{}'.format(i), gnn_layer)
            self.add_module('affinity_{}'.format(i), affinity_layer)
    def forward(self, src, tgt, P_src, P_tgt, G_src, G_tgt, H_src, H_tgt, ns_src, ns_tgt, K_G, K_H, edge_src, edge_tgt, edge_feat1, edge_feat2, perm_mat, type='img'):
        if type == 'img' or type == 'image':
            # extract feature
            src_node = self.node_layers(src)
            src_edge = self.edge_layers(src_node)
            tgt_node = self.node_layers(tgt)
            tgt_edge = self.edge_layers(tgt_node)

            # feature normalization
            src_node = self.l2norm(src_node)
            src_edge = self.l2norm(src_edge)
            tgt_node = self.l2norm(tgt_node)
            tgt_edge = self.l2norm(tgt_edge)
			
			

            # arrange features
            U_src = feature_align(src_node, P_src, ns_src, cfg.PAIR.RESCALE)
            F_src = feature_align(src_edge, P_src, ns_src, cfg.PAIR.RESCALE)
            U_tgt = feature_align(tgt_node, P_tgt, ns_tgt, cfg.PAIR.RESCALE)
            F_tgt = feature_align(tgt_edge, P_tgt, ns_tgt, cfg.PAIR.RESCALE)
        elif type == 'feat' or type == 'feature':
            U_src = src[:, :src.shape[1] // 2, :]
            F_src = src[:, src.shape[1] // 2:, :]
            U_tgt = tgt[:, :tgt.shape[1] // 2, :]
            F_tgt = tgt[:, tgt.shape[1] // 2:, :]
        else:
            raise ValueError('unknown type string {}'.format(type))

        
       
        A_src = torch.bmm(G_src, H_src.transpose(1, 2))
        A_tgt = torch.bmm(G_tgt, H_tgt.transpose(1, 2))
        P1_src = torch.zeros_like(P_src)
        P2_tgt = torch.zeros_like(P_tgt)
        for k in range(P_src.shape[0]):
            for i in range(P_src.shape[1]):
               for j in range(P_tgt.shape[2]):
                  if torch.norm(P_src[k, i, :]) == 0:
                      P1_src[k, i, j] = 0
                      P2_tgt[k, i, j] = 0
                  else:
                      P1_src[k, i, j] = P_src[k, i, j]/torch.norm(P_src[k, i, :])
                      P2_tgt[k, i, j] = P_tgt[k, i, j]/torch.norm(P_tgt[k, i, :])     
        
        ## Node embedding with unary geometric prior
        emb1, emb2 = torch.cat((U_src, F_src, P1_src.transpose(1,2)), dim=1).transpose(1, 2), torch.cat((U_tgt, F_tgt, P2_tgt.transpose(1,2)), dim=1).transpose(1, 2)

        for i in range(self.gnn_layer):

            gnn_layer = getattr(self, 'gnn_layer_{}'.format(i))
            if i==0:
              emb1, emb2 = gnn_layer([A_src, emb1], [A_tgt, emb2])
            else:
              emb1_new = torch.cat((emb1, torch.bmm(s, emb2)), dim=-1)
              emb2_new = torch.cat((emb2, torch.bmm(s.transpose(1, 2), emb1)), dim=-1)              
              emb1, emb2 = gnn_layer([AA_src, emb1_new], [BB_tgt, emb2_new])
            affinity = getattr(self, 'affinity_{}'.format(i))   
            s = affinity(emb1, emb2) 

            AA = torch.ones([s.shape[0], s.shape[1], s.shape[1]]).to(s.device)
            BB = torch.ones([s.shape[0], s.shape[2], s.shape[2]]).to(s.device)

         
            ## Commutative function f
            for kk in range(s.shape[0]):
                for ll in range(s.shape[1]):
                   for qq in range(s.shape[2]):
                        AA[kk,ll,qq] = torch.exp(torch.matmul(emb1[kk,ll,:]/torch.norm(emb1[kk,ll,:]), emb1[kk,qq,:]/torch.norm(emb1[kk,qq,:])))
                        BB[kk,ll,qq] = torch.exp(torch.matmul(emb2[kk,ll,:]/torch.norm(emb2[kk,ll,:]), emb2[kk,qq,:]/torch.norm(emb2[kk,qq,:])))
       
            ## Pairwise structural context
            AA_src = torch.mul(AA, A_src)
            BB_tgt = torch.mul(BB, A_tgt)
       

            if i == 1:
              ## QC-optimization
              X = s
              lb = 0.1  ## Balancing unary term and pairwise term
              for niter in range(3):
                for ik in range(3):       

                   perm_tgt = torch.bmm(torch.bmm(X, BB_tgt), X.transpose(1,2))
                   P = (AA_src - perm_tgt)
                   V = -2*P.cuda()
                   V_X = torch.bmm(V.transpose(1,2), X)
                   V_XB = torch.bmm(V_X, BB_tgt)
                   VX = torch.bmm(V, X)
                   VX_B = torch.bmm(VX, BB_tgt.transpose(1,2))     
                   N = -s
                   G = lb*(V_XB + VX_B) + (1-lb)*N
                   G_sim = -G - torch.min(-G) 
                   S = self.sh_layer(G_sim, ns_src, ns_tgt)
                   lam = 2/(ik+2)
                   Xnew = X + lam*(S - X)
                   X = Xnew
                X = self.sh_layer(X, ns_src, ns_tgt)        
              s = 1*s + 0.5*X  ## For faster convergence
         
            ## Normalization in evaluation
            if self.training == False:    
              for b in range(s.shape[0]):
                 s[b, :, :] = s[b, :, :].clone()/torch.max(s[b, :, :].clone())
          
            s = self.sm_layer(s, ns_src, ns_tgt)
            s = self.sh_layer(s, ns_src, ns_tgt)
            
        return s,  U_src, F_src, U_tgt, F_tgt, AA, BB
