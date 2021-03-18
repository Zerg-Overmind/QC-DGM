import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

class Gconv(nn.Module):
    """
    GCN layer
    """
    def __init__(self, in_features, out_features):
        super(Gconv, self).__init__()
        self.num_inputs = in_features
        self.num_outputs = out_features
        self.a_fc = nn.Linear(self.num_inputs, self.num_outputs)
        self.u_fc = nn.Linear(self.num_inputs, self.num_outputs)
 
    def forward(self, A, x, norm=True):
        if norm is True:
            A = F.normalize(A, p=1, dim=-2)
   #  
        ax = self.a_fc(x)
        ux = self.u_fc(x)
       
        x = torch.bmm(A, F.relu(ax)) + F.relu(ux)
        return x

class Gconv_layer(nn.Module):
    """
    Perform graph convolution on two input graphs (G1, G2)
    """
    def __init__(self, in_features, num_features):
        super(Gconv_layer, self).__init__()
        self.gconv = Gconv(in_features, num_features)

    def forward(self, g1, g2):
        emb1 = self.gconv(*g1)
        emb2 = self.gconv(*g2)
        # embx are tensors of size (bs, N, num_features)
        return emb1, emb2
