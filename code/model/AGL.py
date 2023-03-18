from icecream import ic
import torch
import torch.nn.functional as F
import torch.nn as nn

class AGL(nn.Module):
    def __init__(self, dim_in, dim_out, cheb_k, embed_dim):
        super(AGL, self).__init__()
        self.cheb_k = cheb_k
        self.weights_pool = nn.Parameter(torch.FloatTensor(embed_dim, cheb_k, dim_in, dim_out))
        self.bias_pool = nn.Parameter(torch.FloatTensor(embed_dim, dim_out))
        self.linear = nn.Linear(77, 77)

    def forward(self, x, node_embeddings):

        node_num = node_embeddings.shape[0]
        supports = F.softmax(F.relu(self.linear(torch.mm(node_embeddings, node_embeddings.transpose(0, 1)))), dim=1)
        support_set = [torch.eye(node_num).to(supports.device), supports]

        for k in range(2, self.cheb_k):
            support_set.append(torch.matmul(2 * supports, support_set[-1]) - support_set[-2])
        supports = torch.stack(support_set, dim=0)
        
        weights = torch.einsum('nd,dkio->nkio', node_embeddings, self.weights_pool)  
        bias = torch.matmul(node_embeddings, self.bias_pool)  

        x_g = torch.einsum("knm,bmc->bknc", supports, x)
        x_g = x_g.permute(0, 2, 1, 3) 
        x_gconv = torch.einsum('bnki,nkio->bno', x_g, weights) + bias     
        return x_gconv