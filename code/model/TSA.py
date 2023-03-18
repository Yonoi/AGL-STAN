from icecream import ic
import torch
import torch.nn as nn
from model.AGL import AGL

class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, filter_size):
        super(FeedForwardNetwork, self).__init__()

        self.layer = nn.Sequential(
            nn.Linear(hidden_size, filter_size), 
            nn.ReLU(), 
            nn.Linear(filter_size, hidden_size),
        )

    def forward(self, x):
        return self.layer(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, head_size=6):
        super(MultiHeadAttention, self).__init__()
        self.head_size = head_size

        att_size = hidden_size // head_size
        self.att_size = att_size
        self.scale = att_size ** -0.5

        self.linear_q = nn.Linear(hidden_size, head_size * att_size, bias=False)
        self.linear_k = nn.Linear(hidden_size, head_size * att_size, bias=False)
        self.linear_v = nn.Linear(hidden_size, head_size * att_size, bias=False)

        self.output_layer = nn.Linear(head_size * att_size, hidden_size, bias=False)
    
    def forward(self, q, k, v):
        orig_q_size = q.size()

        d_k = self.att_size
        d_v = self.att_size

        batch_size = q.size(0)
        q = self.linear_q(q).view(batch_size, -1, self.head_size, d_k)
        k = self.linear_k(k).view(batch_size, -1, self.head_size, d_k)
        v = self.linear_v(v).view(batch_size, -1, self.head_size, d_v)

        q = q.transpose(1, 2)
        v = v.transpose(1, 2)
        k = k.transpose(1, 2).transpose(2, 3)

        q.mul_(self.scale)
        x = torch.matmul(q, k) 
        x = torch.softmax(x, dim=3)
        x = x.matmul(v)

        x = x.transpose(1, 2).contiguous()
        x = x.view(batch_size, -1, self.head_size * d_v)

        x = self.output_layer(x)

        return x

class TSALayer(nn.Module):
    def __init__(self, hidden_size, filter_size):
        super(TSALayer, self).__init__()

        self.self_attention_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.self_attention = MultiHeadAttention(hidden_size)

        self.ffn_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.ffn = FeedForwardNetwork(hidden_size, filter_size)

    def forward(self, x):
        y = self.self_attention_norm(x)
        y = self.self_attention(y, y, y)
        x = x + y

        y = self.ffn_norm(x)
        y = self.ffn(y)
        x = x + y

        return x

class AGLLayer(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, embed_dim):
        super(AGLLayer, self).__init__()
        self.node_num = node_num
        self.hidden_dim = dim_out
        self.gconv_layer = AGL(dim_in, dim_out, cheb_k, embed_dim)

    def forward(self, x, node_embeddings):
        
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        if x.shape[2] != self.node_num:
            x = x.view(batch_size, seq_len, self.node_num, -1)
        gconv_lst = []
        for t in range(seq_len):
            gconv_lst.append(self.gconv_layer(x[:, t, :, :], node_embeddings))
        output = torch.stack(gconv_lst, dim=1).view(batch_size, seq_len, -1)
        return output
