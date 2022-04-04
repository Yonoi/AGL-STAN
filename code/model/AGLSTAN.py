import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from icecream import ic
from model.TSA import AGLLayer   
from model.TSA import TSALayer

class AGLTSA(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, embed_dim, hidden_size, filter_size, num_layers):
        super(AGLTSA, self).__init__()
        self.hidden_size = hidden_size
        
        # For positional encoding
        num_timescales = self.hidden_size // 2
        max_timescale = 10000.0
        min_timescale = 1.0
        log_timescale_increment = (
            math.log(float(max_timescale) / float(min_timescale)) /
            max(num_timescales - 1, 1))
        inv_timescales = min_timescale * torch.exp(
            torch.arange(num_timescales, dtype=torch.float32) *
            -log_timescale_increment)
        self.register_buffer('inv_timescales', inv_timescales)

        self.gconv_layers = nn.ModuleList(
            [AGLLayer(node_num, dim_in, dim_out, cheb_k, embed_dim)
            for _ in range(num_layers)]
        ) 

        self.encoders = nn.ModuleList(
            [TSALayer(hidden_size, filter_size)
            for _ in range(num_layers)]
        )

        self.last_norm = nn.LayerNorm(hidden_size, eps=1e-6)
    
    def get_position_encoding(self, x):
        max_length = x.size()[1]
        position = torch.arange(max_length, dtype=torch.float32,
                                device=x.device)
        scaled_time = position.unsqueeze(1) * self.inv_timescales.unsqueeze(0)
        signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)],
                           dim=1)
        signal = F.pad(signal, (0, 0, 0, self.hidden_size % 2))
        signal = signal.view(1, max_length, self.hidden_size)
        return signal

    def forward(self, inputs, node_embeddings):
        """
        inputs : [B, T, N, C]
        """
        # for encoding
        pos_enc = True
        encoder_output = inputs
        for gconv_layer, enc_layer in zip(self.gconv_layers, self.encoders):
            gconv_output = gconv_layer(encoder_output, node_embeddings)
            if pos_enc:
                gconv_output += self.get_position_encoding(gconv_output)
                pos_enc = False
            encoder_output = enc_layer(gconv_output)
        
        return self.last_norm(encoder_output)


class AGLSTAN(nn.Module):
    def __init__(self, args):
        super(AGLSTAN, self).__init__()
        self.num_node = args.num_nodes
        self.batch_size = args.batch_size
        self.input_dim = args.input_dim
        # self.hidden_dim = args.rnn_units
        self.output_dim = args.output_dim
        self.horizon = args.horizon
        self.window = args.window
        self.num_layers = args.num_layers
        self.filter_size = args.filter_size

        self.default_graph = args.default_graph
        self.node_embeddings = nn.Parameter(torch.randn(self.num_node, args.embed_dim), requires_grad=True)

        self.encoder = AGLTSA(args.num_nodes, args.input_dim, args.output_dim, args.cheb_k,
                                args.embed_dim, args.num_nodes * self.output_dim, args.filter_size, args.num_layers)

        self.end_conv = nn.Conv2d(self.window, self.horizon, padding=(2, 2), kernel_size=(5, 5), bias=True)

    def forward(self, source, targets, teacher_forcing_ratio=0.5):
        #source: B, T_1, N, D
        #target: B, T_2, N, D

        output = self.encoder(source, self.node_embeddings)      #B, T, N, hidden
        output = output.view(self.batch_size, self.window, self.num_node, -1)
        output = self.end_conv(output)

        return output