import torch
import torch.nn as nn
import torch.nn.functional as F
from math import floor, ceil

from fast_transformers.attention import AttentionLayer, LinearAttention, FullAttention
from utils.attention import AttentionConv1DLayer

from fast_transformers.transformers import TransformerEncoderLayer, TransformerEncoder
from utils.attention import AttentionConv1D4Layer, AttentionConv1DLayer
class Params:
        class encoder:
            leakiness = 0.2
            ch_in = [2, 16, 32, 64, 128, 256]
            ch_out = [16, 32, 64, 128, 256, 512]
            kernel_size = [(5,5)] *6
            stride = [(2,2), (2,2), (2,2), (2,2), (2,2), (2,2)] 
        class decoder:
            ch_in = [512, 512, 256, 128, 64, 32]
            ch_out = [256, 128, 64, 32, 16, 2]
            kernel_size = [(5,5)] *6
            stride = [(2,2), (2,2), (2,2), (2,2), (2,2), (2,2)] 
        

class Query_UNet(nn.Module):
    def __init__(self):
        super(Query_UNet, self).__init__()
        self.convs = []
        self.deconvs = []
        self.params = Params
        self.frist_conv = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=2, kernel_size=(2,1), stride=1),
            # nn.BatchNorm2d(2),
            nn.ReLU()
        )
        for i in range(len(self.params.encoder.ch_out)):
            self.convs.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels = self.params.encoder.ch_in[i],
                        out_channels = self.params.encoder.ch_out[i], 
                        kernel_size = self.params.encoder.kernel_size[i],
                        stride = self.params.encoder.stride[i]
                    ),
                    nn.BatchNorm2d(self.params.encoder.ch_out[i]),
                    nn.LeakyReLU(self.params.encoder.leakiness),
                )
            )
        for i in range(len(self.params.decoder.ch_out)):
            self.deconvs.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        in_channels = self.params.decoder.ch_in[i],
                        out_channels = self.params.decoder.ch_out[i],
                        kernel_size = self.params.decoder.kernel_size[i],
                        stride = self.params.decoder.stride[i]
                    ),
                    nn.BatchNorm2d(self.params.decoder.ch_out[i]),
                    nn.ReLU()
                )
            )
        self.last_conv = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=2, kernel_size=(2,1), stride=1, padding=(1, 0)), 
            # nn.BatchNorm2d(2),
            nn.ReLU(),
            )
        self.convs = nn.ModuleList(self.convs)
        self.deconvs = nn.ModuleList(self.deconvs)

        self.att = nn.ModuleList()
        for i in range(3):
            self.att.append(TransformerEncoder(
                    [ TransformerEncoderLayer(
                        attention = AttentionLayer(
                            FullAttention(), 
                            d_model=self.params.encoder.ch_out[-1],
                            n_heads=16,
                            ),
                        d_model = self.params.encoder.ch_out[-1],
                        activation="relu",
                        d_ff = 4,
                        dropout=0.2,
                        
                    )],
                    norm_layer= None
                ))

        self.pos_embed = PositionalEncoding(512)            


    def same_padding_conv(self, x, conv):
        dim = len(x.size())
        if dim == 4:
            b, c, h, w = x.size()
        elif dim == 5:
            b, t, c, h, w = x.size()
        elif dim == 3:
            x = x.unsqueeze(1)
            b, c, h, w = x.size()
        else:
            raise NotImplementedError()

        if isinstance(conv[0], nn.Conv2d):
            padding = ((w // conv[0].stride[1] - 1) * conv[0].stride[1] + conv[0].kernel_size[1] - w)
            padding_l = floor(padding / 2)
            padding_r = ceil(padding / 2)
            padding = ((h // conv[0].stride[0] - 1) * conv[0].stride[0] + conv[0].kernel_size[0] - h)
            padding_t = floor(padding / 2)
            padding_b = ceil(padding / 2)
            x = F.pad(x, pad = (padding_l,padding_r,padding_t,padding_b))
            x = conv(x)
        elif isinstance(conv[0], nn.ConvTranspose2d):
            padding = ((w - 1) * conv[0].stride[1] + conv[0].kernel_size[1] - w * conv[0].stride[1])
            padding_l = floor(padding / 2)
            padding_r = ceil(padding / 2)
            padding = ((h - 1) * conv[0].stride[0] + conv[0].kernel_size[0] - h * conv[0].stride[0])
            padding_t = floor(padding / 2)
            padding_b = ceil(padding / 2)
            x = conv(x)
            x = x[:,:,padding_t:-padding_b,padding_l:-padding_r]

        return x
    
    def forward(self, x):
        """
        Args:
            x: (B, C, H, W)
            latent: (B, latent_size)
        """
        B, C, H, W = x.size()
        skip_connections = []
        

        mix = x.detach().clone()
        # print(x.size(), 'init')
        x = self.frist_conv(x)
        for layer_idx, conv in enumerate(self.convs):
            x = self.same_padding_conv(x, conv)
            # print(x.size(), 'conv', layer_idx)
            if layer_idx != len(self.convs) - 1:
                skip_connections.append(x)
        # print(x.size(), 'mid')
        x_ = x.view(x.shape[0], x.shape[1], -1).permute(0,2,1)

        x_ = self.pos_embed(x_)
        for module in self.att:
            x_ = module(x_)
        x = x_.permute(0,2,1).view(x.shape) + x


        for layer_idx, deconv in enumerate(self.deconvs):
            x = self.same_padding_conv(x, deconv)
            # print(x.size(), 'deconv', layer_idx)
            # if layer_idx < 3:
                # x = F.dropout2d(x, p = 0.5)
            if layer_idx != len(self.deconvs)-1:
                x = torch.cat([skip_connections.pop(), x], dim = 1)
        x = self.last_conv(x)
        return mix * F.relu(x), F.relu(x)
    
    
import math
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=1300):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1), :]
        return self.dropout(x)
        
