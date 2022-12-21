from abc import ABC
import torch
import torch.nn as nn
from torch import Tensor

from utils.transformers import TransformerEncoder, TransformerEncoderLayer 
from utils.attention import  AttentionConv1DLayer
from fast_transformers.attention import FullAttention

def get_activation_by_name(activation):
    if activation == "leaky_relu":
        return nn.LeakyReLU
    elif activation == "relu":
        return nn.ReLU
    elif activation == "sigmoid":
        return nn.Sigmoid
    elif activation == "tanh":
        return nn.Tanh
    elif activation == "softmax":
        return nn.Softmax
    elif activation == "identity":
        return nn.Identity
    else:
        return None

class WI_Module(nn.Module, ABC):
    def __init__(self):
        super().__init__()

    def init_weights(self):
        init_weights_functional(self, self.activation)


def init_weights_functional(module, activation='default'):
    if isinstance(activation, nn.ReLU):
        for param in module.parameters():
            if param.dim() > 1:
                nn.init.kaiming_normal_(param, nonlinearity='relu')

    elif activation == 'relu':
        for param in module.parameters():
            if param.dim() > 1:
                nn.init.kaiming_normal_(param, nonlinearity='relu')

    elif isinstance(activation, nn.LeakyReLU):
        for param in module.parameters():
            if param.dim() > 1:
                nn.init.kaiming_normal_(param, nonlinearity='leaky_relu')

    elif activation == 'leaky_relu':
        for param in module.parameters():
            if param.dim() > 1:
                nn.init.kaiming_normal_(param, nonlinearity='leaky_relu')

    elif isinstance(activation, nn.Sigmoid):
        for param in module.parameters():
            if param.dim() > 1:
                nn.init.xavier_normal_(param)

    elif activation == 'sigmoid':
        for param in module.parameters():
            if param.dim() > 1:
                nn.init.xavier_normal_(param)

    elif isinstance(activation, nn.Tanh):
        for param in module.parameters():
            if param.dim() > 1:
                nn.init.xavier_normal_(param)

    elif activation == 'tanh':
        for param in module.parameters():
            if param.dim() > 1:
                nn.init.xavier_normal_(param)

    else:
        for param in module.parameters():
            if param.dim() > 1:
                nn.init.xavier_normal_(param)

class TFC(WI_Module):
    """ [B, in_channels, T, F] => [B, gr, T, F] """

    def __init__(self, in_channels, num_layers, gr, kt, kf, activation, f, isatt=False):
        """
        in_channels: number of input channels
        num_layers: number of densely connected conv layers
        gr: growth rate
        kt: kernel size of the temporal axis.
        kf: kernel size of the freq. axis
        activation: activation function
        """
        super(TFC, self).__init__()

        c = in_channels
        self.H = nn.ModuleList()
        for i in range(num_layers):
            self.H.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=c, out_channels=gr, kernel_size=(kf, kt), stride=1,
                              padding=(kt // 2, kf // 2)),
                    nn.BatchNorm2d(gr),
                    activation(),
                )
            )
            c += gr

        self.activation = self.H[-1][-1]
        self.isatt = isatt
        if isatt:
            self.att = nn.ModuleList()
            for i in range(3):
                self.att.append(
                    TransformerEncoder(
                        [ TransformerEncoderLayer(
                            attention = AttentionConv1DLayer(
                                FullAttention(), 
                                d_model=f,
                                n_heads=16,
                                ),
                            d_model = f,
                            activation="relu",
                            d_ff = 1,
                            dropout=0,
                        )],
                        norm_layer= None
                    )
                )

    def forward(self, x):
        """ [B, in_channels, T, F] => [B, gr, T, F] """
        x_ = self.H[0](x)
        for h in self.H[1:]:
            x = torch.cat((x_, x), 1)
            x_ = h(x)
        
        if self.isatt:
            for att in self.att:
                x_ = x_ + att(torch.flatten(x_,0,1)).reshape(x_.shape)
            
        return x_




class Dense_UNET(nn.Module):
    @staticmethod
    def get_arg_keys():
        return ['n_blocks',
                'input_channels',
                'internal_channels',
                'n_internal_layers',
                'mk_block_f',
                'mk_ds_f',
                'mk_us_f',
                'first_conv_activation',
                'last_activation',
                't_down_scale',
                'f_down_scale']

    def __init__(self,
                 n_fft,  # framework's
                 n_blocks,
                 input_channels,
                 internal_channels,
                 n_internal_layers,
                 mk_block_f,
                 mk_ds_f,
                 mk_us_f,
                 first_conv_activation,
                 last_activation,
                 t_down_layers,
                 f_down_layers
                 ):

        first_conv_activation = get_activation_by_name(first_conv_activation)
        last_activation = get_activation_by_name(last_activation)

        super(Dense_UNET, self).__init__()

        '''num_block should be an odd integer'''
        assert n_blocks % 2 == 1

        ###########################################################
        # Block-independent Section

        dim_f = n_fft // 2
        input_channels = input_channels

        self.first_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=input_channels,
                out_channels=internal_channels,
                kernel_size=(1, 2),
                stride=1
            ),
            nn.BatchNorm2d(internal_channels),
            first_conv_activation(),
        )

        self.encoders = nn.ModuleList()
        self.downsamplings = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.upsamplings = nn.ModuleList()

        self.last_conv = nn.Sequential(

            nn.Conv2d(
                in_channels=internal_channels,
                out_channels=input_channels,
                kernel_size=(1, 2),
                stride=1,
                padding=(0, 1)
            ),
            last_activation()
        )

        self.n = n_blocks // 2

        if t_down_layers is None:
            t_down_layers = list(range(self.n))
        elif n_internal_layers == 'None':
            t_down_layers = list(range(self.n))
        else:
            t_down_layers = string_to_list(t_down_layers)

        if f_down_layers is None:
            f_down_layers = list(range(self.n))
        elif n_internal_layers == 'None':
            f_down_layers = list(range(self.n))
        else:
            f_down_layers = string_to_list(f_down_layers)

        # Block-independent Section
        ###########################################################

        ###########################################################
        # Block-dependent Section

        f = dim_f

        i = 0
        for i in range(self.n):
            self.encoders.append(mk_block_f(internal_channels, internal_channels, f))
            ds_layer, f = mk_ds_f(internal_channels, i, f, t_down_layers)
            self.downsamplings.append(ds_layer)

        self.mid_block = mk_block_f(internal_channels, internal_channels, f, isatt=True)

        for i in range(self.n):
            us_layer, f = mk_us_f(internal_channels, i, f, self.n, t_down_layers)
            self.upsamplings.append(us_layer)
            self.decoders.append(mk_block_f(2 * internal_channels, internal_channels, f))

        # Block-dependent Section
        ###########################################################

        self.activation = self.last_conv[-1]

    def forward(self, x):
        mix = x.detach().clone()
        x = x.permute(0,1,3,2)
        x = self.first_conv(x)
        encoding_outputs = []
        for i in range(self.n):
            x = self.encoders[i](x)
            encoding_outputs.append(x)
            x = self.downsamplings[i](x)
        x = self.mid_block(x)

        for i in range(self.n):
            x = self.upsamplings[i](x)
            x = torch.cat((x, encoding_outputs[-i - 1]), 1)
            x = self.decoders[i](x)
        x = self.last_conv(x)
        return x.permute(0,1,3,2)*mix, x.permute(0,1,3,2)

    def init_weights(self):

        init_weights_functional(self.first_conv, self.first_conv[-1])

        for encoder, downsampling in zip(self.encoders, self.downsamplings):
            encoder.init_weights()
            init_weights_functional(downsampling)

        self.mid_block.init_weights()

        for decoder, upsampling in zip(self.decoders, self.upsamplings):
            decoder.init_weights()
            init_weights_functional(upsampling)

        init_weights_functional(self.last_conv, self.last_conv[-1])


class TFC_NET(Dense_UNET):

    def __init__(self,
                 n_fft,
                 n_blocks, input_channels, internal_channels, n_internal_layers,
                 first_conv_activation, last_activation,
                 t_down_layers, f_down_layers,
                 kernel_size_t, kernel_size_f,
                 tfc_activation,
    ):

        tfc_activation = get_activation_by_name(tfc_activation)

        def mk_tfc(in_channels, internal_channels, f, isatt=False):
            return TFC(in_channels, n_internal_layers, internal_channels, kernel_size_t, kernel_size_f, tfc_activation, f, isatt)


        def mk_ds(internal_channels, i, f, t_down_layers):
            if t_down_layers is None:
                scale = (2, 2)
            else:
                scale = (2, 2) if i in t_down_layers else (1, 2)
            ds = nn.Sequential(
                nn.Conv2d(in_channels=internal_channels, out_channels=internal_channels,
                          kernel_size=scale, stride=scale),
                nn.BatchNorm2d(internal_channels)
            )
            return ds, f // scale[-1]

        def mk_us(internal_channels, i, f, n, t_down_layers):
            if t_down_layers is None:
                scale = (2, 2)
            else:
                scale = (2, 2) if i in [n - 1 - s for s in t_down_layers] else (1, 2)

            us = nn.Sequential(
                nn.ConvTranspose2d(in_channels=internal_channels, out_channels=internal_channels,
                                   kernel_size=scale, stride=scale),
                nn.BatchNorm2d(internal_channels)
            )
            return us, f * scale[-1]

        super(TFC_NET, self).__init__(n_fft,  # framework's
                                      n_blocks,
                                      input_channels,
                                      internal_channels,
                                      n_internal_layers,
                                      mk_tfc,
                                      mk_ds,
                                      mk_us,
                                      first_conv_activation,
                                      last_activation,
                                      t_down_layers,
                                      f_down_layers)