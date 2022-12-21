import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaIN(nn.Module):
    def __init__(self, n_channel, latent_size, affine):
        super(AdaIN. self).__init__()
        self.n_channel = n_channel
        self.latent_size = latent_size
        self.fcs = nn.Linear(latent_size, n_channel, bias=False)
        self.fcb = nn.Linear(latent_size, n_channel, bias=False)
        self.norm = nn.InstanceNorm2d(n_channel)
        self.affine = affine
        
    def forward(self, input):
        """
        Args: 
            input: [x, latent]
                |- x: (B, C, H, W)
                |- latent: (B, latent_size)
        """
        x, latent = input
        B, C = x.size(0), x.size(1)
        x = self.norm(x)
        
        if self.affine:
            # (B, C)
            scale = self.fcs(latent).reshape(B, C, 1, 1)
            bias = self.fcb(latent).reshape(B, C, 1, 1)
            x = x * scale + bias
            
        return x
    
class Query_UNet(nn.Module):
    def __init__(self, config):
        super(Query_UNet, self).__init__()
        self.convs = []
        self.deconvs = []
        self.kernel_size = config.encoder.kernel_size
        self.stride = config.encoder.stride
        for i in range(len(config.encoder.ch_out)):
            self.convs.append(
                nn.Sequential(
                    AdaIN(config.encoder.ch_in[i] + (0 if i != 0 else config.latent_size), config.latent_size, False),
                    nn.Conv2d(
                        in_channels = config.encoder.ch_in[i] + (0 if i != 0 else config.latent_size),
                        out_channels = config.encoder.ch_out[i], 
                        kernel_size = self.kernel_size,
                        stride = self.encoder.stride[i]
                    ),
                    nn.LeakyReLU(config.encoder.leakiness),
                )
            )
        for i in range(len(config.decoder.ch_out)):
            self.deconvs.append(
                nn.Sequential(
                    AdaIN(config.decoder.ch_in[i], config.latent_size, True),
                    nn.ConvTranspose2d(
                        in_channels = config.decoder.ch_in[i],
                        out_channels = config.decoder.ch_out[i],
                        kernel_size = config.decoder.kernel_size,
                        stride = config.decoder.stride
                    ),
                    nn.ReLU()
                )
            )
        self.convs = nn.ModuleList(self.convs)
        self.deconvs = nn.ModuleList(self.deconvs)
        
    def same_padding_conv(x, latent, conv):
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

        if isinstance(conv[1], nn.Conv2d):
            padding = ((w // conv[1].stride[0] - 1) * conv[1].stride[0] + conv[1].kernel_size[0] - w)
            padding_l = floor(padding / 2)
            padding_r = ceil(padding / 2)
            padding = ((h // conv[1].stride[1] - 1) * conv[1].stride[1] + conv[1].kernel_size[1] - h)
            padding_t = floor(padding / 2)
            padding_b = ceil(padding / 2)
            x = F.pad(x, pad = (padding_l,padding_r,padding_t,padding_b))
            input = [x, latent]
            x = conv(input)
        elif isinstance(conv[1], nn.ConvTranspose2d):
            padding = ((w - 1) * conv[1].stride[0] + conv[1].kernel_size[0] - w * conv[1].stride[0])
            padding_l = floor(padding / 2)
            padding_r = ceil(padding / 2)
            padding = ((h - 1) * conv[1].stride[1] + conv[1].kernel_size[1] - h * conv[1].stride[1])
            padding_t = floor(padding / 2)
            padding_b = ceil(padding / 2)
            input = [x, latent]
            x = conv(input)
            x = x[:,:,padding_t:-padding_b,padding_l:-padding_r]

        return x
    
    def forward(self, x, latent):
        """
        Args:
            x: (B, C, H, W)
            latent: (B, latent_size)
        """
        B, C, H, W = x.size()
        skip_connections = []
        
        # (B, latent_size, H, W)
        latent2cat= latent.reshape(B,-1,1,1).repeat(1, 1, H, W)
        x = torch.cat([x, latent2cat], dim=1)
        
        for layer_idx, conv in enumerate(self.convs):
            x = self.same_padding_conv(x, latent, conv)
            #print(x.size(), 'conv', layer_idx)
            if layer_idx != len(self.convs) - 1:
                skip_connections.append(x)
        for layer_idx, deconv in enumerate(self.deconvs):
            x = self.same_padding_conv(x, latent, deconv)
            #print(x.size(), 'deconv', layer_idx)
            if layer_idx < 3:
                x = F.dropout2d(x, p = 0.5)
            if layer_idx != len(self.deconvs)-1:
                x = torch.cat([skip_connections.pop(), x], dim = 1)
        return torch.sigmoid(x)
    
    

        
        