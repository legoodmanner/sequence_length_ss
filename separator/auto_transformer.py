import torch
import torch.nn as nn
import torch.nn.functional as F
from fast_transformers.builders import TransformerEncoderBuilder
from torchlibrosa.stft import Spectrogram, LogmelFilterBank
from torchlibrosa.augmentation import SpecAugmentation
from fast_transformers.transformers import TransformerEncoder, TransformerEncoderLayer, TransformerDecoderLayer, TransformerDecoder
from fast_transformers.attention import AttentionLayer, LinearAttention, FullAttention
import fast_transformers.masking
import math 
from math import sqrt
import numpy as np
from einops.layers.torch import Rearrange

from utils.attention import AttentionConv1DAutoLayer, AttentionConvTranspose1DAutoLayer, AttentionConv1DLayer, AttentionLinearLayer

class TransformerAutoEncoder(nn.Module): 
    def __init__(self, extractor):
        super(TransformerAutoEncoder, self).__init__()
        hidden_size = 512
        cho = 8
        
        self.encoder = []
        self.decoder = []
        self.upsample = []
        self.downsample = []

        self.crop_size = hidden_size
        self.depth = 4


        self.DePreNet_ = nn.Sequential(
            Rearrange('b c w h -> b w (c h)'),
            nn.Linear(2049*2, hidden_size),

            Rearrange('b w h -> b h w'),
            nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=1, stride=1, padding=0, groups=1),
            nn.BatchNorm1d(hidden_size),
            nn.SiLU(inplace=True),
            nn.Dropout(p=0.2),

            # Rearrange('b w h -> b h w'),
            nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=1, stride=1, padding=0, groups=1),
            nn.BatchNorm1d(hidden_size),
            nn.SiLU(inplace=True),
            Rearrange('b h w -> b w h'),
            nn.Dropout(p=0.2),

        )

        self.PosNet = nn.Sequential(
            Rearrange('b w h -> b h w'),
            nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=1, stride=1, padding=0, groups=1),
            nn.BatchNorm1d(hidden_size),
            Rearrange('b h w -> b w h'),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_size,hidden_size),
            
            Rearrange('b w h -> b h w'),
            nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=1, stride=1, padding=0, groups=1),
            Rearrange('b h w -> b w h'),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_size,2049*2),
            nn.ReLU(),
            Rearrange('b w (h c) -> b c h w',c=2),

            
        )

        for i in range(self.depth):

            en_d_model = hidden_size # // (2**i)
            en_out_d_model = hidden_size # // (2**(i+1))
            de_d_model = hidden_size # // (2**(self.depth-i))
            de_out_d_model = hidden_size # // (2**(self.depth-i-1))

            self.encoder.append(
                TransformerEncoder(
                    [ TransformerEncoderLayer(
                        attention = AttentionConv1DAutoLayer(FullAttention(), d_model=en_d_model, out_d_model=en_out_d_model, n_heads=4, d_keys=en_d_model),
                        d_model = en_d_model,
                        out_d_model= en_out_d_model,
                        activation="silu",
                        d_ff = 4,
                        dropout=0.1,
                    )],
                    norm_layer= None
                )
            )
            self.decoder.append(
                TransformerDecoder(
                    [ TransformerDecoderLayer(
                        self_attention = AttentionConv1DLayer(FullAttention(), d_model=de_d_model, n_heads=4, d_keys=de_d_model),
                        cross_attention= AttentionConvTranspose1DAutoLayer(FullAttention(), d_model=de_d_model, out_d_model = de_out_d_model, n_heads=4, d_keys=de_d_model),
                        d_model = de_d_model,
                        out_d_model = de_out_d_model,
                        activation="silu",
                        d_ff = 4,
                        dropout=0.1,
                    )],
                    norm_layer= None
                ),

            )

            self.downsample.append(
                nn.Sequential(
                    Rearrange('b w h -> b h w'),
                    nn.BatchNorm1d(en_out_d_model),
                    Rearrange('b h w -> b w h'),
                    nn.Linear(en_out_d_model, en_out_d_model),
                    nn.SiLU(inplace=True),

                    nn.Dropout(0.2),
                )
                
            )

            self.upsample.append(
                nn.Sequential(
                    Rearrange('b w h -> b h w'),
                    nn.BatchNorm1d(de_out_d_model),
                    Rearrange('b h w -> b w h'),
                    nn.Linear(de_out_d_model, de_out_d_model),
                    nn.SiLU(inplace=True),

                    nn.Dropout(0.2),
                )
            )
            

            
        
        self.encoder = nn.ModuleList(self.encoder)
        self.decoder = nn.ModuleList(self.decoder)
        self.upsample = nn.ModuleList(self.upsample)
        self.downsample = nn.ModuleList(self.downsample)


        self.pos_encoding1 = PositionalEncoding(d_model=hidden_size, max_len=350)
        self.extractor = extractor

    def forward(self, s):
        """
        ARGS: 
            s: input spectrogram of shape
                (nb_samples, nb_channels, nb_bins, nb_frames)
            s: input Mel-spectrogram of shape
                (nb_samples, nb_channels, nb_bins, nb_frames)
        Return:
            (nb_samples, nb_channels, nb_bins, nb_frames)
        """
        mix = s.detach().clone()
        B, C, H, W = s.size()
        
        
        s = self.DePreNet_(s.permute(0,1,3,2))
        s = self.pos_encoding1(s)
        skip_connections = []
        for idx, module in enumerate(self.encoder):
            s = module(s)
            skip_connections.append(s)
            # s = self.downsample[idx](s)
                
        
        for idx, module in enumerate(self.decoder):
            memory = skip_connections.pop()
            s = module(s, memory)
            # s = self.upsample[idx](s)
            
            # result = result + s


        s = self.PosNet(s)  #B, C, H, W
        
        
        
        return  mix*s, s

            
         
            #     x = torch.cat((x,res),-1)
       

        
           #B, C, H, W 
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

    
class Feature_Extractor(nn.Module):
    def __init__(self, rate, n_fft, hop_size, mel_bins):
        super(Feature_Extractor, self).__init__()
        self.spectrogram_extractor = Spectrogram(n_fft=n_fft, hop_length=hop_size, 
            win_length=n_fft, window='hann', center=True, pad_mode='reflect', 
            freeze_parameters=True)

        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(sr=rate, n_fft=n_fft, 
            n_mels=mel_bins, fmin=20, fmax=14000, ref=1.0, amin=1e-10, top_db=None, 
            freeze_parameters=True)

    def forward(self, wav):
        if len(wav.size()) == 3:
            #  [B, L ,2]
            wav = wav.mean(-1)
        S = self.spectrogram_extractor(wav)
        M = self.logmel_extractor(S)

        return M.squeeze()


def diagonalMask_generator(L, l, device):
    mask = torch.zeros(L,L)
    for i in range(L-l+1):
        mask[i:i+l,i:i+l] = 1
    return mask.bool().to(device)


# augment shift
# post Net hidden size & relu
# Transformer activation -> relu
# head -> 1