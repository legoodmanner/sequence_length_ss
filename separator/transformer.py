import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

from torchlibrosa.stft import Spectrogram, LogmelFilterBank
from torchlibrosa.augmentation import SpecAugmentation
from fast_transformers.attention import AttentionLayer, LinearAttention, FullAttention
import fast_transformers.masking
import math 
from math import sqrt
import numpy as np
from einops.layers.torch import Rearrange

from utils.transformers import TransformerEncoder, TransformerEncoderLayer, TransformerDecoderLayer, TransformerDecoder
from utils.attention import AttentionConv1D4Layer, AttentionConv1DLayer

class TransformerEncoderContainer(nn.Module): 
    def __init__(self):
        super(TransformerEncoderContainer, self).__init__()
        hidden_size = 512
        self.layer = 8
        # self.projIn = nn.Linear(1000*2, hidden_size, bias=False)
        # self.bn1 = nn.BatchNorm2d(hidden_size//128, affine=True)
        # self.outnorm = nn.LayerNorm((1292,2049*2))
        # self.bn2 = nn.BatchNorm1d(hidden_size)
        
        
        # self.bn3 = nn.BatchNorm1d(2049*2)
        
        self.encoder = []

        self.nb_bins = self.bandwidth_to_max_bin(rate=44100, n_fft=4096, bandwidth=16000)
        self.crop_size = hidden_size

        self.DePreNet_ = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=1, kernel_size=3, padding=1),
            Rearrange('b c w h -> b w (c h)'),
            nn.Linear(self.nb_bins, hidden_size),
            # nn.Dropout(p=0.5),

            Rearrange('b w h -> b h w'),
            nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=3, stride=1, padding=1, groups=hidden_size),
            nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=1, stride=1, padding=0, groups=1),
            nn.ReLU(inplace=True),

            Rearrange('b h w -> b w h'),
            
        )

        self.PosNet = nn.Sequential(
            Rearrange('b w h -> b h w'),

            nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=1, stride=1, padding=0, groups=1),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),

            nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=3, stride=1, padding=1, groups=hidden_size),
            nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=1, stride=1, padding=0, groups=1),
            nn.ReLU(inplace=True),
          
            nn.Conv1d(in_channels=hidden_size, out_channels=2049, kernel_size=1, stride=1, padding=0, groups=1),
            nn.ReLU(),
            Rearrange('b (h c) w -> b c h w',c=1),
            nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, padding=1),
            nn.ReLU(),
            # Rearrange('(b s) c h w -> b s c h w',s=4),

            
        )

        for i in range(self.layer):
            self.encoder.append(TransformerEncoder(
                        [ TransformerEncoderLayer(
                            attention = AttentionConv1DLayer(
                                FullAttention(), 
                                d_model=hidden_size,
                                n_heads=4,
                                # d_keys=hidden_size,
                                # d_values=hidden_size,
                                layer = 1,
                                ),
                            d_model = hidden_size,
                            activation="relu",
                            d_ff = 4,
                            dropout=0.0,
                            
                        )],
                        norm_layer= None
                    )
                )
            

            
        
        self.encoder = nn.ModuleList(self.encoder)

        self.softmax_temp = nn.Parameter(torch.ones(1).float())


        # self.pos_encoding1 = PositionalEncoding(d_model=hidden_size, max_len=3600)

    def forward(self, s, normalize=False):
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
        
        
        """ x = self.EnPreNet(x)
        x = x.permute(0,3,2,1) # B, W, H, C

        x = x.reshape(B,W,512)
        # x = x.squeeze() # B, W, H


        # x = torch.tanh(x)

        x = self.pos_encoding1(x)
        x = self.encoder(x) """
        s = s.permute(0,1,3,2)
        s = s[..., : self.nb_bins]
        s = self.DePreNet_(s)
        # result = torch.zeros_like(s, device=s.device)
        for idx, module in enumerate(self.encoder):
            s = module(s, softmax_temp=self.softmax_temp, source_indice=None) + s
        s = self.PosNet(s)  #B, C, H, W

        return  mix*s, s


    def bandwidth_to_max_bin(self, rate: float, n_fft: int, bandwidth: float) -> np.ndarray:
        """Convert bandwidth to maximum bin count
        Assuming lapped transforms such as STFT
        Args:
            rate (int): Sample rate
            n_fft (int): FFT length
            bandwidth (float): Target bandwidth in Hz
        Returns:
            np.ndarray: maximum frequency bin
        """
        freqs = np.linspace(0, rate / 2, n_fft // 2 + 1, endpoint=True)

        return np.max(np.where(freqs <= bandwidth)[0]) + 1

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
