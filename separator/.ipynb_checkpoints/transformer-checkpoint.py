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
import numpy as np
from einops.layers.torch import Rearrange

from utils.attention import AttentionConv2DLayer, AttentionConv1DLayer

class TransformerEncoderContainer(nn.Module): 
    def __init__(self, extractor):
        super(TransformerEncoderContainer, self).__init__()
        hidden_size = 512
        cho = 8
        # self.projIn = nn.Linear(1000*2, hidden_size, bias=False)
        # self.bn1 = nn.BatchNorm2d(hidden_size//128, affine=True)
        # self.outnorm = nn.LayerNorm((1292,2049*2))
        # self.bn2 = nn.BatchNorm1d(hidden_size)
        
        self.projOut = nn.Linear(hidden_size, 2049, bias=True)
        # self.bn3 = nn.BatchNorm1d(2049*2)
        
        self.encoder = []
        self.decoder = []
        self.upsample = []
        self.downsample = []

        self.crop_size = hidden_size
        self.n_layer = 6
        # self.salinence_fc = nn.Linear(hidden_size//(4**(self.n_layer-1)),1, bias=False)
        """ self.EnPreNet = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=cho, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(cho),
            nn.LeakyReLU(inplace=True),
            # nn.Dropout(p=0.5),

            nn.Conv2d(in_channels=cho, out_channels=cho, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(cho),
            nn.LeakyReLU(inplace=True),
            # nn.Dropout(p=0.5),

            nn.Conv2d(in_channels=cho, out_channels=2, kernel_size=5, stride=1, padding=2),
            nn.Dropout(p=0.3)
        ) """
        """ self.DePreNet = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=cho, kernel_size=(5,5), stride=1, padding=(2,2)),
            nn.BatchNorm2d(cho),
            nn.LeakyReLU(inplace=True),
            # nn.Dropout(p=0.5),

            nn.Conv2d(in_channels=cho, out_channels=cho*2, kernel_size=(5,5), stride=1, padding=(2,2)),
            nn.BatchNorm2d(cho*2),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(p=0.5),

            nn.Conv2d(in_channels=cho*2, out_channels=cho, kernel_size=(5,5), stride=1, padding=(2,2)),
            nn.BatchNorm2d(cho),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(in_channels=cho, out_channels=1, kernel_size=(5,5), stride=1, padding=(2,2)),
            nn.Linear(2049, hidden_size, bias=True),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(p=0.5),
        ) """
        self.DePreNet_ = nn.Sequential(
            Rearrange('b c w h -> b w (c h)'),
            nn.Linear(2049*2, hidden_size),

            Rearrange('b w h -> b h w'),
            nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=1, stride=1, padding=0, groups=hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.SiLU(inplace=True),
            Rearrange('b h w -> b w h'),
            nn.Linear(hidden_size, hidden_size),
            nn.Dropout(p=0.2),

            Rearrange('b w h -> b h w'),
            nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=1, stride=1, padding=0, groups=hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.SiLU(inplace=True),
            Rearrange('b h w -> b w h'),
            nn.Linear(hidden_size, hidden_size),
            nn.Dropout(p=0.2),

        )

        self.PosNet = nn.Sequential(
            Rearrange('b w h -> b h w'),
            nn.Conv1d(in_channels=2049, out_channels=2049, kernel_size=1, stride=1, padding=0, groups=2049),
            nn.BatchNorm1d(2049),
            Rearrange('b h w -> b w h'),
            nn.SiLU(inplace=True),
            nn.Linear(2049,2049),
            
            Rearrange('b w h -> b h w'),
            nn.Conv1d(in_channels=2049, out_channels=2049, kernel_size=1, stride=1, padding=0, groups=2049),
            nn.BatchNorm1d(2049),
            Rearrange('b h w -> b w h'),
            nn.SiLU(inplace=True),
            nn.Linear(2049,2049*2),
            nn.Sigmoid(),
            Rearrange('b w (h c) -> b c h w',c=2),

            
        )
 
        """ self.encoder =  nn.Sequential(
            TransformerEncoder(
                [ TransformerEncoderLayer(
                        # attention = AttentionLayer(LinearAttention(query_dimensions=hidden_size//(2**(i+1))), d_model=hidden_size//(2**i), n_heads=64),
                        # d_model = hidden_size//(2**i),
                        attention = AttentionConv1DLayer(FullAttention(), d_model=hidden_size, n_heads=4,),
                        d_model = hidden_size,
                        activation="gelu",
                        dropout=0.3,
                        d_ff = 2,
                    ) 
                ] *  self.n_layer,
                norm_layer=None
            ),
            Rearrange('B L D -> B 1 L D'),
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=5, stride=(4,1), padding=2),
            Rearrange('B 1 L D -> B L D')
        )
            
        self.decoder =  TransformerDecoder(
            [ TransformerDecoderLayer(
                    # self_attention = AttentionLayer(LinearAttention(query_dimensions=hidden_size//(2**(self.n_layer-i))), d_model=hidden_size//(2**(self.n_layer-i-1)), n_heads=64),
                    # cross_attention = AttentionLayer(LinearAttention(query_dimensions=hidden_size//(2**(self.n_layer-i))), d_model=hidden_size//(2**(self.n_layer-i-1)), n_heads=64),
                    # d_model = hidden_size//(2**(self.n_layer-i-1)),
                    self_attention = AttentionConv2DLayer(FullAttention(), d_model=hidden_size, n_heads=3),
                    cross_attention = AttentionLayer(FullAttention(), d_model=hidden_size, n_heads=4),
                    d_model = hidden_size,
                    activation="gelu",
                    d_ff = 2,
                    dropout=0.3,
                ) 
            for i in range(self.n_layer)],
            norm_layer= None) """

        for i in range(self.n_layer//2):

            en_d_model = hidden_size//(1**i)
            en_out_d_model = hidden_size//(1**(i+1)) 
            de_d_model = hidden_size//(1**(self.n_layer//2-i)) 
            de_out_d_model = hidden_size//(1**(self.n_layer//2-i-1))

            self.encoder.append(
                TransformerEncoder(
                    [ TransformerEncoderLayer(
                        # self_attention = AttentionLayer(LinearAttention(query_dimensions=hidden_size//(2**(self.n_layer-i))), d_model=hidden_size//(2**(self.n_layer-i-1)), n_heads=64),
                        # cross_attention = AttentionLayer(LinearAttention(query_dimensions=hidden_size//(2**(self.n_layer-i))), d_model=hidden_size//(2**(self.n_layer-i-1)), n_heads=64),
                        # d_model = hidden_size//(2**(self.n_layer-i-1)),
                        attention = AttentionConv1DLayer(FullAttention(), d_model=en_d_model, n_heads=4, d_keys=en_d_model, dilation_exp=i),
                        #cross_attention = AttentionConvLayer(FullAttention(), d_model=hidden_size, n_heads=4),
                        d_model = en_d_model,
                        activation="gelu",
                        d_ff = 4,
                        dropout=0.1,
                    )],
                    norm_layer= None
                )
            )

            self.decoder.append(
                TransformerEncoder(
                    [ TransformerEncoderLayer(
                        # self_attention = AttentionLayer(LinearAttention(query_dimensions=hidden_size//(2**(self.n_layer-i))), d_model=hidden_size//(2**(self.n_layer-i-1)), n_heads=64),
                        # cross_attention = AttentionLayer(LinearAttention(query_dimensions=hidden_size//(2**(self.n_layer-i))), d_model=hidden_size//(2**(self.n_layer-i-1)), n_heads=64),
                        # d_model = hidden_size//(2**(self.n_layer-i-1)),
                        attention = AttentionConv1DLayer(FullAttention(), d_model=de_out_d_model, n_heads=4, d_keys=de_out_d_model, dilation_exp=i),

                        #cross_attention = AttentionConvLayer(FullAttention(), d_model=hidden_size, n_heads=4),
                        d_model = de_out_d_model,
                        activation="gelu",
                        d_ff = 4,
                        dropout=0.1,
                    )],
                    norm_layer= None
                ),

            )

            self.downsample.append(
                nn.Sequential(
                    nn.Linear(en_d_model, en_d_model),
                    Rearrange('b w h -> b h w'),
                    nn.GLU(dim=1),
                    nn.Conv1d(en_d_model//2, en_out_d_model, 1, 1, 0),
                    nn.BatchNorm1d(en_out_d_model),
                    nn.Conv1d(en_out_d_model, en_out_d_model, 1),
                    Rearrange('b h w -> b w h'),
                    nn.Dropout(0.2),
                )
                
            )

            self.upsample.append(
                nn.Sequential(
                    nn.Linear(de_d_model, de_d_model*2),
                    Rearrange('b w h -> b h w'),
                    nn.GLU(dim=1),
                    nn.Conv1d(de_d_model, de_out_d_model, 1, 1, 0),
                    nn.BatchNorm1d(de_out_d_model),
                    nn.Conv1d(de_out_d_model, de_out_d_model, 1),
                    Rearrange('b h w -> b w h'),
                    nn.Dropout(0.2),
                )
            )
            

            
        
        self.encoder = nn.ModuleList(self.encoder)
        self.decoder = nn.ModuleList(self.decoder)
        self.upsample = nn.ModuleList(self.upsample)
        self.downsample = nn.ModuleList(self.downsample)


        self.pos_encoding1 = PositionalEncoding(d_model=hidden_size, max_len=300)
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
        
        
        """ x = self.EnPreNet(x)
        x = x.permute(0,3,2,1) # B, W, H, C

        x = x.reshape(B,W,512)
        # x = x.squeeze() # B, W, H


        # x = torch.tanh(x)

        x = self.pos_encoding1(x)
        x = self.encoder(x) """
        
        


        s = self.DePreNet_(s.permute(0,1,3,2))
        s = self.pos_encoding1(s)
        #s = torch.cat((torch.zeros((B, 1, 1024), device=s.device), s), dim=1)
        skip_connections = []
        result = torch.zeros_like(s, device=s.device)
        for idx, module in enumerate(self.encoder):
            s = module(s)
            
            if idx != self.n_layer//2 -1:
                s = self.downsample[idx](s) + s
                #skip_connections.append(s)
            result = result + s
        
        for idx, module in enumerate(self.decoder):
            
            if idx != 0:
                
                #res = skip_connections.pop()
                #s = torch.cat((s,res),dim=-1)
                s = self.upsample[idx](s) + s
            s = module(s)
            result = result + s


        s = self.projOut(s) 
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
# LayerNorm2d -> LayerNorm1d
# no skip connect
# dropout