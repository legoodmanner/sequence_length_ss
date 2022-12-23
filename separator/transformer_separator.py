import torch
import torch.nn as nn
import torch.nn.functional as F
from fast_transformers.builders import TransformerEncoderBuilder
from torchlibrosa.stft import Spectrogram, LogmelFilterBank
from torchlibrosa.augmentation import SpecAugmentation
from fast_transformers.transformers import TransformerEncoder, TransformerEncoderLayer, TransformerDecoderLayer, TransformerDecoder
from fast_transformers.attention import AttentionLayer, LinearAttention, FullAttention
import math 
from math import sqrt
from einops.layers.torch import Rearrange
from utils.lib import get_attn_func


class TransformerAutoEncoderSeparator(nn.Module): 
    
    def __init__(self, attention, hidden_size):
        super(TransformerAutoEncoderSeparator, self).__init__()
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
                        attention = get_attn_func(attention),
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
                        self_attention = get_attn_func(attention),
                        cross_attention= get_attn_func(attention),
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

class VanillaTransformerSeparator(nn.Module):
    """OpenUnmix Core spectrogram based separation module.
    Args:
        nb_bins (int): Number of input time-frequency bins (Default: `4096`).
        nb_channels (int): Number of input audio channels (Default: `2`).
        hidden_size (int): Size for bottleneck layers (Default: `512`).
        nb_layers (int): Number of Bi-LSTM layers (Default: `3`).
        unidirectional (bool): Use causal model useful for realtime purpose.
            (Default `False`)
        input_mean (ndarray or None): global data mean of shape `(nb_bins, )`.
            Defaults to zeros(nb_bins)
        input_scale (ndarray or None): global data mean of shape `(nb_bins, )`.
            Defaults to ones(nb_bins)
        max_bin (int or None): Internal frequency bin threshold to
            reduce high frequency content. Defaults to `None` which results
            in `nb_bins`
    """

    def __init__(
        self,
        attention,
        n_fft=4096,
        nb_bins=2049,
        nb_channels=2,
        hidden_size=512,
        input_mean=None,
        input_scale=None,
        max_bin=None,
    ):
        super(VanillaTransformerSeparator, self).__init__()
        self.nb_output_bins = nb_bins

        if max_bin:
            self.nb_bins = max_bin
        else:
            self.nb_bins = self.bandwidth_to_max_bin(rate=44100, n_fft=n_fft, bandwidth=16000)
            print(f'nb_bins: {self.nb_bins}')
        self.hidden_size = hidden_size

        self.fc1 = nn.Linear(self.nb_bins * nb_channels, hidden_size, bias=False)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.pos_encoding = PositionalEncoding(d_model=hidden_size, max_len=1800)
        self.lstm = []
        for i in range(3):
            self.lstm.append(TransformerEncoder(
                        [ TransformerEncoderLayer(
                            attention = AttentionLayer(
                                FullAttention(),
                                d_model=hidden_size,
                                n_heads=16,
                                ),
                            d_model = hidden_size,
                            activation="relu",
                            d_ff = 4,
                            dropout=0.2,
                            
                        )],
                        norm_layer= None
                    )
                )
        self.lstm = nn.ModuleList(self.lstm)            

        self.fc2 = nn.Linear(in_features=hidden_size*2, out_features=hidden_size, bias=False)

        self.bn2 = nn.BatchNorm1d(hidden_size)

        self.fc3 = nn.Linear(
            in_features=hidden_size,
            out_features=self.nb_output_bins * nb_channels,
            bias=False,
        )

        self.bn3 = nn.BatchNorm1d(self.nb_output_bins * nb_channels)
        if input_mean is not None:
            input_mean = torch.from_numpy(-input_mean[: self.nb_bins]).float()
        else:
            input_mean = torch.zeros(self.nb_bins)
        if input_scale is not None:
            input_scale = torch.from_numpy(1.0 / input_scale[: self.nb_bins]).float()
        else:
            input_scale = torch.ones(self.nb_bins)

        self.input_mean = nn.Parameter(input_mean)
        self.input_scale = nn.Parameter(input_scale)
        self.output_scale = nn.Parameter(torch.ones(self.nb_output_bins).float())
        self.output_mean = nn.Parameter(torch.ones(self.nb_output_bins).float())

    def freeze(self):
        # set all parameters as not requiring gradient, more RAM-efficient
        # at test time
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: input spectrogram of shape
                `(nb_samples, nb_channels, nb_bins, nb_frames)`
        Returns:
            Tensor: filtered spectrogram of shape
                `(nb_samples, nb_channels, nb_bins, nb_frames)`
        """
        # permute so that batch is last for lstm
        x = x.permute(3, 0, 1, 2)
        # get current spectrogram shape
        nb_frames, nb_samples, nb_channels, nb_bins = x.data.shape

        mix = x.detach().clone()

        # crop
        x = x[..., : self.nb_bins]
        # shift and scale input to mean=0 std=1 (across all bins)
        # x = x + self.input_mean
        # x = x * self.input_scale

        # to (nb_frames*nb_samples, nb_channels*nb_bins)
        # and encode to (nb_frames*nb_samples, hidden_size)
        x = self.fc1(x.reshape(-1, nb_channels * self.nb_bins))
        # normalize every instance in a batch
        x = self.bn1(x)
        x = x.reshape(nb_frames, nb_samples, self.hidden_size)
        # squash range ot [-1, 1]
        # x = y = torch.tanh(x)
        y = x.clone()

        # apply 3-layers of stacked LSTM
        x = x.permute(1,0,2)
        x = self.pos_encoding(x)
        for idx, module in enumerate(self.lstm):
            x = module(x) 

        x = x.permute(1,0,2)

        # lstm skip connection
        x = torch.cat([x, y], -1)

        # first dense stage + batch norm
        x = self.fc2(x.reshape(-1, x.shape[-1]))
        x = self.bn2(x)

        x = F.relu(x)

        # second dense stage + layer norm
        x = self.fc3(x)
        x = self.bn3(x)

        # reshape back to original dim
        x = x.reshape(nb_frames, nb_samples, nb_channels, self.nb_output_bins)

        # apply output scaling
        # x *= self.output_scale
        # x += self.output_mean

        # since our output is non-negative, we can apply RELU
        wav = F.relu(x) * mix
        # permute back to (nb_samples, nb_channels, nb_bins, nb_frames)
        return wav.permute(1, 2, 3, 0), F.relu(x).permute(1, 2, 3, 0)