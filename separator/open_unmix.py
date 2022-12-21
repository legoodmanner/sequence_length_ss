from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import LSTM, BatchNorm1d, Linear, Parameter
import math
from fast_transformers.transformers import TransformerEncoder, TransformerEncoderLayer, TransformerDecoderLayer, TransformerDecoder
from fast_transformers.attention import AttentionLayer, LinearAttention, FullAttention
from utils.attention import AttentionConv1D4Layer, AttentionConv1DLayer

class Vanilla_Transformer(nn.Module):
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
        n_fft=4096,
        nb_bins=2049,
        nb_channels=2,
        hidden_size=512,
        nb_layers=3,
        unidirectional=False,
        input_mean=None,
        input_scale=None,
        max_bin=None,
    ):
        super(Vanilla_Transformer, self).__init__()
        self.nb_output_bins = nb_bins

        if max_bin:
            self.nb_bins = max_bin
        else:
            self.nb_bins = self.bandwidth_to_max_bin(rate=44100, n_fft=n_fft, bandwidth=16000)
            print(f'nb_bins: {self.nb_bins}')
        self.hidden_size = hidden_size

        self.fc1 = Linear(self.nb_bins * nb_channels, hidden_size, bias=False)

        self.bn1 = BatchNorm1d(hidden_size)

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

        self.fc2 = Linear(in_features=hidden_size*2, out_features=hidden_size, bias=False)

        self.bn2 = BatchNorm1d(hidden_size)

        self.fc3 = Linear(
            in_features=hidden_size,
            out_features=self.nb_output_bins * nb_channels,
            bias=False,
        )

        self.bn3 = BatchNorm1d(self.nb_output_bins * nb_channels)

        if input_mean is not None:
            input_mean = torch.from_numpy(-input_mean[: self.nb_bins]).float()
        else:
            input_mean = torch.zeros(self.nb_bins)

        if input_scale is not None:
            input_scale = torch.from_numpy(1.0 / input_scale[: self.nb_bins]).float()
        else:
            input_scale = torch.ones(self.nb_bins)

        self.input_mean = Parameter(input_mean)
        self.input_scale = Parameter(input_scale)

        self.output_scale = Parameter(torch.ones(self.nb_output_bins).float())
        self.output_mean = Parameter(torch.ones(self.nb_output_bins).float())
    def freeze(self):
        # set all parameters as not requiring gradient, more RAM-efficient
        # at test time
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    def forward(self, x: Tensor) -> Tensor:
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