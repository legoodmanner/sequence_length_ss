import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torchlibrosa.stft import Spectrogram

class AdaINLinear(nn.Module):
    def __init__(self, in_feature, latent_size):
        super(AdaINLinear, self).__init__()
        self.fc = nn.Linear(in_feature, in_feature)
        self.bn = nn.BatchNorm1d(in_feature, affine=False)
        self.latent_mu_fc = nn.Linear(latent_size, in_feature, bias=True)
        self.latent_bias_fc = nn.Linear(latent_size, in_feature, bias=True)
        #self.res_scalar = nn.Parameter(torch.rand(in_feature))
        self.drop1 = nn.Dropout(p=0.2, inplace=False)
        self.drop2 = nn.Dropout(p=0.2, inplace=False)
    def forward(self, input):
        x, latent = input
        res = x
        nb_frames, nb_samples, hidden_size = x.size()
        x = self.fc(x.reshape(-1, x.shape[-1]))
        x = self.bn(x)
        x = self.drop1(x)
        x = x.reshape(nb_frames, nb_samples, hidden_size)

        scaler = self.latent_mu_fc(latent).expand(x.size())
        bias = self.latent_bias_fc(latent).expand(x.size())
        x = x * scaler + bias
        x = x + res 
        x = F.relu(x)
        x = self.drop2(x)
        return [x, latent]

class OpenUnmix(nn.Module):
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
        nb_channels=2,
        hidden_size=512,
        nb_layers=3,
        unidirectional=False,
        input_mean=None,
        input_scale=None,
        max_bin=None,
        latent_size = 1024, 

        window_size = 4096,
        hop_size = 1024,
        rate = 32000.0,
        adain_layer = 4

    ):
        super(OpenUnmix, self).__init__()
        
        window = 'hann'
        center = True
        pad_mode = 'reflect'
        nb_bins = window_size//2+1
        self.window_size = window_size
        self.hop_size = hop_size
        
        # spectrogram crop
        max_bin = self.bandwidth_to_max_bin(rate=rate, n_fft=window_size, bandwidth=16000)
        self.nb_output_bins = nb_bins
        if max_bin:
            self.nb_bins = max_bin
        else:
            self.nb_bins = self.nb_output_bins
        # ========================================
        # hidden size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        
        # ========================================
        # latent vecotr -2048 -> hidden_size
        self.fc1 = nn.Linear(self.nb_bins* nb_channels, hidden_size)
        self.latent_mu_fc1 = nn.Linear(self.latent_size, hidden_size, bias=False)
        self.latent_bias_fc1 = nn.Linear(self.latent_size, hidden_size, bias=False)
        
        # ==================================================
        self.bn1 = nn.BatchNorm1d(hidden_size, affine=False)
        
        self.adain_in = nn.Sequential(*([AdaINLinear(hidden_size, latent_size)]*adain_layer))
        
        if unidirectional:
            lstm_hidden_size = hidden_size
        else:
            lstm_hidden_size = hidden_size // 2

        self.lstm = nn.LSTM(
            input_size=hidden_size + latent_size,
            hidden_size=lstm_hidden_size,
            num_layers=nb_layers,
            bidirectional=not unidirectional,
            batch_first=False,
            dropout=0.4 if nb_layers > 1 else 0,
            
        )

        fc2_hiddensize = hidden_size * 2
        self.fc2 = nn.Linear(in_features=fc2_hiddensize, out_features=hidden_size, bias=False)
        self.latent_mu_fc2 = nn.Linear(self.latent_size, hidden_size, bias=False)
        self.latent_bias_fc2 = nn.Linear(self.latent_size, hidden_size, bias=False)
        self.bn2 = nn.BatchNorm1d(hidden_size, affine=False)

        self.adain_out = nn.Sequential(*([AdaINLinear(hidden_size, latent_size)]*adain_layer))
        
        self.fc3 = nn.Linear(
            in_features=hidden_size,
            out_features=self.nb_output_bins * nb_channels,
            bias=False,
        )

        self.bn3 = nn.BatchNorm1d(self.nb_output_bins * nb_channels, affine=False)

        if input_mean is not None:
            input_mean = torch.from_numpy(-input_mean[: self.nb_bins]).float()
        else:
            input_mean = torch.zeros(self.nb_bins)

        if input_scale is not None:
            input_scale = torch.from_numpy(1.0 / input_scale[: self.nb_bins]).float()
        else:
            input_scale = torch.ones(self.nb_bins)

        self.input_mean = nn.parameter.Parameter(input_mean)
        self.input_scale = nn.parameter.Parameter(input_scale)

        self.output_scale = nn.parameter.Parameter(torch.ones(self.nb_output_bins).float())
        self.output_mean = nn.parameter.Parameter(torch.ones(self.nb_output_bins).float())
         
        
    def freeze(self):
        # set all parameters as not requiring gradient, more RAM-efficient
        # at test time
        for p in self.parameters():
            p.requires_grad = False
        self.eval()
    
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
    

    def forward(self, x, latent):
        
        """
        Args:
            x: input spectrogram of shape
                `(nb_samples, nb_channels, nb_bins, nb_frames)`
                
            latent: query embedding vector from pre-trained encoder
                `(nb_samples, dimension)`
        Returns:
            Tensor: filtered spectrogram of shape
                `(nb_samples, nb_channels, nb_bins, nb_frames)`
        """
        # permute so that batch is last for lstm
        x = x.permute(3, 0, 1, 2)
        # get current spectrogram shape
        nb_frames, nb_samples, nb_channels, nb_bins = x.data.shape
        
        # crop
        x = x[..., : self.nb_bins]

        # shift and scale input to mean=0 std=1 (across all bins)
        x = x.reshape(nb_frames * nb_samples, nb_channels * self.nb_bins)
        x = self.fc1(x)
    
        # norm and affine
        x = self.bn1(x)
        x = x.reshape(nb_frames, nb_samples, self.hidden_size)
        # Affine:
        # [nb_sample, hidden_size] -> [nb_frames, nb_sameples, hidden_size]
        scaler1 = self.latent_mu_fc1(latent).expand(x.size())
        bias1 = self.latent_bias_fc1(latent).expand(x.size())
        x = x * scaler1 + bias1
        
        #v = torch.tanh(v_x)
        
        #######========================
        # normalize every instance in a batch 
        
        #x = x.reshape(nb_frames, nb_samples, 2, self.hidden_size)
        # squash range ot [-1, 1]
        
        #######========================
        
        # Query & Vector linear mapping
        _x = [x, latent]
        x, _ = self.adain_in(_x)
        
        # concat
        x = torch.cat([x, latent.expand(x.size(0), x.size(1), -1)], dim=-1)
        #x = torch.tanh(x) ###################
        # apply 3-layers of stacked LSTM
        lstm_out, _,  = self.lstm(x)

        # lstm skip connection
        x = torch.cat([x[...,:self.hidden_size], lstm_out], -1)

        # first dense stage + batch norm
        x = self.fc2(x.reshape(-1, x.shape[-1]))
        x = self.bn2(x)
        x = x.reshape(nb_frames, nb_samples, self.hidden_size)

        scaler2 = self.latent_mu_fc2(latent).expand(x.size())
        bias2 = self.latent_bias_fc2(latent).expand(x.size())
        x = x * scaler2 + bias2
        x = F.relu(x)

        # second dense stage + layer norm
        
        _x = [x, latent]
        x, _ = self.adain_in(_x)
    
        x = self.fc3(x.reshape(-1, x.shape[-1]))
        x = self.bn3(x)

        # reshape back to original dim
        x = x.reshape(nb_frames, nb_samples, nb_channels, self.nb_output_bins)

        # apply output scaling
        x *= self.output_scale
        x += self.output_mean

        # since our output is non-negative, we can apply RELU
        x = torch.sigmoid(x) ################
        
        # permute back to (nb_samples, nb_channels, nb_bins, nb_frames)
        return x.permute(1, 2, 3, 0)