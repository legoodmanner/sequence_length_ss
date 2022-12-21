import torch
import torch.nn as nn
import torch.nn.functional as F
from torchlibrosa.stft import Spectrogram, LogmelFilterBank

class Query_Encoder(nn.Module):
    def __init__(self, config):
        super(Query_Encoder, self).__init__()
        self.kernel_size = config.kernel_size
        self.stride = config.stride
        self.convs =  []
        for i in range(len(self.kernel_size)):
            self.convs.append(
                nn.Sequential(
                    nn.Conv2d(config.ch_in[i], config.ch_out[i], self.kernel_size[i], self.stride[i]),
                    nn.InstanceNorm2d(config.ch_out[i]),
                    nn.ReLU(),
                    nn.Dropout(0.2)
                )
            )
        self.convs = nn.Sequential(*self.convs)
        self.gru = nn.GRU(config.gru_input_size, config.hidden_size//2, batch_first=True, bidirectional=True,)
        #self.fc = nn.Linear(config.hidden_size, config.latent_size)
    def forward(self, x):
        """
        ARGS: x: [B, C, H, W]
        Return:
            latent: [B, latent_size]
            mux:
            varx: 
        """
        self.gru.flatten_parameters()
        x = self.convs(x)
        B, C, H, W = x.size()
        # (B, W, C*H)
        print(x.size())
        x = x.permute(0, 3, 1, 2).reshape(B, W, -1)
        # (B, W, hidden_size)
        x, _ = self.gru(x)
        #x = self.fc(x)
        #out = self.fc(x.reshape(B,-1))
        return x

class Post_Encoder(nn.Module):
    def __init__(self, config):
        super(Post_Encoder, self).__init__()
        hidden_size = config.latent_size
        self.fc1 = nn.Linear(config.latent_size*4, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 20)
    def forward(self, x):
        """
        input: x; shape:[B, 4, latentsize]
        """
        h = x.reshape(x.size(0), -1)
        h = F.leaky_relu(self.fc1(h))
        h = F.leaky_relu(self.fc2(h))
        h = F.leaky_relu(self.fc3(h))
        return h

class CNN_Encoder(nn.Module):
    def __init__(self, config):
        super(CNN_Encoder, self).__init__()
        hidden_size = config.latent_size
        self.conv1 = nn.Conv2d(1,16, 3,(1,2), 1)
        self.conv2 = nn.Conv2d(16,64, 3,(1,2), 1)
        self.conv3 = nn.Conv2d(64, 128, 3,(1,2), 1)
        self.fc = nn.Linear(128,20)
    def forward(self, x):
        if len(x.size()) == 3:
            #(B,H,W) --> (B,C,H,W)
            x = x.unsqueeze(1)
        x = F.leaky_relu(self.conv1(x))
        x = F.max_pool2d(x,2,2)
        #x = F.dropout(x, 0.3)
        x = F.leaky_relu(self.conv2(x))
        x = F.max_pool2d(x,2,2)
        x = F.dropout(x, 0.3)
        x = F.leaky_relu(self.conv3(x))
        x = F.max_pool2d(x,2,2)
        x = F.dropout(x, 0.3)
        x = torch.mean(x, dim=-1)
        x = self.fc(x.squeeze())

        return x
            
class VGGish(nn.Module):
    def __init__(self, n_fft, hop_size, sample_rate, mel_bins, fmin, fmax):
        super(VGGish, self).__init__()

        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None

        self.spectrogram_extractor = Spectrogram(n_fft=n_fft, hop_length=hop_size, 
            win_length=n_fft, window=window, center=center, pad_mode=pad_mode, 
            freeze_parameters=True)

        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(sr=sample_rate, n_fft=n_fft, 
            n_mels=mel_bins, fmin=fmin, fmax=fmax, ref=ref, amin=amin, top_db=top_db, 
            freeze_parameters=True)

            
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(128, 256, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(256, 512, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2)
        )
        self.fc = nn.Sequential(
            nn.Linear(512 * 16 * 4, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 128),
            
        )

    def forward(self, x):
        x = self.spectrogram_extractor(x)
        x = self.logmel_extractor(x)
        x = self.features(x).permute(0, 2, 3, 1).contiguous()
        # [B, 16, 4 , 512]
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x