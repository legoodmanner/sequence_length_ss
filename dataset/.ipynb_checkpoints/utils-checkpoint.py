import torch
from torchlibrosa.stft import Spectrogram


def move_data_to_device(x, device):
    if 'float64' in str(x.dtype):
        x = x.float()
    elif 'float' in str(x.dtype):
        x = torch.Tensor(x)
    elif 'int' in str(x.dtype):
        x = torch.LongTensor(x)
    else:
        return x

    return x.to(device)

class Spectrogram_Extractor():
    def __init__(
                self, 
                 window_size = 4096,
                 hop_size = 1024,
                 window = 'hann',
                 center = True,
                 pad_mode = 'reflect'
                ):
        
        self.extractor = Spectrogram(n_fft=window_size, hop_length=hop_size, 
            win_length=window_size, window=window, center=center, pad_mode=pad_mode, 
            freeze_parameters=True)
    
    def __call__(self, input):
        return self.extractor(input)
    
class STFT():
    def __init__(self, cuda, **kwargs):
        """
        kwargs:  n_fft, hop_length, window, center, cuda
        """
        self.kwargs = kwargs
        self.cuda = cuda

        if 'n_fft' not in self.kwargs:
            self.n_fft = 1024
        else:
            self.n_fft = self.kwargs['n_fft']

    def __call__(self, y):
        """
        Args: [B, time]
        Out: [B, 2, Freq_bin, frame]
        """
        

        s = torch.stft(
                y,
                **self.kwargs,
                return_complex = False
                
            )
        s = s.permute(0, 3, 1, 2)
        s = torch.where(torch.isnan(s), torch.full_like(s, 0), s)
        return s

class ToTensor(torch.nn.Module):
    def __init__(self,):
        super(ToTensor, self).__init__()
    
    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.Tensor(x)
        return x
        