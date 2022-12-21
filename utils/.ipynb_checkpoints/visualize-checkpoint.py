import torch
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
import torchaudio



def figGen(spec, config, mask=False):
        
        if isinstance(spec, np.ndarray):
            b, c, h, w = spec.shape
        elif isinstance(spec, torch.Tensor):
            b, c, h, w = spec.size()
            spec = spec.detach().cpu().numpy()
        else:
            raise NotImplementedError()
            
        b = min(b,4)
        
        fig, ax = plt.subplots(nrows=b, ncols=1, sharex=True)
        spec = spec[:,0,...] + 1j*spec[:,1,...]
        if b == 1:
            librosa.display.specshow(
                librosa.amplitude_to_db(np.abs(spec[0]), ref=np.max), 
                y_axis='log', 
                x_axis='time', 
                ax=ax,
                sr=config.rate,
                hop_length=config.hop_length
            )
        
        else:
            for i in range(b):
                if not mask:
                    s = librosa.amplitude_to_db(np.abs(spec[i]), ref=np.max)
                else:
                    s = spec[i]
                librosa.display.specshow(
                    s,
                    y_axis='log', 
                    x_axis='time', 
                    ax=ax[i],
                    sr=config.rate,
                    hop_length=config.hop_length
                )
        return fig
    
import librosa  
class AudioGen():
    def __init__(self, config):
        # "Input: spec -> [batch, bins, lens, 2]"
        self.config = config
    def __call__(self, X, return_type='tensor'):
        if len(X.size()) == 4:
            if X.size(1) == 2:
                # (B, C, H, W) -> (B, H, W, C)
                X = X.permute(0,2,3,1)
            X = X.detach().cpu().numpy()
            X = X[...,0] + 1j*X[...,1]
            X = np.array([librosa.istft(x, **self.config) for x in X])
            X += 0.001
        else:
            if X.size(0) == 2:
                X = X.permute(1,2,0)
            X = X.detach().cpu().numpy()
            X = X[...,0] + 1j*X[...,1]
            X = librosa.istft(X, **self.config)
        
        if return_type == 'tensor':
            return torch.Tensor(X)
        else:
            return X