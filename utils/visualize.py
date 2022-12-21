import torch
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
import torchaudio



def figGen(spec, config, mask=False,) :
        
        b, c, h, w = spec.shape

        if isinstance(spec, np.ndarray):
            pass
        elif isinstance(spec, torch.Tensor):
            spec = spec.detach().cpu().numpy()
        else:
            raise NotImplementedError()
            
        b = min(b,4)
        
        fig, ax = plt.subplots(nrows=b, ncols=1, sharex=True)
        if spec.shape[1] == 2:
            spec = spec[:,0,...] + 1j*spec[:,1,...]
        elif spec.shape[1] == 1:
            spec = spec[:,0]
        if b == 1:
            img = librosa.display.specshow(
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
                img = librosa.display.specshow(
                    s,
                    y_axis='log', 
                    x_axis='time', 
                    ax=ax[i],
                    sr=config.rate,
                    hop_length=config.hop_length
                )
               
        return fig


def melGen(spec, config, mask=False):
    if isinstance(spec, np.ndarray):
        pass
    elif isinstance(spec, torch.Tensor):
        spec = spec[:,0,...]
        spec = spec.detach().cpu().numpy()
    else:
        raise NotImplementedError()
    b = spec.shape[0]
    b = min(b,4)
    fig, ax = plt.subplots(nrows=b, ncols=1, sharex=True)
    for i in range(b):
    
        img = librosa.display.specshow(
            librosa.amplitude_to_db(np.abs(spec[i]), ref=np.max) if not mask else spec[i],
            y_axis='log', 
            x_axis='time', 
            ax=ax[i],
            sr=config.rate,
            hop_length=config.hop_length
        )
            
    return fig

def salGen(sal, config):
    sal = sal.detach().cpu().numpy()
    b = sal.shape[0]
    b = min(b,4)
    fig, ax = plt.subplots(nrows=b, ncols=1, sharex=True)
    for i in range(b):
        ax[i].imshow(sal[i], interpolation='nearest', aspect='auto')
    return fig
        

    
import librosa  
class AudioGen():
    def __init__(self, config):
        # "Input: spec -> [batch, bins, lens, 2]"
        self.config = config
    def __call__(self, X, return_type='tensor'):
        if len(X.size()) > 4:
            H, W, cmplx = X.shape[-3:]
            remain = X.shape[:-3]
            X = X.reshape(-1,H,W,cmplx)
            X = torch.istft(X, **self.config)
            #X = X.detach().cpu().numpy()
            #X = X[...,0] + 1j*X[...,1]
            #X = np.array([librosa.istft(x, **self.config) for x in X])
            X = X.reshape(remain+(-1,))
            # np.save('exemplify',X.detach().cpu().numpy())
            return  X# (B, nb_sample, channel)
        else:
            
            X = torch.istft(X, **self.config).mean(0)
            #X = X.detach().cpu().numpy()
            #X = X[...,0] + 1j*X[...,1]
            #X = librosa.istft(X, **self.config)
        
            return X