import torch
from torchaudio_augmentations import *
from dataset.utils import ToTensor
CUDA = 0
INST_INDICE = 3
instruments = ['drums', 'bass','other','vocals']
NAME = f'whateverfuckyou'
print(NAME)
########
class Open_Config:
    cuda = CUDA
    project = 'length'
    n_epoch = 10000
    parallel_cuda = [CUDA,2,3]
    
    #dataset/loader
    batch_size = 3
    num_workers = 2
    split_ratio = 0.2
    
    # opt
    lr = 0.0001 
    betas = (0.5, 0.999)
    lr_gamma = 0.3
    lr_step = 80
    weight_decay= 1e-05
    loss_coefficients = [1, 1, 0.5]
    
    # model
    latent_size = 128
    adain_layer = 2
    nb_layers = 3
    
    #STFT
    rate = 44100
    n_fft = 4096
    hop_length = 1024
    wiener_win_len = 1000
    
    instrument_indice = INST_INDICE
    instruments = ['drum', 'bass','other','vocal']
    summary_name = NAME
    

    stft_params = {
        'n_fft': n_fft,
        'hop_length': hop_length, 
        'center': True,
        # 'window': torch.hann_window(n_fft, requires_grad=False, device=cuda),
        'onesided': True,
        'pad_mode': 'reflect'
    }

    istft_params = {
        'n_fft': n_fft,
        'win_length': n_fft,
        'hop_length': hop_length, 
        'center': True,

        'window': torch.hann_window(n_fft, requires_grad=False, device=cuda)
    }

    unmix_kwargs = {   
        'n_fft': n_fft,
        'nb_bins':  n_fft // 2 + 1,
        'nb_channels': 2,
        'hidden_size': 512,
        'nb_layers': 3,
        'unidirectional': False,
        'input_mean': None,
        'input_scale': None,
        'max_bin': None,
    }
    

    transformer_kwargs = {
        'rate': rate,
        'n_fft': n_fft,
        'hop_size': hop_length,
        'mel_bins': 256,

    }
    class separator:
        load_model = False
        load_model_path = f'/home/lego/NAS189/home/MUSDB18/params/{NAME}'
        save_model_path = f'/home/lego/NAS189/home/MUSDB18/params/{NAME}'
        latent_size = 128
    
    class query_encoder:
        load_model = False
        load_model_path = 'params/OpenMic_encoder_UnetVGG.pkl'
        save_model_path = 'params/OpenMic_encoder_UnetVGG.pkl'
        ch_in = [2, 32, 32, 64, 64, 128]
        ch_out = [32, 32, 64, 64, 128, 128]
        kernel_size = [4, 4, 4, 4, 4, 4]
        stride = [(2,2), (2,1), (2,2), (2,1), (2,1), (2,1)]
        gru_input_size = 30 * ch_out[-1] 
        hidden_size = 128
        fc_input_size = hidden_size * 3
        latent_size = 128
    
    class Unet:
        class encoder:
            leakiness = 0.2
            ch_in = [2, 16, 32, 64, 128, 256]
            ch_out = [16, 32, 64, 128, 256, 512]
            kernel_size = [(5,5)] *6
            stride = [(2,2), (2,2), (2,2), (2,1), (2,1), (2,1)] 
        class decoder:
            ch_in = [512, 512, 256, 128, 64, 32]
            ch_out = [256, 128, 64, 32, 16, 2]
            kernel_size = [(5,5)] *6
            stride = [(2,1), (2,1), (2,1), (2,2), (2,2), (2,2)] 
        latent_size = 128



