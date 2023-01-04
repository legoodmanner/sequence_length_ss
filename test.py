import numpy as np
import torch
import os
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from tqdm import tqdm
import argparse
from fast_transformers.events import EventDispatcher, AttentionEvent

from config import Open_Config as config
from dataset.dataset import MUSDB18_vocal_test
from dataset.utils import *
from utils.visualize import figGen, AudioGen, melGen, salGen
from utils.lib import sdr_calc, separator_builder

parser = argparse.ArgumentParser()
parser.add_argument('--cuda', help='determine the available GPU id')
parser.add_argument('--separator', help='determine the type of the separator', choice=['umx', 'umx_transformer', 'auto_transformer'])
parser.add_argument('--attention', help='determine the type of the attention', choice=['qkv2D', 'qkv1D', 'qkvg1D'])
parser.add_argument('--bs', help='batch_size')
parser.add_argument('--src', help='determine the source need to be separated', choice = ['drums', 'bass','other','vocals'])
parser.add_argument('--ckpt_dir', help='determine the path of the directory for model checkpoints')
args = parser.parse_args()

# Model Initialization
print('preparing the models...')
device = torch.device(f'cuda:{args.cuda}')
stft_extractor = STFT(args.cuda, **config.stft_params)
instrument_indice = ['drums', 'bass','other','vocals'].index(args.src)
complexnorm = ComplexNorm()
separator = separator_builder(args)
model_name = f'{args.separator}_{args.attention}_{args.src}'
ckpt_path = os.path.join(args.ckpt_dir, model_name)+'.pkl'

# Loading model
if os.path.isfile(ckpt_path):
    separator.load_state_dict(torch.load(ckpt_path))
else:
    print('The checkpoint file is not found!')
    raise NameError

# Parameter Counting
print(f'paramers: {sum(p.numel() for p in separator.parameters())}')
separator.to(device)
length = 7 * 44100 // 1024
print(f'length: {length}')

# dataset =======
test_dataset = MUSDB18_vocal_test(subsets='test')
eval_loader = DataLoader(test_dataset, 1, num_workers=config.num_workers, shuffle=False)

# criterion ===============
criterion = torch.nn.MSELoss()

# utils =================
writter = SummaryWriter(f'runs/{config.project}/test')
audioGen = AudioGen(config.istft_params)


instrument_indice= config.instrument_indice
print(['drum', 'bass','other','vocal'][instrument_indice])

attentions = []
def save_attention_matrix(event):
    attentions.append(event.attention_matrix.detach().cpu().numpy())
EventDispatcher.get().listen(AttentionEvent, save_attention_matrix)

print('start testing')

# Testing +======================

separator.eval()
stft_extractor.eval()
eval_loss = 0
for batch, wav in enumerate(tqdm(eval_loader)):
    with torch.no_grad():
        v_stft_, m_stft_ = stft_extractor(wav, instrument_indice)
        pred_stft_ = torch.zeros((1,1)+v_stft_.shape[1:]).float()
        nb_frames =  m_stft_.shape[-2]
        pos = 0
        while pos < nb_frames:
            cur_frame = torch.arange(pos, min(nb_frames, pos + length))
            m_stft = move_data_to_device(m_stft_[...,cur_frame,:], device)
            v_stft = move_data_to_device(v_stft_[...,cur_frame,:], device)

            m_norm = complexnorm(m_stft) 
            v_norm = complexnorm(v_stft)
        
            # pred_unmix, mask = unmix(m_norm)
            pred_v, mask = separator(m_norm)

            pred_stft = torch.zeros(m_stft.permute(0,3,2,1,4).shape + (1,), device=m_stft.device)

            for b in range(v_stft.shape[0]):

                pred_stft[b] = wiener(
                    targets_spectrograms=pred_v.permute(0,3,2,1)[b].unsqueeze(-1), 
                    mix_stft=m_stft.permute(0,3,2,1,4)[b],
                    iterations=0,
                    residual=False,
                )
            pred_stft = pred_stft.permute(0, 5, 3, 2, 1, 4).contiguous()
            pred_stft_[...,cur_frame,:] = pred_stft.detach().cpu()
            pos = int(cur_frame[-1]) + 1
            np.save('att_train',np.array(attentions))
            attentions.clear()
        np.save('m_stft', m_stft.detach().cpu().numpy())
        
        if batch == 0:
            sdr, sar, sir = torch.Tensor(sdr_calc(v_stft_, m_stft_, pred_stft_, audioGen))
        else:
            sdr = torch.cat([sdr, torch.Tensor(sdr_calc(v_stft_, m_stft_, pred_stft_, audioGen))[0]])
            sar = torch.cat([sar, torch.Tensor(sdr_calc(v_stft_, m_stft_, pred_stft_, audioGen))[1]])
            sir = torch.cat([sir, torch.Tensor(sdr_calc(v_stft_, m_stft_, pred_stft_, audioGen))[2]])
        print(sdr)
        print(sdr.median())

print('_______________________')
print(f'{config.separator.load_model_path}')
print(['drum', 'bass','other','vocal'][instrument_indice])
print(f'sdr {sdr.median()}')    
print(f'sir {sir.median()}')    
print(f'sar {sar.median()}')    




    

    
    



