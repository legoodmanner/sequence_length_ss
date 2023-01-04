import numpy as np
import torch
import os
import torch.optim as optim
import argparse
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from einops import rearrange

from tqdm import tqdm
from fast_transformers.events import EventDispatcher, AttentionEvent

from config import Open_Config as config
from dataset.dataset import MUSDB18_vocal, MUSDB18_seg, MUSDB18_length
from dataset.utils import *
from utils.visualize import figGen, AudioGen, melGen
from utils.lib import sdr_calc, separator_builder
from utils.criterion import KLD_Loss, SupConLoss, PITLoss


parser = argparse.ArgumentParser()
parser.add_argument('--cuda', help='determine the available GPU id')
parser.add_argument('--separator', help='determine the type of the separator', choice=['umx', 'umx_transformer', 'auto_transformer'])
parser.add_argument('--attention', help='determine the type of the attention', choice=['qkv2D', 'qkv1D', 'qkvg1D'])
parser.add_argument('--bs', help='batch_size')
parser.add_argument('--src', help='determine the source need to be separated', choice = ['drums', 'bass','other','vocals'])
parser.add_argument('--ckpt_dir', help='determine the path of the directory for model checkpoints')
args = parser.parse_args()

print('preparing the models...')
device = torch.device(f'cuda:{args.cuda}')
stft_extractor = STFT(args.cuda, **config.stft_params)
instrument_indice = ['drums', 'bass','other','vocals'].index(args.src)
complexnorm = ComplexNorm()
separator = separator_builder(args)
model_name = f'{args.separator}_{args.attention}_{args.src}'
ckpt_path = os.path.join(args.ckpt_dir, model_name)+'.pkl'
os.makedirs(args.ckpt_dir, exist_ok=True)

if config.separator.load_model:
    separator.load_state_dict(torch.load(ckpt_path))
print(f'paramers: {sum(p.numel() for p in separator.parameters())}')
separator.to(device)

train_dataset = MUSDB18_seg(subsets='train', split='train')
valid_dataset = MUSDB18_seg(subsets='train', split='valid')
train_loader = DataLoader(train_dataset, args.batch_size, num_workers=config.num_workers, shuffle=True, persistent_workers=False)
eval_loader = DataLoader(valid_dataset, args.batch_size, num_workers=config.num_workers, shuffle=False, persistent_workers=False)


# opt =============
opt = optim.Adam(
    [{'params':separator.parameters()},],
    lr=config.lr,
    betas=config.betas,
    weight_decay=config.weight_decay
    )
scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.3, patience=16, cooldown=10)
# criterion ===============
criterion = torch.nn.MSELoss()

# utils =================
audioGen = AudioGen(config.istft_params)
earlystop_count = 0
min_val_loss = 100

print('start training')
for epoch in range(0, config.n_epoch):
    # Training +======================
    
    train_loss, eval_loss = 0, 0
    train_x_loss, train_cycle_loss = 0,0
    eval_x_loss, eval_cycle_loss = 0, 0

    separator.train()
    stft_extractor.train()
    # unmix.eval()
    for batch, wav in enumerate(tqdm(train_loader)):
        opt.zero_grad()
        wav = move_data_to_device(wav, device)
        
        # source_indice = torch.randint(0, 4, [wav.shape[0]], device=wav.device)
        source_indice = torch.tensor([3]*wav.shape[0], device=wav.device)

        with torch.no_grad():
            v_stft, m_stft = stft_extractor(wav, instrument_indice)

        m_norm = rearrange(m_stft, 'b ch h w c -> b (c ch) h w')
        pred_v, mask = separator(m_norm)
        pred_v = rearrange(pred_v, 'b (c ch) h w -> b ch h w c', c=2 )

        batch_loss = criterion(pred_v, v_stft)
        train_loss += batch_loss.item()

        batch_loss.backward()
        opt.step()
        
    print(opt.param_groups[0]['lr'])
    train_loss /= (batch+1)
    # train_x_loss /= (batch+1)
    
    

    print(f'{epoch} T| train loss: {train_loss}' )
    
    if epoch % 5 == 0:
        # evaling =========================== 
        print('evaling...')
        separator.eval()
        stft_extractor.eval()
        # unmix.eval()
        for batch, wav in enumerate(tqdm(eval_loader)):
            with torch.no_grad():
                
                opt.zero_grad()
                wav = move_data_to_device(wav, device)

                source_indice = torch.tensor([3]*wav.shape[0], device=wav.device)
                v_stft, m_stft = stft_extractor(wav, instrument_indice)
                m_norm = rearrange(m_stft, 'b ch h w c -> b (c ch) h w')
                pred_v, mask = separator(m_norm)
                pred_v = rearrange(pred_v, 'b (c ch) h w -> b ch h w c', c=2 )
                x_loss = criterion(pred_v, v_stft)
                batch_loss =  x_loss 
                eval_loss += batch_loss.item()

    

        eval_loss /= (batch+1)
        scheduler.step(eval_loss)

        
        print(f'{epoch} T| eval loss: {eval_loss} | min eval loss: {min_val_loss}' )        
        if min_val_loss > eval_loss:
            min_val_loss = eval_loss
            print(f'changed min eval loss: {min_val_loss}')
            torch.save(separator.state_dict(), ckpt_path)
            earlystop_count = 0
        elif earlystop_count > 50 or epoch > 1000:
            print('early stop')
            break
        earlystop_count += 1