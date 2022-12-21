import numpy as np
import torch
import torch.optim as optim
import random
from torch.utils.data.dataset import Subset
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from tqdm import tqdm

from config import Baseline_Config as config
from separator.dprnn import DPRNN
from separator.open_unmix_baseline import OpenUnmix
from encoder.onehot_encoder import Simple_Encoder
from dataset.dataset import *
from dataset.utils import *
from utils.criterion import PITLoss
from utils.visualize import figGen, AudioGen
from utils.lib import sdr_calc

print('preparing the models...')
device = torch.device(f'cuda:{config.cuda}')

separator = DPRNN(**config.dprnn_kwargs)
separator = torch.nn.DataParallel(separator, config.parallel_cuda)
if config.separator.load_model:
    separator.load_state_dict(torch.load(config.separator.load_model_path))
separator.to(device)

encoder = Simple_Encoder(config.latent_size)
if config.encoder.load_model:
    encoder.load_state_dict(torch.load(config.encoder.load_model_path))
encoder.to(device)

"""train_set = MUSDB18_seg(subset='train', mode='pair')
test_set = MUSDB18_seg(subset='test', mode='pair')"""
dataset = JSBChorale(mode= config.loader_mode, root='/home/lego/NAS189/home/JSBChorale/npy_choir/')
split = len(dataset) // 3 
indices = list(range(len(dataset)))
train_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[split:])
valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[:split])
train_loader = DataLoader(dataset, batch_size=config.batch_size, sampler=train_sampler, num_workers=config.num_workers)
valid_loader = DataLoader(dataset, batch_size=config.batch_size, sampler=valid_sampler, num_workers=config.num_workers)

opt = optim.Adam(
    params=list(separator.parameters()) + list(encoder.parameters()),
    lr=config.lr,
    betas=config.betas,
    weight_decay=config.weight_decay
    )

scheduler = optim.lr_scheduler.StepLR(opt, config.lr_gamma, config.lr_step)
criterion = torch.nn.MSELoss()

writter = SummaryWriter(f'runs/{config.project}/{config.summary_name}')
audioGen = AudioGen(config.istft_params)
stft_extractor = STFT(config.cuda, **config.stft_params)

print('start training')
for epoch in range(1, config.n_epoch):
    # Training +======================
    separator.train()
    encoder.train()
    train_loss = 0
    for batch, (x, m, label) in enumerate(tqdm(train_loader)):

        x = move_data_to_device(x, device)
        m = move_data_to_device(m, device)
        label = move_data_to_device(label, device)

        x = stft_extractor(x)
        m = stft_extractor(m)

        mask = separator(m, encoder(label))
        loss = criterion(mask*m, x)

        train_loss += loss.item()
        loss.backward()
        opt.step()
        opt.zero_grad()
    

        
    train_loss /= (batch+1)
    print(f'{epoch} | train {train_loss}')
    print(label[0])
    
    writter.add_scalar('train_loss', train_loss, epoch)
    if epoch % 10 > 0:
        continue
    
   
    writter.add_figure('train_fig/x', figGen(x, config), epoch)
    writter.add_figure('train_fig/pred', figGen(m*mask, config), epoch)
    writter.add_figure('train_fig/mixture', figGen(m, config), epoch)
    writter.add_figure('train_fig/mask', figGen(mask, config, mask=True), epoch)

    writter.add_audio('train_audio/pred', audioGen((m * mask)[0]), epoch, config.rate)
    writter.add_audio('train_audio/x', audioGen(x[0]), epoch, config.rate)
    writter.add_audio('train_audio/mixture', audioGen(m[0]), epoch, config.rate)
    

    separator.eval()
    encoder.eval()
    eval_loss = 0
    for batch, (x, m, label) in enumerate(tqdm(valid_loader)):

        x = move_data_to_device(x, device)
        m = move_data_to_device(m, device)
        label = move_data_to_device(label, device)

        x = stft_extractor(x)
        m = stft_extractor(m)
        
        mask = separator(m, encoder(label))
        loss = criterion(mask*m, x)
        
        eval_loss += loss.item()

        if batch == 0:
            sdr_a = torch.Tensor(sdr_calc(x, m, mask, audioGen)[0])
            
        else:
            sdr_a = torch.cat([sdr_a, torch.Tensor(sdr_calc(x, m, mask, audioGen)[0])])

            
    eval_loss /= (batch+1)
    print(f'{epoch} | eval {eval_loss}')

    writter.add_scalar('eval_loss', eval_loss, epoch)
    writter.add_scalar('eval_sdr_a', sdr_a.median(), epoch)
    #writter.add_scalar('eval_sdr_b', sdr_b.median(), epoch)
     
    writter.add_figure('eval_fig/x', figGen(x, config), epoch)
    writter.add_figure('eval_fig/pred_a', figGen(mask*m, config), epoch)
    #writter.add_figure('eval_fig/pred_b', figGen((1-mask)*m, config), epoch)
    writter.add_figure('eval_fig/mixture', figGen(m, config), epoch)
    writter.add_figure('eval_fig/mask', figGen(mask, config, mask=True), epoch)

    writter.add_audio('eval_audio/pred_a', audioGen((mask*m)[0]), epoch, config.rate)
    #writter.add_audio('eval_audio/pred_b', audioGen(((1-mask)*m)[0]), epoch, config.rate)
    #writter.add_audio('eval_audio/y', audioGen(y[0]), epoch, config.rate)
    writter.add_audio('eval_audio/x', audioGen(x[0]), epoch, config.rate)
    writter.add_audio('eval_audio/mixture', audioGen(m[0]), epoch, config.rate)
    
    
    torch.save(separator.state_dict(), config.separator.save_model_path)
    torch.save(encoder.state_dict(), config.encoder.save_model_path)
        


        

 