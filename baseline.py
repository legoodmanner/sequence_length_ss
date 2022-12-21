import numpy as np
import torch
import torch.optim as optim
from torch.utils.data.dataset import Subset
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from tqdm import tqdm

from config import Baseline_Config as config

from separator.open_unmix_baseline import OpenUnmix
from dataset.dataset import AmpleGuitar, MUSDB18_seg
from dataset.utils import *
from utils.criterion import PITLoss
from utils.visualize import figGen, AudioGen
from utils.lib import sdr_calc

print('preparing the models...')
device = torch.device(f'cuda:{config.cuda}')

separator = OpenUnmix(**config.unmix_kwargs)
separator = torch.nn.DataParallel(separator, config.parallel_cuda)
if config.separator.load_model:
    separator.load_state_dict(torch.load(config.separator.load_model_path))
separator.to(device)

"""train_set = MUSDB18_seg(subset='train', mode='pair')
test_set = MUSDB18_seg(subset='test', mode='pair')"""
train_set = AmpleGuitar(subset='train', set1='AGT', set2='AGSC')
train_loader = DataLoader(train_set, config.batch_size, shuffle=False, num_workers=config.num_workers)
test_set = AmpleGuitar(subset='test', set1='AGT', set2='AGSC')
test_loader = DataLoader(test_set, config.batch_size, shuffle=True, num_workers=config.num_workers)
stft_extractor = STFT(config.cuda, **config.stft_params)

opt = optim.Adam(
    separator.parameters(),
    lr=config.lr,
    betas=config.betas,
    weight_decay=config.weight_decay
    )

scheduler = optim.lr_scheduler.StepLR(opt, config.lr_gamma, config.lr_step)
#criterion = PITLoss(torch.nn.MSELoss())
criterion = torch.nn.MSELoss()

writter = SummaryWriter(f'runs/{config.project}/{config.summary_name}')
audioGen = AudioGen(config.istft_params)

print('start training')
for epoch in range(1, config.n_epoch):
    # Training +======================
    separator.train()
    train_loss = 0
    for batch, (x, y) in enumerate(tqdm(train_loader)):

        x = move_data_to_device(x, device)
        y = move_data_to_device(y, device)
        
        m = stft_extractor(x+y)
        x = stft_extractor(x)
        y = stft_extractor(y)

        mask = separator(m)
        loss = criterion(mask*m, x)
        #loss, perm = criterion(x, y, mask*m, (1-mask)*m)

        train_loss += loss.item()
        loss.backward()
        opt.step()
        opt.zero_grad()
    train_loss /= (batch+1)

    writter.add_scalar('train_loss', train_loss, epoch)
    print(f'{epoch} | train {train_loss}')
    
    if epoch % 10 > 0:
        continue
    writter.add_figure('train_fig/x', figGen(x, config), epoch)
    writter.add_figure('train_fig/y', figGen(y, config), epoch)
    writter.add_figure('train_fig/pred', figGen(m*mask, config), epoch)
    writter.add_figure('train_fig/mixture', figGen(m, config), epoch)
    writter.add_figure('train_fig/mask', figGen(mask, config, mask=True), epoch)

    writter.add_audio('train_audio/pred', audioGen((m * mask)[0]), epoch, config.rate)
    writter.add_audio('train_audio/y', audioGen(y[0]), epoch, config.rate)
    writter.add_audio('train_audio/x', audioGen(x[0]), epoch, config.rate)
    writter.add_audio('train_audio/mixture', audioGen(m[0]), epoch, config.rate)

    separator.eval()
    eval_loss = 0
    for batch, (x, y) in enumerate(tqdm(test_loader)):

        x = move_data_to_device(x, device)
        y = move_data_to_device(y, device)
        
        m = stft_extractor(x+y)
        x = stft_extractor(x)
        y = stft_extractor(y)

        mask = separator(m)
        loss = criterion(mask*m, x)
        #loss, perm = criterion(x, y, mask*m, (1-mask)*m)

        eval_loss += loss.item()

        if batch == 0:
            sdr_a = torch.Tensor(sdr_calc(x, m, mask, audioGen)[0])
            sdr_b = torch.Tensor(sdr_calc(y, m, (1-mask), audioGen)[0])
            
        else:
            sdr_a = torch.cat([sdr_a, torch.Tensor(sdr_calc(x, m, mask, audioGen)[0])])
            sdr_b = torch.cat([sdr_b, torch.Tensor(sdr_calc(y, m, (1-mask), audioGen)[0])])
            

    
    
    eval_loss /= (batch+1)
    print(f'{epoch} | eval {eval_loss}')

    writter.add_scalar('eval_loss', eval_loss, epoch)
    writter.add_scalar('eval_sdr_a', sdr_a.median(), epoch)
    writter.add_scalar('eval_sdr_b', sdr_b.median(), epoch)
     
    writter.add_figure('eval_fig/x', figGen(x, config), epoch)
    writter.add_figure('eval_fig/y', figGen(y, config), epoch)
    writter.add_figure('eval_fig/pred_a', figGen(mask*m, config), epoch)
    writter.add_figure('eval_fig/pred_b', figGen((1-mask)*m, config), epoch)
    writter.add_figure('eval_fig/mixture', figGen(m, config), epoch)
    writter.add_figure('eval_fig/mask', figGen(mask, config, mask=True), epoch)

    writter.add_audio('eval_audio/pred_a', audioGen((mask*m)[0]), epoch, config.rate)
    writter.add_audio('eval_audio/pred_b', audioGen(((1-mask)*m)[0]), epoch, config.rate)
    writter.add_audio('eval_audio/y', audioGen(y[0]), epoch, config.rate)
    writter.add_audio('eval_audio/x', audioGen(x[0]), epoch, config.rate)
    writter.add_audio('eval_audio/mixture', audioGen(m[0]), epoch, config.rate)
    
    
    torch.save(separator.state_dict(), config.separator.save_model_path)
        


        

