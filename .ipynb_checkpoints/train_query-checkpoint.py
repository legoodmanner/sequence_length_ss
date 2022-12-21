import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchlibrosa.stft import Spectrogram
from tqdm import tqdm

from config import Query_Config as config
from data.dataset import MUSDB18_seg
from data.utils import *
from utils.visualize import figGen, AudioGen
from utils.criterion import KLD_Loss
from separator.query_unet import Query_UNet
from encoder.query import Query_Encoder

print('preparing the models...')
device = torch.device(f'cuda:{config.cuda}')

# Query-encoder
encoder = Query_Encoder(config.query_encoder)
if config.query_encoder.load_model:
    encoder.load_state_dict(torch.load(config.query_encoder.model_path))
encoder.to(device)


# Query-Unet
separator = Query_UNet(config.separator)
separator = torch.nn.DataParallel(separator, config.parallel_cuda)
if config.separator.load_model:
    separator.load_state_dict(torch.load(config.separator.load_model_path))
separator.to(device)

print('preparing data...')
# dataset
dataset = MUSDB18_seg()
loader = DataLoader(dataset, config.batch_size, shuffle=True, num_workers=config.num_workers)
transform = STFT(config.cuda, **config.stft_params)

print('preparing other things...')
# opt
opt = optim.Adam(
    [
        {'params':separator.parameters()},
        {'params':encoder.parameters()}
    ],
    lr=config.lr,
    betas=config.betas
    )
scheduler = optim.lr_scheduler.StepLR(opt, config.lr_gamma, 10000)

# criterion
mse = torch.nn.MSELoss()
kld = KLD_Loss()
l1 = torch.nn.L1Loss(reduction='mean')
c_r, c_kl, c_l = config.loss_coefficients

# utils
writter = SummaryWriter(f'runs/{config.project}/{config.summary_name}')
audioGen = AudioGen(config.stft_params)

print('start training')
for epoch in range(1, config.n_epoch):
    separator.train()
    encoder.train()
    epoch_loss = 0
    for batch, (target, mixture) in enumerate(tqdm(loader)):
        opt.zero_grad()

        target = move_data_to_device(target, device)
        mixture = move_data_to_device(mixture, device)
        
        mixture = transform(mixture)
        target = transform(target)

        latentx, mux, varx = encoder(target)
        #latenty, muy, vary = encoder(y)
        latentz = torch.normal(0, 1, size=latentx.size())
        latentz = move_data_to_device(latentz, device)

        maskt = separator(mixture, latentx)
        #masky = separator(m, latenty)
        maskz = separator(mixture, latentz)
        rloss = l1(mixture * maskt, target) 
        kloss = kld(mux, varx)
        lloss = l1(encoder(mixture * maskz)[0], latentz)

        batch_loss = rloss * c_r + kloss * c_kl + lloss * c_l
        epoch_loss += batch_loss.item()
        batch_loss.backward()

        opt.step()
        
        
    print(f'{epoch} | epoch_loss: {epoch_loss}')
    writter.add_scalar('loss/epoch_loss', epoch_loss, epoch )
    writter.add_scalar('loss/batch_rloss', rloss.item(), epoch)
    writter.add_scalar('loss/batch_kloss', kloss.item(), epoch)
    writter.add_scalar('loss/batch_lloss', lloss.item(), epoch)
    
    
    """writter.add_figure('y/pred', figGen(m * masky), epoch)
    writter.add_figure('y/gt', figGen(y), epoch)
    writter.add_figure('y/mask', figGen(masky), epoch)"""

    writter.add_figure('x/pred', figGen(mixture * maskt), epoch)
    writter.add_figure('x/gt', figGen(target), epoch)
    writter.add_figure('mixture', figGen(mixture), epoch)
    writter.add_figure('x/mask', figGen(maskt), epoch)
    
    """writter.add_audio('pred', audioGen((mixture * maskt)[:2]), epoch)
    writter.add_audio('gt', audioGen(target[:2]), epoch)
    writter.add_audio('mixture', audioGen(mixture[:2]), epoch)"""
    
    torch.save(separator.state_dict(), f'{config.separator.save_model_path}' )
    torch.save(encoder.state_dict(), f'{config.query_encoder.save_model_path}')