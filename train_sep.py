import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from tqdm import tqdm
from asteroid.losses.sdr import SingleSrcNegSDR

from config import Open_Config as config
from dataset.dataset import MUSDB18_seg
from dataset.utils import *
from utils.visualize import figGen, AudioGen
from utils.lib import sdr_calc
from utils.criterion import KLD_Loss, SupConLoss
from separator.open_unmix import OpenUnmix
from separator.dprnn import DPRNN
from encoder.query import Query_Encoder
from torchaudio_augmentations import Compose
#from repr_separator.linear import LinearReLU

print('preparing the models...')
device = torch.device(f'cuda:{config.cuda}')

# encoder
encoder = Query_Encoder(config.query_encoder)
if config.query_encoder.load_model:
    encoder.load_state_dict(torch.load(config.query_encoder.load_model_path))
encoder.to(device)

separator = DPRNN(**config.dprnn_kwargs)
separator = torch.nn.DataParallel(separator, config.parallel_cuda)
if config.separator.load_model:
    separator.load_state_dict(torch.load(config.separator.load_model_path))
separator.to(device)


# dataset =======
#dataset = MUSDB18_seg(transform=torch.nn.Sequential(*config.transforms))
train_set = MUSDB18_seg(subset='train', transform=None)
test_set = MUSDB18_seg(subset='test')
#train_set, eval_set = torch.utils.data.random_split(dataset, [round(len(dataset) * (1-config.split_ratio)), round(len(dataset) * (config.split_ratio))])
train_loader = DataLoader(train_set, config.batch_size, shuffle=True, num_workers=config.num_workers)
eval_loader = DataLoader(test_set, config.batch_size, shuffle=True, num_workers=config.num_workers)
stft_extractor = STFT(config.cuda, **config.stft_params)
 

# opt =============
opt = optim.Adam(
    [{'params':separator.parameters()}, {'params':encoder.parameters()}],
    lr=config.lr,
    betas=config.betas,
    weight_decay=config.weight_decay
    )
scheduler = optim.lr_scheduler.StepLR(opt, config.lr_gamma, config.lr_step)

# criterion ===============
criterion = torch.nn.MSELoss()
similarity = torch.nn.CosineEmbeddingLoss()
kldLoss = KLD_Loss()
astsdr = SingleSrcNegSDR(sdr_type='sisdr', zero_mean=False)
contrastive = SupConLoss(contrast_mode='all')
crossentropy= torch.nn.CrossEntropyLoss()
# utils =================
writter = SummaryWriter(f'runs/{config.project}/{config.summary_name}')
audioGen = AudioGen(config.istft_params)


print('start training')
for epoch in range(1, config.n_epoch):
    # Training +======================
    separator.train()
    encoder.train()
    epoch_tloss, epoch_eloss = 0, 0
    epoch_c_tloss, epoch_c_eloss = 0, 0, 
    epoch_k_tloss, epoch_k_eloss = 0, 0
    epoch_con_tloss, epoch_con_eloss = 0, 0
    for batch, (x, label, m) in enumerate(tqdm(train_loader)):
        opt.zero_grad()
        x = move_data_to_device(x, device)
        m = move_data_to_device(m, device)
        label = move_data_to_device(label, device)

        x = stft_extractor(x)
        m = stft_extractor(m)
        latentx, mux, varx, r = encoder(x)
        maskx = separator(m, latentx)
        reconstruct_loss = criterion(maskx * m, x)
        
        #celoss = crossentropy(r, label-1)
        #contrast_loss = contrastive(latentx.unsqueeze(1), labels=label)
        kloss = kldLoss(mux, varx)
        batch_loss = reconstruct_loss * config.loss_coefficients[0] \
            + kloss * config.loss_coefficients[1] 
        #+ celoss * config.loss_coefficients[1]
        #+ contrast_loss * config.loss_coefficients[1] \
        

        """if epoch > 100 and epoch % 5 == 1:
            latentm, mum, varm = encoder(maskx * m)
            cycle_loss = similarity(mum, mux, torch.ones(1).to(device))
            batch_loss += cycle_loss * config.loss_coefficients[1]
            epoch_c_tloss += cycle_loss.item()"""
        
        epoch_k_tloss += kloss.item()    
        epoch_tloss += reconstruct_loss.item()
        #epoch_con_tloss += celoss.item()
        batch_loss.backward()
        opt.step()
        
        
    epoch_tloss /= (batch+1)
    epoch_c_tloss /= (batch+1)
    epoch_k_tloss /= (batch+1)
    epoch_con_tloss /= (batch+1)
    
    print(f'{epoch} | training loss: {epoch_tloss}')
    writter.add_scalar('train/MSELOSS', epoch_tloss, epoch )
    writter.add_scalar('train/cycleLoss', epoch_c_tloss, epoch)
    writter.add_scalar('train/kLoss', epoch_k_tloss, epoch)
    writter.add_scalar('train/contrastive', epoch_con_tloss, epoch)

    writter.add_figure('train_fig/pred', figGen(m * maskx, config), epoch)
    writter.add_figure('train_fig/gt', figGen(x, config), epoch)
    writter.add_figure('train_fig/mixture', figGen(m, config), epoch)
    writter.add_figure('train_fig/mask', figGen(maskx, config, mask=True), epoch)
    
    # evaling =========================== 
    if epoch % 5 == 1:
        
        writter.add_audio('train_audio/pred', audioGen((m * maskx)[0]), epoch, config.rate)
        writter.add_audio('train_audio/gt', audioGen(x[0]), epoch, config.rate)
        writter.add_audio('train_audio/mixture', audioGen(m[0]), epoch, config.rate)
        
        with torch.no_grad():
            sdsdr = (-1 * astsdr(audioGen(m * maskx), audioGen(x))).median()
        writter.add_scalar('train/SDR', sdsdr , epoch)
        #if epoch % 50 == 1 and epoch > 1:
            #sdsdr = sdr_calc(x, m, maskx, audioGen)
            
        #writter.add_scalar('train_metric/SI-SDR', sisdr , epoch)
    
        # &&&&&&&&&&&&&&&
        
        print('evaling...')
        separator.eval()
        encoder.eval()
        for batch, (x, label, m) in enumerate(tqdm(eval_loader)):
            with torch.no_grad():
                x = move_data_to_device(x, device)
                m = move_data_to_device(m, device)
                x = stft_extractor(x)
                m = stft_extractor(m)
                latentx, mux, varx, r = encoder(x)
                maskx = separator(m, mux)
                latentm, mum, varm, r= encoder(maskx * m)
                
                reconstruct_loss = criterion(maskx * m, x)
                cycle_loss = similarity(mum, mux, torch.ones(1).to(device))
                #kloss = kldLoss(mux, varx)
            
            
            epoch_eloss += reconstruct_loss.item()
            epoch_c_eloss += cycle_loss.item()
            #epoch_k_eloss += kloss.item()
        
        epoch_eloss /= (batch+1)
        epoch_c_eloss /= (batch+1)
        epoch_k_eloss /= (batch+1)
        
        print(f'{epoch} | eval loss: {epoch_eloss}')
        writter.add_scalar('eval/MSEloss', epoch_eloss, epoch )
        writter.add_scalar('eval/cycleLoss', epoch_c_eloss, epoch)
        writter.add_scalar('train/kLoss', epoch_k_eloss, epoch)
        
        #if epoch % 50 == 1 and epoch > 1:
            #sdsdr = sdr_calc(x, m, maskx, audioGen)
        with torch.no_grad():
            sdsdr = (-1 * astsdr(audioGen(m * maskx), audioGen(x))).median()
        writter.add_scalar('eval/SDR', sdsdr , epoch)
            
        
        writter.add_figure('eval_fig/pred', figGen(m * maskx, config), epoch)
        writter.add_figure('eval_fig/gt', figGen(x, config), epoch)
        writter.add_figure('eval_fig/mixture', figGen(m, config), epoch)
        writter.add_figure('eval_fig/mask', figGen(maskx, config, mask=True), epoch)
    
        writter.add_audio('eval_audio/pred', audioGen((m * maskx)[0]), epoch, config.rate)
        writter.add_audio('eval_audio/gt', audioGen(x[0]), epoch, config.rate)
        writter.add_audio('eval_audio/mixture', audioGen(m[0]), epoch, config.rate)
        
        
    
        torch.save(separator.state_dict(), f'{config.separator.save_model_path}' )
        torch.save(encoder.state_dict(), f'{config.query_encoder.save_model_path}')