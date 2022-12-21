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
from utils.criterion import KLD_Loss, SupConLoss, PITLoss
from separator.open_unmix import OpenUnmix
from separator.dprnn import DPRNN
from encoder.query import Query_Encoder
from repr_separator.linear import LinearReLU
from torchaudio_augmentations import Compose
from repr_separator.linear import LinearReLU

print('preparing the models...')
device = torch.device(f'cuda:{config.cuda}')

# encoder
encoder = Query_Encoder(config.query_encoder)
encoder = torch.nn.DataParallel(encoder, config.parallel_cuda)
if config.query_encoder.load_model:
    encoder.load_state_dict(torch.load(config.query_encoder.load_model_path))
encoder.to(device)

# separator
separator = DPRNN(**config.dprnn_kwargs)
separator = torch.nn.DataParallel(separator, config.parallel_cuda)
if config.separator.load_model:
    separator.load_state_dict(torch.load(config.separator.load_model_path))
separator.to(device)

# linear feature-sep
latent_sep = LinearReLU(latent_size=config.latent_size, input_size=config.query_encoder.fc_input_size)
latent_sep.to(device)

# dataset =======
#dataset = MUSDB18_seg(transform=torch.nn.Sequential(*config.transforms))
train_set = MUSDB18_seg(subset='train', mode='pair')
test_set = MUSDB18_seg(subset='test', mode='pair')
#train_set, eval_set = torch.utils.data.random_split(dataset, [round(len(dataset) * (1-config.split_ratio)), round(len(dataset) * (config.split_ratio))])
train_loader = DataLoader(train_set, config.batch_size, shuffle=True, num_workers=config.num_workers)
eval_loader = DataLoader(test_set, config.batch_size, shuffle=True, num_workers=config.num_workers)
stft_extractor = STFT(config.cuda, **config.stft_params)
 

# opt =============
opt = optim.Adam(
    [{'params':encoder.parameters()}, {'params': separator.parameters()}, {'params': latent_sep.parameters()}],
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
pit = PITLoss(criterion=torch.nn.MSELoss())
# utils =================
writter = SummaryWriter(f'runs/{config.project}/{config.summary_name}')
audioGen = AudioGen(config.istft_params)


print('start training')
for epoch in range(1, config.n_epoch):
    # Training +======================
    separator.train()
    encoder.train()
    epoch_pred_tloss, epoch_pred_eloss = 0, 0 
    epoch_double_tloss, epoch_double_eloss = 0, 0
    epoch_single_tloss, epoch_single_eloss = 0, 0
    epoch_pit_tloss, epoch_pit_eloss = 0, 0
    for batch, (x, y, m) in enumerate(tqdm(train_loader)):
        opt.zero_grad()
        x = move_data_to_device(x, device)
        y = move_data_to_device(y, device)
        m = move_data_to_device(m, device)
        
        xy = stft_extractor(x+y)
        x = stft_extractor(x)
        y = stft_extractor(y)
        m = stft_extractor(m)

        latentm, _ = encoder(m)
        latentx, _ = encoder(x)
        latenty, _ = encoder(y)
        latentxy, cxy = encoder(xy)

        pairloss = criterion(latentx + latenty, latentxy)
        pred_latent = latent_sep(latentxy, cxy)
        pitloss, perm = pit(latentx, latenty, pred_latent)

        maskx = separator(m, latentx)
        maskxy = separator(m, latentxy)
        mask_pred_x = separator(m, pred_latent[:,0])

        xy_loss = criterion(maskxy * m, xy)
        x_loss = criterion(maskx * m, x)
        x_pred_loss = criterion(mask_pred_x * m, x)

        #reconstruct_loss = xy_loss + x_loss 
        
        #celoss = crossentropy(r, label-1)
        #contrast_loss = contrastive(latentx.unsqueeze(1), labels=label)
        #kloss = kldLoss(mux, varx)
        batch_loss = xy_loss + x_loss + x_pred_loss + pitloss + pairloss
        #+ celoss * config.loss_coefficients[1]
        #+ contrast_loss * config.loss_coefficients[1] \
        #+ kloss * config.loss_coefficients[1] 

        """if epoch > 100 and epoch % 5 == 1:
            latentm, mum, varm = encoder(maskx * m)
            cycle_loss = similarity(mum, mux, torch.ones(1).to(device))
            batch_loss += cycle_loss * config.loss_coefficients[1]
            epoch_c_tloss += cycle_loss.item()"""
        
        epoch_pred_tloss += x_pred_loss.item()    
        epoch_double_tloss += xy_loss.item()
        epoch_single_tloss += x_loss.item()
        epoch_pit_tloss += pitloss.item()
        batch_loss.backward()
        opt.step()
        
        
    epoch_pred_tloss /= (batch+1)
    epoch_double_tloss /= (batch+1)
    epoch_single_tloss /= (batch+1)
    epoch_pit_tloss /= (batch+1)
    
    
    print(f'{epoch} T| single loss: {epoch_single_tloss} | double loss: {epoch_double_tloss} | latent pred loss: {epoch_pred_tloss}' )
    writter.add_scalar('train/single latent loss', epoch_single_tloss, epoch )
    writter.add_scalar('train/double latent loss', epoch_double_tloss, epoch)
    writter.add_scalar('train/pred latent loss', epoch_pred_tloss, epoch)
    writter.add_scalar('train/latent pitloss', epoch_pit_tloss, epoch)

    writter.add_figure('train_fig/single', figGen(m * maskx, config), epoch)
    writter.add_figure('train_fig/latent_pred', figGen(m * mask_pred_x, config), epoch)
    writter.add_figure('train_fig/gt', figGen(x, config), epoch)
    writter.add_figure('train_fig/mixture', figGen(m, config), epoch)
    writter.add_figure('train_fig/single_mask', figGen(maskx, config, mask=True), epoch)
    writter.add_figure('train_fig/latent_pred_mask', figGen(mask_pred_x, config, mask=True), epoch)
    
    # evaling =========================== 
    if epoch % 5 == 1:
        
        writter.add_audio('train_audio/single', audioGen((m * maskx)[0]), epoch, config.rate)
        writter.add_audio('train_audio/latent_pred', audioGen((m * mask_pred_x)[0]), epoch, config.rate)
        writter.add_audio('train_audio/gt', audioGen(x[0]), epoch, config.rate)
        writter.add_audio('train_audio/mixture', audioGen(m[0]), epoch, config.rate)
        
        with torch.no_grad():
            single_sdsdr = torch.Tensor(sdr_calc(x, m, maskx, audioGen)).median()
            pred_sdsdr = torch.Tensor(sdr_calc(x, m, mask_pred_x, audioGen)).median()
        writter.add_scalar('train/single_SDR', single_sdsdr , epoch)
        writter.add_scalar('train/pred_SDR', pred_sdsdr , epoch)
        #if epoch % 50 == 1 and epoch > 1:
            #sdsdr = sdr_calc(x, m, maskx, audioGen)
            
        #writter.add_scalar('train_metric/SI-SDR', sisdr , epoch)
    
        # &&&&&&&&&&&&&&&
        
        print('evaling...')
        separator.eval()
        encoder.eval()
        for batch, (x, y, m) in enumerate(tqdm(eval_loader)):
            with torch.no_grad():
                x = move_data_to_device(x, device)
                y = move_data_to_device(y, device)
                m = move_data_to_device(m, device)
                
                xy = stft_extractor(x+y)
                x = stft_extractor(x)
                y = stft_extractor(y)
                m = stft_extractor(m)

                latentm, _ = encoder(m)
                latentx, _ = encoder(x)
                latenty, _ = encoder(y)
                latentxy, cxy = encoder(xy)

                #pairloss = criterion(latentx + latenty, latentxy)
                pred_latent = latent_sep(latentxy, cxy)
                pitloss, perm = pit(latentx, latenty, pred_latent)

                maskx = separator(m, latentx)
                maskxy = separator(m, latentxy)
                mask_pred_x = separator(m, pred_latent[:,0])

                xy_loss = criterion(maskxy * m, xy)
                x_loss = criterion(maskx * m, x)
                x_pred_loss = criterion(mask_pred_x * m, x)

                if batch == 0:
                    #epoch_sdr = -1* astsdr(audioGen(m * maskx), audioGen(x))
                    single_sdsdr = torch.Tensor(sdr_calc(x, m, maskx, audioGen))
                    pred_sdsdr = torch.Tensor(sdr_calc(x, m, mask_pred_x, audioGen))
                    #writter.add_embedding(latentx, metadata=label, global_step=epoch, tag='default', metadata_header=None)
                else:
                    single_sdsdr = torch.cat([single_sdsdr, torch.Tensor(sdr_calc(x, m, maskx, audioGen))])
                    pred_sdsdr = torch.cat([pred_sdsdr, torch.Tensor(sdr_calc(x, m, mask_pred_x, audioGen))])
                    #epoch_sdr = torch.cat([epoch_sdr, -1 * astsdr(audioGen(m * maskx), audioGen(x))])
            #epoch_k_eloss += kloss.item()

            epoch_double_eloss += xy_loss.item()
            epoch_single_eloss += x_loss.item()
            epoch_pit_eloss += pitloss.item()
            epoch_pred_eloss += x_pred_loss

        epoch_pred_eloss /= (batch+1)
        epoch_double_eloss /= (batch+1)
        epoch_single_eloss /= (batch+1)
        epoch_pit_eloss /= (batch+1)
        
        print(f'{epoch} E| single loss: {epoch_single_eloss} | double loss: {epoch_double_eloss} | latent pred loss: {epoch_pred_eloss}' )
        writter.add_scalar('eval/single latent loss', epoch_single_eloss, epoch )
        writter.add_scalar('eval/double latent loss', epoch_double_eloss, epoch)
        writter.add_scalar('eval/latent pred loss', epoch_pred_eloss, epoch)
        writter.add_scalar('eval/pit loss', epoch_pit_eloss, epoch)
        
        #if epoch % 50 == 1 and epoch > 1:
            #sdsdr = sdr_calc(x, m, maskx, audioGen)
        writter.add_scalar('eval/single_SDR', single_sdsdr.median() , epoch)
        writter.add_scalar('eval/pred_SDR', pred_sdsdr.median() , epoch)
            
        
        writter.add_figure('eval_fig/single', figGen(m * maskx, config), epoch)
        writter.add_figure('eval_fig/latent_pred', figGen(m * mask_pred_x, config), epoch)
        writter.add_figure('eval_fig/gt', figGen(x, config), epoch)
        writter.add_figure('eval_fig/mixture', figGen(m, config), epoch)
        writter.add_figure('eval_fig/single_mask', figGen(maskx, config, mask=True), epoch)
        writter.add_figure('eval_fig/pred_mask', figGen(mask_pred_x, config, mask=True), epoch)
    
        writter.add_audio('eval_audio/single', audioGen((m * maskx)[0]), epoch, config.rate)
        writter.add_audio('eval_audio/latent_pred', audioGen((m * mask_pred_x)[0]), epoch, config.rate)
        writter.add_audio('eval_audio/gt', audioGen(x[0]), epoch, config.rate)
        writter.add_audio('eval_audio/mixture', audioGen(m[0]), epoch, config.rate)
        
        
    
        torch.save(separator.state_dict(), f'{config.separator.save_model_path}' )
        torch.save(encoder.state_dict(), f'{config.query_encoder.save_model_path}')
        torch.save(latent_sep.state_dict(), 'params/linear.pkl')
