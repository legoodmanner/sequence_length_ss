import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from tqdm import tqdm
from asteroid.losses.sdr import SingleSrcNegSDR

from config import Open_Config as config
from dataset.dataset import OpenMic
from dataset.utils import *
from utils.visualize import figGen, AudioGen
from utils.lib import sdr_calc
from utils.criterion import KLD_Loss, SupConLoss, PITLoss
from separator.open_unmix import OpenUnmix
from separator.dprnn import DPRNN
from separator.query_unet import Query_UNet
from encoder.query import Query_Encoder, VGGish



print('preparing the models...')
device = torch.device(f'cuda:{config.cuda}')

# encoder
encoder = VGGish(**config.vggish_kwargs)
encoder = torch.nn.DataParallel(encoder, config.parallel_cuda)
if config.query_encoder.load_model:
    encoder.load_state_dict(torch.load(config.query_encoder.load_model_path))
encoder.to(device)

# separator
#separator = DPRNN(**config.dprnn_kwargs)
separator = Query_UNet(config.Unet)
separator = torch.nn.DataParallel(separator, config.parallel_cuda)
if config.separator.load_model:
    separator.load_state_dict(torch.load(config.separator.load_model_path))
separator.to(device)

# linear feature-sep

# dataset =======
#dataset = MUSDB18_seg(transform=torch.nn.Sequential(*config.transforms))

dataset = OpenMic(split_ratio=config.split_ratio)
split = int(np.ceil(len(dataset) * (1-config.split_ratio)))
indices = list(range(len(dataset)))
valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[split:])
train_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[:split])
#train_set, eval_set = torch.utils.data.random_split(dataset, [round(len(dataset) * (1-config.split_ratio)), round(len(dataset) * (config.split_ratio))])
train_loader = DataLoader(dataset, config.batch_size, sampler=train_sampler, num_workers=config.num_workers)
eval_loader = DataLoader(dataset, config.batch_size, sampler=valid_sampler, num_workers=config.num_workers)
stft_extractor = STFT(config.cuda, **config.stft_params)
 

# opt =============
opt = optim.Adam(
    [{'params':encoder.parameters()}, {'params': separator.parameters()},],
    lr=config.lr,
    betas=config.betas,
    weight_decay=config.weight_decay
    )
scheduler = optim.lr_scheduler.StepLR(opt, config.lr_gamma, config.lr_step)

# criterion ===============
criterion = torch.nn.MSELoss()
similarity = torch.nn.CosineEmbeddingLoss()
kldLoss = KLD_Loss()
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
    train_loss, eval_loss = 0, 0 
    for batch, (x, y, vggx, vggy) in enumerate(tqdm(train_loader)):
        opt.zero_grad()

        m = move_data_to_device(x+y, device)
        x = move_data_to_device(x, device)
        y = move_data_to_device(y, device)
        vggx = move_data_to_device(vggx, device)
        vggy = move_data_to_device(vggy, device)
        #latentx = encoder(x)
        #latenty = encoder(y)
        #latentxy, cxy = encoder(xy)

        x = stft_extractor(x)
        y = stft_extractor(y)
        m = stft_extractor(m)

        maskx = separator(m, vggx)
        masky = separator(m, vggy)

        x_loss = criterion(maskx * m, x)
        y_loss = criterion(masky * m, y)

        #reconstruct_loss = xy_loss + x_loss 
        
        #celoss = crossentropy(r, label-1)
        #contrast_loss = contrastive(latentx.unsqueeze(1), labels=label)
        #kloss = kldLoss(mux, varx)
        batch_loss =  x_loss + y_loss 
        #+ celoss * config.loss_coefficients[1]
        #+ contrast_loss * config.loss_coefficients[1] \
        #+ kloss * config.loss_coefficients[1] 

        """if epoch > 100 and epoch % 5 == 1:
            latentm, mum, varm = encoder(maskx * m)
            cycle_loss = similarity(mum, mux, torch.ones(1).to(device))
            batch_loss += cycle_loss * config.loss_coefficients[1]
            epoch_c_tloss += cycle_loss.item()"""
        
        train_loss += batch_loss.item()    
        batch_loss.backward()
        opt.step()
        
        
    train_loss /= (batch+1)    
    
    print(f'{epoch} T| train loss: {train_loss}' )
    writter.add_scalar('train_loss', train_loss, epoch)
    
    
    
    if epoch % 5 == 1:

        writter.add_figure('train_fig/pred_x', figGen(m * maskx, config), epoch)
        writter.add_figure('train_fig/gt_x', figGen(x, config), epoch)
        writter.add_figure('train_fig/mixture', figGen(m, config), epoch)
        writter.add_figure('train_fig/mask', figGen(maskx, config, mask=True), epoch)
        """writter.add_figure('train_fig/h', figGen(h, config, mask=True), epoch)
        writter.add_figure('train_fig/r', figGen(r, config, mask=True), epoch)
        writter.add_figure('train_fig/lstm1', figGen(lstm_out1, config, mask=True), epoch)
        writter.add_figure('train_fig/lstm2', figGen(lstm_out2, config, mask=True), epoch)
        """
        writter.add_audio('train_audio/pred_x', audioGen((m * maskx)[0]), epoch, config.rate)
        writter.add_audio('train_audio/pred_y', audioGen((m * masky)[0]), epoch, config.rate)
        writter.add_audio('train_audio/gt_x', audioGen(x[0]), epoch, config.rate)
        writter.add_audio('train_audio/gt_y', audioGen(y[0]), epoch, config.rate)
        writter.add_audio('train_audio/mixture', audioGen(m[0]), epoch, config.rate)
        
        
        #if epoch % 50 == 1 and epoch > 1:
            #sdsdr = sdr_calc(x, m, maskx, audioGen)
            
        #writter.add_scalar('train_metric/SI-SDR', sisdr , epoch)
    

        # evaling =========================== 
        print('evaling...')
        separator.eval()
        encoder.eval()
        for batch, (x, y, vggx, vggy) in enumerate(tqdm(eval_loader)):
            with torch.no_grad():
                m = move_data_to_device(x+y, device)
                x = move_data_to_device(x, device)
                y = move_data_to_device(y, device)
                
                vggx = move_data_to_device(vggx, device)
                vggy = move_data_to_device(vggy, device)

                #latentx = encoder(x)
                #latenty = encoder(y)
                #latentxy, cxy = encoder(xy)

                x = stft_extractor(x)
                y = stft_extractor(y)
                m = stft_extractor(m)

                maskx = separator(m, vggx)
                masky = separator(m, vggy)

                x_loss = criterion(maskx * m, x)
                y_loss = criterion(masky * m, y)
                batch_loss =  x_loss + y_loss 

                if batch == 0:
                    #epoch_sdr = -1* astsdr(audioGen(m * maskx), audioGen(x))
                    sdr = torch.Tensor(sdr_calc(x, m, maskx, audioGen))[0]
                    #writter.add_embedding(latentx, metadata=label, global_step=epoch, tag='default', metadata_header=None)
                else:
                    sdr = torch.cat([sdr, torch.Tensor(sdr_calc(x, m, maskx, audioGen))[0]])
                    #epoch_sdr = torch.cat([epoch_sdr, -1 * astsdr(audioGen(m * maskx), audioGen(x))])
            #epoch_k_eloss += kloss.item()

            eval_loss += batch_loss.item()

        eval_loss /= (batch+1)
        
        print(f'{epoch} E| eval loss: {eval_loss}' )
        
        #if epoch % 50 == 1 and epoch > 1:
            #sdsdr = sdr_calc(x, m, maskx, audioGen)
        writter.add_scalar('eval/SDR', sdr.median() , epoch)
        
        writter.add_scalar('eval_loss', eval_loss, epoch)
        writter.add_figure('eval_fig/pred_x', figGen(m * maskx, config), epoch)
        writter.add_figure('eval_fig/gt_x', figGen(x, config), epoch)
        writter.add_figure('eval_fig/mixture', figGen(m, config), epoch)
        writter.add_figure('eval_fig/mask', figGen(maskx, config, mask=True), epoch)
        """writter.add_figure('eval_fig/h', figGen(h, config, mask=True), epoch)
        writter.add_figure('eval_fig/r', figGen(r, config, mask=True), epoch)
        writter.add_figure('eval_fig/lstm1', figGen(lstm_out1, config, mask=True), epoch)
        writter.add_figure('eval_fig/lstm2', figGen(lstm_out2, config, mask=True), epoch)"""

    
        writter.add_audio('eval_audio/pred_x', audioGen((m * maskx)[0]), epoch, config.rate)
        writter.add_audio('eval_audio/pred_y', audioGen((m * masky)[0]), epoch, config.rate)
        writter.add_audio('eval_audio/gt_x', audioGen(x[0]), epoch, config.rate)
        writter.add_audio('eval_audio/gt_y', audioGen(y[0]), epoch, config.rate)
        writter.add_audio('eval_audio/mixture', audioGen(m[0]), epoch, config.rate)
        
        
    
        torch.save(separator.state_dict(), f'{config.separator.save_model_path}' )
        torch.save(encoder.state_dict(), f'{config.query_encoder.save_model_path}')
