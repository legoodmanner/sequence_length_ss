import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from tqdm import tqdm
from asteroid.losses.sdr import SingleSrcNegSDR

from config import Open_Config as config
from dataset.dataset import OpenMic_Classfication
from dataset.utils import *
from utils.visualize import figGen, AudioGen
from utils.lib import sdr_calc
from utils.criterion import MyWcploss
from separator.open_unmix import OpenUnmix
from separator.dprnn import DPRNN
from encoder.query import Query_Encoder, Post_Encoder, CNN_Encoder
from sklearn.metrics import f1_score


print('preparing the models...')
device = torch.device(f'cuda:{config.cuda}')

# encoder
encoder = Query_Encoder(config.query_encoder)
encoder = torch.nn.DataParallel(encoder, config.parallel_cuda)
encoder.load_state_dict(torch.load(config.query_encoder.load_model_path))
encoder.to(device)

post_encoder = CNN_Encoder(config)
post_encoder = torch.nn.DataParallel(post_encoder, config.parallel_cuda)
post_encoder.to(device)

dataset = OpenMic_Classfication(split_ratio=config.split_ratio)
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
    [{'params':post_encoder.parameters()}],
    lr=config.lr,
    betas=config.betas,
    weight_decay=config.weight_decay
    )
scheduler = optim.lr_scheduler.StepLR(opt, config.lr_gamma, config.lr_step)

# criterion ===============
criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.full((20,), 7).to(device))
# utils =================
writter = SummaryWriter(f'runs/{config.project}/{config.summary_name}_classfication')
audioGen = AudioGen(config.istft_params)


print('start training')
for epoch in range(1, config.n_epoch):
    # Training +======================
    post_encoder.train()
    encoder.eval()
    train_loss, eval_loss = 0, 0 
    for batch, (data, label, vgg) in enumerate(tqdm(train_loader)):
        opt.zero_grad()

        data = move_data_to_device(data, device)
        label = move_data_to_device(label, device)            
        latent = move_data_to_device(torch.empty((data.size(0), data.size(1)*3, config.latent_size)), device)
        with torch.no_grad():
            for i in range(data.size(1)):
                x = stft_extractor(data[:,i,:])
                latentx = encoder(x)
                latent[:,i*3:i*3+3,:] = latentx
         
        pred = post_encoder(latent)
        batch_loss = criterion(pred, label)
        train_loss += batch_loss.item()    
        batch_loss.backward()
        opt.step()
        
        
    train_loss /= (batch+1)    
    
    print(f'{epoch} T| train loss: {train_loss}' )
    writter.add_scalar('train_loss', train_loss, epoch)
    
    if epoch % 1 == 0:
        

        # evaling =========================== 
        print('evaling...')
        post_encoder.eval()
        encoder.eval()
        for batch, (data, label, vgg) in enumerate(tqdm(eval_loader)):
            opt.zero_grad()

            data = move_data_to_device(data, device)
            label = move_data_to_device(label, device)
            latent = move_data_to_device(torch.empty((data.size(0), data.size(1)*3, config.latent_size)), device)
            with torch.no_grad():
                for i in range(data.size(1)):
                    x = stft_extractor(data[:,i,:])
                    latentx = encoder(x)
                    latent[:,i*3:i*3+3,:] = latentx
            
            pred = post_encoder(latent)
            batch_loss = criterion(pred, label)     
            eval_loss += batch_loss.item()    
            thr_pred = torch.where(torch.sigmoid(pred)>=0.5,  torch.ones_like(pred), torch.zeros_like(pred))
            #print(np.where(thr_pred[0].detach().cpu().numpy()==1))
            #print(np.where(label[0].detach().cpu().numpy()==1))
            if batch == 0:
                #epoch_sdr = -1* astsdr(audioGen(m * maskx), audioGen(x))
                thr_preds = thr_pred
                labels = label
                preds = pred
                #writter.add_embedding(latentx, metadata=label, global_step=epoch, tag='default', metadata_header=None)
            else:
                thr_preds = torch.cat((thr_preds, thr_pred), dim=0)
                preds = torch.cat((preds, pred), dim=0)
                labels = torch.cat((labels, label), dim=0)

        f1 = f1_score(labels.detach().cpu().numpy(), thr_preds.detach().cpu().numpy(), average=None)
        eval_loss /= (batch+1)
        
        print(f'{epoch} E| eval loss: {eval_loss}' )
        print(f1)
        #writter.add_histogram('eval/F1', f1, epoch)
        writter.add_pr_curve('eval/F1', labels, preds, epoch)
        
        writter.add_scalar('eval_loss', eval_loss, epoch)
        
        torch.save(post_encoder.state_dict(), f'params/post_encoder.pkl')
