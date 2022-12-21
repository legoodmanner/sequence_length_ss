import numpy as np
import torch
import torchaudio
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from tqdm import tqdm

from config import Open_Config as config
from dataset.dataset import MUSDB18_vocal, MUSDB18_vocal_seg
from dataset.utils import *
from utils.visualize import figGen, AudioGen, melGen, salGen
from utils.lib import sdr_calc
from utils.criterion import KLD_Loss, SupConLoss, PITLoss
from separator.transformer import TransformerEncoderContainer
from separator.open_unmix_baseline import OpenUnmix


print('preparing the models...')
device = torch.device(f'cuda:{config.cuda}')


stft_extractor = STFT(config.cuda, **config.stft_params)
mel_extractor = torchaudio.transforms.MelScale(n_mels=256, sample_rate=config.rate).to(device)
complexnorm = ComplexNorm()
# separator
separator = TransformerEncoderContainer(stft_extractor)
# separator = OpenUnmix(**config.unmix_kwargs)
""" checkpoint = torch.load('params/vocals-c8df74a5.pth', map_location=device)
del checkpoint["sample_rate"]
del checkpoint["stft.window"]
del checkpoint[ "transform.0.window"] 
separator.load_state_dict(checkpoint)  """
separator = torch.nn.DataParallel(separator, config.parallel_cuda)
if config.separator.load_model:
    separator.load_state_dict(torch.load(config.separator.load_model_path))

separator.to(device)

# linear feature-sep

# dataset =======
#dataset = MUSDB18_seg(transform=torch.nn.Sequential(*config.transforms))

train_dataset = MUSDB18_vocal_seg(subsets='train')
test_dataset = MUSDB18_vocal_seg(subsets='test')
# split = int(np.ceil(len(dataset) * (1-config.split_ratio)))
# indices = list(range(len(dataset)))
# valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[split:])
# train_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[:split])
train_loader = DataLoader(train_dataset, config.batch_size, num_workers=config.num_workers, shuffle=True)
eval_loader = DataLoader(test_dataset, config.batch_size, num_workers=config.num_workers, shuffle=True)



# opt =============
opt = optim.Adam(
    [{'params':separator.parameters()},],
    lr=config.lr,
    betas=config.betas,
    weight_decay=config.weight_decay
    )
scheduler = optim.lr_scheduler.StepLR(opt, config.lr_gamma, config.lr_step)

# criterion ===============
criterion = torch.nn.MSELoss()

# utils =================
writter = SummaryWriter(f'runs/{config.project}/{config.summary_name}')
audioGen = AudioGen(config.istft_params)


print('start training')
for epoch in range(1, config.n_epoch):
    # Training +======================
    
    train_loss, eval_loss = 0, 0 
    separator.train()
    for batch, (v, m) in enumerate(tqdm(train_loader)):
        opt.zero_grad()
    
        m = move_data_to_device(m, device)
        v = move_data_to_device(v, device)
        
        v_stft = stft_extractor(v)
        m_stft = stft_extractor(m)
        
        m_norm = complexnorm(m_stft) 
        v_norm = complexnorm(v_stft)

        m_mel = mel_extractor(m_norm)

        pred_v, mask = separator(v_norm, m_mel)

        x_loss = criterion(pred_v, v_norm)
        batch_loss =  x_loss 

        train_loss += batch_loss.item()    
        batch_loss.backward()
        opt.step()
        
    scheduler.step()
    train_loss /= (batch+1)    
    
    print(f'{epoch} T| train loss: {train_loss}' )
    writter.add_scalar('train_loss', train_loss, epoch)
    
    
    
    if epoch % 10 == 0:
        with torch.no_grad():
            pred_stft = torch.zeros(m_stft.permute(0,3,2,1,4).shape + (2,), device=m_stft.device)
            nb_frames = pred_stft.shape[1]
            for b in range(v.shape[0]):
                pos = 0
                while pos < nb_frames:
                    cur_frame = torch.arange(pos, min(nb_frames, pos + config.wiener_win_len))
                    pos = int(cur_frame[-1]) + 1

                    pred_stft[b, cur_frame] = wiener(
                        targets_spectrograms=pred_v.permute(0,3,2,1)[b, cur_frame].unsqueeze(-1), 
                        mix_stft=m_stft.permute(0,3,2,1,4)[b, cur_frame],
                        iterations=1,
                        residual=True
                    )
            pred_stft = pred_stft.permute(0, 5, 3, 2, 1, 4).contiguous() #B, n_src, n_channel, freq, time_len, complex

        writter.add_figure('train_fig/pred_x', melGen(pred_v, config), epoch)
        writter.add_figure('train_fig/gt_x', melGen(v_norm, config), epoch)
        writter.add_figure('train_fig/mixture', melGen(m_norm, config), epoch)
        writter.add_figure('train_fig/mask', melGen(mask, config, mask=True), epoch)
        # writter.add_figure('train_fig/sal', salGen(sal, config), epoch)
        writter.add_audio('train_audio/pred_x', audioGen(pred_stft[0,0]), epoch, config.rate)
        writter.add_audio('train_audio/gt_x', audioGen(v_stft[0]), epoch, config.rate)
        writter.add_audio('train_audio/mixture', audioGen(m_stft[0]), epoch, config.rate) 
    
        
        # if epoch % 50 == 1 and epoch > 1:
        #     sdsdr = sdr_calc(x, pred_v, audioGen)
        # writter.add_scalar('train_metric/SI-SDR', sisdr , epoch)
    

        # evaling =========================== 
        print('evaling...')
        separator.eval()
        for batch, (v,m) in enumerate(tqdm(eval_loader)):
            with torch.no_grad():
                
                opt.zero_grad()

                m = move_data_to_device(m, device)
                v = move_data_to_device(v, device)
                
                v_stft = stft_extractor(v)
                m_stft = stft_extractor(m)
                
                m_norm = complexnorm(m_stft) 
                v_norm = complexnorm(v_stft)

                m_mel = mel_extractor(m_norm)

                pred_v, mask = separator(v_norm, m_mel)

                x_loss = criterion(pred_v, v_norm)
                batch_loss =  x_loss

                
                pred_stft = torch.zeros(m_stft.permute(0,3,2,1,4).shape + (2,), device=m_stft.device)
                nb_frames = pred_stft.shape[1]
                for b in range(v.shape[0]):
                    pos = 0
                    while pos < nb_frames:
                        cur_frame = torch.arange(pos, min(nb_frames, pos + config.wiener_win_len))
                        pos = int(cur_frame[-1]) + 1

                        pred_stft[b, cur_frame] = wiener(
                            targets_spectrograms=pred_v.permute(0,3,2,1)[b, cur_frame].unsqueeze(-1), 
                            mix_stft=m_stft.permute(0,3,2,1,4)[b, cur_frame],
                            iterations=1,
                            residual=True,
                        )
                pred_stft = pred_stft.permute(0, 5, 3, 2, 1, 4).contiguous() #B, n_src, n_channel, freq, time_len, complex

                if epoch % 30 == 0:
                    if batch == 0:
                        sdr = torch.Tensor(sdr_calc(v_stft, m_stft, pred_stft, audioGen))[0]
                    else:
                        sdr = torch.cat([sdr, torch.Tensor(sdr_calc(v_stft, m_stft, pred_stft, audioGen))[0]]) 

                eval_loss += x_loss.item()

        eval_loss /= (batch+1)
        
        if epoch % 30 == 0:
            print(f'{epoch} E| eval loss: {eval_loss}, SDR: {sdr.median()}' )
            writter.add_scalar('eval/SDR', sdr.median() , epoch)
        
        writter.add_scalar('eval_loss', eval_loss, epoch)
        writter.add_figure('eval_fig/pred_x', melGen(pred_v, config), epoch)
        writter.add_figure('eval_fig/gt_x', melGen(v_norm, config), epoch)
        writter.add_figure('eval_fig/mixture', melGen(m_norm, config), epoch) 
        writter.add_figure('eval_fig/mask', melGen(mask, config, mask=True), epoch)
        # writter.add_figure('eval_fig/sal', salGen(sal, config), epoch)


        writter.add_audio('eval_audio/pred_x', audioGen(pred_stft[0,0]), epoch, config.rate)
        writter.add_audio('eval_audio/gt_x', audioGen(v_stft[0]), epoch, config.rate)
        writter.add_audio('eval_audio/mixture', audioGen(m_stft[0]), epoch, config.rate) 
        
        

        torch.save(separator.state_dict(), f'{config.separator.save_model_path}' )

