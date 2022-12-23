import numpy as np
import torch
import torchaudio
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
args = parser.parse_args()

print('preparing the models...')
device = torch.device(f'cuda:{args.cuda}')
stft_extractor = STFT(args.cuda, **config.stft_params)
instrument_indice = ['drums', 'bass','other','vocals'].index(args.src)
complexnorm = ComplexNorm()
separator = separator_builder(args)

if config.separator.load_model:
    """ model_dict = separator.state_dict()
    pretrained_dict = {k: v for k, v in torch.load(config.separator.load_model_path).items() if k in model_dict}
    model_dict.update(pretrained_dict) 
    separator.load_state_dict(model_dict) """
    separator.load_state_dict(torch.load(config.separator.load_model_path+'_0.pkl'))
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

""" attentions = []
def save_attention_matrix(event):
    attentions.append(event.attention_matrix.detach().cpu().numpy())
EventDispatcher.get().listen(AttentionEvent, save_attention_matrix) """

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
        
        # v_stft = move_data_to_device(v_stft, device)
        # m_stft = move_data_to_device(m_stft, device)

        # m_norm = complexnorm(m_stft) 
        # v_norm = complexnorm(v_stft)

        m_norm = rearrange(m_stft, 'b ch h w c -> b (c ch) h w')
        pred_v, mask = separator(m_norm)
        pred_v = rearrange(pred_v, 'b (c ch) h w -> b ch h w c', c=2 )

        batch_loss = criterion(pred_v, v_stft)
        train_loss += batch_loss.item()

        batch_loss.backward()
        opt.step()
        
        """ np.save('att_train',np.array(attentions))
        np.save('mel', m_mel.detach().cpu().numpy())
        np.save('mel_v',v_mel.detach().cpu().numpy())
        attentions.clear() """
        
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

                # v_stft = move_data_to_device(v_stft, device)
                # m_stft = move_data_to_device(m_stft, device)
                
                # m_norm = complexnorm(m_stft) 
                # v_norm = complexnorm(v_stft)

                m_norm = rearrange(m_stft, 'b ch h w c -> b (c ch) h w')
                pred_v, mask = separator(m_norm)
                pred_v = rearrange(pred_v, 'b (c ch) h w -> b ch h w c', c=2 )


                x_loss = criterion(pred_v, v_stft)
                batch_loss =  x_loss 


                eval_loss += batch_loss.item()

    

        eval_loss /= (batch+1)
        scheduler.step(eval_loss)

        
        print(f'{epoch} T| eval loss: {eval_loss} | min eval loss: {min_val_loss}' )
        """ if epoch % 30 == 0:
            print(f'{epoch} E| eval loss: {eval_loss}, SDR: {sdr.median()}' )
            writter.add_scalar('eval/SDR', sdr.median() , epoch) """
        
        """ writter.add_figure('eval_fig/pred_x', melGen(pred_v, config), epoch)
        writter.add_figure('eval_fig/gt_x', melGen(v_norm, config), epoch)
        writter.add_figure('eval_fig/mixture', melGen(m_norm, config), epoch) 
        writter.add_figure('eval_fig/mask', melGen(mask, config, mask=True), epoch)
        # writter.add_figure('eval_fig/sal', salGen(sal, config), epoch)


        writter.add_audio('eval_audio/pred_x', audioGen(pred_stft[0,0]), epoch, config.rate)
        writter.add_audio('eval_audio/gt_x', audioGen(v_stft[0]), epoch, config.rate)
        writter.add_audio('eval_audio/mixture', audioGen(m_stft[0]), epoch, config.rate)  """
        
        
        if min_val_loss > eval_loss:
            min_val_loss = eval_loss
            print(f'changed min eval loss: {min_val_loss}')
            torch.save(separator.state_dict(), f'{config.separator.save_model_path}_0.pkl' )
            earlystop_count = 0
        # elif earlystop_count < 3:
        #     torch.save(separator.state_dict(), f'{config.separator.save_model_path}_{earlystop_count}.pkl' )
        elif earlystop_count > 50 or epoch > 1000:
            print('early stop')
            break
        earlystop_count += 1