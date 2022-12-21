import numpy as np
import torch
import torchaudio
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from einops import rearrange

from tqdm import tqdm
from fast_transformers.events import EventDispatcher, AttentionEvent

from config import Open_Config as config
from dataset.dataset import MUSDB18_vocal, MUSDB18_seg, MUSDB18_length
from dataset.utils import *
from utils.visualize import figGen, AudioGen, melGen
from utils.lib import sdr_calc
from utils.criterion import KLD_Loss, SupConLoss, PITLoss
from separator.transformer import TransformerEncoderContainer
from separator.open_unmix_baseline import OpenUnmix, Separator
from separator.umxtransformer import UmxTransformer
from separator.open_unmix import Vanilla_Transformer
from separator.query_unet import Query_UNet
from separator.dense_unet import TFC_NET

print('preparing the models...')
device = torch.device(f'cuda:{config.cuda}')


stft_extractor = STFT(config.cuda, **config.stft_params)
complexnorm = ComplexNorm()
# separator = Query_UNet()
separator = TFC_NET(
    n_fft=4096,
    n_blocks=9, 
    input_channels=4, 
    internal_channels=16, 
    n_internal_layers=3,
    first_conv_activation='relu', 
    last_activation='relu',
    t_down_layers=None, 
    f_down_layers=None,
    kernel_size_t=3, 
    kernel_size_f=3,
    tfc_activation='relu',
)
# separator = Vanilla_Transformer()
# separator = UmxTransformer()
# separator = TransformerEncoderContainer()
# separator = OpenUnmix(**config.unmix_kwargs)

""" unmix = OpenUnmix(**config.unmix_kwargs)
checkpoint = torch.load('params/vocals-c8df74a5.pth', map_location=device)
del checkpoint["sample_rate"]
del checkpoint["stft.window"]
del checkpoint[ "transform.0.window"] 
unmix.load_state_dict(checkpoint) """ 

separator = torch.nn.DataParallel(separator, config.parallel_cuda)
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
train_loader = DataLoader(train_dataset, config.batch_size, num_workers=config.num_workers, shuffle=True, persistent_workers=False)
eval_loader = DataLoader(valid_dataset, config.batch_size, num_workers=config.num_workers, shuffle=False, persistent_workers=False)

instrument_indice= config.instrument_indice
print(['drum', 'bass','other','vocal'][instrument_indice])

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
writter = SummaryWriter(f'runs/{config.project}/{config.summary_name}')
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
    writter.add_scalar('train_loss', train_loss, epoch)
    
    
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
        
        writter.add_scalar('eval_loss', eval_loss, epoch)
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