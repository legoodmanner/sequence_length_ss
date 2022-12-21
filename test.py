import numpy as np
import torch
import torchaudio
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from tqdm import tqdm
from fast_transformers.events import EventDispatcher, AttentionEvent

from config import Open_Config as config
from dataset.dataset import MUSDB18_vocal_test
from dataset.utils import *
from utils.visualize import figGen, AudioGen, melGen, salGen
from utils.lib import sdr_calc
from utils.criterion import KLD_Loss, SupConLoss, PITLoss
from separator.transformer import TransformerEncoderContainer

from separator.umxtransformer import UmxTransformer
from separator.transformer import TransformerEncoderContainer
from separator.open_unmix_baseline import OpenUnmix, Separator
from separator.umxtransformer import UmxTransformer
from separator.open_unmix import Vanilla_Transformer
from separator.query_unet import Query_UNet

print('preparing the models...')
device = torch.device(f'cuda:0')


stft_extractor = STFT(0, **config.stft_params)
complexnorm = ComplexNorm()

# separator
separator = UmxTransformer()
# separator = TransformerEncoderContainer()
# separator = Vanilla_Transformer()
# separator = OpenUnmix(**config.unmix_kwargs)
# separator = Query_UNet()
length = 7 * 44100 // 1024
# separator = torch.nn.DataParallel(separator, config.parallel_cuda)
""" model_dict = separator.state_dict()
pretrained_dict = {k: v for k, v in torch.load(f'{config.separator.load_model_path}_0.pkl').items() if k in model_dict}
model_dict.update(pretrained_dict)  """
separator.load_state_dict(torch.load(f'{config.separator.load_model_path}_0.pkl'))
print(f'{config.separator.load_model_path}')
print(f'length: {length}')

separator.to(device)
# dataset =======


test_dataset = MUSDB18_vocal_test(subsets='test')
eval_loader = DataLoader(test_dataset, 1, num_workers=config.num_workers, shuffle=False)

# criterion ===============
criterion = torch.nn.MSELoss()

# utils =================
writter = SummaryWriter(f'runs/{config.project}/test')
audioGen = AudioGen(config.istft_params)


instrument_indice= config.instrument_indice
print(['drum', 'bass','other','vocal'][instrument_indice])

attentions = []
def save_attention_matrix(event):
    attentions.append(event.attention_matrix.detach().cpu().numpy())
EventDispatcher.get().listen(AttentionEvent, save_attention_matrix)

print('start training')

    # Training +======================

separator.eval()
stft_extractor.eval()
eval_loss = 0
for batch, wav in enumerate(tqdm(eval_loader)):
    with torch.no_grad():
        # wav = move_data_to_device(wav, device)
        # source_indice = torch.tensor([3]*wav.shape[0], device=wav.device)
        v_stft_, m_stft_ = stft_extractor(wav, instrument_indice)
        pred_stft_ = torch.zeros((1,1)+v_stft_.shape[1:]).float()
        nb_frames =  m_stft_.shape[-2]
        pos = 0
        while pos < nb_frames:
            cur_frame = torch.arange(pos, min(nb_frames, pos + length))
            """ if pos + length > nb_frames:
                break """
            m_stft = move_data_to_device(m_stft_[...,cur_frame,:], device)
            v_stft = move_data_to_device(v_stft_[...,cur_frame,:], device)

            m_norm = complexnorm(m_stft) 
            v_norm = complexnorm(v_stft)
        
            # pred_unmix, mask = unmix(m_norm)
            pred_v, mask = separator(m_norm)

            pred_stft = torch.zeros(m_stft.permute(0,3,2,1,4).shape + (1,), device=m_stft.device)

            for b in range(v_stft.shape[0]):

                pred_stft[b] = wiener(
                    targets_spectrograms=pred_v.permute(0,3,2,1)[b].unsqueeze(-1), 
                    mix_stft=m_stft.permute(0,3,2,1,4)[b],
                    iterations=0,
                    residual=False,
                )
            pred_stft = pred_stft.permute(0, 5, 3, 2, 1, 4).contiguous()
            pred_stft_[...,cur_frame,:] = pred_stft.detach().cpu()
            pos = int(cur_frame[-1]) + 1
            # if pos + 300 < nb_frames:
            
            np.save('att_train',np.array(attentions))
            attentions.clear()
        np.save('m_stft', m_stft.detach().cpu().numpy())
        
        # writter.add_audio('test_audio/pred_x', audioGen(pred_stft_[0,0]), 1, config.rate)
        # writter.add_audio('test_audio/gt', audioGen(v_stft_[0]), 1, config.rate)
        if batch == 0:
            sdr, sar, sir = torch.Tensor(sdr_calc(v_stft_, m_stft_, pred_stft_, audioGen))
        else:
            sdr = torch.cat([sdr, torch.Tensor(sdr_calc(v_stft_, m_stft_, pred_stft_, audioGen))[0]])
            sar = torch.cat([sar, torch.Tensor(sdr_calc(v_stft_, m_stft_, pred_stft_, audioGen))[1]])
            sir = torch.cat([sir, torch.Tensor(sdr_calc(v_stft_, m_stft_, pred_stft_, audioGen))[2]])
        print(sdr)
        print(sdr.median())

print('_______________________')
print(f'{config.separator.load_model_path}')
print(['drum', 'bass','other','vocal'][instrument_indice])
print(f'sdr {sdr.median()}')    
print(f'sir {sir.median()}')    
print(f'sar {sar.median()}')    




        

       
""" print(f'{0} E| eval loss: {eval_loss}, SDR: {sdr.median()}' )
writter.add_histogram('sdr_histogram',sdr,0)
writter.add_scalar('eval/SDR', sdr.median() , 0) """
    

    
    



