import numpy as np
import torch
import torch.optim as optim
from torch.utils.data.dataset import Subset
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from tqdm import tqdm

from config import Baseline_Config as config
from separator.dprnn_baseline import DPRNN_Baseline
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
separator.load_state_dict(torch.load(config.separator.load_model_path))
separator.to(device)

"""train_set = MUSDB18_seg(subset='train', mode='pair')
test_set = MUSDB18_seg(subset='test', mode='pair')"""

test_set = AmpleGuitar(subset='test', set1='AGSC', set2='AGSC')
test_loader = DataLoader(test_set, config.eval_batch_size, shuffle=True, num_workers=config.num_workers)
stft_extractor = STFT(config.cuda, **config.stft_params)

criterion = PITLoss(torch.nn.MSELoss())

writter = SummaryWriter(f'runs/{config.project}/{config.summary_name}')
audioGen = AudioGen(config.istft_params)

print('start training')
for epoch in range(1, 20):
    # Training +======================

    separator.eval()
    eval_loss = 0
    for batch, (x, y) in enumerate(tqdm(test_loader)):
        with torch.no_grad():
            x = move_data_to_device(x, device)
            y = move_data_to_device(y, device)
            
            m = stft_extractor(x+y)
            x = stft_extractor(x)
            y = stft_extractor(y)

            mask = separator(m)
        loss, perm = criterion(x, y, mask*m, (1-mask)*m)

        eval_loss += loss.item()

        if batch == 0:
            sdr_a, sar_a, sir_a = torch.Tensor(sdr_calc(x, m, mask, audioGen))
            sdr_b, sar_b, sir_b = torch.Tensor(sdr_calc(y, m, (1-mask), audioGen))
            
        else:
            sdra, sara, sira = torch.Tensor(sdr_calc(x, m, mask, audioGen))
            sdrb, sarb, sirb = torch.Tensor(sdr_calc(y, m, (1-mask), audioGen))
            sdr_a = torch.cat([sdr_a, sdra])
            sdr_b = torch.cat([sdr_b, sdrb])
            sar_a = torch.cat([sar_a, sara])
            sar_b = torch.cat([sar_b, sarb])
            sir_a = torch.cat([sir_a, sira])
            sir_b = torch.cat([sir_b, sirb])
            

    
    
    eval_loss /= (batch+1)
    print(f'{epoch} | eval {eval_loss}')

    writter.add_scalar('eval_loss', eval_loss, epoch)
    writter.add_scalar('eval_sdr_a/median', sdr_a.median(), epoch)
    writter.add_scalar('eval_sdr_b/median', sdr_b.median(), epoch)
    writter.add_scalar('eval_sar_a/median', sar_a.median(), epoch)
    writter.add_scalar('eval_sar_b/median', sar_b.median(), epoch)
    writter.add_scalar('eval_sir_a/median', sir_a.median(), epoch)
    writter.add_scalar('eval_sir_b/median', sir_b.median(), epoch)

    writter.add_scalar('eval_sdr_a/mean', sdr_a.mean(), epoch)
    writter.add_scalar('eval_sdr_b/mean', sdr_b.mean(), epoch)
    writter.add_scalar('eval_sar_a/mean', sar_a.mean(), epoch)
    writter.add_scalar('eval_sar_b/mean', sar_b.mean(), epoch)
    writter.add_scalar('eval_sir_a/mean', sir_a.mean(), epoch)
    writter.add_scalar('eval_sir_b/mean', sir_b.mean(), epoch)

    writter.add_scalar('eval_sdr_a/std', sdr_a.std(), epoch)
    writter.add_scalar('eval_sdr_b/std', sdr_b.std(), epoch)
    writter.add_scalar('eval_sar_a/std', sar_a.std(), epoch)
    writter.add_scalar('eval_sar_b/std', sar_b.std(), epoch)
    writter.add_scalar('eval_sir_a/std', sir_a.std(), epoch)
    writter.add_scalar('eval_sir_b/std', sir_b.std(), epoch)
    
    writter.add_histogram('eval_sdra', sdr_a, epoch)
    writter.add_histogram('eval_sdrb', sdr_b, epoch)
    

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
    
    
    #torch.save(separator.state_dict(), config.separator.save_model_path)
        


        

