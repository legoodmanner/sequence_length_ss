import numpy as np
import torch
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchlibrosa.stft import Spectrogram
from tqdm import tqdm
from asteroid.losses.sdr import SingleSrcNegSDR

from config import Eval_Config as config
from dataset.dataset import MUSDB18_seg
from dataset.utils import *
from utils.visualize import figGen, AudioGen
from utils.lib import sdr_calc

from separator.open_unmix import OpenUnmix
from separator.dprnn import DPRNN
from encoder.query import Query_Encoder
#from repr_separator.linear import LinearReLU

print('preparing the models...')
device = torch.device(f'cuda:{config.cuda}')

# encoder
encoder = Query_Encoder(config.query_encoder)
encoder.load_state_dict(torch.load(config.query_encoder.load_model_path))
encoder.to(device)

separator = DPRNN(**config.dprnn_kwargs)
separator = torch.nn.DataParallel(separator, [config.cuda])
separator.load_state_dict(torch.load(config.separator.load_model_path))
separator.to(device)


# dataset =======
dataset = MUSDB18_seg(subset='test', mode='all', inst=config.instrument)
loader = DataLoader(dataset, config.batch_size, shuffle=True, num_workers=config.num_workers)
transform = STFT(config.cuda, **config.stft_params)


# criterion ===============
criterion = torch.nn.MSELoss()
astsdr = SingleSrcNegSDR(sdr_type='sisdr', zero_mean=False)

# utils =================
writter = SummaryWriter(f'runs/{config.project}/{config.summary_name}')
audioGen = AudioGen(config.istft_params)

for epoch in range(1, config.n_epoch):
    # Training +======================
    separator.eval()
    encoder.eval()
    epoch_xloss, epoch_yloss = 0, 0
    for batch, data in enumerate(tqdm(loader)):
        if len(data) == 3:
            x, label, m = data
        else:
            x, m = data
        with torch.no_grad():
            x = move_data_to_device(x, device)
            m = move_data_to_device(m, device)
            #y = m - x 
            x = transform(x)
            #y = transform(y)
            m = transform(m)

            latentx, mux, varx = encoder(x)
            #latenty, muym, vary = encoder(y)

            maskx = separator(m, latentx)
            #masky = separator(m, latenty)
            
            if batch == 0:
                epoch_sdr = astsdr(audioGen(m * maskx), audioGen(x))
                writter.add_embedding(latentx, metadata=label, global_step=epoch, tag='default', metadata_header=None)
            else:
                epoch_sdr = torch.cat([epoch_sdr, astsdr(audioGen(m * maskx), audioGen(x))])
            
            
            batchx_loss = criterion(maskx * m, x) 
            #batchy_loss = criterion(masky * m, y)

            epoch_xloss += batchx_loss.item()
            #epoch_sdr += sdr
            #epoch_yloss += batchy_loss.item()

    epoch_xloss /= (batch+1)
    epoch_sdr = epoch_sdr.median()
    # epoch_yloss /= (batch+1)
    print(f'{epoch} |x loss: {epoch_xloss} | sdsr: {epoch_sdr}')
    
    
    writter.add_scalar('test/MSEloss', epoch_xloss, epoch )
    #writter.add_scalar('y/MSEloss', epoch_yloss, epoch )

    #print(maskx[0])
    writter.add_scalar('test/SD-SDR', epoch_sdr, epoch)
    #writter.add_scalar('x/SI-SDR', sisdr , epoch)
    #writter.add_histogram('x/SIR', sir , epoch)
    #writter.add_histogram('x/SAR', sar , epoch)

    #sdr = sdr_calc(y, m, masky, audioGen)
    #writter.add_scalar('y/SDR', sdr , epoch)
    #writter.add_histogram('y/SIR', sir , epoch)
    #writter.add_histogram('y/SAR', sar , epoch)
    
    writter.add_figure('mixture', figGen(m, config), epoch)
    writter.add_audio('mixture_audio', audioGen(m)[0], epoch, config.rate)
    
    writter.add_figure('test_fig/pred', figGen(m * maskx, config), epoch)
    writter.add_figure('test_fig/gt', figGen(x, config), epoch)
    writter.add_figure('test_fig/mask', figGen(maskx, config, mask=True), epoch)
    """writter.add_figure('y/pred', figGen(m * masky, config), epoch)
    writter.add_figure('y/gt', figGen(y, config), epoch)
    writter.add_figure('y/mask', figGen(masky, config, mask=True), epoch)
    """
    writter.add_audio('test_audio/pred_audio', audioGen(m * maskx)[0], epoch, config.rate)
    writter.add_audio('test_audio/gt_audio', audioGen(x)[0], epoch, config.rate)
    #writter.add_audio('y/pred_audio', audioGen((m * masky)[0]), epoch, config.rate)
    #writter.add_audio('y/gt_audio', audioGen(y[0]), epoch, config.rate)
    
    

