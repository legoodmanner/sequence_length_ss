import torch
import torch.nn as nn
import torch.optim as opt
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from data.dataset import PureMic_mixup
from data.utils import *
from config import config
from encoder.cnn14 import Cnn14
from repr_separator.linear import LinearReLU
from utils.criterion import PITLoss

# Encoder
encoder = Cnn14()
checkpoint = torch.load('params/Cnn14_mAP=0.431.pth')
encoder.load_state_dict(checkpoint['model'])

# Separator
separator = LinearReLU()
separator = torch.nn.DataParallel(separator, [2,3])
if config.load_separator:
    separator.load_state_dict(torch.load(config.separator_pkl_loaddir))

# Dataset DataLoader
dataset = PureMic_mixup()
loader = DataLoader(dataset, config.batch_size, shuffle=True, num_workers=8)

# opt
optimizer = opt.Adam(separator.parameters(), lr=config.lr)
scheduler = opt.lr_scheduler.StepLR(optimizer, config.lr_gamma)

# criterion
criterion = PITLoss()

# CUDA 
device = torch.device(f'cuda:{config.cuda}')
encoder.to(device)
encoder = torch.nn.DataParallel(encoder, [2,3])
separator.to(device)


print('GPU number: {}'.format(torch.cuda.device_count()))

writer = SummaryWriter()
separator.train()

for epoch in range(config.epoch_n):
    epoch_loss = 0
    for batch, (y1, y2) in enumerate(loader):
        
        y1 = move_data_to_device(y1, device)
        y2 = move_data_to_device(y2, device)
        
        
        mix = y1 + y2
        # B, 2048
        with torch.no_grad():
            encoder.eval()
            emb_m = encoder(mix)['embedding']
            emb_1 = encoder(y1)['embedding']
            emb_2 = encoder(y2)['embedding']

        """emb_1 = move_data_to_device(emb_1, device)
        emb_2 = move_data_to_device(emb_2, device)
        emb_m = move_data_to_device(emb_m, device)"""
        
        
        # B, 2, 2048
        emb_p = separator(emb_m)
        
        batch_loss = criterion(emb_1,emb_2,emb_p)
        epoch_loss += batch_loss.data
        
        batch_loss.backward()
        optimizer.step()
        
        if batch % 50 == 0:
            print(f'-- {batch} / {len(loader)} | bLoss: {batch_loss.data}')
            writer.add_scalar('Loss', batch_loss)
    print(f'Loss: {epoch_loss/(batch+1)}' )
    scheduler.step()
    torch.save(separator.state_dict(), config.separator_pkl_savedir)
