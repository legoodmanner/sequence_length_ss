from tqdm import tqdm
import os
import numpy as np
import musdb

savedir = '/home/lego/NAS189/home/MUSDB18/22050_3sec_npy/'
train_mus = musdb.DB(root='/home/lego/NAS189/Database/musdb18', subsets='train')
test_mus = musdb.DB(root='/home/lego/NAS189/Database/musdb18', subsets='test')
hop_size = 1024
rate = 22050
window = rate * 3

train_id = 0
test_id = 0

print('splitting trainset...')
for track in tqdm(train_mus):
    # downsample
    source = (track.stems[:,::2, 0] + track.stems[:, ::2, 1]) / 2
    for i in range(0, source.shape[1]//window):
        np.save(os.path.join(savedir, 'train', f'{train_id}.npy'), source[:,i*window:i*window+window])
        train_id += 1
        
        
print('splitting testset...')
for track in tqdm(test_mus):
    # downsample
    source = (track.stems[:,::2, 0] + track.stems[:, ::2, 1]) / 2
    for i in range(0, source.shape[1]//window):
        np.save(os.path.join(savedir, 'test', f'{test_id}.npy'), source[:,i*window:i*window+window])
        test_id += 1
        
