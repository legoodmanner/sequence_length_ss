import torch
import numpy as np
import os
from torch.utils.data import Dataset
from torchaudio_augmentations.apply import RandomApply


class PureMic_mixup(Dataset):
    # rate = 32000
    def __init__(self):
        self.pair_list = self.get_pair_list(config.PureMic_npz_path)
        self.data_root = config.PureMic_data_root
    def __len__(self):
        return len(self.pair_list)
    
    def __getitem__(self, idx):
        id1, id2 = self.pair_list[idx]
        y1 = np.load(os.path.join(self.data_root,id1+'.npy'))
        y2 = np.load(os.path.join(self.data_root,id2+'.npy'))
        return y1, y2
    
    def get_pair_list(self, PureMic_npz_path):
        keys = np.load(PureMic_npz_path)['keys']
        labels = np.load(PureMic_npz_path)['labels']
        head = 0 
        result = []
        perm = []
        for k, l in zip(keys, labels):
            if l[head] == 1:
                perm += [k]
            else:
                result.append(perm)
                perm = []
                head += 1
                
        perm = []
        final = []
        for i in range(len(result)):
            for j in range(len(result[i])):
                perm += [(i,j)]
        for i in range(len(perm)):
            for j in range(len(perm)):
                if perm[i][0] == perm[j][0]:
                    continue
                else:
                    final += [[result[perm[i][0]][perm[i][1]], result[perm[j][0]][perm[j][1]] ]]
        return final
    
import musdb
import random
import librosa

class MUSDB18_mixup(Dataset):
    def __init__(self, rate, root='/home/lego/NAS189/Database/musdb18/', segment_length_in_sec=3):
        self.mus = musdb.DB(root=root)
        self.rate = rate
        self.clip_window = segment_length_in_sec * 44100
    def __getitem__(self, idx):
        track = self.mus[idx]
        
        # random clip
        window_start = random.randint(0,track.stems.shape[1]-self.clip_window)       
        track = track.stems[:, window_start:window_start+self.clip_window, 0].astype('float32')
        
        # randomly pick 2 instruments 
        inst_idxs = random.choices([1,2,3,4],k=2)
        track = track[inst_idxs]
        if not self.rate == 44100:
            inst1 = librosa.resample(track[0], 44100, self.rate)
            inst2 = librosa.resample(track[1], 44100, self.rate)
        else:
            inst1 = track[0]
            inst2 = track[1]
        # resample
        return inst1, inst2
        
    def __len__(self):
        return len(self.mus)
        
class MUSDB18_seg():
    def __init__(self, subset, inst=None, transform=None, root='/home/lego/NAS189/home/MUSDB18/22050_3sec_npy/', mode='single'):
        self.root = os.path.join(root,subset)
        assert os.path.isdir(self.root), f'{subset} data path not exist'
        self.fl = os.listdir(self.root)
        self.fl = [f for f in self.fl if 'npy' in f]
        self.l = len(self.fl)
        self.mode = mode
        self.transform = transform
        self.subset = subset
        if inst is not None:
            inst = {'drum':1, 'bass':2, 'other':3, 'vocal':4}[inst]
        self.inst_idx = inst

    def __len__(self):
        return self.l
    
    def __getitem__(self, idx):
        clip = np.load(f'{self.root}/{idx}.npy')
        clip = np.random.uniform(0.5, 1.25, (5,1)) * clip
        
        if self.mode == 'single':
            if self.inst_idx is None:
                perm_idxs = np.random.permutation([1,2,3,4])
                target_clip = clip[perm_idxs[0]]
                inst_idx = perm_idxs[0]  
            else:
                target_clip = clip[self.inst_idx] 
                inst_idx = self.inst_idx
            mixture_clip = clip[1:5].sum(0)
            
            return target_clip, inst_idx, mixture_clip
        
        elif self.mode == 'pair':
            perm_idxs = np.random.permutation([1,4,])
            target_clip = clip[perm_idxs[[0,1]]]
            inst_idx = perm_idxs[[0,1]]
            return target_clip[0], target_clip[1], #clip[1:5].sum(0)
       
import pandas as pd


class AmpleGuitar():
    def __init__(self, subset, set1='AGSC' ,set2='AGT', root='/home/lego/NAS189/home/Ample/npy/'):
       
        self.root = root
        self.set1, self.set2 = set1, set2
        self.df1 = pd.read_csv(os.path.join(root, set1, f'{subset}_meta.csv'))
        self.df2 = pd.read_csv(os.path.join(root, set2, f'{subset}_meta.csv'))
        self.rate = 22050
    def __getitem__(self, idx):
        filename = self.df1.iloc[idx]['Filename']
        vst = self.df1.iloc[idx]['VST']
        wav1 = np.load(os.path.join(self.root, self.set1, filename))
        start_time = np.random.randint(0, self.rate * 10 - self.rate * 3)
        wav1 = wav1[start_time:start_time+self.rate*3]

        s = self.df2.loc[self.df2['VST'] != vst].sample()
        filename = s['Filename'].item()
        wav2 = np.load(os.path.join(self.root, self.set2, filename))
        start_time = np.random.randint(0, self.rate * 10 - self.rate * 3)
        wav2 = wav2[start_time:start_time+self.rate*3]

        return wav1, wav2
    
    def __len__(self):
        return len(self.df1)

class JSBChorale():
    def __init__(self, mode, root='/home/lego/NAS189/home/JSBChorale/npy_choir/'):
        """ARGS:
            mode; mix, or 1, 2, 3, 4, depend on  which track is desired to be outputed"""
        self.root = root
        self.mode = mode
    def __getitem__(self, idx):
        clip = np.load(os.path.join(self.root, str(idx)+'.npy'))
        clip = np.random.uniform(0.5, 1.25, (4,1)) * clip
        mix = clip.sum(0)

        if self.mode=='mix':
            #label = np.random.binomial(1,0.3,(4))
            label = np.zeros((4))
            label[np.random.randint(0,4)] = 1      
            target  = clip[np.array(label,dtype=bool)].sum(0)
        else:
            label = np.zeros((4))
            label[self.mode] = 1
            target = clip[self.mode]

        
        return target, mix, label
        
    def __len__(self,):
        return len(os.listdir(self.root))


class ChoralSingingDataset():
    def __init__(self, mode, root='/home/lego/NAS189/home/ChoralSingingDataset/'):
        self.root = root
        self.mode = mode
        self.songs_name = os.listdir(root)
        self.songs_num = [len(os.listdir(os.path.join(root, song_name, 'npy'))) for song_name in self.songs_name]
    def __getitem__(self, idx):
        if idx < self.songs_num[0]:
            path = os.path.join(self.root, self.songs_name[0], 'npy')
        elif idx >= self.songs_num[0] and idx < self.songs_num[1] + self.songs_num[0]:
            path = os.path.join(self.root, self.songs_name[1], 'npy')
        else:
            path = os.path.join(self.root, self.songs_name[2], 'npy')
            
        clip = np.load(os.path.join(path, f'{idx}.npy'))
        clip = np.random.uniform(0.5, 1.25, (4,1)) * clip
        mix = clip.sum(0)

        if self.mode=='mix':
            #label = np.random.binomial(1,0.3,(4))
            label = np.zeros((4))
            label[np.random.randint(0,3)] = 1  
            target  = clip[np.array(label,dtype=bool)].sum(0)
        else:
            label = np.zeros((4))
            label[self.mode] = 1
            target = clip[self.mode]

        
        return target, mix, label
    
    def __len__(self):
        return sum(self.songs_num)
