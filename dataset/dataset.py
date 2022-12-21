import torch
import numpy as np
import os
from torch.utils.data import Dataset
from dataset.utils import augment_gain, augment_channelswap




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
    def __init__(self, rate, root='/home/lego/NAS189/Database/musdb18/', segment_length_in_sec=5):
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
        #inst = {'drum':1, 'bass':2, 'other':3, 'vocal':4}[inst]

class MUSDB18_vocal():
    def __init__(self, subsets, split=None, root='/home/lego/NAS189/home/MUSDB18/vocal_accomp_44100',):
        self.root = os.path.join(root, subsets, split or '')
        self.subsets = subsets
        self.split = split
        self.seq_len = 100*44100
    def __len__(self):
        return 86 if self.subsets == 'train' else 50
    
    def __getitem__(self, idx):
        # [2, n_sample]
        # [vocal/accompany, n_sample]
        

        wav = np.load(f'{self.root}/{idx}.npy')
        if wav.shape[-1] < self.seq_len:
            topad = self.seq_len - wav.shape[-1]
            wav = np.pad(wav, ((0,0),(0,0), (0,topad)), 'constant', constant_values=0)
            assert wav.shape[-1] == self.seq_len
        else:
            start = torch.randint(0,wav.shape[-1]-self.seq_len, (1,))
            wav = wav[...,start:start+self.seq_len]
        
        
        return wav

class MUSDB18_vocal_test():
    def __init__(self, subsets='test', root='/home/lego/NAS189/home/MUSDB18/vocal_accomp_44100',):
        self.root = os.path.join(root, subsets)
        self.subsets = subsets
        
    def __len__(self):
        return 50
    
    def __getitem__(self, idx):
        # [2, n_sample]
        # [vocal/accompany, n_sample]
        

        wav = np.load(f'{self.root}/{idx}.npy')
        
        
        
        return wav


class MUSDB18_vocal_seg():
    def __init__(self, subsets, root='/home/lego/NAS189/home/MUSDB18/vocal_accomp_44100',):
        self.subsets = subsets
        self.root = os.path.join(root, subsets, 'seg_20sec')
        self.L = len(os.listdir(self.root))
    def __len__(self):
        return self.L
    def __getitem__(self, idx):
        vocal = np.load(f'{self.root}/{idx}.npy')[0]
        
        if self.subsets == 'test': # valid
            sub_idx = idx
            accomp = np.load(f'{self.root}/{sub_idx}.npy')[1]
        elif self.subsets == 'train': # train
            sub_idx = int(torch.randint(0, self.L, (1,)))
            accomp = np.load(f'{self.root}/{sub_idx}.npy')[1]
            vocal = augmentation(vocal)
            accomp = augmentation(accomp)
        
        mix = vocal + accomp
        return vocal, mix

class MUSDB18_seg():
    def __init__(self, subsets, split=None, root='/home/lego/NAS189/home/MUSDB18/vocal_accomp_44100',):
        self.subsets = subsets
        self.root = os.path.join(root, subsets, split or '','seg_20sec')
        self.L = len(os.listdir(self.root))
    def __len__(self):
        return self.L
    def __getitem__(self, idx):
        wav = np.load(f'{self.root}/{idx}.npy')
        

        # [source, channel, time_length]
        return wav

import pandas as pd
class MUSDB18_length():
    def __init__(self, subsets, split=None, root='/home/lego/NAS189/home/MUSDB18/vocal_accomp_44100',):
        self.subsets = subsets
        self.split= split
        self.root = os.path.join(root, subsets, split or '',)
        self.L = len(os.listdir(self.root))
        self.df = pd.read_csv(os.path.join(self.root,'256.csv'))
        self.seq_length = self.df.iloc[0]['end'] - self.df.iloc[0]['start']
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        series = self.df.iloc[idx]
        wav = np.load(os.path.join(self.root, str(series['songid'])+'.npy'))
        wav = wav[...,series['start']:series['end']]
        if self.seq_length > wav.shape[-1]:
            topad = self.seq_length - wav.shape[-1]
            wav = np.pad(wav, ((0,0),(0,0),(0,topad)), 'constant', constant_values=0)

        return wav

class AmpleGuitar():

    def __init__(self, subset, set1='AGSC' ,set2='AGT', root='/home/lego/NAS189/home/Ample/npy/'):
       
        self.root = root
        self.set1, self.set2 = set1, set2
        self.df1 = pd.read_csv(os.path.join(root, set1, f'{subset}_meta.csv'))
        self.df2 = pd.read_csv(os.path.join(root, set2, f'{subset}_meta.csv'))
        self.rate = 22050
    def __getitem__(self, idx):
        filename = self.df1.iloc[idx]['Filename']

        #vst = self.df1.iloc[idx]['VST']
        group = self.df1.iloc[idx]['VST']

        wav1 = np.load(os.path.join(self.root, self.set1, filename))
        start_time = torch.randint(0, self.rate * 10 - self.rate * 3, (1,))
        wav1 = wav1[start_time:start_time+self.rate*3]

        #s = self.df2.loc[self.df2['VST'] != vst]
        s = self.df2.loc[self.df2['VST'] != group]
        
        s = s.iloc[torch.randint(len(s), (1,))]
        filename = s['Filename'].item()
        wav2 = np.load(os.path.join(self.root, self.set2, filename))
        start_time = torch.randint(0, self.rate * 10 - self.rate * 3, (1,))
        wav2 = wav2[start_time:start_time+self.rate*3]

        return wav1, wav2
    
    def __len__(self):
        return len(self.df1)

class JSBChorale():
    def __init__(self, mode, root='/home/lego/NAS189/home/JSBChorale/npy_Aiko/'):
        """ARGS:
            mode; mix, or 1, 2, 3, 4, depend on  which track is desired to be outputed"""
        self.root = root
        self.mode = mode
    def __getitem__(self, idx):
        clip = np.load(os.path.join(self.root, str(idx)+'.npy'))
        #clip = np.random.uniform(0.5, 1.25, (4,1)) * clip
        clip = librosa.util.normalize(clip, axis=1)
        mix = clip.sum(0)

        if self.mode=='mix':
            label = (torch.zeros(4)+0.3).bernoulli()
            if not label.bool().any():
                label[torch.randint(4,(1,))] = 1      
            target  = clip[label.bool()].sum(0)
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
        clip = librosa.util.normalize(clip, axis=1)
        #clip = np.random.uniform(0.5, 1.25, (4,1)) * clip
        mix = clip.sum(0)

        if self.mode=='mix':
            label = (torch.zeros(4)+0.3).bernoulli()
            if not label.bool().any():
                label[torch.randint(4,(1,))] = 1      
            target = clip[label.bool()].sum(0)
        else:
            label = np.zeros((4))
            label[self.mode] = 1
            target = clip[self.mode]

        
        return target, mix, label
    
    def __len__(self):
        return sum(self.songs_num)

import soundfile as sf

class OpenMic():
    def __init__(self, split_ratio, root='/home/lego/NAS189/Database/OpenMic/'):
        self.root = root
        self.split_ratio = split_ratio
        self.gt = np.load(os.path.join(root,'openmic-2018.npz'), allow_pickle=True)['Y_mask']
        self.sample_key = np.load(os.path.join(root,'openmic-2018.npz'), allow_pickle=True)['sample_key']
        self.L = len(self.sample_key)
        self.vgg = np.load(os.path.join(root,'openmic-2018.npz'), allow_pickle=True)['X']

    def __getitem__(self, idx):
        gt1, key1 =  self.gt[idx], self.sample_key[idx]
        if idx < self.L*(1-self.split_ratio):
            high = np.floor(self.L*(1-self.split_ratio))
            low = 0
        else:
            high = self.L
            low = np.floor(self.L*(1-self.split_ratio))
        while(1):
            sub_idx = torch.randint(int(low), int(high), (1,))
            gt2, key2 = self.gt[sub_idx], self.sample_key[sub_idx]
            if not gt2[np.where(gt1==True)].any():
                break
        audio1 = np.load(os.path.join('/home/lego/NAS189/home/OpenMic/',key1+'.npy'))
        audio2 = np.load(os.path.join('/home/lego/NAS189/home/OpenMic/',key2+'.npy'))
        window = 22050*3
        start_time1 = torch.randint(0,len(audio1)-window, (1,))
        audio1 = audio1[start_time1:start_time1+window]
        start_time2 = torch.randint(0,len(audio2)-window, (1,))
        audio2 = audio2[start_time2:start_time2+window]
        vgg1 = self.vgg[idx].mean(0)
        vgg2 = self.vgg[sub_idx].mean(0)

        """audio1, sr = sf.read(os.path.join(self.root,'audio',key1[0:3],key1+'.ogg'))
        audio2, sr = sf.read(os.path.join(self.root,'audio',key2[0:3],key2+'.ogg'))
        audio1 = audio1[::2] if len(audio1.shape) == 1 else audio1[::2].mean(-1)
        audio2 = audio2[::2] if len(audio2.shape) == 1 else audio2[::2].mean(-1)
        
        window = 22050*3
        if len(audio1) < window:
            tmp = np.zeros((window))
            tmp[:len(audio1)] = audio1
            audio1 = tmp
        else:
            start_time1 = torch.randint(0,len(audio1)-window, (1,))
            audio1 = audio1[start_time1:start_time1+window]
        if len(audio2) < window:
            tmp = np.zeros((window))
            tmp[:len(audio2)] = audio2
            audio2 = tmp
        else:
            start_time2 = torch.randint(0,len(audio2)-window, (1,))
            audio2 = audio2[start_time2:start_time2+window]"""
        return audio1, audio2, vgg1, vgg2
    def __len__(self):
        return self.L


class OpenMic_Classfication():
    def __init__(self, split_ratio, root='/home/lego/NAS189/Database/OpenMic/'):
        self.root = root
        self.split_ratio = split_ratio
        self.gt = np.load(os.path.join(root,'openmic-2018.npz'), allow_pickle=True)['Y_mask']
        self.sample_key = np.load(os.path.join(root,'openmic-2018.npz'), allow_pickle=True)['sample_key']
        self.vgg = np.load(os.path.join(root,'openmic-2018.npz'), allow_pickle=True)['X']

        self.L = len(self.sample_key)
    def __len__(self):
        return self.L
    
    def __getitem__(self, idx):
        gt1, key1 =  self.gt[idx], self.sample_key[idx]
        audio1, sr = sf.read(os.path.join(self.root,'audio',key1[0:3],key1+'.ogg'))
        audio1 = audio1[::2] if len(audio1.shape) == 1 else audio1[::2].mean(-1)
        window = 22050 * 3
        audio1 = np.pad(audio1, (0,4*window-len(audio1)), 'constant', constant_values=(0,0))
        audio1 = np.array(np.split(audio1,4))
        vgg = self.vgg[idx]
        return audio1, gt1.astype('float32'), vgg#[4, 66150]

def augmentation(x):
    x = torch.Tensor(x)
    #x = augment_gain(x)
    x = augment_channelswap(x)
    return x
