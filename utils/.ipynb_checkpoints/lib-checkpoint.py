import torch
import mir_eval
import numpy as np
from asteroid.losses.sdr import SingleSrcNegSDR

def ideal_mask_generator(target, mixture):
    a = torch.div(target,mixture)
    a = torch.where(torch.isinf(a), torch.full_like(a, 0), a)
    a = torch.where(torch.isnan(a), torch.full_like(a, 0), a)
    return a

def sdr_calc(x, m, mask, audioGen):
    pred_spec = mask * m
    pred_audio = audioGen(pred_spec).detach().cpu().numpy()+ 10e-16
    gt_audio = audioGen(x).detach().cpu().numpy()+ 10e-16

    """sdsdr = SingleSrcNegSDR('sdsdr',)(pred_audio, gt_audio)
    sisdr = SingleSrcNegSDR('sisdr',)(pred_audio, gt_audio)"""
    """non_silent_idx = np.where((abs(gt_audio).mean(-1) > 0.002) & (abs(pred_audio).mean(-1) > 0.02))
    print(len(non_silent_idx))
    gt_audio = gt_audio[non_silent_idx] 
    pred_audio = pred_audio[non_silent_idx] """
    #print(non_silent_idx[0].shape)"""
    sdr, sir, sar, perm = mir_eval.separation.bss_eval_sources(gt_audio, pred_audio, False)
    
    #return sdr, sir, sar
    return np.median(sdr)


    
    
    