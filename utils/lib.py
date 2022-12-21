import torch
import mir_eval
import numpy as np
import museval
 

def ideal_mask_generator(target, mixture):
    a = torch.div(target,mixture)
    a = torch.where(torch.isinf(a), torch.full_like(a, 0), a)
    a = torch.where(torch.isnan(a), torch.full_like(a, 0), a)
    return a

def sdr_calc(target, mix, pred_spec, audioGen):
    # pred_spec: [B, n_src, n_channel, freq, frames, cmplx]
    # target [B, n_channel, freq, frames, cmplx]
    # mix [B, n_channel, freq, frames, cmplx]

   
    
    gt_audio = audioGen(target) # B, n_channel, time
    gt_audio = gt_audio.permute(0,2,1).detach().cpu().numpy() + 10e-5 # B, time, n_channel
    gt_audio = gt_audio[:,np.newaxis,...] # B, n_src, time, n_channel

    pred_audio = audioGen(pred_spec) # B, n_src, n_channel, time
    pred_audio = pred_audio.permute(0,1,3,2).detach().cpu().numpy() + 10e-5 # B, n_src, time, n_channel

    sdr = []
    sar = []
    sir = []
    # print(pred_audio.shape, gt_audio.shape)
    for gt, pred in zip(gt_audio, pred_audio):
        sdr_, _, sir_, sar_ = museval.evaluate(
            # references= np.concatenate((gt, acco), axis=0),
            references= gt,
            estimates= pred[[0]],
            
            
        )
        sdr.append(sdr_.item())
        sir.append(sir_.item())
        sar.append(sar_.item())
    return sdr, sar, sir


    
    
    