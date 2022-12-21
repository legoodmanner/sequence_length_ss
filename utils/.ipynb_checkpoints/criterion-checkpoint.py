import torch
import torch.nn as nn
from config import config

class PITLoss(nn.Module):
    def __init__(self, perm_num=2, reduction='mean'):
        super(PITLoss, self).__init__()
        self.perm_num = perm_num
        self.cos = torch.nn.MSELoss()
        self.reduction = reduction
        
    def forward(self, y1, y2, pred):
        b, emb_dim = y1.size()
        batch_loss = torch.zeros(1).cuda(config.cuda)
        for i in range(b):
            l1 = self.cos(pred[i,0], y1[i]) + self.cos(pred[i,1], y2[i])
            l2 = self.cos(pred[i,0], y2[i]) + self.cos(pred[i,1], y1[i])
            
            if l1.data > l2.data:
                batch_loss += l2
            else:
                batch_loss += l1
        
        if self.reduction == 'mean':
            batch_loss /= b
        
        return batch_loss
        
    