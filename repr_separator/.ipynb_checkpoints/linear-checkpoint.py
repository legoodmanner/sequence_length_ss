import torch
import torch.nn as nn
import torch.nn.functional as F

class LinearReLU(nn.Module):
    def __init__(self):
        super(LinearReLU, self).__init__()
        self.l1 = nn.Linear(2048, 4096, bias=False)
        self.l2 = nn.Linear(4096, 4096, bias=False)
        self.l3 = nn.Linear(4096, 4096, bias=False)
        self.l4 = nn.Linear(4096, 4096, bias=False)

    
    def forward(self, input):
        h = F.relu_(self.l1(input))
        h = F.relu_(self.l2(h))
        h = self.l3(h)
        r = self.l4(h)
        return r.view(-1,2,2048)
    
    
    
        