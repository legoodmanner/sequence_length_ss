import torch
import torch.nn as nn
import torch.nn.functional as F

class LinearReLU(nn.Module):
    def __init__(self, latent_size, input_size):
        super(LinearReLU, self).__init__()
        self.latent_size = latent_size
        self.l1 = nn.Linear(input_size + latent_size, input_size, bias=True)
        self.l2 = nn.Linear(input_size, input_size, bias=False)
        self.l3 = nn.Linear(input_size, input_size, bias=True)
        self.l4 = nn.Linear(input_size, latent_size, bias=False)

    
    def forward(self, input, c):
        h = torch.cat([input, c], dim=-1)
        h = F.relu_(self.l1(h))
        h = F.relu_(self.l2(h))
        h = self.l3(h)
        r = F.sigmoid(self.l4(h))
        r = torch.cat([r * input, (1-r) * input])
        return r.view(-1, 2 ,self.latent_size)
    
    
    
        