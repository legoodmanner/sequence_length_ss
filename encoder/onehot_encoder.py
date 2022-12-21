import torch
import torch.nn as nn
import torch.nn.functional as F

class Simple_Encoder(nn.Module):
    def __init__(self, latent_size):
        super(Simple_Encoder, self).__init__()
        self.l1 = nn.Linear(4,128)
        self.l2 = nn.Linear(128, 512)
        self.l3 = nn.Linear(512, latent_size)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(512)
        self.drop = nn.Dropout(0.5)
    def forward(self, x):
        x = F.relu_(self.l1(x))
        x = self.bn1(x)
        x = F.relu_(self.l2(x))
        x = self.bn2(x)
        x = F.relu_(self.l3(x))
        x = self.drop(x)
        return x
