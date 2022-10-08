## generate topology pattern from opticla response

import os
from tkinter import Variable
from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator2(nn.Module):
    def __init__(self, latent):
        super(Generator2, self).__init__()
        self.latent=latent

        self.FC = nn.Sequential(
            nn.Linear(100, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 2*self.latent),
        )

        
    def forward(self, x):
        x = self.FC(x)
        x = x.view(x.size(0),2,self.latent)
        # output = torch.squeeze(output,-3)
        return x

