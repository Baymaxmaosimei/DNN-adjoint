## generate topology pattern from opticla response

import os
from tkinter import Variable
from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.FC = nn.Sequential(
            nn.Embedding(2,100),
            nn.Linear(100, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 2*15*15),
        )
    
    def Fouriertrans(self, x):
        real = x[:,0]
        imag = x[:,1]
        z=torch.complex(real,imag)
        z=torch.reshape(z,(x.size(0),15,15))
        pad = nn.ZeroPad2d(padding=(43,43,43,43))
        img = pad(z)
        img = torch.fft.ifftshift(img,(-2,-1))
        img = torch.real(torch.fft.ifft2(img,[101,101]))
        
        return img
        
    def forward(self, x):
        x = self.FC(x)
        x = x.view(x.size(0),2,15*15)
        output = self.Fouriertrans(x)
        output = torch.tanh(output)
        return output

