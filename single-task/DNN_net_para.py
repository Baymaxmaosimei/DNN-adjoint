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
            nn.Linear(100, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 2*29*15),
        )
        self.CONV = nn.Sequential(
            nn.ConvTranspose2d(4, 8, (3,4), 2 ,1),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(8, 16, 4, 2, 1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(16, 32, 5, 2, 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(32, 1, 3, 1, 1),
            nn.Sigmoid()
        )
    
    def Fouriertrans(self, x):
        real = x[:,0]
        imag = x[:,1]
        z=torch.complex(real,imag)
        z=torch.reshape(z,(x.size(0),29,15))
        pad = nn.ZeroPad2d(padding=(43,43,86,86))
        img = pad(z)
        img = torch.fft.ifftshift(img,(-2,-1))
        img = torch.real(torch.fft.ifft2(img,[201,101]))
        
        return img
        
    def forward(self, x):
        x = self.FC(x)
        x = x.view(x.size(0),2,29*15)
        output = self.Fouriertrans(x)
        output = torch.tanh(output)
        # output = torch.squeeze(output,-3)
        return output

