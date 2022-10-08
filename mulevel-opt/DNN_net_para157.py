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
        self.x1=15
        self.x2=7

        self.FC = nn.Sequential(
            nn.Linear(100, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 2*self.x1*self.x2),
        )
    
    def Fouriertrans(self, x):
        real = x[:,0]
        imag = x[:,1]
        z=torch.complex(real,imag)
        z=torch.reshape(z,(x.size(0),self.x1,self.x2))
        aa = int((101-self.x2)/2)
        bb = int((201-self.x1)/2)
        pad = nn.ZeroPad2d(padding=(aa,aa,bb,bb))
        img = pad(z)
        img = torch.fft.ifftshift(img,(-2,-1))
        img = torch.real(torch.fft.ifft2(img,[201,101]))
        
        return img
        
    def forward(self, x, latent):
        self.x1=latent[0]
        self.x2=latent[1]
        x = self.FC(x)
        x = x.view(x.size(0),2,self.x1*self.x2)
        output = self.Fouriertrans(x)
        output = torch.tanh(output)
        # output = torch.squeeze(output,-3)
        return x

