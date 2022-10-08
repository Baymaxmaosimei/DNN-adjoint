### 2022-02-23 by Simei Mao
### This is the main file


import torch
import torch.nn as nn
from DNN_net_para157 import Generator
from DNN_net_para330 import Generator2
from calFDTD_para2 import calFDTD
from scipy.io import loadmat, savemat
import numpy as np
import random
import os

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

## define global parameters

if __name__ == '__main__':

    ## Define hyper parameters
    LR = 0.02
    EPOCH = 1000
    fomHist = []
    paramHist = []
    binaHist = []

    # initialize the FDTD
    size_x = 4000
    size_y = 2000
    size_z = 220
    x_points=int(size_x/20)+1
    y_points=int(size_y/20)+1
    z_points=int(size_z/40)+1
    x_pos = np.linspace(-size_x/2*1e-9,size_x/2*1e-9,x_points)
    y_pos = np.linspace(-size_y/2*1e-9,size_y/2*1e-9,y_points)
    z_pos = np.linspace(-size_z/2*1e-9,size_z/2*1e-9,z_points)
    xyz = [x_pos,y_pos,z_pos]
    beta = 1
    script = os.path.join(os.path.dirname(__file__), 'Efilter_3D.fsp')
    wavelengths = np.linspace(1260e-9,1360e-9,100)
    callFDTD = calFDTD(base_script=script, xyz=xyz, beta=beta, wavelengths=wavelengths)
    callFDTD.initialize()

    ## Define the DNN models and the optimizer
    generator1 = Generator()
    generator2 = Generator2(330)
    optimizer = torch.optim.Adam(generator2.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma = 0.99)
    checkpoint = torch.load(os.path.join(os.path.dirname(__file__), 'generator.pth'))
    generator1.load_state_dict(checkpoint['model'])

    target_T=np.concatenate([np.ones(50),np.zeros(50)])
    target=torch.unsqueeze(Tensor(target_T),dim=0)
    fom_all = []
    latent1 = [15,7]
    latent2 = [29,15]
    # Train the model
    for epoch in range(EPOCH):

        x1 = generator1(target,latent1)
        x2 = generator2(target)
        z1 = torch.complex(x1[:,0], x1[:,1])
        z1 = torch.reshape(z1,(x1.size(0),latent1[0],latent1[1]))
        z2 = torch.complex(x2[:,0], x2[:,1])
        tl = torch.reshape(z2[:,0:49],(z2.size(0),7,7))
        tr = torch.reshape(z2[:,49:98],(z2.size(0),7,7))
        tu = torch.reshape(z2[:,98:214],(z2.size(0),29,4))
        td = torch.reshape(z2[:,214:330],(z2.size(0),29,4))
        z = torch.cat((tl,z1,tr),1)
        z = torch.cat((tu,z,td),2)
        
        aa = int((101-latent2[1])/2)
        bb = int((201-latent2[1])/2)
        pad = nn.ZeroPad2d(padding=(aa,aa,bb,bb))
        img = pad(z)
        img = torch.fft.ifftshift(img,(-2,-1))
        img = torch.real(torch.fft.ifft2(img,[201,101]))
        gen_image =  torch.tanh(img)

        fom, fom_r, gradients, dis_loss = callFDTD.callable_fom_grad(gen_image, target_T)
        f_loss = torch.sum(torch.mean(gen_image*Tensor(gradients),dim=0))
        bianry_loss = torch.mean(torch.abs(gen_image)*(2-torch.abs(gen_image)))
        binary_penalty = epoch/EPOCH
        g_loss = f_loss - bianry_loss* binary_penalty
 
        fomHist.append(fom)
        paramHist.append(gen_image.data.numpy())
        binaHist.append(dis_loss)
        fom_all.append(fom_r)

        optimizer.zero_grad()
        g_loss.backward()
        optimizer.step()
        scheduler.step()

        if epoch%10==0:
            savemat(os.path.join(callFDTD.workingDir,'results.mat'), {'target_all':target_T, 'fom_all':fom_all, 'fomHist':fomHist, 'paramHist':paramHist, 'binaHist':binaHist})
            state = {'model': generator2.state_dict(), 'optimizer': optimizer.state_dict(),
                     'scheduler': scheduler.state_dict(), 'epoch': epoch}
            torch.save(state, os.path.join(callFDTD.workingDir, 'generator2.pth'))

        # print('Epoch: ', epoch, '| total fom: %.4f, bia_loss: %.4f, fom: %s, dis_loss: %.4f' %(np.mean(fom), bianry_loss, fom, np.mean(dis_loss)))
        print('Epoch: ', epoch, '| total fom: %.4f, bia_loss: %.4f, dis_loss: %.4f' %(np.mean(fom), bianry_loss, np.mean(dis_loss)))


    # save model and results
    savemat(os.path.join(callFDTD.workingDir,'results.mat'), {'target_all':target_T, 'fom_all':fom_all, 'fomHist':fomHist, 'paramHist':paramHist, 'binaHist':binaHist})
    state = {'model': generator2.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
             'epoch': epoch}
    torch.save(state, os.path.join(callFDTD.workingDir, 'generator.pth'))






