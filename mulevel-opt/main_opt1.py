### 2022-02-23 by Simei Mao
### This is the main file


import torch
import torch.nn as nn
from DNN_net_para157 import Generator
from calFDTD_para2 import calFDTD
from scipy.io import loadmat, savemat
import numpy as np
import os

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

## define global parameters

if __name__ == '__main__':

    ## Define hyper parameters
    LR = 0.02
    EPOCH = 400
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
    generator = Generator()
    optimizer = torch.optim.Adam(generator.parameters(), lr=LR)
    binary_penalty = 0
    # Define the scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma = 0.99)

    target_T=np.concatenate([np.ones(50),np.zeros(50)])
    target=torch.unsqueeze(Tensor(target_T),dim=0)
    fom_all = []
    latent = [15,7]

    # Train the model
    for epoch in range(EPOCH):

        x = generator(target,latent)
        real = x[:,0]
        imag = x[:,1]
        z=torch.complex(real,imag)
        z=torch.reshape(z,(x.size(0),latent[0],latent[1]))
        aa = int((101-latent[1])/2)
        bb = int((201-latent[0])/2)
        pad = nn.ZeroPad2d(padding=(aa,aa,bb,bb))
        img = pad(z)
        img = torch.fft.ifftshift(img,(-2,-1))
        img = torch.real(torch.fft.ifft2(img,[201,101]))
        gen_image =  torch.tanh(img)

        fom, fom_r, gradients, dis_loss = callFDTD.callable_fom_grad(gen_image, target_T)
        f_loss = torch.sum(torch.mean(gen_image*Tensor(gradients),dim=0))
        bianry_loss = torch.mean(torch.abs(gen_image)*(2-torch.abs(gen_image)))
        binary_penalty = epoch/EPOCH
        # binary_penalty = 0 if epoch<200 else epoch/EPOCH
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
            state = {'model': generator.state_dict(), 'optimizer': optimizer.state_dict(),
                     'scheduler': scheduler.state_dict(), 'epoch': epoch}
            torch.save(state, os.path.join(callFDTD.workingDir, 'generator.pth'))

        # print('Epoch: ', epoch, '| total fom: %.4f, bia_loss: %.4f, fom: %s, dis_loss: %.4f' %(np.mean(fom), bianry_loss, fom, np.mean(dis_loss)))
        print('Epoch: ', epoch, '| total fom: %.4f, bia_loss: %.4f, dis_loss: %.4f' %(np.mean(fom), bianry_loss, np.mean(dis_loss)))


    # save model and results
    savemat(os.path.join(callFDTD.workingDir,'results.mat'), {'target_all':target_T, 'fom_all':fom_all, 'fomHist':fomHist, 'paramHist':paramHist, 'binaHist':binaHist})
    state = {'model': generator.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
             'epoch': epoch}
    torch.save(state, os.path.join(callFDTD.workingDir, 'generator.pth'))






