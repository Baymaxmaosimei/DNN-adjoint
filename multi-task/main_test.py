### 2022-02-23 by Simei Mao
### This is the main file


import torch
import torch.nn as nn
from DNN_net_para import Generator
from calFDTD_para2 import calFDTD
from scipy.io import loadmat, savemat
import numpy as np
import random
import os

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

## define global parameters

if __name__ == '__main__':

    ## Define hyper parameters
    EPOCH = 100
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
    checkpoint = torch.load(os.path.join(os.path.dirname(__file__), 'generator1000.pth'))
    generator.load_state_dict(checkpoint['model'])

    target_all = []
    fom_all = []
    for epoch in range(EPOCH):

        target_T =[]
        lowpass=np.array(np.zeros(100))
        bandpass=np.array(np.zeros(100))
        lowL=(0.8*np.random.rand(1)+0.1)*100
        lowpass[0:np.int(lowL)]=1
        highpass=1-lowpass
        lifem=np.int((0.7*np.random.rand(1)+0.1)*100)
        rightm=np.int(np.random.rand(1)*(80-lifem)+lifem+10)
        bandpass[lifem:rightm]=1
        bandresist=1-bandpass
        target_T.append(lowpass)
        target_T.append(highpass)
        target_T.append(bandpass)
        target_T.append(bandresist)

        target_all.append(target_T)
        target_T=np.array(target_T)
        target=Tensor(target_T)

        gen_image = generator(target)
        fom, fom_r, dis_loss = callFDTD.callable_fom(gen_image, target_T)
        bianry_loss = torch.mean(torch.abs(gen_image)*(2-torch.abs(gen_image)))

        fomHist.append(fom)
        paramHist.append(gen_image.data.numpy())
        binaHist.append(dis_loss)
        fom_all.append(fom_r)

        print('Epoch: ', epoch, '| total fom: %.4f, bia_loss: %.4f, dis_loss: %.4f' %(np.mean(fom), bianry_loss, np.mean(dis_loss)))
        if epoch%10==0:
            savemat(os.path.join(callFDTD.workingDir,'results.mat'), {'target_all':target_all, 'fom_all':fom_all, 'fomHist':fomHist, 'paramHist':paramHist, 'binaHist':binaHist})

    # save model and results
    savemat(os.path.join(callFDTD.workingDir,'results.mat'), {'target_all':target_all, 'fom_all':fom_all, 'fomHist':fomHist, 'paramHist':paramHist, 'binaHist':binaHist})






