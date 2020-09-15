from DataFidelities.MRIClass import MRIClass
from Regularizers.robjects import *
from iterAlgs import *
from util import *

import os
import numpy as np
import tensorflow as tf
import scipy.io as sio
import matplotlib.pyplot as plt
'''
Python 3.6
tensorflow 1.10~1.13
Windows 10 or Linux
Jiaming Liu (jiaming.liu0327@gmail.com)
github: https://github.com/wustl-cig/RARE
If you have any question, please feel free to contact with me.
(The efficient gpu implementation of the the forward model will release soon.)
Jiaming Liu (jiaming.liu0327@gmail.com)
by Jiaming Liu (16/Jul/2020)
'''
####################################################
####              HYPER-PARAMETERS               ###
####################################################
# Choose Gpu
gpu_ind = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ind # 0,1,2,3

def main():
    ####################################################
    ####              Slice Optimization             ###
    ####################################################
    tau = 0.34
    backtracking = False
    num_spokes = '400'
    reg_mode = 'RED'
    num_slices = [25]
    test_name = 'patient54'
    cnnmodel_name = 'network_A2A'
    phase2show = 6
    stor_root = os.path.join('Results', test_name, reg_mode)
    data_root = os.path.join('data', test_name, num_spokes+'spokes')
    model_root = os.path.join('models', cnnmodel_name+'.h5')
    cs2000 = sio.loadmat('data/cs2000.mat')['gt_norm']
    
    for slices in num_slices:

        save_path = os.path.join(stor_root, 's' + str(slices))
        ## Load forwardmodel
        mriObj = MRIClass(data_root, slices)
        ## MCNUFFT reconstruction
        recon_mcnufft = mriObj.recon_mcnufft
        ## Load 3D-DnCNN as Prior
        rObj = DnCNN3DClass(mriObj.sigSize, model_root)
        ## A2A reconstruction
        recon_a2a = rObj.deartifacts_A2A(recon_mcnufft,useNoise=True)
        ## A2A as initial
        xinit = recon_a2a
        ## start RARE
        recon_rare  = RARE(mriObj, rObj, tau=tau, numIter=9, step=1/(2*tau+1), backtracking=False,
                                accelerate=True, mode=reg_mode, useNoise=False, is_save=True, 
                                save_path=save_path, xref=cs2000, xinit=xinit, clip=False, if_complex='complex', save_iter=1)
        print('\n', 'Finish processing slice: ', slices,'\n')
        
        ## Display the output images
        plot= lambda x: plt.imshow(x,cmap=plt.cm.gray,vmin=0.02,vmax=0.7)
        cal_rPSNR = lambda x: util.cal_rPSNR(cs2000, x, phase=phase2show)

        recon_mcnufft_norm, _, _ = util.to_double(recon_mcnufft, clip=True)
        recon_mcnufft_norm = np.flip(np.abs(recon_mcnufft_norm[160:480, 160:480]), axis=1)[:,:,phase2show]
        rpsnr_mcnufft = cal_rPSNR(recon_mcnufft)

        recon_a2a_norm, _, _ = util.to_double(recon_a2a, clip=True)
        recon_a2a_norm = np.flip(np.abs(recon_a2a_norm[160:480, 160:480]), axis=1)[:,:,phase2show]        
        rpsnr_a2a = cal_rPSNR(recon_a2a)

        recon_rare_norm, _, _ = util.to_double(recon_rare, clip=True)
        recon_rare_norm = np.flip(np.abs(recon_rare_norm[160:480, 160:480]), axis=1)[:,:,phase2show]   
        rpsnr_rare = cal_rPSNR(recon_rare)
        
        plt.clf()
        plt.subplot(1,3,1)
        plot(recon_mcnufft_norm)
        plt.axis('off')
        plt.title('MCNUFFT, rPSNR='+str(rpsnr_mcnufft.round(2))+' dB')
        plt.subplot(1,3,2)
        plot(recon_a2a_norm)
        plt.title('A2A, rPSNR='+str(rpsnr_a2a.round(2))+' dB' )
        plt.axis('off')
        plt.subplot(1,3,3)
        plot(recon_rare_norm)
        plt.title('RARE, rPSNR='+ str(rpsnr_rare.round(2)) +' dB')
        plt.axis('off')
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0,wspace=0)
        plt.show()                

if __name__ == '__main__':
    main()