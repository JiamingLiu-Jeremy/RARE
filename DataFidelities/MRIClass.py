'''
Class for free-breathing 4D MRI reconstruction
Jiaming Liu, CIG, WUSTL, 2019
'''
import sys
import math
import decimal
import numpy as np

from DataFidelities.load_mri import load_mri
from DataFidelities.mtimes import mtimes
from util import *

class MRIClass(object):
    def __init__(self, data_root, slices):
        path_1, path_2 = ['{}/s{}/MCUFFT_Param.mat'.format(data_root, slices), '{}/s{}/sts'.format(data_root, slices)]
        self.param = load_mri(data_root, slices)
        self.y = self.param['param_y']
        self.recon_mcnufft = mtimes(self.param, self.y, adjoint=True)
        self.sigSize = self.recon_mcnufft.shape
    
    def size(self):
        sigSize = self.sigSize
        return sigSize

    def eval(self,x):
        z = x - self.y
        d = 0.5 * np.power(np.linalg.norm(self.y.flatten('F')-z.flatten('F')),2)
        return d
    
    def grad(self, x, mode):
        Hx_y = mtimes(self.param, x, adjoint=False)
        res = Hx_y - self.param['param_y']
        g = mtimes(self.param, res, adjoint=True)
        if mode is 'complex':
            pass
        elif mode is 'real':
            g = g.real
        elif mode is 'imag':
            g = g.imag
        return g

    @staticmethod
    def ftran(path_1, path_2):
        param = load_mri(path_1,path_2)
        y = param['param_y']
        recon_mcnufft = mtimes(param, y, adjoint=True)
        return recon_mcnufft

