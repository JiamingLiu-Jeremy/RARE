# tf_unet is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# tf_unet is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with tf_unet.  If not, see <http://www.gnu.org/licenses/>.


'''
Modified on Feb, 2018 based on the work of jakeret

author: yusun
'''
from __future__ import print_function, division, absolute_import, unicode_literals
from scipy.optimize import fminbound
import numpy as np
import scipy.io as sio
import scipy.misc as smisc
import imageio



evaluatepsnr = lambda xtrue, x: 10*np.log10(1/np.mean((xtrue.flatten('F')-x.flatten('F'))**2))


def to_rgb(img):
    """
    Converts the given array into a RGB image. If the number of channels is not
    3 the array is tiled such that it has 3 channels. Finally, the values are
    rescaled to [0,255) 
    
    :param img: the array to convert [nx, ny, channels]
    
    :returns img: the rgb image [nx, ny, 3]
    """
    img = np.atleast_3d(img)
    channels = img.shape[2]
    if channels < 3:
        img = np.tile(img, 3)
    
    img[np.isnan(img)] = 0
    img -= np.amin(img)
    img /= np.amax(img)
    img *= 255
    return img

def to_double_cnn(img_norm):
    img = img_norm.copy()
    if len(img.shape) == 3: # img.shape = nx*ny*nz
        img[np.isnan(img)] = 0
        img_amin = np.tile(np.amin(img,axis=(0,1),keepdims=True),[img.shape[0],img.shape[1],1])
        img -= img_amin
        img_amax = np.tile(np.amax(img,axis=(0,1),keepdims=True),[img.shape[0],img.shape[1],1])
        img /= img_amax
    else:
        print('Incorrect img.shape')
        exit()
    return img,img_amin,img_amax

def to_double(img, clip=False):
    
    img_norm = img.copy()
    img_norm = np.clip(img_norm,0,np.inf) if clip else img_norm
    if len(img.shape) == 3: # img.shape = nx*ny*nz
        img_norm[np.isnan(img_norm)] = 0
        img_norm_amin = np.amin(img_norm,keepdims=True)
        img_norm -= img_norm_amin
        img_norm_amax = np.amax(img_norm, keepdims=True)
        img_norm /= img_norm_amax
    else:
        img_norm[np.isnan(img_norm)] = 0
        img_norm_amin = np.amin(img_norm,keepdims=True)
        img_norm -= img_norm_amin
        img_norm_amax = np.amax(img_norm, keepdims=True)
        img_norm /= img_norm_amax

    return img_norm, img_norm_amin, img_norm_amax

def save_mat(img, path):
    """
    Writes the image to disk
    
    :param img: the rgb image to save
    :param path: the target path
    """
    
    sio.savemat(path, {'img':img})


def save_img(img, path):
    """
    Writes the image to disk
    
    :param img: the rgb image to save
    :param path: the target path
    """
    # img = img[:,:,1]
    img = to_rgb(img)
    imageio.imwrite(path, img.round().astype(np.uint8))

def addwagon(x,inputSnr):
    noiseNorm = np.linalg.norm(x.flatten('F')) * 10^(-inputSnr/20)
    xBool = np.isreal(x)
    real = True
    for e in np.nditer(xBool):
        if e == False:
            real = False
    if (real == True):
        noise = np.random.randn(np.shape(x)[0],np.shape(x)[1])
    else:
        noise = np.random.randn(np.shape(x)[0],np.shape(x)[1]) + 1j * np.random.randn(np.shape(x)[0],np.shape(x)[1])
    
    noise = noise/np.linalg.norm(noise.flatten('F')) * noiseNorm
    y = x + noise
    return y, noise

def optimizeTau(x, algoHandle, taurange, maxfun=20):
    # maxfun ~ number of iterations for optimization

    # evaluateSNR = lambda x, xhat: 20*np.log10(np.linalg.norm(x.flatten('F'))/np.linalg.norm(x.flatten('F')-xhat.flatten('F')))
    evaluatepsnr = lambda xtrue, x: 10*np.log10(1/np.mean((xtrue.flatten('F')-x.flatten('F'))**2))
    fun = lambda tau: -cal_psnr(x,algoHandle(tau)[0])
    tau = fminbound(fun, taurange[0],taurange[1], xtol = 1e-6, maxfun = maxfun, disp = 3)
    return tau

def cal_rPSNR(xref,x, phase=6):

    if len(x.shape) == len(xref.shape):
        x_norm = np.abs(x.copy())
        x_norm,_,_ = to_double(x_norm)
        x_norm = np.abs(x_norm[160:480,160:480])
        x_norm = np.flip(x_norm,axis=1)
        my_psnr = evaluatepsnr(xref[:,:,phase],x_norm[:,:,phase])
    else:
        print(x.shape,xref.shape)
        exit()
    return my_psnr