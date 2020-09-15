# library
import os
import time
import shutil
from tqdm import tqdm 
import warnings
import numpy as np
import scipy.io as sio
import tensorflow as tf
from numpy import linalg as LA
# scripts
import util

######## Iterative Methods #######

def RARE(dObj, rObj, tau = 0.001, numIter=100, step=100, beta=1e-2, Lipz_total=1, backtracking=True, 
         backtotl=1e-5,  accelerate=False, mode='RED', useNoise=True, is_save=True,save_path='result', 
         xref=None, xinit=None, clip=False, if_complex='complex', save_iter=5):
    """
    Regularized by artifacts removal methods with switch for RED, PGM, Grad

    ### INPUT:
    dObj           ~ data fidelity term, measurement/forward model
    rObj           ~ regularizer term
    tau            ~ control regularizer strength
    numIter        ~ total number of iterations
    step           ~ step-size
    beta           ~ stoping criterion
    Lipz_total     ~ Lipz value of fowardmodel
    backtracking   ~ backtracking linesearch
    backtotl       ~ tolerance
    accelerate     ~ acceleration or not 
    mode           ~ RED update, PROX, or Grad update
    useNoise.      ~ CNN predict noise or image
    is_save        ~ if true save the reconstruction of each iteration
    save_path      ~ the save path for is_save
    xref           ~ the CS2000 of the image, for tracking purpose
    if_complex     ~ Use complex number
    ### OUTPUT:
    x     ~ reconstruction of the algorithm
    """

    ##### HELPER FUNCTION #####

    evaluateTol = lambda x, xnext: np.linalg.norm(x.flatten('F')-xnext.flatten('F'))/np.linalg.norm(x.flatten('F'))
    evaluateSx = lambda s_step: 1/Lipz_total * (dObj.grad(s_step, if_complex) + tau * rObj.deartifacts_A2A(s_step,useNoise=useNoise,clip=clip))

    ##### INITIALIZATION #####

    # initialize save foler
    if is_save:
        abs_save_path = os.path.abspath(save_path)
        if os.path.exists(save_path):
            print("Removing '{:}'".format(abs_save_path))
            shutil.rmtree(abs_save_path, ignore_errors=True)
        # make new path
        print("Allocating '{:}'".format(abs_save_path))
        os.makedirs(abs_save_path)

    #initialize info data
    if xref is not None:
        xrefSet = True
        rPSNR = []
    else:
        xrefSet = False

    dist = []
    timer = []
    relativeChange = []
    norm_d = []
    norm_Sx = []
    
    # initialize variables
    if xinit is not None:
        pass
    else:    
        xinit = np.zeros(dObj.sigSize, dtype=np.complex64) if if_complex else np.zeros(dObj.sigSize, dtype=np.float64)
    # outs = struct(xrefSet)
    x = xinit
    s = xinit            # gradient update
    t = 1.           # controls acceleration
    p = rObj.init()  # dual variable for TV
    pfull = rObj.init()  # dual variable for TV
    count_show = 0

    #Main Loop#
    
    for indIter in tqdm(range(numIter)):
        timeStart = time.time()
        # get gradient
        if mode == 'RED':
            d = evaluateSx(s)
            xnext = s - step*d
            xnext = np.clip(xnext,0,np.inf) if clip else xnext    # clip to [0, inf]
            norm_d.append(LA.norm(d.flatten('F')))
            dist.append(LA.norm(d.flatten('F')))
        elif mode == 'PROX':
            g = dObj.grad(s)
            xnext, p = rObj.prox(s-step*g, step, p, clip, tau = tau)   
        elif mode == 'GRAD':
            g = dObj.grad(s,if_complex)
            xnext = s-step*g
            xnext = np.clip(xnext,0,np.inf) if clip else xnext
        timeEnd = time.time() - timeStart
        timer.append(timeEnd)
        if indIter == 0:
            relativeChange.append(np.inf)
        else:
            relativeChange.append(evaluateTol(x, xnext))        
        # ----- backtracking (damping) ------ #
        if backtracking is True:
            d_update = evaluateSx(xnext)
            while LA.norm(d_update.flatten('F')) > LA.norm(d.flatten('F')) and step >= backtotl:

                step = beta * step
                xnext = s - step*d   # clip to [0, inf]
                d_update = evaluateSx(xnext)

                if step <= backtotl:
                    print('\n','Reach to the backtotl, return x!','\n')
                    return x

        if xrefSet:
            rPSNR.append(util.cal_rPSNR(xref, x))

        # acceleration
        if accelerate:
            tnext = 0.5*(1+np.sqrt(1+4*t*t))
        else:
            tnext = 1
        s = xnext + ((t-1)/tnext)*(xnext-x)

        # update
        t = tnext
        x = xnext

        #save & print
        if is_save and (indIter) % (save_iter) == 0:
            x_temp = np.abs(xnext[160:480,160:480,:])
            x_temp = np.flip(x_temp,axis=1)
            length = int(x_temp.shape[2]/2)
            img_save_1 = x_temp[:,:,0]
            img_save_2 = x_temp[:,:,length]
            for zz in range(1,length):
                img_save_1 = np.concatenate((img_save_1,x_temp[:,:,zz]),axis=1)
                img_save_2 = np.concatenate((img_save_2,x_temp[:,:,zz+length]),axis=1)
            img_save =np.concatenate((img_save_1,img_save_2),axis=0)
            util.save_mat(xnext, abs_save_path+'/iter_{}.mat'.format(indIter+1))
            img_save = np.clip(img_save,0,np.inf)
            util.save_img(img_save, abs_save_path+'/iter_{}_img.tif'.format(indIter+1))
 
    return x