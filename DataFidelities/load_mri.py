
import os
import glob
import os.path
import numpy as np
import scipy.io as sio

def load_mri(data_root,slices):
    path_1 = os.path.join(data_root, 's' + str(slices), 'MCUFFT_Param.mat')
    path_2 = os.path.join(data_root, 's' + str(slices), 'sts')
    mncufft = sio.loadmat(path_1,squeeze_me=False)#, mat_dtype=True, struct_as_record=True)
    #####################
    param_e = {
        "b1": mncufft['b1'],
        "adjoint": mncufft['adjoint'],
        "param_y": mncufft['param_y'],
        "dataSize": mncufft['dataSize'],
        "imSize": mncufft['imSize'],
        "w": mncufft['w'],
    }
    #####################
    sts = {}
    files_sts = glob.glob(os.path.join(path_2, '*.mat'))
    files_sts.sort()
    count = 0
    for name in files_sts:

        st = sio.loadmat(name, squeeze_me=True, mat_dtype=True, struct_as_record=True)

        st_temp = {
            "alpha": st['alpha'],
            "beta": st['beta'],
            "Jd": st['Jd'],
            "kb_alf": st['kb_alf'],
            "kb_m": st['kb_m'],
            "Kd": st['Kd'],
            "kernel": st['kernel'],
            "ktype": st['ktype'],
            "M": st['M'],
            "n_shift": st['n_shift'],
            "Nd": st['Nd'],
            "om": st['om'],
            "p" : st['p'],
            "sn": st['sn'],
            "tol ": st['tol'],
        }
        sts["phase_%d"%(count)] = st_temp
        count = count + 1
    param_e['st'] = sts
    return param_e