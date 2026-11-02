import numpy as np
from scipy.ndimage import gaussian_filter1d

MODES=['wide','zoom','flir-IR','flir-VIS','sionyx','infrared']

def inputs(z):
    f=z['features'].astype(float);center=f-f.mean(0)
    relative=np.clip(center/np.maximum(f.std(0),1e-6),-10,10)
    mode=np.zeros((len(f),len(MODES)));mode[:,MODES.index(str(z['camera_mode']))]=1
    return np.c_[f,center,relative,mode]

def normalize_speed(p,kind):
    p=np.maximum(np.exp(p)-.05 if kind=='log' else p,0)
    p=gaussian_filter1d(p,2.5)
    return p/max(p.mean(),1e-6)
