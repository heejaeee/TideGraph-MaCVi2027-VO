import numpy as np

def wrap(a):return np.arctan2(np.sin(a),np.cos(a))

def frame_axes(rp):
    up=np.median(rp[:,:3,1],axis=0);up/=np.linalg.norm(up)
    f=rp[0,:3,2].copy();f-=up*(f@up);f/=np.linalg.norm(f)
    return f,np.cross(up,f)
