"""Fit a fixed water-plane normal to visual horizons and estimated camera rotations."""
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation,Slerp

def fit_horizon(poses,pose_times,probe_times,horizons,calibration,size=960):
    scale=size/calibration['width'];fy=calibration['fy']*scale;cy=calibration['cy']*scale;h=round(calibration['height']*scale)
    r=Slerp(pose_times,Rotation.from_matrix(poses[:,:3,:3]))(np.clip(probe_times,pose_times[0],pose_times[-1])).as_matrix()
    down=np.median(r[:,:,1],axis=0);down/=np.linalg.norm(down)
    x=np.cross(down,r[0,:,2]);x/=np.linalg.norm(x);z=np.cross(x,down)
    def predict(q):
        n=down+q[0]*x+q[1]*z
        local=np.einsum('nji,j->ni',r,n)
        return cy-fy*local[:,2]/np.maximum(local[:,1],.1)
    q=least_squares(lambda q:np.r_[(predict(q)-horizons)/h,q*.002],np.zeros(2),bounds=(-.8,.8),loss='soft_l1',f_scale=.02).x
    n=down+q[0]*x+q[1]*z;n/=np.linalg.norm(n)
    return predict(q),n,np.abs(predict(q)-horizons)/h
