import numpy as np

def planar_yaw(poses):
    vertical=np.median(poses[:,:3,1],axis=0);vertical/=np.linalg.norm(vertical)
    heading=poses[:,:3,2].copy();heading-=np.outer(heading@vertical,vertical)
    forward=heading[0]/np.linalg.norm(heading[0]);right=np.cross(vertical,forward);right/=np.linalg.norm(right)
    return np.unwrap(np.arctan2(heading@right,heading@forward))
