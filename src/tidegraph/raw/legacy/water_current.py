"""Infer a constant surface-current nuisance vector using the vessel's forward-motion prior."""
import numpy as np

def estimate_current(velocity,heading,down,max_ratio=.5,reject_large=False):
    side=np.cross(np.broadcast_to(down,heading.shape),heading)
    side/=np.maximum(np.linalg.norm(side,axis=1,keepdims=True),1e-8)
    forward=heading[0];right=side[0];basis=np.stack([right,forward],axis=1)
    A=side@basis;b=-np.sum(side*velocity,axis=1)
    scale=max(float(np.median(np.linalg.norm(velocity,axis=1))),1e-6)
    weights=np.ones(len(A));current=np.zeros(2)
    for _ in range(8):
        sw=np.sqrt(weights)
        current=np.linalg.lstsq(A*sw[:,None],b*sw,rcond=.1)[0]
        residual=A@current-b
        delta=max(np.median(np.abs(residual))*1.5,.02*scale)
        weights=1/np.maximum(1,np.abs(residual)/delta)
    current=basis@current
    norm=np.linalg.norm(current)
    if norm>max_ratio*scale:
        if reject_large:current[:]=0
        else:current*=max_ratio*scale/norm
    residual=np.sum(side*(velocity+current),axis=1)
    return current,residual
