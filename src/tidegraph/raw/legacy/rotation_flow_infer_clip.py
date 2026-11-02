import numpy as np

def fit_rotation(a,b,rng,threshold):
    # Row-vector convention b = a @ R.T. RANSAC handles moving water/boats.
    count=len(a)
    if count<8:return np.eye(3),np.zeros(count,dtype=bool)
    samples=np.stack([rng.choice(count,3,replace=False) for _ in range(128)])
    h=np.einsum('nki,nkj->nij',b[samples],a[samples])
    u,_,vt=np.linalg.svd(h);d=np.tile(np.eye(3),(len(samples),1,1));d[:,2,2]=np.linalg.det(u@vt);rs=u@d@vt
    errors=np.linalg.norm(np.einsum('rij,nj->rni',rs,a)-b[None],axis=-1)
    inliers=errors<threshold; idx=np.argmax(inliers.sum(1)); mask=inliers[idx]
    if mask.sum()<6:return rs[idx],mask
    for _ in range(2):
        u,_,vt=np.linalg.svd(b[mask].T@a[mask]);d=np.diag([1,1,np.linalg.det(u@vt)]);r=u@d@vt
        mask=np.linalg.norm(a@r.T-b,axis=1)<threshold
        if mask.sum()<6:break
    return r,mask
