"""Exact numerical routines extracted from the accepted implementation.
See evidence/core_sources.json for source hashes and original function names.
"""
import json
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter1d,median_filter
from scipy.spatial.transform import Rotation
from scipy import sparse
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
import osqp

def planar_yaw(poses):
    vertical=np.median(poses[:,:3,1],axis=0);vertical/=np.linalg.norm(vertical)
    heading=poses[:,:3,2].copy();heading-=np.outer(heading@vertical,vertical)
    forward=heading[0]/np.linalg.norm(heading[0]);right=np.cross(vertical,forward);right/=np.linalg.norm(right)
    return np.unwrap(np.arctan2(heading@right,heading@forward))

def relative_window_yaw(source,key,n):
    """Fuse same-frame relative rotations before integrating; no overlap seam jumps."""
    if not (source/'windows'/key).exists():
        source=Path(json.loads((source/'diagnostics'/f'{key}.json').read_text())['arguments']['output'])
    increments=[[] for _ in range(n-1)]
    for file in sorted((source/'windows'/key).glob('*.npz')):
        with np.load(file) as z:
            start=int(z['start']); rot=z['c2w'][:,:3,:3]
        delta=Rotation.from_matrix(rot[:-1].transpose(0,2,1)@rot[1:]).as_rotvec()
        for j,v in enumerate(delta): increments[start+j].append(v)
    if any(not x for x in increments):raise ValueError('Incomplete VGGT windows')
    poses=np.tile(np.eye(4),(n,1,1))
    for i,values in enumerate(increments):
        poses[i+1,:3,:3]=poses[i,:3,:3]@Rotation.from_rotvec(np.median(values,axis=0)).as_matrix()
    return planar_yaw(poses)

def chain_and_edges(source,key,n):
    windows=[];increments=[[] for _ in range(n-1)]
    for p in sorted((source/f'windows/{key}').glob('*.npz')):
        with np.load(p) as z:start=int(z['start']);r=z['c2w'][:,:3,:3]
        windows.append((start,r))
        dr=Rotation.from_matrix(r[:-1].transpose(0,2,1)@r[1:]).as_rotvec()
        for j,v in enumerate(dr):increments[start+j].append(v)
    assert all(len(v)>0 for v in increments),key
    R=np.tile(np.eye(3),(n,1,1))
    for i,rr in enumerate(increments):R[i+1]=R[i]@Rotation.from_rotvec(np.median(rr,axis=0)).as_matrix()
    vertical=np.median(R[:,:,1],axis=0);vertical/=np.linalg.norm(vertical)
    f=R[0,:,2]-vertical*(R[0,:,2]@vertical);f/=np.linalg.norm(f)
    right=np.cross(vertical,f);right/=np.linalg.norm(right)
    base=np.unwrap(np.arctan2(R[:,:,2]@right,R[:,:,2]@f))
    ii=[];jj=[];zz=[];window_ids=[]
    for wid,(start,local) in enumerate(windows):
        aligned=R[start]@local[0].T@local
        y=np.unwrap(np.arctan2(aligned[:,:,2]@right,aligned[:,:,2]@f))
        a,b=np.triu_indices(len(y),1);z=y[b]-y[a];i,j=a+start,b+start
        z+=2*np.pi*np.round(((base[j]-base[i])-z)/(2*np.pi))
        ii.extend(i);jj.extend(j);zz.extend(z);window_ids.extend([wid]*len(i))
    return base,np.asarray(ii),np.asarray(jj),np.asarray(zz),np.asarray(window_ids)

def solve_yaw(n,i,j,z,initial,delta_deg=3):
    """Huber IRLS of theta[j]-theta[i]=z; theta[0] fixed to zero."""
    ne=len(i);idx=np.arange(ne)
    A=coo_matrix((np.r_[-np.ones(ne),np.ones(ne)],(np.r_[idx,idx],np.r_[i,j])),shape=(ne,n)).tocsr()[:,1:]
    x=initial[1:]-initial[0];w=np.ones(ne);delta=np.deg2rad(delta_deg)
    for it in range(30):
        old=x.copy();Aw=A.multiply(w[:,None]);x=spsolve((A.T@Aw).tocsc(),A.T@(w*z))
        residual=A@x-z;w=np.minimum(1.,delta/np.maximum(abs(residual),1e-12))
        if np.max(abs(x-old))<1e-9:break
    assert np.isfinite(x).all()
    return np.r_[0.,x],dict(edges=ne,iterations=it+1,downweighted_fraction=float(np.mean(w<1-1e-8)),
        residual_median_deg=float(np.rad2deg(np.median(abs(residual)))),residual_p95_deg=float(np.rad2deg(np.quantile(abs(residual),.95))),
        final_step_rad=float(np.max(abs(x-old))),converged=bool(np.max(abs(x-old))<1e-9))

def integrate(t,y,s):
    v=np.c_[np.sin(y),np.zeros(len(y)),np.cos(y)]*s[:,None]
    return np.vstack([np.zeros(3),np.cumsum(.5*(v[:-1]+v[1:])*np.diff(t)[:,None],axis=0)])

def prox(y,lam,order):
    if lam==0:return y.copy(),dict(converged=True,iterations=0,projected_gradient_max=0.)
    n=len(y);m=n-order
    dd=sparse.diags([-np.ones(m),np.ones(m)],[0,1],shape=(m,n),format='csc') if order==1 else sparse.diags([np.ones(m),-2*np.ones(m),np.ones(m)],[0,1,2],shape=(m,n),format='csc')
    solver=osqp.OSQP();solver.setup(P=sparse.triu(dd@dd.T).tocsc(),q=-(dd@y),A=sparse.eye(m,format='csc'),l=-lam*np.ones(m),u=lam*np.ones(m),eps_abs=1e-7,eps_rel=1e-8,max_iter=100000,polishing=True,verbose=False)
    result=solver.solve();assert result.info.status_val==1,(result.info.status,result.info.iter)
    u=result.x;x=y-dd.T@u;pg=float(np.max(abs(u-np.clip(u+dd@x,-lam,lam))))
    assert pg<2e-5,(order,lam,pg)
    return x,dict(converged=True,iterations=int(result.info.iter),projected_gradient_max=pg,solver='OSQP_dual',solver_version=osqp.__version__)

def apply(t,water,method,param):
    # Fixed 2Hz grid; time interpolation handles native midpoint/irregular grids.
    tt=np.linspace(t[0],t[-1],max(3,int(round((t[-1]-t[0])*2))+1));y=np.interp(tt,t,water);mean=float(y.mean())
    if mean<=1e-3*max(float(np.mean(abs(y))),1e-8):return np.ones(len(t)),dict(fallback=True,converged=True,iterations=0)
    y/=mean
    if param==0:x=y;diag=dict(converged=True,iterations=0)
    elif method in ['tv','trend']:x,diag=prox(y,param,1 if method=='tv' else 2)
    else:
        dt=float(tt[1]-tt[0]);size=max(3,int(round(param/dt))//2*2+1);x=gaussian_filter1d(median_filter(y,size=size,mode='nearest'),2/dt,mode='nearest');diag=dict(converged=True,iterations=0,median_window_samples=size,gaussian_sigma_s=2)
    s=np.interp(t,tt,x);m=float(s.mean());fallback=m<=1e-3*max(float(np.mean(abs(s))),1e-8)
    return (np.ones(len(t)) if fallback else s/m),dict(diag,fallback=fallback)
