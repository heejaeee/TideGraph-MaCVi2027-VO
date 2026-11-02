import numpy as np
import cv2

CFG=dict(ransac_iterations=512,pixel_threshold=1.5,min_matches=40,max_matches=240,
    fit_inliers=24,fit_fraction=.4,held_min=8,held_fraction=.5,positive_fraction=.7,
    parallax_deg=.5,horizontal_fraction=.7,agreement_deg=3.,min_cells=4,min_x_fraction=.1)

def normalized(p,k):return np.c_[p,np.ones(len(p))]@np.linalg.inv(k).T

def sampson(a,b,r,t,k):
    ra=a@r.T;ea=np.cross(t,ra);eb=np.cross(b,t)@r
    return np.abs((b*ea).sum(1))/np.maximum(np.sqrt((ea[:,:2]**2).sum(1)+(eb[:,:2]**2).sum(1)),1e-12)*np.sqrt(k[0,0]*k[1,1])

def positive(a,b,r,t):
    x=cv2.triangulatePoints(np.c_[np.eye(3),np.zeros(3)],np.c_[r,t],a[:,:2].T,b[:,:2].T)
    good=abs(x[3])>1e-12;xyz=np.zeros((len(a),3));xyz[good]=x[:3,good].T/x[3,good,None]
    return good&np.isfinite(xyz).all(1)&(xyz[:,2]>0)&((xyz@r.T+t)[:,2]>0)

def free_solver(p,q,k,seed):
    cv2.setRNGSeed(seed)
    e,mask=cv2.findEssentialMat(p,q,k,method=cv2.RANSAC,prob=.999,threshold=CFG['pixel_threshold'],maxIters=2000)
    if e is None or mask is None:return None
    choices=[]
    for ee in np.asarray(e).reshape(-1,3,3):
        count,r,t,m=cv2.recoverPose(ee,p,q,k,mask=mask.copy());choices.append((int(count),r,t.ravel(),m.ravel()>0))
    _,r,t,inliers=max(choices,key=lambda x:x[0]);return r,t,inliers

def inspect_solution(solution,p,q,fit,held,k,ri,axes,meta):
    if solution is None:return None,'no_fit',{}
    r,t,inliers=solution;a=normalized(p,k);b=normalized(q,k)
    detail=dict(fit_inliers=int(inliers.sum()),fit_fraction=float(inliers.mean()))
    if inliers.sum()<CFG['fit_inliers'] or inliers.mean()<CFG['fit_fraction']:return None,'fit_support',detail
    a_fit,b_fit=a[fit][inliers],b[fit][inliers]
    fit_positive=float(positive(a_fit,b_fit,r,t).mean())
    err=sampson(a[held],b[held],r,t,k);valid=err<=CFG['pixel_threshold']
    detail.update(held_median_px=float(np.median(err)),held_inliers=int(valid.sum()),held_fraction=float(valid.mean()),fit_positive=fit_positive)
    if valid.sum()<CFG['held_min'] or valid.mean()<CFG['held_fraction'] or np.median(err)>CFG['pixel_threshold']:return None,'held_residual',detail
    aa,bb=a[held][valid],b[held][valid];pos=positive(aa,bb,r,t)
    detail['held_positive']=float(pos.mean())
    if min(fit_positive,pos.mean())<CFG['positive_fraction']:return None,'cheirality',detail
    aa=aa[pos]@r.T;bb=bb[pos];aa/=np.linalg.norm(aa,axis=1,keepdims=True);bb/=np.linalg.norm(bb,axis=1,keepdims=True)
    parallax=float(np.median(np.rad2deg(np.arctan2(np.linalg.norm(np.cross(aa,bb),axis=1),(aa*bb).sum(1)))))
    detail['parallax_deg']=parallax
    if parallax<CFG['parallax_deg']:return None,'parallax',detail
    fitp=p[fit][inliers];spread=np.ptp(fitp,axis=0)/[meta['width'],meta['height']];cells=len(np.unique(np.floor(fitp/[meta['width'],meta['height']]*[8,4]).astype(int),axis=0))
    detail.update(cells=cells,x_fraction=float(spread[0]))
    if cells<CFG['min_cells'] or spread[0]<CFG['min_x_fraction']:return None,'image_spread',detail
    direction=ri@(-r.T@t);direction/=np.linalg.norm(direction);f,right=axes
    horizontal=float(np.linalg.norm([direction@f,direction@right]));detail['horizontal']=horizontal
    if horizontal<CFG['horizontal_fraction']:return None,'vertical_translation',detail
    weight=min(1,int(inliers.sum())/80)*float(valid.mean())/(1+float(np.median(err)))
    return dict(course=float(np.arctan2(direction@right,direction@f)),weight=weight,direction=direction.tolist()),'accepted',detail
