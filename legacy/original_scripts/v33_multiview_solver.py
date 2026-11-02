"""Generalized multiview joint BA, derived from our V30 solver.
First camera and baseline scale gauge anchored. No trajectory labels used.
"""
from v30_horizon_ba import np,cv2,Rotation,triangulate,project,lil_matrix,least_squares

def bundle(p,r0,c0,k0,focal=False,lines=None,max_nfev=120):
    v,n=p.shape[:2];assert v>=4
    x0=triangulate(p,r0,c0,k0);scale=max(float(np.linalg.norm(c0[-1]-c0[0])),1e-4)
    depth=max(float(np.median(np.linalg.norm(x0-c0[0],axis=1))),scale)
    nc=6*(v-1);npnt=3*n;total=nc+npnt+int(focal);initial=np.r_[np.zeros(nc),x0.ravel(),np.zeros(int(focal))]
    lo=np.full(total,-np.inf);hi=np.full(total,np.inf)
    for j in range(v-1):lo[j*6:j*6+3]=-np.deg2rad(3);hi[j*6:j*6+3]=np.deg2rad(3);lo[j*6+3:j*6+6]=-.5;hi[j*6+3:j*6+6]=.5
    if focal:lo[-1]=np.log(.9);hi[-1]=np.log(1.1)
    lines=lines if lines is not None else [None]*v;valid=[j for j,l in enumerate(lines) if l is not None]
    if len(valid)<2:valid=[]
    def unpack(z):
        delta=z[:nc].reshape(v-1,6);r=r0.copy();c=c0.copy();r[1:]=Rotation.from_rotvec(delta[:,:3]).as_matrix()@r0[1:];c[1:]+=delta[:,3:]*scale
        k=k0.copy();f=float(np.exp(z[-1])) if focal else 1.;k[:,0,0]*=f;k[:,1,1]*=f
        return r,c,k,z[nc:nc+npnt].reshape(n,3),delta,f
    def residual(z):
        r,c,k,x,d,f=unpack(z);uv,depths=project(x,r,c,k)
        # Pixel residuals plus weak pose and landmark priors; retain gauge scale.
        result=[(uv-p).ravel(),(d[:,:3]/np.deg2rad(3)).ravel(),(d[:,3:]/.3).ravel(),((x-x0)/(depth*.5)).ravel(),np.array([(np.linalg.norm(c[-1]-c[0])/scale-1)/.01])]
        if focal:result.append(np.array([np.log(f)/.05]))
        if valid:
            normals=[]
            for j in valid:
                local=k[j].T@lines[j];local/=np.linalg.norm(local);normals.append(r[j]@local)
            reference=r0[valid[0]]@(k[valid[0]].T@lines[valid[0]]);reference/=np.linalg.norm(reference)
            result.append((np.cross(np.array(normals),reference)/np.deg2rad(.5)).ravel())
        return np.concatenate(result)
    count=len(residual(initial));sparsity=lil_matrix((count,total),dtype=int)
    for j in range(v):
        for a in range(n):
            rr=(j*n+a)*2;sparsity[rr:rr+2,nc+a*3:nc+a*3+3]=1
            if j:sparsity[rr:rr+2,(j-1)*6:j*6]=1
            if focal:sparsity[rr:rr+2,-1]=1
    off=2*v*n
    for j in range(v-1):
        sparsity[off+j*3:off+j*3+3,j*6:j*6+3]=np.eye(3,dtype=int)
        sparsity[off+3*(v-1)+j*3:off+3*(v-1)+j*3+3,j*6+3:j*6+6]=np.eye(3,dtype=int)
    off+=nc;sparsity[off:off+npnt,nc:nc+npnt]=np.eye(npnt,dtype=int);off+=npnt
    sparsity[off,(v-2)*6+3:(v-1)*6]=1;off+=1
    if focal:sparsity[off,-1]=1;off+=1
    for j in valid:
        if j:sparsity[off:off+3,(j-1)*6:(j-1)*6+3]=1
        if focal:sparsity[off:off+3,-1]=1
        off+=3
    assert off==count
    sol=least_squares(residual,initial,jac_sparsity=sparsity.tocsr(),bounds=(lo,hi),loss='soft_l1',f_scale=1.,max_nfev=max_nfev,x_scale='jac',ftol=1e-5)
    r,c,k,x,d,f=unpack(sol.x)
    return r,c,k,dict(nfev=sol.nfev,success=bool(sol.success),initial_cost=float(np.mean(residual(initial)[:2*v*n]**2)),final_cost=float(np.mean(residual(sol.x)[:2*v*n]**2)),focal_ratio=f,horizon_views=len(valid))
