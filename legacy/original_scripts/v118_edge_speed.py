"""TRAIN-selected edge-preserving speed filters; VAL used after predictions freeze."""
import argparse
from functools import lru_cache
from pathlib import Path
import numpy as np
from scipy.linalg import cholesky_banded,cho_solve_banded
from scipy.ndimage import median_filter,gaussian_filter1d
from scipy import sparse
import osqp
import v101_protocol as p
import v110_speed_gate as old

ROOT=p.ROOT;OUT=ROOT/'runs/domain_v118_edge_speed';REP=ROOT/'reports/domain_v118_edge_speed'
GRID={'tv':[0,.1,.3,1.,3.,10.,30.],'trend':[0,1.,3.,10.,30.,100.,300.],'median_gaussian':[0,5.,15.,30.,60.]}

def D(x,order):return np.diff(x,n=order)
def DT(z,order):
    x=np.zeros(len(z)+order)
    if order==1:x[:-1]-=z;x[1:]+=z
    else:x[:-2]+=z;x[1:-1]-=2*z;x[2:]+=z
    return x
@lru_cache(maxsize=128)
def factor(n,order,rho):
    # Lower-band storage of I+rho D.T D, with D being first/second differences.
    a=np.zeros((order+1,n));diag=np.zeros(n)
    if order==1:
        diag[:-1]+=1;diag[1:]+=1;a[1,:n-1]=-rho
    else:
        diag[:-2]+=1;diag[1:-1]+=4;diag[2:]+=1
        a[1,:n-2]-=2*rho;a[1,1:n-1]-=2*rho;a[2,:n-2]=rho
    a[0]=1+rho*diag
    return cholesky_banded(a,lower=True,check_finite=False)

def prox_admm(y,lam,order):
    if lam==0:return y.copy(),dict(converged=True,iterations=0,primal_max=0.,stationarity_max=0.)
    x=y.copy();z=D(x,order);u=np.zeros_like(z);rho=1.;converged=False
    for it in range(1,10001):
        x=cho_solve_banded((factor(len(y),order,rho),True),y+rho*DT(z-u,order),check_finite=False)
        dx=D(x,order);previous=z;v=dx+u;z=np.sign(v)*np.maximum(abs(v)-lam/rho,0);u+=dx-z
        primal=dx-z;dual=rho*DT(z-previous,order);pr=float(np.linalg.norm(primal));du=float(np.linalg.norm(dual))
        if np.max(abs(primal))<2e-5 and np.max(abs(x-y+rho*DT(u,order)))<2e-5:
            converged=True;break
        if it%25==0:
            ratio=2. if pr>10*du and rho<4096 else .5 if du>10*pr and rho>1/4096 else 1.
            rho*=ratio;u/=ratio
    diag=dict(converged=converged,iterations=it,primal_max=float(np.max(abs(D(x,order)-z))),stationarity_max=float(np.max(abs(x-y+rho*DT(u,order)))))
    if not converged:raise RuntimeError('ADMM did not converge: '+str(diag))
    return x,diag

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

def checks():
    for order,n in [(1,2),(2,3)]:
        y=np.array([0.,2.]) if order==1 else np.array([0.,2.,0.]);a=np.array([-1.,1.]) if order==1 else np.array([1.,-2.,1.])
        for lam in [.1,1.,10.]:
            expected=y-a*np.clip((a@y)/(a@a),-lam,lam);x,d=prox(y,lam,order);np.testing.assert_allclose(x,expected,atol=8e-5)
    y=np.linspace(.2,1.8,101);x,d=prox(y,30,2);np.testing.assert_allclose(x,y,atol=1e-8)
    for o in [1,2]:
        x,d=prox(np.ones(101),30,o);np.testing.assert_allclose(x,1,atol=1e-8)
    p.save(REP/'solver_checks.json',dict(passed=True,checks='Closed-form two-point TV and three-point trend prox for3lambdas; affine trend and constant preservation. OSQP solved status and projected dual gradient checks.',solver_version=osqp.__version__))

def freeze():
    assert not (REP/'plan.json').exists();files=[Path(__file__),ROOT/'scripts/v110_speed_gate.py']+[old.OUT/f'signals/common/{k}.npz' for k in old.WTRAIN+old.WVAL]
    p.save(REP/'plan.json',dict(created_utc=p.now(),train=old.WTRAIN,val=old.WVAL,grid=GRID,
        definition='Fixed approx2Hz uniform time grid, relative speed mean1. TV objective0.5*sum((x-y)^2)+lambda*sum(abs(diff(x))); l1trend replaces diff with second differences (piecewise linear). Lambda units tied to this fixed sampling. Solve dual box QP with OSQP; solved status and projected gradient<2e-5 required. No clipping signed speeds. Nonpositive mean fallback constant1.',
        solver_revision='Initial hand-written ADMM exceeded10k iterations on TRAIN trend lambda3 at residual2.55e-5. Archived initial code/cache/plan; restarted same grid and objective with OSQP before any VAL evaluation.',solver_version=osqp.__version__,solver_documentation='https://osqp.org/docs/interfaces/python.html',
        median='Nearest-padded centered median with TRAIN-selected window, followed by fixed Gaussian sigma2s. Parameter0 is identity control for every family.',
        selection='Per-family parameter minimizing macro TRAIN46 relative-speed MSE; LOCO selection stability on45 reported with excluded-clip MSE. Final family choice by same TRAIN criterion, no VAL fitting. Existing flow inputs already have6s median and5s Gaussian preprocessing, so this cannot recover information erased upstream.',
        evaluation='Matched common observer TRAIN/VAL primary. Secondary transfer of same fixed parameters onto FINAL V87 speeds, preserving all original routing. Original V87 yaw. Compare21clip mean,wins,maximum harm, and slow/acceleration diagnostics.',
        adoption='Mean improves vs V87,>=11wins and maximum harm<=0.5pp. Separately compare to frozen S_wide60s. User-approved S_wide is generated regardless; these filters are NOT mixed into it.',
        limitations='Existing VAL repeatedly used; not a fresh benchmark. LOCO within TRAIN not verified boat-disjoint. Profile MSE is not trajectory drift. No TEST labels or new TEST filter predictions.',
        dependencies={str(f.relative_to(ROOT)):p.sha(f) for f in files}))

def fit():
    plan=p.read(REP/'plan.json')
    for f,h in plan['dependencies'].items():assert p.sha(ROOT/f)==h
    data={};rows=[]
    for k in old.WTRAIN:
        z=np.load(old.OUT/f'signals/common/{k}.npz');gt,valid,_,_=old.gt_speed(k,z['times']);gt/=max(float(gt.mean()),1e-8);data[k]=(z['times'],z['water'],gt,valid)
    for method,grid in GRID.items():
        for param in grid:
            for key,(t,w,gt,valid) in data.items():
                cache=OUT/f'train_cache/{method}/{param}/{key}.npz';diagfile=cache.with_suffix('.json')
                if cache.exists() and diagfile.exists():z=np.load(cache);s=z['speed'];diag=p.read(diagfile)
                else:
                    s,diag=apply(t,w,method,param);p.npzsave(cache,times=t,speed=s);p.save(diagfile,diag)
                rows.append(dict(key=key,method=method,param=param,mse=float(np.mean((s[valid]-gt[valid])**2)),mae=float(np.mean(abs(s[valid]-gt[valid]))),**diag))
            print('V118_FIT_GRID',method,param,flush=True)
    selected={};cv={};candidates=[]
    for method,grid in GRID.items():
        values={param:[r for r in rows if r['method']==method and r['param']==param] for param in grid}
        means={param:float(np.mean([r['mse'] for r in rr])) for param,rr in values.items()};best=min(grid,key=lambda q:(means[q],q));selected[method]=best;folds=[]
        for key in old.WTRAIN:
            choice=min(grid,key=lambda q:(np.mean([r['mse'] for r in values[q] if r['key']!=key]),q));held=next(r for r in values[choice] if r['key']==key);folds.append(dict(excluded_key=key,selected_param=choice,heldout_mse=held['mse']))
        cv[method]=dict(loco_mean_mse=float(np.mean([r['heldout_mse'] for r in folds])),folds=folds)
        candidates.extend(dict(method=method,param=q,train_mse=means[q]) for q in grid)
    best=min(candidates,key=lambda r:(r['train_mse'],r['method'],r['param']))
    p.save(REP/'trained.json',dict(created_utc=p.now(),selected=selected,selected_family=best['method'],selected_family_param=best['param'],train_keys=old.WTRAIN,candidates=candidates,loco=cv,rows=rows,VAL_labels_accessed=False))
    print('V118_SELECTED',selected,best,flush=True)

def predict():
    p.guard();fit=p.read(REP/'trained.json');files={};diags=[]
    for k in old.WVAL:
        c=p.case(k);t=c['t'];common=np.load(old.OUT/f'signals/common/{k}.npz');signals=dict(times=t,V87=c['speed'])
        for arm,tt,w in [('common',common['times'],common['water']),('routing',t,c['speed'])]:
            for method,param in fit['selected'].items():
                s,diag=apply(tt,w,method,param);s=np.interp(t,tt,s);name=arm+'_'+method;signals[name]=s
                pred=dict(c['pred'],positions=p.integrate(t,c['yaw'],s).tolist());p.validate_prediction(pred,k,p.load_timestamps(p.DATA,c['clip']));dest=OUT/f'predictions/{name}/{k}.json';p.save(dest,pred);files[str(dest.relative_to(ROOT))]=p.sha(dest);diags.append(dict(key=k,arm=arm,method=method,param=param,**diag))
        dest=OUT/f'signals/{k}.npz';p.npzsave(dest,**signals);files[str(dest.relative_to(ROOT))]=p.sha(dest)
    p.save(REP/'predictions_frozen.json',dict(created_utc=p.now(),files=files,fit_sha=p.sha(REP/'trained.json'),rows=diags,reference_access_blocked=True))

def evaluate():
    frozen=p.read(REP/'predictions_frozen.json');assert frozen['fit_sha']==p.sha(REP/'trained.json')
    for f,h in frozen['files'].items():assert p.sha(ROOT/f)==h
    original=p.read(ROOT/'reports/domain_v114_shrink_speed/results.json');base={r['key']:r['drift_pct'] for r in original['rows'] if r['method']=='V87'}
    control=p.read(ROOT/'reports/domain_v114b_preserve_routing/results.json');smooth={r['key']:r['drift_pct'] for r in control['rows'] if r['method']=='routing_full'}
    rows=[];summary={};contexts=[]
    for k in old.WVAL:
        c=p.case(k);t=c['t'];rt,rp=p.load_reference(p.DATA,c['clip']);signals=np.load(OUT/f'signals/{k}.npz');gt=np.load(p.OUT/f'evaluation_only/course_signals/{k}.npz');gn=gt['gt_speed']/gt['gt_speed'].mean();valid=(t>=rt[0]+2)&(t<=rt[-1]-2);rate=np.gradient(gn,t)
        masks={'slow':valid&(gn<.25),'changing':valid&(abs(rate)>=.01),'steady':valid&(abs(rate)<.01)}
        for name in signals.files:
            if name=='times':continue
            s=signals[name];positions=p.integrate(t,c['yaw'],s);row=p.score(k,name,t,positions,rt,rp);rows.append(row);sn=s/max(s.mean(),1e-8)
            contexts.append(dict(key=k,method=name,relative_speed_mse=float(np.mean((sn[valid]-gn[valid])**2)),**{label+'_mae':float(np.mean(abs(sn[m]-gn[m]))) if m.any() else None for label,m in masks.items()},**{label+'_fraction':float(m.mean()) for label,m in masks.items()}))
    for method in sorted({r['method'] for r in rows}):
        rr=[r for r in rows if r['method']==method];d=[r['drift_pct']-base[r['key']] for r in rr];ds=[r['drift_pct']-smooth[r['key']] for r in rr];mean=float(np.mean([r['drift_pct'] for r in rr]));wins=sum(x<-1e-8 for x in d)
        summary[method]=dict(mean_drift_pct=mean,wins=wins,losses=sum(x>1e-8 for x in d),maximum_harm_pp=max(d),worst_clip=rr[int(np.argmax(d))]['key'],wins_vs_Swide=sum(x<-1e-8 for x in ds),max_harm_vs_Swide_pp=max(ds),adopt=bool(mean<np.mean(list(base.values())) and wins>=11 and max(d)<=.5))
    p.save(REP/'results.json',dict(created_utc=p.now(),summary=summary,rows=rows,contexts=contexts,fit=p.read(REP/'trained.json')));print('V118_VAL',summary,flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['checks','freeze','fit','predict','evaluate']);a=ap.parse_args();globals()[a.action]()
