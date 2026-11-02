"""Confidence-gated residual log-speed graph, with an exact prior fallback."""
import datetime
from scipy.optimize import least_squares
from v12_relative_speed import extract
from v12_common import *

CONFIGS=[dict(name=f'prior_alpha{int(a*100)}',alpha=a,reg=0.,blend=0.,gate=True) for a in [.5,.65,.75,1.]]
CONFIGS += [dict(name=f'residual_reg{r:g}_b{b:g}_a{int(a*100)}',alpha=a,reg=r,blend=b,gate=True)
            for a in [.5,.75] for r in [3.,10.] for b in [.5,1.]]
CONFIGS += [dict(name='residual_ungated_r10_a50',alpha=.5,reg=10.,blend=1.,gate=False)]

def gated_prior(case,alpha):
    s=case['speed'] if case['speed_correlation']>=.7 else np.ones(len(case['t']))
    return 1-alpha+alpha*s

def infer(t,prior,rows,reg,blend,gate=True,seed=None):
    fraction=float(np.mean(rows[:,4]>.05));quality=float(np.mean(rows[:,4]))
    accepted=not gate or (fraction>=.5 and quality>=.2)
    diag=dict(accepted=accepted,valid_fraction=fraction,mean_quality=quality)
    if not accepted or not blend:return prior.copy(),diag
    knots=np.linspace(t[0],t[-1],int(np.ceil((t[-1]-t[0])/10))+1);eye=np.eye(len(knots))
    def basis(x):return np.array([np.interp(x,knots,e) for e in eye]).T
    m1=(rows[:,0]+rows[:,1])/2;m2=(rows[:,1]+rows[:,2])/2
    a=basis(m2)-basis(m1);p=np.log(np.maximum(prior,1e-3))
    rhs=rows[:,3]-(np.interp(m2,t,p)-np.interp(m1,t,p));weights=rows[:,4].copy()
    if seed is not None:
        blocks=((rows[:,1]-t[0])/20).astype(int);ids=np.unique(blocks);draw=np.random.default_rng(seed).choice(ids,len(ids),replace=True)
        weights*=np.array([np.sum(draw==b) for b in blocks])
    w=np.sqrt(weights);d2=np.diff(eye,2,axis=0)
    def residual(x):return np.r_[(a@x-rhs)*w,reg*.1*x,reg*d2@x,10*x.mean()]
    # Preserve a genuinely quadratic regularizer; robustify data rows only.
    def loss(z):
        v=np.vstack([z,np.ones(len(z)),np.zeros(len(z))]);n=len(rows);zz=z[:n]
        v[0,:n]=2*(np.sqrt(1+zz)-1);v[1,:n]=1/np.sqrt(1+zz);v[2,:n]=-.5/(1+zz)**1.5;return v
    sol=least_squares(residual,np.zeros(len(knots)),bounds=(-np.log(2),np.log(2)),loss=loss,f_scale=.15,max_nfev=80)
    delta=np.interp(t,knots,sol.x);delta-=delta.mean();out=prior*np.exp(blend*delta)
    return out,{**diag,'max_log_correction':float(np.max(abs(blend*delta))),'speed_cv':float(np.std(out)/np.mean(out))}

def main():
    save_json(REP/'speed_residual_plan.json',dict(configs=CONFIGS,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),hypothesis='Weak static translation evidence should change only a residual around the speed prior. Keep an exact prior fallback for low support. Compare historical prior alpha separately.',gate='At least half of triplets have quality > .05 and mean quality >= .2. No clip IDs in routing.',scope='Repeated TRAIN8 tuning. Alpha change can affect only TRAIN clip_001 under the historical .7 speed agreement gate; audit TEST separately.'))
    for key in TRAIN:
        case,y,_=baseline(key);extract(key);rows=np.load(OUT/'relative_speed/train'/f'{key}.npz')['raw'];diag={}
        for c in CONFIGS:
            s,d=infer(case['t'],gated_prior(case,c['alpha']),rows,c['reg'],c['blend'],c['gate']);emit(key,c['name'],y,s,case);diag[c['name']]=d
        save_json(OUT/'speed_residual'/f'{key}.json',diag)
        print('RESIDUAL',key,diag['residual_reg10_b1_a75'],flush=True)

if __name__=='__main__':main()
