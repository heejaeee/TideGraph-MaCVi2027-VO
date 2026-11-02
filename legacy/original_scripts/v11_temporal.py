"""Image-derived low-frequency yaw bias, separate from trajectory evaluation."""
import argparse,datetime
from scipy.optimize import least_squares
from v5_common import *
from v10_bias import static_edges
from public_dataset import sha256

OUT=ROOT/'runs/domain_v11';REP=ROOT/'reports/domain_v11'
CONFIGS=[dict(name='v10_control',basis=1,strength=4.)]
CONFIGS += [dict(name=f'constant_strength{s}',basis=1,strength=float(s)) for s in [8,16]]
CONFIGS += [dict(name=f'temporal{n}_strength{s}',basis=n,strength=float(s)) for n in [2,3,5] for s in [4,8]]
CONFIGS += [dict(name=f'turn_bias_strength{s}',basis='turn',strength=float(s)) for s in [4,8]]

def observations(key,base,t,subset=None):
    e=np.load(ROOT/'runs/domain_v10/edges'/f'{key}.npz')['edges']
    st=static_edges(key,base,t);ep=(e[:,0].astype(int),e[:,1].astype(int),e[:,4],e[:,9])
    i,j,target,w=tuple(np.r_[a,b] for a,b in zip(st,ep));i=i.astype(int);j=j.astype(int)
    if subset is not None:
        if isinstance(subset,int):mask=np.random.default_rng(subset).random(len(w))<.7
        else:mask=np.floor((.5*(t[i]+t[j])-t[0])/20).astype(int)%2==int(subset[-1])
        i,j,target,w=i[mask],j[mask],target[mask],w[mask]
    return i,j,target,w

def integrated_basis(t,base,basis):
    duration=t[-1]-t[0];u=(t-t[0])/duration
    cols=[t-t[0]]
    if basis=='turn':
        rate=np.abs(np.gradient(gaussian_filter1d(base,10/np.median(np.diff(t))),t))
        rate=np.clip(rate/max(np.quantile(rate,.8),1e-6),0,2);rate-=np.mean(rate)
        cols.append(np.r_[0,np.cumsum(.5*(rate[:-1]+rate[1:])*np.diff(t))])
    else:
        for k in range(1,basis):cols.append(duration*np.sin(k*np.pi*u)/(k*np.pi))
    return np.array(cols).T

def correction(key,c,subset=None):
    case=get_case(key);t=case['t'];r=case['headings']['r3'];i,j,target,w=observations(key,r,t,subset)
    b=integrated_basis(t,r,c['basis']);mat=b[j]-b[i];rhs=target-(r[j]-r[i]);weight=np.sqrt(w/np.maximum(t[j]-t[i],1))
    reg=10*np.sqrt(len(w)*.1)*np.maximum(1,np.arange(b.shape[1])**2)
    def residual(x):return np.r_[(mat@x-rhs)*weight,x*reg]
    sol=least_squares(residual,np.zeros(b.shape[1]),bounds=(-np.deg2rad(.2),np.deg2rad(.2)),loss='soft_l1',f_scale=np.deg2rad(.3),max_nfev=100)
    return b@sol.x*c['strength'],dict(bias_coefficients_deg_s=np.rad2deg(sol.x).tolist(),support=len(w),success=bool(sol.success))

def emit(key,name,yaw,speed,case):
    save_json(OUT/'predictions'/name/f'{key}.json',{**case['pred'],'positions':integrate_velocity(case['t'],yaw,speed).tolist()})
    p=OUT/'signals'/name/f'{key}.npz';p.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(p,times=case['t'],yaw=yaw,speed=speed)

def main():
    save_json(REP/'temporal_plan.json',dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),configs=CONFIGS,hypothesis='A slowly varying image-derived yaw-rate bias may handle changing scene geometry better than a constant bias.',regularization='Cosine integrated basis, degree-squared penalty for higher frequencies. Same robust loss and prior as v10.',scope='Adaptive reused TRAIN8 development. No GT loading in inference. Speed frozen to v10.'))
    frozen={str(p.relative_to(ROOT)):sha256(p) for p in (ROOT/'submissions').glob('*/submission.json')}
    frozen.update({str(p.relative_to(ROOT)):sha256(p) for p in (ROOT/'runs/domain_v10/predictions/robust_hybrid_strength4p0').glob('*.json')})
    save_json(REP/'frozen_hashes.json',frozen)
    for key in TRAIN:
        case=get_case(key);s=case['speed'] if case['speed_correlation']>=.7 else np.ones(len(case['t']));s=.5+.5*s;diagnostics={}
        for c in CONFIGS:
            delta,info=correction(key,c);yaw=.65*case['headings']['r3']+delta;emit(key,c['name'],yaw,s,case);diagnostics[c['name']]=info
            if c['name']=='v10_control':
                old=np.load(ROOT/'runs/domain_v10/signals/robust_hybrid_strength4p0'/f'{key}.npz')['yaw'];assert np.max(abs(yaw-old))<1e-6
        save_json(OUT/'temporal_diagnostics'/f'{key}.json',diagnostics);print('TEMPORAL',key,flush=True)

if __name__=='__main__':main()
