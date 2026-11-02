"""Follow-up: estimate low-dimensional yaw drift from static multi-frame evidence.
Frozen per-clip image-derived fit; no trajectory references or per-clip rules.
"""
import datetime
from scipy.optimize import least_squares
from v10_evidence_graph import OUT,REP
from v5_common import *

CONFIGS=[dict(name=f'bias_{source}_{kind}',source=source,kind=kind) for source in ['static','epipolar','hybrid'] for kind in ['gain','bias','affine']]
CONFIGS += [dict(name=f'bias_{source}_local',source=source,kind='local') for source in ['static','hybrid']]

def static_edges(key,base,t):
    z=np.load(RUN/'cotracker'/f'{key}.npz');rp=np.load(ROOT/'runs/domain_probe_v2/r3/diagnostics'/f'{key}.poses.npz')['camera_to_world']
    vertical=np.median(rp[:,:3,1],axis=0);vertical/=np.linalg.norm(vertical)
    forward=rp[0,:3,2].copy();forward-=vertical*np.dot(forward,vertical);forward/=np.linalg.norm(forward);right=np.cross(vertical,forward)
    i=z['edge_i'];j=z['edge_j'];f=np.einsum('nij,njk,k->ni',rp[i,:3,:3],z['edge_r'].transpose(0,2,1),[0.,0.,1.])
    yaw_j=np.arctan2(f@right,f@forward);yi=np.arctan2(rp[i,:3,2]@right,rp[i,:3,2]@forward)
    dy=(yaw_j-yi+np.pi)%(2*np.pi)-np.pi
    w=np.minimum(z['edge_count']/80,1)
    return i,j,dy,w

def fit(base,t,i,j,target,w,kind,prior=.1):
    x=base[j]-base[i];dt=t[j]-t[i];gain=kind in ['gain','affine'];bias=kind in ['bias','affine','local']
    def residual(params):
        g=params[0] if gain else 1.;b=params[-1] if bias else 0.
        r=(g*x+b*dt-target)*np.sqrt(w/np.maximum(dt,1))
        # Weak prior against unobservable scale or offset, identical across clips.
        prior=[]
        if gain:prior.append((g-1)*np.deg2rad(.5)*np.sqrt(len(x)*prior_strength))
        if bias:prior.append(b*10*np.sqrt(len(x)*prior_strength))
        return np.r_[r,prior]
    prior_strength=prior
    x0=([1.] if gain else [])+([0.] if bias else [])
    lower=([.5] if gain else [])+([-np.deg2rad(.2)] if bias else [])
    upper=([1.5] if gain else [])+([np.deg2rad(.2)] if bias else [])
    sol=least_squares(residual,x0,bounds=(lower,upper),loss='soft_l1',f_scale=np.deg2rad(.3),max_nfev=100)
    return (float(sol.x[0]) if gain else 1.),(float(sol.x[-1]) if bias else 0.)

def main():
    save_json(REP/'bias_plan.json',dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),configs=CONFIGS,reason='v10 graph image residual improvements did not translate to drift. Test low-dimensional gain/bias instead of unconstrained local switching.',stage='Image-conditioned long-horizon yaw calibration, with fixed v2 speed',scope='Adaptive follow-up on repeatedly used TRAIN8; not independent validation.'))
    for key in TRAIN:
        c=get_case(key);t=c['t'];base=c['headings']['r3'];speed=c['speed'] if c['speed_correlation']>=.7 else np.ones(len(t));e=np.load(OUT/'edges'/f'{key}.npz')['edges']
        static=static_edges(key,base,t);epi=(e[:,0].astype(int),e[:,1].astype(int),e[:,4],e[:,9])
        sources={'static':static,'epipolar':epi,'hybrid':tuple(np.r_[a,b] for a,b in zip(static,epi))}
        diagnostics={}
        for cfg in CONFIGS:
            i,j,target,w=sources[cfg['source']];i=i.astype(int);j=j.astype(int)
            if cfg['kind']=='local':
                centers=np.arange(t[0],t[-1]+30,30);bias=[]
                for center in centers:
                    localw=w*np.exp(-(.5*(t[i]+t[j])-center)**2/(2*60**2))
                    _,b=fit(base,t,i,j,target,localw,'bias');bias.append(b)
                b=np.interp(t,centers,bias);yaw=base+np.r_[0,np.cumsum(.5*(b[1:]+b[:-1])*np.diff(t))]
                info=dict(bias_deg_s=np.rad2deg(bias).tolist(),centers=centers.tolist())
            else:
                g,b=fit(base,t,i,j,target,w,cfg['kind']);yaw=g*base+b*(t-t[0]);info=dict(gain=g,bias_deg_s=float(np.rad2deg(b)))
            save_json(OUT/'predictions'/cfg['name']/f'{key}.json',{**c['pred'],'positions':integrate_velocity(t,yaw,speed).tolist()});diagnostics[cfg['name']]=info
            dest=OUT/'signals'/cfg['name']/f'{key}.npz';dest.parent.mkdir(exist_ok=True,parents=True);np.savez_compressed(dest,times=t,yaw=yaw,speed=speed)
        save_json(OUT/'bias_diagnostics'/f'{key}.json',diagnostics);print('BIAS',key,flush=True)

if __name__=='__main__':main()
