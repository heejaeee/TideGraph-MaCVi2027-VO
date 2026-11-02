"""Focal self-calibration from image correspondences, no trajectory labels."""
import time
from scipy.optimize import minimize_scalar
from scipy.spatial.transform import Rotation
from v5_common import *

def costs(logscale,fs,k):
    kk=k.copy();kk[0,0]*=np.exp(logscale);kk[1,1]*=np.exp(logscale)
    e=kk.T[None]@fs@kk[None];sv=np.linalg.svd(e,compute_uv=False)
    return np.abs(np.log(np.maximum(sv[:,0],1e-12)/np.maximum(sv[:,1],1e-12)))
def estimate(fs,k):
    train=fs[::2];audit=fs[1::2];grid=np.linspace(np.log(.5),np.log(4.),101)
    loss=np.array([np.median(costs(x,train,k))+.01*x*x for x in grid]);idx=int(np.argmin(loss));x=float(grid[idx])
    before=float(np.median(costs(0,audit,k)));after=float(np.median(costs(x,audit,k)))
    accepted=len(fs)>=20 and .55<np.exp(x)<3.8 and after<.9*before
    return dict(proposed_ratio=float(np.exp(x)),accepted=bool(accepted),ratio=float(np.exp(x)) if accepted else 1.,validation_constraint_before=before,validation_constraint_after=after,grid_log_scale=grid.tolist(),train_cost=loss.tolist(),constraints=len(fs))
def work(key):
    out=RUN/'selfcal'/f'{key}.json'
    if out.exists():return
    k=np.load(ROOT/'runs/domain_v4/calibrated_rotation_train'/f'{key}.npz')['learned_k'];fs=[];stats=[]
    for path in sorted((RUN/'cotracker_windows'/key).glob('*.npz')):
        z=np.load(path);tr=z['tracks'];vis=z['visible'];start=int(z['start'])
        for i,j in [(0,8),(0,16),(0,24),(8,24)]:
            if j>=len(tr):continue
            good=vis[i]&vis[j];p=tr[i,good];q=tr[j,good]
            if len(p)<50:continue
            cv2.setRNGSeed(start+i+j)
            F,inside=cv2.findFundamentalMat(p,q,cv2.USAC_MAGSAC,1.5,.999,5000)
            if F is None or F.shape!=(3,3) or inside.sum()<max(40,.5*len(p)):continue
            use=inside[:,0].astype(bool)
            if np.ptp(p[use,0])<150 or np.ptp(p[use,1])<40:continue
            fs.append(F);stats.append([start+i,start+j,int(inside.sum())])
    result=estimate(np.array(fs),k) if len(fs)>=20 else dict(accepted=False,ratio=1.,constraints=len(fs),reason='insufficient constraints')
    save_json(out,dict(key=key,**result,pairs=stats,nominal_k=k.tolist(),note='Alternating image constraints for fitting/audit; not independent trajectory holdout'))
    print('SELFCAL',key,'pairs',len(fs),'ratio',result['ratio'],'proposed',result.get('proposed_ratio'),'accepted',result['accepted'],flush=True)
def synthetic_check():
    rng=np.random.default_rng(99);truek=np.array([[1200.,0,480],[0,1200,270],[0,0,1.]])
    fs=[]
    for _ in range(15):
        r=Rotation.from_rotvec(rng.normal(0,.15,3)).as_matrix();t=rng.normal(size=3);skew=np.array([[0,-t[2],t[1]],[t[2],0,-t[0]],[-t[1],t[0],0]])
        fs.append(np.linalg.inv(truek).T@skew@r@np.linalg.inv(truek))
    init=truek.copy();init[0,0]/=1.7;init[1,1]/=1.7
    opt=minimize_scalar(lambda x:np.mean(costs(x,np.array(fs),init)),bounds=(np.log(.5),np.log(4)),method='bounded')
    assert abs(np.exp(opt.x)-1.7)<.001
    save_json(REPORT/'selfcal_synthetic_check.json',dict(true_ratio=1.7,recovered_ratio=float(np.exp(opt.x)),passed=True))
if __name__=='__main__':
    cv2.setNumThreads(1);synthetic_check()
    for key in TRAIN:
        deadline=time.monotonic()+7200
        while not (RUN/'cotracker'/f'{key}.npz').exists():
            if time.monotonic()>deadline:raise TimeoutError(key)
            time.sleep(5)
        work(key)
