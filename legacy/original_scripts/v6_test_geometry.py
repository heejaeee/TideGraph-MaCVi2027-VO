"""Frozen TRAIN-selected self-calibration and rotation fitting on TEST images."""
import concurrent.futures,time
from v6_test_common import *
from v5_selfcal import estimate
from rotation_flow_infer_clip import fit_rotation

def selfcal(key,window_dir,k,dest):
    fs=[];stats=[]
    for path in sorted(window_dir.glob('*.npz')):
        z=np.load(path);tr=z['tracks'];vis=z['visible'];start=int(z['start'])
        for i,j in [(0,8),(0,16),(0,24),(8,24)]:
            if j>=len(tr):continue
            good=vis[i]&vis[j];p=tr[i,good];q=tr[j,good]
            if len(p)<50:continue
            cv2.setRNGSeed(start+i+j);F,inside=cv2.findFundamentalMat(p,q,cv2.USAC_MAGSAC,1.5,.999,5000)
            if F is None or F.shape!=(3,3) or inside.sum()<max(40,.5*len(p)):continue
            use=inside[:,0].astype(bool)
            if np.ptp(p[use,0])<150 or np.ptp(p[use,1])<40:continue
            fs.append(F);stats.append([start+i,start+j,int(inside.sum())])
    result=estimate(np.array(fs),k) if len(fs)>=20 else dict(accepted=False,ratio=1.,constraints=len(fs),reason='insufficient constraints')
    save_json(dest,dict(key=key,**result,pairs=stats,nominal_k=k.tolist(),note='Image constraints only; same TRAIN algorithm, no references'))
    return result

def fit_all(key,times,matches,offsets,ks,output):
    for seed in SEEDS:
        dest=output/f'selfcal_dynamic_s{seed}'/f'{key}.npz'
        if dest.exists():continue
        rotations=[];counts=[]
        for j in range(len(times)-1):
            pq=matches[offsets[j]:offsets[j+1]];a=bearings(pq[:,:2],ks[j]);b=bearings(pq[:,2:],ks[j+1]);r,inside=fit_rotation(a,b,np.random.default_rng(seed+j),1.5/ks[j,0,0]);rotations.append(r);counts.append([len(pq),int(inside.sum())])
        dest.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(dest,times=times,rotations=rotations,counts=counts,k=ks)

def work(key):
    cv2.setNumThreads(1);start=time.monotonic();windows=RUN/'cotracker_windows'/key
    while not (windows/'summary.json').exists():
        if time.monotonic()-start>7200:raise TimeoutError(key)
        time.sleep(2)
    z=np.load(ROOT/'runs/domain_v4/calibrated_rotation_test'/f'{key}.npz');basek=z['learned_k'];times=z['times'];result=selfcal(key,windows,basek,RUN/'selfcal'/f'{key}.json')
    d=json.loads((ROOT/'runs/domain_v4/anycalib_test'/f'{key}.json').read_text());ts=load_timestamps(DATA,CLIPS[key]);vals=np.array(d['intrinsics_per_frame']);tt=ts[d['frame_indices']];w,h=d['active_rect'][2:];ks=np.tile(basek,(len(times),1,1))
    for col,(i,j) in enumerate([(0,0),(1,1),(0,2),(1,2)]):ks[:,i,j]=np.interp(times,tt,vals[:,col])*(960/w if col in [0,2] else round(h*960/w)/h)
    ks[:,0,0]*=result['ratio'];ks[:,1,1]*=result['ratio'];fit_all(key,times,z['matches'],z['offsets'],ks,RUN/'rotations')
    print('GEOMETRY',key,'ratio',result['ratio'],'constraints',result['constraints'],flush=True)

if __name__=='__main__':
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:list(pool.map(work,KEYS))
