"""Frozen v5 DPVO RANDOM96 / fixed AnyCalib intrinsics, three TEST passes."""
import time,torch
from scipy.spatial.transform import Rotation
from v6_test_common import *
sys.path.insert(0,str(ROOT/'repos/DPVO'))
from dpvo.dpvo import DPVO
from dpvo.config import cfg

def work(key,seed):
    dest=RUN/'dpvo'/f'random_s{seed}'/f'{key}.npz'
    if dest.exists():return
    started=time.monotonic()
    while not (RUN/'frames'/key/'index.json').exists():
        if time.monotonic()-started>3600:raise TimeoutError(key)
        time.sleep(2)
    torch.manual_seed(seed);np.random.seed(seed);folder,meta=frames(key);t=np.array(meta['times']);width=640;height=meta['height']*width//960//16*16;start=time.perf_counter()
    k=np.load(ROOT/'runs/domain_v4/calibrated_rotation_test'/f'{key}.npz')['learned_k'];intr=torch.tensor([k[0,0],k[1,1],k[0,2],k[1,2]],dtype=torch.float32,device='cuda')*(width/960)
    config=cfg.clone();config.merge_from_file(str(ROOT/'repos/DPVO/config/default.yaml'));config.BUFFER_SIZE=2048
    model=DPVO(config,str(ROOT/'repos/DPVO/dpvo.pth'),ht=height,wd=width,viz=False);torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        for i in range(len(t)):
            im=cv2.imread(str(folder/f'{i:04d}.png'));im=cv2.resize(im,(width,round(meta['height']*width/960)),interpolation=cv2.INTER_AREA)[:height]
            model(i,torch.from_numpy(im.copy()).permute(2,0,1).cuda(),intr)
        poses,indices=model.terminate()
    assert np.array_equal(np.asarray(indices).astype(int),np.arange(len(t))) and np.isfinite(poses).all()
    matrices=np.tile(np.eye(4),(len(t),1,1));matrices[:,:3,:3]=Rotation.from_quat(poses[:,3:]).as_matrix();matrices[:,:3,3]=poses[:,:3]
    dest.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(dest,times=t,camera_to_world=matrices)
    save_json(dest.with_suffix('.json'),dict(key=key,seed=seed,seconds=time.perf_counter()-start,peak_cuda_gib=torch.cuda.max_memory_allocated()/2**30,settings='Same v5 DPVO RANDOM96,640width,2FPS,fixed learned_k,buffer2048'))
    del model;torch.cuda.empty_cache();print('DPVO',key,seed,round(time.perf_counter()-start,1),flush=True)
if __name__=='__main__':
    torch.set_num_threads(2);cv2.setNumThreads(1)
    for seed in SEEDS:
        for key in KEYS:work(key,seed)
