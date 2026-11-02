"""DPVO ablation: same frozen model, calibrated K, random vs static patch sampling."""
import sys,time,types,traceback
from pathlib import Path
import cv2,numpy as np,torch
from scipy.spatial.transform import Rotation
from scipy.ndimage import gaussian_filter1d
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'repos/DPVO'))
from dpvo.dpvo import DPVO
from dpvo.config import cfg
from dpvo.net import altcorr,coords_grid_with_index,DIM
from v5_common import *

def masked_forward(self,images,patches_per_image=80,disps=None,centroid_sel_strat='RANDOM',return_color=False):
    fmap=self.fnet(images)/4.;imap=self.inet(images)/4.;b,n,c,h,w=fmap.shape;P=self.patch_size
    mask=torch.nn.functional.interpolate(self.current_static.float()[None,None],size=(h,w),mode='nearest')[0,0]>0
    mask[:1]=False;mask[-1:]=False;mask[:,:1]=False;mask[:,-1:]=False
    valid=torch.nonzero(mask,as_tuple=False);self.mask_counts.append(int(len(valid)))
    if len(valid)>=8:
        picked=valid[torch.randint(len(valid),(n,patches_per_image),device='cuda')];y=picked[:,:,0];x=picked[:,:,1]
    else:
        x=torch.randint(1,w-1,size=[n,patches_per_image],device='cuda');y=torch.randint(1,h-1,size=[n,patches_per_image],device='cuda')
    coords=torch.stack([x,y],dim=-1).float();imap=altcorr.patchify(imap[0],coords,0).view(b,-1,DIM,1,1);gmap=altcorr.patchify(fmap[0],coords,P//2).view(b,-1,128,P,P)
    if return_color:clr=altcorr.patchify(images[0],4*(coords+.5),0).view(b,-1,3)
    if disps is None:disps=torch.ones(b,n,h,w,device='cuda')
    grid,_=coords_grid_with_index(disps,device=fmap.device);patches=altcorr.patchify(grid[0],coords,P//2).view(b,-1,3,P,P);index=torch.arange(n,device='cuda').view(n,1).repeat(1,patches_per_image).reshape(-1)
    return (fmap,gmap,imap,patches,index,clr) if return_color else (fmap,gmap,imap,patches,index)

def work(key,mode,seed):
    out=RUN/'dpvo'/f'{mode}_s{seed}'/f'{key}.npz'
    if out.exists():return
    torch.manual_seed(seed);np.random.seed(seed);folder,meta=frames(key);times=np.array(meta['times']);case=get_case(key);beg=time.perf_counter();width=640;height=meta['height']*width//960//16*16
    k=np.load(ROOT/'runs/domain_v4/calibrated_rotation_train'/f'{key}.npz')['learned_k'];intr=torch.tensor([k[0,0],k[1,1],k[0,2],k[1,2]],dtype=torch.float32,device='cuda')*(width/960)
    config=cfg.clone();config.merge_from_file(str(ROOT/'repos/DPVO/config/default.yaml'));config.BUFFER_SIZE=2048
    model=DPVO(config,str(ROOT/'repos/DPVO/dpvo.pth'),ht=height,wd=width,viz=False);torch.cuda.reset_peak_memory_stats()
    info=json.loads((RUN/'segmentation/model.json').read_text())
    if mode=='static':
        model.network.patchify.forward=types.MethodType(masked_forward,model.network.patchify);model.network.patchify.mask_counts=[]
    with torch.no_grad():
        for i in range(len(times)):
            im=cv2.imread(str(folder/f'{i:04d}.png'));im=cv2.resize(im,(width,round(meta['height']*width/960)),interpolation=cv2.INTER_AREA)[:height]
            if mode=='static':
                lab=cv2.imread(str(RUN/'segmentation'/key/f'{i:04d}.png'),0);mask=np.isin(lab,info['static_ids']).astype(np.uint8);mask=cv2.resize(mask,(width,round(meta['height']*width/960)),interpolation=cv2.INTER_NEAREST)[:height];mask=cv2.erode(mask,np.ones((9,9),np.uint8));model.network.patchify.current_static=torch.from_numpy(mask).cuda()
            model(i,torch.from_numpy(im.copy()).permute(2,0,1).cuda(),intr)
        poses,indices=model.terminate()
    indices=np.asarray(indices).astype(int);assert np.array_equal(indices,np.arange(len(times)))
    assert np.isfinite(poses).all()
    matrices=np.tile(np.eye(4),(len(poses),1,1));matrices[:,:3,:3]=Rotation.from_quat(poses[:,3:]).as_matrix();matrices[:,:3,3]=poses[:,:3]
    out.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(out,times=times,camera_to_world=matrices)
    yaw=gaussian_filter1d(planar_yaw(matrices),2/np.median(np.diff(times)));base=case['headings']['r3'];speed=case['speed'] if case['speed_correlation']>=.7 else np.ones(len(times))
    for kind,pos in [('positions',poses[:,:3]),('direct',integrate_velocity(times,yaw,speed)),('blend',integrate_velocity(times,.5*(yaw+base),speed))]:save_json(RUN/'predictions'/f'dpvo_{mode}_{kind}_s{seed}'/f'{key}.json',{**case['pred'],'positions':pos.tolist()})
    save_json(out.with_suffix('.json'),dict(key=key,mode=mode,seed=seed,seconds=time.perf_counter()-beg,peak_cuda_gib=torch.cuda.max_memory_allocated()/2**30,mask_fallback_frames=sum(n<8 for n in getattr(model.network.patchify,'mask_counts',[]))))
    print('DPVO',mode,key,seed,round(time.perf_counter()-beg,1),flush=True)
    del model;torch.cuda.empty_cache()

if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int);args=ap.parse_args()
    cv2.setNumThreads(1);torch.set_num_threads(2)
    for seed in ([args.seed] if args.seed is not None else SEEDS):
        for key in TRAIN:
            for mode in ['random','static']:
                work(key,mode,seed)
