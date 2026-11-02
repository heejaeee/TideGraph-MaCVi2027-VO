import sys,time,types,warnings
from .context import *
from .consensus import OUT


def run(key, mode, seed):
    path = OUT/f'test/poses/{mode}_s{seed}/{key}.npz'
    if path.exists(): return
    v.core.bg.guard()
    warnings.filterwarnings('ignore', category=FutureWarning)
    sys.path.insert(0, str(ROOT/'repos/DPVO'))
    import torch
    from dpvo.dpvo import DPVO as RawDPVO
    from .masked_patch import masked_forward
    ids=json.loads((ROOT/'runs/domain_v5/segmentation/model.json').read_text())['static_ids']
    class DPVO(RawDPVO):
        def __init__(self, config, *args, **kwargs):
            config.KEYFRAME_THRESH=5.0
            super().__init__(config,*args,**kwargs)
            self.network.patchify.forward=types.MethodType(masked_forward,self.network.patchify)
            self.network.patchify.mask_counts=[]
        def __call__(self,tstamp,image,intrinsics):
            path=mask_path(key,int(tstamp),'test');lab=v.cv2.imread(str(path),0)
            if lab is None:raise FileNotFoundError(path)
            mask=np.isin(lab,ids).astype(np.uint8)
            mask=v.cv2.resize(mask,(640,360),interpolation=v.cv2.INTER_NEAREST)[:352]
            mask=v.cv2.erode(mask,np.ones((9,9),np.uint8))
            self.network.patchify.current_static=torch.from_numpy(mask).cuda()
            return super().__call__(tstamp,image,intrinsics)
    from dpvo.config import cfg
    from scipy.spatial.transform import Rotation
    cv2 = v.cv2; cv2.setNumThreads(1); torch.set_num_threads(2)
    torch.manual_seed(seed); np.random.seed(seed)
    folder = frame_root('test')/f'frames/{key}'
    meta = json.loads((folder/'index.json').read_text()); times = np.array(meta['times'])
    assert meta['width']==960 and meta['height']==540
    width = 640; height = meta['height']*width//960//16*16
    k = np.load(ROOT/f'runs/domain_v4/calibrated_rotation_test/{key}.npz')['learned_k']
    intr = torch.tensor([k[0,0],k[1,1],k[0,2],k[1,2]], dtype=torch.float32, device='cuda')*(width/960)
    config = cfg.clone(); config.merge_from_file(str(ROOT/'repos/DPVO/config/default.yaml')); config.BUFFER_SIZE = 2048
    config.LOOP_CLOSURE = mode == 'slam'; config.CLASSIC_LOOP_CLOSURE = False
    started = time.perf_counter(); model = DPVO(config, str(ROOT/'repos/DPVO/dpvo.pth'), ht=height, wd=width, viz=False)
    stats = dict(global_ba_calls=0, global_ba_successes=0, global_ba_errors=[], loop_queries=0, loop_patch_edges=0, loop_events=[])
    original_ba = model._DPVO__run_global_BA
    def observed_ba(self):
        stats['global_ba_calls'] += 1
        try: result = original_ba()
        except Exception as exc:
            stats['global_ba_errors'].append(str(exc)); raise
        stats['global_ba_successes'] += 1
        return result
    model._DPVO__run_global_BA = types.MethodType(observed_ba, model)
    original_edges = model.pg.edges_loop
    def observed_edges(self):
        kk, jj = original_edges(); stats['loop_queries'] += 1
        if kk.numel():
            ii = self.ix[kk]
            pairs = torch.unique(torch.stack([ii,jj], 1), dim=0).cpu().numpy()
            gaps = self.tstamps_[pairs[:,1]]-self.tstamps_[pairs[:,0]]
            stats['loop_patch_edges'] += int(kk.numel())
            stats['loop_events'].append(dict(keyframes=int(self.n), input_frame=int(model.counter), patch_edges=int(kk.numel()), frame_pairs=len(pairs), min_frame_gap=int(gaps.min()), max_frame_gap=int(gaps.max())))
        return kk, jj
    model.pg.edges_loop = types.MethodType(observed_edges, model.pg)
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        for i in range(len(times)):
            im = cv2.imread(str(folder/f'{i:04d}.png'))
            if im is None: raise FileNotFoundError(folder/f'{i:04d}.png')
            im = cv2.resize(im, (width, round(meta['height']*width/960)), interpolation=cv2.INTER_AREA)[:height]
            model(i, torch.from_numpy(im.copy()).permute(2,0,1).cuda(), intr)
            if i%100 == 0:
                print('DPVSLAM', mode, key, seed, i, '/', len(times), 'keyframes',model.n,'BA',stats['global_ba_successes'],'seconds',round(time.perf_counter()-started,1), flush=True)
        poses, indices = model.terminate()
    indices = np.asarray(indices).astype(int)
    assert np.array_equal(indices,np.arange(len(times))) and np.isfinite(poses).all()
    mat = np.tile(np.eye(4),(len(poses),1,1)); mat[:,:3,:3] = Rotation.from_quat(poses[:,3:]).as_matrix(); mat[:,:3,3] = poses[:,:3]
    path.parent.mkdir(parents=True, exist_ok=True); np.savez_compressed(path, times=times, camera_to_world=mat)
    stats.update(mask_fallback_frames=sum(n<8 for n in model.network.patchify.mask_counts),key=key, mode=mode, seed=seed, seconds=time.perf_counter()-started,
        peak_cuda_gib=torch.cuda.max_memory_allocated()/2**30, retained_keyframes=int(model.n), frames=len(times), config=config.dump(), reference_access_blocked=True)
    save_json(path.with_suffix('.json'),stats)
    print('DONE_DPVSLAM',mode,key,seed,'seconds',round(stats['seconds'],1),'BA',stats['global_ba_successes'],'errors',len(stats['global_ba_errors']),flush=True)
