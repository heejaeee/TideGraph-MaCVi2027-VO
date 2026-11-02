"""Only CoTracker query-frame masks and long tracks required for focal estimation."""
import argparse,time,torch
from v6_test_common import *

def wait_frames(key):
    start=time.monotonic()
    while not (RUN/'frames'/key/'index.json').exists():
        if time.monotonic()-start>3600:raise TimeoutError(key)
        time.sleep(2)

def masks():
    from transformers import AutoImageProcessor,Mask2FormerForUniversalSegmentation
    info=json.loads((ROOT/'runs/domain_v5/segmentation/model.json').read_text());name=info['model'];revision=info['revision']
    processor=AutoImageProcessor.from_pretrained(name,revision=revision,use_fast=False,local_files_only=True)
    model=Mask2FormerForUniversalSegmentation.from_pretrained(name,revision=revision,local_files_only=True).eval().cuda()
    save_json(RUN/'segmentation/model.json',info)
    for key in KEYS:
        wait_frames(key);folder,meta=frames(key);dest=RUN/'segmentation'/key;dest.mkdir(parents=True,exist_ok=True);ids=window_starts(len(meta['times']));start=time.perf_counter()
        for i in ids:
            p=dest/f'{i:04d}.png'
            if p.exists():continue
            im=cv2.cvtColor(cv2.imread(str(folder/f'{i:04d}.png')),cv2.COLOR_BGR2RGB);inputs=processor(images=im,return_tensors='pt',size={'shortest_edge':384,'longest_edge':640}).to('cuda')
            with torch.inference_mode():
                result=model(**inputs);lab=processor.post_process_semantic_segmentation(result,target_sizes=[(round(meta['height']/2),480)])[0].byte().cpu().numpy()
            assert cv2.imwrite(str(p),lab);del inputs,result
        save_json(dest/'summary.json',dict(key=key,query_frames=ids,seconds=time.perf_counter()-start,batch=1,note='Only frames used for CoTracker queries; exact v5 model/preprocessing'))
        print('MASKS',key,len(ids),round(time.perf_counter()-start,1),flush=True)

def tracks():
    model=torch.hub.load(str(ROOT/'repos/co-tracker'),'cotracker3_offline',source='local').eval().cuda();info=json.loads((RUN/'segmentation/model.json').read_text())
    for key in KEYS:
        folder,meta=frames(key);w,h=meta['width'],meta['height'];n=len(meta['times']);dest=RUN/'cotracker_windows'/key;dest.mkdir(parents=True,exist_ok=True);start_time=time.perf_counter()
        for start in window_starts(n):
            end=min(n,start+32);cache=dest/f'{start:04d}.npz'
            if cache.exists():continue
            labels=cv2.imread(str(RUN/'segmentation'/key/f'{start:04d}.png'),0);assert labels is not None
            mask=np.isin(labels,info['static_ids']).astype(np.uint8)*255;mask=cv2.resize(mask,(w,h),interpolation=cv2.INTER_NEAREST);mask=cv2.erode(mask,np.ones((3,3),np.uint8));mask[:int(.08*h)]=0;mask[int(.8*h):]=0;mask[:,:int(.05*w)]=0;mask[:,int(.95*w):]=0
            gray=cv2.cvtColor(cv2.imread(str(folder/f'{start:04d}.png')),cv2.COLOR_BGR2GRAY);pts=cv2.goodFeaturesToTrack(gray,256,.003,12,mask=mask)
            if pts is None or len(pts)<8:tr=np.empty((end-start,0,2),np.float32);vis=np.empty((end-start,0),bool)
            else:
                video=np.stack([cv2.cvtColor(cv2.imread(str(folder/f'{i:04d}.png')),cv2.COLOR_BGR2RGB) for i in range(start,end)]);video=torch.from_numpy(video).permute(0,3,1,2)[None].float().cuda();queries=torch.from_numpy(np.c_[np.zeros(len(pts)),pts[:,0]].astype(np.float32))[None].cuda()
                with torch.inference_mode():pred,visible=model(video,queries=queries)
                tr=pred[0].float().cpu().numpy();vis=visible[0].cpu().numpy().astype(bool);del video,queries,pred,visible
            np.savez_compressed(cache,tracks=tr,visible=vis,start=start)
        save_json(dest/'summary.json',dict(key=key,windows=len(window_starts(n)),seconds=time.perf_counter()-start_time))
        print('TRACKS',key,round(time.perf_counter()-start_time,1),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['masks','tracks']);a=ap.parse_args();torch.manual_seed(0);torch.set_num_threads(2);cv2.setNumThreads(1)
    (masks if a.stage=='masks' else tracks)()
