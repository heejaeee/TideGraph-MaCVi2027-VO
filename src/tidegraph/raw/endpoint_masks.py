"""Frozen endpoint-mask pass, preserving the original batch grouping."""
import time,torch
from transformers import AutoImageProcessor,Mask2FormerForUniversalSegmentation
from .context import *
from types import SimpleNamespace
OUT=ROOT/'runs/domain_v16'
REP=ROOT/'reports/raw_endpoint_masks'
test=SimpleNamespace(KEYS=KEYS,RUN=ROOT/'runs/domain_v6_test',frames=frames)

def main():
    torch.manual_seed(0);torch.set_num_threads(2);cv2.setNumThreads(1)
    info=json.loads((ROOT/'runs/domain_v5/segmentation/model.json').read_text())
    args=dict(revision=info['revision'],local_files_only=True)
    processor=AutoImageProcessor.from_pretrained(info['model'],use_fast=False,**args)
    model=Mask2FormerForUniversalSegmentation.from_pretrained(info['model'],**args).eval().cuda()
    counts={};beg=time.perf_counter();verified=False
    for key in test.KEYS:
        required=set()
        for p in sorted((test.RUN/'cotracker_windows'/key).glob('*.npz')):
            with np.load(p) as z:
                st=int(z['start']);n=len(z['tracks'])
                for a,b in [(0,8),(8,16),(16,24)]:
                    if b<n:required.update([st+a,st+b])
        folder,meta=test.frames(key);dest=OUT/'test/extra_masks'/key;dest.mkdir(parents=True,exist_ok=True)
        missing=[i for i in sorted(required) if not (test.RUN/'segmentation'/key/f'{i:04d}.png').exists() and not (dest/f'{i:04d}.png').exists()]
        for start in range(0,len(missing),4):
            batch=missing[start:start+4];images=[]
            for i in batch:
                im=cv2.imread(str(folder/f'{i:04d}.png'));assert im is not None
                images.append(cv2.cvtColor(im,cv2.COLOR_BGR2RGB))
            inp=processor(images=images,return_tensors='pt',size={'shortest_edge':384,'longest_edge':640}).to('cuda')
            with torch.inference_mode():
                out=model(**inp);labs=processor.post_process_semantic_segmentation(out,target_sizes=[(round(meta['height']/2),480)]*len(batch))
                labs=[v.byte().cpu().numpy() for v in labs]
                if not verified:
                    single=processor(images=images[:1],return_tensors='pt',size={'shortest_edge':384,'longest_edge':640}).to('cuda')
                    one=model(**single);lab=processor.post_process_semantic_segmentation(one,target_sizes=[(round(meta['height']/2),480)])[0].byte().cpu().numpy()
                    agreement=float(np.mean(lab==labs[0]));assert agreement>.999
                    save_json(REP/'mask_batch_check.json',dict(pixel_agreement=agreement,passed=True,revision=info['revision']));verified=True
                    del one,single
            for i,lab in zip(batch,labs):assert cv2.imwrite(str(dest/f'{i:04d}.png'),lab)
            del inp,out,labs
        counts[key]=len(missing);print('MASKS',key,len(missing),round(time.perf_counter()-beg,1),flush=True)
    save_json(REP/'extra_mask_provenance.json',dict(model=info['model'],revision=info['revision'],new_frames=counts,seconds=time.perf_counter()-beg,scope='Only missing TEST pair endpoints, no reference access. Existing masks untouched.'))
