import time
from .context import *
from .consensus import OUT
REP=ROOT/'reports/raw_masks'


def main():
    import torch
    from transformers import AutoImageProcessor,Mask2FormerForUniversalSegmentation
    v.core.bg.guard();torch.manual_seed(0);torch.set_num_threads(2);cv2.setNumThreads(1)
    info=json.loads((ROOT/'runs/domain_v5/segmentation/model.json').read_text());args=dict(revision=info['revision'],local_files_only=True)
    processor=AutoImageProcessor.from_pretrained(info['model'],use_fast=False,**args)
    model=Mask2FormerForUniversalSegmentation.from_pretrained(info['model'],**args).eval().cuda()
    counts={};verified=False;beg=time.perf_counter()
    for key in test.KEYS:
        folder=frame_root('test')/f'frames/{key}';meta=json.loads((folder/'index.json').read_text())
        missing=[i for i in range(len(meta['times'])) if not mask_path(key,i,'test').exists()]
        dest=OUT/f'test/masks/{key}';dest.mkdir(parents=True,exist_ok=True)
        for start in range(0,len(missing),4):
            batch=missing[start:start+4];images=[cv2.cvtColor(cv2.imread(str(folder/f'{i:04d}.png')),cv2.COLOR_BGR2RGB) for i in batch]
            inp=processor(images=images,return_tensors='pt',size={'shortest_edge':384,'longest_edge':640}).to('cuda')
            with torch.inference_mode():
                out=model(**inp);labs=processor.post_process_semantic_segmentation(out,target_sizes=[(round(meta['height']/2),480)]*len(batch));labs=[q.byte().cpu().numpy() for q in labs]
                if not verified:
                    single=processor(images=images[:1],return_tensors='pt',size={'shortest_edge':384,'longest_edge':640}).to('cuda');one=model(**single)
                    lab=processor.post_process_semantic_segmentation(one,target_sizes=[(round(meta['height']/2),480)])[0].byte().cpu().numpy()
                    agreement=float(np.mean(lab==labs[0]));assert agreement>.999
                    save_json(REP/'mask_batch_check.json',dict(passed=True,pixel_agreement=agreement,revision=info['revision']));verified=True;del one,single
            for i,lab in zip(batch,labs):assert cv2.imwrite(str(dest/f'{i:04d}.png'),lab)
            del inp,out,labs
            if start%100==0:print('MASK_PROGRESS',key,start,len(missing),round(time.perf_counter()-beg,1),flush=True)
        counts[key]=len(missing);print('MASK_COMPLETE',key,len(missing),flush=True)
    save_json(REP/'mask_provenance.json',dict(model=info['model'],revision=info['revision'],new_frames=counts,seconds=time.perf_counter()-beg,reference_access_blocked=True))
