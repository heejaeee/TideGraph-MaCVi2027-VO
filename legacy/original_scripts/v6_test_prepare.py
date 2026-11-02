import concurrent.futures,time
from v6_test_common import *

def work(key):
    out=RUN/'frames'/key
    if (out/'index.json').exists():return key+' cached'
    out.mkdir(parents=True,exist_ok=True);cv2.setNumThreads(1);c=CLIPS[key];ts=load_timestamps(DATA,c)
    ids=np.unique(np.r_[np.searchsorted(ts,np.arange(ts[0],ts[-1],.5)),len(ts)-1]);wanted={int(v):j for j,v in enumerate(ids)}
    expected=json.loads((V2/'predictions'/f'{key}.json').read_text());assert np.array_equal(ids,expected['frame_indices'])
    x,y,w,h=c['calibration']['active_rect'];width=960;height=round(h*width/w);cap=cv2.VideoCapture(str(DATA/c['video']));start=time.perf_counter()
    for i in range(int(ids[-1])+1):
        if not cap.grab():raise RuntimeError((key,i))
        if i not in wanted:continue
        ok,bgr=cap.retrieve()
        if not ok:raise RuntimeError((key,i))
        im=cv2.resize(bgr[y:y+h,x:x+w],(width,height));assert cv2.imwrite(str(out/f'{wanted[i]:04d}.png'),im,[cv2.IMWRITE_PNG_COMPRESSION,1])
    cap.release();save_json(out/'index.json',dict(key=key,frame_indices=ids.tolist(),times=ts[ids].tolist(),width=width,height=height,source_active_rect=[x,y,w,h],elapsed_seconds=time.perf_counter()-start))
    return key+f' {len(ids)} frames {time.perf_counter()-start:.1f}s'
if __name__=='__main__':
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:
        for result in pool.map(work,KEYS):print(result,flush=True)
