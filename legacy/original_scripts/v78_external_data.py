"""Frozen, time-selected external validation data. No reference data in inference release."""
import argparse, concurrent.futures as cf, datetime, hashlib, io, json, struct, time
import urllib.request, xml.etree.ElementTree as ET, zipfile, zlib, subprocess
from pathlib import Path
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/domain_v78_external'
DATA=ROOT/'data/external_v78'
REP=ROOT/'reports/domain_v78_external'
SURVEY=ROOT/'reports/external_validation_survey'
SOURCES=[('clip_1000','pohang00','thermal_day'),('clip_1001','pohang01','thermal_night'),
         ('clip_1002','canoe-2025-08-20-16-07','rgb_lake'),('clip_1003','canoe-2025-08-28-14-13','rgb_reservoir')]
PB='https://pohang-canal-dataset.s3.amazonaws.com/'
CB='https://canoe-data.s3.us-west-2.amazonaws.com/'

def save(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def get(url,headers=None):
    for retry in range(6):
        try:
            with urllib.request.urlopen(urllib.request.Request(url,headers=headers or {}),timeout=90) as r:
                if headers and 'Range' in headers:assert r.status==206,(url,r.status)
                return r.read()
        except Exception:
            if retry==5:raise
            time.sleep(1+retry)

def download(url,path):
    if not path.exists():
        path.parent.mkdir(parents=True,exist_ok=True);b=get(url);q=path.with_suffix(path.suffix+'.part');q.write_bytes(b);q.replace(path)
    return path

def freeze():
    p=REP/'plan.json'
    if p.exists():return
    names=['v24_pack_uncapped.py','v13_infer.py','domain_fusion_v2.py','fuse_headings.py','v65_submission_inputs.py','v65_static_slam.py','v65_complete_masks.py','v64_course_support_envelope.py','v50_multiview_course.py','v53_course_uncertainty.py','v19_motion.py','v73_pack_independent_speed.py','v74_observable_speed.py','v76_direction_guarded_speed.py','v6_test_dpvo.py','v6_test_models.py','v5_dpvo_masked.py','calibrate_domain_v4.py','submission_r3_infer_clip.py','submission_vggt_infer_clip.py','pi3x_domain_infer.py']
    files=[ROOT/'scripts'/n for n in names]+[ROOT/'runs/domain_v5/segmentation/model.json',ROOT/'repos/DPVO/config/default.yaml',ROOT/'reports/domain_v3/pi3x_source.json']
    save(p,dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        purpose='First external zero-retuning validation. Predictions frozen before reference scoring.',
        methods=['V24','V65','V77','V76_unfiltered_quarter'],
        primary_comparison='Pohang thermal: frozen FLIR branch. CANOE RGB: transfer stress test of the same constant-base-speed branch, NOT exact official wide-camera routing.',
        sequences=[dict(key=k,recording=s,modality=m,offset_seconds=600,duration_seconds=300) for k,s,m in SOURCES],
        selection='Predeclared time window [600,900) seconds after first camera timestamp, first frame at or after each 0.5-second target. No reference positions, errors, or image content used for selection. 4 recordings, one window each. Remaining recordings and windows reserved.',
        inputs='Monocular camera pixels, camera calibration, camera timestamps only. References stored separately and read only by evaluation process.',
        preprocessing='Native supplied 8-bit intensity unchanged. If uint16, fixed whole-clip 1/99 percentile linear display map, no per-frame mapping. Pohang undistorted with supplied five-coefficient Brown model, unchanged K; CANOE already rectified. Lossless RGB mp4 and 960-wide cache preserve aspect. No CLAHE.',
        adapters='Existing generic image model CLIs consume all preselected 2Hz frames using --fps100 to prevent jitter-induced second downsampling; actual timestamps retained. Static DPV-SLAM mask resize generalized to input aspect ratio. Shared original postprocessing with redirected inputs. Verify original TRAIN parity before external scoring.',
        baseline='Three raw DPVO seeds0,104729,209759, then family median with VGGT and Pi3X; Gaussian2sec. Base speed1. R3 supplies pose geometry and original planar coordinate convention. AnyCalib9-frame learned K. No provided navigation speed/heading used.',
        evaluation='Official challenge score_trajectory when supported: one global Sim3 per entire selected 5-minute trajectory, 100-800m segment counts. Also globally aligned ATE and translation RPE at1/10/30sec, 10/25/50m short segment diagnostics. No per-segment realignment. Separate dataset/modality means and full failed-case ledger.',
        limitations=['Pohang SLAM baseline is a proxy reference, not perfect ground truth.','Four fixed windows are a pilot, not the full benchmarks. Two Pohang recordings repeat a route.','CANOE RGB tests branch transfer, not complete deployed wide-camera pipeline.','Pretrained model data overlap not audited.','External scores do not predict official MaCVi percentage.'],
        dependencies={str(q.relative_to(ROOT)):sha(q) for q in files},
        preserved_submissions={str(q.relative_to(ROOT)):sha(q) for q in sorted((ROOT/'submissions').glob('*/submission.json'))}))
    print('FROZEN',p,flush=True)

def listing(prefix,start_after=None):
    import urllib.parse
    params={'list-type':'2','prefix':prefix,'max-keys':'1000'}
    if start_after:params['start-after']=start_after
    r=ET.fromstring(get(CB+'?'+urllib.parse.urlencode(params)))
    ns={'s':'http://s3.amazonaws.com/doc/2006-03-01/'}
    return [x.find('s:Key',ns).text for x in r.findall('s:Contents',ns)]

def select(key,seq,modality):
    p=DATA/'selection'/f'{key}.json'
    if p.exists():return json.loads(p.read_text())
    if seq.startswith('pohang'):
        ts=np.loadtxt(SURVEY/f'{seq}_timestamp.txt');epoch=ts[0,0];times=ts[:,0]
        ids=np.searchsorted(times,epoch+600+np.arange(600)/2)
        items={r['name']:r for r in json.loads((SURVEY/f'{seq}_infrared_index.json').read_text())}
        rows=[dict(timestamp=float(times[i]),source_name=f'infrared/images/{int(ts[i,1]):06d}.png',zip_entry=items[f'infrared/images/{int(ts[i,1]):06d}.png']) for i in ids]
    else:
        prefix=seq+'/cam_left/';first=listing(prefix);first=sorted(x for x in first if x.endswith('.png'));epoch=int(Path(first[0]).stem)/1e6
        names=[];cursor=prefix+str(int((epoch+600)*1e6)-1)+'.png'
        while True:
            batch=listing(prefix,cursor);assert batch
            names.extend(x for x in batch if x.endswith('.png'))
            if int(Path(names[-1]).stem)/1e6>=epoch+901:break
            cursor=batch[-1]
        times=np.array([int(Path(x).stem)/1e6 for x in names]);ids=np.searchsorted(times,epoch+600+np.arange(600)/2)
        rows=[dict(timestamp=float(times[i]),source_name=names[i]) for i in ids]
    assert len(rows)==600 and len({q['source_name'] for q in rows})==600
    rel=np.array([q['timestamp']-rows[0]['timestamp'] for q in rows]);assert np.diff(rel).min()>.1 and np.diff(rel).max()<1
    result=dict(key=key,recording=seq,modality=modality,recording_epoch=epoch,clip_epoch=rows[0]['timestamp'],times=rel.tolist(),rows=rows)
    save(p,result);print('SELECTED',key,seq,rows[0]['timestamp']-epoch,rel[-1],flush=True);return result

def zip_image(seq,row):
    e=row['zip_entry'];start=e['offset'];n=e['compressed']+512
    raw=get(PB+seq+'/infrared.zip',{'Range':f'bytes={start}-{start+n-1}'})
    assert raw[:4]==b'PK\x03\x04'
    nl,el=struct.unpack_from('<HH',raw,26);off=30+nl+el;payload=raw[off:off+e['compressed']]
    assert len(payload)==e['compressed']
    b=zlib.decompress(payload,-15) if e['compression']==8 else payload
    assert len(b)==e['size'] and zlib.crc32(b)==e['crc'];return b

def prepare(key,seq,modality):
    freeze();sel=select(key,seq,modality);native=DATA/'native'/key;native.mkdir(parents=True,exist_ok=True)
    def fetch(pair):
        i,row=pair;p=native/f'{i:04d}.png'
        if not p.exists():
            b=zip_image(seq,row) if seq.startswith('pohang') else get(CB+row['source_name']);p.write_bytes(b)
        return i
    with cf.ThreadPoolExecutor(max_workers=12) as pool:
        for n,i in enumerate(pool.map(fetch,enumerate(sel['rows']))):
            if n%100==0:print('DOWNLOAD',key,n,600,flush=True)
    im=cv2.imread(str(native/'0000.png'),cv2.IMREAD_UNCHANGED);assert im is not None;h,w=im.shape[:2]
    if seq.startswith('pohang'):
        cp=download(PB+seq+'/calibration.zip',DATA/'calibration'/f'{seq}.zip')
        with zipfile.ZipFile(cp) as z:cal=json.loads(z.read('calibration/intrinsics.json'))['infrared']
        f=cal['focal_length'];k=np.array([[f,0,cal['cc_x']],[0,f,cal['cc_y']],[0,0,1.]])
        d=np.array(cal['distortion_coefficients']);maps=cv2.initUndistortRectifyMap(k,d,None,k,(w,h),cv2.CV_32FC1)
    else:
        cp=download(CB+seq+'/calib/P_cam.txt',DATA/'calibration'/f'{seq}_P_cam.txt');k=np.loadtxt(cp)[:3,:3];maps=None
    display=None
    if im.dtype==np.uint16:
        pixels=np.concatenate([cv2.imread(str(native/f'{i:04d}.png'),cv2.IMREAD_UNCHANGED)[::16,::16].ravel() for i in range(600)])
        display=np.percentile(pixels,[1,99]);assert display[1]>display[0]
    else:assert im.dtype==np.uint8,im.dtype
    frame=OUT/'frames'/key;frame.mkdir(parents=True,exist_ok=True);rect=DATA/'rectified'/key;rect.mkdir(parents=True,exist_ok=True)
    out_h=round(h*960/w);hashes=[]
    for i in range(600):
        image=cv2.imread(str(native/f'{i:04d}.png'),cv2.IMREAD_UNCHANGED)
        if display is not None:image=np.clip((image.astype(float)-display[0])/(display[1]-display[0])*255,0,255).astype(np.uint8)
        if image.ndim==2:image=cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
        assert image.shape==(h,w,3)
        if maps is not None:image=cv2.remap(image,*maps,cv2.INTER_LINEAR)
        assert cv2.imwrite(str(rect/f'{i:04d}.png'),image)
        assert cv2.imwrite(str(frame/f'{i:04d}.png'),cv2.resize(image,(960,out_h),interpolation=cv2.INTER_AREA))
        hashes.append(sha(native/f'{i:04d}.png'))
    video=DATA/'release/videos'/f'{key}.mp4';video.parent.mkdir(parents=True,exist_ok=True)
    if not video.exists():
        subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-framerate','2','-i',str(rect/'%04d.png'),'-c:v','libx264rgb','-crf','0','-preset','fast','-threads','4',str(video)],check=True)
    # Lossless decode check on first, middle, last; exact frame count.
    cap=cv2.VideoCapture(str(video));assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT))==600
    for idx in [0,300,599]:
        cap.set(cv2.CAP_PROP_POS_FRAMES,idx);ok,actual=cap.read();assert ok and np.array_equal(actual,cv2.imread(str(rect/f'{idx:04d}.png')))
    cap.release()
    save(frame/'index.json',dict(key=key,times=sel['times'],frame_indices=list(range(600)),width=960,height=out_h,actual_fps=2,source=seq,source_timestamps=[q['timestamp'] for q in sel['rows']]))
    save(DATA/'release/timing'/f'{key}.json',dict(key=key,times=sel['times']))
    cal=dict(fx=float(k[0,0]),fy=float(k[1,1]),cx=float(k[0,2]),cy=float(k[1,2]),width=w,height=h,container_width=w,container_height=h,active_rect=[0,0,w,h],provenance='measured')
    clip=dict(key=key,split='test',camera_mode=modality,calibration=cal,video=f'videos/{key}.mp4',timestamps=f'timing/{key}.json',reference=None)
    save(DATA/'release'/f'{key}_manifest.json',dict(schema_version=1,version='external-v78',counts=dict(train=0,val=0,test=1,total=1),clips=[clip]))
    save(REP/'data'/f'{key}.json',dict(source=seq,frames=600,native_shape=list(im.shape),native_dtype=str(im.dtype),display_limits=None if display is None else display.tolist(),calibration=cal,native_sha256=hashes,video_sha256=sha(video),lossless_decode_check=True,reference_loaded=False))
    print('DATA_READY',key,im.shape,im.dtype,flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['freeze','prepare']);ap.add_argument('--key');a=ap.parse_args()
    if a.action=='freeze':freeze()
    else:
        cv2.setNumThreads(1)
        for k,s,m in SOURCES:
            if a.key is None or a.key==k:prepare(k,s,m)
