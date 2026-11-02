"""TEST-only data plumbing for the frozen v6 candidate. Never loads references."""
import json,sys
from pathlib import Path
import cv2,numpy as np
from scipy.ndimage import gaussian_filter1d
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'repos/lookout-vo-challenge/starter'))
from public_dataset import load_manifest,load_timestamps,validate_prediction,write_json,sha256
from fuse_headings import planar_yaw
from v5_common import bearings,rotation_yaw,save_json,SEEDS
DATA=ROOT/'data/lookout-vo-test-v0.1';MANIFEST=DATA/'release_manifest.json'
CLIPS={c['key']:c for c in load_manifest(MANIFEST)['clips']}
KEYS=[k for k,c in CLIPS.items() if c['camera_mode']=='flir-VIS']
RUN=ROOT/'runs/domain_v6_test';REPORT=ROOT/'reports/domain_v6_test'
BASE=ROOT/'runs/submission_test_v1/fused';V2=ROOT/'runs/domain_test_v2/gated';R3=ROOT/'runs/submission_test_v1/r3'

def frames(key):
    folder=RUN/'frames'/key;return folder,json.loads((folder/'index.json').read_text())

def window_starts(n):
    starts=list(range(0,max(1,n-31),16))
    if starts[-1]!=max(0,n-32):starts.append(max(0,n-32))
    return starts

def get_case(key):
    pred=json.loads((V2/'predictions'/f'{key}.json').read_text());t=np.array(pred['times'])
    rp=json.loads((R3/'predictions'/f'{key}.json').read_text());poses=np.load(R3/'diagnostics'/f'{key}.poses.npz')['camera_to_world'];yaw=gaussian_filter1d(np.interp(t,rp['times'],planar_yaw(poses)),2/np.median(np.diff(t)))
    with np.load(BASE/'diagnostics'/f'{key}.fusion.npz') as z:speed=z['speed'].copy()
    routing=json.loads((BASE/'inference_manifest.json').read_text())['clips'];corr=next(x['speed_correlation'] for x in routing if x['key']==key)
    return dict(pred=pred,t=t,speed=speed,speed_correlation=corr,r3_yaw=yaw)
