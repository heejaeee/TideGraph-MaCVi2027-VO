"""Shared v5 image-only primitives. No trajectory reference loading here."""
import json,sys
from pathlib import Path
import numpy as np
import cv2
from scipy.ndimage import gaussian_filter1d
from evaluate_domain_train import get_case
from domain_fusion_v2 import ROOT
from fuse_headings import planar_yaw
from public_dataset import load_manifest,load_timestamps,write_json
from rotation_flow_infer_clip import fit_rotation
from domain_fusion_v3 import integrate_velocity
RUN=ROOT/'runs/domain_v5'; REPORT=ROOT/'reports/domain_v5'
CLIPS={c['key']:c for c in load_manifest(ROOT/'data/release/release_manifest.json')['clips']}
TRAIN=[k for k,c in CLIPS.items() if c['split']=='train' and c['camera_mode']=='flir-VIS']
SEEDS=[0,104729,209759]
def save_json(p,d):
    import os
    p=Path(p);temp=p.with_name(p.name+f'.tmp.{os.getpid()}');write_json(temp,d);temp.replace(p)
def frames(key):
    folder=RUN/'frames'/key
    meta=json.loads((folder/'index.json').read_text())
    return folder,meta

def bearings(p,k):
    a=np.c_[p,np.ones(len(p))]@np.linalg.inv(k).T
    return a/np.maximum(np.linalg.norm(a,axis=1,keepdims=True),1e-12)

def rotation_yaw(rotations):
    poses=np.tile(np.eye(4),(len(rotations)+1,1,1))
    for i,r in enumerate(rotations): poses[i+1,:3,:3]=poses[i,:3,:3]@r.T
    return planar_yaw(poses)

def make_prediction(key,times,rotations,counts,blend=.5,fallback=True,missing=None):
    case=get_case(key);t=case['t'];base=case['headings']['r3'];y=rotation_yaw(rotations)
    valid=np.asarray(counts)[:,-1]>=8
    if missing is not None: valid &= ~missing
    if fallback:
        delta=np.diff(y);base_delta=np.diff(np.interp(times,t,base));delta=np.where(valid,delta,base_delta)
        y=np.r_[0,np.cumsum(delta)]
    y=gaussian_filter1d(np.interp(t,times,y),2/np.median(np.diff(t)))
    yaw=blend*base+(1-blend)*y
    speed=case['speed'] if case['speed_correlation']>=.7 else np.ones(len(t))
    p=integrate_velocity(t,yaw,speed)
    return {**case['pred'],'positions':p.tolist()},dict(low_support=int((~valid).sum()),pairs=len(valid),yaw=y)
