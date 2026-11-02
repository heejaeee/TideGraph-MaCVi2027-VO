"""Reference-free fusion ablations; original v1 artifacts remain immutable."""
import argparse, json, sys
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation
from fuse_headings import planar_yaw

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'repos/lookout-vo-challenge/starter'))
from public_dataset import load_manifest, select_clips, load_timestamps, validate_prediction, write_json

def relative_window_yaw(source,key,n):
    """Fuse same-frame relative rotations before integrating; no overlap seam jumps."""
    if not (source/'windows'/key).exists():
        source=Path(json.loads((source/'diagnostics'/f'{key}.json').read_text())['arguments']['output'])
    increments=[[] for _ in range(n-1)]
    for file in sorted((source/'windows'/key).glob('*.npz')):
        with np.load(file) as z:
            start=int(z['start']); rot=z['c2w'][:,:3,:3]
        delta=Rotation.from_matrix(rot[:-1].transpose(0,2,1)@rot[1:]).as_rotvec()
        for j,v in enumerate(delta): increments[start+j].append(v)
    if any(not x for x in increments):raise ValueError('Incomplete VGGT windows')
    poses=np.tile(np.eye(4),(n,1,1))
    for i,values in enumerate(increments):
        poses[i+1,:3,:3]=poses[i,:3,:3]@Rotation.from_rotvec(np.median(values,axis=0)).as_matrix()
    return planar_yaw(poses)

def load_case(key,baseline,sources,mode):
    pred=json.loads((baseline/'predictions'/f'{key}.json').read_text())
    t=np.array(pred['times']); sigma=2/np.median(np.diff(t))
    with np.load(baseline/'diagnostics'/f'{key}.fusion.npz') as z:
        base_yaw=z['yaw'].copy(); speed=z['speed'].copy()
    headings={'baseline':base_yaw}
    for name,source in sources.items():
        p=json.loads((source/'predictions'/f'{key}.json').read_text())
        with np.load(source/'diagnostics'/f'{key}.poses.npz') as z:
            yaw=planar_yaw(z['camera_to_world'])
        headings[name]=gaussian_filter1d(np.interp(t,p['times'],yaw),sigma)
    headings['median']=np.median([headings[k] for k in ['r3','vggt','klt']],axis=0)
    headings['vggt_relative']=gaussian_filter1d(relative_window_yaw(sources['vggt'],key,len(t)),sigma)
    records=json.loads((baseline/'inference_manifest.json').read_text())['clips']
    correlation=next(r['speed_correlation'] for r in records if r['key']==key)
    return dict(key=key,pred=pred,t=t,headings=headings,speed=speed,mode=mode,speed_correlation=correlation)

def predict(case,heading='baseline',speed='baseline',scope='all'):
    change=scope=='all' or (scope=='flir' and case['mode'].startswith('flir')) or (scope=='narrow' and case['mode']!='wide')
    if change and heading in ['dpvo_position','dpvo_hybrid']:
        source='dpvo_clahe'
        if source in case.get('raw_positions',{}) and (heading=='dpvo_position' or case['spike_rates'][source]<.01):
            return {**case['pred'],'positions':case['raw_positions'][source].tolist()}
        heading='vggt';speed='constant'
    yaw=case['headings'][heading if change else 'baseline']
    s=case['speed'].copy()
    if change:
        if speed=='constant':s=np.ones_like(s)
        elif speed=='shrink':s=.5*s+.5
        elif speed=='agreement' and not case['speed_correlation']>=.7:s=np.ones_like(s)
    vel=np.stack([np.sin(yaw),np.zeros_like(yaw),np.cos(yaw)],axis=1)*s[:,None]
    t=case['t']; positions=np.vstack([np.zeros(3),np.cumsum(.5*(vel[:-1]+vel[1:])*np.diff(t)[:,None],axis=0)])
    return {**case['pred'],'positions':positions.tolist()}

def add_extra_heading(case,source,name):
    key=case['key'];pp=source/'predictions'/f'{key}.json';zp=source/'diagnostics'/f'{key}.poses.npz'
    if not pp.exists() or not zp.exists():return False
    p=json.loads(pp.read_text())
    with np.load(zp) as z:yaw=planar_yaw(z['camera_to_world'])
    if not np.isfinite(yaw).all():return False
    case['headings'][name]=gaussian_filter1d(np.interp(case['t'],p['times'],yaw),2/np.median(np.diff(case['t'])))
    if name.startswith('dpvo'):
        positions=np.array(p['positions']);t=np.array(p['times'])
        speed=np.linalg.norm(np.diff(positions,axis=0),axis=1)/np.diff(t)
        case.setdefault('spike_rates',{})[name]=float(np.mean(speed>5*max(np.median(speed),1e-8)))
        case.setdefault('raw_positions',{})[name]=np.column_stack([np.interp(case['t'],t,positions[:,j]) for j in range(3)])
        if name=='dpvo_clahe':
            for alias in ['dpvo_position','dpvo_hybrid']:case['headings'][alias]=case['headings'][name]
    return True

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dataset',type=Path,required=True);ap.add_argument('--split',required=True)
    for name in ['baseline','r3','vggt','klt','output']:ap.add_argument('--'+name,type=Path,required=True)
    ap.add_argument('--heading',choices=['baseline','r3','vggt','median','vggt_relative'],default='baseline')
    ap.add_argument('--speed',choices=['baseline','constant','shrink','agreement'],default='baseline')
    ap.add_argument('--scope',choices=['all','flir','narrow'],default='all')
    a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    sources={k:getattr(a,k) for k in ['r3','vggt','klt']}
    for clip in select_clips(load_manifest(a.dataset),a.split):
        case=load_case(clip['key'],a.baseline,sources,clip['camera_mode'])
        pred=predict(case,a.heading,a.speed,a.scope)
        validate_prediction(pred,clip['key'],load_timestamps(a.dataset.parent,clip))
        write_json(a.output/'predictions'/f"{clip['key']}.json",pred)
    write_json(a.output/'settings.json',{k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()})

if __name__=='__main__':main()
