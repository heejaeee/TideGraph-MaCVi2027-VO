"""Portable CPU reproduction of TideGraph #19226 from included estimates.

replay: final yaw/speed -> positions; rebuild: Omega window rotations and
retained legacy speed/IR signals -> graph/TV/gain -> positions.
Neither command is raw-video inference.
"""
import argparse,hashlib,json,sys,time
from pathlib import Path
import numpy as np
import core
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'vendor/lookout_starter'))
from public_dataset import load_manifest,load_timestamps,validate_prediction


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,value):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def verify():
    manifest=read(ROOT/'PACKAGE_MANIFEST.json')
    for name,expected in manifest['files'].items():
        path=ROOT/name
        if not path.is_file() or sha(path)!=expected:raise ValueError('Missing or modified package file: '+name)
    print('PACKAGE_OK',len(manifest['files']),'files')


def run(action,output,config_path):
    output=Path(output).resolve()
    if (output/'submission.json').exists():raise FileExistsError('Choose a new output directory; refusing overwrite')
    config=read(config_path);meta=read(ROOT/'snapshot/metadata/clip_index.json')
    manifest=load_manifest(ROOT/'snapshot/metadata/release_manifest.json')
    clips={c['key']:c for c in manifest['clips']};bundle={};checks=[];tic=time.perf_counter()
    for key in config['clip_order']:
        m=meta[key];n=len(m['times']);t=np.array(m['times']);mode=m['camera_mode'];detail={}
        if action=='replay':
            with np.load(ROOT/f'snapshot/final_signals/{key}.npz',allow_pickle=False) as z:
                np.testing.assert_array_equal(t,z['times']);yaw=z['yaw'];speed=z['speed']
        else:
            with np.load(ROOT/f'snapshot/inputs/{key}.npz',allow_pickle=False) as z:
                np.testing.assert_array_equal(t,z['times']);anchor=float(z['initial_yaw']);speed=z['speed']
                if mode=='flir-IR':yaw=z['legacy_yaw']
            if mode=='wide':
                raw=core.relative_window_yaw(ROOT/'snapshot/omega',key,n)
                yaw=core.gaussian_filter1d(raw,config['yaw_sigma_seconds']/np.median(np.diff(t)))
                yaw=yaw-yaw[0]+anchor
                speed,detail=core.apply(t,speed,'tv',config['wide_tv_lambda'])
            elif mode=='flir-VIS':
                raw,i,j,z,w=core.chain_and_edges(ROOT/'snapshot/omega',key,n)
                raw,detail=core.solve_yaw(n,i,j,z,raw,delta_deg=config['graph_huber_degrees'])
                if not detail['converged']:raise RuntimeError('Yaw solver did not converge: '+key)
                yaw=core.gaussian_filter1d(raw,config['yaw_sigma_seconds']/np.median(np.diff(t)))
                yaw+=anchor-yaw[0]
                gain=config['flir_gains']['A' if key in config['flir_groups']['A'] else 'B']
                yaw=anchor+gain*(yaw-anchor)
            elif mode!='flir-IR':raise ValueError(mode)
        positions=core.integrate(t,yaw,speed)
        pred=dict(key=key,frame_indices=m['frame_indices'],times=m['times'],positions=positions.tolist())
        released=load_timestamps(ROOT/'snapshot/metadata',clips[key]);validate_prediction(pred,key,released)
        coverage=float((min(t[-1],released[-1])-max(t[0],released[0]))/(released[-1]-released[0]))
        if coverage<.95 or np.diff(t).max()>1:raise ValueError('Coverage/gap failure: '+key)
        bundle[key]=pred;checks.append(dict(key=key,mode=mode,coverage=coverage,max_gap_s=float(np.diff(t).max()),solver=detail))
        print(action,key,flush=True)
    output.mkdir(parents=True,exist_ok=True)
    encoded=(json.dumps(bundle,separators=(',',':'),allow_nan=False)+'\n').encode()
    with (output/'submission.json').open('xb') as f:f.write(encoded)
    # Compare only after all predictions are computed. This is an estimate
    # archive, not hidden TEST ground truth, and is never an inference input.
    expected=read(ROOT/'artifacts/submission_19226.json');worst=0.;rows=[]
    for key,pred in bundle.items():
        e=expected[key];assert all(pred[k]==e[k] for k in ['key','times','frame_indices'])
        error=float(np.max(abs(np.asarray(pred['positions'])-np.asarray(e['positions']))))
        worst=max(worst,error);rows.append(dict(key=key,max_position_error=error))
    actual_sha=sha(output/'submission.json');expected_sha=sha(ROOT/'artifacts/submission_19226.json')
    report=dict(action=action,clips=len(bundle),max_position_error=worst,sha256=actual_sha,
                expected_sha256=expected_sha,byte_identical=actual_sha==expected_sha,
                numerical_match=worst<1e-8,elapsed_seconds=time.perf_counter()-tic,
                time_scope='CPU cached-signal reproduction only, NOT video throughput/FPS',checks=checks,parity=rows)
    save(output/'validation.json',report)
    print(json.dumps({k:v for k,v in report.items() if k not in ['checks','parity']},indent=2))
    if config_path.resolve()==(ROOT/'configs/best19226.json').resolve() and not report['numerical_match']:
        raise RuntimeError('Best-config reproduction differs; inspect validation.json')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['verify','replay','rebuild'])
    parser.add_argument('--output',type=Path)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/best19226.json')
    args=parser.parse_args()
    if args.action=='verify':verify()
    else:
        if args.output is None:parser.error('--output is required')
        run(args.action,args.output,args.config)
