"""Choose a FLIR rule on designated TRAIN development clips, then audit once.

Reference labels are confined to this evaluator. Prediction functions never use them.
The audit clips were used by the OLD speed regressor; they are not a pristine holdout.
"""
import argparse,json,sys
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter1d
from domain_fusion_v2 import ROOT,load_case,predict,planar_yaw,add_extra_heading
from public_dataset import load_manifest,load_reference,sha256
from metrics import score_trajectory

def get_case(key):
    base=ROOT/'runs/domain_probe_v2'
    case=load_case(key,base/'baseline',{k:base/k for k in ['r3','vggt','klt']},'flir-VIS')
    sigma=2/np.median(np.diff(case['t']))
    for name in ['dpvo_raw','dpvo_clahe','vggt_clahe_crop']:
        add_extra_heading(case,base/name,name)
    if {'dpvo_raw','dpvo_clahe'}<=set(case['headings']):
        case['headings']['median_dpvo_vggt']=np.median([case['headings'][k] for k in ['dpvo_raw','dpvo_clahe','vggt']],axis=0)
    return case

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--group',choices=['development','audit'],required=True);a=ap.parse_args()
    report=ROOT/'reports/domain_probe_v2';plan=json.loads((report/'experiment_plan.json').read_text())
    manifest=load_manifest(ROOT/'data/release/release_manifest.json');clips={c['key']:c for c in manifest['clips']}
    keys=plan[a.group+'_train_keys'];cases={k:get_case(k) for k in keys}
    if a.group=='development':
        choices=[('baseline','baseline')]+[(h,'constant') for h in ['baseline','r3','vggt','median','vggt_relative','dpvo_raw','dpvo_clahe','vggt_clahe_crop','median_dpvo_vggt','dpvo_position','dpvo_hybrid']]
        choices=[(h,s) for h,s in choices if all(h in case['headings'] for case in cases.values())]
    else:
        selection=json.loads((report/'selection.json').read_text())
        choices=[('baseline','baseline'),(selection['heading'],selection['speed'])]
    rows=[]
    for h,s in choices:
        values=[]
        for key,case in cases.items():
            ref_t,ref_p=load_reference(ROOT/'data/release',clips[key]);p=predict(case,h,s)
            result=score_trajectory(case['t'],np.array(p['positions']),ref_t,ref_p)
            values.append(dict(key=key,drift_pct=result['drift_pct']))
        row=dict(heading=h,speed=s,mean=float(np.mean([v['drift_pct'] for v in values])),worst=max(v['drift_pct'] for v in values),clips=values)
        rows.append(row);print(json.dumps(row),flush=True)
    (report/(a.group+'_results.json')).write_text(json.dumps(rows,indent=2))
    if a.group=='development':
        best=min(rows,key=lambda r:r['mean'])
        path=report/'selection.json'
        with path.open('x') as f:json.dump({**best,'selection_keys':keys,'criterion':'Minimum macro mean on four prespecified TRAIN development clips','frozen_before_audit':True,'plan_sha256':sha256(report/'experiment_plan.json')},f,indent=2)
        print('SELECTED',best['heading'],best['speed'],flush=True)

if __name__=='__main__':main()
