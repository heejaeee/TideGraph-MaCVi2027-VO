"""Run the validation-frozen visual components on TEST, preserving stage provenance."""
import argparse,json,os,shutil,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'repos/lookout-vo-challenge/starter'))
from public_dataset import load_manifest,select_clips,write_json,sha256
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['gpu','cpu'],required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    data=ROOT/'data/lookout-vo-test-v0.1/release_manifest.json';clips=select_clips(load_manifest(data),'test');assert len(clips)==20 and all(c['reference'] is None for c in clips)
    run=a.output.resolve();run.mkdir(parents=True,exist_ok=True);prov=run/'provenance'/a.stage;prov.mkdir(parents=True,exist_ok=True);logs=run/'logs';logs.mkdir(exist_ok=True)
    script_names=['run_submission_inference.py','submission_r3_infer_clip.py','submission_vggt_infer_clip.py','submission_rotation_infer_clip.py','submission_water_infer_clip.py','submission_extract_features.py','rotation_flow_infer_clip.py','extract_speed_features_v2.py','probe_pairs.py','window_geometry.py']
    for name in script_names:
        src=ROOT/'scripts'/name;dst=prov/name
        if dst.exists():assert sha256(src)==sha256(dst),f'Source changed during resume: {name}'
        else:shutil.copy2(src,dst)
    env=os.environ.copy();env['PYTHONPATH']=str(ROOT/'repos/R3_submission_original')+os.pathsep+env.get('PYTHONPATH','')
    revision=subprocess.check_output(['git','-C',str(ROOT/'repos/R3_submission_original'),'rev-parse','HEAD'],text=True).strip()
    assert revision=='e345f1112fbc9c451d44c9c5aa22d9bfec2a954d'
    record_path=run/f'{a.stage}_manifest.json'
    record=json.loads(record_path.read_text()) if record_path.exists() else dict(stage=a.stage,dataset_sha256=sha256(data),r3_revision=revision,settings='Frozen final validation recipe; no TEST training or label access',clips=[])
    done={(r['key'],r['component']) for r in record['clips'] if r['returncode']==0}
    components=[('r3','submission_r3_infer_clip.py'),('vggt','submission_vggt_infer_clip.py')] if a.stage=='gpu' else [('klt','submission_rotation_infer_clip.py'),('water','submission_water_infer_clip.py'),('features','submission_extract_features.py')]
    for key in [c['key'] for c in clips]:
        for component,script in components:
            if (key,component) in done:continue
            out=run/component;out.mkdir(exist_ok=True)
            print('START',key,component,flush=True);began=time.perf_counter()
            cmd=[sys.executable,'-u',str(ROOT/'scripts'/script),'--dataset',str(data),'--key',key,'--output',str(out)]
            with (logs/f'{key}.{component}.log').open('x') as log:r=subprocess.run(cmd,env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            record['clips'].append(dict(key=key,component=component,returncode=r.returncode,seconds=time.perf_counter()-began,command=cmd));write_json(record_path,record)
            print('FINISH',key,component,r.returncode,flush=True)
            if r.returncode:raise RuntimeError(f'{key} {component} failed; inspect log')
    print('STAGE_COMPLETE',a.stage,flush=True)
if __name__=='__main__':main()
