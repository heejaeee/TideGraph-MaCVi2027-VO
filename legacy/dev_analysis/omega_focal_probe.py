"""Omega's own focal estimate per clip (dev vs test), to test a zoom/FOV difference.

Same frames/preprocessing as omega_windows.py (512 flir-VIS / 416 wide balanced), a few
evenly spaced 16-frame windows per clip; reports median predicted fx scaled to the 1920px
source width, next to the release's nominal fx (assumes a 1920x1080 active_rect, no aspect crop).
--dev_clips N limits dev to N evenly spaced dev_split clips of that mode (wide_val / flir_vis_val).

usage: .venv-gpu/bin/python tools/omega_focal_probe.py --out outputs/focal_probe.json
       .venv-gpu/bin/python tools/omega_focal_probe.py --mode wide --dev_clips 16 --out outputs/focal_probe_wide.json
"""
import argparse,json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
import omega_windows as ow

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--windows',type=int,default=8)
    ap.add_argument('--mode',default='flir-VIS',choices=list(ow.RES));ap.add_argument('--dev_clips',type=int)
    a=ap.parse_args()
    sys.path.insert(0,str(ow.DATA['dev']/'starter'))
    from public_dataset import load_manifest,load_timestamps
    torch,preprocess,run=ow.omega_backend(ow.BACKENDS['omega'][1]);model=run.model
    from vggt_omega.utils.pose_enc import encoding_to_camera
    res=ow.RES[a.mode];split=json.loads((ow.ROOT/'configs/dev_split.json').read_text())
    val=split['wide_val' if a.mode=='wide' else 'flir_vis_val']
    if a.dev_clips:val=[val[i] for i in np.linspace(0,len(val)-1,a.dev_clips).round().astype(int)]
    rows=[]
    for data in ['dev','test']:
        for clip in load_manifest(ow.MANIFEST[data])['clips']:
            if clip['camera_mode']!=a.mode or (data=='dev' and a.dev_clips and clip['key'] not in val):continue
            ts=load_timestamps(ow.DATA[data],clip);ids=ow.frame_ids(ts);n=len(ids)
            starts=np.linspace(0,n-16,a.windows).astype(int);need=sorted({int(ids[s+j]) for s in starts for j in range(16)})
            fr=dict(zip(need,ow.frames(ow.DATA[data],clip,np.array(need))))
            fx=[];fy=[]
            for s in starts:
                batch=torch.stack([preprocess(fr[int(ids[s+j])],res) for j in range(16)])
                with torch.inference_mode():
                    out=model(batch.cuda());_,K=encoding_to_camera(out['pose_enc'].float(),batch.shape[-2:])
                K=K[0].cpu().numpy();h,w=batch.shape[-2:]
                fx+=list(K[:,0,0]*1920/w);fy+=list(K[:,1,1]*1080/h)
            x0,y0,w0,h0=clip['calibration']['active_rect']
            row=dict(data=data,key=clip['key'],fx_median=float(np.median(fx)),fx_iqr=[float(np.quantile(fx,.25)),float(np.quantile(fx,.75))],
                     fy_median=float(np.median(fy)),nominal_fx=clip['calibration']['fx'],active_rect=[x0,y0,w0,h0])
            rows.append(row);print(f"{data:4s} {clip['key']} fx {row['fx_median']:7.1f} (IQR {row['fx_iqr'][0]:.0f}-{row['fx_iqr'][1]:.0f})  fy {row['fy_median']:7.1f}  nominal {row['nominal_fx']}",flush=True)
    for d in ['dev','test']:
        v=[r['fx_median'] for r in rows if r['data']==d];print(d,'median fx %.1f  range %.1f-%.1f'%(np.median(v),min(v),max(v)))
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(rows,indent=1))

if __name__=='__main__':main()
