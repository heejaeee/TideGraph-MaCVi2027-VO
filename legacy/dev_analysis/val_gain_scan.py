"""Per-clip yaw-gain-only scan (FLIR-VIS) on regenerated windows; same yaw as val_baseline."""
import json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
import val_baseline as vb
from val_baseline import core,metrics,load_manifest,load_reference,DEV

def main(windows,out):
    windows=Path(windows);clips={c['key']:c for c in load_manifest(DEV/'release_manifest.json')['clips']}
    G=np.round(np.arange(.2,1.61,.02),2);rows=[]
    for info in sorted((windows/'clips').glob('*.json')):
        m=json.loads(info.read_text())
        if m['mode']!='flir-VIS':continue
        key=m['key'];t=np.array(m['times']);n=len(t);rt,rP=load_reference(DEV,clips[key])
        raw,_=vb.raw_yaw(windows,key,'flir-VIS',n,3.);y=core.gaussian_filter1d(raw,2/np.median(np.diff(t)));y-=y[0]
        row=dict(key=key,turn_deg=float(np.rad2deg(y[-1])))
        for sname,s in dict(const=np.ones(n),oracle=vb.oracle_speed(t,rt,rP)).items():
            d=np.array([metrics.score_trajectory(t,core.integrate(t,g*y,s),rt,rP)['drift_pct'] for g in G])
            i=int(np.argmin(d));row[sname]=dict(best_gain=float(G[i]),best=float(d[i]),at_1=float(d[G==1.][0]),at_1p1=float(d[G==1.1][0]),at_1p2=float(d[G==1.2][0]))
        rows.append(row)
        print(f"{key} turn {row['turn_deg']:+7.1f}  const: best g {row['const']['best_gain']:.2f} ({row['const']['best']:.2f}% vs {row['const']['at_1']:.2f}% @1.0)   oracle: best g {row['oracle']['best_gain']:.2f} ({row['oracle']['best']:.2f}% vs {row['oracle']['at_1']:.2f}% @1.0)",flush=True)
    Path(out).parent.mkdir(parents=True,exist_ok=True);Path(out).write_text(json.dumps(rows,indent=1))
    for s in ['const','oracle']:
        g=[r[s]['best_gain'] for r in rows];print(s,'best gain median %.2f  range %.2f-%.2f'%(np.median(g),min(g),max(g)))
if __name__=='__main__':main(*sys.argv[1:3])
