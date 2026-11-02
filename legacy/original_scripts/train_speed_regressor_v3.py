"""TRAIN-only grouped CV for relative-feature speed regressors; no VAL tuning."""
import json,pickle,shutil,sys
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter1d
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'repos/lookout-vo-challenge/starter'))
from public_dataset import load_manifest,load_reference,sha256
MODES=['wide','zoom','flir-IR','flir-VIS','sionyx','infrared']

def inputs(z):
    f=z['features'].astype(float);center=f-f.mean(0)
    relative=np.clip(center/np.maximum(f.std(0),1e-6),-10,10)
    mode=np.zeros((len(f),len(MODES)));mode[:,MODES.index(str(z['camera_mode']))]=1
    return np.c_[f,center,relative,mode]

def normalize_speed(p,kind):
    p=np.maximum(np.exp(p)-.05 if kind=='log' else p,0)
    p=gaussian_filter1d(p,2.5)
    return p/max(p.mean(),1e-6)

def model():
    return HistGradientBoostingRegressor(loss='squared_error',max_iter=220,max_leaf_nodes=15,max_depth=5,min_samples_leaf=40,l2_regularization=1,random_state=0,early_stopping=False)

def main():
    dest=ROOT/'models/flow_speed_v3';dest.mkdir(exist_ok=False)
    clips=[c for c in load_manifest(ROOT/'data/release/release_manifest.json')['clips'] if c['split']=='train'];assert len(clips)==68
    xx=[];yy=[];gg=[];checks={}
    for c in clips:
        path=ROOT/'data/features/flow_speed_v2'/f"{c['key']}.npz"
        with np.load(path) as z:
            assert str(z['split'])=='train';xx.append(inputs(z));t=z['times']
        rt,rp=load_reference(ROOT/'data/release',c);rp=gaussian_filter1d(rp,2/np.median(np.diff(rt)),axis=0)
        target=np.interp(t,rt,np.linalg.norm(np.gradient(rp,rt,axis=0),axis=1));target/=max(target.mean(),.01)
        yy.append(target);gg.extend([c['key']]*len(t));checks[c['key']]=sha256(path)
    X=np.concatenate(xx);y=np.concatenate(yy);g=np.array(gg);keys=np.unique(g)
    splits=list(GroupKFold(3).split(X,y,g));folds=[dict(train_keys=sorted(set(g[tr])),heldout_keys=sorted(set(g[te]))) for tr,te in splits]
    results={};predictions={}
    for kind in ['linear','log']:
        target=np.log(y+.05) if kind=='log' else y;oof=np.zeros(len(y))
        for i,(tr,te) in enumerate(splits):
            assert not set(g[tr])&set(g[te]);m=model();m.fit(X[tr],target[tr]);oof[te]=m.predict(X[te]);print(kind,'FOLD',i,flush=True)
        for k in keys:oof[g==k]=normalize_speed(oof[g==k],kind)
        predictions[kind]=oof
        for alpha in [0.,.25,.5,.75,1.]:
            p=(1-alpha)+alpha*oof
            results[f'{kind}:{alpha}']=float(np.mean([np.mean(np.abs(p[g==k]-y[g==k])) for k in keys]))
    choice=min(results,key=results.get);kind,alpha=choice.split(':');alpha=float(alpha)
    if results[choice]>results['linear:0.0']*.95:alpha=0.
    m=model();m.fit(X,np.log(y+.05) if kind=='log' else y)
    with (dest/'model.pkl').open('wb') as f:pickle.dump(m,f)
    np.savez_compressed(dest/'train_group_cv.npz',target=y,prediction=predictions[kind],groups=g)
    record=dict(training_split='train',train_clip_count=68,feature_cache='flow_speed_v2',train_cache_sha256=checks,folds=folds,cv_clip_macro_relative_speed_mae_by_variant=results,selected_target=kind,selected_blend=alpha,selection_rule='TRAIN 3-fold clip-group CV, require >=5% macro speed MAE gain versus constant speed.',parameters=m.get_params(),feature_count=X.shape[1])
    (dest/'training_record.json').write_text(json.dumps(record,indent=2));shutil.copy2(__file__,dest)
    print(json.dumps(record['cv_clip_macro_relative_speed_mae_by_variant']),flush=True);print('SELECTED',kind,alpha,flush=True)
if __name__=='__main__':main()
