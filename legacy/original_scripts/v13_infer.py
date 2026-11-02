"""Bounded complementary rotation evidence around v2, no .65 heading gain.
Configurations fixed before evaluation. No supervised parameter fitting.
"""
import argparse,datetime
from scipy.optimize import least_squares
from domain_fusion_v2 import relative_window_yaw
from v13_common import *
from public_dataset import sha256

CONFIGS=[dict(name='v2_control',kind='base',candidate=True)]
CONFIGS += [dict(name=f'{src}_direct',kind='source',source=src,cap=None,candidate=False) for src in ['pi3x','vggt','median']]
CONFIGS += [dict(name=f'{src}_cap{cap}',kind='source',source=src,cap=cap,candidate=True) for src in ['pi3x','vggt','consensus'] for cap in [3,5]]
CONFIGS += [dict(name=f'image_bias_cap{cap}',kind='bias',cap=cap,candidate=True) for cap in [3,5]]
CONFIGS += [dict(name='cross_evidence_cap5',kind='cross',cap=5,candidate=True),dict(name='speed_residual_only',kind='speed',candidate=True)]

def sources(key,case,split='train'):
    t=case['t'];n=len(t);sigma=2/np.median(np.diff(t))
    psrc=ROOT/('runs/domain_v3/pi3x_rgb' if split=='train' else 'runs/domain_v13/pi3x_test')
    vsrc=ROOT/('runs/domain_probe_v2/vggt' if split=='train' else 'runs/submission_test_v1/vggt')
    py=gaussian_filter1d(relative_window_yaw(psrc,key,n),sigma);vy=gaussian_filter1d(relative_window_yaw(vsrc,key,n),sigma)
    # Only relative orientation matters; anchor all sources at the v2 start.
    r=case['headings']['r3'];return dict(pi3x=py-py[0]+r[0],vggt=vy-vy[0]+r[0])

def source_correction(r,t,src,kind):
    sigma=5/np.median(np.diff(t));a=gaussian_filter1d(src['pi3x']-r,sigma);b=gaussian_filter1d(src['vggt']-r,sigma);a-=a[0];b-=b[0]
    if kind=='pi3x':return a
    if kind=='vggt':return b
    if kind=='median':return np.median(np.array([np.zeros(len(r)),a,b]),axis=0)
    # Two independently trained sources must agree about the sign and magnitude
    # of an R3 correction. No fixed global turn attenuation.
    sign=(a*b>0);confidence=np.exp(-((a-b)/np.deg2rad(10))**2)
    return sign*confidence*.5*(a+b)

def image_bias(key,t,r,split='train'):
    z=np.load(ROOT/'runs/domain_v11/transfer_features'/split/f'{key}.npz');e=np.concatenate([z['static'],z['epipolar']]);i=e[:,0].astype(int);j=e[:,1].astype(int);dt=t[j]-t[i];w=np.sqrt(e[:,3]/np.maximum(dt,1));rhs=e[:,2]-(r[j]-r[i]);reg=10*np.sqrt(len(e)*.1)
    def loss(z):
        out=np.vstack([2*(np.sqrt(1+z)-1),1/np.sqrt(1+z),-.5/(1+z)**1.5]);out[:,-1]=[z[-1],1,0];return out
    sol=least_squares(lambda x:np.r_[(x[0]*dt-rhs)*w,reg*x[0]],np.zeros(1),bounds=(-np.deg2rad(.2),np.deg2rad(.2)),loss=loss,f_scale=np.deg2rad(.3))
    return sol.x[0]*(t-t[0])

def infer(key,c,case,r,s,src,bias,split='train'):
    t=case['t'];kind=c['kind'];speed=s.copy();delta=np.zeros(len(t))
    if kind=='source':delta=source_correction(r,t,src,c['source'])
    elif kind=='bias':delta=bias.copy()
    elif kind=='cross':
        model=source_correction(r,t,src,'consensus');agreement=(model*bias>0)*np.exp(-((model-bias)/np.deg2rad(10))**2);delta=agreement*.5*(model+bias)
    elif kind=='speed':
        from v12_speed_residual import infer as speed_infer
        from v12_relative_speed import extract
        extract(key,split);edges=np.load(ROOT/'runs/domain_v12/relative_speed'/split/f'{key}.npz')['raw'];speed,_=speed_infer(t,s,edges,10,1,True)
    if c.get('cap') is not None:delta=np.clip(delta,-np.deg2rad(c['cap']),np.deg2rad(c['cap']))
    return r+delta,speed,dict(max_heading_change_deg=float(np.rad2deg(abs(delta).max())),heading_shape_rms_deg=float(np.rad2deg(np.std(delta))),speed_max_change=float(np.max(abs(speed-s))))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--split',default='train',choices=['train','test']);ap.add_argument('--method');args=ap.parse_args()
    configs=[c for c in CONFIGS if args.method is None or c['name']==args.method];assert configs
    if args.split=='train':
        save_json(REP/'plan.json',dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),configs=CONFIGS,baseline='Official v2 14.10%; full R3 heading and original v2 speed. No .65 gain or V11 temporal bias inherited.',selection_rule='Primary eligible methods: min mean TRAIN score with >=5/8 improved, maximum single-clip regression <=.5pp, and average delta excluding the largest improvement <0. Exclude uncapped direct-source diagnostics from promotion. Run leave-one-clip-out selection and report failures; not a pristine holdout.',scope='Repeatedly seen TRAIN8, original learned speed prior already trained on TRAIN. No new learned speed or GT-dependent inference. Boat IDs are unavailable in the manifest; do not infer boat groups from clip numbers.',test_policy='Preserve v2 non-FLIR9 exactly, freeze selected config before TEST inference, no online upload.'))
        frozen=REP/'frozen_hashes.json'
        if not frozen.exists():save_json(frozen,{str(p.relative_to(ROOT)):sha256(p) for p in (ROOT/'submissions').glob('*/submission.json')})
        keys=TRAIN
    else:
        import v6_test_common as test
        keys=test.KEYS
    for key in keys:
        case,r,s=base_case(key,args.split);src=sources(key,case,args.split);bias=image_bias(key,case['t'],r,args.split);diags={}
        for c in configs:
            y,speed,diag=infer(key,c,case,r,s,src,bias,args.split);emit(key,c['name'],y,speed,case,args.split);diags[c['name']]=diag
        save_json(OUT/args.split/'diagnostics'/f'{key}.json',diags);print('V13_INFER',args.split,key,len(configs),flush=True)

if __name__=='__main__':main()
