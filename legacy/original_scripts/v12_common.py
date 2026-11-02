"""V12 reference-free prediction utilities."""
from v5_common import *
OUT=ROOT/'runs/domain_v12';REP=ROOT/'reports/domain_v12'
def baseline(key):
    case=get_case(key);z=np.load(ROOT/'runs/domain_v11/signals/trust_region10'/f'{key}.npz')
    assert np.array_equal(case['t'],z['times'])
    return case,z['yaw'].copy(),z['speed'].copy()
def emit(key,name,yaw,speed,case):
    save_json(OUT/'predictions'/name/f'{key}.json',{**case['pred'],'positions':integrate_velocity(case['t'],yaw,speed).tolist()})
    p=OUT/'signals'/name/f'{key}.npz';p.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(p,times=case['t'],yaw=yaw,speed=speed)
