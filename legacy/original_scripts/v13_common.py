"""V13 starts from official-best v2; never reads trajectory references."""
from v5_common import *
OUT=ROOT/'runs/domain_v13';REP=ROOT/'reports/domain_v13'

def base_case(key,split='train'):
    if split=='train':case=get_case(key)
    else:
        from v6_test_common import get_case as test_case
        case=test_case(key);case['headings']={'r3':case['r3_yaw']}
    t=case['t'];r=case['headings']['r3'].copy();s=case['speed'].copy() if case['speed_correlation']>=.7 else np.ones(len(t))
    return case,r,s

def emit(key,name,yaw,speed,case,split='train'):
    pred={**case['pred'],'positions':integrate_velocity(case['t'],yaw,speed).tolist()}
    save_json(OUT/split/'predictions'/name/f'{key}.json',pred)
    p=OUT/split/'signals'/name/f'{key}.npz';p.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(p,times=case['t'],yaw=yaw,speed=speed)
    return pred
