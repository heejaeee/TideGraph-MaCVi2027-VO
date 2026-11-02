"""Reference-free v3 postprocessing. Pure functions, no label/file access."""
import numpy as np
from piecewise_speed_v3 import piecewise_speed

def integrate_velocity(times,yaw,speed):
    v=np.column_stack([np.sin(yaw),np.zeros(len(times)),np.cos(yaw)])*np.asarray(speed)[:,None]
    return np.vstack([np.zeros(3),np.cumsum((v[1:]+v[:-1])*.5*np.diff(times)[:,None],axis=0)])

def wide_speed(times,speed,guard_stops=True):
    s=np.asarray(speed,dtype=float)
    # A substantial near-stop interval makes amplitude compression unreliable.
    # Preserve the complete v2 profile rather than changing stops/reversals.
    if guard_stops and np.mean(np.abs(s)<.2)>=.1:return s.copy(),False
    a=np.sign(s)*np.maximum(np.abs(s)-.1,0.)
    a=a/(1+.5*np.abs(a))
    return piecewise_speed(times,a,.03),True
