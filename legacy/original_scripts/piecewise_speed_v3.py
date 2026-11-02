"""Reference-free piecewise-constant visual speed prior; preserves reverse motion."""
import numpy as np

def piecewise_speed(times,speed,penalty=.03,min_seconds=10.,stop_threshold=0.):
    t=np.asarray(times);x=np.asarray(speed);n=len(x);min_n=max(2,int(round(min_seconds/np.median(np.diff(t)))))
    cs=np.r_[0.,np.cumsum(x)];cs2=np.r_[0.,np.cumsum(x*x)]
    lam=penalty*n*max(np.var(x),1e-8);cost=np.full(n+1,np.inf);cost[0]=-lam;previous=np.full(n+1,-1,dtype=int)
    for j in range(min_n,n+1):
        starts=np.r_[0,np.arange(min_n,j-min_n+1)];length=j-starts
        values=cost[starts]+cs2[j]-cs2[starts]-(cs[j]-cs[starts])**2/length+lam
        idx=int(np.argmin(values));cost[j]=values[idx];previous[j]=starts[idx]
    if previous[n]<0:return x.copy()
    out=np.empty(n);j=n
    while j>0:
        i=previous[j];out[i:j]=np.mean(x[i:j]);j=i
    if stop_threshold>0:out[np.abs(out)<stop_threshold]=0
    return out
