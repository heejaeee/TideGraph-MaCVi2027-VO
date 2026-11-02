"""Sample sparse probes and enforce a strictly earlier comparison frame."""
import numpy as np

def probe_pairs(times, fps=.5, gap=.125, max_frames=0):
    ids=np.unique(np.searchsorted(times,np.arange(times[0]+.25,times[-1],1/fps)))
    if max_frames:ids=ids[:max_frames]
    assert len(ids)>0 and np.all(ids>0)
    previous=np.minimum(ids-1,np.maximum(0,np.searchsorted(times,times[ids]-gap)))
    assert np.all(previous<ids) and np.all(times[ids]-times[previous]>0)
    return ids,previous
