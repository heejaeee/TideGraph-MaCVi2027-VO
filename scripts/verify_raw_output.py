"""Compare completed raw inference with the archived entry, outside inference.

This is a reproduction/self-consistency check, not a score on hidden TEST labels.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--generated', required=True, type=Path)
    ap.add_argument('--reference', type=Path, default=Path(__file__).resolve().parents[1]/'submission/submission.json')
    ap.add_argument('--output', required=True, type=Path)
    ap.add_argument('--allow-subset', action='store_true')
    a = ap.parse_args()
    actual = json.loads(a.generated.read_text()); expected = json.loads(a.reference.read_text())
    if not a.allow_subset and set(actual) != set(expected):
        raise ValueError('Incomplete raw bundle')
    clips = {}
    for key, pred in actual.items():
        ref = expected[key]
        if pred['times'] != ref['times'] or pred['frame_indices'] != ref['frame_indices']:
            raise ValueError('Timestamp/index mismatch: '+key)
        p, q = np.asarray(pred['positions']), np.asarray(ref['positions'])
        if p.shape != q.shape or not np.isfinite(p).all(): raise ValueError(key)
        pc, qc = p-p.mean(0), q-q.mean(0)
        u, sv, vt = np.linalg.svd(pc.T@qc)
        d = np.ones(3); d[-1] = np.linalg.det(u@vt)
        rot = u@np.diag(d)@vt
        scale = (sv*d).sum()/max((pc*pc).sum(), 1e-12)
        aligned = scale*pc@rot+q.mean(0)
        path_length = np.linalg.norm(np.diff(q, axis=0), axis=1).sum()
        clips[key] = dict(samples=len(p), max_coordinate_difference=float(abs(p-q).max()),
            aligned_rmse_over_path_length=float(np.sqrt(np.mean(np.sum((aligned-q)**2, axis=1)))/max(path_length, 1e-12)),
            alignment_scale=float(scale))
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    report = dict(scope='Self-consistency with submitted estimated trajectories; no TEST ground truth or official rescoring',
        clips=len(clips), complete=set(actual)==set(expected),
        byte_identical=sha(a.generated)==sha(a.reference), generated_sha256=sha(a.generated),
        submitted_sha256=sha(a.reference), per_clip=clips)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
