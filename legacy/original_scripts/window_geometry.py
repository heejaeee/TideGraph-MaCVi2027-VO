"""GT-free overlap alignment using camera rotations and depth scale agreement."""
import numpy as np


def mean_rotation(rotations):
    u, _, vt = np.linalg.svd(np.sum(rotations, axis=0))
    return u @ np.diag([1., 1., np.linalg.det(u @ vt)]) @ vt


def align_window(local, world_overlap, overlap, depth_scale=None):
    """Map local c2w poses into accumulated world coordinates.
    Rotations fix the otherwise unobservable roll of collinear camera centers.
    Scale comes from shared-frame depths, or centered camera motion if requested.
    """
    rotation = mean_rotation(world_overlap[:, :3, :3] @ local[:overlap, :3, :3].transpose(0, 2, 1))
    x = local[:overlap, :3, 3] @ rotation.T
    y = world_overlap[:, :3, 3]
    xc, yc = x-x.mean(0), y-y.mean(0)
    pose_scale = float(np.sum(xc*yc) / max(np.sum(xc*xc), 1e-12))
    scale = float(depth_scale) if depth_scale is not None else pose_scale
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(f'Invalid overlap scale {scale}')
    offset = (y-scale*x).mean(0)
    aligned = local.copy()
    aligned[:, :3, :3] = rotation @ local[:, :3, :3]
    aligned[:, :3, 3] = scale * (local[:, :3, 3] @ rotation.T) + offset
    return aligned, scale, pose_scale


def stitch_windows(windows, scale_mode='depth'):
    world = None
    records = []
    previous = None
    previous_scale = 1.
    for item in windows:
        start, local = int(item['start']), item['c2w'].astype(np.float64)
        if world is None:
            world = local.copy()
            records.append(dict(start=start,scale=1.,pose_scale=1.))
        else:
            overlap = len(world)-start
            if overlap < 2 or overlap >= len(local):
                raise ValueError('Each new window must overlap and extend the trajectory')
            depth_scale = None
            if scale_mode=='depth':
                prev_start=int(previous['start'])
                old=previous['depth'][start-prev_start:start-prev_start+overlap].astype(float)
                new=item['depth'][:overlap].astype(float)
                oc=previous['conf'][start-prev_start:start-prev_start+overlap].astype(float)
                nc=item['conf'][:overlap].astype(float)
                valid=np.isfinite(old)&np.isfinite(new)&(old>1e-6)&(new>1e-6)
                valid &= (oc>=np.quantile(oc,.5))&(nc>=np.quantile(nc,.5))
                if valid.sum()<100:
                    raise ValueError('Insufficient valid overlap depths')
                depth_scale=float(np.median(old[valid]/new[valid]))*previous_scale
            aligned, scale, pose_scale=align_window(local,world[start:],overlap,depth_scale)
            # Keep already emitted poses. Start new motion from the final shared
            # camera center to avoid a seam offset from the overlap least squares.
            aligned[:, :3, 3] += world[-1,:3,3]-aligned[overlap-1,:3,3]
            world=np.concatenate([world,aligned[overlap:]],axis=0)
            previous_scale=scale
            records.append(dict(start=start,scale=scale,pose_scale=pose_scale))
        previous=item
    return world, records
