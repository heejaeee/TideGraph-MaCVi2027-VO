"""Raw video -> per-window camera rotations with frozen VGGT-Omega (GPU stage).

Same procedure as the windows in data/omega: 2 Hz frames at the released
timestamps, active_rect crop, resize to width 960 (wide INTER_AREA, FLIR
INTER_LINEAR), the official 'balanced' preprocessing at 416 (wide) / 512
(flir-VIS) with one checkpoint, 16-frame windows every 8 frames, fp32.
Writes <out>/windows/<key>/<start:06d>.npz (start, c2w (N,4,4)) and <out>/clips/<key>.json.
flir-IR clips are skipped (they use the retained legacy heading).
"""
import gc
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from .starter.public_dataset import load_manifest, load_timestamps

RES = {'wide': 416, 'flir-VIS': 512}
INTERP = {'wide': cv2.INTER_AREA, 'flir-VIS': cv2.INTER_LINEAR}
WEIGHTS_NAME = 'vggt_omega_1b_416_reproduce.pt'


def frame_ids(times):
    return np.unique(np.r_[np.searchsorted(times, np.arange(times[0], times[-1], .5)), len(times) - 1])


def frames(root, clip, ids):
    x, y, w, h = clip['calibration']['active_rect']
    height = round(h * 960 / w)
    wanted = set(int(i) for i in ids)
    cap = cv2.VideoCapture(str(root / clip['video']))
    assert cap.isOpened(), clip['video']
    out = []
    for i in range(int(ids[-1]) + 1):
        if not cap.grab():
            raise RuntimeError((clip['key'], i))
        if i not in wanted:
            continue
        ok, bgr = cap.retrieve()
        assert ok
        out.append(cv2.resize(bgr[y:y + h, x:x + w], (960, height),
                              interpolation=INTERP[clip['camera_mode']])[:, :, ::-1].copy())
    cap.release()
    assert len(out) == len(ids)
    return out


def omega_backend(weights, source_dir, device):
    sys.path.insert(0, str(source_dir))
    import torch
    from PIL import Image
    from torchvision import transforms as TF
    from vggt_omega.models import VGGTOmega
    from vggt_omega.utils import load_fn
    from vggt_omega.utils.pose_enc import encoding_to_camera
    torch.manual_seed(0)
    model = VGGTOmega()
    sd = torch.load(weights, map_location='cpu', weights_only=True)
    model.load_state_dict(sd, strict=True)
    del sd
    gc.collect()
    model = model.eval().to(device)
    to_tensor = TF.ToTensor()

    def preprocess(rgb, res):
        # identical to load_fn.load_and_preprocess_images for one image, minus file IO
        im = load_fn._crop_to_supported_aspect_ratio(Image.fromarray(rgb))
        w, h = im.size
        th, tw = load_fn._balanced_target_shape(h / max(w, 1), res, 16)
        return to_tensor(im.resize((tw, th), Image.Resampling.BICUBIC))

    def run(batch):
        with torch.inference_mode():
            out = model(batch.to(device))
            ex, _ = encoding_to_camera(out['pose_enc'].float(), batch.shape[-2:])
        ex = ex[0].cpu().numpy()
        c2w = np.tile(np.eye(4), (len(ex), 1, 1))
        c2w[:, :3, :3] = ex[:, :, :3].transpose(0, 2, 1)
        c2w[:, :3, 3] = -np.einsum('nij,nj->ni', c2w[:, :3, :3], ex[:, :, 3])
        return c2w

    return torch, preprocess, run


def generate(input_dir, out_dir, weights, source_dir, device='cuda', keys=None, window=16, log=print):
    """Write Omega windows for every wide / flir-VIS clip of the package at input_dir."""
    input_dir, out_dir = Path(input_dir), Path(out_dir)
    clips = {c['key']: c for c in load_manifest(input_dir / 'release_manifest.json')['clips']}
    torch, preprocess, run = omega_backend(weights, source_dir, device)
    for key in keys or sorted(clips):
        clip = clips[key]
        mode = clip['camera_mode']
        if mode not in RES:
            log(f'omega SKIP {key} {mode}')
            continue
        dest = out_dir / 'windows' / key
        info = out_dir / 'clips' / f'{key}.json'
        if info.exists():
            log(f'omega CACHED {key}')
            continue
        dest.mkdir(parents=True, exist_ok=True)
        beg = time.perf_counter()
        ts = load_timestamps(input_dir, clip)
        ids = frame_ids(ts)
        n = len(ids)
        images = torch.stack([preprocess(f, RES[mode]) for f in frames(input_dir, clip, ids)])
        if device.startswith('cuda'):
            torch.cuda.reset_peak_memory_stats()
        for start in range(0, n, window // 2):
            end = min(n, start + window)
            np.savez_compressed(dest / f'{start:06d}.npz', start=np.array(start), c2w=run(images[start:end]))
            if end == n:
                break
        rec = dict(key=key, mode=mode, frames=n, frame_indices=ids.tolist(), times=ts[ids].tolist(),
                   resolution=RES[mode], input_hw=list(images.shape[-2:]), window=window,
                   seconds=time.perf_counter() - beg,
                   peak_cuda_gib=torch.cuda.max_memory_allocated() / 2**30 if device.startswith('cuda') else None)
        info.parent.mkdir(parents=True, exist_ok=True)
        info.write_text(json.dumps(rec, indent=1))
        log(f'omega DONE {key} {mode} {n} frames {rec["seconds"]:.1f} s')
        del images
        gc.collect()

