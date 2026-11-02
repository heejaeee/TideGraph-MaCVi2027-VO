"""R3 long-preset adapter. Reads video/timestamps/calibration, never reference labels."""
import argparse
import gc
import json
import platform
import random
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from omegaconf import OmegaConf
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'repos/lookout-vo-challenge/starter'))
from public_dataset import load_manifest, load_timestamps, select_clips, validate_prediction, write_json
from R3.models.r3 import R3
from R3.utils.input_io import load_images, _resize_pil_image
from R3.utils.pose_enc import pose_encoding_to_extri_intri
from depth_anything_3.utils.geometry import affine_inverse


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--key', required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--fps', type=float, default=2)
    ap.add_argument('--size', type=int, default=504)
    ap.add_argument('--bank', type=int, default=32)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--max-frames', type=int, default=0, help='Smoke diagnostic only; zero means full clip')
    ap.add_argument('--dataset',required=True)
    args = ap.parse_args()
    run = args.output.resolve()
    pred_path = run / 'predictions' / f'{args.key}.json'
    if pred_path.exists():
        raise FileExistsError(pred_path)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    cv2.setNumThreads(2)
    data = Path(args.dataset).resolve().parent
    clip = select_clips(load_manifest(Path(args.dataset)), clips=args.key)[0]
    times = load_timestamps(data, clip)
    indices = np.unique(np.r_[np.searchsorted(times, np.arange(times[0], times[-1], 1 / args.fps)), len(times)-1])
    if args.max_frames:
        indices = indices[:args.max_frames]
    start = time.perf_counter()
    # Preserve released decoded-frame indices. Sequential decode avoids inaccurate video seeking.
    with tempfile.TemporaryDirectory(prefix='r3_frames_', dir=run) as tmp:
        cap = cv2.VideoCapture(str(data / clip['video']))
        if not cap.isOpened():
            raise RuntimeError('Video open failed')
        paths = []
        wanted = set(indices.tolist())
        x, y, w, h = clip['calibration']['active_rect']
        for i in range(int(indices[-1]) + 1):
            ok = cap.grab()
            if not ok:
                raise RuntimeError(f'Decode failed at source frame {i}')
            if i in wanted:
                ok, frame = cap.retrieve()
                if not ok:
                    raise RuntimeError(f'Retrieve failed at source frame {i}')
                frame = frame[y:y+h, x:x+w]
                path = Path(tmp) / f'{i:06d}.png'
                # Same PIL resize as upstream, before lossless PNG storage. This
                # avoids encoding/decoding full-resolution temporary frames.
                # For size=224 upstream uses a different resize; leave it intact.
                if args.size == 224:
                    if not cv2.imwrite(str(path), frame):
                        raise RuntimeError(f'Image write failed: {path}')
                else:
                    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    _resize_pil_image(image, args.size).save(path, compress_level=1)
                paths.append(str(path))
        cap.release()
        views = load_images(paths, args.size, patch_size=14)
        images = torch.stack([v['img'] for v in views], dim=1)
        del views
    print(f'INPUT {args.key}: {len(indices)} frames, shape {tuple(images.shape)}, max gap {np.diff(times[indices]).max():.4f}s', flush=True)
    settings = dict(online_mode=True, online_kv_cache_mode='dynamic', online_recent_frames=0,
        online_verbose=True, bank_initial_frames=1, keyframe_mode='novelty',
        keyframe_novelty_threshold=.985, keyframe_max_interval=30, keyframe_max_keyframes=args.bank,
        online_fallback_enabled=True, drought_length=3, drought_threshold=0,
        drought_threshold_pct=45, num_bridge_frames=10, evict_low_conf_threshold=0,
        fallback_ref_mode='bridge', min_segment_frames=16, max_segment_frames=300,
        fallback_replay_attention='full', disable_segment_pgo=True,
        metric_scale_enabled=True, metric_bootstrap_frames=5,
        metric_model_name=str(ROOT / 'repos/R3/ckpt/da3_metric'))
    model = R3(da3_cfg=OmegaConf.load(ROOT / 'repos/R3/configs/r3-large.yaml'),
               teacher_embed_dim=2048, student_embed_dim=2048, freeze='none', **settings)
    state = load_file(str(ROOT / 'repos/R3/ckpt/r3_long.safetensors'), device='cpu')
    expected = model.state_dict()
    remapped = {}
    training_metadata = {}
    for key, value in state.items():
        if key in {'train_total_images', 'train_total_samples', 'epoch_fraction'}:
            training_metadata[key] = value.item()
            continue
        if key.startswith('net.'):
            key = key[4:]
        if key.startswith('module.'):
            key = key[7:]
        if key.startswith('model.'):
            key = 'da3.' + key[6:]
        if key not in expected and 'da3.' + key in expected:
            key = 'da3.' + key
        remapped[key] = value
    result = model.load_state_dict(remapped, strict=False)
    bad_missing = [k for k in result.missing_keys if not k.startswith('da3_metric.')]
    if bad_missing or result.unexpected_keys:
        raise RuntimeError(f'Checkpoint mismatch: {bad_missing}, {result.unexpected_keys}')
    print(f'CHECKPOINT matched {len(remapped)} R3 tensors; metric model loaded separately', flush=True)
    del state, expected, remapped
    gc.collect()
    model.eval().to('cuda')
    images = images.to('cuda')
    torch.cuda.reset_peak_memory_stats()
    stats_path = run / 'diagnostics' / f'{args.key}.runtime.jsonl'
    stats_path.parent.mkdir(exist_ok=True)
    infer_start = time.perf_counter()
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        model.clear_online_state()
        predictions = model(images, use_ray_pose=False, pose_max_recent=0,
            runtime_stats_every=50, runtime_stats_path=str(stats_path),
            rel_pose_reconstruction_method='greedy',
            rel_pose_reconstruction_kwargs=dict(topn_conf=999, score_mode='auto'))
    torch.cuda.synchronize()
    infer_seconds = time.perf_counter() - infer_start
    extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions['pose_enc'].float(), images.shape[-2:])
    c2w = affine_inverse(extrinsic).detach().cpu().numpy()[0]
    output_ids = np.asarray(predictions.get('output_frame_ids', range(len(c2w))), dtype=int)
    if len(output_ids) != len(c2w) or np.any(np.diff(output_ids) <= 0):
        raise RuntimeError('Invalid R3 output frame mapping')
    source_ids = indices[output_ids]
    prediction = dict(key=args.key, frame_indices=source_ids.tolist(), times=times[source_ids].tolist(),
                      positions=c2w[:, :3, 3].tolist())
    validate_prediction(prediction, args.key, times)
    write_json(pred_path, prediction)
    np.savez_compressed(run / 'diagnostics' / f'{args.key}.poses.npz',
        camera_to_world=c2w, intrinsics=intrinsic.detach().cpu().numpy()[0],
        input_frame_indices=indices, output_frame_ids=output_ids)
    scores = predictions.get('frame_post_scores', {})
    write_json(run / 'diagnostics' / f'{args.key}.json', dict(
        arguments=vars(args) | {'output': str(run)}, settings=settings,
        checkpoint_training_metadata=training_metadata,
        input_frames=len(indices), output_frames=len(output_ids),
        omitted_output_ids=sorted(set(range(len(indices))) - set(output_ids.tolist())),
        pose_confidence={str(k): float(v) if np.isfinite(v) else None for k, v in scores.items()},
        inference_seconds=infer_seconds, elapsed_seconds=time.perf_counter()-start,
        peak_cuda_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
        peak_cuda_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
        environment=dict(python=platform.python_version(), torch=torch.__version__, numpy=np.__version__,
                         cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0)),
        note='No training. Released active rectangle is cropped; intrinsics are estimated by R3. No GT access or trajectory smoothing.'))
    print(f'CLIP_COMPLETE {args.key}: {len(output_ids)}/{len(indices)} poses, inference {infer_seconds:.1f}s, peak {torch.cuda.max_memory_allocated()/2**30:.2f} GiB', flush=True)


if __name__ == '__main__':
    main()
