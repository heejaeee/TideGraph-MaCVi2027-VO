# TideGraph: Maritime Visual Odometry @ MaCVi WACV 2027

This is the supplied reproduction release for the **TideGraph method**, team **UW-IPL** (submission ID 19406, provided by the team) in the
[Maritime Visual Odometry challenge](https://macvi.org/workshop/macvi27/challenges/maritime-visual-odometry).
Its layout follows the [MaCVi 2027 reproduction template](https://github.com/CeviKle/EURS-MaCVi2027-repro-template),
adapted to a trajectory task: the output is one `submission.json` bundle, not per-image files.

| | |
|---|---|
| Final entry | macvi.org submission ID `19406`. Mean drift **10.83 %**, all 20 test clips scored |
| Uploaded file | [`submission/submission.json`](submission/submission.json), SHA-256 `41133948ef57731a18df59c66d919179be3a66730725109ac6621c2b8953bc35` |
| Frozen settings | [`configs/final_entry.json`](configs/final_entry.json) |
| Method | [`METHOD.md`](METHOD.md) |

## Hand-in status

This is a review draft. Cached replay and rebuild were verified again on 2026-11-01: both exactly match the supplied 10.83% artifact. A new **raw-video pipeline is implemented** in [`raw_run.py`](raw_run.py), including legacy speed, infrared heading and the initial gauge. A full-path FLIR pilot passed; the 20-clip raw verification is running. This remains distinct from the older hybrid `video` mode. See `HANDIN_STATUS.txt` and `evidence/handin_cpu_verification.json`. All seven author contact and OpenReview records are complete. Entry 19406 and uploader 2038 were verified on the public leaderboard.

## Repository and artifact access

- Code: https://github.com/heejaeee/TideGraph-MaCVi2027-VO (public)
- Team artifacts: https://huggingface.co/hjkwon1996/TideGraph-MaCVi2027-VO (public)
- Entry: **19406**, supplied by the team; reported score **10.83%**.
- A verified checkpoint companion ZIP is hosted on Hugging Face with component-specific license copies and hashes. It contains Omega, Pi3X, DPVO and the speed regressor. Original videos and reference trajectories are not mirrored.
- Code and team artifacts are public. The checkpoint mirror retains all upstream noncommercial research terms; the upstream Omega download separately uses manual access approval.

`artifacts.lock.json` pins the Hugging Face commit and SHA-256 of both downloads and every extracted cache file. Cached estimates and the TRAIN-fitted model are restored to their original paths before running the existing code.

## Raw videos to submission JSON

See [`RAW_REPRODUCTION.md`](RAW_REPRODUCTION.md) for the CUDA environment, pinned weight setup, one-command inference and independent comparison. A verified four-model companion ZIP can be supplied to setup with `--checkpoint-zip`; other required models are fetched from their pinned upstream sources. No estimated-signal artifact or TRAIN/VAL data is needed for raw inference.

## Quick start (CPU cached verification; no GPU required)

```bash
python3.10 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt -r requirements-download.txt
python scripts/fetch_artifacts.py      # pinned revision + SHA-256 verification
python scripts/fetch_artifacts.py --verify-only
python repro_check.py . --run --input /path/to/lookout-vo-test-v0.1            # mode replay
python repro_check.py . --run --input /path/to/lookout-vo-test-v0.1 --mode rebuild
```

`--input` is the extracted held-out package `lookout-vo-test-v0.1`, the folder containing
`release_manifest.json`, `timing/` and `videos/`.

## What each mode reproduces

| Mode | Starts from | Result vs the uploaded file |
|---|---|---|
| `replay` | 19226 heading/speed signals (`data/final_signals`) + final wide settings | **byte-identical** (SHA-256 above) |
| `rebuild` | Omega window rotations (`data/omega`) + retained signals (`data/inputs`) | max 4.9e-11 m |
| `raw` | raw videos plus frozen weights; regenerates every motion input | representative full-path FLIR pilot verified; full 20-clip verification pending |
| `video` | raw test videos → VGGT-Omega on GPU → `rebuild` | Omega rotations within 0.06° median / 0.32° worst of `data/omega`; drift vs the uploaded trajectories 0.080 % mean, 0.137 % max (RTX 4090, 16 min; `evidence/video_rerun_rtx4090.json`) |

The legacy modes can be called through `run.py`; the new raw mode is documented separately:

```python
from run import main
main(model_dir='weights', input_dir='/path/to/lookout-vo-test-v0.1', output_dir='out', device='cuda', mode='video')
```
```bash
python run.py --input /path/to/lookout-vo-test-v0.1 --output out --mode replay
```

Each call writes `out/submission.json` and `out/report.json`. The report holds the SHA-256, the
per-clip position difference to the upload, gap-aware coverage, solver diagnostics and, for
`video`, the rotation difference to `data/omega`.

### GPU mode (`video`)

```bash
pip install -r requirements.txt -r requirements-gpu.txt   # plus a CUDA build of torch/torchvision
scripts/setup_vggt_omega.sh                                # vggt-omega @ 48b23c8 -> third_party/
HF_TOKEN=... python repro_check.py . --download-weights     # gated FAIR-NC checkpoint, sha256-checked
python run.py --input /path/to/lookout-vo-test-v0.1 --output out_video --mode video --device cuda
```

Run `scripts/fetch_artifacts.py` on the host before building a Docker image so the restored files are copied into the image. Do not bake authentication tokens into images.

Docker files are in `docker/`: `Dockerfile` for CPU replay/rebuild, `Dockerfile.gpu` for video.

## Reproducibility boundary

In the older `video` mode, only the **Omega heading** stages are recomputed. The new `raw` mode recomputes all the following signals. For replay/rebuild/video, these signals are downloaded from the pinned Hugging Face snapshot as intermediate
estimates in `data/inputs`:

* wide relative speed (water-surface flow + TRAIN-fitted regressor),
* the FLIR speed profiles,
* the infrared heading, and
* the initial heading gauge.

The code that produced them is in `legacy/` for review, but it depends on historical caches and
is not a verified single-command raw-video pipeline (`legacy/README.md`). For the metric only the
relative shape of speed and heading matters: one global similarity transform is fitted, so the
gauge and absolute scale have no effect.

## Repository layout

```
run.py                 main(model_dir, input_dir, output_dir, device, mode)
repro_check.py         layout / repro.yaml / weights link / signature / bundle checks, --run, --download-weights
repro.yaml             team, challenge, submission ID, contact
METHOD.md              one-page method, data use, leaderboard-selected settings
configs/               final_entry.json (frozen) + model/source inventories, dev split, A/B grouping evidence
src/tidegraph/         core.py (numerics, unchanged from 19226), pipeline.py, omega_windows.py, starter/ (vendored, MIT)
data/omega/            VGGT-Omega window rotations for the 19 visible-light test clips (estimates, not labels)
data/inputs/           retained legacy speed, initial heading, IR heading per clip
data/final_signals/    heading/speed of entry 19226 (replay)
data/metadata/         released test manifest, timestamps, frame indices (no references)
submission/            the exact uploaded submission.json + SHA256SUMS
weights/               original upstream Omega link; public companion in configs/checkpoint_bundle.json
evidence/              selection history, provenance and validation records
legacy/                original research scripts, speed-prior training code, 19226 package
third_party_licenses/  upstream licences and dataset terms
```

No test reference trajectories exist in this repository. Everything in `data/` is either a
released input or the team's own estimate.

## Data use and disclosure

* No training or self-supervised adaptation on VAL or TEST videos.
* External pretrained models are listed in `METHOD.md`.
* Several global and per-clip settings were chosen with aggregate test-leaderboard feedback,
  including a per-clip gain for clip_092 and clip_103. They are listed in `METHOD.md` and
  `evidence/selection_history.json`.

## License

The team's code is under the MIT License (`LICENSE`). Vendored starter files are MIT
(LOOKOUT contributors). The LOOKOUT dataset-derived metadata in `data/metadata` is CC BY-NC 4.0.
VGGT-Omega and the other upstream models keep their own terms; see `THIRD_PARTY_NOTICES.txt`.

## Team metadata

Registered team: UW-IPL. Team leader: HeeJae Kwon, University of Washington, contact supplied privately, macvi.org user ID 2038. HeeJae Kwon OpenReview: https://openreview.net/profile?id=~HeeJae_Kwon1. The ordered author record is in `configs/author_metadata.json`; author order is confirmed and all seven email/OpenReview records are complete.

## Published-release verification

A fresh GitHub clone, fresh Python 3.10 environment and fresh Hugging Face download cache reproduced all 20 clips with zero position difference and the exact entry SHA-256 in both replay and rebuild modes. See `evidence/fresh_clone_verification.json`. The raw-video reproduction boundary above still applies.

## Author metadata update

The author order is confirmed. Updated emails/OpenReview profiles are in `configs/author_metadata.json` and the factsheet. Team member macvi.org IDs were recovered from the final EURS factsheet at the team's request. The separate EURS Jetson submission is not used as the VO entry identifier; VO remains 19406.

Public access was tested with credentials disabled for Git and a new Hugging Face home with no token. Both downloaded-cache inference modes passed with socket connections blocked; see `evidence/anonymous_verification.json`. This verifies public delivery and offline cached assembly, not full raw-video regeneration.
