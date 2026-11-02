# Raw-video reproduction of submission #19406

`raw_run.py` builds every intermediate from the official TEST videos, released
timestamps/calibration, frozen upstream weights, and the saved TRAIN-trained
speed regressor. It does not use `data/inputs`, `data/final_signals`, `data/omega`,
the archived submission, or the Hugging Face estimated-signals ZIP during inference.
TRAIN/VAL videos and reference trajectories are not required for inference.

Integration status: a representative raw-video FLIR clip has passed all stages
(evidence/raw_video_clip093.json). The full 20-clip run is in progress. The existing replay/rebuild verification remains separate.
Do not interpret a successful cached replay as verification of this path.

## Setup

Use Python 3.10, a CUDA 12.8 toolkit, and an NVIDIA GPU. The original environment
was an RTX 5080 with 16 GB VRAM and PyTorch 2.7.1. A different GPU/software stack
can produce numerical differences; byte identity is not promised for fresh neural
inference. The locked upstream assets total approximately 19.5 GB before source,
environment, videos and generated intermediates.

```bash
python3.10 -m venv .venv-raw
source .venv-raw/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-raw.txt
python scripts/setup_raw_assets.py --output /path/to/raw-assets
CUDA_HOME=/usr/local/cuda-12.8 bash scripts/build_raw_extensions.sh /path/to/raw-assets
```

Setup uses the public, SHA-256-pinned checkpoint companion bundle by default, so
organizers do not need manual upstream Omega approval. The bundle includes the
full FAIR noncommercial research agreement and the other component-specific
terms. You can reuse a local download with
`--checkpoint-zip /path/to/TideGraph-MaCVi2027-VO_checkpoints_verified.zip`.
`--upstream-only` uses the original sources for everything instead; Omega then
requires accepting the upstream license/access request and local authentication.

`configs/raw_assets.lock.json` pins source commits, file sizes and SHA-256 hashes.
Models absent from the companion bundle are obtained from their pinned upstream
sources. An existing DPVO checkpoint can also be supplied with
`--dpvo-weights /path/to/dpvo.pth` when using `--upstream-only`.

The R3 scale-consistency patch and DPVO PyTorch compatibility patch in `configs/`
are applied to the pinned upstream sources. These preserve fixes present in the
original execution environment. No pretrained weights are changed.

## Run

```bash
python raw_run.py \
  --input /path/to/lookout-vo-test-v0.1 \
  --assets /path/to/raw-assets/assets.json \
  --output /path/to/new-raw-run
```

Alternatively, the challenge API can call
`run.main('/path/to/raw-assets', input_dir, output_dir, 'cuda', mode='raw')`.
The other modes retain their previous meaning: `replay` and `rebuild` use archived
estimates; `video` reruns only Omega and retains legacy heading/speed inputs.

After setup, inference is offline. Workers reject network connections and access
to packaged prediction caches. The TEST manifest must contain exactly the final
entry's 20 keys and no reference trajectories. Intermediate files are written
under the new run directory, never under the archived experiment folders.

The complete output is `<output>/submission.json` with checks in `raw_report.json`.
Completed stages have markers in `logs/`; rerunning the command resumes them.
Do not change weights, configuration or completed intermediates within a run.
Use a fresh output directory for changed code/settings. A failed stage leaves its
traceback in `logs/<stage>.log` and does not mark the stage complete.

For a diagnostic pilot, append `--keys clip_093`. It produces
`diagnostic_subset.json`, explicitly not a complete submission. The optional lanes
are `baseline-cpu`, `baseline-gpu`, `geometry`, `omega`, and `assemble`. CPU and GPU
baseline lanes can run concurrently, but use one GPU lane at a time. Geometry
requires completed baseline estimates. The default `all` executes every stage.

## Computation retained from the final entry

1. Decode the released frames and sample their actual timestamps at approximately 2 Hz.
2. Compute R3-long, VGGT-1B, KLT, signed water-plane flow and visual speed features.
   Apply the frozen TRAIN68 speed regressor, horizon/current handling and confidence gate.
3. For FLIR-visible clips, generate image calibration, semantic masks, CoTracker
   tracks, Pi3X windows, three random DPVO passes and three static DPV-SLAM passes.
   Recompute the five spatial-fold course checks and conservative speed-support rule.
   The original final output has nine constant-speed FLIR clips and two very small
   speed corrections; these are recomputed, not hard-coded or discarded.
4. Run VGGT-Omega at 416 for wide and 512 for FLIR-visible. Apply the final median
   rotation chain/all-pairs graph, Gaussian smoothing and frozen gains. Apply wide
   TV speed filtering with lambda 32. Retain the freshly computed R3/KLT infrared
   heading and constant infrared speed.
5. Integrate heading and speed, validate timing/coverage, and write the 20 trajectories.

The final per-clip settings remain those in `configs/final_entry.json`, including
leaderboard-selected gains and overrides. This code does not retune or retrain them.

## Verification

Run comparison **after** inference, in a separate process:

```bash
python scripts/verify_raw_output.py \
  --generated /path/to/new-raw-run/submission.json \
  --output /path/to/new-raw-run/reproduction_comparison.json
```

It compares timestamps, frame indices, coordinates, and globally aligned shape
with the submitted estimates. It does not access hidden TEST ground truth and
does not calculate a new official challenge score. For a pilot file, add
`--allow-subset`. `evidence/raw_port_geometry_parity.json` is specifically a
numerical port regression using archived intermediate fixtures; it is not an
independent raw-video result.
