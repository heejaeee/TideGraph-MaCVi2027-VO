# legacy/: research code behind the retained signals (review only)

These files are what produced the intermediate signals in `data/inputs` and `data/omega`. They
are **not** a runnable pipeline in this repository. They expect the original workspace layout
(`data/`, `runs/`, `repos/`, `models/`, external weights) and historical caches.

| Folder | Contents |
|---|---|
| `original_scripts/` | 73 original modules. Entry points: `run_submission_inference.py` / `submission_fuse.py` (initial R3 / VGGT / KLT / water-flow signals and the wide speed), `v65_*`–`v77_*` (retained FLIR speed corrections), `v81_omega_pipeline.py`, `v82_omega_val.py`, `v83_omega_wide_submission.py` (wide Omega heading), `v87_omega_flir_submission.py`, `v150_flir512_submission.py`, `v152_group_probe.py` (FLIR Omega heading, A/B grouping). `source_index.json` has their hashes. |
| `upstream_patches/` | Patches applied to DPVO and to the lookout-vo-challenge starter. |
| `speed_prior_flow_v3/` | TRAIN-only training script, training record and the fitted sklearn model restored from Hugging Face by `scripts/fetch_artifacts.py` (`model.pkl`, sklearn 1.7.2; a pickle, so load it only if you trust this repository). |
| `dev_analysis/` | Development tools: validation baselines, gain scans, the Omega window tool, direction and far-field calibration probes. |
| `package_19226/` | The earlier entry 19226 package: `team.py`, `best19226.json`, the uploaded 19226 bundle and its description. |
