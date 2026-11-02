# data/: inputs for the trajectory assembly

* `metadata/`: copied from the released test package (CC BY-NC 4.0). It contains the manifest,
  per-clip timestamps, and `clip_index.json`, which lists the 2 Hz frame indices, times and
  camera mode used for every clip.
* `omega/windows/<clip>/<start>.npz`: VGGT-Omega camera-to-world matrices for each 16-frame window
  (stride 8) of the 8 wide and 11 flir-VIS test clips. `run.py --mode video` regenerates them.
* `inputs/<clip>.npz`: `times`, `initial_yaw`, `speed` (wide: before TV; FLIR: final), and `legacy_yaw` (IR).
* `final_signals/<clip>.npz`: `times`, `yaw`, `speed` of entry 19226. Only `--mode replay` uses this.

These are the team's estimates. None of them is a reference trajectory.

Only `metadata/` is tracked in Git. Restore the other three directories with `python scripts/fetch_artifacts.py` from the pinned public Hugging Face snapshot. Every restored file is hash-verified. These caches enable review and cached reproduction, not a claim of full raw-video regeneration.
