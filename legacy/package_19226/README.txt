TideGraph team package — accepted entry19226, reported drift11.29%
===================================================================

This is the best19226 version, not the newer V161 soft-yaw experiment.
Expected official submission SHA-256:
b629c7a73ccd8ee597abfe95d876e9f9977f88eedf178d5dd982af5d6dbe2dde

Quick start (Linux, Python3.10 tested)
----------------------------------
Unzip, enter the TideGraph_team_best19226 folder:
  python3 -m venv .venv
  . .venv/bin/activate
  python -m pip install -r requirements-cpu.txt
  python team.py verify
  python team.py replay --output outputs/replay
  python team.py rebuild --output outputs/rebuild

replay = final cached yaw/speed -> trapezoidal integration -> submission.
rebuild = included Omega per-window rotations -> median/A1 graph -> sigma2
smoothing -> fixed group gains; wide legacy speed ->TV10; then integration.
These commands use no GPU, no raw videos, no GT and no external weights.
outputs/*/validation.json reports numeric error and byte/hash equality.
Frozen configs are in configs/best19226.json. Avoid modifying the only copy;
use a new config and new output folder for experiments. Existing outputs are
never overwritten. Runtime of these commands is NOT full-pipeline FPS.

What is included
----------------
team.py / core.py: portable final postprocessing implementation.
configs/: best settings, explicit A6/B5 membership, model/source versions.
snapshot/omega/windows/: estimated camera-to-world rotations (not GT).
snapshot/inputs/: exact retained legacy speed, initial yaw gauge and IR yaw.
snapshot/final_signals/: final yaw/speed for an independent integration replay.
snapshot/metadata/: released TEST calibration/indices/timestamps; no labels.
artifacts/submission_19226.json: exact accepted submission, for parity checks.
trained_models/: TRAIN68 fitted speed prior, training code and provenance.
workspace_source/: original processing-code dependency archive and repo patches.
evidence/: package validation and source lineage.
licenses/: copied upstream and dataset terms. No new license is assigned here.

Actual final branches
---------------------
Wide8: Omega416 input, median adjacent relative rotations, Gaussian sigma2s;
retained V87 routed speed -> uniform2Hz normalization -> TV lambda10.
FLIR VIS11: same Omega416 checkpoint with input512, W16/stride8, all-pairs
within-window robust yaw graph (Huber3degrees), Gaussian sigma2s;
A1.20/B1.10 gains. Exact speed retained:9 constant, clip095 and clip111 have
tiny nonconstant corrections. Do not replace those2 profiles with ones.
IR1 clip100: retained legacy R3/KLT yaw and constant speed1.
No new soft threshold, deadband, budget normalization or later candidate.

Reproducibility boundary — important
------------------------------------
The portable rebuild starts from estimated Omega windows AND retained legacy
signals. It DOES NOT recreate all speed/calibration/IR/heading-gauge signals
from raw videos. The original workspace code is supplied for review and porting;
it depends on historical cache paths and is NOT a verified one-command raw-video
pipeline. External model weights and video files are not bundled.

To extend toward raw inference, use the released data with its manifest,
configs/omega_weights.json and configs/upstream_revisions.json. Existing raw
adapters: v82_omega_val.py / v83_omega_wide_submission.py for wide frames/Omega;
v6_test_prepare.py / v87_omega_flir_submission.py / v150_flir512_submission.py
for FLIR; run_submission_inference.py and submission_fuse.py for initial raw
R3/VGGT/KLT/water/features; v65_* through v76_* for retained FLIR corrections.
Wide cache extraction used INTER_AREA, historical FLIR cache INTER_LINEAR.
Released active_rect and timestamps must be preserved. Both resolutions use
the same416 reproduction weights, not a separate512 checkpoint.

External model dependencies are listed in configs/model_inventory.json.
The recipient needs their own model access; no credentials are included.
The included sklearn pickle is only needed for original speed inference and
is not deserialized by the portable commands. It was trained on TRAIN68.
Raw GPU inference used torch2.7.1+cu128 / RTX5080; end-to-end FPS is not measured.

Scientific disclosure and sharing
---------------------------------
11.29% is the user-reported official aggregate score of entry19226. No hidden
TEST labels exist in this package; the archive is a prediction, not ground truth.
Settings were selected with reused development and aggregate leaderboard
feedback. A/B partition used image-derived land-visibility ranking before its
gain probes. No VAL/TEST training is performed by these final inference stages.
Keep this as an internal team package unless the owner requests publication.
Portal final selection has intentionally not been updated; this package does
not authenticate, upload to the challenge or change public visibility.

Verified handoff
----------------
Fresh Python3.10 virtual environment, isolated extracted directory, original
project reads blocked: replay and rebuild both reproduced all20 trajectories
with position error0 and the exact b629c7a7... submission hash.
Python needs venv/ensurepip support for the quick-start command. If your Linux
system Python lacks it, use a Python installation with venv or a conda3.10 env.
See evidence/portable_verification.json and isolated_*_validation.json.
