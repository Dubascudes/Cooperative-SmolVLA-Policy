# SO-101 block manipulation starter

**The default backend is now robosuite.** See [ROBOSUITE.md](ROBOSUITE.md) for
the custom SO-101 integration, two-arm scene, action contracts, and research
limitations. Use `--backend mujoco` to run the original standalone scene.

The two-arm viewer now uses a smaller pot. See
[COOPERATIVE_TRAINING.md](COOPERATIVE_TRAINING.md) for scene controls, the 12-action
SmolVLA GPU training diagnostic, and the demonstration-to-policy roadmap.

A single SO-101 arm pushes a red 3 cm cube into a green target in MuJoCo.
The arm uses the actual MuJoCo Menagerie SO-101 model, including its articulated
gripper and collision geometry. The block moves through contact physics: no
teleportation, attachment, or scripted edits to its pose during an episode.

Two controllers share the same environment:

- `scripted`: Cartesian waypoints and numerical IK; a privileged-state baseline
  for validating the simulator and producing demonstrations.
- `smolvla`: real LeRobot inference with `lerobot/smolvla_base` or a fine-tuned
  checkpoint, using RGB cameras, joint state, and the task instruction. There is
  no fallback to the scripted controller. The base model is not assumed to solve
  this new scene without fine-tuning.

This is a **pushing** demo, not yet pick-and-place or cooperative manipulation.

## Run in this workspace

The isolated Python 3.12 environment, dependencies, robot assets, and base model
have been installed locally. No system Python changes are needed.

```bash
source .venv/bin/activate
export HF_HOME="$PWD/.cache/huggingface"

# Interactive robosuite task through the MuJoCo viewer (8-second episode)
so101-demo --backend robosuite --steps 200 --viewer

# Headless physics baseline, with a video
so101-demo --steps 200 --video outputs/robosuite-push.mp4

# Actual base-model inference on the GPU
so101-demo --policy smolvla --device cuda --video outputs/robosuite-smolvla.mp4

# Quick model integration smoke test
so101-demo --policy smolvla --device cuda --steps 2

python -m pytest -q
```

The default headless rendering backend is EGL; a desktop viewer uses GLFW.
Set `MUJOCO_GL` before launching if your platform requires another backend.
This machine has a working RTX 5070 Ti Laptop GPU (12 GB VRAM) and NVIDIA driver
595.84, verified with `nvidia-smi` outside the agent sandbox. The sandbox hides
the GPU device nodes; its initial driver error was not a host driver failure.
The GPU environment uses PyTorch `2.7.1+cu128` and torchvision `0.22.1+cu128`.
Run GPU commands from a normal terminal, or with approved GPU access outside
the agent sandbox. The default `--device auto` selects CUDA when available and
CPU otherwise. Explicit `--device cuda` fails if CUDA is unavailable; it never
silently falls back. Use `--device cpu` to request CPU inference.

Original standalone GPU verification passed: CUDA matrix multiplication and a full 240-step
SmolVLA rollout. The GPU run took 20.33 seconds including model loading and
video recording, with about 1.5 GB GPU process memory observed in `nvidia-smi`.
Its artifacts are `outputs/smolvla-gpu.mp4` and
`outputs/smolvla-gpu-report.json`. The base model made no block contact in this
run; switching compute devices does not provide task-specific training.

Original standalone verification: scripted pushing succeeded on seeds 0–9; six automated tests
passed; the base SmolVLA completed a 240-step CPU rollout but made no block
contact and did not solve the task. This is one integration rollout, not a
benchmark. The saved videos are `outputs/scripted-push.mp4` and
`outputs/smolvla-base.mp4`, with separate JSON reports. Fine-tuning remains the
next step for learned task success.
The exported dataset was reopened successfully with three RGB tensors and
50 × 6 action chunks, including normalization statistics. Headless rendering
and video playback were checked; the interactive desktop viewer was not opened.

The cached model revisions used for verification were SmolVLA
`5e8d12a6e2975b0e5e5fce7c8caf47c371d257b6` and SmolVLM2
`7b375e1b73b11138ff12fe22c8f2822d8fe03467`. The default Hub IDs follow `main`
on a fresh download; retain these cached snapshots for exact model reproduction.

Reports go to `outputs/last-run.json`, or a path supplied with `--report`.
`success` requires the block center to stay within 19 mm of the target center
on both horizontal axes, at table height and below 2.5 cm/s, for 15 control
steps in the standalone backend (0.5 s), or 13 steps at 25 Hz in robosuite.
This approximately keeps the full cube inside the target.
Episodes have a fixed horizon; they continue after success so release/settling
is visible. The report records success at the end of the episode.

## Install on another Linux machine

Use Python 3.12 and [uv](https://docs.astral.sh/uv/). The simulation alone does not
need PyTorch or LeRobot.

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e '.[robosuite,dev]'
python scripts/fetch_assets.py
so101-demo --viewer

# Optional: SmolVLA + dataset tools, CPU installation
uv pip install --torch-backend cpu -e '.[vla]'

# For this RTX 5070 Ti: replace CPU PyTorch with the CUDA 12.8 builds
uv pip install --torch-backend cu128 'torch==2.7.1+cu128' 'torchvision==0.22.1+cu128'
```

`requirements-tested-cpu.txt` and `requirements-tested-cuda.txt` preserve the
original standalone environments. `requirements-tested-robosuite-cuda.txt`
records the current combined environment. Reproduce it with
`uv pip install --torch-backend cu128 -r requirements-tested-robosuite-cuda.txt`,
followed by `uv pip install --no-deps -e .`.
Robot meshes are downloaded only from Menagerie revision
`d7b481bac63d3d600ab404ee74f7474c5efe8a67`; its Apache-2.0 license is retained in
`vendor/robotstudio_so101/LICENSE`. Model downloads are cached under
`.cache/huggingface`. After the first complete download, use `HF_HUB_OFFLINE=1`
to run without network access.

## Observation and action contract

| Item | Convention |
|---|---|
| Joint order | shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper |
| Simulator action | Six absolute position targets in radians; clipped to joint limits |
| Control / physics | robosuite: 25 Hz / 500 Hz; standalone: 30 Hz / 500 Hz |
| `observation.state` | Six measured joint positions, converted to policy units |
| `observation.images.camera1` | Fixed front view, RGB 256 × 256 |
| `observation.images.camera2` | Menagerie wrist camera, RGB 256 × 256 |
| `observation.images.camera3` | Fixed overhead view, RGB 256 × 256 |
| Task | “Push the red block into the green target.” |
| SmolVLA execution | Replan after 10 actions by default; `--action-steps` changes this |

The adapter converts RGB to float CHW in [0, 1], then uses the checkpoint's
pre/postprocessors for tokenization, state normalization, and action
unnormalization. It resets the policy's action queue at episode start. The base
checkpoint expects three cameras and six state/action values.

The default `--units lerobot-range` maps the five arm joints linearly from the
MJCF limits to [-100, 100], and the gripper to [0, 100]. **This is a simulation
calibration convention, not verified physical SO-101 motor calibration or a
guaranteed match to every pretraining dataset.** Camera placement is also new.
Both are distribution shifts for the base model. Record and fine-tune using
this same convention. Use `--units radians` only for checkpoints trained on
recordings made with that option. Never feed normalized neural outputs directly
to MuJoCo controls or reuse this adapter on hardware without calibration.

## Record demonstrations and fine-tune

```bash
so101-demo --steps 200 --seed 0 --record outputs/robosuite-episode-000.npz
so101-demo --steps 200 --seed 1 --record outputs/robosuite-episode-001.npz

python scripts/export_dataset.py outputs/robosuite-episode-*.npz \
  --root outputs/my-dataset --repo-id local/so101-block-push
```

Each NPZ stores pre-action state, the corresponding applied target, three RGB
views, and episode metadata at the backend's control frequency (25 or 30 Hz).
The exporter accepts successful scripted
episodes with consistent units and writes a **local LeRobot v3 dataset** with
image features, statistics, joint names, and `simulation_contract.json`. It
does not upload anything. Choose a new output directory for each export.
An initial standalone dataset remains at `outputs/dataset`. A validated 25 Hz
robosuite dataset is saved at `outputs/robosuite-dataset`; the training example
below uses that one only as a pipeline starting point. Change `--dataset.root`
to your larger exported dataset before a real training run.

A starting fine-tuning command, for a machine with CUDA-enabled PyTorch:

```bash
lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --dataset.repo_id=local/so101-robosuite-push \
  --dataset.root="$PWD/outputs/robosuite-dataset" \
  --output_dir=outputs/train/smolvla-push \
  --job_name=smolvla-push \
  --batch_size=8 --steps=5000 --eval_freq=0 --wandb.enable=false

so101-demo --policy smolvla \
  --checkpoint outputs/train/smolvla-push/checkpoints/last/pretrained_model \
  --device cuda --seed 100 --viewer
```

Training has **not** been run or tuned. Two demonstrations only check the data
pipeline; collect varied successful episodes and reserve disjoint seeds for
evaluation. Current randomization is deliberately narrow: ±8 mm in block x/y,
with the target x aligned to the block. Increase task/visual variation gradually,
and evaluate multiple held-out seeds before claiming generalization. A single
instruction/task does not test language grounding.

## Files and next research step

- `src/so101_demo/env.py`: scene, reset, contacts, success, IK baseline.
- `src/so101_demo/policy.py`: explicit units and SmolVLA preprocessing.
- `src/so101_demo/run.py`: viewer, video, trajectories, JSON metrics.
- `scripts/export_dataset.py`: local training data conversion.
- `tests/test_demo.py`: contact success, no-op failure, reset, timing, invalid
  actions, and unit conversion.

For an undergrad project, first establish a fine-tuned single-arm success rate.
Then add an obstacle and collision metrics, followed by a second arm with
prefixed body/joint names and explicit per-arm observations/actions. Independent
copies of the base policy are not a cooperative controller. Keep joint-space
trajectory prediction and inter-arm collision geometry explicit when adapting
the safety guidance work.

## Upstream references

- [MuJoCo Menagerie SO-101](https://github.com/google-deepmind/mujoco_menagerie/tree/main/robotstudio_so101)
- [SmolVLA base model](https://huggingface.co/lerobot/smolvla_base)
- [LeRobot SmolVLA documentation](https://huggingface.co/docs/lerobot/v0.4.3/en/smolvla)
- [Neuro-Symbolic Safety Guidance for VLAs via Constrained Flow Matching](https://arxiv.org/html/2607.01378)
# Cooperative-SmolVLA-Policy
