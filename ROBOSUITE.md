# SO-101 in robosuite

The demo now defaults to robosuite, while `--backend mujoco` retains the original
standalone environment. Both use the same pinned Menagerie SO-101 geometry.

Verified in this workspace: 15 tests pass, including five successful native
push episodes (seeds 0–4), deterministic resets, preserved end-effector
kinematics, and independent two-arm command routing. CUDA SmolVLA inference
completed 25 steps on the native task; it did not move the block. The new
dataset was reopened successfully. Headless videos were inspected; the
interactive viewer was not opened during verification.

## Launch

```bash
source .venv/bin/activate
export HF_HOME="$PWD/.cache/huggingface"

# SO-101 pushes a cube into a target using contact physics
so101-demo --backend robosuite --steps 200 --viewer

# SmolVLA on the same native robosuite task, with the RTX 5070 Ti
so101-demo --backend robosuite --policy smolvla --device cuda --steps 200 --viewer

# Two SO-101 instances with the scaled cooperative-task pot
python scripts/robosuite_two_arm.py --viewer

# Original standalone MuJoCo scene
so101-demo --backend mujoco --viewer
```

The two-arm demo commands small joint motions. It is a model/control smoke test,
**not a cooperative lifting policy**. The SmolVLA integration remains single-arm:
the base checkpoint has six action outputs, whereas two SO-101s need twelve.
It is not duplicated or padded to pretend to provide cooperation.

The two-arm viewer defaults to `SO101CooperativeLift`, a custom variant with
55%-scale pot geometry (7.7 cm body, about 202 g) and repositioned robot bases.
Use `--object-scale 0.65` to adjust size or `--stock` for original `TwoArmLift`.
See [COOPERATIVE_TRAINING.md](COOPERATIVE_TRAINING.md) for the centralized 12-action
SmolVLA training plan and verified GPU forward/backward diagnostic.

## What was integrated

`src/so101_demo/robosuite_models.py` registers `SO101` as a `ManipulatorModel`
with `FixedBaseRobot`, and `SO101Gripper` as a `GripperModel`. Importing this
module performs registration; no files inside robosuite are modified.

The MJCF converter expands inherited Menagerie defaults, makes asset paths
absolute, separates the arm and articulated gripper, adds robosuite's required
sites/sensors, and maps visual/collision groups to robosuite conventions.
Generated XML lives in `.cache/robosuite_so101`; source meshes and their license
remain in `vendor/robotstudio_so101`. Prefixes isolate each robot and gripper.
Forward kinematics are checked against the standalone model.

Robosuite's `JOINT_POSITION` controller outputs **torques**, so the five arm
position actuators are converted to motors with the original ±2.94 Nm limits.
The controller uses absolute joint targets, proportional gain 200, derivative
gain 2.731, and no gravity/inertia compensation. The proportional gain is lower
than the source servo gain because robosuite applies external torques rather
than MuJoCo's implicit position-servo forces. This is a simulation controller,
not a verified model of physical hardware servo behavior.

`SO101BlockPush` is a custom task built on robosuite's `Lift` environment
infrastructure. It changes the object to a 3 cm cube, places the robot on the
table, adds a target, and replaces the lift reward/success predicate with push
success. It is **not the standard Lift benchmark**. The wrapper routes every
action through native `env.step()`; it does not bypass robosuite to move objects.
State-aware IK is used only by the scripted baseline.

## Interfaces

| Interface | Meaning |
|---|---|
| Policy-facing action | Six absolute radians, same order as the original demo |
| Native single-arm action | Five absolute joint radians, then gripper aperture in [-1, 1] |
| Native gripper sign | -1 open, +1 closed; continuous absolute command, not accumulated velocity |
| Native two-arm action | Robot 0's six values, then robot 1's six values |
| Control frequency | 25 Hz, exactly 20 MuJoCo substeps at 500 Hz |
| Camera 1 | Fixed front view |
| Camera 2 | SO-101 wrist camera |
| Camera 3 | Overhead view |
| Policy images | RGB 256 × 256; renderer output is already upright |

The native arm action bounds reflect joint limits. The policy wrapper validates
and clips radians before converting the gripper coordinate. Policy units and
checkpoint normalization follow the original adapter documented in README.md.
The default SmolVLA chunk execution length of 10 corresponds to 0.4 s here.

The port uses 25 Hz because robosuite truncates non-integer physics/control step
ratios. Keeping the original 30 Hz against 500 Hz would silently advance only
16 physics steps per action. Videos and recordings now use each backend's actual
control frequency. Dataset export rejects mixed frequencies or backends.

```python
import robosuite as suite
import so101_demo.robosuite_models  # registration
from so101_demo.robosuite_models import TWO_ARM_HOME, controller_config

env = suite.make(
    "TwoArmLift",
    robots=["SO101", "SO101"],
    env_configuration="opposed",
    controller_configs=[controller_config(), controller_config()],
    initialization_noise=None,
    has_renderer=False,
    has_offscreen_renderer=False,
    use_camera_obs=False,
)
for robot in env.robots:
    robot.init_qpos = TWO_ARM_HOME.copy()  # start above the larger stock pot
obs = env.reset()
assert env.action_dim == 12
# obs["robot0_joint_pos"] / obs["robot1_joint_pos"] each contain five arm joints.
# Gripper state is separate: obs["robot0_gripper_qpos"], etc.
env.close()
```

Pass `controller_config()` explicitly; generic operational-space controllers
assume pose controllability that this five-joint arm does not have.

## Record and train

```bash
so101-demo --backend robosuite --steps 200 --seed 0 \
  --record outputs/robosuite-episode-000.npz
python scripts/export_dataset.py outputs/robosuite-episode-*.npz \
  --root outputs/my-robosuite-dataset --repo-id local/so101-robosuite-push
```

Use only recordings from this backend and control frequency when creating the
new training dataset. The previous `outputs/dataset` is from the original 30 Hz
standalone simulation. The same LeRobot fine-tuning command in README.md can
be used with the new dataset root and repo ID. No fine-tuning has been run.
The initial 200-frame, 25 Hz smoke dataset is already at
`outputs/robosuite-dataset` and was reopened with 50 × 6 action chunks.

## Dependencies and reproduction

The project pins robosuite commit
`5ce6643f3092639d08f7b0f90ed1c6a84f50552c` (reports version 1.5.2). The PyPI
1.5.2 release requires Mink 0.0.5, which requires NumPy <2; LeRobot's required
Rerun package requires NumPy >=2. The pinned upstream revision makes Mink
optional, avoiding that conflict. Our controller does not use Mink.

```bash
uv pip install -e '.[robosuite,vla,dev]'
uv pip install --torch-backend cu128 'torch==2.7.1+cu128' 'torchvision==0.22.1+cu128'
python scripts/fetch_assets.py
MUJOCO_GL=egl python -m pytest -q
```

`requirements-tested-robosuite-cuda.txt` captures the combined environment.
Robosuite may log notices about optional `robosuite_models`, Mink, and private
macros. Those packages are not required by this integration. Run CUDA commands
outside the agent sandbox, which hides GPU devices.

## Cooperative research scope

The native `TwoArmLift` scene can reset and step with two SO-101s, but the stock
pot and task were designed for larger arms. Its success predicate checks that
the pot bottom is 10 cm above the table; it does not require both grippers to
hold the pot. Neither solvability nor a cooperative policy has been established
for SO-101 on that task.

`SO101CooperativeLift` now supplies scaled geometry and a success predicate
requiring both assigned-handle grasps, nominal bottom clearance ≥5 cm, tilt ≤15°,
and a sustained 13-step hold at 25 Hz. Static IK reaches both handles, but physical
grasp acquisition and cooperative lifting remain unvalidated. Contact checks are
a proxy for support, not proof of equal load sharing. Keep stock and custom-task
results separate. Next build a coordinated demonstration teacher, a 12D dataset,
and a two-arm inference adapter as described in [the training plan](COOPERATIVE_TRAINING.md).
`TwoArmPegInHole`, transport, and a multi-agent API are not yet validated.

References: [robosuite robot interfaces](https://robosuite.ai/docs/modules/robots.html),
[upstream SO-101 request](https://github.com/ARISE-Initiative/robosuite/issues/720),
[pinned robosuite source](https://github.com/ARISE-Initiative/robosuite/tree/5ce6643f3092639d08f7b0f90ed1c6a84f50552c).
