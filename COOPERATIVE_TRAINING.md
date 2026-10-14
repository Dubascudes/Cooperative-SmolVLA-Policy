# Training a cooperative SO-101 policy

Start with **one centralized SmolVLA policy controlling both arms**. This gives
the undergraduate one dataset, one model, and a measurable cooperative task.
Independent communicating policies can be a later experiment against this baseline.

## Current scene and validation

```bash
source .venv/bin/activate
python scripts/robosuite_two_arm.py --viewer
# Adjust all object dimensions together; default is 0.55
python scripts/robosuite_two_arm.py --viewer --object-scale 0.65
# Original benchmark geometry remains available
python scripts/robosuite_two_arm.py --viewer --stock
```

`SO101CooperativeLift` scales the pot body, walls, handles, and resulting
mass/inertia together. At 0.55 scale the body is 7.7 cm wide, the handle bars
are 1.1 cm thick, and mass is about 202 g at the specified density. Bases are
repositioned to keep each handle about 23 cm forward of its arm. Seed 0 admits
downward-tool static IK solutions for both handle bars within joint limits;
this does **not** establish collision-free approaches or reliable grasps.

Success requires each gripper's opposing finger groups to contact its assigned
handle, nominal pot-bottom clearance of at least 5 cm, tilt at most 15 degrees,
and 13 consecutive control steps (0.52 s at 25 Hz). Losing either grasp resets
the hold counter. Contact is a proxy for support, not a measurement of load
sharing. This is a custom task, not a score on stock `TwoArmLift`.

The viewer still runs small joint motions, **not a lifting expert or learned
policy**. The 15-test suite checks scaling, dynamics smoke tests, success logic,
single-arm push behavior, and independent action routing. The cooperative
success test uses mocked contact metrics and does not demonstrate a physical lift.

## 1. Establish a successful demonstration controller

Implement a privileged, coordinated teacher before collecting training data:

1. Move both open grippers to pre-grasp poses above their assigned handles.
2. Descend together and close slowly; wait for both grasp contacts.
3. Raise both handle targets together, keeping relative height fixed.
4. Hold level for at least 0.52 s; stop and retry if either grasp is lost.

Use object pose/contact information for the teacher and success checks only.
The learned policy should receive camera images and joint state, not privileged
object pose. First prove at least 20 successful physical lifts on fixed/narrowly
randomized resets. If grasping fails, tune approach geometry, wrist orientation,
gripper closure and controller tracking before collecting data. Paired teleoperation
is an alternative if scripted acquisition is unreliable. Do not use the current
joint-wiggle video or single-arm push dataset as cooperative demonstrations.

Then start with roughly 100–200 successful, diverse episodes. Treat that as an
initial data budget, not a guarantee. Expand the current ±3 mm object-position
randomization gradually after the teacher is reliable; vary starting joint poses
and approach timing as well. Split by complete episode/reset seed, never by frames.
Keep separate training, validation, and untouched rollout-test seeds.

## 2. Record a consistent 12-dimensional dataset

Proposed LeRobot v3 dataset contract (requires a new two-arm recorder/exporter):

| Feature | Meaning |
|---|---|
| `observation.state`, shape `(12,)` | Measured robot 0 joints, then robot 1 joints |
| `action`, shape `(12,)` | Applied absolute targets in that same order |
| Per-arm joint order | shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper |
| `observation.images.camera1` | Shared fixed view containing both arms and the object |
| `observation.images.camera2` | Robot 0 wrist view |
| `observation.images.camera3` | Robot 1 wrist view |
| Images and timing | RGB 256 × 256, pre-action observations at 25 Hz |
| Task text | “Lift the pot together and keep it level.” |

Use the existing range-based simulation convention for **each six-joint group**:
five arm coordinates in [-100, 100], gripper in [0, 100], derived from MJCF joint
limits. Record measured state separately from targets. Convert policy predictions
back to radians, clip to joint limits, then convert each gripper target to native
robosuite [-1, 1]. Native commands are arm radians plus abstract gripper aperture;
do not normalize that mixed-unit vector as if all six values were radians.

Save scene scale, controller configuration, seed, camera mapping, units and joint
order alongside the dataset. Compute normalization statistics from training
demonstrations only. Keep failed episodes separately for debugging. Record grasp,
height, tilt and collision traces for evaluation, not policy observations.

## 3. Fine-tune the base model on this GPU

The installed SmolVLA has internal state/action capacities of 32, so a 12D
embodiment can reuse the base tensor weights. Its checkpoint declares 6D features
and normalization, which must be replaced with the new dataset features/stats.
In this LeRobot 0.4.3 CLI, explicitly override the state feature to 12D while
retaining the three camera features; output features are dataset-derived.
Passing an empty input-feature dictionary does not work here: the CLI merges it
with the base configuration and retains the six-dimensional state. The trainer
overrides checkpoint normalizers with dataset statistics during fine-tuning.
Twelve-dimensional capacity does not confer pretrained cooperative behavior.

This diagnostic already passed on the RTX 5070 Ti Laptop:

```bash
HF_HUB_OFFLINE=1 python scripts/check_cooperative_smolvla.py \
  --batch-size 2 --report outputs/cooperative-smolvla-gpu-check.json
```

It uses synthetic inputs/identity statistics, verifies finite gradients, executes
one AdamW step and predicts `(2, 50, 12)` actions with bfloat16 autocast.
Peak allocated memory was 1.95 GiB, excluding CUDA context/other processes and
additional real-training overhead. It saves no checkpoint and is not task training.

Once the **new cooperative dataset** exists, this is a starting training command:

```bash
export HF_HOME="$PWD/.cache/huggingface"
ACCELERATE_MIXED_PRECISION=bf16 lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --policy.input_features='{"observation.state":{"type":"STATE","shape":[12]}}' \
  --policy.device=cuda --policy.load_vlm_weights=false \
  --policy.train_expert_only=true --policy.freeze_vision_encoder=true \
  --policy.train_state_proj=true --policy.push_to_hub=false \
  --dataset.repo_id=local/so101-cooperative-lift \
  --dataset.root="$PWD/outputs/cooperative-train-dataset" \
  --output_dir=outputs/train/smolvla-cooperative-lift \
  --job_name=smolvla-cooperative-lift \
  --batch_size=2 --steps=10000 --num_workers=2 \
  --save_freq=1000 --eval_freq=0 --wandb.enable=false
```

Use a fresh output directory. Start with a short pipeline run, check that saved
config/state/action statistics really have 12 dimensions, then run the experiment.
The command uses Accelerate's mixed-precision setting because the installed trainer
uses `accelerator.autocast()`. Keep the pretrained 50-step chunk initially; begin
rollouts by executing only 5–10 actions before replanning. Tune that horizon by
measured inference latency and grasp stability. Validation loss alone is insufficient.

## 4. Evaluate closed-loop cooperation

Build a two-arm inference adapter using the exact training camera mapping, units,
joint order and checkpoint processors. The current `SmolVLAPolicyAdapter` explicitly
accepts only six-action checkpoints and must not be used unchanged. The existing
NPZ exporter is also single-arm. Training's `eval_freq=0` disables the built-in
evaluator; this custom task still needs its own rollout runner.

Evaluate the teacher, a no-op baseline and the fine-tuned policy on the same held-out
seeds. Log success rate (with episode counts/uncertainty), time to success, dropped
grasps, peak tilt, inter-arm collisions, and control/inference timing. Run an arm-disabled
ablation to characterize dependence on both arms. The task predicate enforces two
contacts; that alone does not prove the object is physically impossible to lift
with one arm. Compare decentralized policies or a safety verifier only after this
centralized baseline is reliable.

**Next implementation milestone:** a repeatable two-arm grasp-and-lift teacher,
followed by the 12D recorder/exporter and rollout adapter. No cooperative dataset,
task fine-tuning run, or physically successful cooperative lift has been produced yet.

Reference: [LeRobot 0.4.3 SmolVLA documentation](https://huggingface.co/docs/lerobot/v0.4.3/en/smolvla).
The dimension/processor details above were also checked against the installed source
and the GPU diagnostic in this workspace.
