import argparse
import json
import os
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser(description="SO-101 MuJoCo block-pushing demo")
    parser.add_argument("--backend", choices=["mujoco", "robosuite"], default="robosuite")
    parser.add_argument("--policy", choices=["scripted", "smolvla"], default="scripted")
    parser.add_argument("--checkpoint", default="lerobot/smolvla_base")
    parser.add_argument("--device", default="auto", help="auto selects CUDA when available; or specify cpu/cuda")
    parser.add_argument("--units", choices=["lerobot-range", "radians"], default="lerobot-range")
    parser.add_argument("--action-steps", type=int, default=10)
    parser.add_argument("--steps", type=int, default=240)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--video", type=Path)
    parser.add_argument("--record", type=Path, help="Raw observation/action trajectory (.npz), including RGB")
    parser.add_argument("--report", type=Path, default=Path("outputs/last-run.json"))
    args = parser.parse_args()
    if args.steps < 1 or not 1 <= args.action_steps <= 50:
        parser.error("steps must be positive; action-steps must be between 1 and 50")
    if not args.viewer:
        os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parents[2] / ".cache/huggingface"))
    import imageio.v2 as imageio
    import numpy as np
    from so101_demo.env import BlockPushEnv, ScriptedPush, CAMERAS
    from so101_demo.policy import SmolVLAPolicyAdapter, to_policy_units

    if args.backend == "robosuite":
        from so101_demo.robosuite_env import RobosuiteBlockPushEnv
        env = RobosuiteBlockPushEnv(args.seed)
    else:
        env = BlockPushEnv(args.seed)
    fps = round(1 / env.dt)
    viewer = writer = None
    frames = {"state": [], "action": [], **{key: [] for key in CAMERAS}}
    started = time.monotonic()
    try:
        if args.policy == "smolvla":
            import torch
            if args.device == "auto":
                args.device = "cuda" if torch.cuda.is_available() else "cpu"
            if args.device.startswith("cuda") and not torch.cuda.is_available():
                raise RuntimeError("CUDA is unavailable to this process. Check GPU device access (the agent sandbox hides it) and the PyTorch build, or use --device cpu.")
            torch.manual_seed(args.seed)
            print(f"Loading {args.checkpoint} on {args.device}. Base-model success is not assumed.", flush=True)
            policy = SmolVLAPolicyAdapter(args.checkpoint, args.device, args.units, args.action_steps)
        else:
            policy = ScriptedPush(env)
        if args.viewer:
            import mujoco.viewer
            viewer = mujoco.viewer.launch_passive(env.model, env.data)
            viewer.cam.lookat[:] = [0.18, 0, 0.08]
            viewer.cam.distance = 0.85
            viewer.cam.azimuth = 135
            viewer.cam.elevation = -35
            viewer.opt.geomgroup[3:5] = 0
            if args.backend == "robosuite":
                viewer.cam.lookat[:] = [-0.07, 0, 0.88]
                viewer.opt.geomgroup[0] = 0
        if args.video:
            args.video.parent.mkdir(parents=True, exist_ok=True)
            writer = imageio.get_writer(str(args.video), fps=fps)
        for step in range(args.steps):
            tick = time.monotonic()
            if viewer and not viewer.is_running():
                break
            action = policy.action(env)
            # Store the observation BEFORE executing its paired action.
            if args.record:
                obs = env.observation()
                frames["state"].append(to_policy_units(env.state(), env.limits, args.units))
                # Record the clipped target actually applied by the environment.
                frames["action"].append(to_policy_units(np.clip(action, env.limits[:, 0], env.limits[:, 1]), env.limits, args.units))
                for key in CAMERAS:
                    frames[key].append(obs[f"observation.images.{key}"])
            info = env.step(action)
            if writer:
                writer.append_data(env.render())
            if viewer:
                viewer.sync()
                time.sleep(max(0, env.dt - (time.monotonic() - tick)))
            if step % 30 == 0:
                print(f"step={step:03d} distance={info['distance']:.3f}m success={info['success']}", flush=True)
        if env.steps == 0:
            return
        report = {"policy": args.policy, "checkpoint": args.checkpoint if args.policy == "smolvla" else None,
                  "backend": args.backend, "fps": fps,
                  "device": args.device if args.policy == "smolvla" else "cpu",
                  "seed": args.seed, "units": args.units, "wall_seconds": round(time.monotonic() - started, 2), **info}
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n")
        if args.record:
            args.record.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(args.record, **{k: np.asarray(v) for k, v in frames.items()},
                                metadata=json.dumps(report), fps=fps)
        print(json.dumps(report, indent=2))
    finally:
        if writer:
            writer.close()
        if viewer:
            viewer.close()
        env.close()


if __name__ == "__main__":
    main()
