"""Two SO-101s with a scaled pot. Joint-motion smoke demo, not a lifting policy."""
import argparse
import json
import os
from pathlib import Path
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--viewer", action="store_true")
    p.add_argument("--video", type=Path)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--object-scale", type=float, default=0.55, help="Scale all pot dimensions (default: 0.55)")
    p.add_argument("--stock", action="store_true", help="Use the original stock TwoArmLift instead")
    args = p.parse_args()
    if args.steps < 1:
        p.error("steps must be positive")
    if not args.viewer:
        os.environ.setdefault("MUJOCO_GL", "egl")
    import imageio.v2 as imageio
    import mujoco
    import numpy as np
    import robosuite as suite
    from so101_demo.robosuite_models import TWO_ARM_HOME, controller_config

    if args.stock:
        env = suite.make("TwoArmLift", robots=["SO101", "SO101"], env_configuration="opposed",
                         controller_configs=[controller_config(), controller_config()], initialization_noise=None,
                         has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False, hard_reset=False,
                         control_freq=25, horizon=args.steps, seed=args.seed)
    else:
        from so101_demo.cooperative_env import make_cooperative_env
        env = make_cooperative_env(args.seed, args.object_scale, args.steps)
    viewer = renderer = writer = None
    try:
        for robot in env.robots:
            robot.init_qpos = TWO_ARM_HOME.copy()
        env.reset()
        model, data = env.sim.model._model, env.sim.data._data
        camera = mujoco.MjvCamera()
        camera.lookat[:] = [0, 0, 0.9]
        camera.distance, camera.azimuth, camera.elevation = 1.05, 135, -35
        options = mujoco.MjvOption()
        options.geomgroup[0] = 0
        if args.viewer:
            import mujoco.viewer
            viewer = mujoco.viewer.launch_passive(model, data)
            viewer.cam.lookat[:] = camera.lookat
            viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 1.05, 135, -35
            viewer.opt.geomgroup[0] = 0
        if args.video:
            args.video.parent.mkdir(parents=True, exist_ok=True)
            model.vis.global_.offwidth, model.vis.global_.offheight = 640, 480
            renderer = mujoco.Renderer(model, width=640, height=480)
            writer = imageio.get_writer(str(args.video), fps=25)
        for i in range(args.steps):
            tick = time.monotonic()
            if viewer and not viewer.is_running():
                break
            action = np.tile(np.r_[TWO_ARM_HOME, 0.81818], 2)
            action[0] += 0.08 * np.sin(2 * np.pi * i / 125)
            action[6] -= 0.08 * np.sin(2 * np.pi * i / 125)
            env.step(action)
            if not np.isfinite(data.qpos).all():
                raise RuntimeError("Non-finite simulation state")
            if writer:
                renderer.update_scene(data, camera=camera, scene_option=options)
                writer.append_data(renderer.render())
            if viewer:
                viewer.sync()
                time.sleep(max(0, 0.04 - (time.monotonic() - tick)))
        print(json.dumps({"environment": type(env).__name__, "robots": ["SO101", "SO101"],
                          "object_scale": 1.0 if args.stock else args.object_scale,
                          "pot_mass_kg": float(model.body_mass[env.pot_body_id]),
                          "controller": "joint-motion smoke test (not a cooperative policy)",
                          "action_dim": env.action_dim, "success": bool(env._check_success())}, indent=2))
    finally:
        if writer:
            writer.close()
        if renderer:
            renderer.close()
        if viewer:
            viewer.close()
        env.close()


if __name__ == "__main__":
    main()
