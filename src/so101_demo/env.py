"""MuJoCo SO-101 block push. All public actions are absolute joint radians."""
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[2]
JOINT_NAMES = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
TASK = "Push the red block into the green target."
CAMERAS = {"camera1": "front", "camera2": "wrist_cam", "camera3": "overhead"}


def scene_xml():
    source = ROOT / "vendor/robotstudio_so101/so101.xml"
    if not source.exists():
        raise FileNotFoundError("Run: python scripts/fetch_assets.py")
    root = ET.parse(source).getroot()
    root.find("compiler").set("meshdir", str(source.parent / "assets"))
    root.find("option").set("timestep", "0.002")
    root.find("option").set("iterations", "50")
    visual = root.find("visual")
    ET.SubElement(visual, "global", offwidth="640", offheight="480")
    ET.SubElement(visual, "quality", shadowsize="1024", offsamples="0")
    ET.SubElement(visual, "headlight", ambient="0.4 0.4 0.4", diffuse="0.7 0.7 0.7")
    world = root.find("worldbody")
    ET.SubElement(world, "light", pos="0 -0.4 1.2", dir="0 0 -1", diffuse="0.8 0.8 0.8")
    ET.SubElement(world, "geom", name="table", type="plane", size="0.6 0.6 0.02", rgba="0.24 0.29 0.34 1", friction="0.5 0.005 0.0001")
    ET.SubElement(world, "site", name="goal", type="box", pos="0.25 0.075 0.001", size="0.035 0.035 0.001", rgba="0.15 0.8 0.35 0.6")
    block = ET.SubElement(world, "body", name="block", pos="0.25 -0.055 0.016")
    ET.SubElement(block, "freejoint", name="block_free")
    ET.SubElement(block, "geom", name="block_geom", type="box", size="0.015 0.015 0.015", mass="0.035", rgba="0.85 0.12 0.08 1", friction="0.5 0.005 0.0001", condim="4")
    ET.SubElement(world, "camera", name="front", pos="0.65 -0.65 0.55", xyaxes="0.707 0.707 0 -0.408 0.408 0.816", fovy="48")
    ET.SubElement(world, "camera", name="overhead", pos="0.18 0 0.72", xyaxes="1 0 0 0 1 0", fovy="55")
    return ET.tostring(root, encoding="unicode")


class BlockPushEnv:
    dt = 1 / 30

    def __init__(self, seed=0):
        self.model = mujoco.MjModel.from_xml_string(scene_xml())
        self.data = mujoco.MjData(self.model)
        self.ik_data = mujoco.MjData(self.model)
        self.joints = np.array([self.model.joint(n).id for n in JOINT_NAMES])
        self.qadr = self.model.jnt_qposadr[self.joints]
        self.limits = self.model.jnt_range[self.joints].copy()
        self.site = self.model.site("gripperframe").id
        self.block_adr = self.model.joint("block_free").qposadr[0]
        self.renderers = {}
        self.reset(seed)

    def reset(self, seed=0):
        mujoco.mj_resetData(self.model, self.data)
        rng = np.random.default_rng(seed)
        self.start = np.array([0.25, -0.055, 0.015])
        self.start[:2] += rng.uniform(-0.008, 0.008, 2)
        self.goal = np.array([self.start[0], 0.075, 0.015])
        self.model.site("goal").pos[:2] = self.goal[:2]
        self.data.qpos[self.block_adr:self.block_adr + 3] = self.start + [0, 0, 0.001]
        self.data.qpos[self.block_adr + 3:self.block_adr + 7] = [1, 0, 0, 0]
        home = self.ik([self.start[0], self.start[1] - 0.06, 0.13], seed=np.array([0, -0.5, 0.8, 0.8, 0, 0]))
        self.data.qpos[self.qadr] = home
        self.data.ctrl[:] = home
        mujoco.mj_forward(self.model, self.data)
        self.steps = 0
        self.hold_steps = 0
        self.contact_steps = 0
        return self.state()

    def ik(self, position, seed=None):
        """Position + downward tool axis; five arm joints, closed gripper."""
        if seed is None:
            seed = self.data.qpos[self.qadr].copy()
        target = np.asarray(position)

        def residual(q):
            self.ik_data.qpos[self.qadr[:5]] = q
            self.ik_data.qpos[self.qadr[5]] = 0
            mujoco.mj_forward(self.model, self.ik_data)
            direction = self.ik_data.site_xmat[self.site].reshape(3, 3)[:, 0]
            return np.r_[self.ik_data.site_xpos[self.site] - target,
                         0.1 * (direction - [0, 0, -1])]

        result = least_squares(residual, np.clip(seed[:5], self.limits[:5, 0] + 1e-6, self.limits[:5, 1] - 1e-6),
                               bounds=(self.limits[:5, 0], self.limits[:5, 1]), max_nfev=80)
        return np.r_[result.x, 0.0]

    def state(self):
        return self.data.qpos[self.qadr].astype(np.float32).copy()

    @property
    def block_position(self):
        return self.data.body("block").xpos.copy()

    def step(self, action):
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (6,) or not np.isfinite(action).all():
            raise ValueError("Expected six finite absolute joint targets (radians)")
        self.data.ctrl[:] = np.clip(action, self.limits[:, 0], self.limits[:, 1])
        # Round cumulative time, rather than drifting with a fixed 16 substeps.
        target_time = (self.steps + 1) * self.dt
        touching = False
        block_id = self.model.geom("block_geom").id
        while self.data.time < target_time - self.model.opt.timestep / 2:
            mujoco.mj_step(self.model, self.data)
            for c in self.data.contact:
                if block_id in (c.geom1, c.geom2):
                    other = c.geom2 if c.geom1 == block_id else c.geom1
                    if self.model.geom_bodyid[other] != 0:
                        touching = True
        self.steps += 1
        self.contact_steps += int(touching)
        pos = self.block_position
        speed = np.linalg.norm(self.data.qvel[self.model.joint("block_free").dofadr[0]:][:3])
        inside = bool(np.all(np.abs(pos[:2] - self.goal[:2]) < 0.019) and 0.005 < pos[2] < 0.025)
        self.hold_steps = self.hold_steps + 1 if inside and speed < 0.025 else 0
        return {"success": self.hold_steps >= 15, "distance": float(np.linalg.norm(pos[:2] - self.goal[:2])),
                "block_position": pos.tolist(), "contact_steps": self.contact_steps, "steps": self.steps}

    def render(self, camera="front", width=640, height=480):
        key = (width, height)
        if key not in self.renderers:
            self.renderers[key] = mujoco.Renderer(self.model, height=height, width=width)
        renderer = self.renderers[key]
        options = mujoco.MjvOption()
        options.geomgroup[3:5] = 0
        renderer.update_scene(self.data, camera=camera, scene_option=options)
        return renderer.render().copy()

    def observation(self):
        return {"observation.state": self.state(), "task": TASK,
                **{f"observation.images.{key}": self.render(camera, 256, 256) for key, camera in CAMERAS.items()}}

    def close(self):
        for renderer in self.renderers.values():
            renderer.close()
        self.renderers.clear()


class ScriptedPush:
    """Privileged-state baseline for checking physics, not a learned policy."""
    def __init__(self, env):
        x, y, z = env.start
        surface = z - 0.015
        self.waypoints = [(1.5, [x, y - 0.045, surface + 0.025]),
                          (4.0, [x, env.goal[1] - 0.025, surface + 0.025]),
                          (1.0, [x, env.goal[1] - 0.05, surface + 0.10]),
                          (1.5, [x, env.goal[1] - 0.05, surface + 0.10])]
        self.origin = env.data.site_xpos[env.site].copy()
        self.q = env.state()

    def action(self, env):
        time = env.steps * env.dt
        previous = self.origin
        for duration, target in self.waypoints:
            target = np.asarray(target)
            if time <= duration:
                t = np.clip(time / duration, 0, 1)
                t = t * t * (3 - 2 * t)
                self.q = env.ik(previous + t * (target - previous), seed=self.q)
                return self.q
            time -= duration
            previous = target
        return self.q
