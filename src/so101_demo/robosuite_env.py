"""Native robosuite task and adapter for the existing policy / recording interface."""
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import robosuite as suite
from robosuite.environments.manipulation.lift import Lift

from so101_demo.env import BlockPushEnv, JOINT_NAMES
from so101_demo.robosuite_models import controller_config


class SO101BlockPush(Lift):
    """Custom task, not the standard Lift benchmark. Surface is at z=0.8 m."""

    def _load_model(self):
        super()._load_model()
        body = self.model.worldbody.find(f".//body[@name='{self.cube.root_body}']")
        for geom in body.findall("geom"):
            geom.set("size", "0.015 0.015 0.015")
            geom.set("mass", "0.035" if geom.get("group", "0") == "0" else "0")
            geom.set("rgba", "0.85 0.12 0.08 1")
            geom.set("friction", "0.5 0.005 0.0001")
            geom.set("condim", "4")
            geom.attrib.pop("material", None)
        ET.SubElement(self.model.worldbody, "site", name="push_goal", type="box", pos="0 0.075 0.801",
                      size="0.035 0.035 0.001", rgba="0.15 0.8 0.35 0.6")
        ET.SubElement(self.model.worldbody, "camera", name="push_front", pos="0.4 -0.65 1.35",
                      xyaxes="0.707 0.707 0 -0.408 0.408 0.816", fovy="48")
        ET.SubElement(self.model.worldbody, "camera", name="push_overhead", pos="-0.07 0 1.52",
                      xyaxes="1 0 0 0 1 0", fovy="55")
        option = self.model.root.find("option")
        option.set("integrator", "implicitfast")
        option.set("cone", "elliptic")
        option.set("iterations", "50")
        option.set("impratio", "10")

    def _reset_internal(self):
        super()._reset_internal()
        self.hold_steps = 0
        self.start = np.array([0, -0.055, 0.815])
        self.start[:2] += self.rng.uniform(-0.008, 0.008, 2)
        self.goal = np.array([self.start[0], 0.075, 0.815])
        self.sim.data.set_joint_qpos(self.cube.joints[0], np.r_[self.start + [0, 0, 0.001], [1, 0, 0, 0]])
        self.sim.model.site_pos[self.sim.model.site_name2id("push_goal"), :2] = self.goal[:2]

    def _post_action(self, action):
        pos = self.sim.data.body_xpos[self.cube_body_id]
        adr = self.sim.model.get_joint_qvel_addr(self.cube.joints[0])[0]
        speed = np.linalg.norm(self.sim.data.qvel[adr:adr + 3])
        inside = np.all(np.abs(pos[:2] - self.goal[:2]) < 0.019) and 0.805 < pos[2] < 0.825
        self.hold_steps = self.hold_steps + 1 if inside and speed < 0.025 else 0
        reward, done, info = super()._post_action(action)
        info.update(success=self._check_success(), distance=float(np.linalg.norm(pos[:2] - self.goal[:2])),
                    block_position=pos.tolist())
        return reward, done, info

    def _check_success(self):
        return self.hold_steps >= int(np.ceil(0.5 * self.control_freq))

    def reward(self, action=None):
        return float(self._check_success())


class RobosuiteBlockPushEnv(BlockPushEnv):
    """Same six-radian policy contract, with all physics/control routed through robosuite.step."""
    dt = 1 / 25  # divides robosuite's 500 Hz physics exactly

    def __init__(self, seed=0):
        self.suite = suite.make("SO101BlockPush", robots="SO101", controller_configs=controller_config(),
                                initialization_noise=None, has_renderer=False, has_offscreen_renderer=False,
                                use_camera_obs=False, hard_reset=False, control_freq=25, horizon=10000, seed=seed,
                                table_friction=(0.5, 0.005, 0.0001))
        self.model = self.suite.sim.model._model
        self.data = self.suite.sim.data._data
        self.ik_data = mujoco.MjData(self.model)
        self.qadr = np.array([self.model.joint("robot0_" + name).qposadr[0] for name in JOINT_NAMES[:5]] +
                             [self.model.joint("gripper0_right_gripper").qposadr[0]])
        self.joints = np.array([self.model.joint("robot0_" + name).id for name in JOINT_NAMES[:5]] +
                               [self.model.joint("gripper0_right_gripper").id])
        self.limits = self.model.jnt_range[self.joints].copy()
        self.site = self.model.site("gripper0_right_grip_site").id
        self.block_adr = self.model.joint(self.suite.cube.joints[0]).qposadr[0]
        self.block_id = self.model.body(self.suite.cube.root_body).id
        self.robot_bodies = {i for i in range(self.model.nbody)
                             if self.model.body(i).name.startswith(("robot0_", "gripper0_"))}
        self.renderers = {}
        self.reset(seed)

    def reset(self, seed=0):
        self.suite.rng = np.random.default_rng(seed)
        self.suite.reset()
        self.start = self.suite.start.copy()
        self.goal = self.suite.goal.copy()
        self.steps = self.contact_steps = 0
        return self.state()

    @property
    def block_position(self):
        return self.data.xpos[self.block_id].copy()

    def step(self, action):
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (6,) or not np.isfinite(action).all():
            raise ValueError("Expected six finite absolute joint targets (radians)")
        action = np.clip(action, self.limits[:, 0], self.limits[:, 1])
        suite_action = action.copy()
        suite_action[5] = 1 - 2 * (action[5] - self.limits[5, 0]) / np.ptp(self.limits[5])
        _, reward, done, info = self.suite.step(suite_action)
        self.steps += 1
        # Table is a body in robosuite: count robot contact only.
        touching = any((self.model.geom_bodyid[c.geom1] == self.block_id and self.model.geom_bodyid[c.geom2] in self.robot_bodies) or
                       (self.model.geom_bodyid[c.geom2] == self.block_id and self.model.geom_bodyid[c.geom1] in self.robot_bodies)
                       for c in self.data.contact)
        self.contact_steps += int(touching)
        info.update(steps=self.steps, contact_steps=self.contact_steps, reward=reward, done=done)
        return info

    def render(self, camera="front", width=640, height=480):
        camera = {"front": "push_front", "overhead": "push_overhead", "wrist_cam": "gripper0_right_wrist_cam"}.get(camera, camera)
        key = (width, height)
        if key not in self.renderers:
            self.model.vis.global_.offwidth = max(640, width)
            self.model.vis.global_.offheight = max(480, height)
            self.model.vis.quality.shadowsize = 1024
            self.model.vis.quality.offsamples = 0
            self.renderers[key] = mujoco.Renderer(self.model, height=height, width=width)
        options = mujoco.MjvOption()
        options.geomgroup[0] = 0
        self.renderers[key].update_scene(self.data, camera=camera, scene_option=options)
        return self.renderers[key].render().copy()

    def close(self):
        super().close()
        self.suite.close()
