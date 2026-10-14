"""Small SO-101 two-arm lifting task, separate from the stock benchmark."""
import numpy as np
from robosuite.environments.manipulation.two_arm_lift import TwoArmLift
from robosuite.models.objects import PotWithHandlesObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.placement_samplers import UniformRandomSampler

from so101_demo.robosuite_models import TWO_ARM_HOME, controller_config


class SO101CooperativeLift(TwoArmLift):
    """Scaled geometry, reachable opposing bases, and sustained two-handle grasp success."""

    def __init__(self, *args, object_scale=0.55, **kwargs):
        if not 0.35 <= object_scale <= 1.0:
            raise ValueError("object_scale must lie between 0.35 and 1.0")
        self.object_scale = float(object_scale)
        self.hold_steps = 0
        super().__init__(*args, **kwargs)

    def _load_model(self):
        super()._load_model()
        if len(self.robots) != 2 or self.env_configuration != "opposed":
            raise ValueError("SO101CooperativeLift requires two opposed robots")
        scale = self.object_scale
        # Keep each handle ~23 cm in front of its base as object size changes.
        base_distance = 0.23 + 0.16 * scale
        for robot, sign in zip(self.robots, [-1, 1]):
            robot.robot_model.set_base_xpos([0, sign * base_distance, self.table_offset[2]])
            robot.init_qpos = TWO_ARM_HOME.copy()
        self.pot = PotWithHandlesObject(
            name="pot", body_half_size=np.full(3, 0.07 * scale), handle_radius=0.01 * scale,
            handle_length=0.09 * scale, handle_width=0.09 * scale, thickness=0.01 * scale,
            density=1000, use_texture=False,
        )
        self.placement_initializer = UniformRandomSampler(
            name="SmallPotSampler", mujoco_objects=self.pot, x_range=[-0.003, 0.003], y_range=[-0.003, 0.003],
            rotation=np.pi, ensure_object_boundary_in_range=False, ensure_valid_placement=True,
            reference_pos=self.table_offset, rng=self.rng,
        )
        self.model = ManipulationTask(mujoco_arena=self.model.mujoco_arena,
                                      mujoco_robots=[r.robot_model for r in self.robots], mujoco_objects=self.pot)
        option = self.model.root.find("option")
        option.set("integrator", "implicitfast")
        option.set("cone", "elliptic")
        option.set("iterations", "50")

    def _reset_internal(self):
        super()._reset_internal()
        self.hold_steps = 0

    def cooperation_metrics(self):
        left = self._check_grasp(self.robots[0].gripper["right"], self.pot.handle0_geoms)
        right = self._check_grasp(self.robots[1].gripper["right"], self.pot.handle1_geoms)
        bottom = self.sim.data.site_xpos[self.pot_center_id][2] - self.pot.top_offset[2]
        height = float(bottom - self.sim.data.site_xpos[self.table_top_id][2])
        rotation = self.sim.data.body_xmat[self.pot_body_id].reshape(3, 3)
        tilt = float(np.arccos(np.clip(rotation[2, 2], -1, 1)))
        return {"grasp0": bool(left), "grasp1": bool(right), "lift_height": height,
                "tilt_degrees": float(np.degrees(tilt)), "hold_steps": self.hold_steps}

    def _post_action(self, action):
        metrics = self.cooperation_metrics()
        cooperative = metrics["grasp0"] and metrics["grasp1"] and metrics["lift_height"] >= 0.05 and metrics["tilt_degrees"] <= 15
        self.hold_steps = self.hold_steps + 1 if cooperative else 0
        reward, done, info = super()._post_action(action)
        info.update(self.cooperation_metrics(), success=self._check_success())
        return reward, done, info

    def _check_success(self):
        return self.hold_steps >= int(np.ceil(0.5 * self.control_freq))

    def reward(self, action=None):
        # A sparse task definition. This first training path is supervised imitation.
        return float(self._check_success())


def make_cooperative_env(seed=0, object_scale=0.55, horizon=250):
    import robosuite as suite
    return suite.make("SO101CooperativeLift", robots=["SO101", "SO101"], env_configuration="opposed",
                      controller_configs=[controller_config(), controller_config()], initialization_noise=None,
                      has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False, hard_reset=False,
                      control_freq=25, horizon=horizon, seed=seed, object_scale=object_scale)
