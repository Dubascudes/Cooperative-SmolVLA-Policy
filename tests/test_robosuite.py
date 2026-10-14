import numpy as np
import pytest

pytest.importorskip("robosuite")
import robosuite as suite
from so101_demo.env import BlockPushEnv, ScriptedPush
from so101_demo.robosuite_env import RobosuiteBlockPushEnv
from so101_demo.robosuite_models import TWO_ARM_HOME, controller_config


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_native_contact_push(seed):
    env = RobosuiteBlockPushEnv(seed)
    try:
        policy = ScriptedPush(env)
        for _ in range(200):
            info = env.step(policy.action(env))
        assert info["success"], info
        assert info["contact_steps"] > 20
        assert info["reward"] == 1
        assert abs(env.data.time - 8) < 1e-8
        env.reset(seed)
        initial = env.data.qpos.copy()
        env.step(env.state())
        assert not env.suite._check_success()
        env.reset(seed)
        np.testing.assert_allclose(env.data.qpos, initial)
    finally:
        env.close()


def test_kinematics_preserved():
    import mujoco
    original = BlockPushEnv()
    native = RobosuiteBlockPushEnv()
    try:
        for seed in range(3):
            q = np.random.default_rng(seed).uniform(original.limits[:, 0], original.limits[:, 1])
            for env in [original, native]:
                env.data.qpos[env.qadr] = q
                mujoco.mj_forward(env.model, env.data)
            np.testing.assert_allclose(native.data.site_xpos[native.site],
                                       original.data.site_xpos[original.site] + [-0.25, 0, 0.8], atol=1e-8)
            np.testing.assert_allclose(native.data.site_xmat[native.site], original.data.site_xmat[original.site], atol=1e-8)
    finally:
        original.close()
        native.close()


def test_two_independent_arms():
    env = suite.make("TwoArmLift", robots=["SO101", "SO101"], env_configuration="opposed",
                     controller_configs=[controller_config(), controller_config()], initialization_noise=None,
                     has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False, hard_reset=False, seed=0)
    try:
        for robot in env.robots:
            robot.init_qpos = TWO_ARM_HOME.copy()
        obs = env.reset()
        assert env.action_dim == 12
        assert "robot0_joint_pos" in obs and "robot1_joint_pos" in obs
        assert set(env.robots[0].robot_model.joints).isdisjoint(env.robots[1].robot_model.joints)
        action = np.tile(np.r_[TWO_ARM_HOME, 0.81818], 2)
        action[0] += 0.1
        for _ in range(20):
            obs, _, _, _ = env.step(action)
        assert np.isfinite(env.sim.data.qpos).all()
        # Pot contact can deflect either arm; assert independent command routing, not perfect tracking in contact.
        np.testing.assert_allclose(env.robots[0].part_controllers["right"].goal_qpos, action[:5])
        np.testing.assert_allclose(env.robots[1].part_controllers["right"].goal_qpos, TWO_ARM_HOME)
        assert not env._check_success()
    finally:
        env.close()
