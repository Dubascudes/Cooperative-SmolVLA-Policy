import numpy as np
import pytest

pytest.importorskip("robosuite")
from so101_demo.cooperative_env import make_cooperative_env


def test_scaled_pot_and_stable_controls():
    small, large = make_cooperative_env(), make_cooperative_env(object_scale=1)
    try:
        small.reset()
        large.reset()
        assert small.action_dim == 12
        np.testing.assert_allclose(small.pot.body_half_size, np.full(3, 0.0385))
        assert small.sim.model.body_mass[small.pot_body_id] / large.sim.model.body_mass[large.pot_body_id] == pytest.approx(0.55**3)
        from so101_demo.robosuite_models import TWO_ARM_HOME
        for _ in range(25):
            _, reward, _, info = small.step(np.tile(np.r_[TWO_ARM_HOME, 0.81818], 2))
        assert np.isfinite(small.sim.data.qpos).all()
        assert reward == 0 and not info["success"]
        assert small.sim.data.time == pytest.approx(1.0)
    finally:
        small.close()
        large.close()


def test_cooperative_hold_predicate(monkeypatch):
    env = make_cooperative_env()
    try:
        env.reset()
        metrics = {"grasp0": True, "grasp1": True, "lift_height": 0.06,
                   "tilt_degrees": 0, "hold_steps": 0}
        # Test the task predicate independently of physical grasp acquisition.
        monkeypatch.setattr(env, "cooperation_metrics", lambda: dict(metrics, hold_steps=env.hold_steps))
        action = np.zeros(12)
        for _ in range(12):
            env._post_action(action)
        assert not env._check_success()
        reward, _, info = env._post_action(action)
        assert reward == 1 and info["success"]
        for key, bad in [("grasp0", False), ("grasp1", False), ("lift_height", 0.04), ("tilt_degrees", 16)]:
            old = metrics[key]
            metrics[key] = bad
            env._post_action(action)
            assert env.hold_steps == 0 and not env._check_success()
            metrics[key] = old
        env.hold_steps = 13
        env.reset()
        assert env.hold_steps == 0
    finally:
        env.close()
