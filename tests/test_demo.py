import numpy as np
import pytest

from so101_demo.env import BlockPushEnv, ScriptedPush
from so101_demo.policy import from_policy_units, to_policy_units


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_contact_push(seed):
    env = BlockPushEnv(seed)
    policy = ScriptedPush(env)
    for _ in range(240):
        info = env.step(policy.action(env))
    assert info["success"], info
    assert info["contact_steps"] > 10
    assert abs(env.data.time - 8.0) < env.model.opt.timestep
    assert np.isfinite(env.data.qpos).all()
    env.close()


def test_no_action_does_not_succeed_and_reset_is_deterministic():
    env = BlockPushEnv(8)
    initial = env.data.qpos.copy()
    for _ in range(60):
        info = env.step(env.state())
    assert not info["success"]
    env.reset(8)
    np.testing.assert_allclose(initial, env.data.qpos)
    for bad in [np.zeros(7), np.full(6, np.nan)]:
        with pytest.raises(ValueError):
            env.step(bad)
    for units in ["radians", "lerobot-range"]:
        np.testing.assert_allclose(from_policy_units(to_policy_units(env.state(), env.limits, units), env.limits, units), env.state(), atol=1e-6)
    env.close()
