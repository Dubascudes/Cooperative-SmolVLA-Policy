"""SmolVLA adapter; normalization is loaded from the checkpoint processors."""
import numpy as np


def to_policy_units(q, limits, units):
    q = np.asarray(q)
    if units == "radians":
        return q.astype(np.float32)
    result = 200 * (q - limits[:, 0]) / (limits[:, 1] - limits[:, 0]) - 100
    result[5] = (result[5] + 100) / 2
    return result.astype(np.float32)


def from_policy_units(action, limits, units):
    action = np.asarray(action, dtype=np.float64).copy()
    if action.shape != (6,) or not np.isfinite(action).all():
        raise ValueError(f"Policy must produce six finite values, got {action}")
    if units == "radians":
        return action
    action[5] = 2 * action[5] - 100
    return limits[:, 0] + (action + 100) / 200 * (limits[:, 1] - limits[:, 0])


class SmolVLAPolicyAdapter:
    def __init__(self, checkpoint, device="cpu", units="lerobot-range", action_steps=10):
        import torch
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

        self.torch = torch
        self.units = units
        config = PreTrainedConfig.from_pretrained(checkpoint)
        config.device = device
        config.n_action_steps = action_steps
        # Full policy weights already contain the VLM; avoid a second weight download.
        config.load_vlm_weights = False
        if tuple(config.input_features["observation.state"].shape) != (6,) or tuple(config.output_features["action"].shape) != (6,):
            raise ValueError("This demo requires a six-joint SO-101 checkpoint")
        self.policy = SmolVLAPolicy.from_pretrained(checkpoint, config=config).to(device).eval()
        self.preprocess, self.postprocess = make_pre_post_processors(
            self.policy.config, pretrained_path=checkpoint,
            preprocessor_overrides={"device_processor": {"device": device}},
        )
        self.policy.reset()

    def action(self, env):
        obs = env.observation()
        batch = {"task": obs["task"],
                 "observation.state": self.torch.from_numpy(to_policy_units(obs["observation.state"], env.limits, self.units))}
        for key in self.policy.config.image_features:
            if key not in obs:
                raise ValueError(f"Checkpoint camera {key!r} has no simulator mapping")
            batch[key] = self.torch.from_numpy(obs[key]).permute(2, 0, 1).float() / 255.0
        with self.torch.inference_mode():
            action = self.postprocess(self.policy.select_action(self.preprocess(batch)))
        return from_policy_units(action.squeeze(0).cpu().numpy(), env.limits, self.units)
