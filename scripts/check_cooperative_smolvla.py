"""CUDA training-shape diagnostic only: synthetic inputs, no learned policy saved."""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parents[1] / ".cache/huggingface"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    import torch
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this GPU feasibility check")
    torch.manual_seed(0)
    checkpoint = "lerobot/smolvla_base"
    cfg = PreTrainedConfig.from_pretrained(checkpoint)
    cfg.device, cfg.load_vlm_weights = "cuda", False
    cfg.input_features["observation.state"] = PolicyFeature(type=FeatureType.STATE, shape=(12,))
    cfg.output_features["action"] = PolicyFeature(type=FeatureType.ACTION, shape=(12,))
    cfg.train_expert_only = cfg.freeze_vision_encoder = cfg.train_state_proj = True
    policy = SmolVLAPolicy.from_pretrained(checkpoint, config=cfg).cuda().train()
    # Identity stats are ONLY for this dimension check. Real training must use demonstration statistics.
    stats = {key: {"mean": torch.zeros(12), "std": torch.ones(12)}
             for key in ["observation.state", "action"]}
    features = {**cfg.input_features, **cfg.output_features}
    pre, post = make_pre_post_processors(cfg, pretrained_path=checkpoint, dataset_stats=stats,
        preprocessor_overrides={"device_processor": {"device": "cuda"},
            "normalizer_processor": {"stats": stats, "features": features, "norm_map": cfg.normalization_mapping}},
        postprocessor_overrides={"unnormalizer_processor": {"stats": stats,
            "features": cfg.output_features, "norm_map": cfg.normalization_mapping}})
    batch = {"observation.state": torch.zeros(args.batch_size, 12),
             "action": torch.zeros(args.batch_size, cfg.chunk_size, 12),
             "task": ["Lift the pot together and keep it level."] * args.batch_size}
    batch.update({key: torch.rand(args.batch_size, 3, 256, 256) for key in cfg.image_features})
    batch = pre(batch)
    optimizer = torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad], lr=1e-4)
    torch.cuda.reset_peak_memory_stats()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss, _ = policy(batch)
    loss.backward()
    grads = [p.grad for p in policy.parameters() if p.grad is not None]
    assert torch.isfinite(loss) and grads and all(torch.isfinite(g).all() for g in grads)
    torch.nn.utils.clip_grad_norm_(policy.parameters(), 10)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    policy.eval()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        actions = post(policy.predict_action_chunk(batch))
    assert tuple(actions.shape) == (args.batch_size, cfg.chunk_size, 12)
    assert torch.isfinite(actions).all()
    report = {"diagnostic_only": True, "data": "synthetic; no checkpoint saved",
              "gpu": torch.cuda.get_device_name(), "batch_size": args.batch_size,
              "action_shape": list(actions.shape), "loss": float(loss.detach()),
              "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
              "finite_gradients": True, "optimizer_step": True}
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
