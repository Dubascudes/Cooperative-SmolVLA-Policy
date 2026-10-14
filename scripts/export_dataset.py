"""Convert recorded NPZ episodes to a local LeRobot v3 dataset; no upload."""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parents[1] / ".cache/huggingface"))

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from so101_demo.env import CAMERAS, JOINT_NAMES, TASK


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episodes", nargs="+", type=Path)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--repo-id", default="local/so101-block-push")
    args = parser.parse_args()
    if args.root.exists():
        parser.error("Output root already exists; choose a new directory")
    units = set()
    rates = set()
    backends = set()
    for path in args.episodes:
        with np.load(path, allow_pickle=False) as episode:
            meta = json.loads(str(episode["metadata"]))
            if not meta["success"] or meta["policy"] != "scripted":
                parser.error(f"Expected successful scripted demonstration: {path}")
            units.add(meta["units"])
            rates.add(int(episode["fps"]))
            backends.add(meta.get("backend", "mujoco"))
    if len(units) != 1:
        parser.error("Episodes use different action units")
    if len(rates) != 1 or len(backends) != 1:
        parser.error("Do not mix control frequencies or simulation backends in one dataset")
    fps = rates.pop()
    features = {name: {"dtype": "float32", "shape": (6,), "names": list(JOINT_NAMES)}
                for name in ["observation.state", "action"]}
    features.update({f"observation.images.{key}": {"dtype": "image", "shape": (256, 256, 3),
                                                   "names": ["height", "width", "channels"]} for key in CAMERAS})
    dataset = LeRobotDataset.create(repo_id=args.repo_id, root=args.root, robot_type="so101_follower",
                                    fps=fps, features=features, use_videos=False, image_writer_threads=4)
    try:
        for path in args.episodes:
            with np.load(path, allow_pickle=False) as episode:
                # NPZ lazy reads decompress the entire array; load once per episode.
                arrays = {key: episode[key] for key in ["state", "action", *CAMERAS]}
            for i in range(len(arrays["action"])):
                dataset.add_frame({"observation.state": arrays["state"][i].astype(np.float32),
                                   "action": arrays["action"][i].astype(np.float32), "task": TASK,
                                   **{f"observation.images.{key}": arrays[key][i] for key in CAMERAS}})
            dataset.save_episode()
    finally:
        dataset.finalize()
        dataset.stop_image_writer()
    (args.root / "simulation_contract.json").write_text(json.dumps({"units": units.pop(),
        "fps": fps, "backend": backends.pop(),
        "joint_order": JOINT_NAMES, "camera_mapping": CAMERAS, "task": TASK,
        "note": "Range-based simulation calibration, not physical motor calibration."}, indent=2) + "\n")
    print(f"Local LeRobot dataset saved to {args.root}")


if __name__ == "__main__":
    main()
