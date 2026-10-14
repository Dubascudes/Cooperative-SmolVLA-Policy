"""Fetch only the Apache-2.0 SO-101 model, pinned to a Menagerie revision."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import urllib.request
import xml.etree.ElementTree as ET

REVISION = "d7b481bac63d3d600ab404ee74f7474c5efe8a67"
DEST = Path(__file__).resolve().parents[1] / "vendor" / "robotstudio_so101"
BASE = f"https://raw.githubusercontent.com/google-deepmind/mujoco_menagerie/{REVISION}/robotstudio_so101"


def fetch(name):
    path = DEST / name
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(f"{BASE}/{name}", timeout=60) as response:
            payload = response.read()
        temp = path.with_suffix(path.suffix + ".tmp")
        temp.write_bytes(payload)
        temp.replace(path)
    return path


if __name__ == "__main__":
    xml = fetch("so101.xml")
    meshes = [f"assets/{m.attrib['file']}" for m in ET.parse(xml).findall(".//asset/mesh")]
    with ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(fetch, ["LICENSE", "README.md", *meshes]))
    (DEST / "REVISION").write_text(REVISION + "\n")
    print(f"SO-101 assets ready: {DEST}")
