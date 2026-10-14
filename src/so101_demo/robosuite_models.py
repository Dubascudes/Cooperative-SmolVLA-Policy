"""SO-101 model registration for robosuite 1.5.2, derived from pinned Menagerie assets."""
from copy import deepcopy
import os
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from robosuite.models.robots.manipulators.manipulator_model import ManipulatorModel
from robosuite.models.grippers.gripper_model import GripperModel
from robosuite.models.grippers import register_gripper
from robosuite.robots import register_robot_class

ROOT = Path(__file__).resolve().parents[2]
HOME = np.array([0.507, 0.079, -0.386, 1.658, 0.066])
TWO_ARM_HOME = np.array([0.5, -0.5, 0.0, 1.0, 0.0])


def converted_assets():
    """Expand inherited defaults before splitting arm and gripper; never edit vendor files."""
    source = ROOT / "vendor/robotstudio_so101/so101.xml"
    if not source.exists():
        raise FileNotFoundError("Run python scripts/fetch_assets.py first")
    root = ET.parse(source).getroot()
    classes = {}

    def defaults(node, inherited):
        effective = deepcopy(inherited)
        for child in node:
            if child.tag != "default":
                effective.setdefault(child.tag, {}).update(child.attrib)
        classes[node.get("class", "main")] = effective
        for child in node.findall("default"):
            defaults(child, effective)

    defaults(root.find("default"), {})
    root.remove(root.find("default"))

    def expand(node, inherited="main"):
        childclass = node.get("childclass", inherited)
        original = dict(node.attrib)
        node.attrib.clear()
        node.attrib.update(classes[original.get("class", inherited)].get(node.tag, {}))
        node.attrib.update(original)
        node.attrib.pop("class", None)
        node.attrib.pop("childclass", None)
        for child in node:
            expand(child, childclass)

    expand(root)
    for i, geom in enumerate(root.findall(".//geom")):
        geom.set("name", geom.get("name", f"so101_geom_{i}"))
        collision_mesh_names = {"wrist_roll_follower_so101_gripper_part0_v1": "fixed_jaw_mesh",
                                "moving_jaw_so101_gripper_part0_v1": "moving_jaw_mesh0",
                                "moving_jaw_so101_gripper_part1_v1": "moving_jaw_mesh1"}
        if geom.get("mesh") in collision_mesh_names:
            geom.set("name", collision_mesh_names[geom.get("mesh")])
        visual = geom.get("contype", "1") == "0" and geom.get("conaffinity", "1") == "0"
        geom.set("group", "1" if visual else "0")
    for mesh in root.findall("asset/mesh"):
        mesh.set("name", mesh.get("name", Path(mesh.get("file")).stem))
        mesh.set("file", str(source.parent / "assets" / mesh.get("file")))
    for tag in ["option", "visual"]:
        element = root.find(tag)
        if element is not None:
            root.remove(element)
    root.find("compiler").attrib.pop("meshdir", None)
    ET.SubElement(root.find("worldbody/body"), "site", name="right_center", size="0.001", rgba="0 0 0 0")

    hand = root.find(".//body[@name='gripper']")
    hand.set("name", "right_hand")
    gripper = ET.Element("mujoco", model="so101_gripper")
    gripper.append(deepcopy(root.find("compiler")))
    gripper.append(deepcopy(root.find("asset")))
    gw = ET.SubElement(gripper, "worldbody")
    gb = ET.SubElement(gw, "body", name="so101_gripper")
    for child in list(hand):
        if child.tag not in {"joint", "inertial"}:
            hand.remove(child)
            gb.append(child)
    # Fixed-body geometries must not introduce mass already covered by hand inertia.
    for geom in gb.findall("geom"):
        geom.set("mass", "0")
    eef = ET.SubElement(gb, "body", name="eef", pos="0.012 -0.000218 -0.098127", quat="1 0 1 0")
    for name in ["grip_site", "grip_site_cylinder", "ee", "ee_x", "ee_y", "ee_z"]:
        ET.SubElement(eef, "site", name=name, size="0.002", rgba="0 0 0 0", group="1")
    ET.SubElement(gb, "site", name="ft_frame", size="0.001", rgba="0 0 0 0")
    sensors = ET.SubElement(gripper, "sensor")
    ET.SubElement(sensors, "force", name="force_ee", site="ft_frame")
    ET.SubElement(sensors, "torque", name="torque_ee", site="ft_frame")
    ga = ET.SubElement(gripper, "actuator")
    actuators = root.find("actuator")
    for actuator in list(actuators):
        if actuator.get("joint") == "gripper":
            actuators.remove(actuator)
            ga.append(actuator)
        else:
            # robosuite JOINT_POSITION emits torques; use motor actuators, not position servos.
            name, joint = actuator.get("name"), actuator.get("joint")
            actuator.tag = "motor"
            actuator.attrib.clear()
            actuator.attrib.update(name=name, joint=joint, gear="1", ctrllimited="true", ctrlrange="-2.94 2.94")
    out = ROOT / ".cache/robosuite_so101"
    out.mkdir(parents=True, exist_ok=True)
    for name, xml in [("robot", root), ("gripper", gripper)]:
        ET.indent(xml)
        temporary = out / f"{name}.{os.getpid()}.tmp"
        temporary.write_text(ET.tostring(xml, encoding="unicode"))
        temporary.replace(out / f"{name}.xml")
    return out


@register_gripper
class SO101Gripper(GripperModel):
    def __init__(self, idn=0):
        super().__init__(str(converted_assets() / "gripper.xml"), idn=idn)

    def format_action(self, action):
        # Continuous absolute aperture: -1=open, +1=closed (robosuite convention).
        return -np.clip(np.asarray(action), -1, 1)

    @property
    def init_qpos(self):
        return np.array([0.0])

    @property
    def _important_geoms(self):
        return {"left_finger": [f"fixed_jaw_box{i}" for i in range(1, 8)] +
                               [f"fixed_jaw_sph_tip{i}" for i in range(1, 4)] + ["fixed_jaw_mesh"],
                "right_finger": [f"moving_jaw_box{i}" for i in range(1, 4)] +
                                [f"moving_jaw_sph_tip{i}" for i in range(1, 4)] + ["moving_jaw_mesh0", "moving_jaw_mesh1"],
                "left_fingerpad": ["fixed_jaw_box4", "fixed_jaw_box5", "fixed_jaw_box6"],
                "right_fingerpad": ["moving_jaw_box2", "moving_jaw_box3"]}


@register_robot_class("FixedBaseRobot")
class SO101(ManipulatorModel):
    arms = ["right"]

    def __init__(self, idn=0):
        super().__init__(str(converted_assets() / "robot.xml"), idn=idn)

    @property
    def default_base(self):
        return "NullMount"

    @property
    def default_gripper(self):
        return {"right": "SO101Gripper"}

    @property
    def default_controller_config(self):
        return {"right": "joint_position"}

    @property
    def init_qpos(self):
        return HOME.copy()

    @property
    def base_xpos_offset(self):
        return {"table": lambda length: (-0.25, 0, 0.8), "empty": (-0.25, 0, 0.8), "bins": (-0.25, 0, 0.8)}

    @property
    def top_offset(self):
        return np.array([0, 0, 0.45])

    @property
    def _horizontal_radius(self):
        return 0.4

    @property
    def arm_type(self):
        return "single"


def controller_config():
    # Lower proportional gain for explicit torque control; original derivative gain and torque cap.
    # MuJoCo position servos handle stiffness implicitly, whereas robosuite applies external torques.
    return {"type": "BASIC", "body_parts": {"right": {
        "type": "JOINT_POSITION", "input_type": "absolute",
        "input_min": [-1.91986, -1.7453293, -1.69, -1.658063, -2.7438473],
        "input_max": [1.91986, 1.7453293, 1.69, 1.658063, 2.8412063],
        "kp": 200.0, "kd": 2.731, "use_torque_compensation": False,
        "interpolation": None, "gripper": {"type": "GRIP"}}}}
