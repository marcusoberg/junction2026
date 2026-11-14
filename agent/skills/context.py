"""SkillContext: everything a skill needs (config, scene, kinematics, planner, camera detector), built once.

Nothing here edits the organisers' code; it imports their classical baseline and the oracle's planner and wraps
them. Needs `aaltonomous-humanoid/` on PYTHONPATH (and its third_party URDF, as the baseline does).

    ctx = SkillContext()                       # sim-tuned profile of baselines/classical/config.yaml
    with ctx.tuned(**{"motion.arm_max_vel": 0.6, "grasp.approach_height": 0.10}):
        ...                                    # skills planned in here use these values, then they are restored
"""
from __future__ import annotations

import copy
import threading
from contextlib import contextmanager

from aaltonomous.baselines.classical.config import hand_postures, home_arm, load_config
from aaltonomous.baselines.classical.kinematics import ArmKinematics
from aaltonomous.baselines.classical.perception import Perception
from aaltonomous.interface.calibration import Calibration
from aaltonomous.interface.safety import JointSpec, SafetyConfig
from aaltonomous.scene.scene import Scene

HZ = 30.0
SIDES = ("left", "right")


def _get(cfg, dotted):
    d = cfg
    for k in dotted.split("."):
        d = d[k]
    return d


def _set(cfg, dotted, value):
    keys = dotted.split(".")
    d = cfg
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


class SkillContext:
    def __init__(self, profile="sim", overrides=None, scene=None, calibration=None, safety: SafetyConfig | None = None):
        from .planner import SkillPlanner       # here: planner.py imports this module
        self.cfg = load_config(None, overrides, profile)
        self.scene = scene if isinstance(scene, Scene) else Scene.load(scene)
        self.cal = calibration if isinstance(calibration, Calibration) else Calibration.load(calibration)
        self.spec = JointSpec()
        self.post = hand_postures(self.cfg, self.spec)          # {"open" | "preshape" | "closed" | "release": (14,)}
        self.home = home_arm(self.cfg, self.spec)               # (14,) home pose of both arms
        self.kin = ArmKinematics(self.cfg, self.spec)
        self.safety = safety or SafetyConfig(dt=1.0 / HZ)
        self.planner = SkillPlanner(self.cfg, self.scene, self.post, self.home, self.kin, self.safety, self.spec)
        self.camera = Perception(self.cfg, self.scene, self.cal)
        self.dt = 1.0 / HZ
        self.lock = threading.RLock()            # Pinocchio data and the planner are not thread-safe

    @contextmanager
    def tuned(self, **dotted):
        """Temporarily change config values by dotted key (e.g. "ik.max_tilt_deg"); None values are ignored."""
        missing = object()
        saved = {}
        with self.lock:
            try:
                for k, v in dotted.items():
                    if v is None:
                        continue
                    try:
                        saved[k] = copy.deepcopy(_get(self.cfg, k))
                    except KeyError:
                        saved[k] = missing
                    _set(self.cfg, k, v)
                yield self
            finally:
                for k, v in saved.items():
                    if v is missing:
                        keys = k.split(".")
                        _get(self.cfg, ".".join(keys[:-1])).pop(keys[-1], None) if len(keys) > 1 else self.cfg.pop(k, None)
                    else:
                        _set(self.cfg, k, v)

    def bins(self):
        return list(self.scene.bin_names)

    def side_of(self, arm):
        if arm not in SIDES:
            raise ValueError(f"arm must be 'left' or 'right', got {arm!r}")
        return arm
