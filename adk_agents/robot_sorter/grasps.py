"""Grasp options: every way the hand can take one object, in the order to try them, and a memory of what missed.

One rule for every object instead of a special case per shape:
    options(obj, arms)   -> [Option(arm, kind, angle)]   what could be tried, best first
    robot.failed[id]     -> the options that already missed on that object; they are never tried again
so a retry always changes something (the grip, the hand's direction, or the arm), and when nothing is left the tool
says so instead of repeating itself. In our teammates' comparison of this agent, 17 of 31 missed picks were tried
again straight away and 6 of those with identical arguments; their rule-based policy gained most from not doing that.

Kinds of grip (all are the organisers' planner, with different settings):
    faces   fingers on a flat face of a standing cube or triangular prism. For prisms the angle between the fingers
            and the apex is tried at 105, 90 and 75 deg (the organisers' own sweep: 105 best, 90 next).
    ridge   a block lying on its side: pinch across its two end faces, fingers along the ridge (our teammates' v1.5:
            held 4 of 9 lying prisms, the plain pinch 0 of 14).
    plain   the planner's default pinch from above; with an angle, the fingers are turned that way first.
            Never offered for a standing triangular prism (it held 1 of 15) unless the model asks for a direction.
"""
from __future__ import annotations

import dataclasses
from contextlib import contextmanager

import numpy as np

TRIANGLE_GRIPS_DEG = (105.0, 90.0, 75.0)
PLAIN_TURNS_DEG = (None, 40.0, -40.0)        # the hand's natural direction, then turned either way
# The right arm's plain pinch on a cylinder within 10 cm of the midline held 21 of 52 (40 %), the same pinch turned
# +40 deg 11 of 14 (79 %); on the midline cylinder of layout 1001 the plain pinch missed first time in 11 of 11
# episodes (all saved sim runs, 15 Nov). There the turned pinch goes first.
MIDLINE_M = 0.10
RIGHT_MIDLINE_TURNS_DEG = (40.0, None, -40.0)
FINGER_REF_DEG = {"left": -25.0, "right": 25.0}     # the hand's natural finger direction (as in the oracle)
RIDGE_MAX_DEV_DEG = 95.0


@dataclasses.dataclass(frozen=True)
class Option:
    arm: str
    kind: str                      # faces | ridge | plain
    angle: float | None = None     # faces on a prism: fingers-to-apex angle; plain: turn from the natural direction
    variation: tuple = ()          # what the model changed (offsets, tilt, height): a different variation is a new option

    def text(self):
        what = {"faces": "flat faces", "ridge": "across the ends of the lying block", "plain": "plain pinch",
                "local": "local IK pinch"}[self.kind]
        if self.kind == "faces" and self.angle is not None:
            what += f" at {self.angle:.0f} deg"
        if self.kind in ("plain", "local") and self.angle is not None:
            what += f", hand turned {self.angle:+.0f} deg"
        return f"{self.arm} arm, {what}" + (f", with {dict(self.variation)}" if self.variation else "")


def options(obj, arms, variation=(), direction=None):
    """obj: WorldObject with the measured shape / yaw and extra["ridge_yaw"] for a lying block.
    direction: a finger direction the model asked for (then only the plain pinch in that direction is offered)."""
    out = []
    for arm in arms:
        if direction is not None:
            kinds = [("plain", float(direction) - FINGER_REF_DEG[arm])]
        elif obj.extra.get("ridge_yaw") is not None:
            kinds = [("ridge", None), ("plain", None)]
        elif obj.shape == "triangle" and obj.yaw is not None:
            kinds = [("faces", g) for g in TRIANGLE_GRIPS_DEG]
        elif obj.shape == "cube" and obj.yaw is not None:
            kinds = [("faces", None)] + [("plain", t) for t in PLAIN_TURNS_DEG]
        else:
            near = arm == "right" and abs(float(obj.xyz[1])) < MIDLINE_M
            kinds = [("plain", t) for t in (RIGHT_MIDLINE_TURNS_DEG if near else PLAIN_TURNS_DEG)]
        out += [Option(arm, k, a, tuple(variation)) for k, a in kinds]
    return out


def target_for(obj, opt):
    """The object as the planner should see it for this option."""
    if opt.kind == "faces":
        return obj
    if opt.kind == "ridge":
        return dataclasses.replace(obj, shape="tipped", yaw=float(obj.extra["ridge_yaw"]))
    return dataclasses.replace(obj, shape=None, yaw=None)


def approach_for(approach, opt):
    from agent.skills import Approach
    approach = approach or Approach()
    if opt.kind == "plain" and opt.angle is not None:
        return dataclasses.replace(approach, finger_direction_deg=FINGER_REF_DEG[opt.arm] + opt.angle)
    return approach


@contextmanager
def planner_settings(opt):
    """The oracle planner's module settings for this option, put back afterwards (hold ctx.lock around it)."""
    import aaltonomous.sim.oracle as oracle
    saved = (dict(oracle.FACE_ALIGN), oracle.FACE_MAX_DEV_DEG)
    try:
        if opt.kind == "faces" and opt.angle is not None:
            oracle.FACE_ALIGN["triangle"] = (np.pi / 2 + np.radians(opt.angle), 3)
        if opt.kind == "ridge":
            oracle.FACE_ALIGN["tipped"] = (0.0, 2)             # the two end faces: along the ridge, either way
            oracle.FACE_MAX_DEV_DEG = RIDGE_MAX_DEV_DEG
        yield
    finally:
        oracle.FACE_ALIGN.clear()
        oracle.FACE_ALIGN.update(saved[0])
        oracle.FACE_MAX_DEV_DEG = saved[1]


def accepted(opt, traj):
    """A plan counts for this option only if it is the grip the option names (the planner falls back silently)."""
    return bool(traj.ok and (opt.kind == "plain" or traj.reason.startswith("face-aligned")))
