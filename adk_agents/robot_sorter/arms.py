"""Which arm should act on an object. Kept apart from the tools so the rule can be read and changed in one place.

    order(ctx, xyz, bin_name, arm_q)   arms to try, best first: used by pick_and_place, pick and hand_over when the
                                       model names no arm
    distances / reaches                the facts behind it; the choose_arm tool reports them to the model

The rule: an arm that can put its hand over the target bin comes before one that cannot (the left bin is only
within reach of the left arm, the right bin of the right arm); among those, the arm whose shoulder is nearer the
object goes first. These are exact measurements (forward and inverse kinematics on the robot model), so no model
call is needed for them; what cannot be measured this way (what the object is, whether a grasp held) is what the
Gemini models are asked.
"""
from __future__ import annotations

import numpy as np

SIDES = ("left", "right")


def distances(ctx, xyz, arm_q):
    """Distance (m) from each shoulder to the point xyz (pelvis frame)."""
    with ctx.lock:
        ctx.kin._fk(np.asarray(arm_q, float))
        out = {}
        for side in SIDES:
            p = ctx.kin.data.oMf[ctx.kin.model.getFrameId(f"{side}_shoulder_yaw_link")].translation
            out[side] = float(np.linalg.norm(np.asarray(xyz, float) - p))
    return out


def reaches(ctx, side, bin_name):
    """Can this arm hold its hand above the bin at carry height? (cached inverse-kinematics check)"""
    if not bin_name or bin_name not in ctx.scene.cfg["bins"]["centers"]:
        return True
    with ctx.lock:
        return bool(ctx.planner._reaches_bin(side, bin_name))


def order(ctx, xyz, bin_name, arm_q):
    d = distances(ctx, xyz, arm_q)
    return sorted(SIDES, key=lambda s: (not reaches(ctx, s, bin_name), d[s]))
