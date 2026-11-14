"""SkillPlanner: the oracle's planner with its one monolithic pick-and-place cut into stages.

The organisers' `Planner._plan_side` always plans  to_pregrasp, descend, close, lift, carry, open, retreat, home
in one go. Here the same code (same waypoints, same IK, same clearance and safety checks) takes
`stages=("pick", "place", "home")`, so a skill can plan only the pick (ends holding the object at carry height),
only the place (starts from wherever the hand is), or only the return home. With all three stages the result is
identical to the organisers' planner (agent/tests/test_skills.py checks that).

Extra knob, read from the config so SkillContext.tuned() can vary it per call:
    grasp.pregrasp_offset: [dx, dy]   pregrasp point shifted sideways -> the hand comes down along a slanted line
Everything inherited from the oracle still applies: face-aligned grasps for cubes and triangles when the
detection has a shape and yaw, joint-space carry, drop offsets inside a bin (`drop_offsets`).
"""
from __future__ import annotations

import numpy as np

from aaltonomous.baselines.classical.kinematics import ArmKinematics
from aaltonomous.baselines.classical.planner import Plan, ramp, time_scale
from aaltonomous.sim.oracle import _OraclePlanner

ALL = ("pick", "place", "home")


class SkillPlanner(_OraclePlanner):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.drop_offsets = {}

    def _plan_side(self, det, side, bin_name, arm_q, hand_q, place_xy=None, stages=ALL):
        cfg, g, mo = self.cfg, self.cfg["grasp"], self.cfg["motion"]
        sc, zc = self.scene, self.scene.counter_z
        plan = Plan(False, side=side, color=det.color, target_bin=bin_name if place_xy is None else "counter")
        sl = ArmKinematics.sl(side)
        H, post = self.height, self.post
        arm_q, hand_q = np.asarray(arm_q, float), np.asarray(hand_q, float)
        hand_pre = hand_q.copy(); hand_pre[sl] = post["preshape"][sl]
        hand_cls = hand_q.copy(); hand_cls[sl] = post["closed"][sl]
        hand_open = hand_q.copy(); hand_open[sl] = post["open"][sl]
        hand_rel = hand_q.copy(); hand_rel[sl] = post[g.get("release_posture", "preshape")][sl]
        do_pick, do_place, do_home = "pick" in stages, "place" in stages, "home" in stages

        wall = float(sc.cfg["bins"]["wall_height"])
        down = up = carry = lower = retreat = None
        q_pre = None
        z = zc + g["height_frac"] * H
        z_carry = zc + max(g["carry_height"], wall + H + 0.03)
        if do_pick:
            # grasp height: object mid-height, raised until the preshaped fingers clear the counter
            xy = det.top[:2] + np.asarray((g.get("arm_xy_offset") or {}).get(side, [0.0, 0.0]), float)
            for _ in range(4):
                qg, ok = self._ik(side, np.r_[xy, z], arm_q, "grasp", plan)
                if not ok:
                    plan.reason = f"IK grasp {side}"
                    return plan
                c = self.clearance(side, qg, hand_pre, fingers_only=True)
                if c >= g["fingertip_clearance"] - 1e-3:
                    break
                z += g["fingertip_clearance"] - c
            else:
                plan.reason = "fingers hit the counter"
                return plan
            if z > zc + H - 0.008:
                plan.reason = f"grasp point {100 * (z - zc):.1f} cm above counter: above the object top (point_in_palm?)"
                return plan
            off = np.asarray(g.get("pregrasp_offset") or [0.0, 0.0], float)
            p_pre = np.r_[xy + off, max(z + g["approach_height"], z_carry)]
            p_grasp, p_lift = np.r_[xy, z], np.r_[xy, z_carry]
            q_pre, ok = self._ik(side, p_pre, qg, "pregrasp", plan)       # same IK branch as the grasp
            if not ok:
                plan.reason = f"IK pregrasp {side}"
                return plan
            plan.waypoints["grasp"] = p_grasp
            down = self._cartesian(side, p_pre, p_grasp, q_pre, "descend", plan)
            up = down and self._cartesian(side, p_grasp, p_lift, down[-1], "lift", plan)
            if not up:
                plan.reason = plan.reason or f"IK failed on {[k for k, v in plan.ik.items() if not v[0]]} ({side})"
                return plan
            q_held, p_held = up[-1], p_lift
        else:
            q_held, p_held = arm_q, self.kin.grasp_pose(side, arm_q)[0]

        if do_place:
            z_drop = zc + wall + H / 2 + g["drop_above_rim"]
            if place_xy is not None:      # put it down on the counter (release just above the grasp height)
                bxy, z_drop = np.asarray(place_xy, float), z + 0.005
            else:
                bxy = self.drop_xy(bin_name)
            p_above, p_drop = np.r_[bxy, z_carry], np.r_[bxy, z_drop]
            p_retreat = np.r_[bxy, z_drop + g["retreat_up"]]
            carry = (self._cartesian(side, p_held, p_above, q_held, "carry", plan)
                     or self._joint_carry(side, q_held, p_held, p_above, z_drop, "carry", plan))
            lower = carry and self._cartesian(side, p_above, p_drop, carry[-1], "lower", plan)
            retreat = lower and self._cartesian(side, p_drop, p_retreat, lower[-1], "retreat", plan)
            if not retreat:
                plan.reason = plan.reason or f"IK failed on {[k for k, v in plan.ik.items() if not v[0]]} ({side})"
                return plan
            plan.waypoints["drop"] = p_drop

        vmax, amax, hv = mo["arm_max_vel"], mo["arm_max_acc"], mo["hand_max_vel"]
        home = arm_q.copy(); home[sl] = self.home[sl]
        arm_parts, hand_parts, phases = [], [], []
        cur_a, cur_h = arm_q.copy(), hand_q.copy()

        def add(name, a, h):
            nonlocal cur_a
            if len(a):
                start = sum(len(x) for x in arm_parts)
                arm_parts.append(a); hand_parts.append(h)
                phases.append((name, start, start + len(a)))
                cur_a = a[-1].copy()

        def move(name, Qs, hand_to=None):
            nonlocal cur_h
            a = time_scale([cur_a] + list(Qs), vmax, amax, self.dt)
            h = np.repeat(cur_h[None], len(a), 0)
            if hand_to is not None:       # hand moves during the arm move (ends no later than the arm)
                hr = ramp(cur_h, hand_to, hv, self.dt)
                if len(hr) > len(a):
                    a = np.vstack([a, np.repeat((a[-1] if len(a) else cur_a)[None], len(hr) - len(a), 0)])
                    h = np.repeat(cur_h[None], len(a), 0)
                h[:len(hr)] = hr
                h[len(hr):] = hand_to
                cur_h = np.asarray(hand_to, float).copy()
            add(name, a, h)

        def hand(name, to, hold_s):
            nonlocal cur_h
            h = ramp(cur_h, to, hv, self.dt)
            h = np.vstack([h, np.repeat(np.asarray(to, float)[None], int(round(hold_s / self.dt)), 0)])
            cur_h = np.asarray(to, float).copy()
            add(name, np.repeat(cur_a[None], len(h), 0), h)

        if do_pick:
            move("to_pregrasp", [q_pre], hand_to=hand_pre)
            move("descend", down)
            hand("close", hand_cls, mo["close_hold_s"])
            move("lift", up)
        if do_place:
            move("carry", carry + lower)
            hand("open", hand_rel, mo["open_hold_s"])
            move("retreat", retreat)
        if do_home:
            move("home", [home], hand_to=hand_open if (do_place or not do_pick) else None)
        if not arm_parts:
            plan.reason = "nothing to do"
            return plan
        plan.arm, plan.hand, plan.phases = np.vstack(arm_parts), np.vstack(hand_parts), phases

        # clearance over the whole trajectory (grasp phases use the fingertip margin)
        grasp_ticks = np.zeros(len(plan.arm), bool)
        near = ("descend", "close", "lift") + (("carry", "open", "retreat") if place_xy is not None else ())
        closing = np.zeros(len(plan.arm), bool)
        drop = np.zeros(len(plan.arm), bool)
        for name, s, e in phases:
            if name in near:
                grasp_ticks[s:e] = True
            closing[s:e] |= name == "close" or (place_xy is not None and name == "open")
            drop[s:e] |= name == "open"
        for k in range(0, len(plan.arm), 3):
            need = 0.003 if closing[k] else g["fingertip_clearance"] if grasp_ticks[k] else mo["counter_clearance"]
            c = self.clearance(side, plan.arm[k], plan.hand[k])
            if c < need - 2e-3:
                name = next(n for n, s, e in phases if s <= k < e)
                plan.reason = f"hand {100 * c:.1f} cm above counter during {name} (need {100 * need:.1f})"
                return plan
        for k in range(0, len(plan.arm), 3):
            if drop[k]:
                continue
            c = self.bin_clearance(side, plan.arm[k], plan.hand[k])
            if c < 0.003:
                name = next(n for n, s_, e in phases if s_ <= k < e)
                plan.reason = f"hand {100 * c:.1f} cm above a bin rim during {name}"
                return plan
        plan.safety_clips = self.dry_run_safety(arm_q, hand_q, plan.arm, plan.hand)
        if plan.safety_clips:
            plan.reason = f"{plan.safety_clips} safety clips in dry run"
            return plan
        plan.ok = True
        return plan
