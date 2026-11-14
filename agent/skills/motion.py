"""Motion skills. Each one PLANS and CHECKS a trajectory and returns it; nothing here moves the robot.

    traj = pick_and_place(ctx, obj, "left", arm_q, hand_q)          # the oracle's whole move, with variations
    traj = pick(ctx, obj, arm_q, hand_q, arm="left")                 # ends holding the object at carry height
    traj = place(ctx, "center", traj.end_arm, traj.end_hand, arm="left")
    traj = place_on_counter(ctx, (0.30, 0.0), ...)                   # put down where the other arm can take it
    traj = go_home(ctx, arm_q, hand_q)
    traj = set_hand(ctx, arm_q, hand_q, arm="left", posture="open")

`arm_q` / `hand_q` are the 14-vectors the robot was last COMMANDED to (not the measured ones): the safety filter
limits the step from the last command, so plans must start there.

A returned Trajectory with ok=True has passed: inverse kinematics within tolerance, hand clearance above the counter
and the bin rims, and a dry run through the organisers' SafetyFilter with zero clips. ok=False carries `reason`.

Approach: how to vary a pick without touching the planner code. Unset fields keep the config's value.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from aaltonomous.baselines.classical.planner import Plan

from .perception import WorldObject


@dataclass
class Approach:
    finger_direction_deg: float | None = None   # where the fingers point, pelvis frame: 0 forward, +90 to the left.
                                                #   Ignored for cubes / triangles with a known yaw (face-aligned grasp)
    max_tilt_deg: float | None = None           # how far the palm may tilt from facing straight down (config: 35)
    approach_height: float | None = None        # m above the grasp height where the descent starts (config: 0.08)
    pregrasp_offset: tuple | None = None        # (dx, dy) m: start the descent off to one side (slanted approach)
    speed: float | None = None                  # rad/s arm joint speed (config: 0.97, filter limit 1.0)
    acc: float | None = None                    # rad/s^2 (config: 8.0)
    grasp_height_frac: float | None = None      # grasp height as a fraction of the object height (config: 0.5)
    carry_height: float | None = None           # m above the counter while carrying (config: 0.16)
    xy_offset: tuple | None = None              # (dx, dy) m added to the object position (deliberate offset / jitter)

    def overrides(self):
        o = {"ik.max_tilt_deg": self.max_tilt_deg, "grasp.approach_height": self.approach_height,
             "grasp.pregrasp_offset": None if self.pregrasp_offset is None else [float(v) for v in self.pregrasp_offset],
             "motion.arm_max_vel": None if self.speed is None else float(min(self.speed, 0.98)),
             "motion.arm_max_acc": self.acc, "grasp.height_frac": self.grasp_height_frac,
             "grasp.carry_height": self.carry_height}
        if self.finger_direction_deg is not None:
            a = np.radians(self.finger_direction_deg)
            o["grasp.finger_hints"] = [[float(np.cos(a)), float(np.sin(a)), 0.0]]
        return o

    @classmethod
    def from_json(cls, d):
        d = dict(d or {})
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d and d[k] is not None})


@dataclass
class Trajectory:
    ok: bool
    skill: str
    args: dict = field(default_factory=dict)
    arm: np.ndarray | None = None               # (T, 14) absolute arm joint targets, one row per 30 Hz tick
    hand: np.ndarray | None = None              # (T, 14)
    phases: list = field(default_factory=list)  # (name, first tick, end tick)
    reason: str = ""
    side: str | None = None

    @property
    def ticks(self):
        return 0 if self.arm is None else len(self.arm)

    @property
    def end_arm(self):
        return self.arm[-1].copy()

    @property
    def end_hand(self):
        return self.hand[-1].copy()

    def then(self, other):
        """This trajectory followed by another (planned from this one's end state)."""
        if not (self.ok and other.ok):
            return self if not self.ok else other
        ph = self.phases + [(n, s + self.ticks, e + self.ticks) for n, s, e in other.phases]
        return Trajectory(True, f"{self.skill}+{other.skill}", {"first": self.args, "then": other.args},
                          np.vstack([self.arm, other.arm]), np.vstack([self.hand, other.hand]), ph, side=other.side)

    def to_json(self):
        return {"ok": self.ok, "skill": self.skill, "args": self.args, "arm": self.side, "reason": self.reason,
                "seconds": round(self.ticks / 30.0, 1), "phases": [p[0] for p in self.phases]}


def _from_plan(plan: Plan, skill, args):
    if not plan.ok:
        return Trajectory(False, skill, args, reason=plan.reason or "no plan", side=plan.side)
    return Trajectory(True, skill, args, plan.arm, plan.hand, [(str(n), int(s), int(e)) for n, s, e in plan.phases],
                      reason=plan.reason or "", side=plan.side)


def _det(obj, approach):
    d = obj.detection()
    if approach is not None and approach.xy_offset is not None:
        d.top = d.top + np.r_[np.asarray(approach.xy_offset, float), 0.0]
    if approach is not None and approach.finger_direction_deg is not None:
        d.shape = None                          # an explicit finger direction replaces the face-aligned grasp
    return d


def _mirror_hints(ctx, side, ov):
    """The planner mirrors finger hints in y for the right arm; an absolute direction must be pre-mirrored."""
    if side == "right" and "grasp.finger_hints" in ov and ov["grasp.finger_hints"] is not None:
        ov["grasp.finger_hints"] = [[h[0], -h[1], h[2]] for h in ov["grasp.finger_hints"]]
    return ov


def _sides(ctx, obj, bin_name, arm):
    if arm:
        return [arm]
    by = ctx.planner.bin_xy(bin_name)[1] if bin_name in ctx.scene.cfg["bins"]["centers"] else 0.0
    ref = by if abs(by) > 0.1 else float(obj.xyz[1])
    first = "left" if ref > ctx.cfg["grasp"]["y_split"] else "right"
    return [first, "right" if first == "left" else "left"]


def _try(ctx, obj, bin_name, arm_q, hand_q, arm, approach, skill, args, side_bin=None, **kw):
    approach = approach or Approach()
    last = Trajectory(False, skill, args, reason="no arm tried")
    reasons = []
    for side in _sides(ctx, obj, side_bin or bin_name, arm):
        with ctx.tuned(**_mirror_hints(ctx, side, approach.overrides())):
            plan = ctx.planner._plan_side_hints(_det(obj, approach), side, bin_name, np.asarray(arm_q, float),
                                                np.asarray(hand_q, float), **kw)
        last = _from_plan(plan, skill, {**args, "arm": side})
        if last.ok:
            return last
        reasons.append(f"{side}: {last.reason}")
    last.reason = "; ".join(reasons)
    return last


# ------------------------------------------------------------------------------------------------ skills
def pick_and_place(ctx, obj: WorldObject, bin_name, arm_q, hand_q, arm=None, approach: Approach | None = None,
                   drop_offset=(0.0, 0.0)):
    """Pick `obj` from the counter, drop it in bin `bin_name`, return the arm home. arm=None: the arm on the bin's
    side first (centre bin: the object's side), then the other. drop_offset: (dx, dy) m inside the bin."""
    args = {"object": obj.id, "color": obj.color, "shape": obj.shape, "bin": bin_name,
            "approach": {k: v for k, v in asdict(approach or Approach()).items() if v is not None},
            "drop_offset": [float(v) for v in drop_offset]}
    with ctx.lock:
        ctx.planner.drop_offsets = {bin_name: tuple(drop_offset)}
        try:
            return _try(ctx, obj, bin_name, arm_q, hand_q, arm, approach, "pick_and_place", args)
        finally:
            ctx.planner.drop_offsets = {}


def pick(ctx, obj: WorldObject, arm_q, hand_q, arm=None, approach: Approach | None = None, for_bin=None):
    """Grasp `obj` and lift it to carry height; ends holding it. Follow with place() or place_on_counter().
    for_bin: the bin it is meant for, if known, so the arm that can reach that bin is tried first."""
    args = {"object": obj.id, "color": obj.color, "shape": obj.shape,
            "approach": {k: v for k, v in asdict(approach or Approach()).items() if v is not None}}
    with ctx.lock:
        return _try(ctx, obj, None, arm_q, hand_q, arm, approach, "pick", args, side_bin=for_bin, stages=("pick",))


def _held(ctx, arm, arm_q):
    """A stand-in detection at the hand's current position (the place stages only need the arm)."""
    p = ctx.kin.grasp_pose(arm, np.asarray(arm_q, float))[0]
    return WorldObject("held", "held", p, source="held")


def place(ctx, bin_name, arm_q, hand_q, arm, drop_offset=(0.0, 0.0), home=True, approach: Approach | None = None):
    """From holding an object: carry it over bin `bin_name`, lower, release, retreat (and go home)."""
    args = {"bin": bin_name, "drop_offset": [float(v) for v in drop_offset], "home": bool(home)}
    with ctx.lock:
        ctx.planner.drop_offsets = {bin_name: tuple(drop_offset)}
        try:
            return _try(ctx, _held(ctx, arm, arm_q), bin_name, arm_q, hand_q, arm, approach, "place", args,
                        stages=("place", "home") if home else ("place",))
        finally:
            ctx.planner.drop_offsets = {}


def place_on_counter(ctx, xy, arm_q, hand_q, arm, home=True, approach: Approach | None = None):
    """From holding an object: set it down on the counter at pelvis (x, y), e.g. where the other arm can reach it."""
    args = {"xy": [round(float(v), 3) for v in xy], "home": bool(home)}
    with ctx.lock:
        return _try(ctx, _held(ctx, arm, arm_q), None, arm_q, hand_q, arm, approach, "place_on_counter", args,
                    place_xy=np.asarray(xy, float), stages=("place", "home") if home else ("place",))


def go_home(ctx, arm_q, hand_q, arms="both", open_hands=True):
    """Joint-space move of one or both arms to the home pose (hands to `open`)."""
    arm_q, hand_q = np.asarray(arm_q, float), np.asarray(hand_q, float)
    to_a, to_h = arm_q.copy(), hand_q.copy()
    for side in (("left", "right") if arms == "both" else (arms,)):
        sl = ctx.kin.sl(side)
        to_a[sl] = ctx.home[sl]
        if open_hands:
            to_h[sl] = ctx.post["open"][sl]
    with ctx.lock:
        a, h = ctx.planner.go_to(arm_q, hand_q, to_a, to_h)
        if not len(a):
            a, h = arm_q[None].copy(), hand_q[None].copy()
        clips = ctx.planner.dry_run_safety(arm_q, hand_q, a, h)
    args = {"arms": arms}
    if clips:
        return Trajectory(False, "go_home", args, reason=f"{clips} safety clips in dry run")
    return Trajectory(True, "go_home", args, a, h, [("home", 0, len(a))])


def set_hand(ctx, arm_q, hand_q, arm, posture="open", hold_s=0.3):
    """Move one hand to a named posture (open | preshape | closed | release) with the arm still."""
    from aaltonomous.baselines.classical.planner import ramp
    if posture not in ctx.post:
        return Trajectory(False, "set_hand", {"arm": arm, "posture": posture}, reason=f"unknown posture {posture!r}")
    arm_q, hand_q = np.asarray(arm_q, float), np.asarray(hand_q, float)
    to = hand_q.copy()
    sl = ctx.kin.sl(arm)
    to[sl] = ctx.post[posture][sl]
    h = ramp(hand_q, to, ctx.cfg["motion"]["hand_max_vel"], ctx.dt)
    h = np.vstack([h, np.repeat(to[None], max(1, int(round(hold_s / ctx.dt))), 0)])
    a = np.repeat(arm_q[None], len(h), 0)
    with ctx.lock:
        clips = ctx.planner.dry_run_safety(arm_q, hand_q, a, h)
    args = {"arm": arm, "posture": posture}
    if clips:
        return Trajectory(False, "set_hand", args, reason=f"{clips} safety clips in dry run")
    return Trajectory(True, "set_hand", args, a, h, [(posture, 0, len(h))], side=arm)


def hold(arm_q, hand_q, seconds=0.3):
    n = max(1, int(round(seconds * 30)))
    return Trajectory(True, "hold", {"seconds": seconds}, np.repeat(np.asarray(arm_q, float)[None], n, 0),
                      np.repeat(np.asarray(hand_q, float)[None], n, 0), [("hold", 0, n)])


def handover_spots(ctx, obj: WorldObject, others=()):
    """Counter spots near the centre line, free of other objects, where an object can be set down for the other arm."""
    from aaltonomous.sim.oracle import RELAY_CLEAR, RELAY_POINTS
    out = []
    for xy in RELAY_POINTS:
        xy = np.asarray(xy, float)
        if np.linalg.norm(xy - obj.xyz[:2]) < 0.04:
            continue
        if any(np.linalg.norm(xy - np.asarray(o.xyz[:2], float)) < RELAY_CLEAR for o in others if o.id != obj.id):
            continue
        out.append(tuple(float(v) for v in xy))
    return out


def relocate(ctx, obj: WorldObject, bin_name, arm_q, hand_q, others=(), approach: Approach | None = None):
    """For an object no single arm can take all the way to `bin_name`: pick it with the arm that reaches it and set
    it down at a spot from which the other arm has a complete plan to the bin (checked from the home pose)."""
    args = {"object": obj.id, "color": obj.color, "shape": obj.shape, "bin": bin_name}
    last = Trajectory(False, "relocate", args, reason="no hand-over spot is free")
    home_h = ctx.post["open"].copy()
    for arm in _sides(ctx, obj, None, None):
        other = "right" if arm == "left" else "left"
        pk = pick(ctx, obj, arm_q, hand_q, arm=arm, approach=approach)
        if not pk.ok:
            last = Trajectory(False, "relocate", args, reason=f"{arm} cannot pick it: {pk.reason}")
            continue
        for spot in handover_spots(ctx, obj, others)[:3]:
            there = WorldObject(obj.id, obj.color, np.r_[spot, obj.xyz[2]], obj.shape, None, source=obj.source)
            fin = pick_and_place(ctx, there, bin_name, ctx.home.copy(), home_h, arm=other)
            if not fin.ok:
                last = Trajectory(False, "relocate", args, reason=f"{other} has no plan from {spot}: {fin.reason}")
                continue
            pl = place_on_counter(ctx, spot, pk.end_arm, pk.end_hand, arm=arm)
            if pl.ok:
                out = pk.then(pl)
                out.skill, out.args = "relocate", {**args, "arm": arm, "spot": list(spot), "then_arm": other}
                return out
            last = Trajectory(False, "relocate", args, reason=f"{arm} cannot set it down at {spot}: {pl.reason}")
    return last
