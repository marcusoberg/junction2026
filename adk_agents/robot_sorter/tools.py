"""The agent's tools: the organisers' oracle (aaltonomous/sim/oracle.py), cut into calls a model can make, plus
tools to look through the cameras and read the joint sensors.

Sensing
    look                the oracle's "look": which objects are on the counter and where. The oracle reads exact
                        poses from the simulator, which a submission may not; this uses only the head camera: any
                        saturated colour (not only the sim's red / yellow / blue) + depth for the 3D position.
    view_camera         any camera at full resolution, optionally zoomed in or with the object ids drawn on it.
    read_proprioception joint angles and speeds, where each hand is, how far it is closed, tracking errors.
Acting (each plans with the oracle's planner; see agent/skills)
    pick_and_place      the oracle's single move: pregrasp, descend, close, lift, carry, release, retreat, home.
                        With shape + yaw_deg it uses the oracle's face-aligned grasp for cubes and triangles.
    pick / place        the same move cut in two, so the model can check the hand in between.
    place_on_counter    the oracle's relay put-down, at a point the model chooses.
    hand_over           the oracle's relay: set an object down near the centre line for the other arm.
    go_home, wait, done

Every motion must pass inverse kinematics, hand clearances over the counter and the bin rims, and a zero-clip dry
run through the organisers' safety filter. A call with no safe plan moves nothing and returns status "failed"
with the planner's reason, so the model can try something else.

The tools are async: planning runs in a worker thread and the motion is executed by the 30 Hz loop (robot.py),
so nothing here blocks the control loop or the ADK event loop. A tool returns when its motion has finished.
If the model asks for several tools in one turn they run one after the other, in the order asked.
"""
from __future__ import annotations

import asyncio
import functools
import os
import weakref
from typing import Literal, Optional

import numpy as np

from .robot import HZ, get_robot

Bin = Literal["left", "center", "right"]
Arm = Literal["left", "right"]
Shape = Literal["cube", "cylinder", "triangle"]
Camera = Literal["head", "left_wrist", "right_wrist", "head_depth"]

SAME_OBJECT_M = 0.04          # a detection this close to a known object is that object
# Grasp check from the joint sensors, right after the lift. Fingers that hold something cannot reach their commanded
# closed position: in the Isaac sim they stay 0.017-0.07 rad short, and 0.000 when the hand closed on nothing
# (a first threshold of 0.025 wrongly dropped held objects in run adk-2-lang).
GRASP_CHECK = os.environ.get("ROBOT_GRASP_CHECK", "1") == "1"
GRASP_ERR_RAD = float(os.environ.get("ROBOT_GRASP_ERR_RAD", "0.010"))


# Two arms in one workspace: a planned move is refused if the moving arm would come this close to the other arm
# (measured between hand, finger, wrist and elbow points; nothing in the organisers' safety filter covers it).
ARM_CLEARANCE_M = float(os.environ.get("ROBOT_ARM_CLEARANCE_M", "0.12"))
_ARM_LINKS = ("wrist_yaw_link", "wrist_roll_link", "elbow_link")


def _arm_points(ctx, side, arm_q, hand_q):
    pts = [np.asarray(p, float) for p in ctx.kin.hand_points(side, arm_q, hand_q)]         # palm + fingers (does FK)
    pts += [ctx.kin.data.oMf[ctx.kin.model.getFrameId(f"{side}_{n}")].translation.copy() for n in _ARM_LINKS]
    return np.array(pts)


def _min_arm_distance(ctx, arm, hand):
    """Smallest distance between the two arms over a trajectory (every 3rd tick)."""
    best = np.inf
    with ctx.lock:
        for k in range(0, len(arm), 3):
            a, b = _arm_points(ctx, "left", arm[k], hand[k]), _arm_points(ctx, "right", arm[k], hand[k])
            best = min(best, float(np.linalg.norm(a[:, None] - b[None], axis=2).min()))
    return best


def _park_plan(robot, ctx, arm_q, hand_q):
    """Joint-space move of every arm that holds nothing to its parked pose, hands open. None if already there."""
    from agent.skills.motion import Trajectory
    if robot.aside is None:
        return None
    to_a, to_h = arm_q.copy(), hand_q.copy()
    for side in ("left", "right"):
        if side != robot.holding:
            sl = ctx.kin.sl(side)
            to_a[sl], to_h[sl] = robot.aside[sl], ctx.post["open"][sl]
    if np.abs(to_a - arm_q).max() < 1e-3 and np.abs(to_h - hand_q).max() < 1e-3:
        return None
    with ctx.lock:
        a, h = ctx.planner.go_to(arm_q, hand_q, to_a, to_h)
        if not len(a) or ctx.planner.dry_run_safety(arm_q, hand_q, a, h):
            return None
    return Trajectory(True, "park", {}, a, h, [("park", 0, len(a))])


async def _park(robot):
    """Arms aside: out of each other's way and out of the head camera's view."""
    start = robot.command_state()
    traj = await asyncio.to_thread(_park_plan, robot, robot.ctx, *start)
    if traj is not None:
        await _run(robot, traj, "arms aside", start)


class ToolError(Exception):
    """A call the model got wrong; the message goes back to it."""


_locks = weakref.WeakKeyDictionary()


def _lock():
    """One tool at a time per event loop: several calls in one model turn run in the order they were asked."""
    loop = asyncio.get_running_loop()
    if loop not in _locks:
        _locks[loop] = asyncio.Lock()
    return _locks[loop]


def _tool(fn):
    """Serialise the tools, and turn errors into a result the model can read instead of ending the run."""
    @functools.wraps(fn)
    async def wrapper(*args, **kwargs):
        async with _lock():
            try:
                return await fn(*args, **kwargs)
            except ToolError as e:
                return _result("failed", reason=str(e))
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001  (a planner bug must not take the episode down)
                get_robot().log(f"[tool] {fn.__name__} raised {e!r}")
                return _result("failed", reason=f"internal error: {e!r}"[:300])
    return wrapper


def _result(status, **extra):
    robot = get_robot()
    out = {"status": status, **extra, "holding": f"an object in the {robot.holding} hand" if robot.holding else "nothing"}
    if robot.time_left() is not None:
        out["time_left_s"] = round(robot.time_left(), 1)
    return out


# ---------------------------------------------------------------------------------------------- perception
def _detect(robot):
    """Objects on the counter -> JSON for the model; remembered in robot.objects so tools can take their ids.
    An object keeps its id from one look to the next (same place), so ids stay valid across moves."""
    from agent.skills.perception import detect_camera, xyz_to_point
    ctx, obs = robot.ctx, robot.obs
    # The ONLY source of object positions: the head camera's colour and depth images in `obs`. Nothing here (or
    # anywhere in this package) asks the simulator where an object is.
    found = detect_camera(ctx, obs, shapes=os.environ.get("ROBOT_SHAPES", "1") == "1")
    found.sort(key=lambda o: -o.extra.get("n_px", 0))                    # one object whose hue straddles two colour
    found = [o for i, o in enumerate(found)                               # names is seen twice: keep the larger blob
             if all(np.linalg.norm(o.xyz[:2] - p.xyz[:2]) > 0.03 for p in found[:i])]
    found.sort(key=lambda o: -o.xyz[1])                                   # left to right
    known = dict(robot.objects)
    for o in found:
        near = [(float(np.linalg.norm(np.asarray(k.xyz[:2]) - o.xyz[:2])), i) for i, k in known.items()]
        d, i = min(near, default=(1.0, None))
        if d < SAME_OBJECT_M:
            o.id = i
            known.pop(i)
        else:
            robot.id_count[o.color] = robot.id_count.get(o.color, 0) + 1
            o.id = f"{o.color}_{robot.id_count[o.color]}"
    robot.objects = {o.id: o for o in found}
    hw = obs["head_rgb"].shape[:2]
    out = []
    for o in found:
        y, x = xyz_to_point(ctx, o.xyz, hw)
        out.append({"id": o.id, "color": o.color, "side": "left" if o.xyz[1] > 0 else "right",
                    "position_m": [round(float(o.xyz[0]), 3), round(float(o.xyz[1]), 3)],
                    "point": [int(round(y)), int(round(x))],
                    "shape_guess": o.shape,
                    "yaw_deg_guess": None if o.yaw is None else round(float(np.degrees(o.yaw)), 1)})
    return out


def _expect(robot, object_id, xy):
    """We moved a known object to xy ourselves: let the next look recognise it there and keep its id."""
    seen = robot.objects.get(object_id)
    if seen is not None:
        seen.xyz = np.r_[np.asarray(xy, float), seen.xyz[2]]


def _target(robot, object_id, point, shape, yaw_deg):
    """What to grasp: an id from the last look, or the point the model indicates in the head image."""
    from agent.skills import WorldObject
    from agent.skills.perception import point_to_xyz
    if object_id:
        seen = robot.objects.get(object_id)
        if seen is None:
            raise ToolError(f"unknown object_id {object_id!r}; objects on the counter at the last look: "
                            f"{sorted(robot.objects)}. Call look again, or give a point.")
        xyz, color, source = np.asarray(seen.xyz, float), seen.color, seen.source
    elif point is not None:
        if len(point) != 2:
            raise ToolError("point must be [y, x], each 0..1000 on the head camera image")
        xyz, _ = point_to_xyz(robot.ctx, robot.obs, point)
        color, source = "object", "pointed"
    else:
        raise ToolError("say which object: object_id (from look) or point ([y, x] on the head camera image)")
    yaw = np.radians(yaw_deg) if yaw_deg is not None and shape in ("cube", "triangle") else None
    return WorldObject(object_id or "pointed", color, xyz, shape if yaw is not None else None, yaw, source=source)


def _approach(finger_direction_deg=None, xy_offset_m=None, speed=None):
    from agent.skills import Approach
    if xy_offset_m is not None and len(xy_offset_m) != 2:
        raise ToolError("xy_offset_m must be [dx, dy] in metres")
    return Approach(finger_direction_deg=finger_direction_deg, speed=speed,
                    xy_offset=None if xy_offset_m is None else tuple(float(v) for v in xy_offset_m))


def _pair(value, name):
    if value is None:
        return None
    if len(value) != 2:
        raise ToolError(f"{name} must be [dx, dy] in metres")
    return (float(value[0]), float(value[1]))


def _drop_point(robot, bin_name, asked):
    """Where in the bin to release: what the model asked for, else the oracle's pattern, so that the next object
    does not land on the previous one (a stacked object pokes above the rim and does not count as in the bin)."""
    if asked is not None:
        return asked
    from aaltonomous.sim.oracle import DROP_OFFSETS
    return DROP_OFFSETS[robot.placed.get(bin_name, 0) % len(DROP_OFFSETS)]


def _hand_summary(robot, side):
    """How far a hand is closed, measured and commanded. A hand that stopped short of its command is blocked by
    something between the fingers; one that reached a full close is empty."""
    ctx = robot.ctx
    sl = ctx.kin.sl(side)
    a, b = ctx.post["open"][sl], ctx.post["closed"][sl]

    def frac(q):
        return round(float(np.clip(np.dot(np.asarray(q, float)[sl] - a, b - a) / max(np.dot(b - a, b - a), 1e-9), 0, 1)), 2)

    measured = np.asarray(robot.obs["hand_q"], float)
    return {"closed_fraction": frac(measured), "commanded_closed_fraction": frac(robot.hand_cmd),
            "tracking_error_rad": round(float(np.abs(measured[sl] - robot.hand_cmd[sl]).max()), 3)}


# ---------------------------------------------------------------------------------------------- running a move
async def _ready(robot):
    if not await asyncio.to_thread(robot.ensure_running):
        raise ToolError("no camera images from the robot yet; call wait and try again")
    if robot.busy:
        raise ToolError("the robot is still moving; call wait")


async def _too_close(robot, traj):
    """Reason to refuse a one-arm move that would bring the arms together, else None."""
    if not traj.side or ARM_CLEARANCE_M <= 0:
        return None
    d = await asyncio.to_thread(_min_arm_distance, robot.ctx, traj.arm, traj.hand)
    if d < ARM_CLEARANCE_M:
        return (f"no safe plan: the {traj.side} arm would pass {d * 100:.0f} cm from the other arm "
                f"(limit {ARM_CLEARANCE_M * 100:.0f} cm). Use the other arm, or hand_over.")
    return None


async def _move(label, plan, on_done=None, look_after=False, **echo):
    """Plan with `plan(ctx, arm_q, hand_q) -> Trajectory` in a worker thread, execute it at 30 Hz, report."""
    robot = get_robot()
    await _ready(robot)
    start = robot.command_state()
    traj = await asyncio.to_thread(plan, robot.ctx, *start)
    if not traj.ok:
        return _result("failed", reason=f"no safe plan: {traj.reason}"[:400], **echo)
    seconds = traj.ticks / HZ
    if robot.time_left() is not None and seconds > robot.time_left():
        return _result("failed", reason=f"not enough time left for a {seconds:.0f} s move", **echo)
    close = await _too_close(robot, traj)
    if close:
        return _result("failed", reason=close, **echo)
    outcome = await _run(robot, traj, label, start, settle=not look_after)
    if outcome != "completed":
        robot.holding = robot.held_id = None
        return _result("aborted", reason=outcome, **echo)
    if on_done is not None:
        on_done(traj)
    out = _result("completed", arm=traj.side, seconds=round(seconds, 1), **echo)
    if look_after:
        await _park(robot)
    if look_after and not robot.holding:          # arms are aside again: tell the model what is on the counter now
        try:
            out["objects_on_counter"] = await asyncio.to_thread(_detect, robot)
        except Exception as e:  # noqa: BLE001
            out["objects_on_counter"] = f"look failed: {e!r}"[:200]
    return out


def _grasp_check(robot, side):
    sl = robot.ctx.kin.sl(side)
    err = float(np.abs(np.asarray(robot.obs["hand_q"], float)[sl] - robot.hand_cmd[sl]).max())
    return {"object_in_hand": bool(err >= GRASP_ERR_RAD), "finger_tracking_error_rad": round(err, 3),
            "threshold_rad": GRASP_ERR_RAD}


async def _run(robot, traj, label, start, settle=True):
    try:
        return await asyncio.wrap_future(robot.start(traj, label, start, settle))
    except RuntimeError as e:
        raise ToolError(str(e)) from None


async def _grasp_move(label, plan, on_done, obj, look_after, **echo):
    """A move that starts with a grasp: run it up to the end of the lift, check with the finger sensors that
    something is in the hand, and only then carry on. On a miss: open the hand, go home, say so."""
    from agent.skills.motion import Trajectory
    robot = get_robot()
    await _ready(robot)
    start = robot.command_state()
    traj = await asyncio.to_thread(plan, robot.ctx, *start)
    if not traj.ok:
        return _result("failed", reason=f"no safe plan: {traj.reason}"[:400], **echo)
    seconds = traj.ticks / HZ
    if robot.time_left() is not None and seconds > robot.time_left():
        return _result("failed", reason=f"not enough time left for a {seconds:.0f} s move", **echo)
    # (the mock world has no finger contact, so the check only makes sense on the simulator or the robot)
    close = await _too_close(robot, traj)
    if close:
        return _result("failed", reason=close, **echo)
    lift_end = next((e for n, s, e in traj.phases if n == "lift"), None) if GRASP_CHECK and robot.world is None else None
    if lift_end is None:
        parts = [traj]
    else:
        parts = [Trajectory(True, traj.skill, traj.args, traj.arm[:lift_end], traj.hand[:lift_end], side=traj.side)]
        if lift_end < traj.ticks:
            parts.append(Trajectory(True, traj.skill, traj.args, traj.arm[lift_end:], traj.hand[lift_end:], side=traj.side))
    was = np.asarray(obj.xyz, float).copy()
    outcome = await _run(robot, parts[0], label if len(parts) == 1 else f"{label}: grasp and lift", start)
    check = None
    if outcome == "completed" and lift_end is not None:
        check = _grasp_check(robot, traj.side)
        if not check["object_in_hand"]:
            def recover(ctx, arm_q, hand_q):
                from agent.skills import go_home, hold, set_hand
                rel = set_hand(ctx, arm_q, hand_q, traj.side, "release", hold_s=0.2)
                home = go_home(ctx, rel.end_arm, rel.end_hand) if rel.ok else go_home(ctx, arm_q, hand_q)
                return rel.then(home) if (rel.ok and home.ok) else (home if home.ok else hold(arm_q, hand_q, 0.3))

            back = robot.command_state()
            await _run(robot, await asyncio.to_thread(recover, robot.ctx, *back), f"{label}: missed, back home", back, settle=False)
            robot.holding = robot.held_id = None
            await _park(robot)
            out = _result("missed", reason="the hand closed on nothing (finger sensors): the object is still on the "
                          "counter. The arm is back home.", grasp_check=check, arm=traj.side, **echo)
            try:
                out["objects_on_counter"] = await asyncio.to_thread(_detect, robot)
                now = robot.objects.get(obj.id) or min(robot.objects.values(), default=None,
                                                       key=lambda o: np.linalg.norm(o.xyz[:2] - was[:2]))
                if now is not None and np.linalg.norm(now.xyz[:2] - was[:2]) < 0.06:
                    out["object_moved_m"] = [round(float(v), 3) for v in (now.xyz[:2] - was[:2])]
                    out["object_now"] = now.id
            except Exception as e:  # noqa: BLE001
                out["objects_on_counter"] = f"look failed: {e!r}"[:200]
            return out
        if len(parts) > 1:
            outcome = await _run(robot, parts[1], f"{label}: carry and release", robot.command_state(), settle=False)
    if outcome != "completed":
        robot.holding = robot.held_id = None
        return _result("aborted", reason=outcome, **echo)
    on_done(traj)
    out = _result("completed", arm=traj.side, seconds=round(seconds, 1), **echo)
    if check is not None:
        out["grasp_check"] = check
    await _park(robot)
    if look_after and not robot.holding:
        try:
            out["objects_on_counter"] = await asyncio.to_thread(_detect, robot)
        except Exception as e:  # noqa: BLE001
            out["objects_on_counter"] = f"look failed: {e!r}"[:200]
    return out


async def _do_pick_and_place(robot, obj, bin_name, arm=None, approach=None, drop=None, **echo):
    drop = _drop_point(robot, bin_name, drop)

    def plan(ctx, arm_q, hand_q):
        from agent.skills import pick_and_place as skill
        return skill(ctx, obj, bin_name, arm_q, hand_q, arm=arm, approach=approach, drop_offset=drop)

    def placed(traj):
        robot.holding = robot.held_id = None
        robot.placed[bin_name] = robot.placed.get(bin_name, 0) + 1

    return await _grasp_move(f"pick_and_place {obj.id} -> {bin_name}", plan, placed, obj, True, object=obj.id,
                             bin=bin_name, grasp_at_m=[round(float(v), 3) for v in obj.xyz], **echo)


async def _do_hand_over(robot, obj, bin_name):
    others = [o for o in robot.objects.values() if o.id != obj.id]

    def plan(ctx, arm_q, hand_q):
        from agent.skills import relocate
        return relocate(ctx, obj, bin_name, arm_q, hand_q, others=others)

    def moved(traj):
        robot.holding = robot.held_id = None
        if traj.args.get("spot"):
            _expect(robot, obj.id, traj.args["spot"])

    return await _move(f"hand_over {obj.id} (for {bin_name})", plan, moved, look_after=True, object=obj.id, bin=bin_name)


# ---------------------------------------------------------------------------------------------- sensing tools
@_tool
async def look() -> dict:
    """Detect the objects that are on the counter now, with the head camera.

    Call this at the start and whenever you are unsure what is where. It gives each object an id you can pass to
    pick_and_place, pick and hand_over. An object keeps its id for the whole episode as long as it stays where
    it is. The arms should be at home (out of the camera's view) when you look. It finds anything with a clear
    colour that stands on the counter at object height; if it misses an object you can see, use point instead.

    Returns:
      objects_on_counter: one entry per object: id, color (a rough name from its hue, which can be wrong under
        coloured light: your next live view shows the head image with every id drawn on its object, so check the
        real colour and shape there), side of the robot it lies on, position_m [x forward, y left] in metres from the robot's
        pelvis, point [y, x] in the head image (0..1000), and a rough guess of shape and rotation (often wrong:
        trust the image over it).
      bins: where the three bins are, as position_m.
    """
    robot = get_robot()
    await _ready(robot)
    await _park(robot)                      # arms out of the camera's view (an arm that holds something stays)
    objects = await asyncio.to_thread(_detect, robot)
    robot.view_requests.append({"camera": "head", "region": None, "mark_objects": True})   # ids drawn on the image
    ctx = robot.ctx
    bins = {b: [round(float(v), 3) for v in ctx.planner.bin_xy(b)] for b in ctx.bins()}
    return _result("ok", objects_on_counter=objects, bins=bins, instruction=robot.instruction)


@_tool
async def view_camera(camera: Camera = "head", region: Optional[list[float]] = None, mark_objects: bool = False) -> dict:
    """Look through one of the robot's cameras at full resolution. The picture arrives with your next live view.

    The live view you get on every turn is the head camera, scaled down (plus the wrist camera of a hand that
    holds something). Use this tool for more: a wrist camera to see what is between the fingers, a zoomed region
    to tell a cube from a triangle or to see inside a bin, the depth image to judge heights, or the head image
    with the ids from look drawn on it.

    Args:
      camera: head (colour, looks down at the counter and bins), left_wrist or right_wrist (colour, on the
        forearm, looks along the hand), head_depth (the head camera's depth, coloured: red is near, blue is far).
      region: [ymin, xmin, ymax, xmax], each 0..1000, to zoom into part of the image. Leave out for all of it.
      mark_objects: Draw the object ids from the latest look on the image (head camera only).
    """
    robot = get_robot()
    await _ready(robot)
    key = {"head": "head_rgb", "left_wrist": "left_wrist_rgb", "right_wrist": "right_wrist_rgb",
           "head_depth": "head_depth"}[camera]
    if robot.obs.get(key) is None:
        raise ToolError(f"the {camera} camera is not sending images")
    if region is not None:
        if len(region) != 4 or not (region[0] < region[2] and region[1] < region[3]):
            raise ToolError("region must be [ymin, xmin, ymax, xmax] with ymin < ymax and xmin < xmax, each 0..1000")
        region = [float(np.clip(v, 0, 1000)) for v in region]
    robot.view_requests.append({"camera": camera, "region": region, "mark_objects": bool(mark_objects)})
    h, w = robot.obs[key].shape[:2]
    return _result("ok", camera=camera, image="attached to your next live view", full_resolution=[int(h), int(w)])


@_tool
async def read_proprioception() -> dict:
    """Read the robot's joint sensors: what the arms and hands are doing right now, without a camera.

    Use it to check a grasp (a hand that was told to close but stopped short, with a tracking error, is blocked by
    an object between the fingers; a hand closed all the way is empty), to see whether an arm is at home or
    still moving, and to know where each hand is.

    Returns:
      left / right: for each arm, grasp_point_m [x forward, y left, z up] (where its fingers meet, metres from the
        pelvis), palm_tilt_deg (0 = palm facing straight down), at_home, arm_speed_rad_s (0 = standing still),
        arm_tracking_error_rad (commanded minus measured; large = the arm is blocked or lagging), and hand:
        closed_fraction (0 open, 1 fully closed, measured), commanded_closed_fraction, tracking_error_rad.
      arm_joints_rad / hand_joints_rad: every measured joint angle by name.
    """
    from agent.skills.perception import joint_state
    robot = get_robot()
    await _ready(robot)
    ctx, obs = robot.ctx, robot.obs
    state = joint_state(ctx, obs)
    arm = np.asarray(obs["arm_q"], float)
    speed = np.abs(np.asarray(obs.get("arm_dq", np.zeros(len(arm))), float))
    out = {}
    for side in ("left", "right"):
        sl = ctx.kin.sl(side)
        s = state[side]
        out[side] = {"grasp_point_m": s["grasp_point_m"], "palm_tilt_deg": s["palm_tilt_deg"], "at_home": s["at_home"],
                     "arm_speed_rad_s": round(float(speed[sl].max()), 3),
                     "arm_tracking_error_rad": round(float(np.abs(arm[sl] - robot.arm_cmd[sl]).max()), 3),
                     "hand": _hand_summary(robot, side)}
    out["arm_joints_rad"] = {n: round(float(v), 3) for n, v in zip(ctx.spec.arm_names, arm)}
    out["hand_joints_rad"] = {n: round(float(v), 3) for n, v in zip(ctx.spec.hand_names, np.asarray(obs["hand_q"], float))}
    return _result("ok", **out)


# ---------------------------------------------------------------------------------------------- acting tools
@_tool
async def pick_and_place(
    bin: Bin,
    object_id: Optional[str] = None,
    point: Optional[list[float]] = None,
    arm: Optional[Arm] = None,
    shape: Optional[Shape] = None,
    yaw_deg: Optional[float] = None,
    finger_direction_deg: Optional[float] = None,
    xy_offset_m: Optional[list[float]] = None,
    drop_offset_m: Optional[list[float]] = None,
) -> dict:
    """Pick one object from the counter with a top-down pinch, drop it in a bin and bring the arm home (about 10 s).

    Say which object with object_id (from look; the more precise way) or with point.

    Args:
      bin: The bin to drop it in.
      object_id: Id of the object from the latest look or objects_on_counter list.
      point: [y, x] of the object in the HEAD camera image, each 0..1000; point at the middle of its top face.
      arm: Which arm to use. Leave out to let the planner choose. The left bin can only be reached by the left
        arm and the right bin only by the right arm.
      shape: The object's shape, if you can see it. With yaw_deg, cubes and triangles are gripped on flat faces.
      yaw_deg: Rotation of the object about the vertical axis in degrees (0 = an edge pointing away from the
        robot). Only used with shape cube or triangle.
      finger_direction_deg: Where the fingers point: 0 = away from the robot, +90 = to its left.
      xy_offset_m: [dx, dy] metres added to the grasp position ([x forward, y left]). After a miss that pushed
        the object, shift the grasp the way the object moved (object_moved_m), by about 0.015.
      drop_offset_m: [dx, dy] metres from the bin's centre. Leave out: the drop points are spread automatically.

    Returns:
      status: "completed" (the finger sensors felt an object after the lift and it was carried to the bin and
        released), "missed" (the hand closed on nothing: the move was stopped after the lift, the arm is home,
        object_moved_m says how far the attempt pushed the object), "failed" (no safe plan: nothing moved; reason
        says why) or "aborted".
      objects_on_counter: what is on the counter afterwards.
    """
    robot = get_robot()
    await _ready(robot)
    if robot.holding:
        raise ToolError(f"the {robot.holding} hand is holding an object: place it first")
    obj = _target(robot, object_id, point, shape, yaw_deg)
    return await _do_pick_and_place(robot, obj, bin, arm, _approach(finger_direction_deg, xy_offset_m),
                                    _pair(drop_offset_m, "drop_offset_m"))


@_tool
async def pick(
    object_id: Optional[str] = None,
    point: Optional[list[float]] = None,
    for_bin: Optional[Bin] = None,
    arm: Optional[Arm] = None,
    shape: Optional[Shape] = None,
    yaw_deg: Optional[float] = None,
    finger_direction_deg: Optional[float] = None,
    xy_offset_m: Optional[list[float]] = None,
) -> dict:
    """Pick one object and hold it at carry height (about 5 s). Follow with place or place_on_counter.

    Use this instead of pick_and_place when you want to look at the object in the hand before deciding where it
    goes: the next live view includes that hand's wrist camera. Status "missed" means the hand closed on nothing
    and the arm went back home.

    Args:
      object_id: Id of the object from the latest look or objects_on_counter list.
      point: [y, x] of the object in the HEAD camera image, each 0..1000.
      for_bin: The bin you intend it for, if you already know: the arm that can reach that bin is then used.
      arm: Which arm to use. Leave out to let the planner choose.
      shape: The object's shape, if you can see it.
      yaw_deg: Rotation of the object about the vertical axis in degrees; only with shape cube or triangle.
      finger_direction_deg: Where the fingers point: 0 = away from the robot, +90 = to its left.
      xy_offset_m: [dx, dy] metres added to the grasp position.
    """
    robot = get_robot()
    await _ready(robot)
    if robot.holding:
        raise ToolError(f"the {robot.holding} hand is holding an object: place it first")
    obj = _target(robot, object_id, point, shape, yaw_deg)
    approach = _approach(finger_direction_deg, xy_offset_m)

    def plan(ctx, arm_q, hand_q):
        from agent.skills import pick as skill
        return skill(ctx, obj, arm_q, hand_q, arm=arm, approach=approach, for_bin=for_bin)

    def picked(traj):
        robot.holding, robot.held_id = traj.side, object_id

    return await _grasp_move(f"pick {obj.id}", plan, picked, obj, False, object=obj.id,
                             grasp_at_m=[round(float(v), 3) for v in obj.xyz])


@_tool
async def place(bin: Bin, drop_offset_m: Optional[list[float]] = None) -> dict:
    """Carry the held object over a bin, release it and bring the arm home (about 5 s).

    Args:
      bin: The bin to drop it in. It must be reachable by the arm that holds the object.
      drop_offset_m: [dx, dy] metres from the bin's centre. Leave out: the drop points are spread automatically.
    """
    robot = get_robot()
    await _ready(robot)
    if not robot.holding:
        raise ToolError("nothing is held: pick an object first")
    side, drop = robot.holding, _drop_point(robot, bin, _pair(drop_offset_m, "drop_offset_m"))

    def plan(ctx, arm_q, hand_q):
        from agent.skills import place as skill
        return skill(ctx, bin, arm_q, hand_q, arm=side, drop_offset=drop)

    def placed(traj):
        robot.objects.pop(robot.held_id, None)
        robot.holding = robot.held_id = None
        robot.placed[bin] = robot.placed.get(bin, 0) + 1

    return await _move(f"place -> {bin}", plan, placed, look_after=True, bin=bin)


@_tool
async def place_on_counter(point: Optional[list[float]] = None, position_m: Optional[list[float]] = None) -> dict:
    """Set the held object down on the counter and bring the arm home, e.g. near the middle for the other arm.

    Give either point or position_m.

    Args:
      point: [y, x] in the HEAD camera image, each 0..1000: where on the counter to put it.
      position_m: [x forward, y left] in metres from the robot's pelvis. [0.30, 0.0] is on the centre line,
        where both arms reach.
    """
    from agent.skills.perception import point_to_xyz
    robot = get_robot()
    await _ready(robot)
    if not robot.holding:
        raise ToolError("nothing is held: pick an object first")
    if position_m is not None:
        xy = np.asarray(_pair(position_m, "position_m"))
    elif point is not None and len(point) == 2:
        xy = point_to_xyz(robot.ctx, {**robot.obs, "head_depth": None}, point)[0][:2]
    else:
        raise ToolError("give point ([y, x] on the head image) or position_m ([x, y] metres)")
    side = robot.holding

    def plan(ctx, arm_q, hand_q):
        from agent.skills import place_on_counter as skill
        return skill(ctx, xy, arm_q, hand_q, arm=side)

    def put_down(traj):
        _expect(robot, robot.held_id, xy)
        robot.holding = robot.held_id = None

    return await _move("place_on_counter", plan, put_down, look_after=True,
                       position_m=[round(float(v), 3) for v in xy])


@_tool
async def hand_over(bin: Bin, object_id: Optional[str] = None, point: Optional[list[float]] = None) -> dict:
    """Move an object towards the other arm (about 10 s), for when its bin is on the far side of the body.

    Only needed after pick_and_place failed for that reason. The arm near the
    object picks it and sets it down near the centre line, at a spot from which the other arm has a complete
    plan to the bin. The object keeps its id; afterwards pick_and_place it.

    Args:
      bin: The bin the object should end up in.
      object_id: Id of the object from the latest look or objects_on_counter list.
      point: [y, x] of the object in the HEAD camera image, each 0..1000.
    """
    robot = get_robot()
    await _ready(robot)
    if robot.holding:
        raise ToolError(f"the {robot.holding} hand is holding an object: place it first")
    return await _do_hand_over(robot, _target(robot, object_id, point, None, None), bin)


@_tool
async def go_home() -> dict:
    """Open both hands and move both arms to their resting pose: out to the sides, clear of each other and out
    of the head camera's view.

    The arms go there by themselves after every move, so this is only needed to let go of an object you do not
    want to place (it is dropped where the hand is) or to recover from something unexpected.
    """
    robot = get_robot()
    await _ready(robot)
    robot.holding = robot.held_id = None
    if robot.aside is None:
        def plan(ctx, arm_q, hand_q):
            from agent.skills import go_home as skill
            return skill(ctx, arm_q, hand_q)
        return await _move("go_home", plan, None, look_after=True)
    await _park(robot)
    out = _result("completed")
    out["objects_on_counter"] = await asyncio.to_thread(_detect, robot)
    return out


@_tool
async def wait(seconds: float = 1.0) -> dict:
    """Hold still for a moment, then look at new images (objects still rolling, view blocked).

    Args:
      seconds: How long to hold still, 0.2 to 3.
    """
    seconds = float(np.clip(seconds, 0.2, 3.0))

    def plan(ctx, arm_q, hand_q):
        from agent.skills import hold
        return hold(arm_q, hand_q, seconds)

    return await _move(f"wait {seconds:.1f} s", plan)


@_tool
async def done(summary: str = "") -> dict:
    """End the episode: everything that can be sorted is sorted, or nothing useful is left to do.

    Args:
      summary: One sentence on what was sorted and what was left, for the log.
    """
    robot = get_robot()
    robot.finished = True
    robot.log(f"[robot t={robot.t:6.2f}] done: {summary}")
    out = _result("episode ended", moves=len(robot.moves))
    if robot.world is not None:                 # only the mock world knows the score
        out["mock_world_outcomes"] = robot.world.outcomes()
    return out


TOOLS = [look, view_camera, read_proprioception, pick_and_place, pick, place, place_on_counter, hand_over,
         go_home, wait, done]
