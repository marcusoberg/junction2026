"""The agent's tools: the organisers' oracle (aaltonomous/sim/oracle.py), cut into calls a model can make, plus
tools to look through the cameras and read the joint sensors.

Sensing
    ask_gemini          one closed question about a camera image to the Gemini Decisions API (about a second).
    look                the oracle's "look": which objects are on the counter and where. The oracle reads exact
                        poses from the simulator, which a submission may not; this uses only the head camera: any
                        saturated colour (not only the sim's red / yellow / blue) + depth for the 3D position.
    view_camera         any camera at full resolution, optionally zoomed in or with the object ids drawn on it.
    find_blocks         asks the Gemini Robotics-ER model (er.py) which things on the counter are blocks, and merges
                        its answer with look's detections: for when look reports a hand or a bin edge, or misses a block.
    read_proprioception joint angles and speeds, where each hand is, how far it is closed, tracking errors.
Acting (each plans with the oracle's planner; see agent/skills)
    choose_arm          what each arm can do for an object (distance, reach, plans); the rule lives in arms.py
    check_plan          plan a grasp with all its checks without moving, to find a variation that works
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
import dataclasses
import functools
import os
import weakref
from typing import Literal, Optional

import numpy as np

from google.adk.tools.tool_context import ToolContext

from .robot import HZ, get_robot

Bin = Literal["left", "center", "right"]
Arm = Literal["left", "right"]
Shape = Literal["cube", "cylinder", "triangle"]
Camera = Literal["head", "left_wrist", "right_wrist", "head_depth"]

SAME_OBJECT_M = 0.04          # a detection this close to a known object is that object
# find_blocks: a point of the Robotics-ER model this close (on the table) to a depth detection confirms it. ER's points
# land a median 6-8 mm from the true centre in the sim (worst 13.5 mm) and a block is 45 mm wide; 3.5 cm is the gate
# the evaluation used when it compared the two on the real frames.
ER_MATCH_M = 0.035
# find_blocks: two ER points on ONE detection whose own depth positions lie this far apart are two touching blocks
# that the depth image shows as one object, placed between them (real frame drop3 ep 5: a cylinder and a cube,
# their pointed positions 5.3 cm apart, the detection 2.6 cm from each). Centres of touching blocks are 4.5 cm apart.
ER_SPLIT_M = 0.035
# find_blocks: a pointed-at "block" whose top is this much above a block's height is something else on top of it.
# Block tops read 4.8-5.4 cm above the counter on the real frames (nominal 4.5); a finger over a cylinder read 7.7.
ER_TOP_ABOVE_M = 0.015
# Grasp check from the joint sensors, right after the lift. Fingers that hold something cannot reach their commanded
# closed position: in the Isaac sim they stay 0.017-0.07 rad short, and 0.000 when the hand closed on nothing
# (a first threshold of 0.025 wrongly dropped held objects in run adk-2-lang).
GRASP_CHECK = os.environ.get("ROBOT_GRASP_CHECK", "1") == "1"
GRASP_ERR_RAD = float(os.environ.get("ROBOT_GRASP_ERR_RAD", "0.008"))
# Measured on 272 grasps in the Isaac sim (runs t2, s1, gb2): an empty hand reads 0.000-0.007 rad, a holding hand
# 0.010 and up. At or below GRASP_EMPTY_RAD the hand is empty, at or above GRASP_HELD_RAD it holds something;
# in between the finger sensors are not trusted alone and the Gemini Decisions API is asked to look.
GRASP_EMPTY_RAD = float(os.environ.get("ROBOT_GRASP_EMPTY_RAD", "0.004"))
GRASP_HELD_RAD = float(os.environ.get("ROBOT_GRASP_HELD_RAD", "0.012"))
# (our teammates measured the same on 147 picks: held 0.021-0.100 rad, empty 0.000-0.010; bot-icelli decision_oracle)


def _offset(name):
    """[dx forward, dy left, dz up] in metres from an environment variable, added to where the hand grasps."""
    try:
        v = [float(x) for x in os.environ.get(name, "0,0,0").split(",")]
        return np.array(v if len(v) == 3 else [0.0, 0.0, 0.0])
    except ValueError:
        return np.zeros(3)


# Systematic corrections of the grasp point, found with the grasp bench (sim/grasp_bench.py), for every object
# and for triangular prisms on top of that.
GRASP_OFFSET_M = _offset("ROBOT_GRASP_OFFSET_M")
TRIANGLE_OFFSET_M = _offset("ROBOT_TRI_OFFSET_M")


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


def _park_plan(robot, ctx, arm_q, hand_q, keep=None):
    """Joint-space move of every arm that holds nothing to its parked pose, hands open. None if already there.
    keep: an arm to leave where it is (the one about to move, or one waiting for its next object)."""
    from agent.skills.motion import Trajectory
    if robot.aside is None:
        return None
    to_a, to_h = arm_q.copy(), hand_q.copy()
    for side in ("left", "right"):
        if side != robot.holding and side != keep:
            sl = ctx.kin.sl(side)
            to_a[sl], to_h[sl] = robot.aside[sl], ctx.post["open"][sl]
    if np.abs(to_a - arm_q).max() < 1e-3 and np.abs(to_h - hand_q).max() < 1e-3:
        return None
    with ctx.lock:
        a, h = ctx.planner.go_to(arm_q, hand_q, to_a, to_h)
        if not len(a) or ctx.planner.dry_run_safety(arm_q, hand_q, a, h):
            return None
    return Trajectory(True, "park", {}, a, h, [("park", 0, len(a))])


async def _park(robot, keep=None):
    """Arms aside: out of each other's way and out of the head camera's view."""
    start = robot.command_state()
    traj = await asyncio.to_thread(_park_plan, robot, robot.ctx, *start, keep)
    if traj is not None:
        await _run(robot, traj, "arms aside" if keep is None else f"{'right' if keep == 'left' else 'left'} arm aside", start)
    if robot.waiting != keep:
        robot.waiting = None


CHAIN = os.environ.get("ROBOT_CHAIN", "1") == "1"
# 0 = the agent as it was before find_blocks (for A/B runs only): no tool, no hint to call it, and look does not flag
# detections as probably_not_a_block (so _next_for_arm skips none either)
ER = os.environ.get("ROBOT_ER", "1") == "1"
SELFCAL = os.environ.get("ROBOT_SELFCAL", "1") == "1"     # 0 = the agent as it was before calibrate (for A/B runs only)
CAMERA_CHECK_M = 0.02      # the table-based camera fit sits 8 mm from the calibration file's pose in the nominal sim


async def _next_for_arm(robot, side, placed_id, bin_name, end_arm, end_hand):
    """While an arm carries an object to its bin: should it go back home afterwards, or wait above the bin because
    its next object is right there? Saves the trip home and out again (about 3 s per object).

    Candidates are the other objects on the counter that this arm is the nearer one to. The Gemini Decisions API
    picks which of them to take next (or none), from the head image and the instruction; then the planner checks
    that this arm can grasp it from where it will be. Returns the object id, or None = go home."""
    from . import arms, decisions
    ctx = robot.ctx
    cands = [o for o in robot.seen if o["id"] != placed_id and o["id"] in robot.objects
             and not o.get("probably_not_a_block")            # an arm must not wait above the bin for a hand or a bin edge
             and arms.order(ctx, robot.objects[o["id"]].xyz, None, end_arm)[0] == side]
    if not cands:
        return None, "no other object on this arm's side"
    choice, why = cands[0]["id"], "nearest object on this arm's side"
    if decisions.available() and robot.seen_rgb is not None:
        try:
            options = {o["id"]: f"{(o.get('gemini_label') or {}).get('colour', '')} {(o.get('gemini_label') or {}).get('shape', o.get('shape_guess') or '')}"
                                f" on the robot's {o['side']}".strip() for o in cands}
            options["none"] = f"None of these should be picked next by the {side} arm: send the arm home."
            a = await asyncio.to_thread(
                decisions.ask,
                "A humanoid robot sorts small coloured blocks from a counter into three bins (left, center, right). "
                f"The instruction is: \"{robot.instruction}\". Its {side} arm is carrying one block to the {bin_name} "
                "bin right now (ignore that block).",
                [decisions.image_block(robot.seen_rgb)],
                f"Which block should the {side} arm pick up next? Choose a block that the instruction sends to a "
                f"bin, the easiest to reach first.", "choice", options)
            if float(a.get("confidence", 0)) >= 0.5:
                choice, why = a.get("choice"), f"Gemini Decisions ({a.get('confidence'):.2f})"
        except Exception as e:  # noqa: BLE001
            robot.log(f"[tool] next-object decision failed: {e!r}"[:200])
    if choice in (None, "none") or choice not in robot.objects:
        return None, why
    nxt = _target(robot, choice, None, None, None)
    plan = await asyncio.to_thread(_plan_grasp, robot, nxt, side, _pick_call(ctx), {}, end_arm, end_hand, None, False)
    if not plan.ok:
        return None, f"{choice}: no grasp plan from above the bin"
    return choice, why


async def _show(tool_context, name, robot, request):
    """Save a camera view as an ADK artifact, so it appears in the `adk web` UI next to the tool call."""
    if tool_context is None:
        return
    try:
        from google.genai import types
        from .agent import _render
        await tool_context.save_artifact(name, types.Part.from_bytes(data=_render(robot, request), mime_type="image/jpeg"))
    except Exception:  # noqa: BLE001  (no artifact service: nothing to show, nothing lost)
        pass


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
    robot.recent = (robot.recent + [status])[-4:]
    out = {"status": status, **extra, "holding": f"an object in the {robot.holding} hand" if robot.holding else "nothing"}
    if robot.time_left() is not None:
        out["time_left_s"] = round(robot.time_left(), 1)
    return out


# ---------------------------------------------------------------------------------------------- perception
def _er_memory(robot):
    """What find_blocks has said in this episode: {"added": {id: {"point", "colour", "shape"}} for the blocks only
    the Robotics-ER model saw, "verdict": {id: "confirmed" | "not_confirmed" | "added"}, "note": {id: text}}. Kept
    on the robot and started afresh when robot.reset has made a new id_count (one per episode)."""
    mem = getattr(robot, "er_memory", None)
    if mem is None or mem["episode"] is not robot.id_count:
        mem = robot.er_memory = {"episode": robot.id_count, "added": {}, "verdict": {}, "note": {}}
    return mem


def _pointed_block(ctx, obs, point, says=None):
    """A block the Robotics-ER model points at and the depth detector did not report -> (WorldObject, "") or
    (None, why it cannot be used). It is located from the depth image exactly as an object the agent points at
    (agent.skills.perception.point_to_xyz: the highest surface near the point is its top face). Refused: no
    object-high depth there, something taller than a block there (a hand over it), outside the area of the mat the
    detector covers, inside a bin, or cut by the image border (the model also points at slivers of blocks at the
    image edge, 0-1 per real frame; their visible part says little about where their centre is).
    says: the shape the model gave it. Shape and rotation are measured from the outline of the top face, as for
    every detection, but kept only if the outline is one block's: no larger than a block's top face and of the
    shape the model named. A block that the detector missed usually touches something, and the outline then takes
    in the neighbour: on the real frames it measured 3000-3900 mm2 (a block alone: 1600-1950) and gave a wrong shape
    in 4 of 5 cases, with a meaningless rotation. Without a rotation the grasp is the plain pinch."""
    from agent.skills import WorldObject
    from agent.skills.perception import point_to_xyz
    from .shape import classify_top, top_face_points
    xyz, info = point_to_xyz(ctx, obs, point)
    if info.get("method") != "depth":
        return None, "the depth image shows nothing of a block's height there"
    xyz, sc = np.asarray(xyz, float), ctx.scene
    if xyz[2] > sc.counter_z + sc.object_height() + ER_TOP_ABOVE_M:
        return None, "something taller than a block is at that point (a hand over it?)"
    pm = np.asarray(sc.pelvis_to_mat(xyz[None]))[0]
    W, D = sc.cfg["counter"]["mat_size"]
    if not (-0.02 <= pm[0] <= W + 0.02 and -0.02 <= pm[1] <= D + 0.02):          # the same area as shape.detect_by_depth
        return None, "outside the part of the counter the arms work on"
    half = np.asarray(sc.cfg["bins"]["outer_size"], float) / 2 + 0.008           # the same margin as shape.detect_by_depth
    for name, ctr in sc.cfg["bins"]["centers"].items():
        if np.all(np.abs(pm[:2] - np.asarray(ctr, float)) <= half):
            return None, f"inside the {name} bin or on its wall"
    r = 0.5 * float(sc.cfg["objects"]["size"])                                    # the top face at its nominal size
    corners = xyz + np.array([[sx * r, sy * r, 0.0] for sx in (-1, 1) for sy in (-1, 1)])
    uv = ctx.cal.project(ctx.camera.cam, corners)
    h, w = obs["head_rgb"].shape[:2]
    if uv[:, 0].min() < 0 or uv[:, 0].max() > w - 1 or uv[:, 1].min() < 0 or uv[:, 1].max() > h - 1:
        return None, "cut by the image border"
    o = WorldObject("?", "object", xyz, source="pointed", extra={"top_points": int(info.get("top_points", 0))})
    try:
        shape, yaw, _, feat = classify_top(top_face_points(ctx, obs, xyz))
    except Exception:  # noqa: BLE001
        shape, yaw, feat = None, None, {}
    if feat.get("area_mm2", 0.0) > 1.25 * (2000.0 * r) ** 2 or (says in ("cube", "cylinder", "triangle") and shape != says):
        shape, yaw = None, None
    o.extra["shape_seen"] = shape
    o.shape = shape if shape in ("cube", "cylinder", "triangle") else None
    o.yaw = yaw if o.shape in ("cube", "triangle") else None
    return o, ""


def _keep_er_blocks(robot, found):
    """The blocks only the Robotics-ER model saw (find_blocks) are not in the depth detector's list, so every new
    look would forget them and their ids. Keep each one that is still there: located again from the depth image at
    the point the model gave. It is dropped when the depth detector reports it now (the detection has taken its id),
    when nothing of a block's height is at the point any more (picked, pushed away, hidden by an arm), or when a
    detection sits there (other than one the model did not confirm: two touching blocks seen as one stay next to
    the two blocks that replace them)."""
    mem = _er_memory(robot)
    for i, a in list(mem["added"].items()):
        if any(o.id == i for o in found):
            mem["added"].pop(i)
            continue
        o, _ = _pointed_block(robot.ctx, robot.obs, a["point"], a.get("shape"))
        if o is None or any(np.linalg.norm(o.xyz[:2] - p.xyz[:2]) < SAME_OBJECT_M for p in found
                            if mem["verdict"].get(p.id) != "not_confirmed"):
            mem["added"].pop(i)
            mem["verdict"].pop(i, None)
            continue
        o.id = i
        found.append(o)


def _detect(robot):
    """Objects on the counter -> JSON for the model; remembered in robot.objects so tools can take their ids.
    An object keeps its id from one look to the next (same place), so ids stay valid across moves."""
    from agent.skills.perception import detect_camera, xyz_to_point
    from .shape import classify_top, detect_by_depth, top_face_points
    ctx, obs = robot.ctx, robot.obs
    # The ONLY source of object positions: the head camera's colour and depth images in `obs`. Nothing here (or
    # anywhere in this package) asks the simulator where an object is.
    robot.fit_head_camera(obs.get("head_depth"))
    by_depth = os.environ.get("ROBOT_DETECTOR", "depth") == "depth" and obs.get("head_depth") is not None
    if by_depth:                 # geometry only: no colour thresholds anywhere (shape.detect_by_depth)
        found = detect_by_depth(ctx, obs)
    else:                        # the organisers' colour-blob detector, widened to every hue (older path)
        found = detect_camera(ctx, obs, shapes=False)
        for o in found:
            try:
                shape, yaw, _, _ = classify_top(top_face_points(ctx, obs, o.xyz))
            except Exception:  # noqa: BLE001
                shape, yaw = None, None
            o.extra["shape_seen"] = shape
            o.shape = shape if shape in ("cube", "cylinder", "triangle") else None
            o.yaw = yaw if o.shape in ("cube", "triangle") else None
    found.sort(key=lambda o: -o.extra.get("n_px", 0))                    # one object matched by two masks is seen
    found = [o for i, o in enumerate(found)                               # twice: keep the larger blob
             if all(np.linalg.norm(o.xyz[:2] - p.xyz[:2]) > 0.03 for p in found[:i])]
    found.sort(key=lambda o: -o.xyz[1])                                   # left to right
    known = dict(robot.objects)
    for o in found:
        near = [(float(np.linalg.norm(np.asarray(k.xyz[:2]) - o.xyz[:2])), i) for i, k in known.items()]
        d, i = min(near, default=(1.0, None))
        if d < SAME_OBJECT_M:
            o.id = i
            known.pop(i)
        else:                    # ids say nothing about colour: a colour name guessed from the hue was wrong often
            robot.id_count["obj"] = robot.id_count.get("obj", 0) + 1     # enough (orange read as yellow) to put
            o.id = f"obj_{robot.id_count['obj']}"                         # objects in wrong bins
    _keep_er_blocks(robot, found)
    found.sort(key=lambda o: -o.xyz[1])
    verdict, added = _er_memory(robot)["verdict"], _er_memory(robot)["added"]
    robot.objects = {o.id: o for o in found}
    rgb = obs["head_rgb"]
    hw = rgb.shape[:2]
    out = []
    for o in found:
        y, x = xyz_to_point(ctx, o.xyz, hw)
        u, v = (int(round(c)) for c in o.extra.get("uv", (x / 1000 * (hw[1] - 1), y / 1000 * (hw[0] - 1))))
        # colour of the object, not of the glare on its top face: the most saturated third of a patch that also
        # covers the front face (the lit top face is washed out and made orange read as yellow)
        patch = rgb[max(0, v - 10):v + 22, max(0, u - 12):u + 13].reshape(-1, 3).astype(float)
        if len(patch):
            sat = (patch.max(1) - patch.min(1)) / np.maximum(patch.max(1), 1.0)
            patch = patch[sat >= np.percentile(sat, 67)]
        if o.extra.get("rgb"):
            patch = np.array([o.extra["rgb"]], float)
        out.append({"id": o.id, "side": "left" if o.xyz[1] > 0 else "right",
                    "position_m": [round(float(o.xyz[0]), 3), round(float(o.xyz[1]), 3)],
                    "point": [int(round(y)), int(round(x))],
                    "rgb": [int(c) for c in np.median(patch, axis=0)] if len(patch) else None,
                    "shape_guess": "lying on its side" if o.extra.get("shape_seen") == "tipped" else o.shape,
                    "yaw_deg_guess": None if o.yaw is None else round(float(np.degrees(o.yaw)), 1)})
        lab = robot.labels.get(o.id)
        if lab and lab.get("rgb") and out[-1]["rgb"] and max(abs(a - b) for a, b in zip(lab["rgb"], out[-1]["rgb"])) > 70:
            robot.labels.pop(o.id)      # the id now sits on a different-looking object (two touched): label it afresh
            lab = None
        if o.extra.get("ridge_yaw") is not None:
            out[-1]["lying_ridge_deg"] = round(float(np.degrees(o.extra["ridge_yaw"])), 1)
        if robot.failed.get(o.id):
            out[-1]["grasps_that_missed"] = [f.text() for f in robot.failed[o.id]]
        if lab:                                        # colour and shape from the Gemini Decisions API (see look)
            out[-1]["gemini_label"] = {"colour": lab["colour"], "colour_confidence": lab["colour_confidence"],
                                       "shape": lab["shape"], "shape_confidence": lab["shape_confidence"]}
        # Is it a block at all? A detection the Robotics-ER model did not confirm (find_blocks) is flagged, one it
        # confirmed is not; otherwise the label decides: a hand or a bin edge gets no confident colour. Flagged
        # detections are never removed.
        said = verdict.get(o.id)
        if said:
            out[-1]["find_blocks"] = said
        if _er_memory(robot)["note"].get(o.id):
            out[-1]["note"] = _er_memory(robot)["note"][o.id]
        if o.id in added:
            out[-1]["er_label"] = {"colour": added[o.id]["colour"], "shape": added[o.id]["shape"]}
        # A block the model added is held to the label rule as well: one stray point of the model (on a bin's front
        # wall, say) must not become a block nothing doubts any more.
        if said == "not_confirmed":            # only the ER model's verdict (find_blocks): no rule on colour or labels
            out[-1]["probably_not_a_block"] = True
    robot.trace.frame_dump(obs, robot.t, out)
    robot.seen, robot.seen_rgb = out, rgb
    return out


async def _detect_and_label(robot):
    """Detect, then let the Decisions API label every new object's colour and shape in one request."""
    from . import decisions
    objects = await asyncio.to_thread(_detect, robot)
    new = [o["id"] for o in objects if o["id"] not in robot.labels]
    if new and decisions.available():
        try:
            from .agent import _render
            import cv2
            marked = cv2.imdecode(np.frombuffer(_render(robot, {"camera": "head", "region": None, "mark_objects": True}),
                                                np.uint8), cv2.IMREAD_COLOR)[..., ::-1]
            seconds = await asyncio.to_thread(decisions.label_objects, marked, [o["id"] for o in objects],
                                              robot.instruction, robot.labels)
            if "_difficulty" in robot.labels:
                robot.difficulty = robot.labels.pop("_difficulty")
            for o in objects:                           # remember what each labelled object looked like
                if o["id"] in new and o["id"] in robot.labels:
                    robot.labels[o["id"]]["rgb"] = o["rgb"]
            robot.trace.write("decision_labels", t=round(robot.t, 2), seconds=seconds, difficulty=robot.difficulty,
                              labels={i: robot.labels.get(i) for i in new})
            objects = await asyncio.to_thread(_detect, robot)
        except Exception as e:  # noqa: BLE001  (labels are a help, never a requirement)
            robot.log(f"[tool] Decisions API labels failed: {e!r}"[:300])
            robot.trace.write("decision_labels", t=round(robot.t, 2), error=repr(e)[:300])
    return objects


async def _measure_bins(robot):
    """Find the three bins in the head image (Robotics-ER boxes + depth, bins.py) and use them for every drop.
    -> True if a bin was moved. Never raises; with no answer the bins stay where they are."""
    from . import bins
    obs = robot.obs
    if obs.get("head_depth") is None or obs.get("head_rgb") is None:
        return False
    try:
        try:
            boxes = await asyncio.wait_for(asyncio.to_thread(bins.ask, np.array(obs["head_rgb"])), 8.0)
        except asyncio.TimeoutError:
            boxes = None
        ctx = robot.ctx
        before = {b: [round(float(v), 3) for v in ctx.planner.bin_xy(b)] for b in ctx.bins()}
        with ctx.lock:
            if boxes:
                found = bins.measure(ctx, np.asarray(obs["head_depth"]), boxes)
                changed = bins.apply(ctx, found)
            else:                         # no usable answer from the model: the distance at least, from depth alone
                found = bins.near_wall(ctx, np.asarray(obs["head_depth"]))
                changed = bins.apply_near_wall(ctx, found) if found else {}
        robot.bins_measured = changed
        robot.log(f"[tool] bins measured: {changed} (were {before}); rim {ctx.scene.cfg['bins']['wall_height'] * 100:.1f} cm")
        robot.trace.write("bins", t=round(robot.t, 2), boxes=boxes, found=found, changed=changed, before=before,
                          rim_m=ctx.scene.cfg["bins"]["wall_height"])
        return bool(changed)
    except Exception as e:  # noqa: BLE001
        robot.log(f"[tool] bin measurement failed: {e!r}"[:200])
        robot.bins_measured = {}
        return False


async def _look_result(robot):
    """What look returns: the labelled detections, the bins, and what to check before trusting them."""
    from . import er
    measuring = asyncio.create_task(_measure_bins(robot)) if robot.bins_measured is None else None
    objects = await _detect_and_label(robot)
    if measuring is not None and await measuring:
        objects = await _detect_and_label(robot)       # the bins moved: what counts as "inside a bin" moved with them
    robot.view_requests.append({"camera": "head", "region": None, "mark_objects": True})   # ids drawn on the image
    ctx = robot.ctx
    bins = {b: [round(float(v), 3) for v in ctx.planner.bin_xy(b)] for b in ctx.bins()}
    out = _result("ok", objects_on_counter=objects, bins=bins, instruction=robot.instruction)
    fit = robot.pose_fit or {}
    if SELFCAL and not robot.calibrated and (fit.get("camera_moved_m", 0.0) > CAMERA_CHECK_M or (fit.get("fit_ok") and not fit.get("used"))):
        out["camera_check"] = (f"WARNING: to match the table and bins in the scene file, the camera would have to sit "
                               f"{fit.get('camera_moved_m', 0.0) * 100:.0f} cm from where the robot model has it. Either the camera "
                               f"mount is off or the table / the robot's place is not as in the file; grasps go wrong in "
                               f"the second case. Call calibrate once now, before picking, to measure which.")
    doubtful = [o["id"] for o in objects if o.get("probably_not_a_block") and "find_blocks" not in o]
    if doubtful and ER and er.available():
        out["block_check"] = (f"probably not a block: {', '.join(doubtful)} (a hand, a wrist or a bin edge gets no clear "
                              "colour). Check in the image, or call find_blocks to have every detection checked.")
    return out


def _merge_er(robot, items, obs):
    """Merge what the Robotics-ER model calls blocks (er.point_blocks on the head image of `obs`) with the depth
    detections of the last look. Positions stay the depth detector's (3.2 mm median in the sim against ER's 6-8 mm):
      - an ER point within ER_MATCH_M of a detection confirms it (nearest pairs first, one point per detection);
      - an ER point with no detection near it is a block the detector missed: located from the depth image as a
        pointed-at object is, and given a new id (unless _pointed_block refuses it);
      - a second ER point on a confirmed detection: two touching blocks that the depth image shows as one object,
        placed between them. Both blocks are then located from their own points and replace the detection, if
        their positions are ER_SPLIT_M apart; otherwise the second one is not added;
      - an ER point between ER_MATCH_M and SAME_OBJECT_M from a detection that no point confirmed: the block is
        added at its own position and the detection stays, not confirmed, with a note naming the block;
      - a detection no ER point confirms stays, marked not_confirmed.
    -> {"confirmed": [ids], "added": [ids], "not_confirmed": [ids], "not_added": [{point, colour, shape, reason}]}
    or None, with nothing remembered, when the answer confirms none of the detections and adds nothing: an empty
    or unusable answer is a failed call, not "nothing here is a block" (that verdict would flag every real
    detection for the rest of the episode)."""
    from agent.skills.perception import point_to_xyz
    ctx, mem = robot.ctx, _er_memory(robot)
    where = [np.asarray(point_to_xyz(ctx, obs, it["point"])[0], float)[:2] for it in items]
    pairs = sorted((float(np.linalg.norm(where[k] - o.xyz[:2])), i, k) for i, o in robot.objects.items() for k in range(len(items)))
    of_det, of_item = {}, {}
    for d, i, k in pairs:
        if d <= ER_MATCH_M and i not in of_det and k not in of_item:
            of_det[i], of_item[k] = k, i
    out = {"confirmed": [], "added": [], "not_confirmed": [], "not_added": []}
    replaced = {}                                          # a detection that is two blocks -> the ids of the two

    def add(o, it):
        robot.id_count["obj"] = robot.id_count.get("obj", 0) + 1
        o.id = f"obj_{robot.id_count['obj']}"
        robot.objects[o.id] = o
        mem["added"][o.id], mem["verdict"][o.id] = dict(it), "added"
        out["added"].append(o.id)
        return o.id

    for k, it in enumerate(items):
        if k in of_item:
            continue
        o, why = _pointed_block(ctx, obs, it["point"], it["shape"])
        near = [] if o is None else [i for i, p in robot.objects.items() if i not in replaced
                                     and np.linalg.norm(o.xyz[:2] - p.xyz[:2]) < SAME_OBJECT_M]
        if near:
            i = min(near, key=lambda j: np.linalg.norm(o.xyz[:2] - robot.objects[j].xyz[:2]))
            if i not in of_det and i in mem["added"] and i not in out["added"]:
                of_det[i] = k                              # the block an earlier call added, pointed at a little
                continue                                   # further off this time: the same block, it keeps its id
            if i not in of_det and i not in mem["added"]:
                # A detection that no point confirmed, 3.5-4 cm from this block: a depth blob merged with a hand, a
                # bin wall or a neighbour and so placed off the block. The block gets its own id and position
                # (as it does from 4 cm on); the detection stays, not confirmed.
                replaced[i] = [add(o, it)]
                continue
            first = _pointed_block(ctx, obs, items[of_det[i]]["point"], items[of_det[i]]["shape"])[0] if i in of_det and i not in mem["added"] else None
            if first is not None and np.linalg.norm(first.xyz[:2] - o.xyz[:2]) >= ER_SPLIT_M:
                replaced[i] = [add(first, items[of_det.pop(i)]), add(o, it)]
                continue
            o, why = None, f"the depth image cannot tell it apart from {i} (touching it?): sort {i} first, then look again"
        if o is None:
            out["not_added"].append({**it, "reason": why})
        else:
            add(o, it)
    if robot.objects and not of_det and not out["added"]:
        return None                                        # nothing matched (and nothing is remembered up to here)
    for i in list(robot.objects):
        if i in out["added"]:
            continue
        if i in of_det and i in mem["added"]:              # a block ER added in an earlier call and points at again.
            # Its image point is NOT replaced: the stored one located the block, the new one may not (on real
            # frame drop3 ep 3, 20 of 49 points within 2 cm of a prism at a bin were refused as too tall or inside
            # the bin), and the next look would then drop the block without a word.
            mem["added"][i].update(colour=items[of_det[i]]["colour"], shape=items[of_det[i]]["shape"])
            out["added"].append(i)
        elif i in of_det:
            mem["verdict"][i] = "confirmed"
            mem["note"].pop(i, None)
            out["confirmed"].append(i)
        else:
            mem["verdict"][i] = "not_confirmed"
            out["not_confirmed"].append(i)
            if len(replaced.get(i, [])) == 2:
                mem["note"][i] = (f"two touching blocks that look sees as one object, placed between them: use "
                                  f"{replaced[i][0]} and {replaced[i][1]} instead")
            elif i in replaced:
                mem["note"][i] = (f"look placed this detection off the block (its depth blob takes in something "
                                  f"next to it): the block is {replaced[i][0]}, use that instead")
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
    if yaw is None and object_id and seen.shape in ("cube", "triangle") and seen.yaw is not None \
            and shape in (None, seen.shape):
        # The hand has to be turned to the object's flat faces (a must for triangular prisms): use the shape and
        # rotation measured from the depth image, unless the model says it is something else.
        shape, yaw = seen.shape, float(seen.yaw)
    xyz = np.asarray(xyz, float) + GRASP_OFFSET_M + (TRIANGLE_OFFSET_M if shape == "triangle" and yaw is not None else 0.0)
    extra = {}
    if object_id and shape is None and seen.extra.get("ridge_yaw") is not None:
        extra["ridge_yaw"] = seen.extra["ridge_yaw"]          # lying on its side: the geometry decides, not a label
    return WorldObject(object_id or "pointed", color, xyz, shape if yaw is not None else None, yaw, source=source,
                       extra=extra)


def _plan_grasp(robot, obj, arm, call, info, arm_q, hand_q, bin_name=None, via_home=True, approach=None):
    """A checked plan for a skill that starts with a grasp, or the reasons there is none.
    call(side, arm_q, hand_q, target, approach) -> Trajectory plans the skill for one arm from one start state.
    The grasp options for this object (grasps.py) are tried in order, skipping the ones that already missed on it:
      1. from where the arms are now;
      2. the same again from the planner's own start pose for that arm (hand in front of the robot): the plans were
         tuned from there, and a start from the parked pose sometimes fails a clearance on the way to the object."""
    from agent.skills.motion import Trajectory
    from . import arms, grasps
    ctx = robot.ctx
    sides = [arm] if arm else arms.order(ctx, obj.xyz, bin_name, arm_q)
    variation = tuple(sorted((k, tuple(v) if isinstance(v, (list, tuple)) else v)
                             for k, v in dataclasses.asdict(approach).items()
                             if v is not None and k != "finger_direction_deg")) if approach is not None else ()
    direction = approach.finger_direction_deg if approach is not None else None
    everything = grasps.options(obj, sides, variation, direction)
    missed = robot.failed.get(obj.id, [])
    todo = [o for o in everything if o not in missed]
    if not todo:
        return Trajectory(False, "grasp", {}, side=sides[0], reason=(
            "every grasp option for this object has already missed (" + "; ".join(o.text() for o in everything) + "). "
            "Change something: xy_offset_m, max_tilt_deg, pregrasp_offset_m, grasp_height_frac or finger_direction_deg; "
            "or hand_over so the other arm can try; or leave it."))
    reasons = {}

    def attempt(opts, a_q, h_q):
        for opt in opts:
            with grasps.planner_settings(opt):
                t = call(opt.arm, a_q, h_q, grasps.target_for(obj, opt), grasps.approach_for(approach, opt))
            if grasps.accepted(opt, t):
                info.update(arm=opt.arm, grip=opt.text(), options_left=len(todo) - 1, option=opt)
                return t
            reasons.setdefault(opt.text(), t.reason if not t.ok else "the planner could only offer its default "
                               "pinch here, not this grip")
        return None

    with ctx.lock:
        t = attempt(todo, arm_q, hand_q)
        if t is not None:
            return t
        for side in sides if via_home else []:
            sl = ctx.kin.sl(side)
            if np.abs(arm_q[sl] - ctx.home[sl]).max() < 0.02:
                continue
            a2, h2 = arm_q.copy(), hand_q.copy()
            a2[sl], h2[sl] = ctx.home[sl], ctx.post["open"][sl]
            pa, ph = ctx.planner.go_to(arm_q, hand_q, a2, h2)
            if not len(pa) or ctx.planner.dry_run_safety(arm_q, hand_q, pa, ph):
                continue
            t = attempt([o for o in todo if o.arm == side], a2, h2)
            if t is not None:
                info["start"] = "from the pose in front of the robot"
                k = len(pa)
                return Trajectory(True, t.skill, t.args, np.vstack([pa, t.arm]), np.vstack([ph, t.hand]),
                                  [("to_start", 0, k)] + [(nm, a_ + k, b_ + k) for nm, a_, b_ in t.phases],
                                  reason=t.reason, side=t.side)
        note = f" (already missed: {'; '.join(o.text() for o in missed)})" if missed else ""
        return Trajectory(False, "grasp", {}, side=sides[0],
                          reason="; ".join(f"{what}: {why}" for what, why in reasons.items())[:600] + note)


def _approach(finger_direction_deg=None, xy_offset_m=None, speed=None, max_tilt_deg=None, pregrasp_offset_m=None,
              grasp_height_frac=None):
    from agent.skills import Approach
    for name, v in (("xy_offset_m", xy_offset_m), ("pregrasp_offset_m", pregrasp_offset_m)):
        if v is not None and len(v) != 2:
            raise ToolError(f"{name} must be [dx, dy] in metres")
    return Approach(finger_direction_deg=finger_direction_deg, speed=speed,
                    max_tilt_deg=None if max_tilt_deg is None else float(np.clip(max_tilt_deg, 10, 60)),
                    grasp_height_frac=None if grasp_height_frac is None else float(np.clip(grasp_height_frac, 0.2, 0.9)),
                    pregrasp_offset=None if pregrasp_offset_m is None else tuple(float(v) for v in pregrasp_offset_m),
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
    # The oracle's pattern steps 3 cm for 4.5 cm blocks, and for the right bin it steps towards the wall. In the sim
    # run fix-real-2 two cubes released 1 cm apart ended stacked: the upper one pokes above the rim and scored as
    # "left on the counter". Here the releases are 5.5 cm apart sideways, towards the middle of the bin.
    k = robot.placed.get(bin_name, 0) % 4
    try:
        planner = robot.ctx.planner
        base = np.asarray(super(type(planner), planner).drop_xy(bin_name), float)
        s = float(np.sign(np.asarray(planner.bin_xy(bin_name), float)[1] - base[1]))
    except Exception:  # noqa: BLE001
        s = 0.0
    if abs(s) < 0.5:                                   # the middle bin: its nearest point is its middle
        return [(0.0, 0.0), (0.0, 0.055), (0.0, -0.055), (0.03, 0.0)][k]
    return [(0.0, 0.0), (0.0, 0.055 * s), (0.03, 0.0275 * s), (0.0, 0.09 * s)][k]


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
    if close and robot.waiting and robot.waiting != traj.side:
        await _park(robot, keep=traj.side)            # the other arm was waiting in the way: move it aside, plan again
        start = robot.command_state()
        traj = await asyncio.to_thread(plan, robot.ctx, *start)
        if not traj.ok:
            return _result("failed", reason=f"no safe plan: {traj.reason}"[:400], **echo)
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


# Decisions API checks around a move (measured on saved sim frames, 15 Nov):
#   "is a block held?" from the WRIST camera: 12 of 14 held grasps at or above 0.5 (13 of 14 above 0.3), 14 of 14 empty
#   hands at or below 0.20. From the head camera the hand hides the block: 5 of 14 held grasps recognised. So the
#   wrist image is the one to ask about.
#   "is the hand above the bin?" from the head camera, just before the hand opens: 0.29-0.89 when it was (14 cases),
#   0.06-0.67 when the hand was still above the object. Below ABOVE_BIN_MIN the hand does not open.
# Empty hands read 0.10-0.25 in the sim (26 cases), held blocks 0.28 and up. On the real robot (15 Nov) the first
# pick read 0.24 and the hand let go; the two picks that were carried read 0.43 and 0.32. A hand wrongly called
# holding costs a 4 s empty carry and the next look finds the block; one wrongly called empty drops a block it had.
HELD_MIN = 0.25               # used only where the finger sensors can tell but read in their unclear band (the sim)
STILL_THERE_M = 0.05          # an object detected this close to where the picked one stood is the picked one, not gone


def _spot(robot, xy, r=0.02):
    """What the head camera's depth image shows at one place on the counter: "object" (something of block height
    stands there), "empty" (the bare counter), "hidden" (an arm is in the way) or "unseen" (no depth there)."""
    ctx = robot.ctx
    depth = robot.obs.get("head_depth")
    if depth is None:
        return "unseen"
    cam = ctx.cal.cams[ctx.camera.cam]
    K, T = np.asarray(cam.K, float), np.asarray(cam.T_pelvis_optical, float)
    H, W = depth.shape
    v, u = np.mgrid[0:H:2, 0:W:2]
    z = np.asarray(depth)[::2, ::2].astype(np.float32) * cam.depth_scale
    P = np.stack([(u - K[0, 2]) / K[0, 0] * z, (v - K[1, 2]) / K[1, 1] * z, z], -1) @ T[:3, :3].T + T[:3, 3]
    near = (z > 0.15) & (np.abs(P[..., 0] - xy[0]) < r) & (np.abs(P[..., 1] - xy[1]) < r)
    h = P[..., 2][near] - ctx.scene.counter_z
    if len(h) < 12:
        return "unseen"
    if np.median(h) > 0.09:
        return "hidden"
    return "object" if float(((h > 0.02) & (h < 0.075)).mean()) >= 0.35 else "empty"
ABOVE_BIN_MIN = 0.2
IN_BIN_MIN = 0.2


# How hard the hand closes. The organisers' "closed" posture stops the finger tips 36 mm from the thumb tip: 9 mm
# into a 45 mm block, a light pinch that holds in the sim. On the real hands it does not (run of 15 Nov: "does not
# grip tightly enough"): their fingers are driven softly (kp 1.5), so the force is what the command asks beyond the
# contact, and the operators whose demonstrations hold these blocks command the tips to MEET (1 mm apart, data drop
# 3, 15 grasps: finger tip joints at 1.1-1.75 rad against our 0.6). "firm" curls the two finger tip joints to
# FIRM_TIP_RAD: tips 12 mm from the thumb, the thumb and the hand's place on the block unchanged.
# ROBOT_GRIP: auto = firm on hands whose open-hand sensor reading is not zero (the real ones), firm, or sim.
GRIP = os.environ.get("ROBOT_GRIP", "auto")
# ROBOT_FIRM_TIP_RAD: how far the finger tip joints curl in the firm grip (0.6 = the sim pinch, 1.75 = the joint limit).
# In the sim, forced on, 1.2 held every block but one cube stayed in the opened hand in each of two episodes and
# fell off later (the sim lets the tips sink into the block). Lower it if blocks stick on the real hands, raise it
# if they slip.
FIRM_TIP_RAD = float(os.environ.get("ROBOT_FIRM_TIP_RAD", "1.2"))


def _firm(ctx, traj):
    """The same trajectory with the grasping hand's finger tip joints curled further wherever the hand is closing or
    closed (in proportion to how far the hand has closed in that row). Rows that open the hand are left alone."""
    sl = ctx.kin.sl(traj.side)
    names = [n for n in ctx.spec.hand_names][sl]
    opened, closed = np.asarray(ctx.post["open"], float)[sl], np.asarray(ctx.post["closed"], float)[sl]
    tips = [i for i, n in enumerate(names) if ("index_1" in n or "middle_1" in n)]
    base = [i for i, n in enumerate(names) if ("index_0" in n or "middle_0" in n)][0]      # tells how far the hand has closed
    if not tips or abs(closed[base] - opened[base]) < 1e-6:
        return traj
    hand = np.array(traj.hand, float)
    rows = hand[:, sl]
    f = np.clip((rows[:, base] - opened[base]) / (closed[base] - opened[base]), 0.0, 1.0)
    lo, hi = np.asarray(ctx.spec.hand_lower, float)[sl], np.asarray(ctx.spec.hand_upper, float)[sl]
    for i in tips:
        firm = float(np.clip(np.sign(closed[i]) * FIRM_TIP_RAD, lo[i], hi[i]))
        # ADDED to the planned row in proportion to how far the hand is closed, so the tips follow the hand both
        # ways without a jump. (A first version replaced the rows and jumped back at the release: the organisers'
        # safety filter clipped 31 steps for finger speed and aborted the episode, sim run fix-real-1.)
        want = rows[:, i] + f * (firm - closed[i])
        step = 0.85 * float(ctx.safety.hand_max_vel) / HZ             # under the safety filter's finger speed limit
        for t in range(1, len(want)):                                 # the tips follow at that speed at most
            want[t] = np.clip(want[t], want[t - 1] - step, want[t - 1] + step)
        rows[:, i] = want
    hand[:, sl] = rows
    return _within_hand_speed(ctx, traj, hand)


def _within_hand_speed(ctx, traj, hand):
    """The trajectory with the new hand rows, if no finger joint's command changes faster from one tick to the next
    than the organisers' safety filter allows (hand_max_vel, 3 rad/s: a faster command is clipped, and clipped steps
    abort the episode); else the trajectory as it was."""
    limit = 0.9 * float(ctx.safety.hand_max_vel) / HZ
    step = float(np.abs(np.diff(hand, axis=0)).max(initial=0.0))
    return dataclasses.replace(traj, hand=hand) if step <= limit else traj


async def _decide(robot, text, image, question):
    """One yes/no question to the Decisions API -> probability of yes, or None (not available, failed, too slow)."""
    from . import decisions
    if not decisions.available() or image is None:
        return None
    try:
        a = await asyncio.wait_for(asyncio.to_thread(decisions.ask, text, [decisions.image_block(image)], question, "noul"), 4.0)
        return float(a.get("noul", 0.5))
    except Exception as e:  # noqa: BLE001  (a check must never stop a move)
        robot.log(f"[tool] Decisions check failed: {e!r}"[:200])
        return None


def _needs_control(robot, why, side=None):
    """A check went wrong: the controlling model gets the reason, the wrist view of that hand, and more thinking."""
    robot.needs_control.append(why)
    if side:
        robot.view_requests.append({"camera": f"{side}_wrist", "region": None, "mark_objects": False})


async def _grasp_check(robot, side):
    """After the lift: is something in the hand? Two witnesses: the finger sensors (a hand that holds something
    stops short of its command) and the Gemini Decisions API looking through that hand's wrist camera.
    The sensors decide when they can tell: in the sim an open or empty hand reads 0.000. Where a hand that holds
    nothing still reads an error (the real hands: at least 0.029 rad in every frame of data drop 3) they cannot, and
    the Decisions API decides. When the two disagree the controlling model is told and shown the wrist view."""
    sl = robot.ctx.kin.sl(side)
    err = float(np.abs(np.asarray(robot.obs["hand_q"], float)[sl] - robot.hand_cmd[sl]).max())
    out = {"finger_tracking_error_rad": round(err, 3)}
    robot.trace.snapshot("grasp", robot.obs, robot.t)
    seen = await _decide(robot, f"The camera on a humanoid robot's {side} wrist looks along the fingers of its {side} hand.",
                         robot.obs.get(f"{side}_wrist_rgb"), "Is a small coloured block held between the fingers of this hand?")
    if seen is not None:
        out["decisions_probability_object_in_hand"] = round(seen, 2)
    # an open hand reads (almost) no error. ROBOT_FINGER_SENSORS=0 treats the sensors as on the real hands, to run that
    # path in the sim, where the truth is known.
    sensors_tell = (os.environ.get("ROBOT_FINGER_SENSORS", "1") == "1"
                    and robot.finger_baseline.get(side, 0.0) <= GRASP_EMPTY_RAD)
    if sensors_tell and err >= GRASP_HELD_RAD:
        out.update(object_in_hand=True, decided_by="finger sensors")
    elif sensors_tell and err <= GRASP_EMPTY_RAD:
        out.update(object_in_hand=False, decided_by="finger sensors")
    elif not sensors_tell:
        # The real hands. Their finger sensors read an error with nothing in the hand, and the wrist view was wrong
        # there: in the run of 15 Nov it read 0.43 and 0.32 for two hands that held nothing (both were carried to
        # the bin and "placed"), and 0.24 for the third. So nothing is decided here: the carry goes ahead and the
        # table is asked afterwards whether the object has left its place (_spot, and the look after the move).
        out.update(object_in_hand=True, unverified=True,
                   decided_by="not decided at the lift (finger sensors and wrist view cannot tell on this hand): "
                              "checked on the counter after the carry")
    elif seen is not None:
        out.update(object_in_hand=bool(seen >= HELD_MIN), decided_by="Gemini Decisions, wrist camera")
    else:
        out.update(object_in_hand=bool(err >= GRASP_ERR_RAD), decided_by="finger sensors (unclear)")
    if seen is not None and "finger sensors" == out["decided_by"] and (seen >= 0.8) != out["object_in_hand"] and (seen <= 0.2) == out["object_in_hand"]:
        out["witnesses_disagree"] = True
        _needs_control(robot, f"grasp check: the finger sensors say the {side} hand is {'holding' if out['object_in_hand'] else 'empty'}, "
                              f"the wrist camera check says the opposite ({seen:.2f}); the wrist view is attached", side)
    robot.trace.write("grasp_check", t=round(robot.t, 2), arm=side, **out)
    return out


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
    used = (echo.get("grasp") or {}).pop("option", None)       # which grasp option this plan is (for the memory)
    if not traj.ok:
        return _result("failed", reason=f"no safe plan: {traj.reason}"[:400], **echo)
    seconds = traj.ticks / HZ
    if robot.time_left() is not None and seconds > robot.time_left():
        return _result("failed", reason=f"not enough time left for a {seconds:.0f} s move", **echo)
    close = await _too_close(robot, traj)
    if close and any(np.abs(start[0][robot.ctx.kin.sl(o)] - robot.aside[robot.ctx.kin.sl(o)]).max() > 1e-3
                     for o in ("left", "right") if o != traj.side and robot.aside is not None):
        await _park(robot, keep=traj.side)            # the other arm was waiting in the way: move it aside, plan again
        start = robot.command_state()
        traj = await asyncio.to_thread(plan, robot.ctx, *start)
        used = (echo.get("grasp") or {}).pop("option", used)
        if not traj.ok:
            return _result("failed", reason=f"no safe plan: {traj.reason}"[:400], **echo)
        close = await _too_close(robot, traj)
    if close:
        return _result("failed", reason=close, **echo)
    # (the mock world has no finger contact, so the grasp check only makes sense on the simulator or the robot)
    lift_end = next((e for n, s, e in traj.phases if n == "lift"), None) if GRASP_CHECK and robot.world is None else None
    sl_ = robot.ctx.kin.sl(traj.side)             # what the finger sensors read with the hand open, before it closes
    open_err = float(np.abs(np.asarray(robot.obs["hand_q"], float)[sl_] - robot.hand_cmd[sl_]).max())
    # the smallest reading seen: a hand that has only just opened (it released the last object a moment ago) still
    # reads an error for a few ticks; in the sim episode of 15 Nov that made one check distrust good sensors.
    robot.finger_baseline[traj.side] = min(open_err, robot.finger_baseline.get(traj.side, open_err))
    if GRIP == "firm" or (GRIP == "auto" and robot.finger_baseline[traj.side] > GRASP_EMPTY_RAD):
        traj = _firm(robot.ctx, traj)             # the real hands (their sensors never read zero): squeeze harder
    open_start = (next((s for n, s, e in traj.phases if n == "open"), None)
                  if GRASP_CHECK and robot.world is None and bin_name_of(echo) else None)
    home_start = next((s for n, s, e in traj.phases if n == "home"), None) if CHAIN and bin_name_of(echo) else None
    cut = lambda i, j: Trajectory(True, traj.skill, traj.args, traj.arm[i:j], traj.hand[i:j], side=traj.side)  # noqa: E731
    was = np.asarray(obj.xyz, float).copy()
    robot.waiting = None
    first_end = lift_end if lift_end is not None else (home_start if home_start is not None else traj.ticks)
    outcome = await _run(robot, cut(0, first_end), label if first_end == traj.ticks else f"{label}: grasp and lift"
                         if lift_end is not None else f"{label}: to the bin", start, settle=first_end == lift_end)
    check = None
    if outcome == "completed" and lift_end is not None:
        check = await _grasp_check(robot, traj.side)

    async def missed(reason, arm_is_back=False):
        """The hand has nothing: open it, bring the arm back, remember that this grip missed, and say what is left."""
        if not arm_is_back:
            def recover(ctx, arm_q, hand_q):
                from agent.skills import go_home, hold, set_hand
                rel = set_hand(ctx, arm_q, hand_q, traj.side, "release", hold_s=0.2)
                home = go_home(ctx, rel.end_arm, rel.end_hand) if rel.ok else go_home(ctx, arm_q, hand_q)
                return rel.then(home) if (rel.ok and home.ok) else (home if home.ok else hold(arm_q, hand_q, 0.3))

            back = robot.command_state()
            await _run(robot, await asyncio.to_thread(recover, robot.ctx, *back), f"{label}: missed, back home", back, settle=False)
        robot.holding = robot.held_id = None
        if used is not None:                        # never try this arm + grip on this object again
            robot.failed.setdefault(obj.id, []).append(used)
        await _park(robot)
        out = _result("missed", reason=reason, grasp_check=check, arm=traj.side, **echo)
        try:
            out["objects_on_counter"] = await asyncio.to_thread(_detect, robot)
            now = robot.objects.get(obj.id) or min(robot.objects.values(), default=None,
                                                   key=lambda o: np.linalg.norm(o.xyz[:2] - was[:2]))
            if now is not None and np.linalg.norm(now.xyz[:2] - was[:2]) < 0.06:
                out["object_moved_m"] = [round(float(v), 3) for v in (now.xyz[:2] - was[:2])]
                out["object_now"] = now.id
                if now.id != obj.id and robot.failed.get(obj.id):      # pushed far enough to get a new id
                    robot.failed[now.id] = robot.failed.pop(obj.id)
                    out["objects_on_counter"] = await asyncio.to_thread(_detect, robot)
            left = [o.text() for o in _options_left(robot, now)] if now is not None else []
            out["other_grasps_to_try"] = left or "none: change an argument, hand it over, or leave it"
        except Exception as e:  # noqa: BLE001
            out["objects_on_counter"] = f"look failed: {e!r}"[:200]
        return out

    if check is not None and not check["object_in_hand"]:
        return await missed(f"the hand closed on nothing ({check['decided_by']}): the object is still on the counter. "
                            "The arm is back home.")
    unverified = bool(check and check.get("unverified"))
    stay, why, landed = None, "", None
    if outcome == "completed" and first_end < traj.ticks:
        mid = home_start if home_start is not None else traj.ticks
        decide = None
        if home_start is not None and not unverified:   # decided while the arm carries the object: no time is lost
            decide = asyncio.create_task(_next_for_arm(robot, traj.side, obj.id, bin_name_of(echo),
                                                       traj.arm[home_start - 1].copy(), traj.hand[home_start - 1].copy()))
        if mid > first_end and open_start is not None and first_end < open_start < mid:
            # carry, then ask before the hand opens: is it above the bin? A release beside the bin costs the object.
            outcome = await _run(robot, cut(first_end, open_start), f"{label}: carry", robot.command_state(), settle=True)
            if outcome == "completed" and unverified:
                state = await asyncio.to_thread(_spot, robot, was[:2])
                robot.trace.write("table_check", t=round(robot.t, 2), arm=traj.side, when="after the carry", place=state)
                check["counter_after_carry"] = state
                if state == "object":
                    return await missed("after the carry the object is still at its place on the counter (depth image): "
                                        "the hand closed on nothing. The arm is back home.")
            if outcome == "completed":
                bin_ = bin_name_of(echo)
                p = await _decide(robot, "The head camera of a humanoid robot looks down at its own hands, a counter and three "
                                  "open bins (left, center, right as the robot sees them).", robot.obs.get("head_rgb"),
                                  f"The robot's {traj.side} hand is holding a small block. Is that hand directly above the opening "
                                  f"of the {bin_} bin, so that the block would fall into that bin if the hand opened now?")
                robot.trace.write("above_bin_check", t=round(robot.t, 2), arm=traj.side, bin=bin_, probability=p)
                if p is not None and p < ABOVE_BIN_MIN:
                    robot.holding, robot.held_id = traj.side, obj.id
                    if decide is not None:
                        decide.cancel()
                    _needs_control(robot, f"above-bin check: the {traj.side} hand holds {obj.id} but does not look to be above the "
                                          f"{bin_} bin ({p:.2f}); the hand did NOT open. Look at the views and decide: place (it "
                                          f"releases into the bin from where the hand is), or place_on_counter", traj.side)
                    return _result("check_failed", reason=f"the hand did not open: it does not look to be above the {bin_} bin "
                                   f"(Decisions API {p:.2f}). The {traj.side} hand still holds {obj.id}.", arm=traj.side,
                                   above_bin_probability=round(p, 2), **echo)
                echo["above_bin_probability"] = None if p is None else round(p, 2)
                outcome = await _run(robot, cut(open_start, mid), f"{label}: release", robot.command_state(), settle=False)
                if outcome == "completed":        # asked while the arm goes home: costs no episode time
                    landed = asyncio.create_task(_decide(
                        robot, "The head camera of a humanoid robot looks down at a counter and three open bins (left, center, "
                        "right as the robot sees them).", np.array(robot.obs.get("head_rgb")),
                        f"The robot's {traj.side} hand has just released a small block above the {bin_} bin. Is that block "
                        f"inside the {bin_} bin now (not on the counter, not on the bin's edge)?"))
        elif mid > first_end:
            outcome = await _run(robot, cut(first_end, mid), f"{label}: carry and release", robot.command_state(), settle=False)
        if decide is not None:
            try:
                stay, why = await decide
            except Exception as e:  # noqa: BLE001
                stay, why = None, f"decision failed: {e!r}"[:120]
        if outcome == "completed" and mid < traj.ticks and stay is None:
            outcome = await _run(robot, cut(mid, traj.ticks), f"{label}: back home", robot.command_state(), settle=False)
    if outcome != "completed":
        robot.holding = robot.held_id = None
        return _result("aborted", reason=outcome, **echo)
    if unverified and not bin_name_of(echo):
        # The move ends with the object (perhaps) in the hand: parking now would open that hand. drop_in asks the
        # counter once the hand has left the place.
        robot.unverified = {"id": obj.id, "xy": was[:2].copy(), "side": traj.side}
        check["note"] = "not verified yet: drop_in checks the counter after the carry and reports missed if the object never left"
    elif unverified:
        await _park(robot)
        try:
            await asyncio.to_thread(_detect, robot)
            still = [o for o in robot.objects.values() if np.linalg.norm(o.xyz[:2] - was[:2]) < STILL_THERE_M]
        except Exception:  # noqa: BLE001
            still = []
        robot.trace.write("table_check", t=round(robot.t, 2), arm=traj.side, when="after the move",
                          place="object" if still else "empty")
        if still:
            return await missed("after the move the object is still on the counter where it stood: it was never in the "
                                "hand (nothing was placed).", arm_is_back=True)
        check.update(decided_by=check["decided_by"] + ": the object has left its place", unverified=False)
    on_done(traj)
    out = _result("completed", arm=traj.side, seconds=round(seconds, 1), **echo)
    if check is not None:
        out["grasp_check"] = check
    try:
        p_in = await landed if landed is not None else None
    except Exception:  # noqa: BLE001
        p_in = None
    if p_in is not None:
        out["in_bin_probability"] = round(p_in, 2)
        robot.trace.write("in_bin_check", t=round(robot.t, 2), arm=traj.side, bin=bin_name_of(echo), probability=p_in)
        if p_in < IN_BIN_MIN:
            _needs_control(robot, f"in-bin check: {obj.id} does not look to be inside the {bin_name_of(echo)} bin after the "
                                  f"release ({p_in:.2f}); look again and pick it up if it lies on the counter")
    if stay is not None:
        # The arm waits above the bin for its next object instead of going home and out again. No new look (the
        # arm is in the camera's view): the list is the last one, without the object just placed.
        robot.waiting = traj.side
        robot.objects.pop(obj.id, None)
        robot.seen = [o for o in robot.seen if o["id"] != obj.id]
        out["arm_waiting"] = {"arm": traj.side, "next_object": stay, "chosen_by": why,
                              "note": f"the {traj.side} arm is waiting above the bin: pick_and_place {stay} next "
                                      "saves about 3 s. Any other call works too."}
        out["objects_on_counter"] = robot.seen
        robot.trace.write("arm_waiting", t=round(robot.t, 2), arm=traj.side, next_object=stay, chosen_by=why)
        return out
    if home_start is not None:
        robot.trace.write("arm_home", t=round(robot.t, 2), arm=traj.side, why=why)
    await _park(robot)
    if look_after and not robot.holding:
        try:
            out["objects_on_counter"] = await asyncio.to_thread(_detect, robot)
        except Exception as e:  # noqa: BLE001
            out["objects_on_counter"] = f"look failed: {e!r}"[:200]
    return out


def _options_left(robot, seen):
    """Grasp options not yet missed on a detected object, for either arm (what a retry would use)."""
    from . import grasps
    obj = _target(robot, seen.id, None, None, None)
    return [o for o in grasps.options(obj, ("left", "right")) if o not in robot.failed.get(seen.id, [])]


def bin_name_of(echo):
    return echo.get("bin")


def _pick_and_place_call(ctx, bin_name, drop):
    from agent.skills import pick_and_place as skill
    return lambda side, a, h, target, approach: skill(ctx, target, bin_name, a, h, arm=side, approach=approach,
                                                        drop_offset=drop)


def _pick_call(ctx, for_bin=None):
    from agent.skills import pick as skill
    return lambda side, a, h, target, approach: skill(ctx, target, a, h, arm=side, approach=approach, for_bin=for_bin)


async def _do_pick_and_place(robot, obj, bin_name, arm=None, approach=None, drop=None, **echo):
    drop = _drop_point(robot, bin_name, drop)

    info = {}

    def plan(ctx, arm_q, hand_q):
        return _plan_grasp(robot, obj, arm, _pick_and_place_call(ctx, bin_name, drop), info, arm_q, hand_q,
                           bin_name, approach=approach)

    def placed(traj):
        robot.holding = robot.held_id = None
        robot.placed[bin_name] = robot.placed.get(bin_name, 0) + 1
        _er_memory(robot)["added"].pop(obj.id, None)          # it is in the bin: nothing to keep track of on the counter

    return await _grasp_move(f"pick_and_place {obj.id} -> {bin_name}", plan, placed, obj, True, object=obj.id,
                             bin=bin_name, grasp_at_m=[round(float(v), 3) for v in obj.xyz], grasp=info, **echo)


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
async def look(tool_context: ToolContext = None) -> dict:
    """Detect the objects that are on the counter now, with the head camera.

    Call this at the start and whenever you are unsure what is where. It gives each object an id you can pass to
    pick_and_place, pick and hand_over. An object keeps its id for the whole episode as long as it stays where
    it is. The arms should be at home (out of the camera's view) when you look. It finds anything with a clear
    colour that stands on the counter at object height; if it misses an object you can see, use point instead.

    Returns:
      objects_on_counter: one entry per object: id (says nothing about colour or shape), gemini_label (colour
        and shape as judged by a fast Gemini vision model that compared all the objects in the image, each with
        a confidence 0-1), rgb (the raw colour the camera measured, 0-255), side of the robot it lies on, position_m [x forward, y left] in metres from the robot's
        pelvis, point [y, x] in the head image (0..1000), and a rough guess of shape and rotation (often wrong:
        trust the image over it). Your next live view shows the head image with every id drawn on its object:
        decide each object's colour and shape from that image.
        probably_not_a_block (only when true): find_blocks was called and its model did not call this detection a
        block (a hand, a wrist, the edge of a bin). Do not pick it unless the image shows a block there.
      block_check: present when a detection is flagged that way: call find_blocks to have all of them checked.
      bins: where the three bins are, as position_m.
    """
    if os.environ.get("ROBOT_ER_LOOK", "0") == "1":     # for A/B runs: every look also asks the Robotics-ER model
        return await find_blocks.__wrapped__(tool_context=tool_context)
    robot = get_robot()
    await _ready(robot)
    await _park(robot)                      # arms out of the camera's view (an arm that holds something stays)
    out = await _look_result(robot)
    await _show(tool_context, "look.jpg", robot, {"camera": "head", "region": None, "mark_objects": True})
    return out


@_tool
async def find_blocks(tool_context: ToolContext = None) -> dict:
    """Ask the embodied-reasoning model (Gemini Robotics-ER) which things on the counter are blocks, when look
    seems wrong. It looks again as look does and checks every detection against that model's answer. It takes
    about 2 s (4 s at most), during which the robot stands still.

    look finds objects by height in the depth image alone, so it can report a hand, a wrist or the edge of a bin
    as an object, and it can miss a block that touches a bin, a hand or another block. Call this when look lists
    more objects than you can see blocks in the image, when an id sits on a hand or a bin edge, when you see a
    block that has no id, or when look's block_check names detections that may not be blocks. Not needed when
    look's list matches the image. Once per scene is enough; call it again only after the scene has changed in a
    way look does not explain.

    Returns (as look, with the same ids, positions and labels):
      objects_on_counter: the blocks: every detection the model confirmed (find_blocks "confirmed"; position
        still measured from depth) and every block it saw that look had missed (find_blocks "added": a new id,
        position from the depth image at the model's point, er_label = that model's colour and shape, which is
        less reliable than gemini_label). Use the ids with pick_and_place as usual.
      not_confirmed: detections the model did not call a block (probably_not_a_block true). They keep their ids
        and can still be picked: pick one only if the image clearly shows a block there. One with a note is a
        detection that look placed off its block, or two touching blocks that look saw as one object: use the
        added id or ids its note names instead.
      not_added: blocks the model pointed at that could not be given a position (cut by the image border, no
        depth, touching another block), with the reason and the point [y, x].
      find_blocks_failed: only if the model gave no usable answer; the result is then look's, unchanged.
    """
    from . import er
    robot = get_robot()
    await _ready(robot)
    await _park(robot)                      # arms out of the camera's view, as for look
    obs = robot.obs                         # the image the model gets and the depth image its points are located in
    asked = asyncio.create_task(asyncio.to_thread(er.point_blocks, obs["head_rgb"], robot.instruction))
    try:                                    # look's own detections and labels, while the model answers
        out = await _look_result(robot)
    finally:
        try:                                # a deadline for the whole call: the client's timeout is per socket read
            items = await asyncio.wait_for(asyncio.shield(asked), er.TIMEOUT_S + 1.0)
        except asyncio.TimeoutError:
            items = None                    # the worker thread ends by itself; never raises otherwise
    seconds, error = er.last["seconds"], er.last["error"]
    merged = None if items is None else await asyncio.to_thread(_merge_er, robot, items, obs)
    if merged is None:                      # fail open: no answer, or one that matches nothing look found
        if items is not None:
            error = f"the answer ({len(items)} points) confirmed none of look's {len(robot.objects)} detections and added nothing"
        robot.log(f"[tool] find_blocks: Robotics-ER call failed after {seconds} s: {error}"[:300])
        robot.trace.write("find_blocks", t=round(robot.t, 2), seconds=seconds, error=error, **({} if items is None else {"er_items": items}))
        out["find_blocks_failed"] = ("the embodied-reasoning model gave no usable answer; this is look's result, "
                                     "unchanged. Judge the detections from the image yourself.")
        out.pop("block_check", None)
        return out
    # the list again, now with the model's verdicts; a new block gets a colour and shape label like every other object
    objects = await _detect_and_label(robot)
    robot.trace.write("find_blocks", t=round(robot.t, 2), seconds=seconds, er_items=items, **merged)
    robot.log(f"[tool] find_blocks ({seconds} s): {len(items)} blocks seen; confirmed {merged['confirmed']}, "
              f"added {merged['added']}, not confirmed {merged['not_confirmed']}, not added {len(merged['not_added'])}")
    out["objects_on_counter"] = [o for o in objects if o.get("find_blocks") != "not_confirmed"]
    out["not_confirmed"] = [o for o in objects if o.get("find_blocks") == "not_confirmed"]
    if merged["not_added"]:
        out["not_added"] = merged["not_added"]
    out.pop("block_check", None)
    await _show(tool_context, "find_blocks.jpg", robot, {"camera": "head", "region": None, "mark_objects": True})
    return out


@_tool
async def view_camera(camera: Camera = "head", region: Optional[list[float]] = None, mark_objects: bool = False,
                      tool_context: ToolContext = None) -> dict:
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
    await _show(tool_context, f"view_{camera}.jpg", robot, robot.view_requests[-1])
    h, w = robot.obs[key].shape[:2]
    return _result("ok", camera=camera, image="attached to your next live view", full_resolution=[int(h), int(w)])


@_tool
async def ask_gemini(
    question: str,
    kind: Literal["yes_no", "choice", "score"] = "yes_no",
    options: Optional[list[str]] = None,
    camera: Camera = "head",
    region: Optional[list[float]] = None,
    mark_objects: bool = False,
) -> dict:
    """Ask a fast Gemini vision model one closed question about what a camera sees now (about a second).

    Use it to settle something visual without spending a full turn of your own: "Is there a block inside the
    left bin?", "Is the right hand holding a block?" (camera right_wrist), "Which of these colours is the block
    labelled obj_3?" (mark_objects true), "How far is obj_2 from the bin rim?" (score).

    Args:
      question: The question. Refer to objects by their ids only if mark_objects is true.
      kind: yes_no, choice (give options) or score (give options as levels, lowest first).
      options: The answers to choose from, for choice and score. Add a catch-all such as "none of these".
      camera: head, left_wrist or right_wrist (head_depth is not useful here).
      region: [ymin, xmin, ymax, xmax], each 0..1000, to ask about part of the image only.
      mark_objects: Draw the object ids from the latest look on the image (head camera only).

    Returns:
      yes_no: answer (true/false) and probability_yes. choice: answer and probabilities. score: score (the
      expected level index) and probabilities. Always a confidence 0-1: below about 0.7, do not rely on it.
    """
    from . import decisions
    from .agent import _render
    import cv2
    robot = get_robot()
    await _ready(robot)
    if not decisions.available():
        raise ToolError("the Decisions API is not available (no key, or ROBOT_DECISIONS=0)")
    if kind != "yes_no" and not options:
        raise ToolError("give options for kind choice or score")
    key = {"head": "head_rgb", "left_wrist": "left_wrist_rgb", "right_wrist": "right_wrist_rgb"}.get(camera)
    if key is None or robot.obs.get(key) is None:
        raise ToolError(f"the {camera} camera cannot be used here")
    shown = cv2.imdecode(np.frombuffer(_render(robot, {"camera": camera, "region": region, "mark_objects": mark_objects}),
                                       np.uint8), cv2.IMREAD_COLOR)[..., ::-1]
    text = (f"This is the {camera.replace('_', ' ')} camera of a humanoid robot standing at a counter with small "
            "coloured blocks and three orange bins (left, center, right from the robot's point of view)."
            + (" Blocks on the counter have a white circle and an id (obj_1, ...) drawn on them." if mark_objects else ""))
    try:
        a = await asyncio.to_thread(decisions.ask, text, [decisions.image_block(shown)], question,
                                    {"yes_no": "noul", "choice": "choice", "score": "score"}[kind], options)
    except Exception as e:  # noqa: BLE001
        raise ToolError(f"Decisions API call failed: {e!r}"[:300]) from None
    out = {"confidence": round(float(a.get("confidence", 0)), 2), "latency_s": a.get("latency_s")}
    if kind == "yes_no":
        out.update(answer=bool(float(a.get("noul", 0.5)) >= 0.5), probability_yes=round(float(a.get("noul", 0.5)), 2))
    elif kind == "choice":
        out.update(answer=a.get("choice"), probabilities={k: round(float(v), 2) for k, v in (a.get("probabilities") or {}).items()})
    else:
        out.update(score=a.get("score"), probabilities=a.get("probabilities"))
    robot.trace.write("decision_ask", t=round(robot.t, 2), question=question, kind=kind, options=options, answer=out)
    return _result("ok", question=question, **out)


# Where a hand goes for calibrate: [arm, hand position (x forward, y left, height above the counter) m, palm down?].
# Reaching across to the middle puts the forearm in the middle of the head image (1200-1500 model points in view; the
# poses of a normal pick show 30-280). 15 cm up clears the objects and the bin rims even if the table is 5 cm higher
# than the scene file says.
CAL_POSES = [("left", (0.50, 0.0, 0.15), True), ("right", (0.50, 0.0, 0.15), True),
             ("left", (0.50, 0.06, 0.18), False), ("right", (0.50, -0.06, 0.18), False)]
CAL_HOLD_S = 1.5


def _cal_plan(robot, ctx, arm_q, hand_q, side, xyz, top_down):
    """One arm to a calibration pose, the other aside, then hold still. None if not reachable or not safe."""
    from agent.skills.motion import Trajectory
    if robot.aside is None:
        return None
    to_a, to_h = np.asarray(robot.aside, float).copy(), np.asarray(ctx.post["open"], float).copy()
    with ctx.lock:
        sol = ctx.kin.solve(side, np.array([xyz[0], xyz[1], ctx.scene.counter_z + xyz[2]]), to_a, top_down=top_down)
        if not sol[1]:
            return None
        to_a = np.asarray(sol[0], float)
        a, h = ctx.planner.go_to(arm_q, hand_q, to_a, to_h)
        if not len(a) or ctx.planner.dry_run_safety(arm_q, hand_q, a, h):
            return None
    if _min_arm_distance(ctx, a, h) < ARM_CLEARANCE_M:
        return None
    n = int(round(CAL_HOLD_S * HZ))
    a, h = np.vstack([a, np.repeat(a[-1:], n, 0)]), np.vstack([h, np.repeat(h[-1:], n, 0)])
    return Trajectory(True, "calibrate", {}, a, h, [("calibrate", 0, len(a))])


@_tool
async def calibrate() -> dict:
    """Measure the scene against the robot's own arms and use the result for every later move.

    From the depth images of its own arms and the joint angles it works out where the camera really is, and from
    that how the table and bins differ from the scene file: higher or lower, shifted, tilted (a robot that leans).
    What is larger than the measurement's own error is then corrected: a tilt goes into the robot model (upper-body
    pitch and roll), a shift moves the scene; grasp heights, object positions and drop points all follow. If the
    camera has not yet seen the arms standing still in three poses, each arm first reaches into the camera's view
    and holds still (about 15 s in all).

    Call it when look reports a camera_check warning, or when grasps keep missing in the same way (closing above or
    below the objects, or always off to one side). Once is enough; the result is kept. Put down anything you hold
    first. Afterwards call look: every position from before is void.

    Returns:
      measurements: each measurement made (table tilt, height and shift against the scene file, how well the arm
        model fits the depth points) and what it corrected;
      upper_body_pitch_roll_deg, scene_offset_mm [x forward, y left, z up]: the corrections now in force.
    """
    robot = get_robot()
    await _ready(robot)
    if robot.holding:
        return _result("failed", reason="a hand is holding an object: put it down first")
    # First with what the camera has already seen of the arms (on the real robot the hands are in view from the
    # start): a rough correction, so that the calibration poses below are planned in a frame that is about right.
    first = await asyncio.to_thread(_measure_and_apply, robot)
    held = 0
    if first is None:                 # the arms have not been seen standing still: bring them into view
        await _park(robot)
        for side, xyz, top_down in CAL_POSES:
            start = robot.command_state()
            traj = await asyncio.to_thread(_cal_plan, robot, robot.ctx, *start, side, xyz, top_down)
            if traj is None:
                continue
            await _run(robot, traj, f"calibrate: {side} arm in view", start, settle=False)
            held += 1
        await _park(robot)
        second = await asyncio.to_thread(_measure_and_apply, robot)
    else:
        second = None
    done = [d for d in (first, second) if d is not None]
    if not done:
        robot.trace.write("calibrate", t=round(robot.t, 2), ok=False, poses=held)
        return _result("failed", reason="could not measure the camera against the arms. Nothing was changed.", poses_held=held)
    robot.calibrated = True
    changed = any(d["tilt_applied_deg"] != [0.0, 0.0] or any(d["offset_applied_mm"]) for d in done)
    out = dict(measurements=done, poses_held=held,
               upper_body_pitch_roll_deg=[round(float(np.degrees(v)), 1) for v in robot.waist],
               scene_offset_mm=[int(v) for v in np.round(robot.scene_offset * 1000)],
               note=("the robot model and the scene were corrected by what was measured: call look again" if changed else
                     "the files are right within this measurement's error: nothing was changed"))
    robot.trace.write("calibrate", t=round(robot.t, 2), ok=True, **out)
    robot.log(f"[tool] calibrate: upper body {out['upper_body_pitch_roll_deg']} deg, scene offset {out['scene_offset_mm']} mm")
    return _result("completed", **out)


CAL_MOMENTS = 3            # calibrate skips its own arm motion if the camera has already seen this many still arm poses
TILT_MIN_DEG = 2.5         # a table tilt below this is within the measurement's error (nominal sim reads 0.4-1.9 deg)


def _measure_and_apply(robot):
    """Measure the camera against the arms and the scene against the camera; put a tilt of the table into the robot
    model (upper-body pitch / roll), measure again in that frame, and move the scene by what is left. None if the
    camera could not be measured against the arms."""
    from . import selfcal
    est, arm, why = robot.measure_scene()
    if est is None or est.cam_from != "arms":
        return None
    out = dict(moments=arm.frames, fitted=robot.arm_fit_model, points_on_arm_pct=round(100 * arm.matched),
               rms_mm=round(arm.rms_m * 1000, 1), table_tilt_deg=round(est.tilt_deg, 1), tilt_applied_deg=[0.0, 0.0])
    if est.tilt_deg > TILT_MIN_DEG:
        pitch, roll = selfcal.level_angles(est.S)
        robot.set_waist(robot.waist[0] + pitch, robot.waist[1] + roll)
        out["tilt_applied_deg"] = [round(float(np.degrees(pitch)), 1), round(float(np.degrees(roll)), 1)]
        est2, arm2, _ = robot.measure_scene()
        if est2 is not None and est2.cam_from == "arms":
            est = est2
            out["table_tilt_after_deg"] = round(est.tilt_deg, 1)
    t = robot.apply_scene(est, robot.last_fit_T, set_camera=out["tilt_applied_deg"] != [0.0, 0.0])
    out.update(table_height_vs_file_mm=round(est.table_dz_m * 1000),
               shift_vs_file_mm=[round(est.shift_xy_m[0] * 1000), round(est.shift_xy_m[1] * 1000)],
               uncertainty_mm=round(max(est.sigma_dz_m, est.sigma_xy_m) * 1000), offset_applied_mm=[int(v) for v in np.round(t * 1000)])
    return out


@_tool
async def measure_scene() -> dict:
    """Measure where the camera and the table really are, instead of trusting the configuration files.

    It calibrates the head camera against the robot's own arms (the depth image shows the arm, the joint angles say
    where it is), measures the table and the bins relative to the camera, and combines both with their uncertainty.
    It reads only: it changes nothing about how the robot moves. It needs a few moments at which an arm stood still
    in the camera's view, so it works after two or three picks, not at the start; it takes a few seconds.

    Call it at most once or twice an episode, when grasps keep missing in the same direction or closing above or
    below the objects, to find out whether the table is higher or lower, shifted, or tilted (a leaning robot).

    Returns:
      table_height_vs_file_mm, shift_vs_file_mm [x forward, y left], tilt_deg, yaw_deg: how the real scene differs
        from the scene file, each with its uncertainty; camera_measured_from ("arms", or "calibration file only" when
        no arm has been seen standing still yet: then the numbers are only as good as the files);
      verdict: whether the difference is larger than this measurement can tell.
    """
    robot = get_robot()
    await _ready(robot)
    est, arm, why = await asyncio.to_thread(robot.measure_scene)
    if est is None:
        return _result("failed", reason=why)
    # In the nominal sim, where the scene is exactly the file's, this measurement reads about +8 mm, (+6, +5) mm and
    # 1.7 deg (two episodes, repeatable to 1-2 mm; cause not found). Anything inside that is not a finding.
    big = (abs(est.table_dz_m) > 0.015 or max(abs(est.shift_xy_m[0]), abs(est.shift_xy_m[1])) > 0.015 or est.tilt_deg > 2.5)
    measured = est.cam_from == "arms"
    out = dict(camera_measured_from=est.cam_from,
               table_height_vs_file_mm=round(est.table_dz_m * 1000), table_height_uncertainty_mm=round(est.sigma_dz_m * 1000),
               shift_vs_file_mm=[round(est.shift_xy_m[0] * 1000), round(est.shift_xy_m[1] * 1000)],
               shift_uncertainty_mm=round(est.sigma_xy_m * 1000),
               tilt_deg=round(est.tilt_deg, 1), tilt_uncertainty_deg=round(est.sigma_tilt_deg, 1), yaw_deg=round(est.yaw_deg, 1),
               verdict=("not measured: " + why + ". These numbers only repeat the files." if not measured else
                        "the scene differs from the scene file by more than this measurement's error: grasps planned "
                        "from the file will be off by about that much" if big else
                        "the scene agrees with the scene file within this measurement's error (about 15 mm, 2.5 deg)"),
               note="report only: the robot's motions still use the scene file")
    if arm is not None and arm.ok:
        out["arm_fit"] = {"moments_used": arm.frames, "depth_points_on_the_arm_pct": round(100 * arm.matched),
                          "rms_mm": round(arm.rms_m * 1000, 1)}
    robot.trace.write("scene_measured", t=round(robot.t, 2), **out)
    robot.log(f"[tool] measure_scene: {est.summary()}" + (f" ({why})" if why else ""))
    return _result("ok", **out)


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
    max_tilt_deg: Optional[float] = None,
    pregrasp_offset_m: Optional[list[float]] = None,
    grasp_height_frac: Optional[float] = None,
) -> dict:
    """Pick one object from the counter with a top-down pinch, drop it in a bin and bring the arm home (about 10 s).

    Say which object with object_id (from look; the more precise way) or with point.

    Args:
      bin: The bin to drop it in.
      object_id: Id of the object from the latest look or objects_on_counter list.
      point: [y, x] of the object in the HEAD camera image, each 0..1000; point at the middle of its top face.
      arm: Which arm to use. Leave out to let the planner choose. The left bin can only be reached by the left
        arm and the right bin only by the right arm.
      shape: Leave out normally: the tool measures shape and rotation from the depth image and turns the hand to
        the flat faces of cubes and triangular prisms by itself. Pass it only to overrule a wrong shape_guess.
      yaw_deg: Rotation of the object about the vertical axis in degrees (0 = an edge pointing away from the
        robot). Only used with shape cube or triangle.
      finger_direction_deg: Where the fingers point: 0 = away from the robot, +90 = to its left.
      xy_offset_m: [dx, dy] metres added to the grasp position ([x forward, y left]). After a miss that pushed
        the object, shift the grasp the way the object moved (object_moved_m), by about 0.015.
      drop_offset_m: [dx, dy] metres from the bin's centre. Leave out: the drop points are spread automatically.
      max_tilt_deg: How far the palm may tilt from facing straight down, 10 to 60 (default 35). More tilt lets the
        hand reach an object next to a bin wall or near the robot's body.
      pregrasp_offset_m: [dx, dy] metres: start the descent off to one side and come down at a slant, e.g.
        [-0.03, 0] to come from the robot's side when a bin rim is in the way behind the object.
      grasp_height_frac: Where on the object the fingers close, 0.2 (low) to 0.9 (near the top); default 0.5.

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
    approach = _approach(finger_direction_deg, xy_offset_m, None, max_tilt_deg, pregrasp_offset_m, grasp_height_frac)
    return await _do_pick_and_place(robot, obj, bin, arm, approach, _pair(drop_offset_m, "drop_offset_m"))


@_tool
async def choose_arm(bin: Optional[Bin] = None, object_id: Optional[str] = None, point: Optional[list[float]] = None) -> dict:
    """Which arm should take this object? Measures and plans, moves nothing (1 to 4 s).

    For each arm: how far its shoulder is from the object, whether it can reach the bin, whether a grasp can be
    planned, and whether the whole pick-and-place can. pick_and_place uses the same rule by itself when you name
    no arm; call this when a move failed and you want to know what each arm can do before trying again.

    Args:
      bin: The bin the object is meant for, if you know it.
      object_id: Id of the object from the latest look or objects_on_counter list.
      point: [y, x] of the object in the HEAD camera image, each 0..1000.

    Returns:
      nearest_arm, and for left and right: distance_to_object_m, reaches_bin, can_grasp, can_pick_and_place and
      the planner's reason when it cannot. recommendation says what to call next.
    """
    from . import arms
    robot = get_robot()
    await _ready(robot)
    obj = _target(robot, object_id, point, None, None)

    def work(ctx, arm_q, hand_q):
        d = arms.distances(ctx, obj.xyz, arm_q)
        out = {}
        for side in arms.SIDES:
            e = {"distance_to_object_m": round(d[side], 3)}
            g = _plan_grasp(robot, obj, side, _pick_call(ctx, bin), {}, arm_q, hand_q, bin, via_home=False)
            e["can_grasp"] = bool(g.ok)
            if not g.ok:
                e["grasp_problem"] = g.reason[:220]
            if bin:
                e["reaches_bin"] = arms.reaches(ctx, side, bin)
                t = _plan_grasp(robot, obj, side, _pick_and_place_call(ctx, bin, _drop_point(robot, bin, None)),
                                {}, arm_q, hand_q, bin) if g.ok and e["reaches_bin"] else None
                ok = bool(t is not None and t.ok)
                if ok and ARM_CLEARANCE_M > 0 and _min_arm_distance(ctx, t.arm, t.hand) < ARM_CLEARANCE_M:
                    ok, e["pick_and_place_problem"] = False, "it would pass too close to the other arm"
                elif t is not None and not t.ok:
                    e["pick_and_place_problem"] = t.reason[:220]
                e["can_pick_and_place"] = ok
            out[side] = e
        return out

    info = await asyncio.to_thread(work, robot.ctx, *robot.command_state())
    nearest = min(arms.SIDES, key=lambda s_: info[s_]["distance_to_object_m"])
    full = [s_ for s_ in arms.SIDES if info[s_].get("can_pick_and_place")]
    graspers = [s_ for s_ in arms.SIDES if info[s_]["can_grasp"]]
    if full:
        best = nearest if nearest in full else full[0]
        advice = f"pick_and_place with arm='{best}'"
    elif bin and graspers:
        best = nearest if nearest in graspers else graspers[0]
        advice = (f"the {best} arm can grasp it but not finish: hand_over (it is set down near the middle for the "
                  f"other arm), then pick_and_place")
    elif graspers:
        advice = f"pick with arm='{nearest if nearest in graspers else graspers[0]}'"
    else:
        advice = ("no arm has a plan with the default grasp: try check_plan with max_tilt_deg 50, a pregrasp_offset_m "
                  "away from whatever is in the way, or another finger_direction_deg")
    return _result("ok", object=obj.id, bin=bin, nearest_arm=nearest, left=info["left"], right=info["right"],
                   recommendation=advice)


@_tool
async def check_plan(
    bin: Optional[Bin] = None,
    object_id: Optional[str] = None,
    point: Optional[list[float]] = None,
    arm: Optional[Arm] = None,
    finger_direction_deg: Optional[float] = None,
    xy_offset_m: Optional[list[float]] = None,
    max_tilt_deg: Optional[float] = None,
    pregrasp_offset_m: Optional[list[float]] = None,
    grasp_height_frac: Optional[float] = None,
) -> dict:
    """Would this grasp be possible? Plans it with every safety check, moves nothing (about a second).

    Use it after "no safe plan" to find a variation that works before spending a move: call it several times in
    one turn with different arguments (they are checked one after the other) and then run the one that is ok with
    pick_and_place, using the same arguments. The reason of a failed plan says what was in the way: a bin rim, the
    counter, the other arm, or the arm's reach.

    Args:
      bin: The bin to drop it in. Leave out to check the grasp alone.
      object_id: Id of the object from the latest look or objects_on_counter list.
      point: [y, x] of the object in the HEAD camera image, each 0..1000.
      arm: Which arm. Leave out to let the planner choose.
      finger_direction_deg: Where the fingers point: 0 = away from the robot, +90 = to its left.
      xy_offset_m: [dx, dy] metres added to the grasp position.
      max_tilt_deg: How far the palm may tilt from straight down, 10 to 60 (default 35).
      pregrasp_offset_m: [dx, dy] metres: come down at a slant from that side.
      grasp_height_frac: Where on the object the fingers close, 0.2 to 0.9 (default 0.5).
    """
    robot = get_robot()
    await _ready(robot)
    obj = _target(robot, object_id, point, None, None)
    approach = _approach(finger_direction_deg, xy_offset_m, None, max_tilt_deg, pregrasp_offset_m, grasp_height_frac)
    info = {}

    def plan(ctx, arm_q, hand_q):
        call = _pick_and_place_call(ctx, bin, _drop_point(robot, bin, None)) if bin else _pick_call(ctx)
        return _plan_grasp(robot, obj, arm, call, info, arm_q, hand_q, bin, approach=approach)

    traj = await asyncio.to_thread(plan, robot.ctx, *robot.command_state())
    info.pop("option", None)
    if not traj.ok:
        return _result("failed", reason=f"no safe plan: {traj.reason}"[:400], object=obj.id, bin=bin)
    close = await _too_close(robot, traj)
    if close:
        return _result("failed", reason=close, object=obj.id, bin=bin)
    return _result("ok", object=obj.id, bin=bin, arm=traj.side, seconds=round(traj.ticks / HZ, 1), grasp=info,
                   note="nothing moved: run it with pick_and_place and the same arguments")


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

    info = {}

    def plan(ctx, arm_q, hand_q):
        return _plan_grasp(robot, obj, arm, _pick_call(ctx, for_bin), info, arm_q, hand_q, for_bin, approach=approach)

    def picked(traj):
        robot.holding, robot.held_id = traj.side, object_id
        _er_memory(robot)["added"].pop(obj.id, None)          # in the hand now, no longer at the point ER gave

    return await _grasp_move(f"pick {obj.id}", plan, picked, obj, False, object=obj.id,
                             grasp_at_m=[round(float(v), 3) for v in obj.xyz], grasp=info)


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


# ---------------------------------------------------------------------------------------------- hand control by local IK
# The tools below move a hand from where it is with localik.py: the joint solution nearest to the present one, the
# hand turned as little as the target allows. They do not go through the organisers' planner, so they also work
# where it has no plan (a bin beyond the reach of a palm-down hand, a start pose it was not tuned for).
LOCAL_STEP_M = 0.02           # straight-line moves are solved every 2 cm
LOCAL_JOINT_STEP = 0.6        # rad between two such solutions: more is a jump to another arm configuration (the largest
                              # step seen on good paths was 0.45 rad per 2 cm, close past the body with the robot leaning)
GRAB_TURNS_DEG = (0.0, 45.0, -45.0, 90.0, -90.0)   # the hand turned about the vertical, tried in this order
GRAB_ORI_RAD = 0.15           # how strongly a grasp holds the palm down (localik.solve's ori_rad; smaller = stronger)
GRAB_MAX_TILT_DEG = 25.0      # as the organisers' planner: a pinch from above needs the palm about level
DROP_RAISE_M = (0.0, 0.04, 0.08)   # release this much higher when the bin is out of reach (1 cm of reach per 4 cm)
DROP_WALL_M = 0.008           # the block's near side clears the bin's front wall by at least this at the release


def _heights(ctx):
    """(counter z, object size, carry z, release z) as the organisers' planner sets them."""
    g, zc, H = ctx.cfg["grasp"], ctx.scene.counter_z, ctx.planner.height
    wall = float(ctx.scene.cfg["bins"]["wall_height"])
    return zc, H, zc + max(g["carry_height"], wall + H + 0.03), zc + wall + H / 2 + g["drop_above_rim"]


def _local_plan(ctx, side, arm_q, hand_q, steps, skill, args, near=(), floor=None):
    """A checked trajectory made of local-IK moves and hand postures. Call with ctx.lock held.
    steps: ("go", name, xyz, R_want, ori_rad, hand posture to reach on the way | None, straight line: bool)
           ("hand", name, posture, hold seconds)
    near: phases in which the fingers may come down to the counter (a grasp). floor: {phase: lowest z of the grasp
    point} (a carry must not dip into a bin wall). Checked like the organisers' plans: hand above the counter and the
    bin rims at every third tick, no safety-filter clips in a dry run."""
    from aaltonomous.baselines.classical.planner import ramp, time_scale
    from agent.skills.motion import Trajectory
    from . import localik
    kin, mo, g = ctx.kin, ctx.cfg["motion"], ctx.cfg["grasp"]
    sl = kin.sl(side)
    cur_a, cur_h = np.asarray(arm_q, float).copy(), np.asarray(hand_q, float).copy()
    parts_a, parts_h, phases, n, turned = [], [], [], 0, 0.0

    def fail(why):
        return Trajectory(False, skill, args, side=side, reason=why)

    for step in steps:
        if step[0] == "go":
            _, name, xyz, R, ori, hand_to, straight = step
            rs = (localik.path(kin, side, cur_a, [xyz], R, step_m=LOCAL_STEP_M, ori_rad=ori) if straight
                  else [localik.solve(kin, side, cur_a, xyz, R, ori_rad=ori)])
            if not rs or not rs[-1].ok:
                return fail(f"{name}: out of the {side} arm's reach" + (f" by {rs[-1].pos_err_m * 1000:.0f} mm" if rs else ""))
            Qs = [cur_a] + [r.arm_q for r in rs]
            if straight and float(np.abs(np.diff(Qs, axis=0)).max()) > LOCAL_JOINT_STEP:
                return fail(f"{name}: the arm would have to swing round on the way")
            turned = max(turned, rs[-1].turned_deg)
            a = time_scale(Qs, mo["arm_max_vel"], mo["arm_max_acc"], ctx.dt)
            h = np.repeat(cur_h[None], len(a), 0)
            if hand_to is not None:                          # the hand changes posture during the arm move
                to = cur_h.copy()
                to[sl] = ctx.post[hand_to][sl]
                hr = ramp(cur_h, to, mo["hand_max_vel"], ctx.dt)
                if len(hr) > len(a):
                    a = np.vstack([a.reshape(-1, 14), np.repeat((a[-1] if len(a) else cur_a)[None], len(hr) - len(a), 0)])
                    h = np.repeat(cur_h[None], len(a), 0)
                h[:len(hr)] = hr
                h[len(hr):] = to
                cur_h = to
        else:
            _, name, posture, hold_s = step
            to = cur_h.copy()
            to[sl] = ctx.post[posture][sl]
            h = np.vstack([ramp(cur_h, to, mo["hand_max_vel"], ctx.dt).reshape(-1, 14),
                           np.repeat(to[None], max(1, int(round(hold_s / ctx.dt))), 0)])
            a, cur_h = np.repeat(cur_a[None], len(h), 0), to
        if len(a):
            parts_a.append(a)
            parts_h.append(h)
            phases.append((name, n, n + len(a)))
            n += len(a)
            cur_a = a[-1].copy()
    if not parts_a:
        return fail("the hand is already there")
    arm, hand = np.vstack(parts_a), np.vstack(parts_h)
    for name, s, e in phases:
        need = 0.003 if name == "close" else g["fingertip_clearance"] if name in near else mo["counter_clearance"]
        for k in range(s, e, 3):
            c = ctx.planner.clearance(side, arm[k], hand[k])
            if c < need - 2e-3:
                return fail(f"the hand would be {100 * c:.1f} cm above the counter during {name} (need {100 * need:.1f})")
            if name != "open" and ctx.planner.bin_clearance(side, arm[k], hand[k]) < 0.003:
                return fail(f"the hand would touch a bin rim during {name}")
            if floor and name in floor and kin.grasp_pose(side, arm[k])[0][2] < floor[name]:
                return fail(f"the hand would dip below the bin rim during {name}")
    clips = ctx.planner.dry_run_safety(arm_q, hand_q, arm, hand)
    if clips:
        return fail(f"{clips} safety clips in dry run")
    return Trajectory(True, skill, {**args, "hand_turned_deg": round(turned, 1)}, arm, hand, phases, side=side)


def _grab_plan(ctx, side, arm_q, hand_q, obj, finger_deg, missed=()):
    """Local-IK grasp of one object with one arm: over it, down, close, lift to carry height. ctx.lock held.
    The fingers keep the direction they have (a cube: the nearest direction square to its faces); when the hand
    cannot come down like that (a bin rim behind the object, the counter, the reach) the hand is turned in steps.
    missed: the grasps.Option(side, "local", turn) that already closed on nothing on this object: not tried again."""
    from agent.skills.motion import Trajectory
    from . import grasps, localik
    kin, g = ctx.kin, ctx.cfg["grasp"]
    zc, H, z_carry, _ = _heights(ctx)
    sl = kin.sl(side)
    xy = np.asarray(obj.xyz[:2], float)
    cube = finger_deg is None and obj.shape == "cube" and obj.yaw is not None
    R0 = localik.palm_down(kin, side, arm_q, float(np.degrees(obj.yaw)) if cube else finger_deg, 4 if cube else 1)
    turns = (0.0,) if finger_deg is not None else (0.0, 90.0, -90.0) if cube else GRAB_TURNS_DEG
    pre = np.asarray(hand_q, float).copy()
    pre[sl] = ctx.post["preshape"][sl]
    why = []
    for turn in turns:
        if finger_deg is None and grasps.Option(side, "local", turn) in missed:
            why.append(f"fingers turned {turn:+.0f} deg already missed")
            continue
        R = localik._about(np.array([0.0, 0.0, 1.0]), np.radians(turn)) @ R0
        z = zc + g["height_frac"] * H                          # mid-height, raised until the open fingers clear the counter
        for _ in range(4):
            r = localik.solve(kin, side, arm_q, np.r_[xy, z], R, ori_rad=GRAB_ORI_RAD)
            c = ctx.planner.clearance(side, r.arm_q, pre, fingers_only=True) if r.ok else 0.0
            if not r.ok or c >= g["fingertip_clearance"] - 1e-3:
                break
            z += g["fingertip_clearance"] - c
        tilt = localik.tilt_deg(kin, side, r.arm_q)
        if not r.ok:
            why.append(f"the object is out of reach by {r.pos_err_m * 1000:.0f} mm")
            continue
        if c < g["fingertip_clearance"] - 1e-3 or tilt > GRAB_MAX_TILT_DEG or z > zc + H - 0.008:
            why.append(f"no pinch from above there (palm tilted {tilt:.0f} deg, fingers {100 * c:.1f} cm above the counter)")
            continue
        hover = np.r_[xy, max(z + g["approach_height"], z_carry)]
        args = {"object": obj.id, "grasp_height_m": round(float(z - zc), 3), "palm_tilt_deg": round(tilt, 1),
                "fingers_turned_deg": turn}
        for straight in (False, True):                         # the short way in joint space first, else a straight line
            t = _local_plan(ctx, side, arm_q, hand_q,
                            [("go", "to_hover", hover, R, GRAB_ORI_RAD, "preshape", straight),
                             ("go", "descend", np.r_[xy, z], R, GRAB_ORI_RAD, None, True),
                             ("hand", "close", "closed", ctx.cfg["motion"]["close_hold_s"]),
                             ("go", "lift", np.r_[xy, z_carry], R, GRAB_ORI_RAD, None, True)],
                            "grab", args, near=("descend", "close", "lift"))
            if t.ok:
                return t
            if not straight:
                why.append(t.reason)
    return Trajectory(False, "grab", {}, side=side, reason="; ".join(dict.fromkeys(why))[:300])


def _drop_plan(ctx, side, arm_q, hand_q, bin_name, drop):
    """Local-IK release of the held object into a bin: carry over it, lower, open, back up. ctx.lock held.
    The wanted release point is the planner's (inside the bin's front wall, spread sideways). When the arm cannot
    get there the release is raised, then brought nearer to the front wall, as long as the block still clears it."""
    from agent.skills.motion import Trajectory
    from . import localik
    kin, sc, mo = ctx.kin, ctx.scene, ctx.cfg["motion"]
    zc, H, z_carry, z_drop = _heights(ctx)
    ctx.planner.drop_offsets = {bin_name: tuple(drop)}
    try:
        want = np.asarray(ctx.planner.drop_xy(bin_name), float)
    finally:
        ctx.planner.drop_offsets = {}
    lo, hi = sc.bin_inner_box(bin_name)
    wall_x = min(sc.mat_to_pelvis(np.r_[lo[:2], 0.0])[0], sc.mat_to_pelvis(np.r_[hi[:2], 0.0])[0])   # inside of the front wall
    nearest = wall_x + H / 2 + DROP_WALL_M
    R = localik.hand_pose(kin, side, arm_q)[1]                 # the hand stays as it is, as far as the reach allows
    last = None
    for back in np.arange(0.0, max(want[0] - nearest, 0.0) + 1e-9, 0.01):
        for up in DROP_RAISE_M:
            above = np.r_[want[0] - back, want[1], z_carry + up]
            if not localik.solve(kin, side, arm_q, above, R).ok:
                last = last or Trajectory(False, "drop_in", {}, side=side, reason=f"the {bin_name} bin is out of the {side} arm's reach")
                continue
            args = {"bin": bin_name, "release_at_m": [round(float(v), 3) for v in above[:2]],
                    "behind_front_wall_m": round(float(above[0] - wall_x), 3), "wanted_m": [round(float(v), 3) for v in want]}
            for lower in (True, False):                        # down to just above the rim if the hand can, else from carry height
                for straight in (False, True):
                    steps = [("go", "carry", above, R, 0.5, None, straight)]
                    if lower:
                        steps.append(("go", "lower", np.r_[above[:2], z_drop], R, 0.5, None, True))
                    steps += [("hand", "open", ctx.cfg["grasp"].get("release_posture", "release"), mo["open_hold_s"]),
                              ("go", "retreat", np.r_[above[:2], z_carry + up + (0.0 if lower else 0.03)], R, 0.5, None, True)]
                    t = _local_plan(ctx, side, arm_q, hand_q, steps, "drop_in",
                                    {**args, "release_height_above_rim_m": round(float((z_drop if lower else above[2]) - zc
                                                                                     - sc.cfg["bins"]["wall_height"]), 3)},
                                    floor={"carry": z_drop - 0.005})
                    if t.ok:
                        return t
                    last = t
    return last or Trajectory(False, "drop_in", {}, side=side, reason=f"the {bin_name} bin is out of the {side} arm's reach")


def _reaches_bin_local(ctx, side, bin_name, arm_q):
    """Can this arm release into the bin at all (the nearest point that still clears the front wall, raised as far
    as drop_in would raise it)? ctx.lock held."""
    from . import localik
    zc, H, z_carry, _ = _heights(ctx)
    lo, hi = ctx.scene.bin_inner_box(bin_name)
    a, b = ctx.scene.mat_to_pelvis(np.r_[lo[:2], 0.0]), ctx.scene.mat_to_pelvis(np.r_[hi[:2], 0.0])
    y = float(np.clip(0.0, min(a[1], b[1]) + 0.05, max(a[1], b[1]) - 0.05))       # the part of the bin nearest the midline
    target = np.r_[min(a[0], b[0]) + H / 2 + DROP_WALL_M, y, z_carry + DROP_RAISE_M[-1]]
    return localik.solve(ctx.kin, side, arm_q, target).ok


def _hand_at(robot, side):
    """Where a hand's grasp point is commanded: [x, y, height above the counter] in metres."""
    ctx = robot.ctx
    with ctx.lock:
        p = ctx.kin.grasp_pose(side, robot.command_state()[0])[0]
    return [round(float(p[0]), 3), round(float(p[1]), 3), round(float(p[2] - ctx.scene.counter_z), 3)]


@_tool
async def find_bins() -> dict:
    """Measure where the three bins are (about 2 s): a Gemini Robotics-ER model marks them in the head image and the
    depth image gives their front wall, width and rim height. Every later drop uses the result.

    look does this once by itself. Call it again when a release fell short of a bin or beside it, or when the bins
    may have been moved. The arms should be out of the camera's view (go_home).

    Returns:
      bins: for each bin its centre [x forward, y left] in metres, where a block is released, and for each arm
        whether it can reach that release point from above (reach_short_m: by how much it falls short).
    """
    from . import localik
    robot = get_robot()
    await _ready(robot)
    robot.bins_measured = None
    moved = await _measure_bins(robot)
    ctx = robot.ctx
    arm_q, _h = robot.command_state()
    out = {}
    with ctx.lock:
        _zc, _H, z_carry, _zd = _heights(ctx)
        for b in ctx.bins():
            drop = np.asarray(ctx.planner.drop_xy(b), float)
            reach = {}
            for side in ("left", "right"):
                r = localik.solve(ctx.kin, side, arm_q, np.r_[drop, z_carry])
                reach[side] = True if r.ok else {"reach_short_m": round(r.pos_err_m, 3)}
            out[b] = {"centre_m": [round(float(v), 3) for v in ctx.planner.bin_xy(b)],
                      "release_point_m": [round(float(v), 3) for v in drop], "arm_reaches": reach}
    return _result("ok", bins=out, rim_height_m=round(float(ctx.scene.cfg["bins"]["wall_height"]), 3),
                   measured_now=robot.bins_measured or "no bin could be measured: the positions are the ones from before",
                   changed=bool(moved))


@_tool
async def grab(object_id: Optional[str] = None, point: Optional[list[float]] = None, for_bin: Optional[Bin] = None,
               arm: Optional[Arm] = None, finger_direction_deg: Optional[float] = None) -> dict:
    """Pick one object with a pinch from above and hold it at carry height (about 5 s), moving the arm as little
    as possible: the hand goes from where it is now, turned only as far as needed to face the counter, to the
    joint solution nearest to the present one. Follow with drop_in.

    Use it when pick_and_place returned "failed: no safe plan" or has missed an object: this tool does not use that
    planner. Cubes and cylinders; a triangular prism needs pick_and_place's grip on its flat faces.

    Args:
      object_id: Id of the object from the latest look or objects_on_counter list.
      point: [y, x] of the object in the HEAD camera image, each 0..1000.
      for_bin: The bin it is going to. Give it: the arm that can reach that bin is then used (an arm cannot
        release into the bin on the far side of the body).
      arm: Which arm. Leave out: an arm that reaches for_bin first, the nearer one first.
      finger_direction_deg: Where the fingers point: 0 = away from the robot, +90 = to its left. Leave out: the
        fingers keep their present direction (for a cube: the nearest direction square to its faces).

    Returns:
      status: "completed" (the hand holds the object, or grasp_check says it could not be verified yet: drop_in
        then checks the counter), "missed" (the hand closed on nothing; the arm is back: call grab again, it turns
        the hand differently each time), or "failed" (nothing moved; reason says why).
    """
    from agent.skills.motion import Trajectory
    from . import arms, grasps
    robot = get_robot()
    await _ready(robot)
    if robot.holding:
        raise ToolError(f"the {robot.holding} hand is holding an object: drop_in or place it first")
    obj = _target(robot, object_id, point, None, None)
    if arm and for_bin:
        ctx = robot.ctx
        with ctx.lock:
            can = _reaches_bin_local(ctx, arm, for_bin, robot.command_state()[0])
        if not can:
            raise ToolError(f"the {arm} arm cannot release into the {for_bin} bin. Use the other arm; if the other arm cannot "
                            f"get at the object, grab it with the {arm} arm WITHOUT for_bin, place_on_counter near the "
                            "middle ([0.30, 0.0]), then grab it with the other arm.")
    info = {}

    def plan(ctx, arm_q, hand_q):
        sides = [arm] if arm else arms.order(ctx, obj.xyz, None, arm_q)
        why = []
        with ctx.lock:
            if for_bin and not arm:                            # stable sort: reaching arms first, each group nearest first
                sides = sorted(sides, key=lambda s_: not _reaches_bin_local(ctx, s_, for_bin, arm_q))
            for side in sides:
                t = _grab_plan(ctx, side, arm_q, hand_q, obj, finger_direction_deg, robot.failed.get(obj.id, ()))
                if t.ok:
                    info.update(arm=side, grip="local IK pinch from above", **t.args,
                                option=grasps.Option(side, "local", t.args["fingers_turned_deg"]))
                    return t
                why.append(f"{side} arm: {t.reason}")
        return Trajectory(False, "grab", {}, side=sides[0], reason="; ".join(why))

    def picked(traj):
        robot.holding, robot.held_id = traj.side, object_id
        _er_memory(robot)["added"].pop(obj.id, None)

    return await _grasp_move(f"grab {obj.id}", plan, picked, obj, False, object=obj.id,
                             grasp_at_m=[round(float(v), 3) for v in obj.xyz], grasp=info)


@_tool
async def drop_in(bin: Bin, drop_offset_m: Optional[list[float]] = None, release_anyway: bool = False) -> dict:
    """Carry the held object over a bin and release it (about 5 s), moving the arm as little as possible from where
    it is. Works where place has no plan: when the release point is out of reach it releases higher, or nearer to
    the bin's front wall, as long as the block still falls inside (the result says where it released).

    Before the hand opens a vision check asks whether the hand is above the bin; if it does not look so the hand
    stays closed and the status is "check_failed": look at the views, then call drop_in again with
    release_anyway=true if the hand is above the bin, or move_hand to correct it first.

    Args:
      bin: The bin to drop it in.
      drop_offset_m: [dx, dy] metres from the usual release point. Leave out: the releases are spread automatically.
      release_anyway: Open the hand even if the above-bin check doubts it.

    Returns:
      status: "completed", "missed" (the object never left the counter: the hand had closed on nothing), "failed"
        (nothing moved; the bin is out of this arm's reach or the path is blocked) or "check_failed".
    """
    from agent.skills.motion import Trajectory
    robot = get_robot()
    await _ready(robot)
    if not robot.holding:
        raise ToolError("nothing is held: grab or pick an object first")
    side, held = robot.holding, robot.held_id
    drop = _drop_point(robot, bin, _pair(drop_offset_m, "drop_offset_m"))
    start = robot.command_state()

    def plan(ctx, arm_q, hand_q):
        with ctx.lock:
            return _drop_plan(ctx, side, arm_q, hand_q, bin, drop)

    traj = await asyncio.to_thread(plan, robot.ctx, *start)
    if not traj.ok:
        return _result("failed", reason=f"no safe plan: {traj.reason}"[:400], bin=bin, arm=side)
    echo = {"bin": bin, "arm": side, **{k: v for k, v in traj.args.items() if k != "bin"}}
    if robot.time_left() is not None and traj.ticks / HZ > robot.time_left():
        return _result("failed", reason=f"not enough time left for a {traj.ticks / HZ:.0f} s move", **echo)
    close = await _too_close(robot, traj)
    if close:
        return _result("failed", reason=close, **echo)
    opens = next(s for n, s, e in traj.phases if n == "open")
    cut = lambda i, j: Trajectory(True, traj.skill, traj.args, traj.arm[i:j], traj.hand[i:j], side=side)  # noqa: E731
    robot.waiting = None
    outcome = await _run(robot, cut(0, opens), f"drop_in {bin}: carry", start, settle=True)
    doubt = robot.unverified if robot.unverified and robot.unverified.get("side") == side else None
    robot.unverified = None

    async def nothing_held(reason):
        robot.holding = robot.held_id = None
        await _park(robot)                                    # opens the hand and brings the arm back
        out = _result("missed", reason=reason, **echo)
        try:
            out["objects_on_counter"] = await asyncio.to_thread(_detect, robot)
        except Exception as e:  # noqa: BLE001
            out["objects_on_counter"] = f"look failed: {e!r}"[:200]
        return out

    if outcome == "completed" and doubt is not None:          # the grasp could not be verified at the lift: ask the counter
        state = await asyncio.to_thread(_spot, robot, doubt["xy"])
        robot.trace.write("table_check", t=round(robot.t, 2), arm=side, when="after the carry", place=state)
        echo["counter_after_carry"] = state
        if state == "object":
            return await nothing_held("after the carry the object is still at its place on the counter (depth image): the "
                                      "hand had closed on nothing. Nothing was released; the arm is back.")
    if outcome == "completed" and not release_anyway and robot.world is None:
        p = await _decide(robot, "The head camera of a humanoid robot looks down at its own hands, a counter and three open "
                          "bins (left, center, right as the robot sees them).", robot.obs.get("head_rgb"),
                          f"The robot's {side} hand is holding a small block. Is that hand directly above the opening of the "
                          f"{bin} bin, so that the block would fall into that bin if the hand opened now?")
        robot.trace.write("above_bin_check", t=round(robot.t, 2), arm=side, bin=bin, probability=p, tool="drop_in")
        echo["above_bin_probability"] = None if p is None else round(p, 2)
        if p is not None and p < ABOVE_BIN_MIN:
            _needs_control(robot, f"above-bin check: the {side} hand does not look to be above the {bin} bin ({p:.2f}); the "
                                  f"hand did NOT open. It is at {_hand_at(robot, side)} [x, y, height]; the bin's release "
                                  f"point is {echo.get('release_at_m')}", side)
            return _result("check_failed", reason=f"the hand did not open: it does not look to be above the {bin} bin "
                           f"(Decisions API {p:.2f}). The {side} hand still holds the object.", **echo)
    landed = None
    if outcome == "completed":
        outcome = await _run(robot, cut(opens, traj.ticks), f"drop_in {bin}: release", robot.command_state(), settle=False)
        if outcome == "completed" and robot.world is None:
            landed = asyncio.create_task(_decide(
                robot, "The head camera of a humanoid robot looks down at a counter and three open bins (left, center, right "
                "as the robot sees them).", np.array(robot.obs.get("head_rgb")),
                f"The robot's {side} hand has just released a small block above the {bin} bin. Is that block inside the "
                f"{bin} bin now (not on the counter, not on the bin's edge)?"))
    robot.holding = robot.held_id = None
    if outcome != "completed":
        return _result("aborted", reason=outcome, **echo)
    await _park(robot)
    out = _result("completed", **echo)
    try:
        out["objects_on_counter"] = await asyncio.to_thread(_detect, robot)
    except Exception as e:  # noqa: BLE001
        out["objects_on_counter"] = f"look failed: {e!r}"[:200]
    if doubt is not None and any(np.linalg.norm(o.xyz[:2] - doubt["xy"]) < STILL_THERE_M for o in robot.objects.values()):
        robot.trace.write("table_check", t=round(robot.t, 2), arm=side, when="after the move", place="object")
        out.update(status="missed", reason="after the move the object is still on the counter where it stood: it was never "
                                           "in the hand (nothing was placed).")
        robot.recent[-1:] = ["missed"]
        return out
    robot.objects.pop(held, None)
    robot.placed[bin] = robot.placed.get(bin, 0) + 1
    try:
        p_in = await landed if landed is not None else None
    except Exception:  # noqa: BLE001
        p_in = None
    if p_in is not None:
        out["in_bin_probability"] = round(p_in, 2)
        robot.trace.write("in_bin_check", t=round(robot.t, 2), arm=side, bin=bin, probability=p_in, tool="drop_in")
        if p_in < IN_BIN_MIN:
            _needs_control(robot, f"in-bin check: the object does not look to be inside the {bin} bin after the release "
                                  f"({p_in:.2f}); look again and pick it up if it lies on the counter")
    return out


@_tool
async def move_hand(arm: Arm, position_m: list[float], palm_down: bool = False) -> dict:
    """Move one hand to a position, as directly as possible (local IK: the joint solution nearest to the present
    one). The position is that of the hand's grasp point, between the finger tips and the thumb. The hand's fingers
    do not change. With grip this gives full manual control: e.g. to correct a hand that stopped short of a bin.

    The hand stays where you put it until the next call; go_home brings both arms back.

    Args:
      arm: Which hand.
      position_m: [x forward, y left, height above the counter] in metres from the robot's pelvis. Objects stand at
        heights 0 to 0.045; carry height is about 0.16; look and find_bins give x and y of objects and bins.
      palm_down: true = turn the hand so the palm faces the counter (needed before closing on an object from
        above); false = keep the hand's present orientation as far as the reach allows.

    Returns:
      hand_at_m: where the grasp point is afterwards. "failed" = nothing moved (out of reach, or the hand would
        touch the counter, a bin rim or the other arm); reason says which.
    """
    from . import localik
    if len(position_m) != 3:
        raise ToolError("position_m must be [x, y, height above the counter] in metres")
    x, y, up = (float(v) for v in position_m)
    if not (0.0 <= up <= 0.45 and 0.05 <= x <= 0.75 and abs(y) <= 0.6):
        raise ToolError("position_m is outside the workspace: x 0.05..0.75, y -0.6..0.6, height 0..0.45")

    def plan(ctx, arm_q, hand_q):
        with ctx.lock:
            R = localik.palm_down(ctx.kin, arm, arm_q) if palm_down else localik.hand_pose(ctx.kin, arm, arm_q)[1]
            target = np.array([x, y, ctx.scene.counter_z + up])
            for straight in (False, True):
                t = _local_plan(ctx, arm, arm_q, hand_q, [("go", "move_hand", target, R, GRAB_ORI_RAD if palm_down else 0.5,
                                                         None, straight)], "move_hand", {}, near=("move_hand",))
                if t.ok:
                    break
            return t

    out = await _move(f"move_hand {arm} -> [{x:.2f}, {y:.2f}, {up:.2f}]", plan)
    out["hand_at_m"] = _hand_at(get_robot(), arm)
    return out


@_tool
async def grip(arm: Arm, action: Literal["close", "open"]) -> dict:
    """Close one hand on what is between its fingers, or open it; the arm does not move (about 1 s).

    Args:
      arm: Which hand.
      action: "close" pinches (firmly on the real hands) and then checks whether something is held; "open" lets go:
        what the hand held falls where the hand is.

    Returns:
      grasp_check (after close): object_in_hand and what decided it. holding says what the robot now counts as held.
    """
    from agent.skills import set_hand
    robot = get_robot()
    await _ready(robot)
    ctx = robot.ctx
    start = robot.command_state()
    sl = ctx.kin.sl(arm)
    closing = action == "close"
    traj = await asyncio.to_thread(set_hand, ctx, *start, arm, "closed" if closing else
                                   ctx.cfg["grasp"].get("release_posture", "release"), 0.4)
    if not traj.ok:
        return _result("failed", reason=traj.reason, arm=arm, action=action)
    if closing:
        err = float(np.abs(np.asarray(robot.obs["hand_q"], float)[sl] - robot.hand_cmd[sl]).max())
        robot.finger_baseline[arm] = min(err, robot.finger_baseline.get(arm, err))
        if GRIP == "firm" or (GRIP == "auto" and robot.finger_baseline[arm] > GRASP_EMPTY_RAD):
            traj = _firm(ctx, traj)
    outcome = await _run(robot, traj, f"grip {action} ({arm})", start, settle=True)
    if outcome != "completed":
        return _result("aborted", reason=outcome, arm=arm, action=action)
    check = None
    if not closing:
        if robot.holding == arm:
            robot.holding = robot.held_id = None
    elif GRASP_CHECK and robot.world is None:
        check = await _grasp_check(robot, arm)
        robot.holding, robot.held_id = (arm, None) if check["object_in_hand"] else (None, None)
    else:
        robot.holding, robot.held_id = arm, None
    out = _result("completed", arm=arm, action=action, hand=_hand_summary(robot, arm), hand_at_m=_hand_at(robot, arm))
    if check is not None:
        out["grasp_check"] = check
    return out


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


TOOLS = [look, find_blocks, find_bins, view_camera, ask_gemini, read_proprioception, measure_scene, calibrate, choose_arm, check_plan, pick_and_place, pick, place, place_on_counter, hand_over,
         grab, drop_in, move_hand, grip, go_home, wait, done]
# ROBOT_TOOLS_OFF: tool names, comma separated, that the model does not get (for A/B runs only; e.g.
# "pick_and_place,pick,place,hand_over" leaves only the local-IK way of moving an object).
_OFF = {n.strip() for n in os.environ.get("ROBOT_TOOLS_OFF", "").split(",") if n.strip()}
TOOLS = [t for t in TOOLS if t.__name__ not in _OFF]
if not SELFCAL:
    TOOLS = [t for t in TOOLS if t not in (calibrate, measure_scene)]
if not ER:
    TOOLS = [t for t in TOOLS if t is not find_blocks]
