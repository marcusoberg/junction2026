"""Check `find_blocks` (the Gemini Robotics-ER model merged with the depth detections) and look's
probably_not_a_block flag on saved frames. No simulator and no motion; real API calls: per frame one Decisions
request (look's labels) and one Robotics-ER request, about 3 s a frame.

    PYTHONPATH=aaltonomous-humanoid:adk_agents:. .venv/bin/python sim/replay_er.py \
        sim_out/new-1855ff4 sim_out/t2-tri-nominal sim_out/t2-venue-fixed --real data_drops/drop3 [-v] [--out DIR]

Each frame is put in front of the robot as its current camera image (arms parked, so nothing moves) and the two
tools are called the way the agent calls them: tools.look(), then tools.find_blocks().

Sim runs (first-look frame of every episode; the true layout from eval.json is used for scoring only): objects
right / missed / extra in look's list and in find_blocks' list, how many real objects look flagged
probably_not_a_block and how many find_blocks did not confirm (both should be 0), position error of both lists.

Real frames (--real: frame 0 of every episode of a data drop, both hands in view; there is no truth): detections,
flagged, confirmed, added, not confirmed, next to the number of blocks the evaluation's Robotics-ER call returned
on the same frame (--reference: er_eval/ermeas/results_real.json, "points/minimal"; checked by eye there: every
visible block and no hand or bin, plus 0-1 slivers of blocks cut by the image edge per frame).

--out DIR writes one picture per frame: green = confirmed, cyan = added by find_blocks, red = not confirmed,
"?" = flagged by look, grey cross = pointed at by the model but not added.
"""
import argparse
import asyncio
import json
import os
import sys
from collections import Counter
from pathlib import Path

os.environ.pop("ROBOT_TRACE_DIR", None)                 # a replay writes nothing into a run's trace

import cv2
import numpy as np

from aaltonomous.scene.scene import Scene
from robot_sorter import er, tools
from robot_sorter.robot import get_robot

GATE = 0.04                                             # m: a listed object this close to a true one is that one (as replay_perception.py)
REAL_INSTRUCTION = ("Put the yellow objects in the left bin, the purple objects in the center bin and the orange "
                    "objects in the right bin.")       # the real objects' colours (the organisers' real mapping)


def show(robot, rgb, depth, instruction):
    """A saved frame becomes what the robot sees now; the arms are where look parks them, so no motion is planned."""
    ctx = robot.ctx
    robot.reset(instruction, time_limit=None)
    q = np.asarray(robot.aside if robot.aside is not None else ctx.home, float)
    robot.obs = {"head_rgb": np.ascontiguousarray(rgb), "head_depth": depth, "arm_q": q.copy(),
                 "hand_q": ctx.post["open"].copy(), "t": 0.0}
    robot.arm_cmd, robot.hand_cmd = q.copy(), ctx.post["open"].copy()
    robot._ready.set()


async def both_tools():
    first = await tools.look()
    return first, await tools.find_blocks()


def match(truth_xy, objects):
    """One-to-one, nearest pairs first, within GATE -> {truth index: (object, distance m)}."""
    pairs = sorted((float(np.linalg.norm(np.asarray(o["position_m"]) - xy)), i, k)
                   for i, xy in enumerate(truth_xy) for k, o in enumerate(objects))
    out, used = {}, set()
    for d, i, k in pairs:
        if d <= GATE and i not in out and k not in used:
            out[i] = (objects[k], d)
            used.add(k)
    return out


def picture(path, rgb, first, second):
    img = cv2.resize(np.ascontiguousarray(rgb[..., ::-1]), None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    h, w = rgb.shape[:2]
    px = lambda p: (int(round(p[1] / 1000 * (w - 1) * 2)), int(round(p[0] / 1000 * (h - 1) * 2)))  # noqa: E731
    flagged = {o["id"] for o in first["objects_on_counter"] if o.get("probably_not_a_block")}
    for o in second.get("objects_on_counter", []) + second.get("not_confirmed", []):
        said = o.get("find_blocks")
        colour = {"confirmed": (0, 220, 0), "added": (255, 220, 0), "not_confirmed": (0, 0, 255)}.get(said, (255, 255, 255))
        cv2.circle(img, px(o["point"]), 16, colour, 2)
        text = o["id"] + ("?" if o["id"] in flagged else "")
        for c, th in (((0, 0, 0), 3), ((255, 255, 255), 1)):
            cv2.putText(img, text, (px(o["point"])[0] + 18, px(o["point"])[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, th, cv2.LINE_AA)
    for n in second.get("not_added", []):
        cv2.drawMarker(img, px(n["point"]), (200, 200, 200), cv2.MARKER_TILTED_CROSS, 16, 2)
    cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 88])


def sim_frames(run):
    evals, frames = sorted(run.glob("runs/eval/*/eval.json")), run / "agent_trace" / "frames"
    if not evals or not frames.is_dir():
        return
    for i, t in enumerate(json.loads(evals[-1].read_text())["trials"]):
        f = sorted(x for x in frames.glob(f"ep{i:03d}_00_*.jpg") if "depth" not in x.name)
        if f:
            yield i, t, f[0]


def run_sim(robot, runs, verbose, out):
    sc = Scene.load()
    tot, err = {}, {}
    for run in runs:
        c, e = tot.setdefault(run.name, Counter()), err.setdefault(run.name, {"look": [], "find_blocks": [], "added": []})
        for i, t, f in sim_frames(run):
            rgb = cv2.imread(str(f))[..., ::-1]
            show(robot, rgb, cv2.imread(str(f).replace(".jpg", "_depth.png"), cv2.IMREAD_UNCHANGED), t["instruction"])
            first, second = asyncio.run(both_tools())
            truth = [np.asarray(sc.mat_to_pelvis(sc.cube_spawn_mat(o)))[:2] for o in t["layout"]]
            c["episodes"] += 1
            c["objects"] += len(truth)
            a = first.get("objects_on_counter") or []
            m = match(truth, a)
            c["look right"] += len(m)
            c["look missed"] += len(truth) - len(m)
            c["look extra"] += len(a) - len(m)
            real_ids = {o["id"] for o, _ in m.values()}
            c["real flagged"] += sum(bool(o.get("probably_not_a_block")) for o, _ in m.values())
            c["extra flagged"] += sum(bool(o.get("probably_not_a_block")) for o in a if o["id"] not in real_ids)
            c["hint given"] += "block_check" in first
            e["look"] += [d for _, d in m.values()]
            if "find_blocks_failed" in second or second.get("status") != "ok":
                c["ER failed"] += 1
                print(f"  {run.name} ep{i}: find_blocks failed: {er.last['error'] or second.get('reason')}")
                continue
            c["ER calls"] += 1
            c["ER seconds"] += er.last["seconds"]
            b, unsure = second["objects_on_counter"], second.get("not_confirmed", [])
            m2, lost = match(truth, b), match(truth, unsure)
            c["after right"] += len(m2)
            c["after missed"] += len(truth) - len(m2)
            c["after extra"] += len(b) - len(m2)
            c["real not confirmed"] += len([i_ for i_ in lost if i_ not in m2])
            c["added"] += sum(o.get("find_blocks") == "added" for o in b)
            c["added real"] += sum(o.get("find_blocks") == "added" for o, _ in m2.values())
            c["not added"] += len(second.get("not_added", []))
            e["find_blocks"] += [d for _, d in m2.values()]
            e["added"] += [d for o, d in m2.values() if o.get("find_blocks") == "added"]
            if verbose:
                for o, _ in m.values():
                    if o.get("probably_not_a_block"):
                        print(f"  {run.name} ep{i}: REAL object {o['id']} flagged; label {o.get('gemini_label')}")
                for i_ in lost:
                    if i_ not in m2:
                        print(f"  {run.name} ep{i}: REAL object {lost[i_][0]['id']} not confirmed ({t['layout'][i_]['color']} {t['layout'][i_]['shape']})")
                for n in second.get("not_added", []):
                    print(f"  {run.name} ep{i}: not added {n}")
                print(f"  {run.name} ep{i}: look {len(m)}/{len(truth)} extra {len(a) - len(m)} | find_blocks {len(m2)}/{len(truth)} "
                      f"extra {len(b) - len(m2)} not confirmed {len(unsure)} ({er.last['seconds']} s)")
            if out:
                picture(out / f"{run.name}_ep{i:03d}.jpg", rgb, first, second)
    med = lambda v: f"{np.median(v) * 1000:.1f}" if len(v) else "n/a"  # noqa: E731
    print("\nSIM, first-look frames. look = today's list; find_blocks = its objects_on_counter (confirmed + added)")
    print("run               episodes objects | look: right missed extra | flagged: real extra  hint | find_blocks: right missed"
          " extra | real not confirmed  added (real)  not added | position mm median: look  find_blocks | ER s/call  failed")
    for name, c in tot.items():
        calls = max(c["ER calls"], 1)
        print(f"{name:18s}{c['episodes']:8d}{c['objects']:8d} |{c['look right']:12d}{c['look missed']:7d}{c['look extra']:6d} |"
              f"{c['real flagged']:14d}{c['extra flagged']:6d}{c['hint given']:6d} |{c['after right']:19d}{c['after missed']:7d}"
              f"{c['after extra']:6d} |{c['real not confirmed']:19d}{c['added']:7d} ({c['added real']}){c['not added']:11d} |"
              f"{med(err[name]['look']):>25s}{med(err[name]['find_blocks']):>13s} |{c['ER seconds'] / calls:10.2f}{c['ER failed']:8d}")
    return tot


def run_real(robot, root, reference, instruction, verbose, out):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from real_data_check import Drop
    drop = Drop(str(root))
    ref = {}
    if reference and Path(reference).is_file():
        for r in json.loads(Path(reference).read_text()):
            if r["tag"].endswith("_f0000") and "points/minimal" in r.get("er", {}):
                ref[int(r["tag"][7:9])] = r["er"]["points/minimal"]["n_items"]
    tot, rows = Counter(), []
    for e in sorted(int(x) for x in np.unique(drop.epi)):
        rgb, depth = drop.frames("observation.images.cam_high", e, [0], "rgb24")[0], drop.depth(e, [0])[0]
        show(robot, rgb, depth, instruction)
        first, second = asyncio.run(both_tools())
        a = first.get("objects_on_counter") or []
        flagged = {o["id"] for o in a if o.get("probably_not_a_block")}
        row = {"ep": e, "detections": len(a), "flagged": len(flagged), "hint": "block_check" in first, "reference": ref.get(e)}
        if "find_blocks_failed" in second or second.get("status") != "ok":
            row["failed"] = er.last["error"] or second.get("reason")
            tot["ER failed"] += 1
        else:
            b, unsure = second["objects_on_counter"], second.get("not_confirmed", [])
            row.update(confirmed=sum(o.get("find_blocks") == "confirmed" for o in b),
                       added=sum(o.get("find_blocks") == "added" for o in b), not_confirmed=len(unsure),
                       not_added=[n["reason"] for n in second.get("not_added", [])], seconds=er.last["seconds"],
                       flagged_not_confirmed=sum(o["id"] in flagged for o in unsure),
                       flagged_confirmed=sum(o["id"] in flagged for o in b),
                       unflagged_not_confirmed=sum(o["id"] not in flagged for o in unsure),
                       # an added block is held to look's label rule too (tools._detect): how many real ones it costs
                       added_flagged=sum(o.get("find_blocks") == "added" and bool(o.get("probably_not_a_block")) for o in b))
            row["blocks"] = row["confirmed"] + row["added"]
            for k in ("confirmed", "added", "not_confirmed", "blocks", "flagged_not_confirmed", "flagged_confirmed",
                      "unflagged_not_confirmed", "added_flagged", "seconds"):
                tot[k] += row[k]
            tot["not_added"] += len(row["not_added"])
            tot["ER calls"] += 1
            if row["reference"] is not None:
                tot["reference"] += row["reference"]
                tot["blocks on reference frames"] += row["blocks"]
                tot["not_added on reference frames"] += len(row["not_added"])
                tot["detections on reference frames"] += row["detections"]
            if verbose:
                for o in b + unsure:
                    print(f"  ep{e:02d} {o['id']:7s}{o.get('find_blocks', ''):14s}{'flagged' if o['id'] in flagged else '':8s}"
                          f"at {o['position_m']} point {o['point']} label {o.get('gemini_label')} er {o.get('er_label')}")
                for n in second.get("not_added", []):
                    print(f"  ep{e:02d} not added: {n}")
            if out:
                picture(out / f"real_ep{e:02d}.jpg", rgb, first, second)
        tot["detections"] += row["detections"]
        tot["flagged"] += row["flagged"]
        tot["hint"] += row["hint"]
        rows.append(row)
    print(f"\nREAL, frame 0 of every episode of {root} (no truth). blocks = confirmed + added; reference = blocks the "
          f"evaluation's Robotics-ER call returned on that frame (eye-checked there)")
    print(" ep  detections  flagged  hint | confirmed  added  not confirmed  not added | blocks  reference | "
          "flagged & not confirmed  flagged & confirmed  unflagged & not confirmed | ER s")
    for r in rows:
        if "failed" in r:
            print(f"{r['ep']:3d}{r['detections']:12d}{r['flagged']:9d}{r['hint']:6d} | find_blocks failed: {r['failed']}")
            continue
        print(f"{r['ep']:3d}{r['detections']:12d}{r['flagged']:9d}{r['hint']:6d} |{r['confirmed']:10d}{r['added']:7d}{r['not_confirmed']:15d}"
              f"{len(r['not_added']):11d} |{r['blocks']:7d}{str(r['reference'] if r['reference'] is not None else '-'):>11s} |"
              f"{r['flagged_not_confirmed']:24d}{r['flagged_confirmed']:21d}{r['unflagged_not_confirmed']:27d} |{r['seconds']:5.1f}"
              + (f"   not added: {'; '.join(r['not_added'])}" if r["not_added"] else ""))
    print(f"all{tot['detections']:12d}{tot['flagged']:9d}{tot['hint']:6d} |{tot['confirmed']:10d}{tot['added']:7d}{tot['not_confirmed']:15d}"
          f"{tot['not_added']:11d} |{tot['blocks']:7d}{'':>11s} |{tot['flagged_not_confirmed']:24d}{tot['flagged_confirmed']:21d}"
          f"{tot['unflagged_not_confirmed']:27d} |{tot['seconds'] / max(tot['ER calls'], 1):5.1f}   (failed calls: {tot['ER failed']})")
    print(f"added blocks that their own label then flags probably_not_a_block: {tot['added_flagged']} of {tot['added']}")
    if tot["reference"]:
        print(f"on the frames with a reference: detections {tot['detections on reference frames']}, blocks after find_blocks "
              f"{tot['blocks on reference frames']} (+ {tot['not_added on reference frames']} pointed at but not added), reference {tot['reference']}")
    return tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*", help="sim runs with agent_trace/frames and runs/eval/*/eval.json")
    ap.add_argument("--real", help="a data drop (LeRobot v3), e.g. data_drops/drop3")
    ap.add_argument("--reference", default="/private/tmp/claude-501/-Users-marcusoberg--claude-spares-spare-PjEtDl/"
                    "341d4124-30c5-468f-86de-10e4f75081a7/scratchpad/er_eval/ermeas/results_real.json")
    ap.add_argument("--instruction", default=REAL_INSTRUCTION, help="the instruction for the real frames")
    ap.add_argument("--out", help="directory for one annotated picture per frame")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()
    if not er.available():
        sys.exit("no API key (robot_sorter/.env), or ROBOT_ER=0: nothing to check")
    out = Path(a.out) if a.out else None
    if out:
        out.mkdir(parents=True, exist_ok=True)
    robot = get_robot()
    robot.external, robot.log = True, (lambda m: None)   # no mock world: the frames come from here
    robot.ctx
    if a.runs:
        run_sim(robot, [Path(r) for r in a.runs], a.verbose, out)
    if a.real:
        run_real(robot, Path(a.real), a.reference, a.instruction, a.verbose, out)


if __name__ == "__main__":
    main()
