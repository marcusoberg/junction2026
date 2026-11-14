"""Check the agent's perception on camera frames saved from earlier sim runs. No simulator, no model, seconds.

    PYTHONPATH=aaltonomous-humanoid:adk_agents:. .venv/bin/python sim/replay_perception.py sim_out/t1-* [--frame 0]

Every traced run (sim/modal_sim.py --action eval-adk) saves, for each detection, the head colour image and the
depth image (sim_out/<run>/agent_trace/frames/). The evaluator's eval.json has the true layout of each episode.
This runs the CURRENT `look` code on the first frame of every episode and compares with the truth: objects found,
position error, shape, rotation, and the colour measured for each true colour. Change perception code, rerun this,
and only then spend sim time. (The truth is used here, offline, for scoring only.)
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from aaltonomous.scene.scene import Scene
from robot_sorter import tools
from robot_sorter.robot import Robot

SYM = {"cube": 90.0, "triangle": 120.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--frame", type=int, default=0, help="which saved frame of each episode (0 = the first look)")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()
    robot, sc = Robot(), Scene.load()
    robot.log = lambda m: None
    ctx = robot.ctx
    tot = defaultdict(Counter)
    errs, yaw_errs, colours, shapes = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(Counter)
    for run in map(Path, a.runs):
        evals, frames = sorted(run.glob("runs/eval/*/eval.json")), run / "agent_trace" / "frames"
        if not evals or not frames.is_dir():
            continue
        ev = json.loads(evals[-1].read_text())
        for i, t in enumerate(ev["trials"]):
            f = sorted(frames.glob(f"ep{i:03d}_{a.frame:02d}_*.jpg"))
            if not f:
                continue
            rgb = cv2.imread(str(f[0]))[..., ::-1]
            depth = cv2.imread(str(f[0]).replace(".jpg", "_depth.png"), cv2.IMREAD_UNCHANGED)
            robot.reset("")
            robot.obs = {"head_rgb": np.ascontiguousarray(rgb), "head_depth": depth}
            dets = tools._detect(robot)
            truth = [(c["color"], c["shape"], np.asarray(sc.mat_to_pelvis(sc.cube_spawn_mat(c)))[:2],
                      -90.0 - np.degrees(c["yaw"])) for c in t["layout"]]     # layout.mat_yaw_to_pelvis
            used = set()
            for color, shape, xy, yaw in truth:
                tot[run.name]["objects"] += 1
                cand = [(np.linalg.norm(np.array(d["position_m"]) - xy), k) for k, d in enumerate(dets) if k not in used]
                dist, k = min(cand, default=(9.0, None))
                if dist > 0.04:
                    tot[run.name]["missed"] += 1
                    if a.verbose:
                        print(f"  {run.name} ep{i}: MISSED {color} {shape} at ({xy[0]:.3f},{xy[1]:+.3f})")
                    continue
                used.add(k)
                d = dets[k]
                errs[run.name].append(dist)
                colours[(run.name.split("-")[1], color)].append(d["rgb"])
                seen = d["shape_guess"] or "none"
                shapes[shape][seen] += 1
                if shape in SYM and seen == shape and d["yaw_deg_guess"] is not None:
                    e = (d["yaw_deg_guess"] - yaw + SYM[shape] / 2) % SYM[shape] - SYM[shape] / 2
                    yaw_errs[shape].append(abs(e))
            tot[run.name]["false"] += len(dets) - len(used)
            if a.verbose:
                for k, d in enumerate(dets):
                    if k not in used:
                        near = min(np.linalg.norm(np.array(d["position_m"]) - x[2]) for x in truth)
                        print(f"  {run.name} ep{i}: FALSE detection at {d['position_m']} rgb {d['rgb']} shape {d['shape_guess']}"
                              f" ({near * 100:.0f} cm from the nearest real object)")
            tot[run.name]["episodes"] += 1
    print("run                 episodes objects missed false-detections  position error median / 90 % (mm)")
    for name, c in tot.items():
        e = np.array(errs[name]) * 1000 if errs[name] else np.array([np.nan])
        print(f"{name:20s}{c['episodes']:6d}{c['objects']:8d}{c['missed']:7d}{c['false']:12d}"
              f"{np.median(e):22.1f} / {np.percentile(e, 90):.1f}")
    print("\nshape seen (columns) for each true shape (rows):")
    cols = ["cube", "cylinder", "triangle", "lying on its side", "none"]
    print("            " + "".join(f"{c[:9]:>10s}" for c in cols))
    for s in ("cube", "cylinder", "triangle"):
        print(f"  {s:10s}" + "".join(f"{shapes[s][c]:10d}" for c in cols))
    for s, v in yaw_errs.items():
        print(f"rotation error, {s}s recognised as such: median {np.median(v):.1f} deg, 90 % {np.percentile(v, 90):.1f} deg, n={len(v)}")
    print("\ncolour measured (median rgb) per true colour, per scene:")
    for (scene, color), v in sorted(colours.items()):
        print(f"  {scene:8s}{color:8s}{np.median(np.array(v), axis=0).astype(int).tolist()}   n={len(v)}")


if __name__ == "__main__":
    main()
