"""Check the Decisions API labels (colour and shape per object) on frames saved from earlier sim runs, against the
true layout. No simulator; one Decisions request per episode.

    PYTHONPATH=aaltonomous-humanoid:adk_agents:. .venv/bin/python sim/replay_labels.py sim_out/t2-venue-fixed sim_out/t1-nominal
"""
import json
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from aaltonomous.scene.scene import Scene
from robot_sorter import agent as A, decisions as D, tools
from robot_sorter.robot import Robot


def main(runs):
    robot, sc = Robot(), Scene.load()
    robot.log = lambda m: None
    robot.ctx
    jobs = []
    for run in map(Path, runs):
        evals, frames = sorted(run.glob("runs/eval/*/eval.json")), run / "agent_trace" / "frames"
        if not evals or not frames.is_dir():
            continue
        ev = json.loads(evals[-1].read_text())
        for i, t in enumerate(ev["trials"]):
            f = sorted(frames.glob(f"ep{i:03d}_00_*.jpg"))
            if not f:
                continue
            robot.reset(t["instruction"])
            robot.obs = {"head_rgb": np.ascontiguousarray(cv2.imread(str(f[0]))[..., ::-1]),
                         "head_depth": cv2.imread(str(f[0]).replace(".jpg", "_depth.png"), cv2.IMREAD_UNCHANGED)}
            dets = tools._detect(robot)
            marked = cv2.imdecode(np.frombuffer(A._render(robot, {"camera": "head", "region": None, "mark_objects": True}),
                                                np.uint8), cv2.IMREAD_COLOR)[..., ::-1]
            truth = {}
            for c in t["layout"]:
                xy = np.asarray(sc.mat_to_pelvis(sc.cube_spawn_mat(c)))[:2]
                d = min(dets, key=lambda d: np.linalg.norm(np.array(d["position_m"]) - xy))
                truth[d["id"]] = (c["color"], c["shape"])
            jobs.append((run.name, t["seed"], marked, [d["id"] for d in dets], t["instruction"], truth))

    def one(job):
        name, seed, marked, ids, instruction, truth = job
        cache = {}
        try:
            seconds = D.label_objects(marked, ids, instruction, cache)
        except Exception as e:  # noqa: BLE001
            return name, seed, truth, {}, None, repr(e)[:200]
        return name, seed, truth, cache, seconds, ""

    with ThreadPoolExecutor(6) as pool:
        results = list(pool.map(one, jobs))
    tot, lat, conf = defaultdict(Counter), [], defaultdict(list)
    for name, seed, truth, cache, seconds, err in results:
        if err:
            print(f"{name} seed {seed}: ERROR {err}")
            continue
        lat.append(seconds)
        for oid, (color, shape) in truth.items():
            lab = cache.get(oid)
            if lab is None:
                tot[name]["no label"] += 1
                continue
            ok_c, ok_s = lab["colour"] == color, lab["shape"] == shape
            tot[name]["colour right"] += ok_c
            tot[name]["shape right"] += ok_s
            tot[name]["objects"] += 1
            conf["right" if ok_c else "wrong"].append(lab["colour_confidence"])
            if not ok_c:
                print(f"  {name} seed {seed}: {color} {shape} labelled {lab['colour']} ({lab['colour_confidence']}) {lab['colour_probabilities']}")
    print("\nrun                  objects  colour right  shape right  no label")
    for name, c in tot.items():
        print(f"{name:20s}{c['objects']:8d}{c['colour right']:14d}{c['shape right']:13d}{c['no label']:10d}")
    if lat:
        print(f"\nlatency per request (one per episode, all objects): median {np.median(lat):.2f} s, max {max(lat):.2f} s")
    for k, v in conf.items():
        print(f"colour confidence when {k}: median {np.median(v):.2f}, min {min(v):.2f}, n={len(v)}")


if __name__ == "__main__":
    main(sys.argv[1:])
