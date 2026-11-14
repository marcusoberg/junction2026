"""What failed in a sim run of the ADK agent, and why. Offline, after the run: it joins the evaluator's eval.json
(true layout, instruction, mapping, per-object outcome) with the agent's policy.log (every tool call and result).

    PYTHONPATH=aaltonomous-humanoid .venv/bin/python sim/analyze_runs.py sim_out/adk-3 sim_out/var-real-shape ...

The true object positions are used HERE ONLY, to tell which object a tool call was about. The agent never sees them.
Per object it prints: shape, colour, the bin it belongs in, the outcome, and each attempt (tool, bin chosen, status).
Then totals by shape, and a list of every decision that sent an object to a bin the instruction did not ask for.
"""
import ast
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from aaltonomous.scene.scene import Scene, resolve_bin

CALL = re.compile(r"\[agent t=\s*([\d.]+)\] (pick_and_place|pick|place|hand_over|place_on_counter)\((\{.*\})\)\s*$")
RES = re.compile(r"\[agent t=\s*([\d.]+)\] (pick_and_place|pick|place|hand_over|place_on_counter) -> \{'status': '(\w+)'(.*)$")


def parse_log(path):
    """-> one list of attempts per episode: {t, tool, args, status, reason, grasp_at, object, err}"""
    episodes, pending = [], None
    for line in Path(path).read_text(errors="replace").splitlines():
        if "] reset task=" in line:
            episodes.append([])
            continue
        if not episodes:
            continue
        m = CALL.search(line)
        if m:
            try:
                args = ast.literal_eval(m.group(3))
            except Exception:
                args = {}
            pending = {"t": float(m.group(1)), "tool": m.group(2), "args": args}
            continue
        m = RES.search(line)
        if m and pending and pending["tool"] == m.group(2):
            rest = m.group(4)
            g = re.search(r"'grasp_at_m': \[([-\d., ]+)\]", rest)
            r = re.search(r"'reason': ['\"](.*?)['\"],? ?'", rest)
            e = re.search(r"'finger_tracking_error_rad': ([\d.]+)", rest)
            pending.update(status=m.group(3), reason=(r.group(1) if r else "")[:110],
                           grasp_at=None if g is None else np.array([float(v) for v in g.group(1).split(",")])[:2],
                           err=None if e is None else float(e.group(1)))
            episodes[-1].append(pending)
            pending = None
    return episodes


def main(run_dirs):
    by_shape, by_shape_attempt, wrong_choice, fail_reasons = defaultdict(Counter), defaultdict(Counter), [], Counter()
    for run in map(Path, run_dirs):
        evals = sorted(run.glob("runs/eval/*/eval.json"))
        if not evals or not (run / "policy.log").exists():
            print(f"{run}: no eval.json / policy.log")
            continue
        ev = json.loads(evals[-1].read_text())
        sc = Scene.load()
        episodes = parse_log(run / "policy.log")
        print(f"\n================ {run.name}: {ev.get('task', '')} {ev.get('variant', '')}")
        for i, t in enumerate(ev["trials"]):
            mapping = {(tuple(k.split("/")) if "/" in k else k): v for k, v in (t.get("mapping") or {}).items()}
            print(f"\n seed {t['seed']}  {t.get('instruction', '')}")
            truth = []
            for c, o in zip(t["layout"], t["outcomes"]):
                xy = np.asarray(sc.mat_to_pelvis(sc.cube_spawn_mat(c)))[:2]
                truth.append({"color": c["color"], "shape": c["shape"], "yaw": np.degrees(c["yaw"]), "xy": xy,
                              "bin": resolve_bin(mapping, c["color"], c["shape"]), "outcome": None, "attempts": []})
            names = Counter()
            for o in t["outcomes"]:                       # outcomes are named obj_<colour>_<shape>_<k>
                _, col, shp, _ = o["name"].split("_")
                cands = [x for x in truth if x["color"] == col and x["shape"] == shp and x["outcome"] is None]
                if cands:
                    cands[0]["outcome"] = o["outcome"]
            attempts = episodes[i] if i < len(episodes) else []
            where = {}                                     # the agent's object id -> truth entry (first grasp position)
            for a in attempts:
                oid = a["args"].get("object_id") or "pointed"
                if a.get("grasp_at") is not None and oid not in where:
                    where[oid] = min(truth, key=lambda x: np.linalg.norm(x["xy"] - a["grasp_at"]))
                if oid in where:
                    where[oid]["attempts"].append(a)
            for x in truth:
                by_shape[x["shape"]][x["outcome"]] += 1
                tries = []
                for a in x["attempts"]:
                    chosen = a["args"].get("bin") or a["args"].get("for_bin")
                    extra = [k for k in ("shape", "yaw_deg", "finger_direction_deg", "xy_offset_m", "arm") if k in a["args"]]
                    tries.append(f"{a['tool']}->{chosen} {a['status']}" + (f"[{','.join(extra)}]" if extra else ""))
                    if a["tool"] in ("pick_and_place", "pick"):
                        by_shape_attempt[x["shape"]][a["status"]] += 1
                    if a["status"] == "failed":
                        fail_reasons[re.sub(r"[\d.]+ cm", "N cm", a["reason"])[:70]] += 1
                    if chosen and a["tool"] in ("pick_and_place", "place") and chosen != x["bin"]:
                        wrong_choice.append(f"{run.name} seed {t['seed']}: {x['color']} {x['shape']} belongs in {x['bin']}, "
                                            f"agent chose {chosen} (ended: {x['outcome']})")
                print(f"   {x['color']:7s}{x['shape']:9s}yaw {x['yaw']:+4.0f}  -> {str(x['bin']):7s}{str(x['outcome']):8s} "
                      + ("; ".join(tries) if tries else "never attempted"))
    print("\n================ totals")
    for shape, c in sorted(by_shape.items()):
        n = sum(c.values())
        a = by_shape_attempt[shape]
        na = sum(a.values())
        print(f" {shape:9s} objects {n:3d}: " + ", ".join(f"{k} {v}" for k, v in c.most_common())
              + f"   | grasp attempts {na}: " + ", ".join(f"{k} {v}" for k, v in a.most_common()))
    print("\n no-safe-plan reasons:")
    for r, n in fail_reasons.most_common(8):
        print(f"   {n:3d}  {r}")
    print(f"\n decisions for a bin the instruction did not ask for: {len(wrong_choice)}")
    for w in sorted(set(wrong_choice)):
        print("   " + w)


if __name__ == "__main__":
    main(sys.argv[1:])
