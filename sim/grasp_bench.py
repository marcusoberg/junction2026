"""Single-object grasp bench for OUR tools in the Isaac sim: one object per episode (shape x yaw x position), the
scripted agent (adk_agents/robot_sorter/bench.py: look, pick_and_place, retry) behind the policy server, no model
calls. Modelled on the organisers' aaltonomous/sim/grasp_test.py, which does the same for their oracle.

    python /opt/sim/grasp_bench.py --shapes triangle --yaws -60,-30,0,30,60 --xs 0.30,0.45,0.60 [--ys 0.15]
    (inside the sim container, sim running, policy server on :8000 with ROBOT_AGENT_MODEL=bench;
     sim/modal_sim.py --action grasp-bench --bench "..." sets all of that up)

This harness places the object and reads the outcome through the sim's scene service; the policy under test still
sees only its cameras. Per case it prints the outcome and how many grasps it took; then totals by shape.
"""
import argparse
import itertools
import json
import os
import re
import tempfile
import threading
from pathlib import Path

import numpy as np

from aaltonomous.sim import evaluate as ev


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", default="cube,cylinder,triangle")
    ap.add_argument("--colors", default="red,yellow,blue")
    ap.add_argument("--yaws", default="-45,0,30", help="deg, mat frame")
    ap.add_argument("--xs", default=None, help="mat x (m); default 4 points across layout.region_x")
    ap.add_argument("--ys", default=None, help="mat y (m); default the middle of layout.region_y")
    ap.add_argument("--time-limit", type=float, default=45.0)
    ap.add_argument("--policy", default="ws://localhost:8000")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--out", default="/tmp/grasp_bench.json")
    a = ap.parse_args(argv)
    from aaltonomous.deploy.runner import Recorder, run_episode
    from aaltonomous.interface import env as env_mod
    from aaltonomous.interface.env import AaltoEnv
    from aaltonomous.sim.scene_client import SceneClient

    ev.apply_instance_env()
    sc = SceneClient(a.host)
    env = AaltoEnv(backend="sim", task="sort_fixed", seed=0, time_limit=a.time_limit, realtime=True,
                   img_server_ip=a.host, head_rgbd=True, cameras=True,
                   head_rgbd_port=int(os.environ.get("AALTO_RGBD_PORT", "55560")))
    L = env.scene.cfg["layout"]
    xs = [float(v) for v in a.xs.split(",")] if a.xs else list(np.linspace(*L["region_x"], 4))
    ys = [float(v) for v in a.ys.split(",")] if a.ys else [float(np.mean(L["region_y"]))]
    client = ev.PolicyClient(a.policy, 120.0)
    tmp = Path(tempfile.mkdtemp(prefix="grasp_bench_"))
    cases = list(itertools.product(a.shapes.split(","), [float(y) for y in a.yaws.split(",")], xs, ys))
    colors = itertools.cycle(a.colors.split(","))
    rows, stop = [], threading.Event()
    log = Path("/tmp/policy.log")
    try:
        for i, (shape, yaw, x, y) in enumerate(cases):
            color = next(colors)
            lay = [{"color": color, "shape": shape, "x": x, "y": y, "yaw": float(np.radians(yaw))}]
            env_mod.sample_layout = lambda rng, _l=lay, **kw: [dict(o) for o in _l]
            ev.home_arms(env, sc)
            mark = log.stat().st_size if log.exists() else 0
            summ, _ = run_episode(env, client, i, i, Recorder(tmp / f"ep{i:03d}"), stop, "sort_fixed")
            ev.settle(sc, 3.0, float(env.scene.cfg["success"]["at_rest_speed"]))
            out = sc.evaluate("sort_fixed", env.mapping)["outcomes"][0]["outcome"]
            text = log.read_text(errors="replace")[mark:] if log.exists() else ""
            status = re.findall(r"pick_and_place -> \{'status': '(\w+)'", text)
            grips = re.findall(r"'finger_to_apex_deg': ([\d.]+)", text)
            r = {"shape": shape, "color": color, "yaw": yaw, "x": round(x, 3), "y": round(y, 3), "outcome": out,
                 "attempts": status, "grips": grips, "sim_s": summ.get("sim_s"), "end": summ.get("end_reason")}
            rows.append(r)
            print(f"{i:3d} {shape:8s} {color:6s} yaw={yaw:+4.0f} x={x:.2f} y={y:.2f} -> {out:8s} attempts={','.join(status)}"
                  + (f" grips={','.join(grips)}" if grips else ""), flush=True)
    finally:
        client.close()
        env.close()
    print("\nshape     sorted   first grasp held   grasps per sorted object")
    for s in dict.fromkeys(r["shape"] for r in rows):
        v = [r for r in rows if r["shape"] == s]
        ok = [r for r in v if r["outcome"] == "correct"]
        first = sum(1 for r in v if r["attempts"][:1] == ["completed"])
        n_att = sum(len(r["attempts"]) for r in ok)
        print(f"{s:8s} {len(ok):3d}/{len(v):<3d}  {first:3d}/{len(v):<3d}            {n_att / max(len(ok), 1):.1f}")
    json.dump(rows, open(a.out, "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
