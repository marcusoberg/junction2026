"""Check the agent's DECISIONS (which object goes to which bin) on frames saved from earlier sim runs.
No simulator: the real ADK agent and the real Gemini model, on a saved head image, with the motions faked.
About a minute for 20 episodes (episodes run in parallel processes; a few cents of model calls).

    PYTHONPATH=aaltonomous-humanoid:adk_agents:. .venv/bin/python sim/replay_decisions.py sim_out/t2-venue-fixed sim_out/t1-nominal

For each episode: the first saved frame (all objects on the counter), the episode's instruction, the current
`look` code and the current prompt. `pick_and_place` and the other motion tools are replaced by stand-ins with the
same declarations that always "succeed" and paint the object out of the image. What is scored is the bin the agent
chose for each object against the bin the instruction asks for (the evaluator's mapping): right, wrong, or left on
the counter. Use it to test prompt changes and colour / shape reasoning before spending sim time.
"""
import argparse
import asyncio
import functools
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np


def episode(job):
    run, i, verbose = job
    import cv2
    from google.adk.agents import Agent
    from google.adk.runners import InMemoryRunner
    from google.genai import types
    from aaltonomous.scene.scene import Scene, resolve_bin
    from robot_sorter import agent as A, tools as T
    from robot_sorter.robot import get_robot

    run = Path(run)
    ev = json.loads(sorted(run.glob("runs/eval/*/eval.json"))[-1].read_text())
    t = ev["trials"][i]
    f = sorted((run / "agent_trace" / "frames").glob(f"ep{i:03d}_00_*.jpg"))
    if not f:
        return None
    sc = Scene.load()
    robot = get_robot()
    robot.external, robot.log = True, (lambda m: None)
    ctx = robot.ctx
    robot.reset(t["instruction"], time_limit=None)
    rgb = np.ascontiguousarray(cv2.imread(str(f[0]))[..., ::-1])
    depth = cv2.imread(str(f[0]).replace(".jpg", "_depth.png"), cv2.IMREAD_UNCHANGED)
    home_h = ctx.post["open"].copy()
    robot.obs = {"head_rgb": rgb, "head_depth": depth, "arm_q": ctx.home.copy(), "hand_q": home_h, "t": 0.0}
    robot.arm_cmd, robot.hand_cmd = ctx.home.copy(), home_h.copy()
    robot._ready.set()
    decisions, gone, said = {}, set(), []

    def remaining():
        return [o for o in T._detect(robot) if o["id"] not in gone]

    def take(object_id, point):
        if object_id and object_id in robot.objects:
            return object_id
        if point is not None and len(point) == 2 and robot.objects:      # pointed: the nearest detection
            from agent.skills.perception import point_to_xyz
            xyz, _ = point_to_xyz(ctx, robot.obs, point)
            return min(robot.objects, key=lambda k: np.linalg.norm(robot.objects[k].xyz[:2] - xyz[:2]))
        return None

    def erase(object_id):
        o = robot.objects[object_id]
        u, v = (int(round(c)) for c in o.extra["uv"])
        h, w = rgb.shape[:2]
        around = rgb[max(0, v - 45):v + 46:6, max(0, u - 45):u + 46:6].reshape(-1, 3)
        cv2.circle(rgb, (u, v + 8), 34, [int(c) for c in np.median(around, axis=0)], -1)
        cv2.circle(depth, (u, v + 8), 34, int(np.median(depth[max(0, v - 45):v + 46:6, max(0, u - 45):u + 46:6])), -1)
        gone.add(object_id)

    def stand_in(real):
        @functools.wraps(real)
        async def tool(*args, **kw):
            name = real.__name__
            if name == "look":
                objects = remaining()
                robot.view_requests.append({"camera": "head", "region": None, "mark_objects": True})
                return T._result("ok", objects_on_counter=objects, instruction=robot.instruction,
                                 bins={b: [round(float(x), 3) for x in ctx.planner.bin_xy(b)] for b in ctx.bins()})
            if name in ("view_camera", "read_proprioception"):
                return await real(*args, **kw)
            if name == "done":
                said.append(kw.get("summary", ""))
                robot.finished = True
                return T._result("episode ended")
            if name in ("pick_and_place", "pick", "hand_over"):
                oid = take(kw.get("object_id"), kw.get("point"))
                if oid is None or oid in gone:
                    return T._result("failed", reason=f"unknown object; on the counter: {[o['id'] for o in remaining()]}")
                if name == "pick_and_place":
                    decisions[oid] = kw.get("bin")
                    erase(oid)
                    return T._result("completed", object=oid, bin=kw.get("bin"), grasp_check={"object_in_hand": True},
                                     objects_on_counter=remaining())
                return T._result("completed", object=oid, objects_on_counter=remaining())
            return T._result("completed", objects_on_counter=remaining())
        return tool

    agent = Agent(name="robot_sorter", model=A.MODEL, instruction=A.INSTRUCTION, tools=[stand_in(x) for x in T.TOOLS],
                  generate_content_config=A._generation_config(), before_model_callback=[A.attach_live_view])
    runner = InMemoryRunner(agent=agent, app_name="robot_sorter")

    async def go():
        s = await runner.session_service.create_session(app_name="robot_sorter", user_id="replay")
        text = f"New episode. Instruction: {t['instruction']}"
        for _ in range(4):
            if robot.finished:
                break
            msg = types.Content(role="user", parts=[types.Part(text=text)])
            async for _e in runner.run_async(user_id="replay", session_id=s.id, new_message=msg):
                pass
            text = "Carry on: call the next tool. Call done only when nothing useful is left to do."

    T._detect(robot)
    first = {k: (o.xyz[:2].copy(), o.extra.get("shape_seen")) for k, o in robot.objects.items()}
    try:
        asyncio.run(go())
        error = ""
    except Exception as e:  # noqa: BLE001
        error = repr(e)[:200]
    mapping = {(tuple(k.split("/")) if "/" in k else k): v for k, v in (t.get("mapping") or {}).items()}
    rows = []
    for c in t["layout"]:
        xy = np.asarray(sc.mat_to_pelvis(sc.cube_spawn_mat(c)))[:2]
        oid = min(first, key=lambda k: np.linalg.norm(first[k][0] - xy), default=None)
        want = resolve_bin(mapping, c["color"], c["shape"])
        chose = decisions.get(oid) if oid is not None and np.linalg.norm(first[oid][0] - xy) < 0.04 else None
        rows.append({"color": c["color"], "shape": c["shape"], "want": want, "chose": chose,
                     "verdict": "left" if chose is None else "right" if chose == want else "WRONG"})
    return {"run": run.name, "seed": t["seed"], "instruction": t["instruction"], "rows": rows, "said": said, "error": error}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()
    jobs = []
    for run in a.runs:
        evals = sorted(Path(run).glob("runs/eval/*/eval.json"))
        if evals and (Path(run) / "agent_trace" / "frames").is_dir():
            jobs += [(run, i, a.verbose) for i in range(len(json.loads(evals[-1].read_text())["trials"]))]
    with ProcessPoolExecutor(a.workers) as pool:
        results = [r for r in pool.map(episode, jobs) if r]
    totals = {}
    for r in results:
        bad = [x for x in r["rows"] if x["verdict"] != "right"]
        tot = totals.setdefault(r["run"], {"right": 0, "WRONG": 0, "left": 0})
        for x in r["rows"]:
            tot[x["verdict"]] += 1
        if bad or r["error"] or a.verbose:
            print(f"\n{r['run']} seed {r['seed']}: {r['instruction']}")
            for x in r["rows"]:
                print(f"    {x['color']:7s}{x['shape']:9s} belongs in {str(x['want']):7s} agent chose {str(x['chose']):7s} {x['verdict']}")
            if r["said"]:
                print(f"    agent's summary: {r['said'][-1][:260]}")
            if r["error"]:
                print(f"    ERROR: {r['error']}")
    print("\nrun                      right  WRONG  left   (decisions only: every grasp is assumed to succeed)")
    for name, tot in totals.items():
        print(f"{name:24s}{tot['right']:6d}{tot['WRONG']:7d}{tot['left']:6d}")


if __name__ == "__main__":
    sys.exit(main())
