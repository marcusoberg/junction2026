"""Record the organisers' oracle sorting objects in a running Aalto sim, as demonstrations for training.

Runs in the organisers' `isaac` environment next to the sim (`aalto sim` first). Each episode goes through exactly
the code their evaluator uses (aaltonomous/sim/evaluate.py): home_arms -> seed the layout -> DR `randomize` if the
sim runs with AALTO_DR -> deploy.runner.run_episode (AaltoEnv + SafetyFilter, 30 Hz on the SIM clock) -> settle ->
scene-service scoring. The only difference is the recorder: it also keeps the three camera images of every tick.
(Their own data/record_demos.py samples on the WALL clock, so with the sim at 0.3x real time it would record every
motion three times too slow; that is why this does not use it.)

    python /opt/sim/gen_demos.py --task sort_fixed --seed 100000 --episodes 20 --out /data/sim_raw/<name>/shard_00

One folder per episode, <out>/ep_s<seed>/ (written elsewhere first, moved in when complete):
  meta.json      instruction, mapping, layout, per-object outcomes, score, success, rates, DR parameters,
                 segments = which object the oracle was handling from which tick (for later sub-task labels)
  lowdim.npz     per tick: t (sim s), arm_q, arm_dq, hand_q (observation AFTER the tick), arm_q_cmd, hand_q_cmd (the
                 command sent IN the tick, after the safety filter), hold, clipped, head_repeat,
                 <camera>_end (byte offset where that tick's JPEG ends in <camera>.mjpg)
  head_rgb.mjpg, left_wrist_rgb.mjpg, right_wrist_rgb.mjpg     the JPEGs of all ticks, back to back
success = every object in its correct bin, the oracle ended the episode itself, and no safety clip.
Convert with sim/to_dataset.py (by default it keeps the successful pick-and-place segments of every episode).

--policy skills-random records our skill library (agent/) instead of the oracle: same planner and checks, but a
random object order and random approach parameters per pick, with every pick's skill arguments in the labels.

One deliberate change to the oracle: --max-attempts (default 2) tries per OBJECT. Its own limit counts tries per
table position, so an object that moves when a grasp fails counts as new and is retried until the time runs out
(seen with every triangular prism: 0 of 7 sorted, 5 tries each). After the limit the object is hidden from the oracle.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import ssl  # noqa: F401  before DDS, as in sim/evaluate.py
import sys
import threading
import time
import traceback
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

CAMS = ("head_rgb", "left_wrist_rgb", "right_wrist_rgb")
FORMAT = "aalto-simdemo-v1"


def _js(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


class DemoRecorder:
    """Takes the place of deploy.runner.Recorder in run_episode (it only calls .add and reads .rows)."""

    def __init__(self, path: Path, quality: int):
        import cv2
        self.cv2, self.path, self.quality = cv2, Path(path), int(quality)
        self.path.mkdir(parents=True, exist_ok=True)
        self.rows, self.repeat, self.last_crc = [], [], None
        self.files = {c: open(self.path / f"{c}.mjpg", "wb") for c in CAMS}
        self.ends = {c: [] for c in CAMS}
        self.pool = ThreadPoolExecutor(len(CAMS))

    def _jpeg(self, img):
        ok, buf = self.cv2.imencode(".jpg", np.ascontiguousarray(img[..., ::-1]),
                                    [self.cv2.IMWRITE_JPEG_QUALITY, self.quality])
        if not ok:
            raise RuntimeError("jpeg encoding failed")
        return buf.tobytes()

    def add(self, row, obs):
        self.rows.append(row)
        for c, data in zip(CAMS, self.pool.map(self._jpeg, [obs[c] for c in CAMS])):
            self.files[c].write(data)
            self.ends[c].append(self.files[c].tell())
        crc = zlib.crc32(np.ascontiguousarray(obs["head_rgb"][::4, ::4]).tobytes())
        self.repeat.append(crc == self.last_crc)      # same head image as the tick before = no new camera frame
        self.last_crc = crc

    def close(self):
        for f in self.files.values():
            f.close()
        self.pool.shutdown()

    def save(self, meta):
        self.close()
        cols = {k: np.array([r[k] for r in self.rows]) for k in self.rows[0]}
        cols["head_repeat"] = np.array(self.repeat)
        for c in CAMS:
            cols[f"{c}_end"] = np.array(self.ends[c], np.int64)
        np.savez_compressed(self.path / "lowdim.npz", **cols)
        (self.path / "meta.json").write_text(json.dumps(meta, indent=1, default=_js))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", default="sort_fixed", choices=["sort_fixed", "sort_language"])
    ap.add_argument("--seed", type=int, default=100000, help="first seed; episode i uses seed + i (evaluator: 1000+)")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--out", required=True)
    ap.add_argument("--time-limit", type=float, default=120.0)
    ap.add_argument("--settle", type=float, default=3.0)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--jpeg-quality", type=int, default=92)
    ap.add_argument("--max-attempts", type=int, default=2, help="oracle tries per object (0 = its own behaviour)")
    ap.add_argument("--policy", default="oracle", choices=["oracle", "skills-rule", "skills-random"],
                    help="who solves the episode: the organisers' oracle, or our skill library (agent/) with exact "
                         "poses and the rule chooser / the random chooser (varied order, approach, speed, drop point)")
    ap.add_argument("--variety", type=float, default=1.0, help="skills-random: 0..1, how far parameters stray")
    ap.add_argument("--tmp", default="/tmp/aalto_demo_tmp")
    a = ap.parse_args(argv)

    from aaltonomous.deploy.runner import run_episode
    from aaltonomous.interface.env import AaltoEnv
    from aaltonomous.sim import evaluate as E
    from aaltonomous.sim.oracle import OraclePolicy
    from aaltonomous.sim.scene_client import SceneClient

    out, tmp = Path(a.out), Path(a.tmp)
    out.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(tmp, ignore_errors=True)
    E.apply_instance_env()
    sc = SceneClient(a.host)
    try:
        sim_info = sc.call(cmd="info")
    except RuntimeError:
        sim_info = {"dr": None}
    sim_info.pop("ok", None)
    variant = (sim_info.get("dr") or {}).get("name") or "nominal"

    # Same construction as evaluate.evaluate()
    env = AaltoEnv(backend="sim", task=a.task, seed=a.seed, time_limit=a.time_limit, realtime=True,
                   img_server_ip=a.host, head_rgbd=True,
                   head_rgbd_port=int(os.environ.get("AALTO_RGBD_PORT", "55560")), cameras=True)
    skills = a.policy != "oracle"
    if skills:
        from agent.runtime import RandomChooser, RuleChooser, SkillPolicy
        from agent.skills import SkillContext
        pol = SkillPolicy(ctx=SkillContext(safety=env.filter.cfg), perception="sim", scene_client=SceneClient(a.host),
                          time_limit=a.time_limit, max_attempts=max(1, a.max_attempts), log=None)
        pol.mapping_source = "parsed"
    else:
        pol = OraclePolicy(SceneClient(a.host), safety=env.filter.cfg, time_limit=a.time_limit,
                           privileged_mapping=lambda: env.mapping, log=None)
    client = E.LocalClient(pol, name=a.policy)
    spec = env.filter.spec
    rest = float(env.scene.cfg["success"]["at_rest_speed"])
    stop = threading.Event()

    cur = {"rec": None, "segments": []}
    start_plan = pol._start

    def start_skill(traj, decision, t):       # SkillPolicy keeps its own labels; add the recorder's tick
        n = len(pol.segments)
        start_plan(traj, decision, t)
        if len(pol.segments) > n:
            pol.segments[-1]["tick"] = len(cur["rec"].rows) if cur["rec"] else None
            cur["segments"].append(pol.segments[-1])

    def start_and_log(plan, t):               # which object the oracle starts on, and when
        ph = [(str(n), int(s), int(e)) for n, s, e in (getattr(plan, "phases", None) or [])]
        cur["segments"].append({"t": round(float(t), 3), "tick": len(cur["rec"].rows) if cur["rec"] else None,
                                "object": getattr(plan, "obj", None), "color": getattr(plan, "color", None),
                                "bin": str(plan.target_bin), "side": getattr(plan, "side", None),
                                "relay": getattr(plan, "relay", None), "plan_ticks": len(plan.arm), "phases": ph})
        return start_plan(plan, t)
    pol._start = start_skill if skills else start_and_log
    if a.max_attempts > 0 and not skills:
        detect = pol.perception.detect

        def detect_untried(*args, **kw):
            tried = {}
            for s in cur["segments"]:
                tried[s["object"]] = tried.get(s["object"], 0) + 1
            return [d for d in detect(*args, **kw) if tried.get(getattr(d, "name", None), 0) < a.max_attempts]
        pol.perception.detect = detect_untried

    print(f"[gen] sim {sim_info}  variant {variant}  task {a.task}  -> {out}", flush=True)
    done, good, crashes, t_all = 0, 0, 0, time.monotonic()
    try:
        for i in range(a.episodes):
            seed = a.seed + i
            name = f"ep_s{seed:06d}"
            if (out / name / "meta.json").exists():
                print(f"[gen] {name}: already recorded, skipping", flush=True)
                continue
            rec = None
            try:
                home_err = E.home_arms(env, sc)
                env.rng = np.random.default_rng(seed)
                dr = sc.call(cmd="randomize", seed=seed).get("params") if sim_info.get("dr") else None
                rec = DemoRecorder(tmp / name, a.jpeg_quality)
                cur["rec"], cur["segments"] = rec, []
                if skills:
                    pol.chooser = RandomChooser(seed, a.variety, pol.max_attempts) if a.policy == "skills-random" \
                        else RuleChooser(pol.max_attempts)
                t0 = time.monotonic()
                summary, events = run_episode(env, client, i, seed, rec, stop, a.task)
                rtf = summary.get("rtf") or 1.0
                E.settle(sc, a.settle / max(min(rtf, 1.0), 0.2), rest)
                ev = sc.evaluate(a.task, env.mapping)
                outc = [o["outcome"] for o in ev["outcomes"]]
                clips = (summary.get("safety") or {}).get("clipped_steps") or 0
                success = bool(ev["all_correct"]) and summary["end_reason"] == "done" and clips == 0
                meta = {"format": FORMAT, "source": f"sim-{a.policy}", "task": a.task, "seed": seed, "variant": variant,
                        "instruction": summary["instruction"], "mapping": env.mapping, "layout": env.layout,
                        "success": success, "outcomes": ev["outcomes"], "score": ev["score"],
                        "n_objects": len(outc), "n_correct": outc.count("correct"), "all_correct": ev["all_correct"],
                        "end_reason": summary["end_reason"], "ticks": len(rec.rows), "clipped_steps": clips,
                        "head_repeat_frac": round(float(np.mean(rec.repeat)), 3) if rec.repeat else None,
                        "home_err": round(float(home_err), 3), "episode_wall_s": round(time.monotonic() - t0, 1),
                        "summary": summary, "safety_events": events[:20], "segments": cur["segments"],
                        "oracle_events": pol.events[-40:], "mapping_source": pol.mapping_source, "dr_params": dr,
                        "sim": sim_info, "max_attempts": a.max_attempts, "arm_names": list(spec.arm_names), "hand_names": list(spec.hand_names),
                        "cameras": list(CAMS), "fps": 30}
                rec.save(meta)
                part = out / (name + ".partial")          # complete on the volume or not there at all
                shutil.rmtree(part, ignore_errors=True)
                shutil.copytree(tmp / name, part)
                part.rename(out / name)
                shutil.rmtree(tmp / name, ignore_errors=True)
                done, good, crashes = done + 1, good + success, 0
                print(f"[gen] {name}: {meta['n_correct']}/{meta['n_objects']} correct  end={meta['end_reason']}  "
                      f"ticks={meta['ticks']} ({meta['ticks'] / 30:.0f} sim s, {meta['episode_wall_s']:.0f} s wall, "
                      f"rtf {summary.get('rtf')})  clips={clips}  hz={summary.get('command_hz')}  "
                      f"head_repeat={meta['head_repeat_frac']}  success={success}", flush=True)
            except KeyboardInterrupt:
                break
            except Exception:
                traceback.print_exc()
                crashes += 1
                if rec is not None:
                    rec.close()
                shutil.rmtree(tmp / name, ignore_errors=True)
                print(f"[gen] {name}: crashed, not kept ({crashes} in a row)", flush=True)
                if crashes >= 3:
                    print("[gen] three crashes in a row: giving up on this sim", flush=True)
                    break
    finally:
        client.close()
        env.close()
    print(f"[gen] recorded {done} episodes ({good} successful) in {time.monotonic() - t_all:.0f} s -> {out}", flush=True)
    return 0 if done else 1


if __name__ == "__main__":
    sys.exit(main())
