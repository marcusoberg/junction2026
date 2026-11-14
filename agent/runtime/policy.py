"""SkillPolicy: look -> let a chooser decide -> plan the skill -> execute, as a policy for the organisers' runner.

    pol = SkillPolicy()                                   # camera perception + RuleChooser: a submittable policy
    pol = SkillPolicy(perception="sim", scene_client=SceneClient(), chooser=RandomChooser(seed))   # demonstrations
    pol.reset({"instruction": obs["instruction"]})
    action = pol.act(obs)       # {"arm_q_target": (k,14), "hand_q_target": (k,14), "done": bool}; NEVER blocks

act() only hands out ticks of the trajectory being executed, or holds (with threaded=True, the default). Looking, asking the chooser (possibly a
model over the network) and planning run in one background thread per decision. Every trajectory has passed the
skill checks (IK, clearances, safety dry run) before its first tick is sent.

`pol.segments` records every executed skill (tick, skill, arguments, object, colour, shape, bin, arm, phases),
the labels sim/gen_demos.py stores with a recording.
"""
from __future__ import annotations

import threading
import time

import numpy as np

from agent.skills import (Approach, SkillContext, detect_camera, detect_sim, go_home, hold, pick_and_place,
                          relocate)
from aaltonomous.interface import tasks

from .choosers import RuleChooser
from .language import parse_mapping

HZ = 30.0
DROP_OFFSETS = [(0.0, 0.0), (0.03, 0.03), (-0.03, -0.03), (0.03, -0.03), (-0.03, 0.03)]


class SkillPolicy:
    name = "skills"

    def __init__(self, ctx: SkillContext | None = None, chooser=None, perception="camera", scene_client=None,
                 time_limit=120.0, max_attempts=2, warmup_ticks=15, log=None, shapes=False, threaded=True):
        self.ctx = ctx or SkillContext()
        self.chooser = chooser or RuleChooser(max_attempts)
        if perception == "sim" and scene_client is None:
            raise ValueError("perception='sim' needs a scene_client")
        self.perception, self.client, self.shapes = perception, scene_client, shapes
        self.threaded = threaded     # False: decide inside act() (offline tests on a virtual clock ONLY)
        self.time_limit, self.max_attempts, self.warmup_ticks = float(time_limit), int(max_attempts), int(warmup_ticks)
        self.log = log or (lambda *a: None)
        mo = self.ctx.cfg["motion"]
        self.chunk, self.hold_ticks = int(mo["chunk"]), max(1, int(mo.get("hold_ticks", 1)))
        self.settle_ticks = max(1, int(round(mo["settle_s"] * HZ)))
        self.action_horizon = self.chunk
        self.reset()

    # ------------------------------------------------------------------ protocol
    def reset(self, info=None, task=None, seed=None):
        if isinstance(info, str):
            info = {"instruction": info}
        info = dict(info or {})
        job = getattr(self, "job", None)                 # a decision still being planned belongs to the old episode
        if job is not None:
            job["cancelled"] = True
        self.instruction = info.get("instruction") or tasks.FIXED_INSTRUCTION
        self.task = info.get("task") or task
        self.mapping = info.get("mapping") or parse_mapping(self.instruction) or dict(tasks.FIXED_MAPPING)
        self.chooser.reset(self.instruction, self.task)
        self.state = "warmup"
        self.queue_a = self.queue_h = None
        self.arm_cmd = self.hand_cmd = None
        self.sent, self.job, self.homed = 0, None, False
        self.attempts, self.placed, self.tracks, self.budget = {}, {}, [], None
        self.segments, self.events, self.history = [], [], []
        self.errors = 0

    def act(self, obs):
        if self.arm_cmd is None:        # start from the measured pose, inside the limits (as the baseline does)
            sp = self.ctx.spec
            self.arm_cmd = np.clip(np.asarray(obs["arm_q"], float), sp.arm_lower, sp.arm_upper)
            self.hand_cmd = np.clip(np.asarray(obs["hand_q"], float), sp.hand_lower, sp.hand_upper)
        t = float(obs.get("t", 0.0))
        if self.queue_a is not None and len(self.queue_a):
            return self._pop()
        if self.state == "done":
            return self._hold(1, done=True)
        if self.state == "warmup":      # the sim's first camera frames still show the previous scene
            self.state = "think"
            return self._enqueue_hold(self.warmup_ticks)
        if self.state == "settle":      # arm lag + objects coming to rest before the next look
            self.state = "think"
            return self._enqueue_hold(self.settle_ticks)
        if self.job is None:
            self.job = {"result": None, "cancelled": False}
            if self.threaded:
                threading.Thread(target=self._think_job, args=(self.job, obs, t), daemon=True).start()
            else:
                self._think_job(self.job, obs, t)
        if self.job["result"] is None:
            return self._hold(self.hold_ticks)
        res, self.job = self.job["result"], None
        kind = res[0]
        if kind == "trajectory":
            _, traj, decision = res
            self._start(traj, decision, t)
            return self._pop()
        if kind == "error":
            self.errors += 1
            self._note(t, f"error while deciding: {res[1]}")
            if self.errors >= 3:
                self.state = "done"
            return self._hold(self.hold_ticks, done=self.state == "done")
        self._note(t, res[1] if len(res) > 1 else "nothing left to do")
        self.state = "done"
        return self._hold(1, done=True)

    # ------------------------------------------------------------------ deciding (background thread)
    def _think_job(self, job, obs, t):
        try:
            job["result"] = self._think(obs, t)
        except Exception as e:          # never kill the control loop
            job["result"] = ("error", repr(e))

    def detect(self, obs):
        if self.perception == "sim":
            return detect_sim(self.ctx, self.client)
        return detect_camera(self.ctx, obs, shapes=self.shapes)

    def _key(self, obj, create=False):
        """Which physical object this is, for counting attempts: its name with simulator perception; with the
        camera, the previously attempted object of the same colour within 6 cm (a missed grasp nudges it)."""
        if obj.source == "sim":
            return obj.id
        xy = np.asarray(obj.xyz[:2], float)
        for i, (color, p) in enumerate(self.tracks):
            if color == obj.color and np.linalg.norm(p - xy) < 0.06:
                if create:
                    self.tracks[i] = (color, xy)
                return i
        if create:
            self.tracks.append((obj.color, xy))
            return len(self.tracks) - 1
        return None

    def _world(self, objs, t, rejected):
        return {"instruction": self.instruction, "task": self.task, "mapping": dict(self.mapping),
                "time_left_s": round(self.time_limit - t, 1),
                "bins": {b: [round(float(v), 3) for v in self.ctx.planner.bin_xy(b)] for b in self.ctx.bins()},
                "objects": [{**o.to_json(), "attempts": self.attempts.get(self._key(o), 0)} for o in objs],
                "rejected": rejected, "history": self.history[-8:]}

    def _think(self, obs, t):
        ctx = self.ctx
        if not self.homed:
            self.homed = True
            if np.abs(self.arm_cmd - ctx.home).max() > 0.02:
                g = go_home(ctx, self.arm_cmd, self.hand_cmd)
                if g.ok:
                    return ("trajectory", g, {"skill": "go_home"})
        objs = self.detect(obs)
        by_id = {o.id: o for o in objs}
        if self.budget is None:                  # hard cap on tries per episode, whatever the perception says
            self.budget = self.max_attempts * len(objs) + 1
        if len(self.segments) >= self.budget:
            return ("done", f"attempt budget used up ({self.budget})")
        rejected = []
        for _ in range(2 * max(1, len(objs)) + 1):
            decision = self.chooser.decide(self._world(objs, t, rejected))
            skill = decision.get("skill")
            if skill == "done":
                return ("done", f"nothing left to sort ({len(objs)} objects seen)")
            obj, b = by_id.get(decision.get("object")), decision.get("bin")
            if skill not in ("pick_and_place", "relocate") or obj is None or b not in ctx.bins():
                rejected.append({"decision": decision, "reason": "unknown skill, object or bin"})
                continue
            ap = Approach.from_json(decision.get("approach"))
            if skill == "pick_and_place":
                off = decision.get("drop_offset") or DROP_OFFSETS[self.placed.get(b, 0) % len(DROP_OFFSETS)]
                traj = pick_and_place(ctx, obj, b, self.arm_cmd, self.hand_cmd, arm=decision.get("arm"),
                                      approach=ap, drop_offset=off)
            else:
                traj = relocate(ctx, obj, b, self.arm_cmd, self.hand_cmd, others=objs, approach=ap)
            if traj.ok and t + traj.ticks / HZ > self.time_limit:
                traj.ok, traj.reason = False, f"not enough time ({traj.ticks / HZ:.0f} s)"
            if traj.ok:
                return ("trajectory", traj, {**decision, "color": obj.color, "shape": obj.shape,
                                             "key": self._key(obj, create=True)})
            rejected.append({"decision": {k: decision.get(k) for k in ("skill", "object", "bin", "arm")},
                             "reason": traj.reason[:160]})
        return ("done", f"no valid plan for anything left ({len(rejected)} tried)")

    # ------------------------------------------------------------------ executing
    def _start(self, traj, decision, t):
        self.queue_a, self.queue_h = traj.arm.copy(), traj.hand.copy()
        self.state = "settle"
        skill = decision.get("skill")
        if skill in ("pick_and_place", "relocate"):
            key = decision.get("key")
            self.attempts[key] = self.attempts.get(key, 0) + 1
            if skill == "pick_and_place":
                self.placed[decision["bin"]] = self.placed.get(decision["bin"], 0) + 1
            self.segments.append({"t": round(t, 3), "tick": self.sent, "skill": skill, "args": traj.args,
                                  "object": decision.get("object"), "color": decision.get("color"),
                                  "shape": decision.get("shape"),
                                  "bin": decision["bin"] if skill == "pick_and_place" else f"relay -> {decision['bin']}",
                                  "side": traj.side, "relay": skill == "relocate", "plan_ticks": traj.ticks,
                                  "phases": traj.phases})
            self.history.append({"skill": skill, "object": decision.get("object"), "bin": decision.get("bin"),
                                 "arm": traj.side, "t": round(t, 1)})
        self._note(t, f"execute {skill} {decision.get('object', '')} -> {decision.get('bin', '')} "
                      f"({traj.side} arm, {traj.ticks / HZ:.1f} s)")

    def _pop(self):
        k = self.chunk
        a, h = self.queue_a[:k], self.queue_h[:k]
        self.queue_a, self.queue_h = self.queue_a[k:], self.queue_h[k:]
        self.arm_cmd, self.hand_cmd = a[-1].copy(), h[-1].copy()
        self.sent += len(a)
        return {"arm_q_target": a.copy(), "hand_q_target": h.copy(), "done": False}

    def _enqueue_hold(self, n):
        tr = hold(self.arm_cmd, self.hand_cmd, n / HZ)
        self.queue_a, self.queue_h = tr.arm, tr.hand
        return self._pop()

    def _hold(self, k, done=False):
        k = 1 if done else k
        self.sent += k
        return {"arm_q_target": np.repeat(self.arm_cmd[None], k, 0), "hand_q_target": np.repeat(self.hand_cmd[None], k, 0),
                "done": bool(done)}

    def _note(self, t, msg):
        self.events.append((round(t, 2), msg))
        self.log(f"[skills t={t:6.2f}] {msg}")
