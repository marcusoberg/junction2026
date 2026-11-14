"""AgentPolicy: a multimodal model drives the oracle-based skills in real time, looking at the camera images itself.

Two models, two speeds (both are plain interfaces; agent/runtime/gemini.py has the Gemini adapters):

    brain   slow, smart, calls tools.   brain.decide(view) -> {"name": <tool>, "args": {...}}
            Gets the raw head and wrist camera images plus a short state/history, and answers with ONE tool call.
            It says WHICH object by pointing at it in the head image ([y, x], 0..1000); the pointed pixel and the
            depth image give the 3D grasp target. Nothing about the scene is turned into text for it.
    reflex  fast, closed questions.     reflex.answer(view, questions) -> {key: {"value", "confidence"}}
            Asked DURING a motion with the current images (right after the lift: "is the hand holding an object?").
            A confident "no" aborts the rest of the move, the hand opens, the arm goes home, and the brain is told.

act() never waits for either model: it hands out ticks of the current trajectory or holds. Every trajectory comes
from agent/skills and has passed the oracle planner's checks and a zero-clip safety dry run before its first tick.

    pol = AgentPolicy(brain=GeminiBrain(), reflex=GeminiReflex())
    pol.reset({"instruction": obs["instruction"]});  action = pol.act(obs)
"""
from __future__ import annotations

import concurrent.futures
import threading

import numpy as np

from agent.skills import (Approach, SkillContext, WorldObject, go_home, hold, pick, pick_and_place, place,
                          place_on_counter, set_hand)
from agent.skills.perception import encode_image, point_to_xyz
from aaltonomous.interface import tasks

HZ = 30.0
POINT = {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2,
         "description": "[y, x] of the object in the HEAD camera image, each normalised to 0..1000. Point at the "
                        "middle of the object's top face."}
APPROACH = {"type": "object", "description": "Optional grasp variations; leave out for the default.", "properties": {
    "finger_direction_deg": {"type": "number", "description": "Where the fingers point: 0 = away from the robot, +90 = to its left."},
    "pregrasp_offset": {"type": "array", "items": {"type": "number"}, "description": "(dx, dy) m: come down from one side."},
    "speed": {"type": "number", "description": "Arm joint speed in rad/s, 0.3 to 0.97 (default 0.97)."},
    "max_tilt_deg": {"type": "number", "description": "Allowed palm tilt from straight down (default 35)."},
    "grasp_height_frac": {"type": "number", "description": "Grasp height as a fraction of the object height (default 0.5)."},
    "xy_offset": {"type": "array", "items": {"type": "number"}, "description": "(dx, dy) m correction if the last grasp was off."}}}
SHAPE = {"type": "string", "enum": ["cube", "cylinder", "triangle"],
         "description": "The object's shape if you can see it; with yaw_deg it lets the fingers close on flat faces."}
YAW = {"type": "number", "description": "Rotation of the object about the vertical axis in degrees, robot frame (0 = "
                                         "an edge pointing away from the robot). Only with shape cube or triangle."}
ARM = {"type": "string", "enum": ["left", "right"],
       "description": "Optional. The left bin can only be reached by the left arm, the right bin only by the right arm."}
BIN = {"type": "string", "enum": ["left", "center", "right"]}

TOOLS = [
    {"type": "function", "name": "pick_and_place",
     "description": "Pick the object you point at with a top-down pinch, drop it in a bin and return the arm home "
                    "(about 10 s). The grasp is checked right after lifting; a miss aborts the move.",
     "parameters": {"type": "object", "required": ["point", "bin"], "properties": {
         "point": POINT, "bin": BIN, "arm": ARM, "shape": SHAPE, "yaw_deg": YAW, "approach": APPROACH}}},
    {"type": "function", "name": "pick",
     "description": "Pick the object you point at and hold it at carry height (about 5 s). Follow with place or "
                    "place_on_counter after looking at the new images.",
     "parameters": {"type": "object", "required": ["point"], "properties": {
         "point": POINT, "bin": {**BIN, "description": "The bin you intend it for, if you already know: the arm that "
                                                       "can reach that bin is then used."},
         "arm": ARM, "shape": SHAPE, "yaw_deg": YAW, "approach": APPROACH}}},
    {"type": "function", "name": "place",
     "description": "Carry the held object over a bin, release it and return the arm home (about 5 s).",
     "parameters": {"type": "object", "required": ["bin"], "properties": {
         "bin": BIN, "drop_offset": {"type": "array", "items": {"type": "number"},
                                     "description": "(dx, dy) m inside the bin, e.g. to avoid stacking."}}}},
    {"type": "function", "name": "place_on_counter",
     "description": "Set the held object down on the counter at the point you indicate (for example near the middle, "
                    "so the other arm can take it), then return the arm home.",
     "parameters": {"type": "object", "required": ["point"], "properties": {"point": POINT}}},
    {"type": "function", "name": "go_home",
     "description": "Open the hands and move both arms to the home pose, out of the way of the head camera.",
     "parameters": {"type": "object", "properties": {}}},
    {"type": "function", "name": "wait",
     "description": "Hold still and look again after a moment (objects still moving, view blocked).",
     "parameters": {"type": "object", "properties": {"seconds": {"type": "number", "description": "0.2 to 3"}}}},
    {"type": "function", "name": "done",
     "description": "Everything that can be sorted is sorted, or nothing useful is left. Ends the episode.",
     "parameters": {"type": "object", "properties": {}}},
]

SYSTEM_PROMPT = """You control a humanoid robot (two arms, three-finger hands) standing at a counter. Small coloured
objects (cubes, cylinders, triangular prisms; about 4.5 cm) lie on the counter in front of three identical orange
bins. "left", "center" and "right" are from the robot's point of view, which is also the head camera's: the left
bin is on the left of the head image.

Each turn you get the current images (head camera first, then the two wrist cameras), the instruction, what the
robot is holding, how much time is left, and what happened on earlier turns. Answer with exactly one tool call.

- Do what the instruction says. A correctly placed object scores; an object in a wrong bin or knocked off the
  counter costs a point; an object left on the counter costs nothing. If you are not sure about an object, leave it.
- Say which object you mean by pointing at it in the HEAD image. Objects already inside a bin stay there.
- The left bin is reachable only with the left arm and the right bin only with the right arm. For an object on the
  far side from its bin: pick it with the near arm, place_on_counter near the middle, then pick it with the other arm.
- Look at the history. If a call failed or was aborted, do something different: another object, the other arm, a
  different approach, or a small xy_offset if the hand closed beside the object.
- A move takes 5 to 10 seconds. Call done when nothing useful is left or time is nearly up.
"""


def holding_question(arm):
    return {"holding": {"type": "noul",
                        "instructions": f"The robot's {arm} hand has just closed and lifted. Is that hand holding an object "
                                        f"between its fingers?",
                        "criteria": {"true_description": "An object is clearly gripped in the hand, lifted off the counter.",
                                     "false_description": "The hand is empty; the object is still on the counter or fell."}}}


class AgentPolicy:
    name = "agent"

    def __init__(self, brain, reflex=None, ctx: SkillContext | None = None, time_limit=120.0, warmup_ticks=15,
                 brain_timeout_s=25.0, reflex_timeout_s=2.0, abort_confidence=0.75, wrist_images=True,
                 image_side=640, threaded=True, log=None):
        self.brain, self.reflex = brain, reflex
        self.ctx = ctx or SkillContext()
        self.time_limit, self.warmup_ticks = float(time_limit), int(warmup_ticks)
        self.brain_timeout_s, self.reflex_timeout_s = float(brain_timeout_s), float(reflex_timeout_s)
        self.abort_confidence, self.wrist_images, self.image_side = float(abort_confidence), wrist_images, image_side
        self.threaded = threaded            # False only for offline tests on a virtual clock
        self.log = log or (lambda *a: None)
        mo = self.ctx.cfg["motion"]
        self.chunk, self.hold_ticks = int(mo["chunk"]), max(1, int(mo.get("hold_ticks", 1)))
        self.settle_ticks = max(1, int(round(mo["settle_s"] * HZ)))
        self.action_horizon = self.chunk
        self.pool = concurrent.futures.ThreadPoolExecutor(2)           # brain calls
        self.reflex_pool = concurrent.futures.ThreadPoolExecutor(2)    # reflex calls (never queue behind the brain)
        self.stats = {"brain_calls": 0, "brain_failures": 0, "reflex_calls": 0, "aborts": 0}
        self.reset()

    # ------------------------------------------------------------------ protocol
    def reset(self, info=None, task=None, seed=None):
        if isinstance(info, str):
            info = {"instruction": info}
        info = dict(info or {})
        for job in (getattr(self, "job", None), getattr(self, "reflex_job", None)):
            if job is not None:
                job["cancelled"] = True
        self.instruction = info.get("instruction") or tasks.FIXED_INSTRUCTION
        self.task = info.get("task") or task
        self.state = "warmup"
        self.queue_a = self.queue_h = None
        self.arm_cmd = self.hand_cmd = None
        self.sent, self.job, self.reflex_job, self.traj = 0, None, None, None
        self.holding, self.history, self.events, self.segments = None, [], [], []

    def act(self, obs):
        if self.arm_cmd is None:
            sp = self.ctx.spec
            self.arm_cmd = np.clip(np.asarray(obs["arm_q"], float), sp.arm_lower, sp.arm_upper)
            self.hand_cmd = np.clip(np.asarray(obs["hand_q"], float), sp.hand_lower, sp.hand_upper)
        t = float(obs.get("t", 0.0))
        self._reflexes(obs, t)
        if self.queue_a is not None and len(self.queue_a):
            return self._pop()
        if self.traj is not None:
            self._finish(t)
        if self.state == "done":
            return self._hold(1, done=True)
        if self.state in ("warmup", "settle"):
            n = self.warmup_ticks if self.state == "warmup" else self.settle_ticks
            self.state = "think"
            return self._enqueue(hold(self.arm_cmd, self.hand_cmd, n / HZ))
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
            _, traj, call, checks = res
            self._start(traj, call, checks, t)
            return self._pop()
        if kind == "done":
            self._note(t, "the model called done")
            self.state = "done"
            return self._hold(1, done=True)
        if kind == "failed":                      # the tool call had no safe plan: tell the model, ask again
            self.history.append({"tool": res[1]["name"], "args": res[1].get("args", {}), "result": f"failed: {res[2]}"})
            self._note(t, f"{res[1]['name']} failed: {res[2]}")
            return self._enqueue(hold(self.arm_cmd, self.hand_cmd, 0.5))      # do not hammer the model
        if kind == "retry":                       # model call failed or timed out: hold and ask again
            self.stats["brain_failures"] += 1
            self._note(t, f"model call failed: {res[1]}")
            return self._enqueue(hold(self.arm_cmd, self.hand_cmd, 0.5))
        return self._hold(self.hold_ticks)

    # ------------------------------------------------------------------ what the models see
    def view(self, obs, t):
        images = [("head camera", encode_image(obs["head_rgb"], self.image_side))]
        if self.wrist_images:
            for key, label in (("left_wrist_rgb", "left wrist camera"), ("right_wrist_rgb", "right wrist camera")):
                if obs.get(key) is not None:
                    images.append((label, encode_image(obs[key], min(self.image_side, 480))))
        state = {"instruction": self.instruction, "time_left_s": round(self.time_limit - t, 1),
                 "holding": f"an object in the {self.holding} hand" if self.holding else "nothing",
                 "history": self.history[-8:]}
        return {"images": images, "state": state}

    # ------------------------------------------------------------------ slow loop: one tool call per move
    def _think_job(self, job, obs, t):
        try:
            job["result"] = self._think(obs, t)
        except Exception as e:
            job["result"] = ("retry", repr(e))

    def _think(self, obs, t):
        self.stats["brain_calls"] += 1
        fut = self.pool.submit(self.brain.decide, self.view(obs, t))
        try:
            call = fut.result(timeout=self.brain_timeout_s)
        except Exception as e:
            fut.cancel()
            return ("retry", repr(e))
        if not isinstance(call, dict) or "name" not in call:
            return ("retry", f"not a tool call: {str(call)[:80]}")
        call = {"name": call["name"], "args": dict(call.get("args") or {})}
        return self._plan(call, obs, t)

    def _target(self, a, obs, name="object"):
        xyz, info = point_to_xyz(self.ctx, obs, a["point"])
        yaw = np.radians(a["yaw_deg"]) if a.get("yaw_deg") is not None and a.get("shape") in ("cube", "triangle") else None
        return WorldObject(name, a.get("color", "object"), xyz, a.get("shape") if yaw is not None else None, yaw,
                           source="pointed", extra=info)

    def _plan(self, call, obs, t):
        ctx, name, a = self.ctx, call["name"], call["args"]
        arm_q, hand_q = self.arm_cmd, self.hand_cmd
        checks = []
        if name == "done":
            return ("done",)
        if name == "wait":
            return ("trajectory", hold(arm_q, hand_q, float(np.clip(a.get("seconds", 1.0), 0.2, 3.0))), call, checks)
        if name == "go_home":
            traj = go_home(ctx, arm_q, hand_q)
        elif name in ("pick_and_place", "pick"):
            if self.holding:
                return ("failed", call, f"the {self.holding} hand is holding an object: place it first")
            if not isinstance(a.get("point"), (list, tuple)) or len(a["point"]) != 2:
                return ("failed", call, "point must be [y, x]")
            obj = self._target(a, obs)
            ap = Approach.from_json(a.get("approach"))
            if name == "pick_and_place":
                if a.get("bin") not in ctx.bins():
                    return ("failed", call, f"unknown bin {a.get('bin')!r}")
                traj = pick_and_place(ctx, obj, a["bin"], arm_q, hand_q, arm=a.get("arm"), approach=ap)
            else:
                traj = pick(ctx, obj, arm_q, hand_q, arm=a.get("arm"), approach=ap,
                            for_bin=a.get("bin") if a.get("bin") in ctx.bins() else None)
            call["args"]["resolved_xyz_m"] = [round(float(v), 3) for v in obj.xyz]
            if traj.ok:
                lift_end = next((e for n, s, e in traj.phases if n == "lift"), None)
                if lift_end is not None and self.reflex is not None:
                    checks.append((lift_end, "holding", traj.side))
        elif name == "place":
            if not self.holding:
                return ("failed", call, "nothing is held: pick an object first")
            if a.get("bin") not in ctx.bins():
                return ("failed", call, f"unknown bin {a.get('bin')!r}")
            traj = place(ctx, a["bin"], arm_q, hand_q, arm=self.holding, drop_offset=tuple(a.get("drop_offset") or (0.0, 0.0)))
        elif name == "place_on_counter":
            if not self.holding:
                return ("failed", call, "nothing is held: pick an object first")
            xyz, _ = point_to_xyz(ctx, {**obs, "head_depth": None}, a["point"])
            traj = place_on_counter(ctx, xyz[:2], arm_q, hand_q, arm=self.holding)
        else:
            return ("failed", call, f"unknown tool {name!r}")
        if traj.ok and t + traj.ticks / HZ > self.time_limit:
            return ("failed", call, f"not enough time left for a {traj.ticks / HZ:.0f} s move")
        if not traj.ok:
            return ("failed", call, traj.reason[:200])
        return ("trajectory", traj, call, checks)

    # ------------------------------------------------------------------ executing
    def _start(self, traj, call, checks, t):
        self.traj = {"call": call, "traj": traj, "checks": list(checks), "tick": 0, "t": t, "aborted": None}
        self.queue_a, self.queue_h = traj.arm.copy(), traj.hand.copy()
        if call["name"] not in ("wait",):
            self.history.append({"tool": call["name"], "args": call.get("args", {}), "arm": traj.side, "result": "executing"})
            self.segments.append({"t": round(t, 3), "tick": self.sent, "skill": call["name"], "args": call.get("args", {}),
                                  "side": traj.side, "plan_ticks": traj.ticks, "phases": traj.phases})
        self._note(t, f"execute {call['name']} {call.get('args', {})} ({traj.side} arm, {traj.ticks / HZ:.1f} s)")

    def _finish(self, t):
        tr, self.traj = self.traj, None
        name = tr["call"]["name"]
        if name == "wait":
            self.state = "think"
            return
        if tr["aborted"]:
            result = f"aborted: {tr['aborted']}"
            self.holding = None
        else:
            result = "completed"
            if name == "pick":
                self.holding = tr["traj"].side
            elif name in ("place", "place_on_counter", "pick_and_place", "go_home"):
                self.holding = None
        if self.history and self.history[-1]["result"] == "executing":
            self.history[-1]["result"] = result
        self._note(t, f"{name}: {result}")
        self.state = "settle"

    # ------------------------------------------------------------------ fast loop: closed questions during a move
    def _reflexes(self, obs, t):
        job = self.reflex_job
        if job is not None and job["result"] is not None:
            self.reflex_job = None
            if not job["cancelled"] and self.traj is not None and job["traj"] is self.traj:
                ans = (job["result"] or {}).get("holding") if isinstance(job["result"], dict) else None
                if ans and ans.get("value") is False and float(ans.get("confidence", 0)) >= self.abort_confidence:
                    self._abort(job["arm"], "the hand closed on nothing (seen in the camera images)", t)
        if (self.traj is not None and self.traj["checks"] and self.reflex is not None and self.reflex_job is None
                and self.traj["tick"] >= self.traj["checks"][0][0]):
            _, kind, arm = self.traj["checks"].pop(0)
            job = {"result": None, "cancelled": False, "traj": self.traj, "arm": arm}
            self.reflex_job = job
            self.stats["reflex_calls"] += 1
            view, questions = self.view(obs, t), holding_question(arm)

            def run():
                fut = self.reflex_pool.submit(self.reflex.answer, view, questions)
                try:
                    job["result"] = fut.result(timeout=self.reflex_timeout_s) or {}
                except Exception as e:           # no answer in time = carry on with the plan
                    fut.cancel()
                    job["result"] = {"error": repr(e)}
            if self.threaded:
                threading.Thread(target=run, daemon=True).start()
            else:
                run()

    def _abort(self, arm, why, t):
        """Stop the current move where it is (a hold is always safe), open that hand, go home."""
        self.stats["aborts"] += 1
        self.traj["aborted"] = why
        rel = set_hand(self.ctx, self.arm_cmd, self.hand_cmd, arm, "release", hold_s=0.2)
        home = go_home(self.ctx, rel.end_arm if rel.ok else self.arm_cmd, rel.end_hand if rel.ok else self.hand_cmd)
        rec = rel.then(home) if (rel.ok and home.ok) else (home if home.ok else hold(self.arm_cmd, self.hand_cmd, 0.3))
        self.queue_a, self.queue_h = rec.arm.copy(), rec.hand.copy()
        self._note(t, f"abort: {why}")

    # ------------------------------------------------------------------ plumbing
    def _enqueue(self, traj):
        self.queue_a, self.queue_h = traj.arm, traj.hand
        return self._pop()

    def _pop(self):
        k = self.chunk
        a, h = self.queue_a[:k], self.queue_h[:k]
        self.queue_a, self.queue_h = self.queue_a[k:], self.queue_h[k:]
        self.arm_cmd, self.hand_cmd = a[-1].copy(), h[-1].copy()
        self.sent += len(a)
        if self.traj is not None:
            self.traj["tick"] += len(a)
        return {"arm_q_target": a.copy(), "hand_q_target": h.copy(), "done": False}

    def _hold(self, k, done=False):
        k = 1 if done else k
        self.sent += k
        return {"arm_q_target": np.repeat(self.arm_cmd[None], k, 0), "hand_q_target": np.repeat(self.hand_cmd[None], k, 0),
                "done": bool(done)}

    def _note(self, t, msg):
        self.events.append((round(t, 2), msg))
        self.log(f"[agent t={t:6.2f}] {msg}")
