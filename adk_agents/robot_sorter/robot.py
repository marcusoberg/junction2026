"""The real-time side of the ADK agent: one Robot object shared by the 30 Hz control loop and the agent's tools.

    30 Hz loop (policy server, or the mock world below)        ADK agent (model + tools, its own thread / event loop)
        action = robot.act(obs)   never waits                      fut = robot.start(trajectory)   from a tool
            hands out ticks of the current trajectory,             await fut                       -> "completed"
            or holds the last command                              robot.obs                       latest cameras

A tool plans a trajectory with the oracle-based skills (agent/skills), which is slow (0.5-3 s) and is done off the
control loop; the robot holds still meanwhile. Only a trajectory that passed the planner's checks and a zero-clip
dry run through the organisers' safety filter is ever handed to start().

MockWorld runs the organisers' mock world (rendered head RGB-D, simple grasp physics) at 30 Hz in a thread, so the
agent can be developed in `adk web` with no simulator. With a real simulator or robot the organisers' evaluator
drives the loop instead: see serve.py.
"""
from __future__ import annotations

import concurrent.futures
import os
import threading

import numpy as np

HZ = 30.0


class Robot:
    def __init__(self):
        self._lock = threading.RLock()
        self._ctx, self._ctx_lock = None, threading.Lock()
        self._ready = threading.Event()        # set once camera frames of the current episode can be trusted
        self.aside = None                      # (14,) arm pose with both arms parked out of the way; see _aside_pose
        self.external = False                  # True: a policy server feeds act() (serve.py); False: MockWorld
        self.world = None
        self.session = None                    # ADK session the mock episode belongs to
        self.warmup_ticks = 15                 # the sim's first frames after a reset still show the previous scene
        self.default_time_limit = float(os.environ.get("AALTO_TIME_LIMIT", 120.0))
        self.log = lambda msg: print(msg, flush=True)
        self.trace = Trace()
        self.motion = None
        self.reset()

    # ------------------------------------------------------------------ skills context (heavy: URDF, planner)
    @property
    def ctx(self):
        with self._ctx_lock:
            if self._ctx is None:
                from agent.skills import SkillContext
                self._ctx = SkillContext()
                if os.environ.get("ROBOT_ANY_COLOUR", "1") == "1":
                    # The organisers' detector knows the sim's red / yellow / blue. The real objects are purple,
                    # yellow and orange and the held-out sim may differ, so cover the rest of the hue circle
                    # too (OpenCV H 0-180). Their ranges are kept as tuned; bins are masked out by position.
                    self._ctx.camera.cfg["hsv"].update({
                        "orange": [[[9, 70, 70], [17, 255, 255]]], "green": [[[36, 70, 60], [85, 255, 255]]],
                        "cyan": [[[86, 70, 60], [99, 255, 255]]], "purple": [[[131, 60, 50], [155, 255, 255]]],
                        "pink": [[[156, 70, 70], [169, 255, 255]]]})
                    self._ctx.camera = _any_colour_camera(self._ctx)
                self.aside = self._aside_pose(self._ctx)
                self.head_T0 = np.array(self._ctx.cal._T(self._ctx.camera.cam), float)     # the calibration file's pose
            return self._ctx

    def fit_head_camera(self, depth):
        """Once per episode, at the first look: re-fit the head camera's pose from the depth image itself (table
        plane + bin walls, bot-icelli's headpose.py) and use it if the correction is plausible. The calibration
        file's pose is a URDF default; the sim's randomisation moves the camera by ~1 cm / 1 deg and the real
        camera is reported to point about 10 deg further down. Without this the bins' walls leak out of their
        masked footprint as false objects and positions are off by 1-2 cm."""
        if self.pose_fit is not None or depth is None or os.environ.get("ROBOT_HEADPOSE", "1") != "1":
            return
        from .headpose import fit_head_pose, plausible
        ctx = self.ctx
        cam = ctx.cal.cams[ctx.camera.cam]
        try:
            cam.T_pelvis_optical = self.head_T0.copy()
            fit = fit_head_pose(depth, ctx.cal, ctx.scene)
            used = bool(plausible(fit))
            if used:
                cam.T_pelvis_optical = np.asarray(fit.T, float)
            self.pose_fit = {"used": used, "summary": fit.summary()}
        except Exception as e:  # noqa: BLE001
            cam.T_pelvis_optical = self.head_T0.copy()
            self.pose_fit = {"used": False, "summary": f"fit failed: {e!r}"}
        self.log(f"[robot t={self.t:6.2f}] head camera pose fit: {'used' if self.pose_fit['used'] else 'NOT used'}; {self.pose_fit['summary']}")
        self.trace.write("head_pose_fit", t=round(self.t, 2), **self.pose_fit)

    @staticmethod
    def _aside_pose(ctx):
        """Where an arm rests between moves: hand above the counter, well out to its own side. At the task's home
        pose the two hands hover 13 cm apart in front of the robot, inside the head camera's view, and an arm
        reaching for the middle passes within 10 cm of the other hand (run adk-1: the right wrist camera mount hit
        the left thumb). Parked aside, the arms stay about 30 cm apart and the camera sees the whole counter.
        None (feature off) if no such pose is reachable or ROBOT_PARK=0."""
        if os.environ.get("ROBOT_PARK", "1") != "1":
            return None
        try:
            q = np.asarray(ctx.home, float).copy()
            for side, sign in (("left", 1.0), ("right", -1.0)):
                sol, ok = ctx.kin.solve(side, np.array([0.27, sign * 0.33, ctx.scene.counter_z + 0.22]), ctx.home)[:2]
                if not ok:
                    return None
                sol = np.asarray(sol, float)
                sl = ctx.kin.sl(side)
                q[sl] = sol[sl] if len(sol) == len(q) else sol
            return q
        except Exception:  # noqa: BLE001
            return None

    # ------------------------------------------------------------------ episode
    def reset(self, instruction=None, time_limit="default"):
        with self._lock:
            self._cancel("the episode was reset")
            self._ready.clear()
            self.instruction = instruction
            self.time_limit = self.default_time_limit if time_limit == "default" else time_limit
            self.obs, self.t, self.ticks = None, 0.0, 0
            self.arm_cmd = self.hand_cmd = None
            self.queue_a = self.queue_h = None
            self.holding = None                # "left" | "right" | None
            self.held_id = None                # id of the object in that hand, if it came from a look
            self.objects = {}                  # id -> WorldObject, from the last look
            self.id_count = {}                 # ids handed out, so an id is never reused in an episode
            self.tries = {}                    # object id -> grasps that missed (the next one uses another grip)
            self.labels = {}                   # object id -> colour / shape label from the Gemini Decisions API
            self.placed = {}                   # bin -> objects we dropped in it (spreads the drop points)
            self.view_requests = []            # camera views the agent asked for; shown with the next model call
            self.pose_fit = None               # head camera pose fit of this episode (see fit_head_camera)
            self.finished = False              # the agent called done
            self.moves = []                    # what was executed, for logs

    def time_left(self):
        return None if self.time_limit is None else max(0.0, self.time_limit - self.t)

    # ------------------------------------------------------------------ 30 Hz side
    def act(self, obs):
        """One request of the control loop. Returns a chunk of absolute joint targets; never waits for anything."""
        ctx = self.ctx
        with self._lock:
            self.obs, self.t = obs, float(obs.get("t", 0.0))
            if self.arm_cmd is None:
                sp = ctx.spec
                self.arm_cmd = np.clip(np.asarray(obs["arm_q"], float), sp.arm_lower, sp.arm_upper)
                self.hand_cmd = np.clip(np.asarray(obs["hand_q"], float), sp.hand_lower, sp.hand_upper)
            if self.queue_a is not None and len(self.queue_a):
                k = int(ctx.cfg["motion"]["chunk"])
                a, h = self.queue_a[:k], self.queue_h[:k]
                self.queue_a, self.queue_h = self.queue_a[k:], self.queue_h[k:]
                self.arm_cmd, self.hand_cmd = a[-1].copy(), h[-1].copy()
                self.ticks += len(a)
                return {"arm_q_target": a.copy(), "hand_q_target": h.copy(), "done": False}
            if self.motion is not None:        # every tick was sent, and `obs` was taken after the last one
                self._finish("completed")
            if self.ticks >= self.warmup_ticks:
                self._ready.set()
            k = 1 if self.finished else max(1, int(ctx.cfg["motion"].get("hold_ticks", 1)))
            self.ticks += k
            return {"arm_q_target": np.repeat(self.arm_cmd[None], k, 0),
                    "hand_q_target": np.repeat(self.hand_cmd[None], k, 0), "done": bool(self.finished)}

    # ------------------------------------------------------------------ tool side
    @property
    def busy(self):
        return self.motion is not None

    def command_state(self):
        """The joint targets last commanded: every plan must start from these (the safety filter limits the step)."""
        with self._lock:
            if self.arm_cmd is None:
                raise RuntimeError("no observation from the robot yet")
            return self.arm_cmd.copy(), self.hand_cmd.copy()

    def start(self, traj, label, planned_from, settle=True):
        """Hand a checked trajectory to the 30 Hz loop. Returns a Future -> "completed" | "cancelled: <why>"."""
        with self._lock:
            if self.motion is not None:
                raise RuntimeError(f"the robot is still executing {self.motion['label']}")
            if not (np.allclose(planned_from[0], self.arm_cmd, atol=1e-6)
                    and np.allclose(planned_from[1], self.hand_cmd, atol=1e-6)):
                raise RuntimeError("the robot moved while this was being planned; plan again")
            n = max(1, int(round(float(self.ctx.cfg["motion"]["settle_s"]) * HZ))) if settle else 1   # stand still before the next look
            self.queue_a = np.vstack([traj.arm, np.repeat(traj.arm[-1:], n, 0)])
            self.queue_h = np.vstack([traj.hand, np.repeat(traj.hand[-1:], n, 0)])
            fut = concurrent.futures.Future()
            self.motion = {"label": label, "future": fut, "t": self.t, "arm": traj.side, "ticks": traj.ticks}
            self.log(f"[robot t={self.t:6.2f}] start {label} ({traj.side or 'both'} arm, {traj.ticks / HZ:.1f} s)")
            return fut

    def _finish(self, outcome):
        m, self.motion = self.motion, None
        self.moves.append({"t": round(m["t"], 2), "move": m["label"], "arm": m["arm"], "outcome": outcome})
        self.trace.write("motion", t=round(m["t"], 2), t_end=round(self.t, 2), move=m["label"], arm=m["arm"], outcome=outcome)
        self.log(f"[robot t={self.t:6.2f}] {m['label']}: {outcome}")
        if not m["future"].done():
            m["future"].set_result(outcome)

    def _cancel(self, why):
        """Drop what is left of the current move. The robot then holds its last command, which is always safe."""
        if self.motion is not None:
            self.queue_a = self.queue_h = None
            self._finish(f"cancelled: {why}")

    # ------------------------------------------------------------------ who drives the loop
    def ensure_running(self, session=None, timeout=90.0):
        """Blocking (call it off the event loop). Starts the mock world if no policy server feeds act(), starts a
        new mock episode when the ADK session changes, and waits until the camera frames can be trusted."""
        if not self.external:
            if self.world is None:
                self.world = MockWorld(self, seed=int(os.environ.get("ROBOT_MOCK_SEED", "0")))
                self.world.start()
            elif session is not None and self.session not in (None, session):
                self.world.new_episode()
            if session is not None:
                self.session = session
        return self._ready.wait(timeout)


class MockWorld:
    """The organisers' mock world stepped in real time, calling robot.act() the way the evaluator would."""

    def __init__(self, robot: Robot, seed=0):
        self.robot, self.seed, self.env = robot, seed, None
        self._new, self._started, self._stop = threading.Event(), threading.Event(), False
        self.thread = threading.Thread(target=self._run, name="mock-world", daemon=True)

    def start(self):
        self.thread.start()

    def new_episode(self, timeout=30.0):
        self._started.clear()
        self._new.set()
        self._started.wait(timeout)

    def outcomes(self):
        """Where each object ended up (correct | wrong | counter | off): only the mock world can tell."""
        from aaltonomous.interface.tasks import FIXED_MAPPING
        env = self.env
        return list(env.backend.outcomes(env.mapping or FIXED_MAPPING)) if env is not None else []

    def _run(self):
        from aaltonomous.baselines.classical.mock_world import make_mock_env
        self.robot.ctx                                  # build the planner before the clock starts
        while not self._stop:
            env = make_mock_env(seed=self.seed, realtime=True)
            env.time_limit = float("inf")               # interactive: no 120 s limit
            obs = env.reset()
            self.env = env
            self.robot.reset(obs["instruction"], time_limit=None)
            self.robot.log(f"[mock] episode seed {self.seed}: {len(env.backend.objects)} objects; {obs['instruction']}")
            self._new.clear()
            self._started.set()
            while not self._stop and not self._new.is_set():
                action = self.robot.act(obs)
                action["done"] = False                  # `done` ends the agent's work, not the mock world
                obs, info = env.step(action)
                if info["episode_over"]:
                    self.robot.log(f"[mock] episode ended: {info['end_reason']}; starting another")
                    break
            self.seed += 1


def _any_colour_camera(ctx):
    """The organisers' detector with one more mask: every pixel that none of the hue ranges claims. The detector
    itself keeps only pixels whose depth puts them at object height above the counter and outside the bins, so
    this adds objects of ANY colour, including washed-out, white, grey or black ones, found by their geometry."""
    import cv2
    from aaltonomous.baselines.classical.perception import Perception

    class AnyColour(Perception):
        def masks(self, rgb):
            out = super().masks(rgb)
            claimed = np.zeros(np.asarray(rgb).shape[:2], np.uint8)
            for m in out.values():
                claimed |= m
            out["other"] = cv2.bitwise_not(claimed)
            return out

    return AnyColour(ctx.cfg, ctx.scene, ctx.cal)


class Trace:
    """Everything the agent did, untruncated, for working out afterwards what failed and why.
    <dir>/episode_NNN.jsonl: one JSON object per line (reset, tool call with full arguments, full tool result,
    what the model said, model errors, each robot motion). <dir>/frames/: the head colour image (jpg) and depth
    image (16-bit png, mm) behind every detection, with the detections next to them (json)."""

    def __init__(self):
        self.dir = os.environ.get("ROBOT_TRACE_DIR", "")
        self.episode, self.frame, self._lock = -1, 0, threading.Lock()

    def reset(self, **info):
        self.episode += 1
        self.frame = 0
        self.write("reset", **info)

    def write(self, kind, **data):
        if not self.dir or self.episode < 0:
            return
        import json
        import time
        try:
            os.makedirs(self.dir, exist_ok=True)
            with self._lock, open(os.path.join(self.dir, f"episode_{self.episode:03d}.jsonl"), "a") as f:
                f.write(json.dumps({"wall": round(time.time(), 2), "kind": kind, **data}, default=str) + "\n")
        except Exception:  # noqa: BLE001  (tracing must never break a run)
            pass

    def frame_dump(self, obs, t, detections):
        if not self.dir or self.episode < 0:
            return
        import json
        import cv2
        try:
            d = os.path.join(self.dir, "frames")
            os.makedirs(d, exist_ok=True)
            stem = os.path.join(d, f"ep{self.episode:03d}_{self.frame:02d}_t{t:06.2f}")
            self.frame += 1
            cv2.imwrite(stem + ".jpg", np.ascontiguousarray(obs["head_rgb"][..., ::-1]), [cv2.IMWRITE_JPEG_QUALITY, 92])
            if obs.get("head_depth") is not None:
                cv2.imwrite(stem + "_depth.png", np.ascontiguousarray(obs["head_depth"]).astype(np.uint16))
            with open(stem + ".json", "w") as f:
                json.dump({"t": t, "detections": detections}, f)
        except Exception:  # noqa: BLE001
            pass


_robot = None
_robot_lock = threading.Lock()


def get_robot() -> Robot:
    global _robot
    with _robot_lock:
        if _robot is None:
            _robot = Robot()
        return _robot
