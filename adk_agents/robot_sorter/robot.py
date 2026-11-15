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
        self._arm_model = None                 # the arms' surface from the robot model, built on first use
        self.arm_cam_T = None                  # the last camera pose measured against the arms
        self.scene_offset = np.zeros(3)        # how far calibrate has moved the scene from the scene file
        self.waist = (0.0, 0.0)                # upper-body pitch, roll (rad) calibrate has put into the robot model
        self.arm_fit_model = None              # which arm model the last camera fit used
        self.calibrated = False                # calibrate has measured the scene against the arms (kept across episodes)
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
                # ROBOT_SCENE_FILE: the scene file this policy believes (default: the organisers' scene.yaml). The
                # sim tests with a moved table give the policy the original file while the sim uses the edited one.
                self._ctx = SkillContext(scene=os.environ.get("ROBOT_SCENE_FILE") or None)
                self.scene_origin0 = np.asarray(self._ctx.scene.cfg["counter"]["mat_origin_in_pelvis"], float).copy()
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
        from .headpose import camera_move_ok, fit_head_pose
        ctx = self.ctx
        cam = ctx.cal.cams[ctx.camera.cam]
        try:
            cam.T_pelvis_optical = self.head_T0.copy()
            fit = fit_head_pose(depth, ctx.cal, ctx.scene)
            # 20 deg: on the real robot's frames (data drop 3, 15 episodes) this fit reads 14.2-15.2 deg, so the
            # default limit of 15 threw one in fifteen away and the rest passed by under a degree.
            used = bool(camera_move_ok(fit, self.head_T0, max_tilt_deg=20.0))
            if used:
                cam.T_pelvis_optical = np.asarray(fit.T, float)
            self.pose_fit = {"used": used, "summary": fit.summary(), "fit_ok": bool(fit.ok),
                             "camera_moved_m": float(np.linalg.norm(np.asarray(fit.T, float)[:3, 3] - self.head_T0[:3, 3]))}
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
            self.failed = {}                   # object id -> grasp options that missed on it (never repeated)
            self.recent = []                   # the last tool results' statuses (drives the thinking level)
            self.difficulty = None             # how hard the scene looks, 0..2, from the Decisions API at look
            self.labels = {}                   # object id -> colour / shape label from the Gemini Decisions API
            self.seen, self.seen_rgb = [], None  # the last look: objects as reported to the model, and its image
            self.waiting = None                # an arm waiting above a bin for its next object ("left" | "right")
            self.placed = {}                   # bin -> objects we dropped in it (spreads the drop points)
            self.view_requests = []            # camera views the agent asked for; shown with the next model call
            self.pose_fit = None               # head camera pose fit of this episode (see fit_head_camera)
            self.arm_views, self.arm_view_t = [], -1.0   # moments with an arm still in view (see measure_scene)
            self.arm_hist = []                 # recent (t, arm_q), to tell whether the arms are standing still
            self.finished = False              # the agent called done
            self.moves = []                    # what was executed, for logs

    def _still(self, obs, window_s=0.3, tol_rad=0.003):
        """Have both arms kept their pose for the last window_s? From the joint angles themselves: the sim's joint
        speed signal never reads zero (a resting arm shows 0.1-0.2 rad/s), and the camera image can lag a tick or
        two behind the joints, so a moving arm must not be used for calibration."""
        q = np.asarray(obs["arm_q"], float)
        if self.arm_hist and self.arm_hist[-1][0] == self.t:        # the same observation again (the loop is waiting)
            self.arm_hist.pop()
        self.arm_hist = [(t, a) for t, a in self.arm_hist if self.t - t <= window_s + 0.1] + [(self.t, q)]
        old = [a for t, a in self.arm_hist if self.t - t >= window_s - 0.05]
        return bool(old) and all(float(np.abs(a - q).max()) < tol_rad for _, a in self.arm_hist)

    def _keep_arm_view(self, obs, every_s=0.4, most=10):
        """Remember a few moments at which the arms stood still in the head camera's view: the depth image and the
        joint angles of the same observation. measure_scene() calibrates the camera against the arm with them."""
        still = self._still(obs)
        if not still or self.ticks < self.warmup_ticks or obs.get("head_depth") is None or self.t - self.arm_view_t < every_s:
            return
        q = np.asarray(obs["arm_q"], float)
        if any(float(np.abs(q - a).max()) < 0.1 for _, a, _ in self.arm_views):
            return                                         # the same pose again adds nothing
        self.arm_view_t = self.t
        self.arm_views.append((np.array(obs["head_depth"]), np.asarray(obs["arm_q"], float).copy(),
                               np.asarray(obs["hand_q"], float).copy()))
        del self.arm_views[:-most]

    def measure_scene(self):
        """Camera pose from the robot's own arms, the scene from the depth image, both with their uncertainty
        (selfcal.py). Reads only; nothing the robot does changes. -> (SceneEstimate | None, ArmFit | None, why)."""
        from . import selfcal
        from .headpose import fit_head_pose
        ctx = self.ctx
        cam = ctx.cal.cams[ctx.camera.cam]
        with self._lock:
            views, depth = list(self.arm_views), None if self.obs is None else self.obs.get("head_depth")
        if depth is None:
            return None, None, "no depth image"
        K = np.asarray(cam.K, float)
        if self._arm_model is None:              # two models of the arm: without the hands (sim) and with them (real)
            self._arm_model = [selfcal.ArmModel(ctx.kin, lock=ctx.lock), selfcal.ArmModel(ctx.kin, lock=ctx.lock, hands=True)]
        used = cam.T_pelvis_optical.copy()
        try:
            cam.T_pelvis_optical = self.head_T0.copy()
            fit = fit_head_pose(np.asarray(depth), ctx.cal, ctx.scene)
        finally:
            cam.T_pelvis_optical = used
        if not fit.ok:
            return None, None, f"table not found in the depth image ({fit.why})"
        self.last_fit_T = np.asarray(fit.T, float)
        frames = [(selfcal.arm_points(d, K, cam.depth_scale), aq, hq) for d, aq, hq in views]
        frames = [(P[::max(1, len(P) // 4000)], aq, hq) for P, aq, hq in frames]      # enough for the fit, and fast
        guesses = selfcal.starts(self.head_T0, self.arm_cam_T, np.asarray(fit.T, float))
        arm = None
        for model in self._arm_model if frames else ():
            f = selfcal.camera_from_arms(frames, model, guesses)
            ok = f.ok and f.matched >= selfcal.MIN_MATCHED and f.rms_m <= selfcal.MAX_RMS_M
            if arm is None or (ok, f.matched - f.rms_m / 0.01 if f.ok else -1) > arm[0]:
                arm = ((ok, f.matched - f.rms_m / 0.01 if f.ok else -1), f, "arms and hands" if model.hands else "arms without hands")
        arm, self.arm_fit_model = (arm[1], arm[2]) if arm else (None, None)
        good = arm is not None and arm.ok and arm.matched >= selfcal.MIN_MATCHED and arm.rms_m <= selfcal.MAX_RMS_M
        if good:
            self.arm_cam_T = np.asarray(arm.T, float)      # the camera does not move on the robot: start here next time
        est = selfcal.fuse(self.head_T0, np.asarray(fit.T, float), [arm] if good else [])
        why = "" if good else ("no moment yet at which an arm stood still in the camera's view" if arm is None or not arm.ok
                               else f"arm fit too poor to use ({100 * arm.matched:.0f} % on the arm, {arm.rms_m * 1000:.1f} mm)")
        return est, arm, why

    def set_waist(self, pitch, roll):
        """Tell the robot model that the upper body (arms and head) is pitched / rolled by these angles (rad) relative
        to a pelvis frame that is level with the table. The organisers' kinematics takes waist joint angles; after
        this every position the planner uses (grasp points, the table, straight-down approaches) is in the level
        frame, and the joint targets it computes are the ones that get the real arms there."""
        ctx = self.ctx
        kin, pin = ctx.kin, ctx.kin.pin
        fid = kin.model.getFrameId("torso_link")
        with ctx.lock:
            def torso():
                pin.framesForwardKinematics(kin.model, kin.data, kin.q0)
                M = kin.data.oMf[fid]
                T = np.eye(4)
                T[:3, :3], T[:3, 3] = np.asarray(M.rotation), np.asarray(M.translation)
                return T
            before = torso()
            for k in {id(ctx.kin): ctx.kin, id(ctx.planner.kin): ctx.planner.kin}.values():
                k.q0[k._iq("waist_pitch_joint")], k.q0[k._iq("waist_roll_joint")] = float(pitch), float(roll)
            W = torso() @ np.linalg.inv(before)            # how everything on the upper body moved in the pelvis frame
            ctx.planner.__dict__.pop("_reach", None)
        self.head_T0 = W @ self.head_T0                    # the camera rides on the upper body
        if self.arm_cam_T is not None:
            self.arm_cam_T = W @ self.arm_cam_T
        ctx.cal.cams[ctx.camera.cam].T_pelvis_optical = W @ np.asarray(ctx.cal.cams[ctx.camera.cam].T_pelvis_optical, float)
        self.waist = (float(pitch), float(roll))
        self.aside = self._aside_pose(ctx)
        self.objects, self.seen = {}, []
        self.log(f"[robot t={self.t:6.2f}] upper body set to pitch {np.degrees(pitch):+.1f} deg, roll {np.degrees(roll):+.1f} deg in the robot model")
        self.trace.write("waist_set", t=round(self.t, 2), pitch_deg=float(np.degrees(pitch)), roll_deg=float(np.degrees(roll)))

    def apply_scene(self, est, fit_T, min_m=0.015, set_camera=False):
        """Use a scene measurement for the robot's motions: move the scene (table, bins) by the measured offset and
        keep the camera consistent with it. Only offsets larger than the measurement's own error are applied
        (in the nominal sim it reads up to 8 mm where the truth is 0), each of x, y, z by itself. A tilt is not
        handled here but before, in the robot model (set_waist). set_camera: take fit_T as the camera pose even if
        nothing is moved (after set_waist the camera pose in use is stale).
        -> the offset applied, metres [x, y, z] in the pelvis frame (zeros = nothing changed)."""
        ctx = self.ctx
        t = np.array([est.shift_xy_m[0], est.shift_xy_m[1], est.table_dz_m], float)
        t[np.abs(t) < min_m] = 0.0
        if not t.any() and not set_camera:
            return t
        origin = np.asarray(ctx.scene.cfg["counter"]["mat_origin_in_pelvis"], float) + t
        with ctx.lock:
            ctx.scene.cfg["counter"]["mat_origin_in_pelvis"] = [float(v) for v in origin]
            T = np.asarray(fit_T, float).copy()            # the camera relative to the scene, now in the moved scene
            T[:3, 3] += t
            ctx.cal.cams[ctx.camera.cam].T_pelvis_optical = T
            ctx.planner.__dict__.pop("_reach", None)       # which arm reaches which bin was worked out for the old scene
        self.aside = self._aside_pose(ctx)
        self.scene_offset = origin - self.scene_origin0
        self.pose_fit = {"used": True, "fit_ok": True, "camera_moved_m": 0.0,
                         "summary": f"scene moved by {np.round(self.scene_offset * 1000).astype(int).tolist()} mm after calibrate"}
        self.objects, self.seen = {}, []                   # every position measured before is in the old frame
        self.log(f"[robot t={self.t:6.2f}] scene moved by {np.round(t * 1000).astype(int).tolist()} mm (total "
                 f"{np.round(self.scene_offset * 1000).astype(int).tolist()} mm from the scene file)")
        self.trace.write("scene_applied", t=round(self.t, 2), offset_m=t.tolist(), total_m=self.scene_offset.tolist())
        return t

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
            if self.motion is not None or self.waiting or not self.seen:     # an arm is in the camera's view
                self._keep_arm_view(obs)
                self.trace.arm_frame(obs, self.t, self.motion["label"] if self.motion else "waiting" if self.waiting else "start",
                                     still=self.arm_view_t == self.t)
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
        self.arm_frames, self.arm_frame_t = 0, -1.0
        self.write("reset", **info)

    def arm_frame(self, obs, t, label, still=False, every_s=0.3, most=120):
        """Head colour + depth image together with the joint angles of the same observation, a few per second while
        an arm is in the camera's view: <dir>/arm_frames/. This is the data for calibrating the camera against the
        robot's own hand (sim/replay_selfcal.py). Written by a background thread; act() only copies the arrays."""
        if not self.dir or self.episode < 0 or obs.get("head_depth") is None or obs.get("head_rgb") is None:
            return
        if self.arm_frames >= most or t <= self.arm_frame_t or (t - self.arm_frame_t < every_s and not still):
            return
        self.arm_frames, self.arm_frame_t = self.arm_frames + 1, t
        item = (os.path.join(self.dir, "arm_frames", f"ep{self.episode:03d}_t{t:06.2f}"),
                np.array(obs["head_rgb"]), np.array(obs["head_depth"]),
                {"t": t, "move": label, "still": bool(still), "arm_q": np.asarray(obs["arm_q"], float).tolist(),
                 "hand_q": np.asarray(obs["hand_q"], float).tolist(),
                 "arm_dq": np.asarray(obs.get("arm_dq", np.zeros(14)), float).tolist()})
        if getattr(self, "_writer", None) is None:
            import queue
            self._q = queue.Queue()
            self._writer = threading.Thread(target=self._write_arm_frames, name="arm-frames", daemon=True)
            self._writer.start()
        self._q.put(item)

    def _write_arm_frames(self):
        import json
        import cv2
        while True:
            stem, rgb, depth, info = self._q.get()
            try:
                os.makedirs(os.path.dirname(stem), exist_ok=True)
                cv2.imwrite(stem + ".jpg", np.ascontiguousarray(rgb[..., ::-1]), [cv2.IMWRITE_JPEG_QUALITY, 90])
                cv2.imwrite(stem + "_depth.png", np.ascontiguousarray(depth).astype(np.uint16))
                with open(stem + ".json", "w") as f:
                    json.dump(info, f)
            except Exception:  # noqa: BLE001  (tracing must never break a run)
                pass

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

    def snapshot(self, tag, obs, t):
        """The camera images at a moment worth looking at afterwards (e.g. the grasp check after a lift)."""
        if not self.dir or self.episode < 0:
            return
        import cv2
        try:
            d = os.path.join(self.dir, "frames")
            os.makedirs(d, exist_ok=True)
            for key in ("head_rgb", "left_wrist_rgb", "right_wrist_rgb"):
                if obs.get(key) is not None:
                    cv2.imwrite(os.path.join(d, f"ep{self.episode:03d}_{tag}_t{t:06.2f}_{key[:-4]}.jpg"),
                                np.ascontiguousarray(obs[key][..., ::-1]), [cv2.IMWRITE_JPEG_QUALITY, 88])
        except Exception:  # noqa: BLE001
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
                d = np.ascontiguousarray(obs["head_depth"]).astype(np.uint16)
                cv2.imwrite(stem + "_depth.png", d)              # the data: 16-bit millimetres (looks black in a viewer)
                cv2.imwrite(stem + "_depth_view.jpg", depth_view(d))     # the same for human eyes
            with open(stem + ".json", "w") as f:
                json.dump({"t": t, "detections": detections}, f)
        except Exception:  # noqa: BLE001
            pass


def depth_view(depth_mm):
    """A depth image (uint16, millimetres) as a picture: red is near (0.3 m), blue is far (1.2 m), black = no reading."""
    import cv2
    m = np.asarray(depth_mm, np.float32) * 0.001
    near = np.clip(1.0 - (m - 0.3) / 0.9, 0.0, 1.0)
    img = cv2.applyColorMap((255 * near).astype(np.uint8), cv2.COLORMAP_JET)
    img[m <= 0] = 0
    return img


_robot = None
_robot_lock = threading.Lock()


def get_robot() -> Robot:
    global _robot
    with _robot_lock:
        if _robot is None:
            _robot = Robot()
        return _robot
