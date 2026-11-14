"""Policy server for the Aaltonomous Humanoid submission: a chunking VLA (pi0.5 via LeRobot) behind the
organisers' websocket protocol, built so the two sim-gate rules hold whatever the model does:

  * >= 27 Hz command rate   act() never waits for the model. Inference runs in a background thread; act() hands
                            out the next few ticks of the current plan, or hold targets when there is none.
  * 0 clipped steps         every target goes through the organisers' own SafetyFilter here first (with a slightly
                            lower speed cap), so what the runner's filter receives is already inside its limits.

Run
    PYTHONPATH=<starter repo root> python submission/server.py                  # pi0.5 from $AALTO_CKPT
    PYTHONPATH=<starter repo root> AALTO_MODEL=stub python submission/server.py # no weights: plumbing tests

Environment
    AALTO_MODEL       pi05 | stub                       (default pi05)
    AALTO_CKPT        checkpoint dir = lerobot-train's .../checkpoints/<step>/pretrained_model   (default /app/weights)
    AALTO_DEVICE      cuda | cpu | mps                  (default cuda if available)
    AALTO_ACT_TICKS   ticks returned per act() call     (default 8: the model sees an observation every 0.27 s).
                      Do not go low: the evaluator's "policy real-time rate" comes out as about
                      ACT_TICKS x 30 x (sim real-time factor), whatever our speed. Measured 34-38 Hz with 4 ticks
                      at a real-time factor of 0.30; the gate needs 27.
    AALTO_EXEC_TICKS  ticks of a chunk executed before the next one is requested   (default 25 of 50)
    AALTO_BLEND       ticks over which a new chunk is cross-faded into the old one  (default 6)
    AALTO_LATENCY_TICKS  extra ticks skipped at the start of a new chunk (observation age)   (default 1)
    AALTO_SAMPLES        predictions drawn per inference and combined (see combine_samples)           (default 1)
    AALTO_WARMUP_TICKS   ticks to hold at episode start before the first inference: the sim's first camera
                         frames after a reset still show the previous layout                (default 15)
    AALTO_ARM_VEL / AALTO_HAND_VEL   local speed caps, rad/s   (default 0.98 / 2.7; the runner allows 1.0 / 3.0)
    PORT              (default 8000)
"""
import logging
import os
import threading
import time
from pathlib import Path

import numpy as np

from aaltonomous.deploy.policy_server import Policy, serve
from aaltonomous.interface.safety import JointSpec, SafetyConfig, SafetyFilter

log = logging.getLogger("submission")
DT = 1.0 / 30.0
CAMERAS = ("head_rgb", "left_wrist_rgb", "right_wrist_rgb")
# camera names used by some data drops -> the names in the runner's observation
CAMERA_ALIASES = {"cam_high": "head_rgb", "cam_left_wrist": "left_wrist_rgb", "cam_right_wrist": "right_wrist_rgb"}


def obs_camera(feature_key):
    """'observation.images.cam_high' -> 'head_rgb' (the key of that image in the runner's obs)."""
    name = feature_key.rsplit(".", 1)[-1]
    return CAMERA_ALIASES.get(name, name)


def env_num(name, default, cast=float):
    return cast(os.environ.get(name, default))


# ------------------------------------------------------------------------------------------------ models
class StubModel:
    """No weights: a slow wave around the measured pose, after a fake inference delay. For plumbing tests."""
    name, horizon = "stub", 50

    def __init__(self, latency_s=0.15, amplitude=0.6):
        self.latency_s, self.amplitude = latency_s, amplitude

    def infer(self, obs, instruction):
        time.sleep(self.latency_s)
        q = np.concatenate([obs["arm_q"], obs["hand_q"]]).astype(np.float64)
        phase = 2 * np.pi * (float(obs["t"]) + np.arange(self.horizon) * DT) / 4.0
        # deliberately far too fast and partly out of range: the safety layer has to make it legal
        return q[None] + self.amplitude * np.sin(phase)[:, None] * np.linspace(1.0, -1.0, 28)[None]


def combine_samples(chunks):
    """(n, horizon, 28) predictions for ONE observation -> one chunk. A single sample scatters by a few
    hundredths of a radian around what the model means (measured: centimetres at the hand); averaging removes
    that, but averaging samples that chose DIFFERENT objects gives a move toward neither. So: average only the
    half of the samples that lie closest to the per-joint median (the majority choice)."""
    if len(chunks) == 1:
        return chunks[0]
    dist = np.abs(chunks[:, :, :14] - np.median(chunks[:, :, :14], axis=0)).mean(axis=(1, 2))
    keep = np.argsort(dist)[:max(1, len(chunks) // 2)]
    return chunks[keep].mean(axis=0)


class Pi05Model:
    """A lerobot-train pi0.5 checkpoint. One observation in, one (horizon, 28) chunk of absolute joint targets out."""
    name = "pi05"

    def __init__(self, ckpt, device=None, samples=None):
        os.environ.setdefault("HF_HUB_OFFLINE", "1")          # the finals machine has no network
        self.samples = int(samples if samples is not None else env_num("AALTO_SAMPLES", 1, int))
        import torch
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy

        ckpt = Path(ckpt)
        if not (ckpt / "model.safetensors").exists():
            raise SystemExit(f"no checkpoint in {ckpt} (expected lerobot's pretrained_model folder)")
        if not (ckpt / "tokenizer").is_dir():
            log.warning("%s has no tokenizer/ folder: the tokenizer would have to be downloaded, which fails offline", ckpt)
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        t0 = time.monotonic()
        self.policy = PI05Policy.from_pretrained(str(ckpt)).to(self.device).eval()
        self.pre, self.post = make_pre_post_processors(
            self.policy.config, pretrained_path=str(ckpt),
            preprocessor_overrides={"device_processor": {"device": self.device}})
        cfg = self.policy.config
        self.horizon = int(cfg.chunk_size)
        self.image_keys = list(cfg.image_features)
        missing = [k for k in self.image_keys if obs_camera(k) not in CAMERAS]
        if missing:
            raise SystemExit(f"checkpoint expects cameras the robot does not have: {missing}")
        if self.device == "cuda":
            self.name = f"pi05 ({torch.cuda.get_device_name(0)}, torch {torch.__version__}, cuda {torch.version.cuda})"
        log.info("loaded %s on %s in %.1f s; cameras %s; chunk %d", ckpt, self.device, time.monotonic() - t0,
                 [f"{k.rsplit('.', 1)[-1]}<-{obs_camera(k)}" for k in self.image_keys], self.horizon)

    def infer(self, obs, instruction):
        torch = self.torch
        batch = {"observation.state": torch.from_numpy(np.concatenate([obs["arm_q"], obs["hand_q"]]).astype(np.float32)),
                 "task": instruction}
        for key in self.image_keys:                           # uint8 HWC RGB -> float CHW in 0..1, as in training
            img = np.ascontiguousarray(obs[obs_camera(key)])
            batch[key] = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        with torch.inference_mode():
            x = self.pre(batch)
            if self.samples > 1:                              # same observation, independent flow-matching noise
                x = {k: (v.repeat_interleave(self.samples, 0) if torch.is_tensor(v) else
                         v * self.samples if isinstance(v, list) else v) for k, v in x.items()}
            chunks = self.post(self.policy.predict_action_chunk(x)).float().cpu().numpy().astype(np.float64)
        return combine_samples(chunks)

    def warm_up(self):
        blank = {k: np.zeros((480, 640, 3), np.uint8) for k in CAMERAS}
        t0 = time.monotonic()
        self.infer({**blank, "arm_q": np.zeros(14, np.float32), "hand_q": np.zeros(14, np.float32)}, "warm up")
        t1 = time.monotonic()
        self.infer({**blank, "arm_q": np.zeros(14, np.float32), "hand_q": np.zeros(14, np.float32)}, "warm up")
        log.info("warm-up inference %.2f s, then %.3f s per chunk", t1 - t0, time.monotonic() - t1)


class BackgroundLoaded:
    """Loads the real model in a thread so the port opens at once (startup limit: listening within 120 s; a
    14.5 GB checkpoint took 105 s to load from a network volume in our test). Until it is ready the executor
    simply holds. If loading fails the process exits, so the failure is visible instead of an endless hold."""

    def __init__(self, factory, name, horizon=50):
        self.name, self.horizon, self.model = name, horizon, None
        self.ready = threading.Event()
        threading.Thread(target=self._load, args=(factory,), name="model-load", daemon=True).start()

    def _load(self, factory):
        try:
            t0 = time.monotonic()
            self.model = factory()
            self.horizon = self.model.horizon
            log.info("model ready %.1f s after start: %s", time.monotonic() - t0, self.model.name)
            self.ready.set()
        except BaseException:
            log.exception("model failed to load; exiting")
            os._exit(1)

    def infer(self, obs, instruction):
        self.ready.wait()
        return self.model.infer(obs, instruction)


# ------------------------------------------------------------------------------------------------ executor
def make_shadow_filter(arm_vel, hand_vel):
    """The organisers' filter with our own (lower) speed caps and no abort: used to pre-shape every target."""
    logging.getLogger("aaltonomous.safety").setLevel(logging.ERROR)   # clipping here is routine, not an event
    cfg = SafetyConfig(arm_max_vel=arm_vel, hand_max_vel=hand_vel, max_clipped_steps=10 ** 9)
    try:
        return SafetyFilter(JointSpec(), cfg)
    except Exception as e:   # no Pinocchio / URDF: limits and speed are still enforced, the workspace box is not
        log.error("workspace check unavailable (%s: %s): running WITHOUT the workspace pre-check", type(e).__name__, e)
        cfg.workspace = False
        return SafetyFilter(JointSpec(), cfg)


class ChunkPolicy(Policy):
    def __init__(self, model):
        self.model = model
        self.act_ticks = env_num("AALTO_ACT_TICKS", 8, int)
        self.exec_ticks = env_num("AALTO_EXEC_TICKS", 25, int)
        self.blend = env_num("AALTO_BLEND", 6, int)
        self.latency_ticks = env_num("AALTO_LATENCY_TICKS", 1, int)
        self.warmup_ticks = env_num("AALTO_WARMUP_TICKS", 15, int)
        self.shadow = make_shadow_filter(env_num("AALTO_ARM_VEL", 0.98), env_num("AALTO_HAND_VEL", 2.7))
        self.name = f"chunk-vla/{model.name}"
        self.action_horizon = self.act_ticks
        self.lock = threading.Lock()
        self.wake = threading.Condition(self.lock)
        self.episode = 0            # bumped on reset: results of an older episode are dropped
        self.instruction = ""
        self._clear()
        threading.Thread(target=self._infer_loop, name="inference", daemon=True).start()

    def _clear(self):
        self.plan = np.zeros((0, 28))   # future targets, row 0 = next tick to send
        self.plan_age = 0               # ticks sent since the current plan was installed
        self.sent = 0                   # ticks handed to the runner this episode
        self.request = None             # (obs, sent, episode) waiting for the inference thread
        self.busy = False
        self.started = False
        self.lag = 6.0                  # ticks that go out while one inference runs (running average)
        self.stats = {"chunks": 0, "hold_ticks": 0, "shaped_ticks": 0, "latency": [], "reasons": {}}

    # -- protocol ------------------------------------------------------------------------------------
    def reset(self, instruction, task=None, seed=None):
        with self.lock:
            if self.sent:
                s, lat = self.stats, self.stats["latency"]
                log.info("episode end: %d ticks, %d chunks, %d hold ticks, %d ticks shaped by the safety layer, "
                         "inference %.0f ms median; shaped by: %s", self.sent, s["chunks"], s["hold_ticks"],
                         s["shaped_ticks"],
                         1e3 * float(np.median(lat)) if lat else float("nan"), s["reasons"] or "nothing")
            self.episode += 1
            self.instruction = instruction
            self._clear()
        log.info("reset: task=%s instruction=%r", task, instruction)

    def act(self, obs):
        with self.lock:
            if not self.started:        # the runner's filter starts from the measured pose: so do we
                self.shadow.reset(np.asarray(obs["arm_q"], float), np.asarray(obs["hand_q"], float), 0.0)
                self.started = True
            remaining = len(self.plan)
            # ask early enough that the next chunk lands before this one runs out
            want_new = self.plan_age >= self.exec_ticks or remaining <= self.lag + 2 * self.act_ticks
            # The sim's cameras lag a scene reset: the first observations of an episode still show the PREVIOUS
            # layout (seen in the runner's video: frames 0-1). Hold for warmup_ticks before the first inference.
            if want_new and not self.busy and self.sent >= self.warmup_ticks:
                self.request = (obs, self.sent, self.episode)
                self.wake.notify()
            arm, hand = [], []
            for _ in range(self.act_ticks):
                if len(self.plan):
                    target, self.plan = self.plan[0], self.plan[1:]
                else:                   # nothing planned yet / model too slow: hold where we are
                    target = np.concatenate([self.shadow.arm_cmd, self.shadow.hand_cmd])
                    self.stats["hold_ticks"] += 1
                res = self.shadow.filter({"arm_q_target": target[:14], "hand_q_target": target[14:]},
                                         self.sent * DT)
                self.stats["shaped_ticks"] += bool(res.clipped)
                for r in res.reasons:   # which rule shaped the target ("left_elbow velocity" -> "velocity")
                    key = str(r).split(":")[0].split()[-1]
                    self.stats["reasons"][key] = self.stats["reasons"].get(key, 0) + 1
                arm.append(res.arm_q_cmd)
                hand.append(res.hand_q_cmd)
                self.sent += 1
                self.plan_age += 1
            self.shadow.events.clear()
        return {"arm_q_target": np.stack(arm), "hand_q_target": np.stack(hand)}

    # -- background inference --------------------------------------------------------------------------
    def _infer_loop(self):
        while True:
            with self.lock:
                while self.request is None:
                    self.wake.wait()
                (obs, sent_at, episode), self.request = self.request, None
                instruction, self.busy = self.instruction, True
            t0 = time.monotonic()
            try:
                chunk = np.asarray(self.model.infer(obs, instruction), float)
                if chunk.ndim != 2 or chunk.shape[1] != 28 or not np.all(np.isfinite(chunk)):
                    raise ValueError(f"bad chunk from the model: shape {chunk.shape}, finite {np.isfinite(chunk).all()}")
            except Exception:
                log.exception("inference failed; holding")
                chunk = None
            with self.lock:
                self.busy = False
                if chunk is None or episode != self.episode:
                    continue
                self.stats["latency"].append(time.monotonic() - t0)
                # chunk[0] belongs to the moment of the observation; `skip` ticks have gone out since then
                self.lag = 0.7 * self.lag + 0.3 * min(self.sent - sent_at, len(chunk) // 2)
                skip = (self.sent - sent_at) + self.latency_ticks
                new = chunk[skip:]
                if len(new) < self.act_ticks:
                    log.warning("chunk arrived too late to use (%d ticks old, horizon %d)", skip, len(chunk))
                    continue
                n = min(self.blend, len(self.plan), len(new))
                if n:                   # cross-fade from what we were about to do into the new chunk
                    w = (np.arange(1, n + 1) / (n + 1))[:, None]
                    new[:n] = (1 - w) * self.plan[:n] + w * new[:n]
                self.plan, self.plan_age = new, 0
                self.stats["chunks"] += 1


def build_model():
    kind = os.environ.get("AALTO_MODEL", "pi05")
    if kind == "stub":
        return StubModel(latency_s=env_num("AALTO_STUB_LATENCY", 0.15))
    if kind != "pi05":
        raise SystemExit(f"AALTO_MODEL={kind!r}: expected pi05 or stub")
    ckpt = Path(os.environ.get("AALTO_CKPT", "/app/weights"))
    if not (ckpt / "model.safetensors").exists():       # checked up front: a missing checkpoint should not start
        raise SystemExit(f"no checkpoint in {ckpt} (expected lerobot's pretrained_model folder)")

    def load():
        model = Pi05Model(ckpt, os.environ.get("AALTO_DEVICE"))
        model.warm_up()                 # first CUDA call done before any episode needs it
        return model

    return BackgroundLoaded(load, "pi05")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    serve(ChunkPolicy(build_model()), port=env_num("PORT", 8000, int))
