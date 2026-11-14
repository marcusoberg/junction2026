"""Who decides what to do next. A chooser sees the world as plain data and returns ONE decision:

    {"skill": "pick_and_place", "object": "<id>", "bin": "left|center|right",
     "arm": "left|right" (optional), "approach": {...Approach fields...} (optional)}
    {"skill": "relocate", "object": "<id>", "bin": "..."}          hand the object across for the other arm
    {"skill": "done"}

`world` (see SkillPolicy._world): objects (id, color, shape, where, position_m, yaw_deg, attempts), mapping
(colour -> bin), time_left_s, rejected (decisions this look that had no valid plan, with the reason), history.

    RuleChooser     the oracle's rule: the object closest to its bin first. No network, the fallback for everything.
    RandomChooser   random order and random approach parameters, for recording varied demonstrations.
    LLMChooser      asks a model (any callable world -> decision), with a timeout and a fallback chooser.
"""
from __future__ import annotations

import concurrent.futures
import time

import numpy as np

BINS_Y = {"left": 0.21, "center": 0.0, "right": -0.21}      # approximate pelvis y of the bins, for ordering only


def bin_distance(world, obj, b):
    bx, by = (world.get("bins") or {}).get(b, (0.45, BINS_Y.get(b, 0.0)))
    return float(np.hypot(obj["position_m"][0] - bx, obj["position_m"][1] - by))


def target_bin(world, obj):
    m = world["mapping"]
    for key in (f"{obj['color']} {obj.get('shape')}", obj["color"], obj.get("shape")):
        if key in m:
            return m[key]
    return None


def candidates(world, max_attempts=2):
    rejected = {(r["decision"].get("object"), r["decision"].get("skill")) for r in world.get("rejected", [])}
    out = []
    for o in world["objects"]:
        if o["where"] != "counter" or o.get("attempts", 0) >= max_attempts:
            continue
        b = target_bin(world, o)
        if b is None:
            continue
        if (o["id"], "pick_and_place") not in rejected:
            out.append((o, b, "pick_and_place"))
        elif (o["id"], "relocate") not in rejected:
            out.append((o, b, "relocate"))
    return out


class RuleChooser:
    name = "rule"

    def __init__(self, max_attempts=2):
        self.max_attempts = max_attempts

    def reset(self, instruction=None, task=None):
        pass

    def decide(self, world):
        c = candidates(world, self.max_attempts)
        if not c:
            return {"skill": "done"}
        # direct moves before hand-overs, then the object nearest its bin
        o, b, skill = min(c, key=lambda t: (t[2] != "pick_and_place", bin_distance(world, t[0], t[1])))
        return {"skill": skill, "object": o["id"], "bin": b}


class RandomChooser(RuleChooser):
    """Varied but valid behaviour for demonstrations: any sortable object next, random approach parameters.
    `variety` in 0..1 scales how far the parameters stray from the oracle's."""
    name = "random"

    def __init__(self, seed=0, variety=1.0, max_attempts=2):
        super().__init__(max_attempts)
        self.rng, self.variety = np.random.default_rng(seed), float(variety)

    def decide(self, world):
        c = candidates(world, self.max_attempts)
        if not c:
            return {"skill": "done"}
        direct = [t for t in c if t[2] == "pick_and_place"] or c
        o, b, skill = direct[int(self.rng.integers(len(direct)))]
        d = {"skill": skill, "object": o["id"], "bin": b}
        if skill == "pick_and_place":
            r, v = self.rng, self.variety
            ap = {"speed": float(r.uniform(0.97 - 0.42 * v, 0.97))}
            if r.random() < 0.5 * v:
                ap["pregrasp_offset"] = [float(r.uniform(-0.04, 0.02)), float(r.uniform(-0.03, 0.03))]
            if r.random() < 0.3 * v:
                ap["carry_height"] = float(r.uniform(0.16, 0.20))
            if r.random() < 0.3 * v:
                ap["grasp_height_frac"] = float(r.uniform(0.45, 0.6))
            d["approach"] = ap
            d["drop_offset"] = [float(r.uniform(-0.03, 0.03)), float(r.uniform(-0.03, 0.03))]
        return d


class LLMChooser:
    """`ask(world) -> decision dict` is any function that calls a model (see agent/skills/tools.py for the tool
    declarations and a prompt). It runs with a timeout; a late, failed or invalid answer falls back to `fallback`
    for that decision. `validate` is applied by SkillPolicy as well (unknown object / bin -> rejected)."""
    name = "llm"

    def __init__(self, ask, fallback=None, timeout_s=4.0, log=None):
        self.ask, self.fallback, self.timeout_s = ask, fallback or RuleChooser(), float(timeout_s)
        self.pool = concurrent.futures.ThreadPoolExecutor(1)
        self.log = log or (lambda *a: None)
        self.stats = {"asked": 0, "used": 0, "fallback": 0, "latency_s": []}

    def reset(self, instruction=None, task=None):
        self.fallback.reset(instruction, task)

    def decide(self, world):
        self.stats["asked"] += 1
        t0 = time.monotonic()
        fut = self.pool.submit(self.ask, world)
        try:
            d = fut.result(timeout=self.timeout_s)
            if not isinstance(d, dict) or "skill" not in d:
                raise ValueError(f"not a decision: {str(d)[:80]}")
            self.stats["used"] += 1
            self.stats["latency_s"].append(time.monotonic() - t0)
            return d
        except Exception as e:                # timeout, network, bad answer: never stall the episode
            fut.cancel()
            self.stats["fallback"] += 1
            self.log(f"[llm] falling back to {self.fallback.name}: {e!r}")
            return self.fallback.decide(world)
