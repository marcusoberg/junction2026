"""Offline tests of the model-driven loop (AgentPolicy) with stand-in models: no network, no key, no simulator.

    PYTHONPATH=aaltonomous-humanoid:. python -m unittest agent.tests.test_agent -v

PointingBrain plays the tool-calling model: it knows where the mock world's objects are and answers the way a
vision model would, by POINTING at one in the head image ([y, x], 0..1000) and naming a bin. What is tested is
everything on our side of that call: image packaging, pointing -> depth -> 3D target, planning, execution,
the mid-motion reflex and the abort, and that act() never waits.
"""
import time
import unittest

import numpy as np

from aaltonomous.baselines.classical.mock_world import make_mock_env
from aaltonomous.interface.tasks import FIXED_MAPPING

from agent.runtime.agent_policy import TOOLS, AgentPolicy
from agent.skills import SkillContext
from agent.skills.perception import point_to_xyz, xyz_to_point


class PointingBrain:
    def __init__(self, ctx, split=False, delay=0.0):
        self.ctx, self.split, self.delay = ctx, split, delay
        self.env, self.tried, self.carrying, self.views = None, {}, None, 0

    def decide(self, view):
        if self.delay:
            time.sleep(self.delay)
        self.views += 1
        assert view["images"][0][0] == "head camera" and view["images"][0][1][:2] == b"\xff\xd8"      # a real JPEG
        sc, b = self.ctx.scene, self.env.backend
        if view["state"]["holding"] != "nothing":
            return {"name": "place", "args": {"bin": FIXED_MAPPING[self.carrying]}}
        held = {h[0] for h in b.held.values() if h is not None}
        for i, o in enumerate(b.objects):
            m = sc.pelvis_to_mat(o["pos"])
            if i in held or sc.bin_of(m) is not None or not sc.on_counter(m) or self.tried.get(i, 0) >= 2:
                continue
            self.tried[i] = self.tried.get(i, 0) + 1
            point = xyz_to_point(self.ctx, o["pos"] + np.array([0.0, 0.0, o["height"] / 2]))
            if self.split:
                self.carrying = o["color"]
                return {"name": "pick", "args": {"point": point, "bin": FIXED_MAPPING[o["color"]]}}
            return {"name": "pick_and_place", "args": {"point": point, "bin": FIXED_MAPPING[o["color"]]}}
        return {"name": "done", "args": {}}


class Reflex:
    def __init__(self, holding=True):
        self.holding, self.calls = holding, 0

    def answer(self, view, questions):
        self.calls += 1
        assert "holding" in questions and questions["holding"]["type"] == "noul"
        return {"holding": {"value": self.holding, "confidence": 0.95}}


def run(pol, brain, seed, realtime=False, time_limit=None):
    env = make_mock_env(seed=seed, realtime=realtime)
    if time_limit:
        env.time_limit = pol.time_limit = time_limit
    brain.env = env
    obs = env.reset()
    pol.reset({"instruction": obs["instruction"]})
    slow = 0.0
    while True:
        t0 = time.perf_counter()
        act = pol.act(obs)
        slow = max(slow, time.perf_counter() - t0)
        obs, info = env.step(act)
        if info["episode_over"]:
            return env, info, slow


class TestAgent(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ctx = SkillContext()

    def test_pointing_gives_the_grasp_target(self):
        env = make_mock_env(seed=0)
        obs = env.reset()
        for o in env.backend.objects:
            top = o["pos"] + np.array([0.0, 0.0, o["height"] / 2])
            for jitter in ([0, 0], [8, -8], [-10, 6]):          # pointing a little off-centre still finds the top face
                p = np.array(xyz_to_point(self.ctx, top)) + jitter
                xyz, info = point_to_xyz(self.ctx, obs, p)
                self.assertEqual(info["method"], "depth")
                self.assertLess(np.linalg.norm(xyz[:2] - top[:2]), 0.012, (o["color"], jitter, xyz, top))

    def test_episodes_by_pointing(self):
        correct = total = 0
        for seed in (0, 1, 2):
            brain, reflex = PointingBrain(self.ctx), Reflex(True)
            pol = AgentPolicy(brain, reflex, ctx=self.ctx, threaded=False)
            env, info, _ = run(pol, brain, seed)
            self.assertEqual(info["safety_summary"]["clipped_steps"], 0, info["safety_summary"])
            out = env.backend.outcomes(FIXED_MAPPING)
            self.assertNotIn("wrong", out)
            self.assertNotIn("off", out)
            correct, total = correct + out.count("correct"), total + len(out)
            self.assertGreater(reflex.calls, 0)
            self.assertEqual(pol.stats["aborts"], 0)
        self.assertGreaterEqual(correct, 0.85 * total, f"{correct}/{total}")

    def test_pick_look_place(self):
        brain = PointingBrain(self.ctx, split=True)
        pol = AgentPolicy(brain, Reflex(True), ctx=self.ctx, threaded=False)
        env, info, _ = run(pol, brain, 3)
        self.assertEqual(info["safety_summary"]["clipped_steps"], 0)
        out = env.backend.outcomes(FIXED_MAPPING)
        self.assertGreaterEqual(out.count("correct"), len(out) - 1, out)
        tools = [h["tool"] for h in pol.history]
        self.assertIn("pick", tools)
        self.assertIn("place", tools)

    def test_reflex_no_aborts_the_move_and_the_model_is_told(self):
        brain = PointingBrain(self.ctx)
        pol = AgentPolicy(brain, Reflex(False), ctx=self.ctx, threaded=False)
        env, info, _ = run(pol, brain, 0, time_limit=40.0)
        self.assertEqual(info["safety_summary"]["clipped_steps"], 0)
        self.assertGreater(pol.stats["aborts"], 0)
        self.assertTrue(any(str(h["result"]).startswith("aborted") for h in pol.history))
        self.assertNotIn("wrong", env.backend.outcomes(FIXED_MAPPING))

    def test_bad_tool_calls_are_reported_not_executed(self):
        class Confused(PointingBrain):
            def decide(self, view):
                self.views += 1
                calls = [{"name": "place", "args": {"bin": "left"}},                       # nothing is held
                         {"name": "pick_and_place", "args": {"point": [500, 500], "bin": "top"}},
                         {"name": "fly", "args": {}}, {"name": "pick", "args": {"point": [5, 5]}}]   # the ceiling
                return calls[self.views - 1] if self.views <= len(calls) else {"name": "done", "args": {}}
        brain = Confused(self.ctx)
        pol = AgentPolicy(brain, None, ctx=self.ctx, threaded=False)
        env, info, _ = run(pol, brain, 0)
        self.assertEqual(info["safety_summary"]["clipped_steps"], 0)
        self.assertEqual(len([h for h in pol.history if str(h["result"]).startswith("failed")]), 4)
        self.assertEqual(env.backend.outcomes(FIXED_MAPPING).count("counter"), len(env.backend.objects))

    def test_act_never_waits_for_the_models(self):
        brain = PointingBrain(self.ctx, delay=1.5)                 # a model that takes 1.5 s per call
        pol = AgentPolicy(brain, Reflex(True), ctx=self.ctx)       # threaded (the real configuration)
        env, info, slow = run(pol, brain, 1, realtime=True, time_limit=30.0)
        self.assertLess(slow, 0.05, "act() blocked")
        self.assertEqual(info["safety_summary"]["clipped_steps"], 0)
        self.assertGreaterEqual(env.backend.outcomes(FIXED_MAPPING).count("correct"), 1)

    def test_tool_declarations(self):
        for t in TOOLS:
            self.assertEqual(t["type"], "function")
            self.assertTrue(t["name"] and t["description"] and t["parameters"]["type"] == "object")


if __name__ == "__main__":
    unittest.main()
