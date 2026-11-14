"""Offline tests (no simulator, no GPU): the organisers' kinematic mock world + their SafetyFilter.

    PYTHONPATH=aaltonomous-humanoid:. python -m unittest agent.tests.test_skills -v
"""
import time
import unittest

import numpy as np

from aaltonomous.baselines.classical.mock_world import make_mock_env
from aaltonomous.interface.tasks import FIXED_MAPPING, sample_layout
from aaltonomous.sim.oracle import _OraclePlanner

from agent.runtime import LLMChooser, RandomChooser, RuleChooser, SkillPolicy, parse_mapping
from agent.skills import (Approach, SkillContext, WorldObject, go_home, pick, pick_and_place, place, relocate,
                          set_hand)
from agent.skills.tools import TOOLS, validate


def layout_objects(ctx, seed):
    H, out = ctx.scene.object_height(), []
    for c in sample_layout(np.random.default_rng(seed), **ctx.scene.layout_kwargs()):
        p = ctx.scene.mat_to_pelvis(np.array([c["x"], c["y"], 0.0]))
        out.append(WorldObject(f"{c['color']}_{c['shape']}_{len(out)}", c["color"],
                               np.r_[p[:2], ctx.scene.counter_z + H], c["shape"], c["yaw"], source="sim"))
    return out


def run_episode(pol, seed, realtime=False, time_limit=None):
    env = make_mock_env(seed=seed, realtime=realtime)
    if time_limit:
        env.time_limit = pol.time_limit = time_limit
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


class TestSkills(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ctx = SkillContext()
        cls.hand = cls.ctx.post["open"].copy()

    def test_whole_move_is_the_oracle_planners(self):
        ctx = self.ctx
        ref = _OraclePlanner(ctx.cfg, ctx.scene, ctx.post, ctx.home, ctx.kin, ctx.safety, ctx.spec)
        ref.drop_offsets = {}
        for obj in layout_objects(ctx, 0) + layout_objects(ctx, 1):
            b = FIXED_MAPPING[obj.color]
            t = pick_and_place(ctx, obj, b, ctx.home, self.hand)
            r = ref.plan(obj.detection(), ctx.home, self.hand, bin_name=b, relocate=False)
            self.assertEqual(t.ok, r.ok)
            if t.ok:
                self.assertTrue(np.allclose(t.arm, r.arm) and np.allclose(t.hand, r.hand))

    def test_pick_then_place_and_variations_pass_the_safety_filter(self):
        ctx = self.ctx
        aps = [None, Approach(speed=0.5), Approach(pregrasp_offset=(-0.04, 0.02)), Approach(finger_direction_deg=20),
               Approach(carry_height=0.19, grasp_height_frac=0.55)]
        done = 0
        for i, obj in enumerate(layout_objects(ctx, 2) + layout_objects(ctx, 3)):
            b = FIXED_MAPPING[obj.color]
            whole = pick_and_place(ctx, obj, b, ctx.home, self.hand, approach=aps[i % len(aps)])
            if not whole.ok:
                continue
            pk = pick(ctx, obj, ctx.home, self.hand, arm=whole.side)
            self.assertTrue(pk.ok, pk.reason)
            self.assertEqual([p[0] for p in pk.phases], ["to_pregrasp", "descend", "close", "lift"])
            pl = place(ctx, b, pk.end_arm, pk.end_hand, arm=whole.side)
            self.assertTrue(pl.ok, pl.reason)
            for tr, a0, h0 in ((whole, ctx.home, self.hand), (pk.then(pl), ctx.home, self.hand)):
                self.assertEqual(ctx.planner.dry_run_safety(a0, h0, tr.arm, tr.hand), 0)
                self.assertLessEqual(np.abs(np.diff(tr.arm, axis=0)).max() * 30, 1.0 + 1e-6)
            done += 1
        self.assertGreaterEqual(done, 5)
        self.assertEqual(ctx.cfg["motion"]["arm_max_vel"], 0.97)          # tuned() restored the config

    def test_small_skills(self):
        ctx = self.ctx
        away = ctx.home + 0.2 * np.sign(ctx.home + 1e-9)
        away = np.clip(away, ctx.spec.arm_lower + 0.1, ctx.spec.arm_upper - 0.1)
        self.assertTrue(go_home(ctx, away, self.hand).ok)
        self.assertTrue(set_hand(ctx, ctx.home, self.hand, "left", "closed").ok)
        self.assertFalse(set_hand(ctx, ctx.home, self.hand, "left", "fist").ok)

    def test_cross_body_object_is_relocated(self):
        ctx = self.ctx
        # a red object (-> left bin) far on the robot's right: the left arm cannot pick it there
        p = ctx.scene.mat_to_pelvis(np.array([0.30, 0.15, 0.0]))
        obj = WorldObject("red_cylinder_0", "red", np.r_[p[:2], ctx.scene.counter_z + ctx.scene.object_height()],
                          "cylinder", 0.0, source="sim")
        direct = pick_and_place(ctx, obj, "left", ctx.home, self.hand)
        if not direct.ok:
            r = relocate(ctx, obj, "left", ctx.home, self.hand)
            self.assertTrue(r.ok, r.reason)
            self.assertEqual(ctx.planner.dry_run_safety(ctx.home, self.hand, r.arm, r.hand), 0)


class TestRuntime(unittest.TestCase):
    def test_instruction_parsing(self):
        from aaltonomous.interface import tasks
        rng = np.random.default_rng(0)
        for _ in range(60):
            text, m = tasks.make_instruction("sort_language", rng)
            self.assertEqual(parse_mapping(text), m)
        self.assertEqual(parse_mapping("The left bin is for yellow; the centre bin is for red; the right bin is for blue."),
                         {"yellow": "left", "red": "center", "blue": "right"})
        self.assertIsNone(parse_mapping("Sort them however you like."))

    def test_camera_episodes_in_the_mock_world(self):
        correct = total = 0
        for seed in (0, 1, 2):
            pol = SkillPolicy(log=None, threaded=False)       # virtual clock: decide inside act()
            env, info, slow = run_episode(pol, seed)
            self.assertIn(info["end_reason"], ("done", "time_limit"))
            self.assertEqual(info["safety_summary"]["clipped_steps"], 0, info["safety_summary"])
            out = env.backend.outcomes(pol.mapping)
            self.assertNotIn("wrong", out)
            self.assertNotIn("off", out)
            correct, total = correct + out.count("correct"), total + len(out)
            self.assertTrue(all(s["plan_ticks"] > 0 and s["phases"] for s in pol.segments))
        self.assertGreaterEqual(correct, 0.6 * total, f"{correct}/{total}")

    def test_random_chooser_varies_and_stays_safe(self):
        pol = SkillPolicy(chooser=RandomChooser(seed=3), log=None, threaded=False)
        env, info, _ = run_episode(pol, 4)
        self.assertEqual(info["safety_summary"]["clipped_steps"], 0)
        self.assertTrue(pol.segments)
        self.assertTrue(any(s["args"].get("approach") for s in pol.segments))

    def test_act_never_waits_for_a_slow_or_broken_model(self):
        calls = []

        def model(world):                     # first call hangs, every later one fails
            calls.append(1)
            if len(calls) == 1:
                time.sleep(3.0)
            raise RuntimeError("no network")

        ch = LLMChooser(model, fallback=RuleChooser(), timeout_s=0.3)
        pol = SkillPolicy(chooser=ch, log=None)
        env, info, slow = run_episode(pol, 1, realtime=True, time_limit=25.0)     # real 30 Hz for 25 s
        self.assertLess(slow, 0.05, "act() blocked")
        self.assertGreater(ch.stats["fallback"], 0)
        self.assertEqual(info["safety_summary"]["clipped_steps"], 0)
        self.assertGreaterEqual(env.backend.outcomes(pol.mapping).count("correct"), 1)

    def test_tool_declarations_and_validation(self):
        names = {t["name"] for t in TOOLS}
        self.assertTrue({"pick_and_place", "relocate", "done"} <= names)
        world = {"objects": [{"id": "red_1", "where": "counter"}], "bins": {"left": [0, 0]}}
        self.assertIsNone(validate({"skill": "pick_and_place", "object": "red_1", "bin": "left"}, world))
        self.assertIsNotNone(validate({"skill": "pick_and_place", "object": "blue_9", "bin": "left"}, world))
        self.assertIsNotNone(validate({"skill": "fly"}, world))


if __name__ == "__main__":
    unittest.main()
