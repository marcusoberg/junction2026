"""Offline test of the ADK agent: the real ADK runner, tools, camera callback and 30 Hz bridge, with a scripted
stand-in for the Gemini model and the organisers' mock world for the robot. No network, no key, no simulator.

    cd adk_agents && ../.venv/bin/python -m unittest robot_sorter.tests.test_robot_sorter -v

ScriptedModel answers the way a tool-calling model would: look, then pick_and_place each listed object into the bin
of the fixed mapping (hand_over when a pick_and_place has no safe plan), then done. What is tested is everything
on our side of the model call.
"""
import asyncio
import json
import time
import unittest

import robot_sorter  # noqa: F401  (puts the project and the organisers' repo on sys.path)
from aaltonomous.baselines.classical.mock_world import make_mock_env
from aaltonomous.interface.tasks import FIXED_MAPPING
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_response import LlmResponse
from google.adk.runners import InMemoryRunner
from google.genai import types
from pydantic import Field

from robot_sorter.agent import build_agent
from robot_sorter.robot import get_robot
from robot_sorter.serve import AdkPolicy


class ScriptedModel(BaseLlm):
    model: str = "scripted"
    max_moves: int = 99
    calls: list = Field(default_factory=list)          # (tool name, args) the "model" asked for
    images: list = Field(default_factory=list)         # camera images attached to each request
    tries: dict = Field(default_factory=dict)
    relayed: set = Field(default_factory=set)

    def _next(self, last):
        if last is None:
            return "look", {}
        name, res = last
        if name == "done":
            return None
        moves = sum(1 for n, _ in self.calls if n in ("pick_and_place", "hand_over"))
        if name == "pick_and_place" and res.get("status") == "failed" and res.get("object") not in self.relayed:
            self.relayed.add(res["object"])             # e.g. the bin is on the far side: use the oracle's relay
            return "hand_over", {"object_id": res["object"], "bin": res["bin"]}
        objects = res.get("objects_on_counter")
        if not isinstance(objects, list):
            return "look", {}
        for o in objects:
            if self.tries.get(o["id"], 0) < 2 and moves < self.max_moves:
                self.tries[o["id"]] = self.tries.get(o["id"], 0) + 1
                return "pick_and_place", {"object_id": o["id"], "bin": FIXED_MAPPING[o["color"]]}
        return "done", {"summary": "scripted run finished"}

    async def generate_content_async(self, llm_request, stream=False):
        last = None
        for content in llm_request.contents:
            for part in content.parts or []:
                if part.function_response is not None:
                    last = (part.function_response.name, dict(part.function_response.response or {}))
        tail = llm_request.contents[-1]
        self.images.append(sum(1 for p in tail.parts or [] if p.inline_data and p.inline_data.mime_type == "image/jpeg"))
        nxt = self._next(last)
        if nxt is None:
            yield LlmResponse(content=types.Content(role="model", parts=[types.Part(text="Finished.")]))
            return
        self.calls.append(nxt)
        yield LlmResponse(content=types.Content(role="model", parts=[
            types.Part.from_function_call(name=nxt[0], args=nxt[1])]))


class TestPolicyServerMode(unittest.TestCase):
    """serve.py's AdkPolicy against the mock world, the way the organisers' evaluator drives it."""

    def run_episode(self, seed, model=None):
        model = model or ScriptedModel()
        pol = AdkPolicy(agent=build_agent(model), log_fn=lambda m: None)
        robot = pol.robot
        env = make_mock_env(seed=seed)                  # virtual clock: stepped only while the robot moves
        obs = env.reset()
        pol.reset(obs["instruction"])
        slow, info, deadline = 0.0, None, time.monotonic() + 180
        while time.monotonic() < deadline:
            t0 = time.perf_counter()
            action = pol.act(obs)
            slow = max(slow, time.perf_counter() - t0)
            if robot.busy or action["done"] or robot.ticks <= robot.warmup_ticks + 6:
                obs, info = env.step(action)
                if info["episode_over"]:
                    break
            else:
                time.sleep(0.002)                       # the "model" is thinking or a tool is planning: time stands still
        pol.close()
        return model, pol, env, info, slow

    def test_sorts_with_zero_safety_clips(self):
        correct = total = 0
        for seed in (0, 1):
            model, pol, env, info, slow = self.run_episode(seed)
            self.assertIsNotNone(info)
            self.assertEqual(info["end_reason"], "done", (info, model.calls))
            self.assertEqual(info["safety_summary"]["clipped_steps"], 0, info["safety_summary"])
            out = env.backend.outcomes(FIXED_MAPPING)
            self.assertNotIn("wrong", out)
            self.assertNotIn("off", out)
            correct, total = correct + out.count("correct"), total + len(out)
            self.assertEqual(model.calls[0][0], "look")
            self.assertEqual(model.calls[-1][0], "done")
            self.assertTrue(all(n >= 1 for n in model.images), model.images)      # every request carried live images
            self.assertLess(slow, 0.05, "act() waited for the agent")
            self.assertEqual(pol.stats["model_errors"], 0)
        self.assertGreaterEqual(correct, 0.75 * total, f"{correct}/{total}")

    def test_bad_calls_are_reported_and_move_nothing(self):
        class Confused(ScriptedModel):
            def _next(self, last):
                script = [("place", {"bin": "left"}),                                     # nothing is held
                          ("pick_and_place", {"bin": "left"}),                             # no object named
                          ("pick_and_place", {"object_id": "green_9", "bin": "left"}),     # unknown id
                          ("pick", {"point": [5, 5]}),                                     # the ceiling
                          ("view_camera", {"camera": "head", "region": [900, 0, 100, 500]}),        # upside down
                          ("done", {})]
                return script[len(self.calls)] if len(self.calls) < len(script) else None

        model = Confused()
        pol = AdkPolicy(agent=build_agent(model), log_fn=lambda m: None)
        env = make_mock_env(seed=0)
        obs = env.reset()
        pol.reset(obs["instruction"])
        deadline = time.monotonic() + 60
        while not pol.robot.finished and time.monotonic() < deadline:
            action = pol.act(obs)
            if pol.robot.busy or pol.robot.ticks <= pol.robot.warmup_ticks + 6:
                obs, _ = env.step(action)
            else:
                time.sleep(0.002)
        pol.close()
        self.assertTrue(pol.robot.finished)
        self.assertEqual(pol.robot.moves, [])
        self.assertEqual(env.backend.outcomes(FIXED_MAPPING).count("counter"), len(env.backend.objects))


class TestAdkWebMode(unittest.TestCase):
    """What `adk web` / `adk run` do: a plain ADK runner; the tools start the mock world themselves (real time)."""

    def test_one_move_in_the_mock_world(self):
        robot = get_robot()
        robot.external, robot.world, robot.session = False, None, None
        robot.log = lambda m: None
        model = ScriptedModel(max_moves=1)
        runner = InMemoryRunner(agent=build_agent(model), app_name="robot_sorter")

        async def chat():
            session = await runner.session_service.create_session(app_name="robot_sorter", user_id="dev")
            message = types.Content(role="user", parts=[types.Part(text="Sort the objects.")])
            results = []
            async for event in runner.run_async(user_id="dev", session_id=session.id, new_message=message):
                results += [(r.name, dict(r.response or {})) for r in event.get_function_responses() or []]
            return results

        try:
            results = asyncio.run(chat())
        finally:
            if robot.world is not None:
                robot.world._stop = True
        names = [n for n, _ in results]
        self.assertEqual(names[0], "look")
        self.assertEqual(names[-1], "done")
        self.assertIn("pick_and_place", names)
        look = results[0][1]
        self.assertGreaterEqual(len(look["objects_on_counter"]), 1, json.dumps(look)[:300])
        self.assertNotIn("time_left_s", look)                                    # the mock world has no time limit
        outcomes = results[-1][1]["mock_world_outcomes"]
        self.assertNotIn("wrong", outcomes)
        moved = [r for n, r in results if n == "pick_and_place" and r["status"] == "completed"]
        if moved:
            self.assertGreaterEqual(outcomes.count("correct"), 1, outcomes)


if __name__ == "__main__":
    unittest.main()
