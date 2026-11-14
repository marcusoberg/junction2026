"""The ADK agent behind the organisers' policy-server protocol (websocket + msgpack, port 8000).

    PYTHONPATH=adk_agents .venv/bin/python -m robot_sorter.serve

This is how the agent controls the simulator or the real robot: the organisers' evaluator connects, sends
observations at 30 Hz and gets joint targets back. act() only hands out ticks (robot.py); the ADK agent runs in
its own thread and event loop, one ADK session per episode, and keeps calling tools until it calls done or the
time is up.

Environment: GOOGLE_API_KEY (or GEMINI_API_KEY) and ROBOT_AGENT_MODEL, read from robot_sorter/.env unless already set;
AALTO_TIME_LIMIT (default 120), PORT (default 8000).
In the Isaac sim on Modal:  modal run sim/modal_sim.py --action eval-adk   (see sim/modal_sim.py).
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys
import threading
import uuid

# The policy must not know where objects are except through its cameras. The simulator's scene service (ground-truth
# poses) is only reachable with zmq, so make zmq unimportable in this process: any attempt would raise ImportError.
sys.modules["zmq"] = None

from aaltonomous.deploy.policy_server import Policy, serve
from google.adk.runners import InMemoryRunner
from google.genai import types

from .robot import get_robot

log = logging.getLogger("robot_sorter")
APP = "robot_sorter"
NUDGE = "Carry on: call the next tool. Call done only when nothing useful is left to do."


class AdkPolicy(Policy):
    """The organisers' Policy interface: reset(instruction) starts an agent episode, act(obs) hands out ticks."""

    def __init__(self, agent=None, log_fn=None):
        from .agent import root_agent
        self.agent = agent or root_agent
        self.name = f"adk/{getattr(self.agent.model, 'model', self.agent.model)}"
        self.robot = get_robot()
        self.robot.external = True
        if log_fn is not None:
            self.robot.log = log_fn
        self.robot.ctx                                   # build the planner before the first episode
        self.action_horizon = int(self.robot.ctx.cfg["motion"]["chunk"])
        self.runner = InMemoryRunner(agent=self.agent, app_name=APP)
        self.loop = asyncio.new_event_loop()
        threading.Thread(target=self.loop.run_forever, name="adk-agent", daemon=True).start()
        self.episode = None
        self.stats = {"model_errors": 0, "runs": 0}

    def reset(self, instruction, task=None, seed=None):
        if self.episode is not None:
            self.episode.cancel()
            self.robot.log(f"[agent] episode end: {len(self.robot.moves)} moves {self.robot.moves}; stats {self.stats}")
        self.robot.log(f"[agent] reset task={task} seed={seed}: {instruction}")
        self.robot.reset(instruction)
        self.episode = asyncio.run_coroutine_threadsafe(self._episode(instruction), self.loop)

    def act(self, obs):
        return self.robot.act(obs)

    def close(self, timeout=10.0):
        """Let the current episode's last model turn finish (or cancel it), then stop the agent's event loop."""
        if self.episode is not None:
            try:
                self.episode.result(timeout)
            except Exception:  # noqa: BLE001  (timed out, cancelled or failed: all fine when closing)
                self.episode.cancel()
        self.loop.call_soon_threadsafe(self.loop.stop)

    async def _episode(self, instruction):
        robot = self.robot
        session = await self.runner.session_service.create_session(app_name=APP, user_id="robot",
                                                                   session_id=uuid.uuid4().hex)
        text, idle = f"New episode. Instruction: {instruction}", 0
        while not robot.finished and (robot.time_left() is None or robot.time_left() > 2.0):
            self.stats["runs"] += 1
            called = False
            try:
                message = types.Content(role="user", parts=[types.Part(text=text)])
                async for event in self.runner.run_async(user_id="robot", session_id=session.id, new_message=message):
                    for call in event.get_function_calls() or []:
                        called = True
                        robot.log(f"[agent t={robot.t:6.2f}] {call.name}({dict(call.args or {})})")
                    for resp in event.get_function_responses() or []:
                        robot.log(f"[agent t={robot.t:6.2f}] {resp.name} -> {str(resp.response)[:300]}")
                    if event.is_final_response() and event.content and event.content.parts:
                        said = " ".join(p.text for p in event.content.parts if p.text)
                        if said:
                            robot.log(f"[agent t={robot.t:6.2f}] says: {said[:300]}")
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001  (a failed model call: hold still, ask again)
                self.stats["model_errors"] += 1
                robot.log(f"[agent t={robot.t:6.2f}] model call failed: {e!r}"[:400])
            if called:
                idle = 0
            else:                                         # failed, or answered in words only: do not hammer the model
                idle += 1
                await asyncio.sleep(min(5.0, 0.5 * 2 ** (idle - 1)))
            text = NUDGE                                  # the model stopped without calling done
        robot.finished = True


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    policy = AdkPolicy(log_fn=log.info)
    log.info("%s ready", policy.name)
    serve(policy, port=int(os.environ.get("PORT", 8000)))
