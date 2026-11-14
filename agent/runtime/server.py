"""SkillPolicy behind the organisers' policy-server protocol (websocket + msgpack, port 8000).

    PYTHONPATH=aaltonomous-humanoid:. python -m agent.runtime.server

Environment:
    AGENT_MODE         gemini = AgentPolicy: a Gemini model looks at the camera images and calls the skills
                       (needs GEMINI_API_KEY; models via AGENT_BRAIN_MODEL / AGENT_REFLEX_MODEL, see gemini.py;
                       AGENT_REFLEX=0 turns the mid-motion Decisions-API check off)
                       skills (default) = SkillPolicy with the rule/random chooser below, no model
    AGENT_PERCEPTION   camera (default; what a submission may use) | sim (exact poses from the scene service)
    AGENT_CHOOSER      rule (default) | random          (an LLM chooser is wired in code: see agent/skills/tools.py)
    AGENT_SHAPES       1 = estimate shape and yaw from the camera image (default 0, see perception.estimate_shape)
    AGENT_SEED         seed of the random chooser
    AALTO_TIME_LIMIT   episode length in s (default 120), PORT (default 8000)
"""
from __future__ import annotations

import logging
import os
import time

import numpy as np

from aaltonomous.deploy.policy_server import Policy, serve

from .choosers import RandomChooser, RuleChooser
from .policy import SkillPolicy

log = logging.getLogger("agent")


class SkillServerPolicy(Policy):
    name = "skills"

    def __init__(self):
        if os.environ.get("AGENT_MODE", "skills") == "gemini":
            from .agent_policy import AgentPolicy
            from .gemini import GeminiBrain, GeminiReflex
            brain = GeminiBrain()
            reflex = GeminiReflex() if os.environ.get("AGENT_REFLEX", "1") == "1" else None
            self.pol = AgentPolicy(brain, reflex, time_limit=float(os.environ.get("AALTO_TIME_LIMIT", 120.0)),
                                   log=lambda m: log.info(m))
            self.name = f"agent/{brain.model}" + (f"+{reflex.model}" if reflex else "")
            self.action_horizon = self.pol.action_horizon
            return
        perception = os.environ.get("AGENT_PERCEPTION", "camera")
        client = None
        if perception == "sim":
            from aaltonomous.sim.scene_client import SceneClient
            client = SceneClient(os.environ.get("AGENT_SIM_HOST", "127.0.0.1"))
        chooser = (RandomChooser(int(os.environ.get("AGENT_SEED", "0"))) if os.environ.get("AGENT_CHOOSER") == "random"
                   else RuleChooser())
        self.pol = SkillPolicy(chooser=chooser, perception=perception, scene_client=client,
                               time_limit=float(os.environ.get("AALTO_TIME_LIMIT", 120.0)),
                               shapes=os.environ.get("AGENT_SHAPES", "0") == "1", log=lambda m: log.info(m))
        self.name = f"skills/{perception}/{chooser.name}"
        self.action_horizon = self.pol.action_horizon

    def reset(self, instruction, task=None, seed=None):
        job, t0 = self.pol.job, time.monotonic()        # let a decision from the last episode finish planning
        while job is not None and job["result"] is None and time.monotonic() - t0 < 30.0:
            time.sleep(0.02)
        if self.pol.segments:
            log.info("episode end: %d skills executed: %s; stats %s", len(self.pol.segments),
                     [(s.get("skill", "pick_and_place"), s.get("object") or s.get("args", {}).get("bin"), s["side"])
                      for s in self.pol.segments], getattr(self.pol, "stats", None))
        self.pol.reset({"instruction": instruction, "task": task})
        log.info("reset task=%s seed=%s mapping=%s", task, seed, getattr(self.pol, "mapping", None))

    def act(self, obs):
        a = self.pol.act(obs)
        return {"arm_q_target": np.asarray(a["arm_q_target"], np.float64),
                "hand_q_target": np.asarray(a["hand_q_target"], np.float64), "done": bool(a.get("done", False))}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    t0 = time.monotonic()
    pol = SkillServerPolicy()
    log.info("%s ready in %.1f s", pol.name, time.monotonic() - t0)
    serve(pol, port=int(os.environ.get("PORT", 8000)))
