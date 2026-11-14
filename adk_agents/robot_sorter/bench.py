"""A scripted stand-in for the Gemini model: the same agent, tools and robot bridge, with fixed rules instead of
model calls. For measuring the TOOLS (grasps, plans, retries) in the sim, quickly and repeatably.

    ROBOT_AGENT_MODEL=bench python -m robot_sorter.serve        # sim/modal_sim.py --action grasp-bench does this

Rules: look; then, for every object on the counter, pick_and_place it into the bin the fixed nominal mapping gives
its measured colour (red -> left, yellow -> center, blue -> right); after a miss try the same object again, up to
BENCH_TRIES times in all (the tools change the grip between tries); after "no safe plan" hand it over once; done.
"""
from __future__ import annotations

import os

from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_response import LlmResponse
from google.genai import types
from pydantic import Field


def _bin_for(rgb):
    r, g, b = (float(v) for v in (rgb or (0, 0, 0)))
    if b > r and b > g:
        return "right"                       # blue
    return "center" if g > 0.75 * r else "left"     # yellow : red


class BenchModel(BaseLlm):
    model: str = "bench"
    tries: dict = Field(default_factory=dict)        # object id -> pick_and_place calls made
    relayed: set = Field(default_factory=set)
    max_tries: int = int(os.environ.get("BENCH_TRIES", "3"))

    def _next(self, last):
        if last is None:
            self.tries.clear()
            self.relayed.clear()
            return "look", {}
        name, res = last
        if name == "done":
            return None
        if name == "pick_and_place" and res.get("status") == "failed" and res.get("object") not in self.relayed \
                and "not enough time" not in str(res.get("reason")):
            self.relayed.add(res["object"])
            return "hand_over", {"object_id": res["object"], "bin": res["bin"]}
        objects = res.get("objects_on_counter")
        if not isinstance(objects, list):
            return "look", {}
        for o in objects:
            if self.tries.get(o["id"], 0) < self.max_tries:
                self.tries[o["id"]] = self.tries.get(o["id"], 0) + 1
                return "pick_and_place", {"object_id": o["id"], "bin": _bin_for(o.get("rgb"))}
        return "done", {"summary": f"bench: tries per object {self.tries}"}

    async def generate_content_async(self, llm_request, stream=False):
        last, fresh = None, False
        for content in llm_request.contents:
            for part in content.parts or []:
                if part.function_response is not None:
                    last = (part.function_response.name, dict(part.function_response.response or {}))
                elif part.text and part.text.startswith("New episode"):
                    last = None                       # a new episode in the same session history never happens,
        nxt = self._next(last)                        # but a new session starts here
        if nxt is None:
            yield LlmResponse(content=types.Content(role="model", parts=[types.Part(text="Bench episode finished.")]))
            return
        yield LlmResponse(content=types.Content(role="model", parts=[types.Part.from_function_call(name=nxt[0], args=nxt[1])]))
