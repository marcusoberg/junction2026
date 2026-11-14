"""Gemini adapters for AgentPolicy. WRITTEN FROM THE DOCUMENTATION, NOT RUN: nobody on the team had tried the API
when this was written (no key on the machine it was written on). Expect to fix small things on first contact;
`python -m agent.runtime.gemini <image.jpg>` makes one brain call and one reflex call so you can see what comes back.

What the docs (aistudio.google.com/docs, November 2026) say, and what this file relies on:
  * google-genai SDK >= 3.2, key in the GEMINI_API_KEY environment variable.
  * Tool calling with images: client.interactions.create(model=..., input=[blocks], tools=[{"type": "function",
    "name", "description", "parameters"}]); the reply's `steps` contain {type: "function_call", name, arguments}.
    Models: gemini-4-hydrogen (fast, the default here), gemini-4-argon, gemini-4-carbon (slowest, most deliberate),
    gemini-robotics-er-2-preview (embodied reasoning; points are [y, x] normalised to 0..1000, as our tools expect).
  * Decisions API: client.decisions.create(model="gemini-decision-1", input=[blocks], questions={key: {"type":
    "noul" | "choice" | "score", "instructions", "criteria"}}) with api_version v1beta; answers[key] has .noul
    (probability of yes) or .choice, and .confidence. 150-250 ms per request. It cannot call tools.
  * Image block: {"type": "image", "data": <base64>, "mime_type": "image/jpeg"}; text block: {"type": "text", "text"}.

Environment: GEMINI_API_KEY (required), AGENT_BRAIN_MODEL (default gemini-4-hydrogen), AGENT_BRAIN_THINKING
(low | medium | high; default low, for latency), AGENT_REFLEX_MODEL (default gemini-decision-1).
"""
from __future__ import annotations

import base64
import json
import os

from .agent_policy import SYSTEM_PROMPT, TOOLS


def blocks(view, preamble=None):
    """A view from AgentPolicy.view() as API input blocks: the images themselves, each after a one-line label."""
    out = []
    if preamble:
        out.append({"type": "text", "text": preamble})
    for label, jpeg in view["images"]:
        out.append({"type": "text", "text": f"\n[{label}]\n"})
        out.append({"type": "image", "data": base64.b64encode(jpeg).decode("ascii"), "mime_type": "image/jpeg"})
    out.append({"type": "text", "text": "\nState:\n" + json.dumps(view["state"], indent=1)})
    return out


def _get(obj, key, default=None):
    return obj.get(key, default) if isinstance(obj, dict) else getattr(obj, key, default)


class GeminiBrain:
    """decide(view) -> {"name", "args"}: one tool call from a Gemini model that sees the camera images."""

    def __init__(self, model=None, thinking=None, client=None):
        from google import genai
        self.client = client or genai.Client()
        self.model = model or os.environ.get("AGENT_BRAIN_MODEL", "gemini-4-hydrogen")
        self.thinking = thinking or os.environ.get("AGENT_BRAIN_THINKING", "low")

    def decide(self, view):
        kw = {"model": self.model, "input": blocks(view, SYSTEM_PROMPT), "tools": TOOLS}
        if self.thinking:
            kw["generation_config"] = {"thinking_level": self.thinking}
        try:
            it = self.client.interactions.create(**kw)
        except TypeError:                         # SDK without generation_config on this call
            kw.pop("generation_config", None)
            it = self.client.interactions.create(**kw)
        for step in _get(it, "steps", None) or []:
            if _get(step, "type") == "function_call":
                args = _get(step, "arguments") or {}
                if isinstance(args, str):
                    args = json.loads(args)
                return {"name": _get(step, "name"), "args": dict(args)}
        raise RuntimeError(f"no tool call in the reply: {str(_get(it, 'output_text', it))[:200]}")


class GeminiReflex:
    """answer(view, questions) -> {key: {"value": bool | str | float, "confidence": float}} via the Decisions API."""

    def __init__(self, model=None, client=None):
        from google import genai
        self.client = client or genai.Client(http_options={"api_version": "v1beta"})
        self.model = model or os.environ.get("AGENT_REFLEX_MODEL", "gemini-decision-1")

    def answer(self, view, questions):
        res = self.client.decisions.create(model=self.model, input=blocks(view), questions=questions)
        answers, out = _get(res, "answers", {}) or {}, {}
        for key, q in questions.items():
            a = _get(answers, key)
            if a is None:
                continue
            conf = float(_get(a, "confidence", 0.0) or 0.0)
            if q["type"] == "noul":
                out[key] = {"value": float(_get(a, "noul", 0.5)) >= 0.5, "confidence": conf}
            elif q["type"] == "choice":
                out[key] = {"value": _get(a, "choice"), "confidence": conf}
            else:
                out[key] = {"value": float(_get(a, "score", 0.0)), "confidence": conf}
        return out


if __name__ == "__main__":      # one call to each model on a saved head-camera image
    import sys
    import time

    from .agent_policy import holding_question
    jpeg = open(sys.argv[1], "rb").read()
    view = {"images": [("head camera", jpeg)],
            "state": {"instruction": "Put the red objects in the left bin, the yellow objects in the center bin and "
                                     "the blue objects in the right bin.", "time_left_s": 110.0, "holding": "nothing",
                      "history": []}}
    t0 = time.monotonic()
    print("brain :", GeminiBrain().decide(view), f"({time.monotonic() - t0:.2f} s)")
    t0 = time.monotonic()
    print("reflex:", GeminiReflex().answer(view, holding_question("left")), f"({time.monotonic() - t0:.2f} s)")
