"""The Gemini Decisions API (model gemini-decision-1): closed questions about text and images, answered with
calibrated probabilities and no generated text, so it is fast and cheap. Written from Google's documentation
(https://aistudio.google.com/docs/decisions-api, read 14 Nov 2026):

    POST https://generativelanguage.googleapis.com/v1beta/decisions      header x-goog-api-key
    {"model": "gemini-decision-1",
     "input": [{"type": "text", "text": ...}, {"type": "image", "data": <base64>, "mime_type": "image/jpeg",
                                                 "resolution": "low" | "medium" | "high" | "ultra_high"}],
     "questions": {<key>: {"type": "noul" | "choice" | "score", "instructions": ..., "criteria": ...}}}
       noul    yes/no. criteria optional: {"true_description", "false_description"}
       choice  criteria = {option id: description}, 2-255 options. Probability is always spread over the
               options, so include a catch-all option and look at the confidence.
       score   criteria = [level descriptions, lowest first]
    -> {"answers": {<key>: {"noul" | "choice" | "score", "probabilities", "confidence"}}, "usage": ...}
    Up to 1,000 questions per request share one input and are scored independently; the documentation recommends
    putting all questions about the same input in ONE request (the input is counted once, latency stays flat).
    The Python SDK has client.decisions.create from google-genai 3.2; ADK 2.14 pins an older SDK, so this file
    uses the REST endpoint.

How the agent uses it:
    label_objects   after `look`: ONE request with the head image (object ids drawn on it) and two questions per
                    object, its colour and its shape. Cached per object for the episode.
    ask             the ask_gemini tool: any closed question about a camera image, on demand.
A failed or slow call never breaks anything: the caller simply gets no labels.
"""
from __future__ import annotations

import base64
import json
import os
import re
import time
import urllib.error
import urllib.request

import numpy as np

URL = os.environ.get("ROBOT_DECISION_URL", "https://generativelanguage.googleapis.com/v1beta/decisions")
MODEL = os.environ.get("ROBOT_DECISION_MODEL", "gemini-decision-1")
ENABLED = os.environ.get("ROBOT_DECISIONS", "1") == "1"

COLOUR_WORDS = ("red", "orange", "yellow", "green", "blue", "purple", "violet", "pink", "brown", "white", "grey",
                "gray", "black")
SHAPES = {"cube": "A cube: a box with a square top face and four flat vertical sides.",
          "cylinder": "A cylinder: a round top face, a curved side, no corners.",
          "triangle": "A triangular prism standing upright: the top face is a triangle with three corners.",
          "other": "Something else: a block lying on its side, a different object, or nothing clear."}


def _key():
    k = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY") or ""
    return "" if k == "PASTE_YOUR_KEY_HERE" else k.strip()


def available():
    return ENABLED and bool(_key())


def image_block(rgb, resolution="high", quality=90):
    import cv2
    ok, buf = cv2.imencode(".jpg", np.ascontiguousarray(np.asarray(rgb)[..., ::-1]), [cv2.IMWRITE_JPEG_QUALITY, quality])
    return {"type": "image", "data": base64.b64encode(buf.tobytes()).decode(), "mime_type": "image/jpeg",
            "resolution": resolution}


def create(text, images, questions, timeout=6.0):
    """One Decisions request -> (answers dict as returned by the API, seconds). Raises on HTTP / network errors."""
    body = {"model": MODEL, "input": [{"type": "text", "text": text}] + list(images), "questions": questions}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), method="POST",
                                 headers={"x-goog-api-key": _key(), "Content-Type": "application/json"})
    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            out = json.load(r)
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"Decisions API HTTP {e.code}: {e.read().decode(errors='replace')[:300]}") from None
    return out.get("answers") or {}, time.monotonic() - t0


def colour_options(instruction):
    """The colour words of the instruction (or the common ones) as choice options, plus a catch-all."""
    named = [w for w in COLOUR_WORDS if re.search(rf"\b{w}\b", (instruction or "").lower())]
    named = ["purple" if w == "violet" else "grey" if w == "gray" else w for w in named]
    opts = {w: "" for w in dict.fromkeys(named or ["red", "orange", "yellow", "green", "blue", "purple"])}
    opts["other"] = "None of the colours listed."
    return opts


def label_objects(marked_rgb, ids, instruction, cache):
    """One request: the head image with the object ids drawn on it, and for every id not in `cache` two questions.
    cache[id] = {"colour", "colour_confidence", "colour_probabilities", "shape", "shape_confidence"}."""
    todo = [i for i in ids if i not in cache]
    if not todo or not available():
        return None
    text = ("A robot's head camera looks down at a counter with small coloured blocks (about 4.5 cm; cubes, "
            "cylinders and upright triangular prisms) in front of three orange bins. Each block on the counter has "
            "a white circle on its top and its id written next to it (obj_1, obj_2, ...). The task instruction is: "
            f"\"{instruction}\". The camera and the light shift and wash out colours, so judge each block's colour "
            "by comparing the blocks with each other and choose the option it is nearest to.")
    colours = colour_options(instruction)
    questions = {}
    for i in todo:
        questions[f"colour_{i}"] = {"type": "choice", "instructions": f"What colour is the block labelled {i}?",
                                    "criteria": colours}
        questions[f"shape_{i}"] = {"type": "choice", "instructions": f"What shape is the block labelled {i}?",
                                   "criteria": SHAPES}
    answers, seconds = create(text, [image_block(marked_rgb)], questions)
    for i in todo:
        c, s = answers.get(f"colour_{i}"), answers.get(f"shape_{i}")
        if c and s:
            cache[i] = {"colour": c.get("choice"), "colour_confidence": round(float(c.get("confidence", 0)), 2),
                        "colour_probabilities": {k: round(float(v), 2) for k, v in (c.get("probabilities") or {}).items()
                                                 if float(v) >= 0.05},
                        "shape": s.get("choice"), "shape_confidence": round(float(s.get("confidence", 0)), 2)}
    return round(seconds, 2)


def ask(text, images, question, kind="noul", options=None):
    """A single closed question. kind: noul (yes/no), choice (options: list or {id: description}) or score
    (options: list of level descriptions, lowest first). Returns the API's answer plus latency_s."""
    q = {"type": kind, "instructions": question}
    if kind == "choice":
        q["criteria"] = options if isinstance(options, dict) else {str(o): "" for o in (options or [])}
    elif kind == "score":
        q["criteria"] = [str(o) for o in (options or [])]
    answers, seconds = create(text, images, {"q": q})
    out = dict(answers.get("q") or {})
    out["latency_s"] = round(seconds, 2)
    return out
