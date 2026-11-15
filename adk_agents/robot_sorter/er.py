"""Gemini Robotics-ER (model gemini-robotics-er-2-preview): an embodied-reasoning vision model that POINTS at things.
The agent's find_blocks tool asks it which things on the counter are blocks, when the depth detector seems wrong.
Written from Google's documentation (https://aistudio.google.com/docs/robotics-overview and
.../docs/robotics-spatial, read 15 Nov 2026) and checked live with the request below:

    POST https://generativelanguage.googleapis.com/v1beta/interactions      header x-goog-api-key
    {"model": "gemini-robotics-er-2-preview",
     "input": [{"type": "image", "data": <base64>, "mime_type": "image/jpeg"}, {"type": "text", "text": ...}],
     "generation_config": {"thinking_level": "minimal"},
     "response_format": {"type": "text", "mime_type": "application/json", "schema": <JSON schema>},
     "store": false}
    -> {"status": "completed", "steps": [{"type": "thought", ...}, {"type": "model_output",
        "content": [{"type": "text", "text": <the JSON>}]}], "usage": ...}
    Points are [y, x], each normalised to 0-1000 (the documentation's convention; the same as the agent's `point`).
    The Python SDK has client.interactions.create; like decisions.py this file uses the REST endpoint, so the
    request path needs nothing but the standard library.

What was measured before this was built (20 saved sim first-look frames, 9 real frames of data drop 3):
    - sim: it finds the same objects as the depth detector (82 of 82 both) but places them worse (6-8 mm median
      after snapping its point to the depth image, against 3.2 mm), so positions keep coming from the depth image;
    - real robot, both hands in view: the depth detector reports hands, wrists and bin edges as objects (17 of 51
      detections on 8 frames) and misses 7 of 41 visible blocks; this prompt returned all 41 and no hand or bin,
      plus slivers of blocks cut by the image edge (0-1 per frame);
    - 1.75 s per call at thinking level "minimal" (worst 2.7 s); "low" and "high" found nothing more and took
      2.8 and 4.3-5.9 s;
    - its colours are weak without the instruction (purple read as blue 12 of 12, yellow as green 13 of 28 in the
      sim): the colour is a hint only, the Decisions API labels (decisions.py) stay the colour source.

A failed or slow call never breaks anything: point_blocks returns None and the caller keeps what it had. A call
is given up on TIMEOUT_S after it was sent, however the answer arrives (_send).
"""
from __future__ import annotations

import base64
import json
import os
import threading
import time
import urllib.error
import urllib.request

import numpy as np

from . import decisions

URL = os.environ.get("ROBOT_ER_URL", "https://generativelanguage.googleapis.com/v1beta/interactions")
MODEL = os.environ.get("ROBOT_ER_MODEL", "gemini-robotics-er-2-preview")
ENABLED = os.environ.get("ROBOT_ER", "1") == "1"
THINKING = os.environ.get("ROBOT_ER_THINKING", "minimal")
# The robot stands still while this call runs, so this is a deadline for the whole call (see _send), not a limit
# per socket read. Measured 1.4-3.2 s over about 185 calls (the evaluation, sim/replay_er.py, the mock world and
# the review; worst 3.19 s): a call that has not answered after 4 s is given up on.
TIMEOUT_S = float(os.environ.get("ROBOT_ER_TIMEOUT_S", "4.0"))

COLOURS = ("red", "orange", "yellow", "green", "blue", "purple", "other")
SHAPES = ("cube", "cylinder", "triangle", "other")
# The prompt the measurements above were made with (word for word), then the documentation's sentence on points.
SCENE = ("This image is from the head camera of a humanoid robot looking down at a table. Small foam blocks (about "
         "4.5 cm; cubes, cylinders and triangular prisms) lie on the table in front of three bins. ")
RULES = ("Do not include the bins, anything inside a bin, or the robot's own hands and arms. For each block give its "
         "colour and its shape (use 'triangle' for a triangular prism). ")
POINTS = "The points are in [y, x] format normalized to 0-1000."

last = {"seconds": None, "error": ""}        # the latest call: how long it took and why it failed (for logs)


def _key():
    return decisions._key()


def available():
    """As decisions.available(): switched on, a key is there, and the API is not known to be down. Both endpoints
    are one host with one key, so the Decisions breaker (decisions.link()) speaks for this one too: once a
    Decisions call has failed, find_blocks no longer waits for its own timeout and look no longer offers it.
    A failed call here does not close that breaker (it is the Decisions endpoint's own). Never waits."""
    link = getattr(decisions, "link", None)
    return ENABLED and bool(_key()) and (link is None or bool(link().up))


def request(instruction=None):
    """-> (prompt, JSON schema of the answer). With an instruction, the colours to choose from are the ones it
    names, in the words the Decisions API labels are asked with: on the sim's venue colours purple was then no
    longer read as blue (0 of 12 right without, 8 of 12 with), and the same blocks were found (21 of 21)."""
    colours = list(COLOURS)
    prompt = SCENE + "Point to every block that is on the table. " + RULES
    if instruction:
        colours = list(decisions.colour_options(instruction))
        prompt += (f'The task instruction is: "{instruction}". The camera and the light shift and wash out colours, '
                   "so judge each block's colour by comparing the blocks with each other and choose the option it "
                   "is nearest to. ")
    schema = {"type": "object", "required": ["items"], "properties": {"items": {"type": "array", "items": {
        "type": "object", "required": ["point", "colour", "shape"], "properties": {
            "point": {"type": "array", "items": {"type": "integer"}, "description": "[y, x] format normalized to 0-1000"},
            "colour": {"type": "string", "enum": colours}, "shape": {"type": "string", "enum": list(SHAPES)}}}}}}
    return prompt + POINTS, schema


def _send(req, timeout):
    """The request, given up on `timeout` seconds after it was sent, whatever happens (as decisions._send).
    urlopen's own timeout is per socket operation: a name lookup can hang for longer, and an answer that arrives
    slowly, a byte at a time, never trips it (a test endpoint that trickled for 16 s kept the robot standing for
    16 s). The request runs in its own thread, which is left to finish by itself."""
    box = {}

    def work():
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                box["out"] = json.load(r)
        except BaseException as e:  # noqa: BLE001  (handed to the caller's thread below)
            box["error"] = e

    t = threading.Thread(target=work, name="er-call", daemon=True)
    t.start()
    t.join(timeout)
    if "out" in box:
        return box["out"]
    raise box.get("error") or TimeoutError(f"no answer from Robotics-ER in {timeout:.1f} s")


def create(jpeg, prompt, schema, timeout=TIMEOUT_S):
    """One request -> (the model's text, seconds). Raises on HTTP / network errors, when `timeout` seconds have
    passed, and on an answer with no text."""
    body = {"model": MODEL,
            "input": [{"type": "image", "data": base64.b64encode(jpeg).decode(), "mime_type": "image/jpeg"},
                      {"type": "text", "text": prompt}],
            "generation_config": {"thinking_level": THINKING},
            "response_format": {"type": "text", "mime_type": "application/json", "schema": schema},
            "store": False}                    # nothing to continue later: do not keep the image on the server
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), method="POST",
                                 headers={"x-goog-api-key": _key(), "Content-Type": "application/json"})
    t0 = time.monotonic()
    try:
        out = _send(req, timeout)
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"Robotics-ER HTTP {e.code}: {e.read().decode(errors='replace')[:300]}") from None
    text = out.get("output_text") or "".join(
        c.get("text", "") for s in out.get("steps") or [] if s.get("type") == "model_output"
        for c in s.get("content") or [] if c.get("type") == "text")
    if not text.strip():
        raise RuntimeError(f"Robotics-ER answered without text (status {out.get('status')!r})")
    return text, time.monotonic() - t0


def parse(text):
    """The model's JSON -> [{"point": [y, x], "colour", "shape"}], points inside 0..1000. Items that are not a
    point are skipped; an answer that is not JSON raises, and so does one whose items are ALL unusable (points in
    another form): that is a broken answer, not "no blocks on the table"."""
    t = text.strip()
    if t.startswith("```"):                    # without a schema the model wraps its JSON in a code fence
        t = t.split("\n", 1)[1].rsplit("```", 1)[0]
    j = json.loads(t)
    items = j.get("items") if isinstance(j, dict) else j
    out = []
    for it in items or []:
        try:
            y, x = (float(v) for v in it["point"])
        except Exception:  # noqa: BLE001
            continue
        if 0 <= y <= 1000 and 0 <= x <= 1000:
            out.append({"point": [int(round(y)), int(round(x))], "colour": str(it.get("colour") or "other"),
                        "shape": str(it.get("shape") or "other")})
    if items and not out:
        raise ValueError(f"none of the {len(items)} items is a [y, x] point in 0..1000")
    return out


def point_blocks(rgb, instruction=None, timeout=TIMEOUT_S):
    """The blocks on the table in one head camera image (H, W, 3 RGB) -> [{"point": [y, x] 0..1000, "colour",
    "shape"}], possibly empty; or None on ANY failure (no key, switched off, API known to be down, network, no
    answer within `timeout` seconds in all, bad answer), so the caller keeps its own detections. Blocking (about
    2 s, never longer than `timeout`): call it from a worker thread, never from robot.act."""
    last.update(seconds=None, error="")
    if not available():
        last["error"] = "no API key, ROBOT_ER=0, or the API is known to be down"
        return None
    t0 = time.monotonic()
    try:
        import cv2
        ok, buf = cv2.imencode(".jpg", np.ascontiguousarray(np.asarray(rgb)[..., ::-1]), [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            raise RuntimeError("could not encode the image")
        prompt, schema = request(instruction)
        text, _ = create(buf.tobytes(), prompt, schema, timeout)
        items = parse(text)
    except Exception as e:  # noqa: BLE001  (fail open: whatever went wrong, the depth detections stand)
        last.update(seconds=round(time.monotonic() - t0, 2), error=f"{type(e).__name__}: {e}".replace(_key() or "\0", "<key>")[:300])
        return None
    last["seconds"] = round(time.monotonic() - t0, 2)
    return items
