"""The ADK agent: a Gemini model that sees the robot's cameras and calls the oracle-based tools in tools.py.

    adk web adk_agents                    chat with it in the browser; the robot is the organisers' mock world
    adk run adk_agents/robot_sorter       the same in the terminal
    python -m robot_sorter.serve          policy server on port 8000 for the organisers' evaluator (see serve.py)

Before every model call, `attach_live_view` adds the CURRENT head camera image, a short robot state and whatever
the model asked for with view_camera. These are not stored in the session, so the history stays small (fast
calls) and the model always decides on what the cameras show now.

Environment (robot_sorter/.env): ROBOT_AGENT_MODEL (default gemini-3.8-flash). How much the model thinks is chosen
per request by thinking_level(): least on routine turns, more when a move went wrong or the scene is unclear.
"""
from __future__ import annotations

import asyncio
import json
import os

import numpy as np
from google.adk.agents import Agent
from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types

from .robot import get_robot
from .tools import TOOLS, _hand_summary

MODEL = os.environ.get("ROBOT_AGENT_MODEL", "gemini-3.8-flash")
CAMERA_KEYS = {"head": "head_rgb", "left_wrist": "left_wrist_rgb", "right_wrist": "right_wrist_rgb",
               "head_depth": "head_depth"}

INSTRUCTION = """You control a humanoid robot (two arms, three-finger hands) standing at a counter. Small coloured
objects (cubes, cylinders, triangular prisms; about 4.5 cm; red, yellow, blue) lie on the counter in front of three
identical orange bins. "left", "center" and "right" are from the robot's point of view, which is also the head
camera's: the left bin is on the left of the head image.

Your job is to carry out the sorting instruction by calling tools until it is done. Do not ask questions and do
not describe what you are about to do: call tools. The episode is short (about two minutes) and the robot stands
still while you think, so do not spend turns on things you do not need. When you are finished, call done.

What you see: every request ends with a live view: the current head camera image and the robot state (what is
held, how far each hand is closed, time left). It is always current; older images are not kept. For more, call
view_camera (wrist cameras, a zoomed region, depth, object ids drawn on the image), ask_gemini (one closed
question about a camera image, answered in about a second) or read_proprioception (joint sensors: where the
hands are, whether a hand closed on something).

How to work, one step at a time (look, act, check, repeat):
1. Call look. It lists the objects on the counter with ids, positions, a colour and shape label and the raw
   colour the camera measured, and your next live view shows the head image with every id drawn on its object.
   Check the labels against that image, and whether look missed an object you can see.
2. Pick the object that is nearest its bin and call pick_and_place for it. Every result lists the objects that
   are on the counter after the move, so you can go straight on to the next object.
3. Read the status of each move. "completed": the finger sensors felt the object and it was carried to the bin;
   go on. "missed": the hand closed on nothing and the object is still on the counter. The tools remember every
   grip that missed on an object and never repeat it: calling pick_and_place again for the same object uses the
   next grip or the other arm by itself (other_grasps_to_try lists what is left). So simply call it again, or sort
   the other objects first and come back. When no grip is left, change an argument (xy_offset_m the way
   object_moved_m shows it was pushed, max_tilt_deg, grasp_height_frac, finger_direction_deg) or hand it over.
   A block lying on its side (lying_ridge_deg) is gripped across its two ends automatically.
   If look returns a camera_check WARNING, call calibrate once before the first pick, then look again.
4. "failed: no safe plan" means nothing moved; the reason names what was in the way. Find a plan that works
   before moving: call choose_arm to see what each arm can do, and check_plan with variations (several calls in
   one turn are fine): the other arm, max_tilt_deg 50, a pregrasp_offset_m away from the bin rim or the body that
   was in the way, a different finger_direction_deg. Run the one that is ok with pick_and_place and the same
   arguments. If only the grasp is possible (the bin is out of that arm's reach), call hand_over and then
   pick_and_place. If nothing works, leave the object and go on.
5. Call done only when every object is sorted, or every remaining object has no grasp left to try, or less than
   about 12 s remain. Do not stop while time is left and something can still be tried.

Rules:
- "failed" means no safe plan was found and nothing moved; the reason says why. Do something different.
- Sort the easy objects first and come back to a difficult one at the end.
- The left bin is reachable only with the left arm and the right bin only with the right arm.
- Colour and shape: look gives every object a gemini_label, the colour and shape judged by a fast Gemini vision
  model that compared all the objects in the image, with a confidence. Use it together with the image. The
  camera and the light wash colours out (yellow can look pale green, purple can look blue, orange can look tan),
  and every object on the counter is one of the task's colours: so do not look for an exact match, give each
  object the named colour it is nearest to. When a label's confidence is below about 0.7, or two objects with
  the same label look different, settle it first: zoom in with view_camera or ask a closed question with
  ask_gemini.
- Scoring: an object in the right bin scores; one in a wrong bin or knocked off the counter costs a point; one
  left on the counter scores nothing. So sort every object you can assign. Leave an object only when it could
  equally well be two of the named colours (or shapes) and the bins differ.
- The instruction may name colours, shapes or both, and the colours may be any (not only red, yellow, blue).
  Read it each episode. If it is about something the cameras must settle (a shape, which object is which),
  zoom in with view_camera before you decide.
- Things can change while you work (an object rolls, a grasp fails, something is not where look said): always
  act on the latest result and live view, never on an earlier plan.
- Watch time_left_s when it is given: a move takes about 10 s. Call done when nothing useful is left.
"""


def _render(robot, request):
    """One requested camera view -> JPEG bytes: full resolution, optional crop, ids drawn on, depth coloured."""
    import cv2
    from agent.skills.perception import encode_image, xyz_to_point
    camera, obs = request["camera"], robot.obs
    img = obs[CAMERA_KEYS[camera]]
    if camera == "head_depth":
        metres = img.astype(np.float32) * 0.001
        near = np.clip(1.0 - (metres - 0.3) / 0.9, 0.0, 1.0)             # 0.3 m -> red, 1.2 m -> blue
        coloured = cv2.applyColorMap((255 * near).astype(np.uint8), cv2.COLORMAP_JET)
        coloured[metres <= 0] = 0                                        # no depth reading: black
        img = coloured[..., ::-1]
    img = np.ascontiguousarray(img)
    h, w = img.shape[:2]
    if request.get("mark_objects") and camera == "head":
        for o in robot.objects.values():
            y, x = xyz_to_point(robot.ctx, o.xyz, (h, w))
            px = (int(round(x / 1000 * (w - 1))), int(round(y / 1000 * (h - 1))))
            cv2.circle(img, px, 9, (255, 255, 255), 1)
            for colour, thick in (((0, 0, 0), 3), ((255, 255, 255), 1)):
                cv2.putText(img, o.id, (px[0] + 11, px[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, thick, cv2.LINE_AA)
    if request.get("region"):
        y0, x0, y1, x1 = request["region"]
        a, b = int(y0 / 1000 * h), max(int(y1 / 1000 * h), int(y0 / 1000 * h) + 8)
        c, d = int(x0 / 1000 * w), max(int(x1 / 1000 * w), int(x0 / 1000 * w) + 8)
        img = np.ascontiguousarray(img[a:b, c:d])
        if max(img.shape[:2]) < 640:                                     # a small crop: enlarge it for the model
            s = 640 / max(img.shape[:2])
            img = cv2.resize(img, (int(round(img.shape[1] * s)), int(round(img.shape[0] * s))), interpolation=cv2.INTER_CUBIC)
    return encode_image(img, 1024, quality=90)


def live_view(robot):
    """The parts added to every request: the head camera, the wrist camera of a hand that holds something, the
    views asked for with view_camera, and a short state including the hands' proprioception."""
    from agent.skills.perception import encode_image
    obs = robot.obs
    parts = [types.Part(text="Live view of the robot, taken just now."), types.Part(text="[head camera]"),
             types.Part.from_bytes(data=encode_image(obs["head_rgb"], 640), mime_type="image/jpeg")]
    if robot.holding and obs.get(f"{robot.holding}_wrist_rgb") is not None:
        parts.append(types.Part(text=f"[{robot.holding} wrist camera: the hand that holds an object]"))
        parts.append(types.Part.from_bytes(data=encode_image(obs[f"{robot.holding}_wrist_rgb"], 480), mime_type="image/jpeg"))
    requests, robot.view_requests = robot.view_requests, []
    for request in requests:
        label = f"[view_camera: {request['camera']}" + (f", region {request['region']}" if request.get("region") else "") \
                + (", object ids drawn on" if request.get("mark_objects") else "") + "]"
        try:
            image = _render(robot, request)
        except Exception as e:  # noqa: BLE001
            parts.append(types.Part(text=f"{label} could not be shown: {e!r}"[:200]))
            continue
        parts.append(types.Part(text=label))
        parts.append(types.Part.from_bytes(data=image, mime_type="image/jpeg"))
    state = {"instruction": robot.instruction,
             "holding": f"an object in the {robot.holding} hand" if robot.holding else "nothing"}
    try:
        state["hands_closed_fraction"] = {s: _hand_summary(robot, s)["closed_fraction"] for s in ("left", "right")}
    except Exception:  # noqa: BLE001  (state is a convenience; never fail a model call over it)
        pass
    if robot.time_left() is not None:
        state["time_left_s"] = round(robot.time_left(), 1)
    parts.append(types.Part(text="Robot state: " + json.dumps(state)))
    return parts


async def require_api_key(callback_context: CallbackContext, llm_request: LlmRequest):
    """before_model_callback: a readable answer instead of an API error while .env still has the placeholder."""
    if os.environ.get("GOOGLE_API_KEY") == "PASTE_YOUR_KEY_HERE" and not os.environ.get("GEMINI_API_KEY"):
        return LlmResponse(content=types.Content(role="model", parts=[types.Part(
            text="No API key yet: put your Gemini key in adk_agents/robot_sorter/.env (GOOGLE_API_KEY=...) "
                 "and restart.")]))
    return None


async def attach_live_view(callback_context: CallbackContext, llm_request: LlmRequest):
    """before_model_callback: put the current camera images and robot state at the end of the request."""
    robot = get_robot()
    session = getattr(getattr(callback_context, "session", None), "id", None)
    if not await asyncio.to_thread(robot.ensure_running, session):
        return None                                    # no robot yet: let the model answer without images
    llm_request.contents.append(types.Content(role="user", parts=await asyncio.to_thread(live_view, robot)))
    if isinstance(callback_context_agent_model(callback_context), str):       # a real Gemini model, not a stand-in
        level, why = thinking_level(robot)
        llm_request.config.thinking_config = types.ThinkingConfig(thinking_level=level)
        robot.trace.write("thinking", t=round(robot.t, 2), level=level, why=why)
    return None


def callback_context_agent_model(callback_context):
    try:
        return callback_context._invocation_context.agent.model
    except Exception:  # noqa: BLE001
        return MODEL


def thinking_level(robot):
    """How much the model should think on THIS request. The robot stands still while it thinks, so routine turns
    get the least and only turns that need it get more. Signals, all already at hand (no extra call):
      - the Decisions API's view of the scene from the last look: its difficulty score (0 easy .. 2 hard) and how
        confident it was about each object's colour and shape;
      - whether the last moves went wrong (missed, failed, aborted);
      - whether the instruction has exceptions ("the rest", "other", "whatever", "except")."""
    # gemini-4-hydrogen supports low, medium and high (it rejects "minimal": checked live, 14 Nov).
    if not robot.seen:
        return "low", "nothing looked at yet: the only sensible call is look"
    conf = [min(l.get("colour_confidence", 1.0), l.get("shape_confidence", 1.0))
            for i, l in robot.labels.items() if i in robot.objects and isinstance(l, dict)]
    trouble = sum(1 for st in robot.recent[-2:] if st in ("missed", "failed", "aborted"))
    tricky = any(w in (robot.instruction or "").lower() for w in (" rest", "other", "whatever", "except", "everything else"))
    unsure = (robot.difficulty or 0.0) >= 1.3 or (conf and min(conf) < 0.7)
    # Measured offline on saved frames (sim/replay_decisions.py --thinking): "medium" on every turn decided no
    # better than "low" (hard scene 13 right / 7 wrong against 14 / 6; easy scene 20 / 0 both) at about the same
    # time per turn. So "low" is the rule and "medium" is kept for the two cases where a second look is cheap
    # insurance: two moves in a row went wrong, or the scene looks unsure AND the instruction has exceptions.
    if trouble >= 2 or (unsure and tricky):
        return "medium", "two moves in a row went wrong, or an unsure scene with an instruction that has exceptions"
    return "low", "routine"


def build_agent(model=None) -> Agent:
    """`model`: a model name, or any ADK BaseLlm (the offline test passes a scripted stand-in)."""
    return Agent(
        name="robot_sorter",
        model=model or MODEL,
        description="Sorts coloured objects into bins with a Unitree G1 humanoid, by calling oracle-based motion skills.",
        instruction=INSTRUCTION,
        tools=list(TOOLS),
        before_model_callback=[attach_live_view] if model is not None else [require_api_key, attach_live_view],
    )


root_agent = build_agent()
