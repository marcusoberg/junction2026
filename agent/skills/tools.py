"""The skills as tools for a language model: declarations (name, description, JSON-schema parameters), a system
prompt, and validation of what the model answers. Provider-neutral on purpose: turn TOOLS into whatever your model
API wants (Gemini function declarations take this shape almost verbatim), call the model, and hand the chosen call
back as a decision dict  {"skill": <tool name>, **arguments}  to agent.runtime.LLMChooser.

    ask(world) -> decision          # you write this: build the request from SYSTEM_PROMPT, TOOLS and `world`
    chooser = LLMChooser(ask, fallback=RuleChooser(), timeout_s=4)
    policy = SkillPolicy(chooser=chooser)

`world` is what SkillPolicy passes to the chooser: instruction, mapping, time_left_s, bins, objects, rejected, history.
"""
from __future__ import annotations

APPROACH_SCHEMA = {
    "type": "object", "description": "Optional variations of the grasp. Leave out for the default.",
    "properties": {
        "finger_direction_deg": {"type": "number", "description": "Where the fingers point: 0 forward, +90 left."},
        "max_tilt_deg": {"type": "number", "description": "Allowed palm tilt from straight down (default 35)."},
        "pregrasp_offset": {"type": "array", "items": {"type": "number"},
                            "description": "(dx, dy) m: start the descent off to one side."},
        "speed": {"type": "number", "description": "Arm joint speed rad/s, at most 0.97."},
        "carry_height": {"type": "number", "description": "m above the counter while carrying (default 0.16)."},
        "grasp_height_frac": {"type": "number", "description": "Grasp height as a fraction of object height (0.5)."},
        "xy_offset": {"type": "array", "items": {"type": "number"},
                      "description": "(dx, dy) m correction of where to grasp, if the detected position looks off."}}}

TOOLS = [
    {"name": "pick_and_place",
     "description": "Pick one object from the counter with a top-down pinch, drop it in a bin, return the arm home. "
                    "About 10 s. Fails (and is reported back in `rejected`) if no safe plan exists.",
     "parameters": {"type": "object", "required": ["object", "bin"], "properties": {
         "object": {"type": "string", "description": "id from world.objects with where == 'counter'"},
         "bin": {"type": "string", "enum": ["left", "center", "right"]},
         "arm": {"type": "string", "enum": ["left", "right"],
                 "description": "Optional. Default: the arm on the bin's side (centre bin: the object's side). "
                                "The left bin is reachable only by the left arm, the right bin only by the right."},
         "approach": APPROACH_SCHEMA,
         "drop_offset": {"type": "array", "items": {"type": "number"},
                         "description": "(dx, dy) m inside the bin, to avoid stacking objects."}}}},
    {"name": "relocate",
     "description": "For an object whose bin is on the other side of the body: pick it with the near arm and set it "
                    "down near the centre line, where the other arm can pick it on a later turn. About 10 s.",
     "parameters": {"type": "object", "required": ["object", "bin"], "properties": {
         "object": {"type": "string"}, "bin": {"type": "string", "enum": ["left", "center", "right"]}}}},
    {"name": "done",
     "description": "Nothing sensible is left to do (everything sorted, or the rest cannot be grasped). Ends the episode.",
     "parameters": {"type": "object", "properties": {}}},
]

SYSTEM_PROMPT = """You control a humanoid robot that sorts small coloured objects on a counter into three bins
(left, center, right, named from the robot's point of view). You are called once per move with the current world
state as JSON and must answer with exactly one tool call.

Rules:
- Follow `instruction`. `mapping` is our reading of it (colour or shape -> bin); correct it if it is wrong.
- Only objects with where == "counter" can be picked. Objects already in a bin stay there.
- A correct object scores, an object in a wrong bin or knocked off the counter costs a point, an object left on the
  counter costs nothing. When unsure about an object, leave it.
- `rejected` lists calls from this turn that had no safe plan, with the reason. Do not repeat them; choose another
  object, another arm, or `relocate`.
- An object with attempts >= 2 has been tried and missed; prefer others.
- Each move takes about 10 s; `time_left_s` is what remains. Call `done` when nothing useful is left.
"""


def validate(decision, world):
    """None if the decision is usable, else why not."""
    if not isinstance(decision, dict):
        return "not an object"
    skill = decision.get("skill")
    if skill == "done":
        return None
    if skill not in ("pick_and_place", "relocate"):
        return f"unknown skill {skill!r}"
    ids = {o["id"] for o in world.get("objects", []) if o.get("where", "counter") == "counter"}
    if decision.get("object") not in ids:
        return f"object {decision.get('object')!r} is not on the counter (have {sorted(ids)})"
    if decision.get("bin") not in (world.get("bins") or {"left": 0, "center": 0, "right": 0}):
        return f"unknown bin {decision.get('bin')!r}"
    if decision.get("arm") not in (None, "left", "right"):
        return f"unknown arm {decision.get('arm')!r}"
    return None
