# Prompt for Google AI Studio: Gemini agent + skill tools for the G1 sorting challenge

How to use: start a new chat in AI Studio with the team's hackathon key, attach the files listed under
"Files to attach", then paste everything below the line. The files matter more than the summary: the model
should read the real code, not our description of it.

**Files to attach** (all under `aaltonomous-humanoid/aaltonomous/` unless noted):
`sim/oracle.py`, `baselines/classical/planner.py`, `baselines/classical/kinematics.py`,
`baselines/classical/policy.py`, `baselines/classical/perception.py`, `baselines/classical/config.yaml`,
`interface/safety.py`, `interface/tasks.py`, `interface/calibration.yaml`, `interface/joint_spec.yaml`,
`scene/scene.yaml`, `deploy/policy_server.py`, `deploy/protocol.py`, `sim/scene_client.py`,
and from our own code: everything under `agent/` (`README.md`, `skills/*.py`, `runtime/*.py`,
`tests/test_skills.py`), `submission/server.py` and `sim/gen_demos.py`.

---

You are helping a four-person hackathon team (Junction 2026, "Aaltonomous Humanoid" challenge, deadline tomorrow)
design and write a Gemini-driven agent that controls a humanoid robot through a library of motion and perception
tools. Read the attached files first; they are the ground truth. Where this text and the code disagree, the code
wins, and tell us about the disagreement.

## 1. The task

A Unitree G1 with two Dex3 three-finger hands stands at a counter. On the counter are 3 to 5 objects, each a
4.5 cm cube, cylinder or triangular prism in red, yellow or blue, and behind them three identical orange bins
(left, centre, right; 21 cm square, 6 cm walls). Three difficulty levels:

1. `sort_fixed`: red → left bin, yellow → centre, blue → right. +1 per correct object, −1 wrong bin, −1 knocked
   off the counter, 0 if left on the counter.
2. `sort_language`: the colour → bin mapping is given in a natural-language instruction, different every episode;
   the finals use sentence templates we have not seen (colour first, possibly shapes, no negation). +2 per correct.
3. `insert_shapes`: one bin has a lid with a square, a round and a triangular hole (1 cm clearance); each object
   goes through the hole of its shape. +3 inserted, −1 forced or dropped.

An episode lasts 120 s. Finalists are chosen by a simulation run (Isaac Sim) on a held-out variant of the scene,
then run on the real robot.

## 2. The interface we must implement

- A policy server in a Docker container: websocket + msgpack on port 8000 (`deploy/protocol.py`,
  `deploy/policy_server.py`). Subclass `Policy`, implement `reset(instruction, task, seed)` and `act(obs)`.
- `obs`: `head_rgb` (480×640×3 uint8), `head_depth` (480×640 uint16, mm, aligned to the colour image),
  `left_wrist_rgb`, `right_wrist_rgb`, `arm_q` (14), `arm_dq` (14), `hand_q` (14), `instruction` (str), `t` (s).
- `act` returns absolute joint targets in radians: `arm_q_target` (14,) or (k,14) and `hand_q_target` likewise,
  executed at 30 Hz, optionally `done: True`. Order: left arm 7, right arm 7; left hand thumb0-2, middle0-1,
  index0-1; right hand thumb0-2, index0-1, middle0-1. Left fingers close with negative angles, right with positive.
- The runner calls `act` again when the returned chunk is nearly used up and holds the last target while we
  compute. **`act` must return within a few milliseconds every time**; anything slow has to run in a background
  thread while `act` keeps returning holds or the rest of the current trajectory. Our `submission/server.py` shows
  this pattern for a neural policy.
- Safety filter (`interface/safety.py`), applied by the runner to everything we send: joint limits minus a margin,
  at most 1.0 rad/s of command change on arm joints and 3.0 rad/s on fingers, a workspace box, and a keep-out
  volume below the counter surface. More than 30 clipped ticks aborts the episode, and the sim qualification gate
  requires **zero** clipped ticks and a command rate of at least 27 Hz. So every trajectory must be checked
  against a local copy of the filter before it is sent (`Planner.dry_run_safety` already does this).
- Frames: everything is in the pelvis frame (x forward, y left, z up). The counter surface is at z ≈ +0.01 m.
  Head camera intrinsics and extrinsics are in `interface/calibration.yaml`; wrist camera extrinsics are unknown.
- Reach limits we measured: only the near half of each bin is reachable, the left bin only by the left arm and the
  right bin only by the right arm.
- The first few camera frames after an episode reset in the sim still show the previous scene; ignore the first
  15 ticks.
- Internet at the finals: the organisers said it "can have" network, but it is not guaranteed. The agent must
  therefore keep working, with reduced intelligence, when the Gemini API is slow or unreachable.

## 3. What already exists (in the attached files)

**A scripted solver, the "oracle"** (`sim/oracle.py`, built on `baselines/classical/`). It scores about 97 % on
cubes, 91 % on cylinders and 35 % on triangular prisms in the sim. It cannot be submitted as is, because it reads
exact object poses from the simulator. How it works:

- `PrivilegedPerception.detect()` → list of `Detection` (colour, shape, name, top-face centre in the pelvis frame,
  yaw) from the sim's scene service.
- `Planner.plan(det, arm_q, hand_q, bin_name)` → a `Plan` with `arm` (T,14) and `hand` (T,14) joint targets at
  30 Hz and named `phases`: to_pregrasp, descend, close, lift, carry, open, retreat, home. Internally
  `_plan_side(det, side, bin_name, arm_q, hand_q, place_xy=None)` builds six waypoints for the grasp point
  (pregrasp, grasp, lift, above_bin, drop, retreat), solves them with `ArmKinematics.solve` (Pinocchio IK,
  position target plus "palm faces down within N degrees" plus a preferred finger direction), connects them with
  `_cartesian` or `_joint_carry`, time-scales them with `time_scale` and `ramp`, and rejects the plan if the hand
  comes too close to the counter (`clearance`) or a bin rim (`bin_clearance`) or if `dry_run_safety` reports a clip.
- Oracle additions: face-aligned grasps for cubes and triangles (`_plan_side_faces`), a hand-over via a counter
  spot both arms reach when one arm cannot do the whole move (`_plan_relay`), drop points spread inside the bin.
- `ClassicalPolicy.act(obs)` is the executor: look, plan in a background thread, hand out the plan in chunks of 10
  ticks, hold while planning. It runs open loop: a missed grasp is only noticed at the next look.
- The grasp is always a claw from above: thumb against one finger, palm down. Tunable through `config.yaml` and
  `ORACLE_OVERRIDES`: finger direction hints, maximum tilt, approach / carry / drop heights, speeds and
  accelerations, arm preference, hand postures, grasp point in the palm.

**A camera-based detector** (`baselines/classical/perception.py`): HSV colour blobs + depth → `Detection` in the
pelvis frame (colour and position, no shape). The organisers measured about 4 mm error in the nominal sim.

**Our neural policy** (π0.5 fine-tuned on oracle demonstrations, served by `submission/server.py`). It performs
the pick-and-place motion but is still imprecise (3 of 12 objects sorted). It could become one more tool
("run the learned skill with this prompt for N seconds").

**Our skill library and runtime** (`agent/`, read `agent/README.md` first). Already written and tested offline:
- `agent/skills/`: `SkillContext`; `detect_camera`, `detect_sim`, `joint_state`; checked motion skills
  `pick_and_place`, `pick`, `place`, `place_on_counter`, `relocate`, `go_home`, `set_hand` returning a `Trajectory`;
  `Approach` (finger direction, tilt, slanted descent, speed, carry height, grasp height, position correction).
  With default arguments `pick_and_place` is identical to the oracle planner.
- `agent/runtime/`: `SkillPolicy` (non-blocking `act()`, one background decision per move, attempt tracking, a
  hard attempt budget), `RuleChooser`, `RandomChooser`, `LLMChooser(ask, fallback, timeout_s)`, `parse_mapping`
  (instruction → colour/shape → bin, no model), and a policy server.
- `agent/skills/tools.py`: tool declarations (`pick_and_place`, `relocate`, `done`), a system prompt, `validate`.
- `agent/runtime/agent_policy.py`: **`AgentPolicy`, the loop we want to run.** A tool-calling model
  (`brain.decide(view)`) gets the raw head and wrist images and answers with one tool call, naming an object by
  pointing at it in the head image (`[y, x]`, 0..1000); pointing + depth gives the 3D target. A fast closed-question
  model (`reflex.answer(view, questions)`) is asked during a move (after the lift: is the hand holding an object?)
  and a confident "no" aborts the move. Tested offline with stand-in models.
- `agent/runtime/gemini.py`: `GeminiBrain` (`client.interactions.create` with tools and image blocks, default
  `gemini-4-hydrogen`) and `GeminiReflex` (`client.decisions.create`, `gemini-decision-1`). **Written from the
  docs, never run.** Check every call in it against the current API and fix it.
- In the Isaac sim the skills with a rule-based chooser sort 16 of 20 objects with zero safety clips.
- **What we want from you:** make `gemini.py` correct; improve the prompts and tool descriptions; more reflex
  questions during a move (object slipping, about to hit a bin wall, object knocked over) and what to do on each
  answer; the skills listed in 4A that do not exist yet; level 3.

**A demonstration recorder** (`sim/gen_demos.py`): runs the oracle in the sim and stores every tick (three camera
images, joint states, commands) plus, per pick-and-place, the object, colour, shape, bin, arm and phase timings.

## 4. What we want from you

### A. Extend the skill library in `agent/`

Review what is there, tell us what is wrong or fragile, and extend it in the same style: wrap, do not edit the
organisers' files; each skill is a plain Python function with typed arguments and a JSON-serialisable result, so
the same function can be (1) called by the Gemini agent as a tool, (2) called by the rule-based fallback, (3)
called with randomised parameters to record more varied demonstrations. The lists below are the full set we would
like; the README says which already exist.

Perception and state tools, for example:
- `get_joint_state()` → arm and hand angles, grasp-point pose of each hand (forward kinematics), what each hand
  is probably holding.
- `get_camera_image(camera)` → the current frame, for sending to Gemini.
- `detect_objects(source="camera" | "sim")` → objects with id, colour, shape (if known), position, yaw, which bin
  they are in or "counter". `sim` is only for recording demonstrations.
- `check_reachability(object_id, arm, bin)` → whether a full plan exists and why not.

Motion tools, each returning a checked trajectory handle rather than moving directly, for example:
- `pick(object_id, arm, approach)` where `approach` covers finger direction, tilt, approach height, speed.
- `place(bin, arm, offset)`, `place_on_counter(xy, arm)`, `hand_over(object_id, from_arm, to_arm)`.
- `move_hand(arm, position, tilt, finger_direction)`, `set_hand(arm, posture)`, `go_home(arm | "both")`.
- `insert(object_id, hole)` for level 3: align the object's shape and yaw with the hole, lower through it.
- `execute(handle)` / `cancel()` / `status()`: the executor hands the trajectory to `act()` chunk by chunk.

Tell us which of these the existing planner can do with configuration only, which need small extensions, and which
need new planning code. For the new ones, write the code.

We also want more variety than the oracle has today: side and angled approaches, different pre-grasp paths,
re-grasping after a miss, pushing an object away from a bin wall or from a neighbour before picking it. Propose
which of these are worth having for this task and implement the ones that are.

### B. The agent runtime

- `agent/runtime/policy.py` already gives a `Policy` whose `act()` never blocks and a background decision per
  move through a chooser. Build on it: write the Gemini-backed `ask(world)` for `LLMChooser`, and change the loop
  where the model needs more than one call per move (for example look, then decide).
- The agent loop observes (tools above, plus head camera images when useful), asks Gemini what to do
  next, validates the answer, and queues skills. One decision per pick-and-place is the right granularity; the
  30 Hz loop is never waiting on the network.
- We want to use the **Gemini Decisions API** that Google released this week, and we have a hackathon API key. We
  have not used it and cannot vouch for any detail of it. Use the current official documentation, show exactly how
  you call it (model name, request, tool or schema declaration, how the result comes back), and say plainly which
  parts you are sure of and which you are assuming. If plain function calling or structured output with a Gemini
  model is the more reliable choice for some step, say so and use it there. If a Gemini robotics or
  embodied-reasoning model is better for locating objects, shapes and lid holes in the images, use it for that.
- Level 1: fixed mapping, no language needed. Level 2: parse the instruction into a colour (and maybe shape) → bin
  mapping once at `reset`, with a rule-based parser as fallback. Level 3: recognise shape and yaw, choose the hole.
- **Fallbacks, in this order:** Gemini decision → cached mapping plus a rule-based planner (nearest object to its
  bin first, as the oracle does) → hold. A timeout or a malformed answer must never stall or crash the episode.
- Everything the model proposes passes the same checks as the oracle's plans (clearances and the safety dry run)
  before a single tick is sent.
- Log every decision with its inputs, the answer, latency, and the outcome seen at the next observation.

### C. A demonstration mode

The same skills driven by a random or scripted chooser instead of Gemini, in the sim, to record more varied data
for the neural policy: random object order, arm choice where both work, approach direction and height, speed,
drop position, plus recoveries after a missed grasp. Keep `sim/gen_demos.py`'s recording format and add the skill
name and arguments of every segment to its labels, so each segment can be given its own language prompt.

### D. Tests

Offline tests that need no simulator (the repo has a kinematic mock in `baselines/classical/mock_world.py`): every
skill's trajectory passes the safety dry run; `act()` returns within 5 ms while a fake slow model takes 3 s; the
fallback takes over when the model call raises or times out.

## 5. Constraints you must respect

- Python 3.12, numpy, Pinocchio (already used), no ROS. Keep new dependencies minimal.
- Never send a target the safety filter would clip: zero clips is a hard requirement.
- Never block `act()`.
- Do not rely on privileged simulator state outside demonstration mode.
- API keys come from an environment variable; never write one into code or logs.
- We have one day. Prefer a small design that works for level 1 and 2 end to end over a complete design for all
  three levels. Mark clearly what is finished, what is a stub, and what you could not verify.

## 6. What to give us

1. A short design: the tool list with signatures, the agent loop, the fallback chain, and the latency budget.
2. A table mapping each tool to existing code (function names in the attached files) and what is new.
3. The code, as files under `agent/` (`skills/`, `runtime/`, `tests/`), ready to run.
4. The exact Gemini calls, with the caveats asked for in 4B.
5. A list of open questions for us, and of anything in the attached code that looked wrong.
