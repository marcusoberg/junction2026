# agent/: oracle-based skills and a loop that lets Gemini drive them from the camera images

The organisers' oracle planner, cut into importable skills, and a runtime in which a multimodal model looks at the
raw camera images and calls those skills as tools. Nothing in `aaltonomous-humanoid/` is edited; this imports it.
No training is involved.

## How the loop works (`runtime/agent_policy.py`)

```
30 Hz   act(obs)            never waits: hands out ticks of the current trajectory, or holds
~10 s   brain.decide(view)  Gemini 4 Hydrogen / Carbon (tool calling). Gets the head and wrist images as images,
                            the instruction, what is held, time left and the result of earlier calls.
                            Answers with ONE tool call. It names an object by POINTING at it in the head image.
~0.2 s  reflex.answer(...)  Gemini Decisions API (gemini-decision-1): a closed question on the current images
                            during a move. Right after the lift: "is the hand holding an object?". A confident
                            no stops the move, opens the hand, sends the arm home, and the brain is told.
```

- The model's point (`[y, x]`, 0..1000 on the head image, Gemini's pointing convention) plus the depth image gives
  the 3D grasp target (`skills/perception.py: point_to_xyz`). Nothing about the scene is turned into text.
- Every tool call is planned by the oracle-based skills and must pass inverse kinematics, hand clearances and a
  zero-clip dry run through the organisers' safety filter before a tick is sent. A call with no safe plan is not
  executed; the model gets the reason in the next turn's history.
- A model call that fails or times out costs half a second of holding still, then it is asked again.

Tools the model has: `pick_and_place(point, bin, ...)`, `pick(point, ...)`, `place(bin)`,
`place_on_counter(point)`, `go_home()`, `wait(seconds)`, `done()`; optional `arm`, `shape`, `yaw_deg` and an
`approach` (finger direction, slanted descent, speed, tilt, grasp height, position correction).
Declarations and the system prompt are at the top of `runtime/agent_policy.py`.

## Files

```
skills/context.py       SkillContext: config, scene, kinematics, planner (build once)
skills/planner.py       SkillPlanner: the oracle planner with stages (pick / place / home) and slanted approaches
skills/motion.py        pick_and_place, pick, place, place_on_counter, relocate, go_home, set_hand, hold; Approach
skills/perception.py    point_to_xyz, encode_image, joint_state; detect_sim (demonstrations); detect_camera (fallback)
runtime/agent_policy.py AgentPolicy + tool declarations + system prompt
runtime/gemini.py       GeminiBrain (Interactions API, tools + images), GeminiReflex (Decisions API)
runtime/server.py       policy server on port 8000 (AGENT_MODE=gemini for the loop above)
runtime/policy.py, choosers.py, language.py   SkillPolicy: the same skills without a model (rule / random chooser)
tests/test_agent.py     the loop with stand-in models;  tests/test_skills.py   the skills
```

## Status: what has been run and what has not

| Piece | Status |
|---|---|
| Skills (pick, place, pick-and-place, hand-over, home) | Offline tests pass. With default arguments identical to the oracle planner. In the Isaac sim: 16 of 20 objects, 0 safety clips, gate passed (5 episodes, rule chooser). |
| `AgentPolicy` loop: pointing → 3D target, planning, execution, reflex abort, bad calls, non-blocking | Offline tests pass with stand-in models (`tests/test_agent.py`). **Not yet run in the Isaac sim.** |
| `gemini.py` (the actual API calls) | **Never run.** Written from the documentation; no API key was available. Expect small fixes. |

## First steps with the key

```bash
export GEMINI_API_KEY=...            # the team key; never commit it
pip install -U google-genai

# 1. one call to each model on a saved head-camera image from the sim
PYTHONPATH=aaltonomous-humanoid:. python -m agent.runtime.gemini agent/tests/data/head_example.jpg

# 2. the offline tests (no key needed)
PYTHONPATH=aaltonomous-humanoid:. python -m unittest agent.tests.test_agent agent.tests.test_skills

# 3. in the Isaac sim on Modal, scored by the organisers' evaluator, with video
modal secret create gemini GEMINI_API_KEY=...
modal run --detach sim/modal_sim.py --action eval-agent --trials 2 --seed 1000 --tag agent-1 --background
mkdir -p sim_out && modal volume get aalto-sim-out agent-1 sim_out      # policy.log has every decision
#    other models:  --server-env "AGENT_BRAIN_MODEL=gemini-4-carbon"   (default gemini-4-hydrogen)
#                   --server-env "AGENT_BRAIN_MODEL=gemini-robotics-er-2-preview"
#    no reflex:     --server-env "AGENT_REFLEX=0"
```

What the docs say about the models (November 2026; check before relying on it): `gemini-decision-1` answers
closed questions about text and images in 150–250 ms and cannot call tools; `gemini-4-hydrogen` (fast),
`gemini-4-argon` and `gemini-4-carbon` (most deliberate) take images and call tools through
`client.interactions.create`; `gemini-robotics-er-2-preview` is the embodied-reasoning model with pointing output.

## Known limits

- One brain decision per move, about every 5–10 s. The reflex is the only thing that reacts within a move, and
  today it asks one question (holding or not, after the lift). More checks are a few lines each: add a tick and a
  question in `AgentPolicy._plan`, handle the answer in `_reflexes`.
- Latency of the brain model is unknown to us. While it thinks, the robot holds still.
- Grasps are top-down pinches only. Triangular prisms fail often in the sim even for the oracle (35 %). Passing
  `shape` and `yaw_deg` enables the oracle's face-aligned grasp, if the model's yaw is accurate.
- No side grasps, pushing, re-grasping in the air, or insertion (level 3) yet.
- Level 2 needs nothing extra: the model reads the instruction itself.
