# adk_agents/: a Google ADK agent that drives the robot with oracle-based tools

`robot_sorter/` is an ADK agent (ADK 2.14). A Gemini model sees the robot's live camera images and calls tools;
each tool is a piece of the organisers' oracle (`aaltonomous/sim/oracle.py`), reached through the skills in the
project's `agent/skills`. Nothing in `aaltonomous-humanoid/` or `agent/` is edited.

## Setup (once)

ADK and the planner's dependencies are in the project's `.venv` (created with `uv venv .venv`, then
`uv pip install google-adk numpy==2.2.6 scipy==1.15.2 opencv-python-headless==4.10.0.84 pyyaml msgpack websockets pin==3.3.1`).

Put your Gemini API key in `robot_sorter/.env` in place of `PASTE_YOUR_KEY_HERE`. The model is set there too
(`ROBOT_AGENT_MODEL`).

## Run

```bash
# chat with the agent in the browser; the robot is the organisers' mock world, started automatically at 30 Hz
.venv/bin/adk web adk_agents
# the same in the terminal
.venv/bin/adk run adk_agents/robot_sorter
# policy server on port 8000 for the organisers' evaluator (simulator or real robot)
PYTHONPATH=adk_agents .venv/bin/python -m robot_sorter.serve
# score it in the Isaac sim on Modal with the organisers' evaluator (key and model are read from robot_sorter/.env)
modal run --detach sim/modal_sim.py --action eval-adk --trials 3 --seed 1000 --tag adk-1
mkdir -p sim_out && modal volume get aalto-sim-out adk-1 sim_out     # scores, videos, policy.log with every decision
```

In `adk web`, type for example "Put the red objects in the left bin, yellow in the center, blue in the right."
A new session starts a new mock episode with the next layout.

## How it controls the robot in real time

```
30 Hz   robot.act(obs)            never waits: hands out ticks of the current trajectory, or holds
        a tool call               plans with the oracle's planner in a worker thread (0.5-3 s, robot holds),
                                  queues the checked trajectory, returns when the motion has finished
        before every model call   the CURRENT head camera image and robot state are attached to the request
```

| File | What it holds |
|---|---|
| `agent.py` | `root_agent`, the instruction, the callback that attaches the live camera images |
| `tools.py` | the tools (below) |
| `robot.py` | `Robot`: the bridge between the 30 Hz loop and the tools; `MockWorld` for `adk web` |
| `serve.py` | `AdkPolicy` + policy server: one ADK session per episode, runs until `done` or time is up |
| `.env` | API key and model (never commit) |

## Tools and what they take from the oracle

| Tool | Oracle part |
|---|---|
| `look` | the oracle's look, but from the head camera: any saturated colour + depth (the organisers' detector, extended beyond the sim's red / yellow / blue). `ROBOT_PERCEPTION=sim` uses the oracle's exact simulator poses (not allowed in a submission) |
| `view_camera` | not from the oracle: head, wrist or depth image at full resolution, optional zoom region, object ids drawn on |
| `read_proprioception` | not from the oracle: joint angles and speeds, where each hand is, how far it is closed, tracking errors |
| `pick_and_place` | the whole planned move; with `shape` + `yaw_deg`, the face-aligned grasp for cubes and triangles |
| `pick`, `place` | the same move cut in two, so the model can check the hand in between |
| `place_on_counter` | the relay put-down |
| `hand_over` | the relay for an object whose bin is on the far side of the body |
| `go_home`, `wait`, `done` | |

The model works step by step: look, one move, check the result and the live view, next move. An object is named by its id from `look` (or from the `objects_on_counter` list every motion tool returns), or by
pointing at it in the head image (`point`, `[y, x]` 0..1000). A call with no safe plan moves nothing and returns
`failed` with the planner's reason.

## What the tools do by themselves on every move

- **Arms aside.** Between moves both arms are parked out to the sides: clear of each other (at the task's home pose
  the hands hover 13 cm apart and one arm's wrist camera hit the other hand's thumb) and outside the head camera's
  view. A planned move that would bring the arms within 12 cm of each other is refused (`ROBOT_ARM_CLEARANCE_M`).
- **Grasp check.** After the lift, the finger sensors say whether something is in the hand (fingers that hold an
  object stay 0.017-0.07 rad short of their command in the Isaac sim; an empty hand closes fully). On a miss the
  move stops there, the hand opens, the arm goes back, and the tool returns `missed` with how far the attempt
  pushed the object. Threshold `ROBOT_GRASP_ERR_RAD` (default 0.010); it has not been measured on the real hands.
  `ROBOT_GRASP_CHECK=0` turns it off.
- **No ground truth.** Object positions come only from the head camera's colour and depth images. The policy
  process cannot import `zmq`, the only way to reach the simulator's scene service, and the Modal environment
  does not install it. What it does take from the organisers' files: the bins' positions and size, the counter
  height, the camera calibration and the nominal object size.

## Varied scenes

`sim/scene_variant.py` edits the sim container's scene before it starts: the real object colours (yellow, purple,
orange, with the organisers' real mapping), wider shades and lighting, wider placements, and instructions that
name shapes or colour and shape. Example:

```bash
modal run --detach sim/modal_sim.py --action eval-adk --eval-task sort_language --trials 5 --seed 4000 --dr wide \
    --variant "--colors real --dr wide --placement wide --instructions shape" --tag var-real-shape
```

## Status

Scores from the Isaac sim are the only ones that count; see `sim_out/adk-*/runs/eval/*/eval.json` once a run has been
fetched. In the organisers' mock world (no real grasp physics) `gemini-4-hydrogen` sorts every object, with 3-4 s
per decision at `ROBOT_AGENT_THINKING=low`.
