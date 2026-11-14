# Junction 2026, Aaltonomous Humanoid: where we are (Sat 14 Nov, 17:40)

Read this top to bottom once; after that the command list in section 4 is all you need day to day.

## 0. Getting this repository to run

```bash
git clone https://github.com/zupermiez/aaltonomous-humanoid.git      # the organisers' repo, next to this file (not committed here)
uv venv .venv --python 3.11
uv pip install --python .venv/bin/python google-adk==2.14.1 numpy==2.2.6 scipy==1.15.2 opencv-python-headless==4.10.0.84 pyyaml msgpack "websockets>=12" pin==3.3.1
cp adk_agents/robot_sorter/.env.example adk_agents/robot_sorter/.env   # then put the Gemini API key in it
```

Not in the repository: the API key (`.env`), the virtual environment, and run outputs (`sim_out/`, `sim/out/`:
videos and logs, fetched from the Modal volume `aalto-sim-out`). The current work is the ADK agent in `adk_agents/`;
start with `adk_agents/README.md`.

## 1. The challenge in short

- Robot: Unitree G1 with two Dex3 hands, standing at a counter. Task: sort coloured objects (cube, cylinder,
  triangular prism; red, yellow, blue) into three identical orange bins (left, centre, right).
- Three difficulty levels:
  1. `sort_fixed`: red → left, yellow → centre, blue → right. +1 per correct object, −1 wrong bin, −1 knocked off.
  2. `sort_language`: the colour → bin mapping comes from the instruction sentence. +2 per correct object.
  3. `insert_shapes`: put each object through the matching hole in a lid on one bin. +3 per object.
- Submission: a Docker image with a policy server (websocket + msgpack, port 8000). 30 times a second it gets
  camera images and joint angles and must return joint targets for 14 arm joints and 14 finger joints.
- Episodes last 120 s. A safety filter clips anything too fast or out of bounds; the sim "gate" requires zero
  clipped steps and a command rate of at least 27 Hz.
- Finalists are chosen by a **simulation** evaluation on a held-out variant, then run on the real robot on Sunday.

## 2. What is in the project folder

| Path | What it is |
|---|---|
| `aaltonomous-humanoid/` | The organisers' repo, unmodified. Sim, evaluator, oracle, safety filter, docs. |
| `sim/modal_sim.py` | Runs the organisers' Isaac sim on Modal: scene check, oracle baseline, evaluating our model, recording demonstrations. |
| `sim/gen_demos.py` | Records the oracle solving episodes (all three cameras, joints, commands, per-object labels). |
| `sim/to_dataset.py` | Turns those recordings into a LeRobot v3.0 training dataset. |
| `training/modal_pi05.py` | Fine-tunes π0.5 (LeRobot 0.7.0) on Modal. |
| `training/openloop_check.py` | Offline check: feeds dataset frames through the server code and compares predicted with recorded actions. |
| `submission/server.py`, `submission/Dockerfile` | Our policy server and its image. **The image has never been built** (no Docker on the laptop used so far). |
| `ORGANIZER_ISSUES.md` | Everything we found wrong or unclear in the organisers' material (A1–A11, B–E). Send them this. |
| `sim/out/` | Videos and frame sheets: the oracle with all three cameras, and our checkpoints in the sim. |
| `agent/` | The oracle-based skills (`agent/skills/`), which the ADK agent's tools call, and an earlier hand-written Gemini loop (`agent/runtime/`) that has never been run and is superseded by `adk_agents/`. |
| `agent/AI_STUDIO_PROMPT.md` | Prompt for Google AI Studio to review and extend the agent. |
| `adk_agents/robot_sorter/` | **Current focus.** A Google ADK agent: `gemini-4-hydrogen`, through the Gemini API with the key in its `.env`, looks at the camera images and calls the skills as tools. It runs in the Isaac sim on Modal (`--action eval-adk`); scores are in section 5. Start with `adk_agents/README.md`. |

Everything heavy lives on Modal volumes, not on a laptop: `aalto-data` (datasets, raw recordings), `aalto-runs`
(checkpoints), `aalto-sim-out` (sim results and videos), `aalto-kitcache`, `aalto-hf-cache`.

## 3. How the pieces fit

1. **Sim** (Modal, one L40S GPU per sim): the organisers' scene, started with their `aalto sim`. It runs at about
   0.3× real time and is limited by the CPU, not the GPU (GPU at 11 %).
2. **Evaluator** (`aalto eval`): places objects from a seed, runs the 30 Hz loop for up to 120 s, scores where the
   objects ended up. This is what picks the finalists, so we use it unchanged.
3. **Oracle**: the organisers' scripted solver. It reads exact object positions from the sim (not allowed in a
   submission) and plans with inverse kinematics. We use it as a reference and to record demonstrations.
4. **Demonstrations**: oracle episodes recorded tick by tick, converted to a dataset in the same format as the
   organisers' real-robot data drop (so the two can be merged).
5. **Training**: π0.5 fine-tuned on that dataset.
6. **Policy server**: loads a checkpoint, answers the evaluator. It predicts 50 steps at a time, re-plans about
   every 25 steps, and never sends anything the safety filter would clip.

## 4. Commands

Needs the `modal` CLI logged in to the team workspace, and the workspace secret `huggingface` (a Hugging Face
token with access to `google/paligemma-3b-pt-224`). Run from the project folder.

```bash
# sim: scene check + camera snapshots / oracle baseline
modal run sim/modal_sim.py --action boot
modal run sim/modal_sim.py --action baseline --trials 3

# record demonstrations: 8 sims in parallel, 16 episodes each (about 1 h, about $18)
modal run --detach sim/modal_sim.py --action datagen --name sim_fixed_v4 --shards 8 --episodes 16 --seed 400000 --dr mix --background
# make it trainable
modal run training/modal_pi05.py --action prepare --src sim_fixed_v4 --name sim_fixed_v4
# several prepared datasets into one
modal run training/modal_pi05.py --action merge --names sim_fixed_v2,sim_fixed_v3 --name sim_fixed_v23

# train (B200). --init continues from one of our checkpoints instead of the base model
modal run --detach training/modal_pi05.py --action train --mode full --dataset sim_fixed_v23 --run pi05-sim-v23-full \
    --init pi05-sim-v2-full/checkpoints/006000 --steps 6000 --save-freq 1000 --gpu B200 --workers 40 --background
modal run training/modal_pi05.py --action ls --run pi05-sim-v23-full        # which checkpoints exist

# evaluate a checkpoint in the sim (3 episodes, about 25 min) and fetch scores + videos
modal run --detach sim/modal_sim.py --action eval-model --ckpt pi05-sim-v2-full/checkpoints/003000 --trials 3 --tag full-3000 --background
mkdir -p sim_out && modal volume get aalto-sim-out full-3000 sim_out
#   scores: sim_out/full-3000/runs/eval/<run>/eval.json   videos: episode_NNN.mp4 next to it

# offline sanity check of a checkpoint against its training data
modal run training/openloop_check.py --ckpt pi05-sim-v2-full/checkpoints/003000 --dataset sim_fixed_v2
```

Always use `--detach` together with `--background` for anything long. Without `--background` the job is cancelled
when the laptop that started it sleeps or loses its network (this killed runs twice).

## 5. Results so far

Oracle (the ceiling for imitation), 125 recorded episodes: cubes 97 %, cylinders 91 %, triangular prisms 35 %.
The organisers' 3-episode baseline shows 92 % only because those three layouts contain no triangles.

Our π0.5 checkpoints in the sim, rung 1, seeds 1000–1002, gate passed in every run (0 clips, 30 Hz):

| Checkpoint | Trained on | Objects sorted | Mean score |
|---|---|---|---|
| `pi05-pickcube-v1` 2,000 | 7 real episodes | 0 of 8 | 0 |
| `pi05-sim-v1a` 1,000 / 2,500 | 148 single pick-and-places (flawed cut, see 6) | 0 of 8 | −0.5 |
| `pi05-sim-v2` 2,000 | 125 episodes, fixed cut | 0 of 8 | −0.5 |
| `pi05-sim-v2` 4,000 | same | 1 of 12 | −0.33 |
| `pi05-sim-v2` 5,000 | same | 3 of 12 | +0.33 |
| `pi05-sim-v2-full` 1,000 (whole model, on top of the 5,000) | same | 2 of 12 | +0.33 |

| `pi05-sim-v2-full` 3,000 | same | 4 of 12 | +1.33 |

Not learned, same evaluator, seeds 1000–1004 (5 episodes, 20 objects), gate passed, 0 clips:

| Policy | Objects sorted | Mean score |
|---|---|---|
| Organisers' classical baseline (camera colour blobs + their planner; their reference submission) | 16 of 20 | 3.2 |
| Our skill library, rule-based chooser, camera positions (`agent/`, no model) | 16 of 20 | 3.2 |
| Our skill library, rule-based chooser, exact simulator positions | 16 of 20 | 3.2 |
| **ADK agent, `gemini-4-hydrogen`** (`adk_agents/`), seeds 1000–1002, run `adk-1` | 10 of 12 | 3.33 |
| ADK agent, `sort_language` (2 points per object), seeds 1000–1002, run `adk-3-lang` | 9 of 11 | 6.0 |
| Earlier hand-written Gemini loop (`agent/runtime/agent_policy.py`) | never run; superseded by the ADK agent | |

So the planner-based route is at 80 % in the sim, the Gemini agent on top of it at 82-83 % (it retries missed
grasps and reads the instruction itself), the neural policy at about 33 %. Every agent run above made real Gemini API
calls; none failed.

The neural policy now runs the full motion (reach, grasp, lift, carry, release, return) and keeps working for the whole
episode. What still fails: grasps miss by a centimetre or two, and red objects tend to land in the centre bin.
12 objects per evaluation is a small sample; differences of one object mean nothing.

## 6. Things we learned the hard way

- **How you cut the demonstrations matters.** One training episode per pick-and-place, each ending at the home
  pose, taught the model to stay at home (this π0.5 implementation also trains on the padded "hold" actions
  after an episode ends). Fixed by joining each episode's successful pick-and-places back to back and cutting out
  all waiting. `sim/to_dataset.py --mode segments` does this.
- **Failed grasps must not be in the data.** The oracle retries a missed triangle up to five times. We cap it at
  two tries per object and keep only attempts that ended with the object in the right bin.
- **π0.5 reads joint angles as text.** In LeRobot's π0.5 the joint positions enter only as numbers in the prompt.
  With the language model frozen (`--mode expert`) that path is weak; `--mode full` trains everything.
- **Training speed on Modal:** H100 19 samples/s; B200 with `--no-grad-ckpt` (action head only) 55 samples/s;
  B200 `--mode full` 22 samples/s (1.46 s per step, 49 GB). The B200 is the one to use.
- **The sim's first camera frames after a reset show the previous scene.** The server now waits 15 ticks.
- **The gate's "policy real-time rate"** is roughly chunk length × 30 × sim speed; return at least 8 ticks per
  call or it fails on a slow machine.
- **`modal volume get` fails if the destination folder does not exist.** `mkdir -p` first.
- **The workspace limit is 10 GPUs.** 8 data sims + training + one evaluation is exactly that; more just queue.
- The real-robot data drop (`zupermiez/g1_pick_cube`, 20 episodes) shows a blue cube going to the LEFT bin,
  which contradicts rung 1, uses different camera names than the docs, and has only 7 clean episodes.

## 7. Running right now (15:00)

- `pi05-sim-v23-full`: whole-model fine-tune on both demonstration batches (`sim_fixed_v23`), 8,000 steps on the
  B200, started 14:48, about 3.3 hours and about $20 if left to finish. Nobody needs it for the agent route; stop
  it from the Modal dashboard (app `aalto-pi05`) if the budget matters. Checkpoints every 1,000 steps are kept.
- Sim evaluations of its checkpoints are started by helper scripts on one laptop.
- Nothing else. The second data batch (`sim_fixed_v3`, 7 × 18 episodes) is finished and merged.

## 8. Open problems and ideas, most useful first

0. **The ADK agent on varied scenes.** In the nominal scene it sorts 82-83 % of the objects, never into a wrong
   bin, gate passed. With the real object colours (yellow, purple, orange), varied shades, lighting and placements
   (`sim/scene_variant.py`) the first runs are much worse and include wrong bins; the cause is not known yet.
   Runs are fetched with `modal volume get aalto-sim-out <tag> sim_out`; every decision is in `policy.log`.
1. **More data and longer training.** Scores only started moving after 4,000–5,000 steps; 125 episodes is small.
2. **Tell the model which object and which bin.** Rung 1's instruction is the same sentence in every episode, so
   it carries no information and the model has to work out "which object next, which bin" from pixels alone. The
   recordings already label every pick-and-place with object, colour, shape, bin and arm, so a dataset with
   per-object prompts ("Pick up the red cube and put it in the left bin") is one converter option away. Something
   then has to choose the prompts at run time: a rule-based planner, or the Gemini agent.
3. **A Gemini agent calling skills as tools** (also a sponsor prize). The oracle's planner can be split into
   reusable skills (detect, pick, place, hand over, go home) that an agent calls; see `agent/AI_STUDIO_PROMPT.md`.
   The same skills, randomised, would give more varied demonstrations.
4. **Triangles.** A third of all objects; the oracle gets 35 %. Either improve its triangle grasp or accept the cap.
5. **Rungs 2 and 3.** No data recorded yet. Rung 2 is one flag (`--eval-task sort_language`). The oracle does not
   do rung 3 at all.
6. **Build and test the Docker image** on a machine with Docker and an NVIDIA GPU.
7. **Real robot.** Everything above is sim only. Sim wrist cameras do not look like the real ones.

## 9. How the oracle works (for whoever generalises it)

Code: `aaltonomous-humanoid/aaltonomous/sim/oracle.py` on top of `baselines/classical/{planner,kinematics,policy}.py`
and `config.yaml`. Videos: `sim/out/oracle_all_cameras_5of5.mp4`, `sim/out/oracle_all_cameras_triangle_failures.mp4`.

- **Look:** asks the sim for every object's exact position, shape and yaw; keeps those on the counter.
- **Choose:** the object closest to its bin first. Arm: the bin's side; for the centre bin, the object's side.
- **Plan one pick-and-place** as six waypoints of the grasp point (pregrasp above the object, grasp, lift, above
  the bin, drop, retreat), solved with Pinocchio inverse kinematics. The grasp is always a claw from above: palm
  facing down within 35°, fingers pointed along a preferred direction, thumb against one finger.
- **Shape handling:** for cubes and triangles it aligns the fingers with a flat face; cylinders need no alignment.
- **Time it** into 30 Hz joint targets: to pregrasp, descend, close, lift, carry, open, retreat, home. About 10 s.
- **Check it** before moving: hand clearance above the counter and bin rims, and a dry run through the safety
  filter. If any check fails it tries the other finger directions, then the other arm, then a hand-over via a spot
  both arms reach.
- **Execute blindly.** There is no feedback during the motion; a missed grasp is only noticed at the next look.

What can be varied by configuration alone: finger direction, allowed tilt, approach and carry heights, speeds and
accelerations, arm preference, where in the bin it drops, hand postures, the grasp point in the palm.
What needs a little code: a different object order (it is one hard-coded sort), jittered waypoints, curved paths.
What needs real new code: side or angled approaches, pushing or sliding, re-grasping, reacting during a motion,
and insertion for rung 3.
