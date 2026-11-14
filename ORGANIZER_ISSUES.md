# Aaltonomous Humanoid — discrepancies and open questions for the organisers

Checked against `zupermiez/aaltonomous-humanoid` @ `a2d1dee` (2026-11-13, `aaltonomous/VERSION` = `dc41d88 +dirty`)
and the Junction challenge page. Everything below was found by reading the repo and running its offline tools
(unit tests, `scene.check_layout`, the classical baseline in the mock world, Pinocchio IK on the pinned URDF) on a
laptop. **Nothing here was run in Isaac Sim or on the robot** — we have not had the sim image yet.

Severity: **A** = can change scores or break a build, **B** = sim/real mismatch we need a number for, **C** = stale docs.

## A. Things that look wrong

### A1. The repo's own unit test for head-camera visibility fails (and that test is part of the image build)
- `python -m unittest aaltonomous/scene/test_scene.py` → `TestHeadCameraView.test_cubes_and_bins_visible` **FAILS**.
- Cause: `interface/calibration.yaml` was switched to the factory intrinsics on 2026-11-12
  (fx 604.8, fy 604.3, cx 325.0, **cy 250.9**). With them, an object at the near edge of the layout band
  (mat y = 0.12, pelvis x = 0.26) has its centre at image row v ≈ 440 of 480 and its near-bottom edge at v ≈ 480.2,
  i.e. on/over the bottom edge of the head image.
- Consequences:
  - `setup_instance.sh verify` (the last `RUN` of `cloud/Dockerfile`, and `aalto test`) runs this test, so a fresh
    image build or bare-metal setup from the current repo should stop with a test failure. Was `aalto-sim:v1` built
    from an older calibration? Which calibration does the image actually render with?
  - With the training DR camera jitter (±1 cm, ±1° ≈ ±10 px) or any real head-pose error, objects in the near part of
    the band are partly cut off in `head_rgb`. The head extrinsics are still "URDF, not yet checked with the AprilTag".
- Ask: move the layout band ~2–3 cm away from the robot (or confirm the real head pitch), and re-run the test.

### A2. Only the near half of each bin is reachable, and each side bin only by the arm on its side
- `scene.yaml` itself says the side-bin centres are at r = 0.57 m, past the ~0.55 m straight-arm reach.
- `python -m aaltonomous.scene.check_layout` prints **"bins: 0% of cells seen + reachable"** (its criterion: palm
  frame 8 cm above the cell), and `TestReach.test_cube_region_and_bins_reachable` is marked
  `expectedFailure  # KNOWN 2026-11-05: default counter is out of reach`. The tool and the test still describe the
  scene as unreachable.
- Our own check with the baseline's IK (Pinocchio, pinned URDF, joint limits − 0.08 rad, grasp point = the baseline's
  `grasp.point_in_palm`, release height z = 0.133 m, no collision checking), 3 cm × 4 cm grid:
  - the grasp point reaches at most **x ≈ 0.49–0.52 m**; the bins span x = 0.40–0.61 m, so only the **near ~50 %** of
    each bin's usable inner area can be reached and the far half cannot be reached at all (allowing any hand
    orientation does not help: it is arm length, not wrist tilt);
  - left bin: left arm 50 %, **right arm 0 %**; right bin: right arm 50 %, **left arm 0 %**; centre bin: either arm ~46 %.
- The classical baseline works by releasing 5 cm inside the nearest inner corner (grasp point at pelvis (0.45, 0.00) /
  (0.45, ±0.21), z = 0.13). In the kinematic mock world that is feasible for every layout we tried (seeds 0–19: all
  objects "correct", 0 clips) — but the mock has no collisions, no self-collision and no grasp physics.
- Consequences: (1) every object whose colour maps to the bin on the other side of the body needs a cross-body pick or
  a hand-over (the oracle has a "relay" for this; rung 2 mappings make it common); (2) releases happen right at the near
  wall, and nothing in the safety filter covers bin walls (only a counter keep-out up to z = 0.03) or self-collision;
  (3) at the home pose both hands sit in the head image in front of the bin row.
- Ask: are the real boxes really 40 cm from the waist joint, and how tall are they? Did the teleop operators manage
  cross-body sorts? If not, consider moving the row 5–8 cm closer; either way please update `check_layout` / `TestReach`
  so they describe the scene that is actually used.

### A3. Pelvis-to-table height is an assumption, and docs disagree on the robot's posture
- `scene.yaml`: table surface = pelvis + 0.01 m, from "table 0.80 m, pelvis assumed 0.79 m (G1 standing)", marked *(default)*.
- Challenge page: the robot **stands** behind a counter. `data/DATA_CARD.md`: "**seated** behind an 80 cm table".
  `data/RECORDING.md`: "seated or on the gantry". Baseline README: "robot on the gantry".
- A few cm of height error changes reach (A2), the head view (A1), the counter keep-out (z < 0.03) and every sim image.
- Ask: one measured number — table surface height relative to the pelvis/waist joint in the finals setup.

### A4. Wrist cameras in the sim are placeholders
- `calibration.yaml`: wrist `T_link_optical: null`, intrinsics fx = fy = 400 "(default) unknown lens, resized".
- `sim/aalto_task/scene_cfg.py` uses the upstream `CameraPresets.*_dex3_wrist_camera()` presets, not the real mounts.
- So 2 of the 3 policy cameras do not match the robot in the sim. Sim-generated data is of limited use for any policy
  that looks at the wrist images until this lands. DR pixel noise is also applied to the head RGB-D stream only.
- Ask: wrist camera mount pose + FOV/intrinsics (even rough), and an ETA for the sim update.

### A5. Start pose is not pinned down
- `INTERFACE.md`: sim home = `ARM_HOME` and "the real robot uses the same home pose".
- `sim/aalto_task/layout.py`: `ARM_HOME` is "(default) — replace with the real robot's home pose once defined".
- `baselines/classical/config.yaml`: the robot-side default `home_arm_q` is a different pose (arms out to the sides,
  hands out of the head view); the sim profile overrides it with `ARM_HOME`, where both hands are inside the head image.
- `RECORDING.md`: demos start wherever the operator "homes" the arms with teleop.
- Ask: which arm/hand pose does the runner put the robot in before `reset` in the finals, and do the demos start there?

### A6. Sim timing: oracle/baseline parameters were tuned against the wrong clock
- `sim/README.md` "Still open": "Earlier tuning ran ~1.8x too fast in physics time (wall pacing at RTF ~0.55);
  re-check grasp parameters on the sim clock", oracle ~84 % on rung 1.
- The sim head camera runs at ~17 fps (`sim/README.md`) / 28.6 Hz render, while the policy loop is 30 Hz, so `obs`
  contains repeated head frames in sim. What is the real head RGB-D rate through `head_rgbd_server`?

### A7. The repo cannot be installed from scratch: the constraints file conflicts with Isaac Sim's own pins
- Found 2026-11-14 building the sim image from the repo (the Discord advice: "no need to wait for the docker image,
  you can build everything based on the repo").
- `bash aaltonomous/cloud/setup_instance.sh isaacsim rest` stops with pip `ResolutionImpossible`:
  `isaacsim-kernel 6.2.0.0 depends on click==8.3.3`, but `cloud/isaac-constraints.txt` pins `click==8.5.1`.
  The file is a `pip freeze` of a working machine, not a set pip can resolve on a fresh install.
- Affects both paths in the Quickstart: the bare-metal `setup_instance.sh` and `cloud/Dockerfile` (which runs it).
- Our workaround: drop the conflicting constraint line and retry. A full build needed six lines dropped:
  `click==8.5.1`, `urllib3==2.8.0`, `psutil==7.2.2`, `llvmlite==0.50.0`, `fsspec==2026.9.0` (all at `isaacsim rest`)
  and `rerun-sdk==0.23.4` (at `isaaclab`). With those gone the sim boots and `aalto check` passes (9 PASS, 0 FAIL;
  the Quickstart says 8).
- Ask: publish the sim image link, or regenerate the constraints so a clean install works.

### A8. The gate's "policy real-time rate" measures the sim's speed, not the policy's
- `deploy/runner.py` takes a call's latency as the time from sending the request to the next loop iteration that
  finds the reply. For a chunking policy the request goes out while a queued tick executes, so the measured latency
  is never less than one tick of wall time, i.e. 1 / (30 x real-time factor), however fast the server answers.
- So `policy_hz_rt` comes out as roughly (chunk length) x 30 x RTF. Measured on Modal (L40S, RTF 0.30): our server
  answers in a few ms and returned 4 ticks per call, and scored 33.7-38.3 Hz; the oracle (10-tick chunks) 88-95 Hz.
- Consequence: a fast policy with short chunks fails the 27 Hz gate whenever the evaluation machine is slow or
  shared (4-tick chunks fail below RTF 0.225), which is the opposite of what the check is meant to catch.
- Ask: measure the server's reply time directly (timestamp when the reply arrives), or tell teams the RTF you
  evaluate at.

### A9. The oracle almost never sorts triangular prisms, and `aalto baseline` does not show it
- Measured on Modal (L40S, RTF 0.25-0.30), `sort_fixed`, oracle from `sim/evaluate.py` unchanged:
  - seeds 1000-1002 (what `aalto baseline` runs): 11 of 12 objects, but these three layouts contain no triangle.
  - seeds 100000, 100001, 101000, 101001: cubes and cylinders 8 of 8, triangles 2 of 9.
- A missed triangle is pushed a little by the failed grasp. The retry limit (`policy.max_attempts: 2`) counts tries
  per table position (`same_object_radius`), so the moved object counts as new: the oracle tried the same triangle 5
  times and spent the rest of the 120 s on it (episode seed 101000: 1 of 5 objects sorted, 12 grasp attempts).
- Consequence for teams: a third of all objects are triangles (`sample_layout` draws shapes uniformly), so the
  oracle's real per-object rate is well below the 92 % the baseline command prints, and demonstrations recorded
  from it contain mostly failed triangle grasps.
- Ask: is the triangle grasp expected to work in the sim (pad friction, `_tri_shift`)? Will the held-out sim
  variant and the finals contain triangles at all (data drop 1 has none)? And count retries per object name.

### A10. The sim head camera gives a new image on only about 83 % of control ticks
- Recording every tick of 4 oracle episodes (sim clock, 30 Hz), 17 % of ticks returned exactly the same head image
  as the tick before (16.8-17.1 % in every episode), so the effective head-camera rate is about 25 fps of sim time.
- Ask: is that intended (it matches no real-camera rate), and does the held-out evaluation run the same way?

### A11. The first camera frames of a sim episode still show the previous episode's objects
- In the runner's `--video` of a sim episode, frames 0 and 1 show the layout of the episode before; the new layout
  appears from frame 2 (Modal, RTF 0.3). The scene reset returns before the cameras have rendered the new scene,
  so `env.reset()` and the first `step()` hand the policy a stale image.
- In `sim/evaluate.py` the objects are parked first (`home_arms`), so there the first observations show an EMPTY
  counter. A policy that plans from its first observation acts on the wrong scene for its whole first chunk; a
  policy that sends `done` when it sees nothing to sort would end the episode at once.
- Ask: wait for a fresh frame from every camera before `reset()` returns. (We now hold for 15 ticks ourselves.)

### A12. Nothing prevents the two arms from hitting each other, and the oracle's own moves do

- Seen in the Isaac sim, run `adk-1`, `episode_000.mp4`, t = 19-20 s: the right arm carries an object to the centre bin
  and its wrist camera mount hits the left hand's thumb, which is resting at the home pose.
- Numbers (forward kinematics on the recorded joint angles, hand / finger / wrist / elbow link origins): at the task's
  home pose (`layout.ARM_HOME`) the two hands are 13 cm apart in front of the robot. During that move the right wrist
  comes within 9.8 cm of a left-hand link origin; during the next one (blue object to the right bin) 6.8 cm. The links
  and the camera mount have width, so these are contacts or near-contacts.
- The trajectories are the oracle planner's, unchanged. `interface/safety.py` says self-collision (arm vs arm) is "not
  covered (yet)", and the planner checks only the moving hand against the counter and the bin rims.
- The home pose also puts both hands inside the head camera's view, partly hiding objects in the middle of the counter.
- What we do about it: between moves we park both arms out to the sides (hand at x 0.27, y +-0.33, 0.22 m above the
  counter): about 30 cm between the arms during the same motions, and the hands are outside the head image. We also
  refuse a planned move that brings the arms within 12 cm of each other.
- Question: will the real-robot episodes start from `ARM_HOME` as well, and is a submission expected to avoid arm-arm
  contact by itself?

## B. Numbers marked "(default)" that we need measured

| Item | Value in repo | Where |
|---|---|---|
| Bin wall height | 0.06 m | `scene.yaml bins.wall_height` (page only gives the 21 cm width) |
| Bin wall / floor thickness | 3 mm | `scene.yaml` |
| Object size | 45 mm (`DATA_CARD.md` says "45 to 50 mm") | `scene.yaml objects.size` |
| Lid hole clearance (rung 3) | +10 mm | `scene.yaml lid.hole_clearance` |
| Head camera extrinsics | URDF `d435_link`, unverified | `calibration.yaml` |
| Dex3 friction / PD gains, object mass (50 g) and friction (1.2 / 1.0) | guesses, "not yet fitted" | `scene_cfg.py`, `STARTER_PATHS.md` |
| DR ranges | "guesses until the venue capture" | `sim/dr/train.yaml` |
| Bin colour | RGB (230, 110, 30) "take from a venue photo" | `scene.yaml` |

DR also does not move the bins, the table height or the robot gains — only lights, colours, object mass/friction,
±5 mm object jitter and head-camera pose/noise.

## C. Stale or inconsistent docs

1. `INTERFACE.md` §6 says head intrinsics are fx = fy = 615, cx 320, cy 240 and "Sim uses exactly these values";
   `calibration.yaml` (what the sim loads) has 604.8 / 604.3 / 325.0 / 250.9.
2. `SUBMISSION.md` §7 says the baseline image scores **0/24** in Isaac ("grasps slip"); `FAQ.md` and the baseline
   README say **76–82 %** after a same-day fix. Which is current for image v1?
3. `interface/tasks.py::sample_layout` defaults (`n_max=6`, region x 0.20–0.60 / y 0.10–0.22) claim to "mirror
   scene.yaml" but don't (3–5 objects, x 0.28–0.62, y 0.12–0.19). All real callers pass `Scene.layout_kwargs()`, so
   it only bites someone calling it directly.
4. `scene.yaml` comment says the band is pelvis x 0.26–0.35; the numbers give 0.26–0.33 (as `INTERFACE.md` says).
5. URDF revision: `joint_spec.yaml` / `INTERFACE.md` say `xr_teleoperate@76a63c4`; `setup_instance.sh` pins `817fb00`.
6. `data/to_groot.py` and its generated `new_embodiment_config.py` target Isaac-GR00T **N1.5/N1.6**; GR00T `main` is
   now **N1.7** (new model package and data interface). Say which branch the GR00T copy is meant for (`n1d6`?).
7. openpi (π0 / π0.5) still pins LeRobot at a May-2025 commit that reads only **v2.1** datasets, so the LeRobot
   **v3.0** drop does not load there; the `_groot` (v2.1) copy is the one openpi users need. Worth a line in `STARTER_PATHS.md`.
8. Referenced but not in the public repo: `docs/ISAAC_TWINFLOW.md`, `sim/test_sim_offline.py`,
   `cloud/make_bundle.sh`, `scripts/teleop_safe.py`, `scripts/record.sh`, `sim/dr/eval_heldout.yaml` (the last one on purpose).
9. `aalto eval` defaults to `--seed 1000`, and the real rung-1 demo cards are seeds 1000+ made by the same sampler,
   so the default local eval layouts are the demo layouts. Fine, but worth knowing (and saying).
10. `deploy/test_deploy.py::test_stuck_policy_latency_cap` is timing-sensitive (took 11.1 s against an 8 s bound on
    our laptop); `test_end_to_end_hold` flaked once under load.

## D. Questions about the finals deployment (Wendy)

You mentioned the container will be deployed with Wendy. The docs say linux/amd64, CUDA ≥ 12.8, one RTX 5090, port
8000, no network. Wendy's tooling is aimed at Jetson/ARM edge devices and runs apps through containerd, so please confirm:
- target architecture is still **x86_64** (not the robot's Jetson);
- the GPU is passed through and the image is started exactly as `docker run --gpus all -p <port>:8000 <image>` would;
- limits are unchanged: ≤ 40 GB image, ≤ 32 GB VRAM, listening within 120 s, weights inside the image.
- which NVIDIA driver the finals PC runs: PyTorch's default PyPI wheel (2.15) now bundles CUDA 13 and needs a 580+
  driver, while the docs only say "CUDA ≥ 12.8" (we build against the cu128 wheels to be safe).

## E. Data drop 1 (`zupermiez/g1_pick_cube`, posted 2026-11-13 23:44 UTC) vs the docs

Checked 2026-11-14 from the dataset's metadata, joint data and sampled head/wrist frames (not full video review).

**Format differs from `data/DATA_CARD.md` and `INTERFACE.md`** (the brief says "the obs/action keys and the data format will not change"):
- cameras are `observation.images.cam_high` / `cam_left_wrist` / `cam_right_wrist`, not `head_rgb` / `left_wrist_rgb` / `right_wrist_rgb`;
- no `observation.images.head_depth`, no `observation.arm_dq`;
- no `meta/aalto_episodes.jsonl` (success label, layout, instruction, card seed, filter check) and no GR00T copy;
- one task string for all 20 episodes: `pick up cube.` — not a challenge instruction;
- `robot_type` is `Unitree_G1_Dex3_Mono`, not `unitree_g1_29dof_dex3`: recorded with a different pipeline than `record_demos.py`?
  If so, is `action` still the commanded joint target, resampled the same way?
- Joint order does match `joint_spec.yaml` (left arm, right arm, left hand thumb/middle/index, right hand thumb/index/middle).

**The demonstrated behaviour contradicts rung 1:** the left hand puts a **blue** object into the **left** bin. Rung 1 says
blue → right (and scores −1 for a wrong bin). Is this drop only a format sample, or is the mapping different from the docs?

**The real scene does not look like `scene.yaml` / the sim:**
- the "red" objects look **orange** in the head camera, close to the bin colour (sim red is RGB 200, 30, 30);
- 6 objects on the table (rules: 3–5), touching or a few cm apart (rules: ≥ 8 cm between centres), only cubes and
  cylinders, no triangular prisms;
- no printed mat / 5 cm grid visible, the surface is a plain pinkish board;
- the bins are orange outside with brown cardboard inside and look clearly taller than the 6 cm in `scene.yaml`;
- the head camera view changes between episodes (0–9, 10–16 and 17–19 are three different views), so the head/robot
  pose is not fixed;
- the wrist cameras look along the fingers with a wide lens and mostly show the hand, the bins and the venue/people —
  nothing like the upstream presets the sim uses.

**The demos do not respect the safety limits they are supposed to stay inside** (`RECORDING.md`: teleop capped at 0.9 rad/s):
- arm commands exceed 1 rad/s on 11–229 ticks in every episode that moves; peaks of 20.8 and 27.8 rad/s in episodes 16 and 13;
- finger commands leave the URDF range on ~10 % of frames (left index 0/1);
- the start pose differs per episode — there is no fixed home pose (see A5).

**Content:** episode 19 has no motion; by our reading only 7 of 20 are clean "blue cube → left bin" demos, 2 move the
blue cylinder instead, and in episode 0 a person reaches into the scene and resets the cube mid-recording.
