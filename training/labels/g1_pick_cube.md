# Episode labels: `zupermiez/g1_pick_cube` (data drop 1, 20 episodes)

The drop ships no success labels. These are ours, made on 2026-11-14 from 5–8 head-camera frames per episode plus
the joint data — **not from watching the full videos**, so treat "clean" as "looks clean", and re-check before relying
on it. What the demos show: the **left hand picks a blue object and drops it in the LEFT bin**. Task text in the
dataset is `pick up cube.` for every episode.

| Ep | Length | Label | What we saw |
|---:|---:|---|---|
| 0 | 36.6 s | messy | Cube goes into the left bin, then a person's hand enters the view and the cube is back on the table at the end |
| 1 | 15.3 s | messy tail | Cube ends up on the bin rim / back on the table in the last frames |
| 2 | 14.9 s | **clean** | Blue cube → left bin |
| 3 | 16.0 s | messy tail | Cube in the left bin, then the hand goes back into the bin and grabs it again |
| 4 | 20.5 s | **clean** | Blue cube → left bin |
| 5 | 17.1 s | **clean** | Blue cube → left bin |
| 6 | 15.6 s | **clean** | Blue cube → left bin |
| 7 | 11.9 s | **clean** | Blue cube → left bin |
| 8 | 12.6 s | **clean** | Blue cube → left bin |
| 9 | 19.2 s | **clean** | Blue cube → left bin |
| 10 | 25.2 s | fail | Cube never grasped; head camera pose differs from episodes 0–9 |
| 11 | 22.0 s | fail | Cube lifted, put down on the table in front of the bins |
| 12 | 15.9 s | fail | Cube pushed around, still on the table at the end |
| 13 | 31.7 s | fail + glitch | Command jumps up to 27.8 rad/s; arm swings up, nothing sorted |
| 14 | 16.3 s | wrong object | Blue **cylinder** → left bin (cube stays) |
| 15 | 19.7 s | fail + glitch | No grasp; command jump 7.1 rad/s |
| 16 | 23.7 s | fail + glitch | Objects knocked about; command jump 20.8 rad/s |
| 17 | 15.4 s | wrong object | Blue **cylinder** → left bin (cube stays) |
| 18 | 31.5 s | fail | Cylinder pushed over, nothing picked |
| 19 | 9.5 s | junk | No motion at all |

Training copy used so far: `--keep 2,4,5,6,7,8,9` (7 episodes, 3,353 frames, 1.9 minutes) with the instruction
rewritten to `Pick up the blue cube and put it in the left bin.`

Things that apply to every episode:
- Arm commands exceed the 1 rad/s safety cap on 11–229 ticks per moving episode (peaks 2.6–3.8 rad/s in the kept ones).
- The start pose differs from episode to episode; there is no fixed home pose.
- The right arm barely moves; its wrist camera mostly shows the venue and people.
- Finger commands leave the URDF joint range on about 10 % of frames (left index).
