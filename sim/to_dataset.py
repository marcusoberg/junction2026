"""Raw sim demonstrations (sim/gen_demos.py) -> LeRobot v3.0 dataset (lerobot 0.7.0) that trains with
training/modal_pi05.py and merges with the organisers' real-robot data.

    python to_dataset.py RAW_DIR --root OUT_DIR                       # every ep_*/ under RAW_DIR
    python to_dataset.py --merge DS_A DS_B ... --root OUT_DIR         # several converted datasets into one
    python to_dataset.py --verify OUT_DIR

Frame k of an episode = (cameras + joint positions after tick k, the command sent in tick k+1): what the policy sees,
and what it has to output next. `action` is the absolute joint target after the safety filter, as in the real data.

--schema drop1 (default): feature names and robot type exactly as in the organisers' first real drop
    (zupermiez/g1_pick_cube: cam_high / cam_left_wrist / cam_right_wrist), so the two can be merged.
--schema docs: the names their to_lerobot.py writes (head_rgb / left_wrist_rgb / right_wrist_rgb).
--mode segments (default): per recorded episode, its SUCCESSFUL pick-and-places (home pose -> grasp -> bin -> home
    pose, about 10 s each) joined back to back into one dataset episode. A pick-and-place counts as successful when
    it is the oracle's last try at an object that ended in its correct bin. Failed grasps and all waiting at the
    home pose are cut out, so the policy never imitates a miss and, at the home pose with objects left, always sees
    "start the next one". (One dataset episode per pick-and-place taught the opposite: each ended resting at home,
    and lerobot's pi0.5 also trains on the padded hold actions after an episode's end, so "at home -> stay" won;
    measured with training/openloop_check.py.) Kept as well: the relay move of an object that ended up sorted (the oracle hands an object
    across when the arm on its side cannot reach its bin). The instruction stays the episode's full instruction.
--mode episodes: whole episodes as recorded, failures inside included (success label per episode).
--max-idle N: where the command does not change for more than N ticks in a row (the oracle waiting for its
    planner), keep only the first N. Long motionless stretches teach a policy to stand still. 0 keeps everything.
meta/aalto_episodes.jsonl has one row per dataset episode: success, seed, layout, and its pick-and-places with their
first frame, object, colour, shape, bin and arm (training/modal_pi05.py `prepare` drops rows without success).
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

H, W, FPS = 480, 640, 30
CAMS = ("head_rgb", "left_wrist_rgb", "right_wrist_rgb")
SIDECAR = "aalto_episodes.jsonl"
DROP1_JOINTS = (
    [f"k{s}{j}" for s in ("Left", "Right")
     for j in ("ShoulderPitch", "ShoulderRoll", "ShoulderYaw", "Elbow", "WristRoll", "WristPitch", "WristYaw")]
    + [f"kLeftHand{j}" for j in ("Thumb0", "Thumb1", "Thumb2", "Middle0", "Middle1", "Index0", "Index1")]
    + [f"kRightHand{j}" for j in ("Thumb0", "Thumb1", "Thumb2", "Index0", "Index1", "Middle0", "Middle1")])
SIDE_KEYS = ("success", "source", "task", "instruction", "mapping", "seed", "variant", "layout", "outcomes", "score",
             "n_objects", "n_correct", "end_reason", "clipped_steps", "head_repeat_frac", "dr_params")


def schema(name, arm_names, hand_names):
    """-> (robot_type, features, camera -> feature key)"""
    if name == "drop1":
        cam = {"head_rgb": "cam_high", "left_wrist_rgb": "cam_left_wrist", "right_wrist_rgb": "cam_right_wrist"}
        img = {"dtype": "video", "shape": (H, W, 3), "names": ["height", "width", "channel"]}
        vec = {"dtype": "float32", "shape": (28,), "names": [list(DROP1_JOINTS)]}
        feats = {"observation.state": dict(vec), "action": dict(vec)}
        feats.update({f"observation.images.{v}": dict(img) for v in cam.values()})
        return "Unitree_G1_Dex3_Mono", feats, cam
    if name == "docs":
        cam = {c: c for c in CAMS}
        img = {"dtype": "video", "shape": (H, W, 3), "names": ["height", "width", "channels"]}
        feats = {f"observation.images.{v}": dict(img) for v in cam.values()}
        feats["observation.state"] = {"dtype": "float32", "shape": (28,), "names":
                                      [f"arm_q.{n}" for n in arm_names] + [f"hand_q.{n}" for n in hand_names]}
        feats["action"] = {"dtype": "float32", "shape": (28,), "names":
                           [f"arm_q_target.{n}" for n in arm_names] + [f"hand_q_target.{n}" for n in hand_names]}
        return "unitree_g1_29dof_dex3", feats, cam
    raise SystemExit(f"unknown schema {name!r} (drop1 | docs)")


def kept_ticks(cmd, max_idle):
    """Ticks k (0..n-2) to turn into frames: all, minus the tail of each run where the next command is unchanged."""
    moving = np.abs(np.diff(cmd, axis=0)).max(1) > 1e-6          # moving[k]: command k+1 differs from command k
    if max_idle <= 0:
        return np.arange(len(moving))
    keep, idle = [], 0
    for k, m in enumerate(moving):
        idle = 0 if m else idle + 1
        if idle <= max_idle:
            keep.append(k)
    return np.array(keep, int)


def good_segments(meta, n):
    """[(first tick, end tick, segment)] of the successful pick-and-places in one recorded episode."""
    segs = [s for s in meta.get("segments", []) if s.get("tick") is not None]
    correct = {o["name"] for o in meta["outcomes"] if o["outcome"] == "correct"}
    last = {s["object"]: i for i, s in enumerate(segs)}
    out = []
    for i, s in enumerate(segs):
        # the last try at a sorted object, and the relay move before it (an object the arm on its side cannot take
        # to its bin is first moved to the middle; with 2 tries per object, relay + final try both worked)
        if s["object"] in correct and (last[s["object"]] == i or str(s["bin"]).startswith("relay")):
            end = segs[i + 1]["tick"] if i + 1 < len(segs) else min(n - 1, s["tick"] + s["plan_ticks"] + 15)
            if end - s["tick"] >= FPS:
                out.append((s["tick"], min(end, n - 1), s))
    return out


def read_sidecar(root):
    p = Path(root) / "meta" / SIDECAR
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()] if p.exists() else []


def convert(raw, root, repo_id, schema_name="drop1", max_idle=15, only_success=False, limit=0, streaming=True,
            mode="segments"):
    import cv2
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    eps = sorted(p.parent for p in Path(raw).glob("**/ep_*/meta.json") if not p.parent.name.endswith(".partial"))
    eps = [(d, json.loads((d / "meta.json").read_text())) for d in eps]
    if only_success:
        eps = [(d, m) for d, m in eps if m.get("success")]
    if limit:
        eps = eps[:limit]
    if not eps:
        raise SystemExit(f"no recorded episodes under {raw}")
    root = Path(root)
    shutil.rmtree(root, ignore_errors=True)
    robot_type, feats, cam = schema(schema_name, eps[0][1]["arm_names"], eps[0][1]["hand_names"])
    kw = {"streaming_encoding": True} if streaming else {"image_writer_threads": 8}
    ds = LeRobotDataset.create(repo_id, FPS, feats, root=root, robot_type=robot_type, use_videos=True, **kw)
    side = []
    for d, meta in eps:
        low = dict(np.load(d / "lowdim.npz"))
        n = len(low["t"])
        if n < FPS:
            print(f"skip {d.name}: only {n} ticks")
            continue
        state = np.concatenate([low["arm_q"], low["hand_q"]], 1).astype(np.float32)
        cmd = np.concatenate([low["arm_q_cmd"], low["hand_q_cmd"]], 1).astype(np.float32)
        blobs = {c: np.fromfile(d / f"{c}.mjpg", np.uint8) for c in CAMS}
        starts = {c: np.r_[0, low[f"{c}_end"][:-1]] for c in CAMS}
        marks = None
        if mode == "segments":
            keeps, marks = [], []
            for a, b, seg in good_segments(meta, n):
                moving = np.flatnonzero(np.abs(np.diff(cmd[a:b + 1], axis=0)).max(1) > 1e-6)
                if not len(moving):
                    continue
                # only the motion itself: from one tick before the command first changes to one after it last does
                lo, hi = a + max(int(moving[0]) - 1, 0), a + min(int(moving[-1]) + 2, b - a)
                k = lo + kept_ticks(cmd[lo:hi + 1], max_idle)
                marks.append({"frame": int(sum(len(x) for x in keeps)), "frames": int(len(k)), "object": seg["object"],
                              "shape": seg["object"].split("_")[2] if seg["object"].count("_") >= 2 else None,
                              "color": seg["color"], "bin": seg["bin"], "arm": seg["side"],
                              "relay": str(seg["bin"]).startswith("relay"),
                              "skill": seg.get("skill", "pick_and_place"), "args": seg.get("args")})
                keeps.append(k)
            if not keeps:
                print(f"{d.name}: {meta.get('n_correct')}/{meta.get('n_objects')} objects sorted -> nothing kept", flush=True)
                continue
            keep = np.concatenate(keeps)
        else:
            keep = kept_ticks(cmd, max_idle)
        for k in keep:
            fr = {"observation.state": state[k], "action": cmd[k + 1], "task": meta["instruction"]}
            for c in CAMS:
                img = cv2.imdecode(blobs[c][starts[c][k]:low[f"{c}_end"][k]], cv2.IMREAD_COLOR)
                fr[f"observation.images.{cam[c]}"] = np.ascontiguousarray(img[..., ::-1])
            ds.add_frame(fr)
        ds.save_episode()
        row = {"episode_index": ds.meta.total_episodes - 1, "source_dir": d.name, "frames": int(len(keep)), "ticks": n,
               **{k: meta.get(k) for k in SIDE_KEYS}}
        if marks is not None:
            row.update(success=True, episode_success=meta.get("success"), segments=marks)
        side.append(row)
        print(f"{d.name}: {meta.get('n_correct')}/{meta.get('n_objects')} objects sorted in "
              f"{len(meta.get('segments', []))} tries -> "
              + (f"{len(marks)} pick-and-places, {len(keep)} frames of {n} ticks" if marks is not None
                 else f"whole episode, {len(keep)} frames, success={meta.get('success')}"), flush=True)
    if not side:                                # e.g. a shard where the oracle sorted nothing
        print(f"{raw}: nothing to convert in mode {mode}")
        shutil.rmtree(root, ignore_errors=True)
        return None
    ds.finalize()
    with open(root / "meta" / SIDECAR, "w") as f:
        f.writelines(json.dumps(s) + "\n" for s in side)
    ok = sum(bool(s["success"]) for s in side)
    print(f"-> {root}: {len(side)} episodes ({ok} successful), {sum(s['frames'] for s in side)} frames")
    return root


def merge(roots, dst, repo_id):
    roots, dst = [Path(r) for r in roots], Path(dst)
    shutil.rmtree(dst, ignore_errors=True)
    if len(roots) == 1:
        shutil.copytree(roots[0], dst)
    else:
        cmd = [sys.executable, "-m", "lerobot.scripts.lerobot_edit_dataset", "--new_repo_id", repo_id,
               "--new_root", str(dst), "--operation.type", "merge",
               "--operation.repo_ids", json.dumps([f"local/{r.name}" for r in roots]),
               "--operation.roots", json.dumps([str(r) for r in roots])]
        print("$", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)
        side, offset = [], 0                   # merge keeps the input order: number the sidecar rows the same way
        for r in roots:
            side += [{**s, "episode_index": offset + s["episode_index"], "shard": r.name} for s in read_sidecar(r)]
            offset += json.loads((r / "meta" / "info.json").read_text())["total_episodes"]
        with open(dst / "meta" / SIDECAR, "w") as f:
            f.writelines(json.dumps(s) + "\n" for s in side)
    return dst


def verify(root):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    ds = LeRobotDataset("local/verify", root=root)
    side = read_sidecar(root)
    print(f"{root}: {ds.num_episodes} episodes, {ds.num_frames} frames ({ds.num_frames / ds.fps / 60:.1f} min), "
          f"fps {ds.fps}, {sum(bool(s.get('success')) for s in side)} of {len(side)} labelled successful")
    for i in (0, ds.num_frames - 1):
        x = ds[i]
        print(f" frame {i}: " + ", ".join(f"{k.split('.')[-1]} {tuple(v.shape)}" for k, v in x.items()
                                          if hasattr(v, "shape") and v.ndim) + f" | task: {x['task'][:60]}")
    return ds


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("raw", nargs="?", help="folder with ep_*/ episode folders (searched recursively)")
    ap.add_argument("--root", help="output dataset folder")
    ap.add_argument("--repo-id", default="local/sim_oracle")
    ap.add_argument("--schema", default="drop1", choices=("drop1", "docs"))
    ap.add_argument("--mode", default="segments", choices=("segments", "episodes"))
    ap.add_argument("--max-idle", type=int, default=15)
    ap.add_argument("--only-success", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-streaming", action="store_true", help="encode videos from PNG files instead of on the fly")
    ap.add_argument("--merge", nargs="+", metavar="DATASET")
    ap.add_argument("--verify", metavar="DATASET")
    a = ap.parse_args(argv)
    if a.verify:
        verify(a.verify)
    elif a.merge:
        verify(merge(a.merge, a.root, a.repo_id))
    else:
        root = convert(a.raw, a.root, a.repo_id, a.schema, a.max_idle, a.only_success, a.limit, not a.no_streaming,
                       a.mode)
        if root:
            verify(root)


if __name__ == "__main__":
    main()
