"""What the organisers' real-robot data says about the real setup, against the robot model and the sim.

    # once: the data (drop 3 has head depth and tactile; drops 1 and 2 only joints and colour video)
    D=data_drops/drop3; B=https://huggingface.co/datasets/zupermiez/g1_junction_drop_3/resolve/main
    for f in data/chunk-000/file-000.parquet meta/episodes/chunk-000/file-000.parquet meta/info.json \\
             videos/observation.images.cam_high/chunk-000/file-000.mp4 \\
             videos/observation.images.head_depth/chunk-000/file-000.mp4 \\
             videos/observation.images.head_depth/chunk-000/file-001.mp4; do
        mkdir -p $D/$(dirname $f); curl -sL $B/$f -o $D/$f; done
    # then (needs pyarrow, pandas, av in the environment):
    PYTHONPATH=aaltonomous-humanoid:adk_agents:. .venv/bin/python sim/real_data_check.py data_drops/drop3 [sim_out/<run> ...]

It trusts the robot model (URDF, meshes) and the recorded joint angles, and measures:
  1. the head camera's pose relative to the arms (robot_sorter/selfcal.py, here with the hands: the real hands carry no
     bracket the URDF lacks), against the calibration file;
  2. the table plane in the robot's frame with that camera pose: height and tilt, against scene.yaml;
  3. the same tilt from the joint data alone: how high the hand is when it closes on an object, by distance ahead;
  4. how the joints follow their commands (lag, error), against the sim runs given as further arguments;
  5. the start pose of every episode, against the sim's home pose.
"""
import glob
import os
import re
import sys

import numpy as np
import pandas as pd

from robot_sorter import selfcal
from robot_sorter.robot import get_robot


class Drop:
    """A LeRobot v3 data drop: joint state / action per frame, head colour and depth frames by (episode, frame)."""

    def __init__(self, root):
        self.root = root
        self.d = pd.read_parquet(os.path.join(root, "data/chunk-000/file-000.parquet"))
        self.ep = pd.read_parquet(os.path.join(root, "meta/episodes/chunk-000/file-000.parquet"))
        self.state = np.stack(self.d["observation.state"].to_numpy()).astype(float)
        self.action = np.stack(self.d["action"].to_numpy()).astype(float)
        self.epi, self.fi = self.d["episode_index"].to_numpy(), self.d["frame_index"].to_numpy()

    def rows(self, e):
        return np.flatnonzero(self.epi == e)

    def frames(self, key, e, idx, fmt):
        import av
        r = self.ep[self.ep["episode_index"] == e].iloc[0]
        f = os.path.join(self.root, f"videos/{key}/chunk-{int(r[f'videos/{key}/chunk_index']):03d}/"
                                    f"file-{int(r[f'videos/{key}/file_index']):03d}.mp4")
        first = int(round(float(r[f"videos/{key}/from_timestamp"]) * 30))
        want, out = {first + int(i): int(i) for i in idx}, {}
        c = av.open(f)
        s = c.streams.video[0]
        tb = float(s.time_base)
        c.seek(int(max(0, (min(want) - 5) / 30) / tb), stream=s, any_frame=False, backward=True)
        for fr in c.decode(s):
            n = int(round(float(fr.pts * tb) * 30))
            if n in want:
                out[want[n]] = fr.to_ndarray(format=fmt)
            if n >= max(want):
                break
        c.close()
        return out

    def depth(self, e, idx):                 # uint16 millimetres (lossless 12-bit video)
        return {k: v.astype(np.uint16) for k, v in self.frames("observation.images.head_depth", e, idx, "gray12le").items()}


def angles(T, T0):
    R = T0[:3, :3].T @ T[:3, :3]
    return (np.degrees(np.arctan2(R[2, 1], R[2, 2])), np.degrees(np.arcsin(np.clip(-R[2, 0], -1, 1))),
            np.degrees(np.arctan2(R[1, 0], R[0, 0])))


def lag(cmd, meas, most=20):
    """Per joint: the delay (ticks) at which the measured angle matches the command best, the rms error there."""
    out = []
    for j in range(cmd.shape[1]):
        c, m = cmd[:, j], meas[:, j]
        if np.ptp(c) < 0.15:
            out.append((np.nan, np.nan))
            continue
        errs = [np.sqrt(np.mean((m[k:] - c[:len(c) - k]) ** 2)) for k in range(most)]
        out.append((int(np.argmin(errs)), min(errs)))
    return np.array(out)


def main(root, sim_runs):
    robot = get_robot()
    robot.log = lambda m: None
    ctx = robot.ctx
    kin = ctx.kin
    cam = ctx.cal.cams[ctx.camera.cam]
    K, T0 = np.asarray(cam.K, float), np.asarray(robot.head_T0, float)
    D = Drop(root)
    episodes = sorted(np.unique(D.epi))
    print(f"{root}: {len(episodes)} episodes, {len(D.state)} frames")

    # ---- 1. camera against the arms
    if os.path.isdir(os.path.join(root, "videos/observation.images.head_depth")):
        picks = []
        for e in episodes:                     # up to three moments per episode with both arms still, in different poses
            q = D.state[D.rows(e)][:, :14]
            sp = np.array([np.abs(q[min(len(q) - 1, i + 5)] - q[max(0, i - 5)]).max() for i in range(len(q))])
            kept = []
            for i in np.argsort(sp):
                if sp[i] < 0.02 and all(np.abs(q[i] - q[j]).max() > 0.25 for j in kept):
                    kept.append(int(i))
                if len(kept) == 3:
                    break
            picks += [(e, i) for i in kept]
        frames, depths = [], []
        for e in sorted({p[0] for p in picks}):
            idx = sorted(i for ee, i in picks if ee == e)
            dep = D.depth(e, idx)
            for i in idx:
                st = D.state[D.rows(e)[i]]
                frames.append((selfcal.arm_points(dep[i], K, cam.depth_scale), st[:14], st[14:]))
                depths.append(dep[i])
        model = selfcal.ArmModel(kin, lock=ctx.lock, hands=True)

        def fit(sub):
            best = None
            for Ti in selfcal.starts(T0):
                f = selfcal.camera_from_arms(sub, model, Ti)
                if f.ok and selfcal.mount_ok(f.T, T0) and (best is None or f.matched - f.rms_m / 0.01 > best.matched - best.rms_m / 0.01):
                    best = f
            return best
        full = fit(frames)
        print(f"\n1. head camera against the arms: {len(frames)} still moments, {full.pairs} depth points, "
              f"{100 * full.matched:.0f} % on the arm model, rms {full.rms_m * 1000:.1f} mm")
        p, y, r = angles(full.T, T0)
        print(f"   camera vs calibration file: pitch {p:+.1f}, pan {y:+.1f}, roll {r:+.1f} deg; moved "
              f"{np.round((full.T[:3, 3] - T0[:3, 3]) * 1000).astype(int).tolist()} mm")
        parts = [fit(frames[k::4]) for k in range(4)]
        A = np.array([[*angles(f.T, T0), *((f.T[:3, 3] - T0[:3, 3]) * 1000)] for f in parts if f is not None])
        print(f"   four separate quarters of the moments: std pitch {A[:, 0].std():.1f}, pan {A[:, 1].std():.1f}, roll "
              f"{A[:, 2].std():.1f} deg; position {np.round(A[:, 3:].std(0), 1).tolist()} mm")

        # ---- 2. the table in the robot's frame
        rows = []
        for dep in depths[::2]:
            pl = selfcal.table_plane(selfcal.optical_points(dep, K, cam.depth_scale))
            if pl is None:
                continue
            for k, T in enumerate((T0, full.T)):
                n = T[:3, :3] @ pl[0]
                d = pl[1] - n @ T[:3, 3]
                rows.append((k, np.degrees(np.arcsin(n[0])), np.degrees(np.arcsin(n[1])),
                             *[-(d + n[0] * x) / n[2] for x in (0.20, 0.35, 0.50)], pl[1]))
        R = np.array(rows)
        print(f"\n2. the table in the robot's frame (scene.yaml: level, {ctx.scene.counter_z * 1000:+.0f} mm), {len(R) // 2} frames")
        for k, name in ((0, "calibration file's camera pose"), (1, "camera pose measured from the arms")):
            B = R[R[:, 0] == k]
            print(f"   {name:36s}: rises {-B[:, 1].mean():.1f} deg away from the robot (std {B[:, 1].std():.1f}), "
                  f"{-B[:, 2].mean():+.1f} deg to the left; height at 0.20 / 0.35 / 0.50 m ahead "
                  f"{B[:, 3].mean() * 1000:+.0f} / {B[:, 4].mean() * 1000:+.0f} / {B[:, 5].mean() * 1000:+.0f} mm")
        print(f"   camera above the table plane (needs no pose): {R[:, 6].mean() * 1000:.0f} mm; "
              f"in the sim's nominal scene {(T0[2, 3] - ctx.scene.counter_z) * 1000:.0f} mm")

    # ---- 3. the same tilt from the joints alone
    ev = []
    for e in episodes:
        q = D.state[D.rows(e)]
        for k, side in enumerate(("left", "right")):
            hand = q[:, 14 + 7 * k: 21 + 7 * k]
            clos = np.abs(hand - np.median(hand[:20], axis=0)).sum(1)
            if clos.max() < 0.8:
                continue
            i = int(np.argmax(clos > 0.5 * clos.max()))            # the hand closes for the first time: the pick
            with ctx.lock:
                zs = np.array([kin.grasp_pose(side, qq[:14])[0] for qq in q[max(0, i - 20): i + 10]])
            ev.append(zs[int(np.argmin(zs[:, 2]))])
    E = np.array(ev)
    E = E[(E[:, 2] < 0.15) & (E[:, 0] < 0.42)]
    if len(E) >= 8:
        coef = np.linalg.lstsq(np.c_[E[:, 0], E[:, 1], np.ones(len(E))], E[:, 2], rcond=None)[0]
        print(f"\n3. joints only: hand height when it closes on an object, {len(E)} picks, x {E[:, 0].min():.2f}-{E[:, 0].max():.2f} m: "
              f"rises {np.degrees(np.arctan(coef[0])):+.1f} deg away from the robot, {np.degrees(np.arctan(coef[1])):+.1f} deg "
              f"to the left; {1000 * (coef[0] * 0.30 + coef[2]):+.0f} mm at 0.30 m ahead (level table at the file's height: "
              f"about {(ctx.scene.counter_z + 0.0225) * 1000:+.0f} mm, slope 0)")

    # ---- 4. how the joints follow their commands
    def report(tag, eps):
        L = np.array([lag(c, m) for c, m in eps])
        print(f"   {tag:28s}: arms lag {np.nanmedian(L[:, :14, 0]):.1f} ticks ({33.3 * np.nanmedian(L[:, :14, 0]):.0f} ms), "
              f"rms error {np.nanmedian(L[:, :14, 1]):.3f} rad; hands lag {np.nanmedian(L[:, 14:, 0]):.1f} ticks, "
              f"rms error {np.nanmedian(L[:, 14:, 1]):.3f} rad")
    print("\n4. measured joint angle against the command")
    report("real robot (this drop)", [(D.action[D.rows(e)], D.state[D.rows(e)]) for e in episodes])
    sim = []
    for run in sim_runs:
        for f in sorted(glob.glob(os.path.join(run, "**", "episode_*.npz"), recursive=True)):
            z = np.load(f)
            sim.append((np.c_[z["arm_q_cmd"], z["hand_q_cmd"]], np.c_[z["arm_q"], z["hand_q"]]))
    if sim:
        report(f"Isaac sim ({len(sim)} episodes)", sim)

    # ---- 5. the start pose
    first = D.state[D.fi == 0][:, :14]
    with ctx.lock:
        palms = np.array([[np.asarray(kin.data.oMf[kin.palm[s]].translation).copy() for s in ("left", "right")]
                          for q in first if kin._fk(q) is not None])
        kin._fk(np.asarray(ctx.home, float))
        home = [np.round(np.asarray(kin.data.oMf[kin.palm[s]].translation), 3).tolist() for s in ("left", "right")]
    print(f"\n5. start pose of {len(first)} episodes")
    print(f"   arm joints, mean: {np.round(first.mean(0), 2).tolist()}")
    print(f"   arm joints, std : {np.round(first.std(0), 2).tolist()}")
    print(f"   sim home pose   : {np.round(np.asarray(ctx.home, float), 2).tolist()}")
    print(f"   palms at start (x, y, z m): left {np.round(palms[:, 0].mean(0), 3).tolist()}, right "
          f"{np.round(palms[:, 1].mean(0), 3).tolist()} (std {np.round(palms.std(0).max() * 1000)} mm); sim home: {home}")

    # ---- 6. what calibrate would do on these frames, and whether the arms would then go to the right place
    if os.path.isdir(os.path.join(root, "videos/observation.images.head_depth")):
        replay_calibrate(robot, D, episodes)


def replay_calibrate(robot, D, episodes):
    """For every episode: the agent's own code on the real frames, without and with calibrate. calibrate gets only
    the first frame (what the robot has before it moves). The result is judged on two later still moments of the
    same episode: how far is the arm the camera sees from where the joint angles put it, in the frame and with the
    camera pose the agent would be working with? That distance is what a hand misses by when it reaches for
    something the camera sees. (Nearest-surface distance: a hand shifted along its own forearm still lies close to
    the forearm, so it understates the shift.)"""
    from scipy.spatial import cKDTree
    from robot_sorter import tools
    ctx = robot.ctx
    cam = ctx.cal.cams[ctx.camera.cam]
    K, file_T = np.asarray(cam.K, float), np.asarray(robot.head_T0, float).copy()
    model = selfcal.ArmModel(ctx.kin, lock=ctx.lock, hands=True)

    def fresh():
        robot.set_waist(0.0, 0.0)
        ctx.scene.cfg["counter"]["mat_origin_in_pelvis"] = [float(v) for v in robot.scene_origin0]
        robot.head_T0, robot.arm_cam_T, robot.calibrated = file_T.copy(), None, False
        robot.scene_offset = np.zeros(3)
        cam.T_pelvis_optical = file_T.copy()
        robot.reset("replay")
        robot.ticks = robot.warmup_ticks + 100

    def hand_eye(T, moments):
        out = []
        for dep, aq, hq in moments:
            P = selfcal.arm_points(dep, K, cam.depth_scale)
            P = P[::max(1, len(P) // 4000)]
            d = cKDTree(model.surface(aq, hq)[0]).query(P @ T[:3, :3].T + T[:3, 3])[0]
            out.append((np.median(d), (d < 0.01).mean()))
        return np.mean(out, 0)

    res, cal = {"today": [], "calibrated": []}, []
    for e in episodes:
        q = D.state[D.rows(e)]
        sp = np.array([np.abs(q[min(len(q) - 1, i + 5), :14] - q[max(0, i - 5), :14]).max() for i in range(len(q))])
        held = []
        for i in np.argsort(sp):
            if i > 30 and sp[i] < 0.03 and all(np.abs(q[i, :14] - q[j, :14]).max() > 0.25 for j in [0] + held):
                held.append(int(i))
            if len(held) == 2:
                break
        dep = D.depth(e, sorted([0] + held))
        moments = [(dep[i], q[i, :14], q[i, 14:]) for i in held]
        for mode in ("today", "calibrated"):
            fresh()
            robot.obs = {"head_depth": dep[0], "arm_q": q[0, :14], "hand_q": q[0, 14:], "t": 0.0}
            if mode == "calibrated":
                robot.arm_views = [(dep[0], q[0, :14], q[0, 14:])]
                out = tools._measure_and_apply(robot)
                if out is None:
                    cal.append(None)
                    continue
                cal.append((np.degrees(robot.waist[0]), np.degrees(robot.waist[1]), *(robot.scene_offset * 1000)))
            else:
                robot.fit_head_camera(dep[0])
            if moments:
                res[mode].append(hand_eye(np.asarray(cam.T_pelvis_optical, float), moments))
    fresh()
    C = np.array([c for c in cal if c is not None])
    print(f"\n6. calibrate on the real frames (the agent's own code), from the first frame of each episode alone: "
          f"{len(C)} of {len(cal)} episodes gave a fit")
    print(f"   upper body pitch {C[:, 0].mean():+.1f} deg (std {C[:, 0].std():.1f}, {C[:, 0].min():+.1f} to {C[:, 0].max():+.1f}), roll "
          f"{C[:, 1].mean():+.1f} (std {C[:, 1].std():.1f}); scene moved x {C[:, 2].mean():+.0f} (std {C[:, 2].std():.0f}), "
          f"y {C[:, 3].mean():+.0f} (std {C[:, 3].std():.0f}), z {C[:, 4].mean():+.0f} (std {C[:, 4].std():.0f}) mm")
    print("   on two later moments of each episode, the arm the camera sees against where the joint angles put it:")
    for mode in ("today", "calibrated"):
        R = np.array(res[mode])
        print(f"   {mode:11s} ({len(R)} episodes): median {np.median(R[:, 0]) * 1000:.0f} mm apart (worst episode "
              f"{R[:, 0].max() * 1000:.0f} mm); {100 * R[:, 1].mean():.0f} % of its points within 10 mm")


if __name__ == "__main__":
    main(sys.argv[1].rstrip("/"), sys.argv[2:])
