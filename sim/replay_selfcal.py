"""Self-calibration on recorded sim frames: how well can the robot measure its camera and the scene by itself?

    PYTHONPATH=aaltonomous-humanoid:adk_agents:. .venv/bin/python sim/replay_selfcal.py sim_out/<run>

Needs a run made with the arm-frame recording (agent_trace/arm_frames/: head depth + joint angles while an arm is in
view) and its look frames (agent_trace/frames/). Offline, seconds, no model calls. It prints:

  1. the camera pose from the robot's own arms (robot_sorter/selfcal.py), from different starting guesses and from
     different subsets of frames: the spread is the measurement's repeatability;
  2. camera and scene together (GTSAM): how far the scene is from scene.yaml. In a nominal sim run the true answer is
     "not at all", so what it reports there is the error of the whole chain;
  3. the same with files that are wrong on purpose (scene.yaml with the table lower and nearer, a calibration file
     with the camera 10 deg off): what the estimate recovers, against what today's fit (which trusts scene.yaml) does.
"""
import copy
import glob
import json
import os
import sys

import cv2
import numpy as np

from robot_sorter import selfcal
from robot_sorter.headpose import fit_head_pose
from robot_sorter.robot import get_robot

STILL_RAD = 0.003        # both arms within this of their pose from 0.27 s before to 0.1 s after the frame


def rot(axis, deg):
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    R = np.eye(4)
    i, j = [(1, 2), (2, 0), (0, 1)][axis]
    R[i, i] = R[j, j] = c
    R[i, j], R[j, i] = -s, s
    return R


def diff(A, B):
    """Translation (mm) and rotation (deg) between two poses."""
    d = np.linalg.inv(A) @ B
    return (float(np.linalg.norm(A[:3, 3] - B[:3, 3]) * 1000),
            float(np.degrees(np.arccos(np.clip((np.trace(d[:3, :3]) - 1) / 2, -1, 1)))))


def main(run):
    robot = get_robot()
    robot.log = lambda m: None
    ctx = robot.ctx
    cam = ctx.cal.cams[ctx.camera.cam]
    K, T0 = np.asarray(cam.K, float), np.asarray(robot.head_T0, float)
    model = selfcal.ArmModel(ctx.kin, lock=ctx.lock)
    tdir = os.path.join(run, "agent_trace")
    episodes = sorted({os.path.basename(f)[:5] for f in glob.glob(os.path.join(tdir, "arm_frames", "ep*.json"))})
    if not episodes:
        sys.exit(f"no arm frames in {tdir}/arm_frames")
    for ep in episodes:
        print(f"\n================ {os.path.basename(run)} {ep}")
        frames, info = [], []
        log = glob.glob(os.path.join(run, "runs", "eval", "*", f"episode_{ep[2:]}.npz"))     # the evaluator's tick log
        ticks = np.load(log[0]) if log else None
        for f in sorted(glob.glob(os.path.join(tdir, "arm_frames", f"{ep}_*.json"))):
            j = json.load(open(f))
            depth = cv2.imread(f[:-5] + "_depth.png", cv2.IMREAD_UNCHANGED)
            if depth is None:
                continue
            if ticks is not None:
                k = int(np.argmin(np.abs(ticks["t"] - j["t"])))
                lo, hi = max(0, k - 8), min(len(ticks["t"]) - 1, k + 3)
                still = float(np.abs(ticks["arm_q"][lo:hi + 1] - ticks["arm_q"][k]).max()) < STILL_RAD
            else:
                still = bool(j.get("still", False))
            P = selfcal.arm_points(depth, K, cam.depth_scale)
            frames.append((P, np.array(j["arm_q"]), np.array(j["hand_q"])))
            info.append({"t": j["t"], "still": still, "n": len(P), "move": j["move"]})
        usable = [k for k, i in enumerate(info) if i["n"] >= 300]
        still = [k for k in usable if info[k]["still"]]
        print(f" arm frames: {len(frames)} recorded, {len(usable)} with the arm in view, {len(still)} of those with the arm still")
        if not usable:
            continue
        look = sorted(glob.glob(os.path.join(tdir, "frames", f"{ep}_00_*_depth.png")))
        look_depth = cv2.imread(look[0], cv2.IMREAD_UNCHANGED) if look else None
        cam.T_pelvis_optical = T0.copy()
        scene_fit = fit_head_pose(look_depth, ctx.cal, ctx.scene) if look_depth is not None else None
        T_fit = np.asarray(scene_fit.T, float) if scene_fit is not None and scene_fit.ok else T0
        print(f" today's fit (trusts scene.yaml): {scene_fit.summary() if scene_fit else 'no look frame'}")

        # 1. camera from the arms
        if not still:
            print(" no frame with both arms still: nothing to calibrate against")
            continue
        pick = []
        for k in still:                                    # one frame per arm pose, as the robot keeps them
            if all(float(np.abs(frames[k][1] - frames[m][1]).max()) >= 0.1 for m in pick):
                pick.append(k)
        sub = pick
        p_tab = np.array([0.35, 0.0, ctx.scene.counter_z])
        p_opt = np.linalg.inv(T_fit) @ np.r_[p_tab, 1.0]
        fits = {}
        for name, Ti in (("today's fit", T_fit), ("calibration file", T0),
                         ("all guesses (as the robot does)", selfcal.starts(T0, T_fit)),
                         ("all guesses, file 10 deg off", selfcal.starts(T0 @ rot(0, 10.0))),
                         ("all guesses, file 7 deg off the other way", selfcal.starts(T0 @ rot(0, -7.0)))):
            fit = selfcal.camera_from_arms([frames[k] for k in sub], model, Ti)
            fits[name] = fit
            print(f" camera from the arms, {len(sub)} frames, start = {name:32s}: " + (
                f"{fit.pairs} points ({100 * fit.matched:.0f} % on the arm), rms {fit.rms_m * 1000:.1f} mm; "
                f"vs today's fit {diff(T_fit, fit.T)[0]:.1f} mm, {diff(T_fit, fit.T)[1]:.2f} deg; a point on the table 35 cm "
                f"ahead lands {np.round(((fit.T @ p_opt)[:3] - p_tab) * 1000, 1).tolist()} mm from where today's fit puts it"
                if fit.ok else f"FAILED: {fit.why}"))
        ref = fits["all guesses (as the robot does)"]
        if not ref.ok:
            continue
        singles = []
        for k in pick:
            f1 = selfcal.camera_from_arms([frames[k]], model, ref.T, gates=selfcal.GATES_M, min_pairs=300)
            if f1.ok and f1.matched >= selfcal.MIN_MATCHED:
                singles.append(f1)
        if singles:
            # what matters for grasping: where a point on the table ends up
            q_opt = np.linalg.inv(ref.T) @ np.r_[p_tab, 1.0]
            spread = np.array([(f.T @ q_opt)[:3] for f in singles]) - p_tab
            print(f" one frame at a time ({len(singles)} of {len(pick)} frames give a fit): the table point moves by (std, mm) "
                  f"x {spread[:, 0].std() * 1000:.1f}  y {spread[:, 1].std() * 1000:.1f}  z {spread[:, 2].std() * 1000:.1f}"
                  f"   (worst {np.abs(spread).max() * 1000:.1f} mm)")
        print(f" still frames at t = {[round(info[k]['t'], 1) for k in pick]} s")

        # 2. camera and scene together, files as they are
        if scene_fit is None or not scene_fit.ok:
            continue
        est = selfcal.fuse(T0, T_fit, [ref])
        print(f" scene, files as they are   : {est.summary()}")
        print(f" scene, without the arms    : {selfcal.fuse(T0, T_fit, []).summary()}")

        # 3. files wrong on purpose
        for label, d_scene, d_cam in (("scene.yaml: table 30 mm too low, 20 mm too near", (-0.02, 0.0, -0.03), None),
                                      ("scene.yaml: table 20 mm too high, bins 30 mm to the left", (0.0, 0.03, 0.02), None),
                                      ("calibration file: camera 10 deg off", None, rot(0, 10.0)),
                                      ("both: table 30 mm too low and camera 10 deg off", (0.0, 0.0, -0.03), rot(0, 10.0))):
            scene = copy.deepcopy(ctx.scene)
            if d_scene is not None:
                o = np.asarray(scene.cfg["counter"]["mat_origin_in_pelvis"], float) + d_scene
                scene.cfg["counter"]["mat_origin_in_pelvis"] = o.tolist()
            T0w = T0 @ d_cam if d_cam is not None else T0
            cam.T_pelvis_optical = T0w.copy()
            fw = fit_head_pose(look_depth, ctx.cal, scene, T0=T0w)
            cam.T_pelvis_optical = T0.copy()
            if not fw.ok:
                print(f" {label}: today's fit fails ({fw.why})")
                continue
            arm = selfcal.camera_from_arms([frames[k] for k in sub], model, selfcal.starts(T0w, np.asarray(fw.T, float)))
            arm.ok = arm.ok and arm.matched >= selfcal.MIN_MATCHED and arm.rms_m <= selfcal.MAX_RMS_M
            e = selfcal.fuse(T0w, np.asarray(fw.T, float), [arm] if arm.ok else [])
            want = -np.asarray(d_scene if d_scene is not None else (0, 0, 0), float) + [est.shift_xy_m[0], est.shift_xy_m[1], est.table_dz_m]
            got = np.array([e.shift_xy_m[0], e.shift_xy_m[1], e.table_dz_m])
            # today: the scene is taken from the file, so every object is placed wrong by the file's error
            today = np.linalg.norm(d_scene) * 1000 if d_scene is not None else 0.0
            print(f" {label}:\n     measured: {e.summary()}\n     should be (x, y, z) {np.round(want * 1000).astype(int).tolist()} mm; "
                  f"error of the measured scene {np.linalg.norm(got - want) * 1000:.1f} mm; today's error {today:.0f} mm")


if __name__ == "__main__":
    for r in sys.argv[1:]:
        main(r.rstrip("/"))
