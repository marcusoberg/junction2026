"""Self-calibration: measure the head camera and the scene instead of reading them from files.

What is trusted: the robot model (URDF: link lengths, meshes) and the joint angles. Nothing else.
What is measured:
    1. camera_from_arms   the head camera's pose in the robot's frame, from the robot's own arms: the depth image
                          shows the arm and hand, the joint angles say where they are, and the pose that puts one
                          on the other is the camera pose. No assumption about the table or the bins.
    2. headpose.fit_head_pose (existing)   the scene (table plane, bin walls) relative to the camera.
    3. fuse               a GTSAM factor graph over two poses, camera C and scene S (how the real scene sits
                          relative to the one in scene.yaml). Measurements 1 and 2 are factors; the calibration file
                          and scene.yaml are loose priors only. It returns both poses with their uncertainty.

Why two steps: one depth image cannot tell a camera that looks 2 deg further down from a robot that leans 2 deg
forward. Until now we always blamed the camera, which is wrong when the table is higher, the robot leans or stands
a few cm off. The arm is the reference that separates the two.

Offline check on recorded sim frames: sim/replay_selfcal.py.
"""
from __future__ import annotations

import dataclasses
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

# The rigid, well-modelled parts of the arm. The hands are left out on purpose: in the sim (and on the real robot) a
# wrist-camera bracket sits on each hand that the URDF does not have, and a hand may hold an object. On recorded sim
# frames a fit on the hands alone was 3-4 cm off; on the arm without the hands it repeats to 1-2 mm.
ARM_LINK = re.compile(r"^(left|right)_(shoulder|elbow|wrist)_")
ARM_AND_HAND_LINK = re.compile(r"^(left|right)_(shoulder|elbow|wrist|hand)_")
HAND_RADIUS_M = 0.14           # depth points this close to a palm are ignored (hand, bracket, held object)
# On the real robot it is the other way round (data drop 3): the head camera sees mostly hands, little forearm, and
# the hands carry nothing the URDF lacks. ArmModel(hands=True) is the model for that; the robot fits both and
# keeps the one that explains the depth points better (sim: without hands 98 %, real: with hands 81 % vs 31 %).
ABOVE_TABLE_M = 0.075          # arm points are taken above this height: clear of the objects and the bin rims
GATES_M = (0.02, 0.012, 0.008)             # the fit only converges from within about 2 cm, so it is started from
PITCH_STARTS_DEG = (-12, -10, -8, -6, -4, -2, 2, 4, 6, 8, 10, 12)   # many guesses (see starts) and the best one is kept
MIN_MATCHED, MAX_RMS_M = 0.6, 0.005   # an arm fit is used only if most points away from the hands lie on the arm model


# ------------------------------------------------------------------ the arm's surface, from the robot model
def _load_stl(path):
    b = Path(path).read_bytes()
    n = int(np.frombuffer(b[80:84], "<u4")[0]) if len(b) >= 84 else 0
    if len(b) == 84 + 50 * n:                                           # binary
        rec = np.frombuffer(b[84:], dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]))
        return rec["v"].astype(float)
    v = np.array(re.findall(rb"vertex\s+(\S+)\s+(\S+)\s+(\S+)", b), float)
    return v.reshape(-1, 3, 3)


def _rpy(r, p, y):
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def _sample(tris, spacing, rng):
    """Points spread evenly over a triangle mesh (about one per spacing^2), with the triangle normals."""
    e1, e2 = tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]
    nrm = np.cross(e1, e2)
    area = 0.5 * np.linalg.norm(nrm, axis=1)
    total = float(area.sum())
    if total <= 0:
        return np.zeros((0, 3)), np.zeros((0, 3))
    n = max(50, int(total / spacing ** 2))
    k = rng.choice(len(tris), size=n, p=area / total)
    u, v = rng.random(n), rng.random(n)
    flip = u + v > 1
    u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
    pts = tris[k, 0] + u[:, None] * e1[k] + v[:, None] * e2[k]
    return pts, nrm[k] / np.maximum(np.linalg.norm(nrm[k], axis=1, keepdims=True), 1e-12)


class ArmModel:
    """Surface points of both arms and hands in the pelvis frame for any joint angles (URDF meshes + FK)."""

    def __init__(self, kin, urdf=None, spacing=0.004, lock=None, hands=False):
        from aaltonomous.interface.safety import DEFAULT_URDF
        self.kin, self.lock, self.hands = kin, lock, hands
        self.hand_radius = 0.0 if hands else HAND_RADIUS_M
        pattern = ARM_AND_HAND_LINK if hands else ARM_LINK
        urdf = Path(urdf or DEFAULT_URDF)
        rng = np.random.default_rng(0)
        self.links = []                                    # (frame id, points, normals) in the link frame
        for link in ET.parse(urdf).getroot().findall("link"):
            name = link.get("name", "")
            if not pattern.match(name) or not kin.model.existFrame(name):
                continue
            for vis in link.findall("visual"):
                mesh = vis.find("geometry/mesh")
                if mesh is None:
                    continue
                o = vis.find("origin")
                xyz = np.array((o.get("xyz", "0 0 0") if o is not None else "0 0 0").split(), float)
                rpy = np.array((o.get("rpy", "0 0 0") if o is not None else "0 0 0").split(), float)
                scale = np.array(mesh.get("scale", "1 1 1").split(), float)
                path = urdf.parent / mesh.get("filename")
                if not path.exists():
                    continue
                pts, nrm = _sample(_load_stl(path) * scale, spacing, rng)
                R = _rpy(*rpy)
                self.links.append((kin.model.getFrameId(name), pts @ R.T + xyz, nrm @ R.T))
        if not self.links:
            raise RuntimeError(f"no arm meshes found next to {urdf}")

    def palms(self, arm_q, hand_q):
        """Both palm positions (2, 3) in the pelvis frame."""
        def run():
            self.kin._fk(np.asarray(arm_q, float), np.asarray(hand_q, float))
            return np.array([np.asarray(self.kin.data.oMf[self.kin.palm[s]].translation).copy() for s in ("left", "right")])
        if self.lock is not None:
            with self.lock:
                return run()
        return run()

    def surface(self, arm_q, hand_q):
        """-> (points (N,3), outward normals (N,3)) in the pelvis frame."""
        P, N = [], []

        def run():
            self.kin._fk(np.asarray(arm_q, float), np.asarray(hand_q, float))
            for fid, pts, nrm in self.links:
                M = self.kin.data.oMf[fid]
                R, t = np.asarray(M.rotation), np.asarray(M.translation)
                P.append(pts @ R.T + t)
                N.append(nrm @ R.T)
        if self.lock is not None:
            with self.lock:
                run()
        else:
            run()
        return np.vstack(P), np.vstack(N)


# ------------------------------------------------------------------ what the camera sees of the arm
def optical_points(depth, K, depth_scale=0.001, step=2, max_range_m=1.3):
    h, w = depth.shape
    v, u = np.mgrid[0:h:step, 0:w:step]
    z = depth[::step, ::step].astype(np.float32) * depth_scale
    ok = (z > 0.15) & (z < max_range_m)
    x = (u - K[0, 2]) / K[0, 0] * z
    y = (v - K[1, 2]) / K[1, 1] * z
    return np.stack([x, y, z], axis=-1)[ok]


def table_plane(P, rng=None, tol=0.005):
    """The largest plane in the optical-frame points: (unit normal n, d) with n.p + d = height above it, positive on
    the camera's side. None if no plane holds a fifth of the points. Uses nothing but the depth image."""
    rng = rng or np.random.default_rng(0)
    S = P[rng.choice(len(P), size=min(len(P), 6000), replace=False)]
    best, best_n = None, 0
    for _ in range(200):
        a, b, c = S[rng.integers(len(S), size=3)]
        n = np.cross(b - a, c - a)
        ln = np.linalg.norm(n)
        if ln < 1e-8:
            continue
        n = n / ln
        k = int((np.abs((S - a) @ n) < tol).sum())
        if k > best_n:
            best, best_n = (n, a), k
    if best is None or best_n < 0.2 * len(S):
        return None
    n, a = best
    inl = P[np.abs((P - a) @ n) < tol]
    c0 = inl.mean(0)
    _, _, vt = np.linalg.svd(inl - c0, full_matrices=False)
    n = vt[2]
    d = -float(n @ c0)
    if d < 0:                                              # the camera (origin) is above the table
        n, d = -n, -d
    return n, d


def arm_points(depth, K, depth_scale=0.001, min_h=ABOVE_TABLE_M, max_h=0.6):
    """Optical-frame points that stand well above the table: the arms (and whatever a hand holds)."""
    P = optical_points(depth, K, depth_scale)
    if len(P) < 2000:
        return np.zeros((0, 3))
    plane = table_plane(P)
    if plane is None:
        return np.zeros((0, 3))
    h = P @ plane[0] + plane[1]
    return P[(h > min_h) & (h < max_h)]


# ------------------------------------------------------------------ 1. camera pose from the arms
@dataclasses.dataclass
class ArmFit:
    T: np.ndarray                  # 4x4 pelvis <- optical
    ok: bool
    pairs: int = 0                 # depth points matched to the arm surface
    matched: float = 0.0           # fraction of the above-table points that lie on the arm model
    rms_m: float = 0.0
    frames: int = 0
    cov: np.ndarray | None = None  # 6x6, [rotation, translation] of a small motion in the pelvis frame
    why: str = ""


def _kabsch(P, M):
    cp, cm = P.mean(0), M.mean(0)
    U, _, Vt = np.linalg.svd((P - cp).T @ (M - cm))
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, cm - R @ cp
    return T


def _skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0.0]])


def starts(T0, *others):
    """Starting guesses for the camera pose: any poses already at hand (the last result, the table-based fit), the
    calibration file's, and the calibration file's pitched up and down. The camera sits at a known place on the
    robot; what a mount gets wrong is mostly its pitch (the real one is reported to look about 10 deg further down)."""
    out = [np.asarray(T, float) for T in others if T is not None] + [np.asarray(T0, float)]
    for deg in PITCH_STARTS_DEG:
        a = np.radians(deg)
        R = np.eye(4)
        R[1, 1] = R[2, 2] = np.cos(a)
        R[1, 2], R[2, 1] = -np.sin(a), np.sin(a)
        out.append(np.asarray(T0, float) @ R)
    return out


def mount_ok(T, T0, max_move_m=0.03, max_pitch_deg=15.0, max_other_deg=4.0):
    """Is camera pose T one the camera's mount allows, given the robot model's pose T0? The camera is bolted to the
    head: its position is known to a few cm and it cannot roll or pan much; its pitch is what mounts get wrong."""
    if np.linalg.norm(np.asarray(T)[:3, 3] - np.asarray(T0)[:3, 3]) > max_move_m:
        return False
    R = np.asarray(T0)[:3, :3].T @ np.asarray(T)[:3, :3]              # in the camera's own axes: x pitch, y pan, z roll
    pitch = np.degrees(np.arctan2(R[2, 1], R[2, 2]))
    pan = np.degrees(np.arcsin(np.clip(-R[2, 0], -1, 1)))
    roll = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
    return abs(pitch) <= max_pitch_deg and abs(pan) <= max_other_deg and abs(roll) <= max_other_deg


def camera_from_arms(frames, model, T_init, gates=GATES_M, min_pairs=400, correlated=30.0):
    """frames: [(above-table points in the optical frame (N,3), arm_q (14,), hand_q (14,))], one or more, taken while
    both arms stood still. The pose that lays every frame's points on the arm surface the joint angles predict
    (trimmed ICP: a point counts only if it ends up within the gate of the surface and on a side of it that faces
    the camera). T_init: one pose, or a list of starting guesses (see starts): the fit that puts the most points
    on the arm wins. correlated: neighbouring depth points share their errors; the covariance treats this many as one."""
    if isinstance(T_init, (list, tuple)):
        # Several guesses end in different poses (a forearm is close to a cylinder: it fits itself in more than one
        # way). Keep the fits that leave the camera where the robot model has it, then the one that explains the
        # depth points best: most of them on the arm, with the smallest distance.
        T0 = np.asarray(T_init[-len(PITCH_STARTS_DEG) - 1], float) if len(T_init) > len(PITCH_STARTS_DEG) else None
        best = None
        for Ti in T_init:
            fit = camera_from_arms(frames, model, Ti, gates, min_pairs, correlated)
            good = fit.ok and fit.matched >= MIN_MATCHED and fit.rms_m <= MAX_RMS_M and (T0 is None or mount_ok(fit.T, T0))
            score = (good, (fit.matched - fit.rms_m / 0.01) if fit.ok else -1.0)
            if best is None or score > best[0]:
                best = (score, fit)
        fit = best[1]
        if fit.ok and not best[0][0]:
            fit.ok, fit.why = False, (f"no fit that keeps the camera on its mount and lays the points on the arm "
                                      f"(best: {100 * fit.matched:.0f} % on the arm, {fit.rms_m * 1000:.1f} mm)")
        return fit
    from scipy.spatial import cKDTree
    T = np.asarray(T_init, float).copy()
    data, kept = [], []
    for P, aq, hq in frames:
        if len(P) < 100:
            continue
        Q = np.asarray(P, float) @ T[:3, :3].T + T[:3, 3]
        keep = (np.all([np.linalg.norm(Q - c, axis=1) > model.hand_radius + max(gates) for c in model.palms(aq, hq)], axis=0)
                if model.hand_radius > 0 else np.ones(len(Q), bool))
        if keep.sum() < 100:
            continue
        M, N = model.surface(aq, hq)
        data.append((np.asarray(P, float)[keep], M, N, cKDTree(M)))
        kept.append((np.asarray(P, float), aq, hq))
    if not data:
        return ArmFit(T, False, why="no arm in view")
    sel_P = sel_M = None
    for gate in gates:
        for _ in range(12):
            SP, SM = [], []
            cam = T[:3, 3]
            for P, M, N, tree in data:
                Q = P @ T[:3, :3].T + T[:3, 3]
                dist, idx = tree.query(Q, distance_upper_bound=gate)
                ok = np.isfinite(dist)
                idx = np.where(ok, idx, 0)
                facing = np.einsum("ij,ij->i", N[idx], cam - M[idx]) > -0.02
                ok &= facing
                SP.append(P[ok])
                SM.append(M[idx[ok]])
            sel_P, sel_M = np.vstack(SP), np.vstack(SM)
            if len(sel_P) < min_pairs:
                return ArmFit(T, False, pairs=len(sel_P), frames=len(data), why="too few points on the arm")
            T_new = _kabsch(sel_P, sel_M)
            step = np.linalg.norm(T_new[:3, 3] - T[:3, 3]) + np.linalg.norm(T_new[:3, :3] - T[:3, :3])
            T = T_new
            if step < 1e-5:
                break
    Q = sel_P @ T[:3, :3].T + T[:3, 3]
    res = Q - sel_M
    rms = float(np.sqrt((res ** 2).sum(1).mean()))
    seen = on_arm = 0                                      # judged at the final pose, so that fits can be compared
    for (P, aq, hq), (_, M, _, tree) in zip(kept, data):
        Qa = P @ T[:3, :3].T + T[:3, 3]
        if model.hand_radius > 0:
            Qa = Qa[np.all([np.linalg.norm(Qa - c, axis=1) > model.hand_radius + 0.02 for c in model.palms(aq, hq)], axis=0)]
        seen += len(Qa)
        on_arm += int((tree.query(Qa)[0] < gates[-1]).sum()) if len(Qa) else 0
    H = np.zeros((6, 6))
    for q in Q[:: max(1, len(Q) // 4000)]:
        J = np.hstack([-_skew(q), np.eye(3)])
        H += J.T @ J
    H *= len(Q) / max(1, len(Q[:: max(1, len(Q) // 4000)]))
    sigma2 = max(rms, 0.002) ** 2 / 3.0                    # per axis
    cov = np.linalg.inv(H / (sigma2 * correlated))
    return ArmFit(T, True, pairs=len(sel_P), matched=on_arm / max(1, seen), rms_m=rms, frames=len(data), cov=cov)


def level_angles(S):
    """The upper body's pitch and roll (rad, as the waist joints turn: roll about x, then pitch about y) that make
    the measured table level: S is the real scene as a motion of scene.yaml's in the robot's frame (fuse). A robot
    that leans forward sees the table rise away from it; pitching the model's waist by that angle puts the pelvis
    frame, in which the planner works, back level with the table."""
    n = np.asarray(S, float)[:3, :3] @ np.array([0.0, 0.0, 1.0])
    pitch = float(np.arctan2(-n[0], n[2]))
    z1 = -np.sin(pitch) * n[0] + np.cos(pitch) * n[2]
    return pitch, float(np.arctan2(n[1], z1))


# ------------------------------------------------------------------ 3. camera and scene together (GTSAM)
@dataclasses.dataclass
class SceneEstimate:
    T_cam: np.ndarray              # 4x4 pelvis <- optical
    S: np.ndarray                  # 4x4: where the real scene is, as a motion of the scene in scene.yaml (pelvis frame)
    cam_from: str                  # "arms" or "calibration file only"
    table_dz_m: float              # table height minus scene.yaml's, at `at`
    shift_xy_m: tuple              # how far the table / bins sit from scene.yaml, at `at`
    tilt_deg: float                # table tilt in the robot's frame (a leaning robot shows up here)
    tilt_dir_deg: float            # direction in which the table rises, pelvis frame (0 = away from the robot)
    yaw_deg: float                 # rotation of the bins about the vertical
    sigma_dz_m: float
    sigma_xy_m: float
    sigma_tilt_deg: float

    def summary(self):
        return (f"camera from {self.cam_from}; table {self.table_dz_m * 1000:+.0f} mm (+-{self.sigma_dz_m * 1000:.0f}) vs "
                f"scene.yaml, shifted ({self.shift_xy_m[0] * 1000:+.0f}, {self.shift_xy_m[1] * 1000:+.0f}) mm "
                f"(+-{self.sigma_xy_m * 1000:.0f}), tilt {self.tilt_deg:.1f} deg (+-{self.sigma_tilt_deg:.1f}), "
                f"yaw {self.yaw_deg:+.1f} deg")


# Priors: how far each thing may be from its file. Loose on purpose; the measurements decide.
CAM_PRIOR = dict(rot_deg=(12.0, 3.0, 3.0), trans_m=0.03)          # optical axes: x = pitch, y = pan, z = roll
SCENE_PRIOR = dict(rot_deg=(4.0, 4.0, 8.0), trans_m=0.08)         # pelvis axes: x, y = tilt, z = yaw
SCENE_FIT_SIGMA = dict(rot_deg=0.15, trans_m=0.002)               # table plane + bin walls, from the depth image


def fuse(T0, T_fit, arm_fits=(), at=(0.40, 0.0, 0.0), cam_prior=CAM_PRIOR, scene_prior=SCENE_PRIOR):
    """T0: camera pose from the calibration file. T_fit: fit_head_pose's camera pose, i.e. the camera relative to
    the scene AS IF the scene were exactly scene.yaml's, so that  S^-1 * C = T_fit.  arm_fits: ArmFit results.
    at: a point on the table (pelvis frame) at which the height and shift are reported."""
    import gtsam
    nm = gtsam.noiseModel
    C, S = gtsam.symbol("c", 0), gtsam.symbol("s", 0)

    def sig(rot_deg, trans_m):
        r = np.radians(np.broadcast_to(np.asarray(rot_deg, float), (3,)))
        t = np.broadcast_to(np.asarray(trans_m, float), (3,))
        return nm.Diagonal.Sigmas(np.r_[r, t])

    g = gtsam.NonlinearFactorGraph()
    g.add(gtsam.PriorFactorPose3(C, gtsam.Pose3(np.asarray(T0, float)), sig(**cam_prior)))
    g.add(gtsam.PriorFactorPose3(S, gtsam.Pose3(), sig(**scene_prior)))
    g.add(gtsam.BetweenFactorPose3(S, C, gtsam.Pose3(np.asarray(T_fit, float)), sig(**SCENE_FIT_SIGMA)))
    used = [f for f in arm_fits if f.ok and f.cov is not None]
    for f in used:
        pose = gtsam.Pose3(np.asarray(f.T, float))
        A = np.linalg.inv(pose.AdjointMap())               # pelvis-frame motion -> the pose's own frame (GTSAM)
        cov = A @ f.cov @ A.T
        noise = nm.Robust.Create(nm.mEstimator.Huber.Create(1.5), nm.Gaussian.Covariance(cov))
        g.add(gtsam.PriorFactorPose3(C, pose, noise))
    init = gtsam.Values()
    c0 = np.asarray(used[0].T if used else T_fit, float)
    init.insert(C, gtsam.Pose3(c0))
    init.insert(S, gtsam.Pose3(c0 @ np.linalg.inv(np.asarray(T_fit, float))))
    result = gtsam.LevenbergMarquardtOptimizer(g, init).optimize()
    marg = gtsam.Marginals(g, result)
    Tc, Ts = result.atPose3(C).matrix(), result.atPose3(S).matrix()
    cov_s = marg.marginalCovariance(S)                     # [rotation, translation] in S's own frame
    p = np.asarray(at, float)
    moved = Ts[:3, :3] @ p + Ts[:3, 3] - p
    Jp = Ts[:3, :3] @ np.hstack([-_skew(p), np.eye(3)])
    cov_p = Jp @ cov_s @ Jp.T
    up = Ts[:3, :3] @ np.array([0.0, 0.0, 1.0])            # the table's normal in the robot's frame
    tilt = float(np.degrees(np.arccos(np.clip(up[2], -1, 1))))
    return SceneEstimate(
        T_cam=Tc, S=Ts, cam_from="arms" if used else "calibration file only",
        table_dz_m=float(moved[2]), shift_xy_m=(float(moved[0]), float(moved[1])),
        tilt_deg=tilt, tilt_dir_deg=float(np.degrees(np.arctan2(-up[1], -up[0]))) if tilt > 1e-6 else 0.0,
        yaw_deg=float(np.degrees(np.arctan2(Ts[1, 0], Ts[0, 0]))),
        sigma_dz_m=float(np.sqrt(cov_p[2, 2])), sigma_xy_m=float(np.sqrt(max(cov_p[0, 0], cov_p[1, 1]))),
        sigma_tilt_deg=float(np.degrees(np.sqrt(max(cov_s[0, 0], cov_s[1, 1])))))
