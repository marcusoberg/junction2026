# Copied unchanged from our team's other repository, bot-icelli: policy/headpose.py at commit 407ff32
# (fit by Emils, plausibility limits by the policy authors). Used by robot.py at the first look of every episode.
"""Head-camera pose from one depth image: table plane (tilt, roll, height) + bin walls (x, y, yaw).

Copied from Emīls's `policy/decision_oracle/camera.py` (`fit_head_pose` and helpers, commit c787468), because
that folder stays out of the submission image. Keep the two in sync. Our only addition is `plausible()`: the fit
is used only if the correction it applies is within what a mis-calibrated head camera can be.

Why: Aalto's `calibration.yaml` head pose is a URDF default. The real camera (and our venue-look sim) looks about
10° further down (`sim/venue_look/README.md`), which moves every object position by several cm. On 115 venue-look
sim frames this fit found 9.79° ± 0.02° tilt (`policy/decision_oracle/README.md`).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

CAM = "head"


@dataclass
class PoseFit:
    T: np.ndarray                 # 4x4 pelvis <- optical
    ok: bool
    plane_inliers: float = 0.0    # fraction of sampled points on the table plane
    wall_score: float = 0.0       # fraction of wall-band points within 4 mm of a bin outline
    n_wall: int = 0
    tilt_deg: float = 0.0         # rotation applied to the calibration pose by the plane fit
    dxy: tuple = (0.0, 0.0)       # shift applied by the bin fit (m)
    dyaw_deg: float = 0.0
    why: str = ""

    def summary(self):
        return (f"tilt {self.tilt_deg:.1f} deg, shift ({self.dxy[0] * 100:+.1f}, {self.dxy[1] * 100:+.1f}) cm, "
                f"yaw {self.dyaw_deg:+.1f} deg, table {self.plane_inliers:.2f}, walls {self.wall_score:.2f}"
                + (f" ({self.why})" if self.why else ""))


def plausible(fit, max_tilt_deg=15.0, max_shift_m=0.06, max_yaw_deg=6.0):
    """Accept only corrections a mis-mounted / mis-calibrated head camera can need (real one: ~10° tilt)."""
    return (fit.ok and fit.tilt_deg <= max_tilt_deg and max(abs(fit.dxy[0]), abs(fit.dxy[1])) <= max_shift_m
            and abs(fit.dyaw_deg) <= max_yaw_deg)


def _points(cal, depth, step=1):
    """Optical-frame points (h, w, 3) for a depth image (mm), subsampled by `step`."""
    c = cal.cams[CAM]
    h, w = depth.shape
    v, u = np.mgrid[0:h:step, 0:w:step]
    z = depth[::step, ::step].astype(np.float32) * c.depth_scale
    x = (u - c.K[0, 2]) / c.K[0, 0] * z
    y = (v - c.K[1, 2]) / c.K[1, 1] * z
    return np.stack([x, y, z], axis=-1)


def _rot_a_to_b(a, b):
    """Smallest rotation taking unit vector a to unit vector b."""
    v, c = np.cross(a, b), float(a @ b)
    s = np.linalg.norm(v)
    if s < 1e-9:
        return np.eye(3)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]]) / s
    return np.eye(3) + s * K + (1 - c) * K @ K


def _bin_outlines(scene, res=0.002):
    """Points (pelvis xy) along the wall centre lines of every bin, `res` apart."""
    b = scene.cfg["bins"]
    half = np.asarray(b["outer_size"], float) / 2 - float(b.get("wall_thickness", 0.003)) / 2
    pts = []
    for cx, cy in b["centers"].values():
        t = np.arange(-1, 1, res / half.max())
        for sx in (-1, 1):
            pts.append(np.stack([np.full_like(t, cx + sx * half[0]), cy + t * half[1]], 1))
            pts.append(np.stack([cx + t * half[0], np.full_like(t, cy + sx * half[1])], 1))
    pm = np.vstack(pts)
    return scene.mat_to_pelvis(np.c_[pm, np.zeros(len(pm))])[:, :2]


def fit_head_pose(depth, cal, scene, T0=None, rng=None, max_range_mm=1600):
    """Fit the head camera pose (pelvis <- optical) to one depth image. Returns PoseFit; .ok False keeps T0."""
    import cv2
    T0 = np.asarray(cal._T(CAM) if T0 is None else T0, float)
    rng = rng or np.random.default_rng(0)
    P = _points(cal, depth, step=4).reshape(-1, 3)
    d = depth[::4, ::4].reshape(-1)
    P = P[(d > 150) & (d < max_range_mm)]
    if len(P) < 2000:
        return PoseFit(T0, False, why="too little valid depth")
    Q = P @ T0[:3, :3].T + T0[:3, 3]              # pelvis frame according to the calibration
    zc = scene.counter_z

    # 1. table plane: RANSAC among planes roughly horizontal and near the expected table height
    best, best_n = None, 0
    for _ in range(150):
        a, b, c = Q[rng.integers(len(Q), size=3)]
        n = np.cross(b - a, c - a)
        ln = np.linalg.norm(n)
        if ln < 1e-8:
            continue
        n = n / ln
        if n[2] < 0:
            n = -n
        if n[2] < 0.85:                         # more than ~32 deg off the calibration's vertical
            continue
        off = float(n @ a)
        if abs(off - zc) > 0.25:
            continue
        k = int((np.abs(Q @ n - off) < 0.005).sum())
        if k > best_n:
            best, best_n = (n, off), k
    if best is None or best_n < 0.15 * len(Q):
        return PoseFit(T0, False, plane_inliers=best_n / len(Q), why="no table plane")
    n, off = best
    inl = np.abs(Q @ n - off) < 0.006
    c0 = Q[inl].mean(0)                           # refine: least squares on the inliers
    _, _, vt = np.linalg.svd(Q[inl] - c0, full_matrices=False)
    n = vt[2] if vt[2][2] > 0 else -vt[2]
    R1 = _rot_a_to_b(n, np.array([0.0, 0.0, 1.0]))
    T1 = np.eye(4)
    T1[:3, :3] = R1 @ T0[:3, :3]
    T1[:3, 3] = R1 @ T0[:3, 3]
    T1[2, 3] += zc - float((R1 @ c0)[2])          # table plane -> counter height
    tilt = float(np.degrees(np.arccos(np.clip(n[2], -1, 1))))
    fit = PoseFit(T1, True, plane_inliers=float(inl.mean()), tilt_deg=tilt)

    # 2. bins: wall points between object height and the rim, seen from above, against the known box outlines
    Q1 = P @ T1[:3, :3].T + T1[:3, 3]
    wall = float(scene.cfg["bins"]["wall_height"])
    hobj = scene.object_height()
    h = Q1[:, 2] - zc
    band = (h > min(hobj + 0.006, wall - 0.03)) & (h < wall - 0.006)
    W = Q1[band][:, :2]
    out = _bin_outlines(scene)
    lo, hi = out.min(0) - 0.15, out.max(0) + 0.15
    W = W[np.all((W > lo) & (W < hi), axis=1)]
    fit.n_wall = len(W)
    if len(W) < 150:
        fit.why = "too few bin wall points: kept calibration x, y, yaw"
        return fit
    res = 0.002
    size = np.ceil((hi - lo) / res).astype(int)   # (nx, ny)
    tpl = np.full((size[0], size[1]), 255, np.uint8)
    ij = np.floor((out - lo) / res).astype(int)
    tpl[ij[:, 0], ij[:, 1]] = 0
    dist = cv2.distanceTransform(tpl, cv2.DIST_L2, 3) * res            # m to the nearest outline
    cap = 0.02
    cost_map = np.minimum(dist, cap).astype(np.float32)

    def cost(yaw, dx, dy):
        c, s = np.cos(yaw), np.sin(yaw)
        X = W @ np.array([[c, s], [-s, c]]) + [dx, dy]
        ij = np.floor((X - lo) / res).astype(int)
        ok = (ij[:, 0] >= 0) & (ij[:, 0] < size[0]) & (ij[:, 1] >= 0) & (ij[:, 1] < size[1])
        cst = np.full(len(X), cap, np.float32)
        cst[ok] = cost_map[ij[ok, 0], ij[ok, 1]]
        return float(cst.mean()), cst

    best = (cost(0.0, 0.0, 0.0)[0], 0.0, 0.0, 0.0)
    for span_yaw, st_yaw, span, st in ((8.0, 1.0, 0.10, 0.01), (1.0, 0.25, 0.01, 0.002), (0.25, 0.05, 0.002, 0.0005)):
        _, y0, x0, yy0 = best[0], best[1], best[2], best[3]
        for yaw in np.radians(np.arange(-span_yaw, span_yaw + 1e-9, st_yaw)) + y0:
            for dx in np.arange(-span, span + 1e-9, st) + x0:
                for dy in np.arange(-span, span + 1e-9, st) + yy0:
                    c_ = cost(yaw, dx, dy)[0]
                    if c_ < best[0]:
                        best = (c_, yaw, dx, dy)
    _, yaw, dx, dy = best
    cst = cost(yaw, dx, dy)[1]
    fit.wall_score = float((cst < 0.004).mean())
    if fit.wall_score < 0.35:
        fit.why = f"bin outline fit weak ({fit.wall_score:.2f}): kept calibration x, y, yaw"
        return fit
    c, s = np.cos(yaw), np.sin(yaw)
    R2 = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
    T2 = np.eye(4)
    T2[:3, :3] = R2 @ T1[:3, :3]
    T2[:3, 3] = R2 @ T1[:3, 3] + [dx, dy, 0.0]
    fit.T, fit.dxy, fit.dyaw_deg = T2, (float(dx), float(dy)), float(np.degrees(yaw))
    return fit
