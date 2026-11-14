"""Shape and rotation of an object from the outline of its top face, seen from above.

`classify_top` is copied from our team's other repository, bot-icelli (policy/decision_oracle/camera.py, written by
Emils; commit 407ff32). Measured there on 115 saved sim frames: shape right on 153 of 162 objects, rotation error
under 0.5 deg median, and a block lying on its side is recognised ("tipped"). The rotation follows the sim's
convention, which is what the oracle's face-aligned grasp expects: cube faces at yaw + k*90 deg, triangle corners
at yaw + 90 + k*120 deg.

`top_face_points` is ours: the 3D points of one detected object's top face, from the head depth image only.
"""
from __future__ import annotations

import numpy as np


def classify_top(xy):
    """Top-face points (n, 2) in m, seen from above -> (shape, yaw, centre, features).
    shape: cube | cylinder | triangle | tipped (a ridge, not a face) | None (too few points)."""
    import cv2
    if len(xy) < 25:
        return None, None, None, {}
    pts = (np.asarray(xy, np.float32) * 1000.0)                 # mm
    hull = cv2.convexHull(pts)
    A = float(cv2.contourArea(hull))
    if A < 1.0:
        return None, None, None, {}
    (rc, (rw, rh), rang) = cv2.minAreaRect(hull)
    (_, cr) = cv2.minEnclosingCircle(hull)
    At, tri = cv2.minEnclosingTriangle(hull)
    m = cv2.moments(hull)
    centre = np.array([m["m10"] / m["m00"], m["m01"] / m["m00"]]) / 1000.0
    f = {"area_mm2": A, "rect": A / max(rw * rh, 1e-6), "circle": A / max(np.pi * cr * cr, 1e-6),
         "tri": A / max(float(At), 1e-6), "short_mm": float(min(rw, rh)), "long_mm": float(max(rw, rh))}
    if f["short_mm"] < 22.0 or f["short_mm"] < 0.45 * f["long_mm"]:
        return "tipped", None, centre, f
    # distance to the ideal (rect, circle, tri) fill ratios of each outline
    ideal = {"cube": (1.0, 0.637, 0.5), "cylinder": (0.785, 1.0, 0.605), "triangle": (0.5, 0.413, 1.0)}
    v = np.array([f["rect"], f["circle"], f["tri"]])
    d = {k: float(np.linalg.norm(v - np.array(i))) for k, i in ideal.items()}
    shape = min(d, key=d.get)
    f["dist"] = d
    yaw = 0.0
    if shape == "cube":
        a = np.radians(rang)
        yaw = float((a + np.pi / 4) % (np.pi / 2) - np.pi / 4)
    elif shape == "triangle":
        t = tri.reshape(3, 2) / 1000.0
        ang = np.arctan2(t[:, 1] - centre[1], t[:, 0] - centre[0]) - np.pi / 2
        # the three corners agree modulo 120 deg: circular mean of 3 * angle
        a3 = np.arctan2(np.sin(3 * ang).mean(), np.cos(3 * ang).mean()) / 3
        yaw = float((a3 + np.pi / 3) % (2 * np.pi / 3) - np.pi / 3)
    return shape, yaw, centre, f


def ridge_yaw(xy):
    """Top-strip points (n, 2) of a block lying on its side -> (direction of its ridge in rad, mod pi; how many
    times longer than wide the strip is). The idea and the 2.0 ratio limit are our teammates' (bot-icelli v1.5:
    47 of 48 lying prisms found offline, 0.8 deg median error)."""
    p = np.asarray(xy, float)
    c = p - p.mean(0)
    w, v = np.linalg.eigh(c.T @ c)
    ax = v[:, -1]
    return float((np.arctan2(ax[1], ax[0]) + np.pi / 2) % np.pi - np.pi / 2), float(np.sqrt(w[-1] / max(w[0], 1e-12)))


def top_face_points(ctx, obs, xyz, radius_m=0.045, band_m=0.006, window_px=60):
    """(n, 2) pelvis-frame xy of the depth pixels on the top face of the object whose top centre is `xyz`."""
    depth = obs.get("head_depth")
    if depth is None:
        return np.zeros((0, 2))
    cam = ctx.camera.cam
    u0, v0, _ = ctx.cal.project(cam, np.asarray(xyz, float)[None])[0]
    h, w = depth.shape[:2]
    ua, ub = int(max(0, u0 - window_px)), int(min(w, u0 + window_px + 1))
    va, vb = int(max(0, v0 - window_px)), int(min(h, v0 + window_px + 1))
    if ua >= ub or va >= vb:
        return np.zeros((0, 2))
    d = depth[va:vb, ua:ub].astype(float)
    vv, uu = np.nonzero((d >= 150) & (d <= 2000))
    if not len(uu):
        return np.zeros((0, 2))
    P = ctx.cal.pixel_to_pelvis(cam, uu + ua, vv + va, d[vv, uu])
    near = np.linalg.norm(P[:, :2] - np.asarray(xyz, float)[:2], axis=1) < radius_m
    top = np.abs(P[:, 2] - float(xyz[2])) <= band_m
    return P[near & top][:, :2]


def detect_by_depth(ctx, obs, cell_m=0.0025, min_cells=40):
    """Objects on the counter from the depth image alone: no colour thresholds of any kind.

    Every depth pixel becomes a 3D point; the points that stand at object height above the counter, on the mat and
    outside the bins are dropped onto a top-down grid, and each connected patch is one object. Its top face gives
    the centre, the shape and the rotation; the colour is only MEASURED (for the model to judge), never used to
    find or split objects. Returns WorldObjects like agent.skills.perception.detect_camera."""
    import cv2
    from agent.skills.perception import WorldObject
    depth, rgb = obs.get("head_depth"), obs["head_rgb"]
    if depth is None:
        return []
    cam, sc = ctx.camera.cam, ctx.scene
    H = float(sc.object_height())
    d = depth.astype(float)
    vv, uu = np.nonzero((d >= 150) & (d <= 2000))
    P = ctx.cal.pixel_to_pelvis(cam, uu, vv, d[vv, uu])
    h = P[:, 2] - sc.counter_z
    pm = sc.pelvis_to_mat(P)
    W, D = sc.cfg["counter"]["mat_size"]
    ok = (h >= 0.3 * H) & (h <= 2.0 * H) & (pm[:, 0] >= -0.02) & (pm[:, 0] <= W + 0.02) & (pm[:, 1] >= -0.02) & (pm[:, 1] <= D + 0.02)
    half = np.asarray(sc.cfg["bins"]["outer_size"], float) / 2 + 0.008
    for ctr in sc.cfg["bins"]["centers"].values():
        ok &= ~np.all(np.abs(pm[:, :2] - np.asarray(ctr, float)) <= half, axis=1)
    P, uu, vv = P[ok], uu[ok], vv[ok]
    if len(P) < min_cells:
        return []
    lo = P[:, :2].min(0) - cell_m
    ij = np.floor((P[:, :2] - lo) / cell_m).astype(int)
    grid = np.zeros(ij.max(0) + 2, np.uint8)
    grid[ij[:, 0], ij[:, 1]] = 255
    grid = cv2.morphologyEx(grid, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))      # depth holes, sparse far pixels
    n, lab = cv2.connectedComponents(grid, connectivity=8)
    which = lab[ij[:, 0], ij[:, 1]]
    out = []
    for k in range(1, n):
        m = which == k
        if int((lab == k).sum()) < min_cells or m.sum() < 30:
            continue
        Q = P[m]
        z_top = np.percentile(Q[:, 2], 92)
        top = Q[:, 2] >= z_top - 0.006
        if top.sum() < 15:
            continue
        try:
            shape, yaw, centre, feat = classify_top(Q[top][:, :2])
        except Exception:                     # OpenCV's triangle fit can fail on a thin strip: a lying object
            shape, yaw, centre, feat = "tipped", None, Q[top][:, :2].mean(0), {}
        ridge = None
        if shape == "tipped" and top.sum() >= 25:
            direction, ratio = ridge_yaw(Q[top][:, :2])
            ridge = direction if ratio >= 2.0 else None
        xy = centre if centre is not None else Q[top][:, :2].mean(0)
        if feat.get("long_mm", 0) > 110:                       # far too big for one object (an arm, a tool, a heap)
            continue
        pix = rgb[vv[m], uu[m]].astype(float)
        sat = (pix.max(1) - pix.min(1)) / np.maximum(pix.max(1), 1.0)
        pix = pix[sat >= np.percentile(sat, 67)]
        o = WorldObject("?", "object", np.array([xy[0], xy[1], float(np.median(Q[top][:, 2]))]), source="depth",
                        extra={"uv": (float(np.median(uu[m][top])), float(np.median(vv[m][top]))), "n_px": int(m.sum()),
                               "shape_seen": shape, "ridge_yaw": ridge,
                               "rgb": [int(c) for c in np.median(pix, axis=0)]})
        o.shape = shape if shape in ("cube", "cylinder", "triangle") else None
        o.yaw = yaw if o.shape in ("cube", "triangle") else None
        out.append(o)
    return out
