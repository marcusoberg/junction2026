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
