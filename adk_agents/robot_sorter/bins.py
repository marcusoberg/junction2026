"""Where the three bins really are, measured from the head camera instead of read from scene.yaml.

Why: on the real robot the boxes are not the scene file's (run of 15 Nov: the hand released short of the boxes; on
the real frames of data drop 3 the boxes are about 20 cm wide and the right one starts 8 cm nearer the centre than
the file says, so a release at the file's position lands on its outer wall). The table-and-box template fit in
headpose.py cannot fix that: it matches the file's boxes to the image (score 0.36-0.47 on real frames, 0.6-0.74 in
the sim) and shifts everything by one common offset.

How: the Gemini Robotics-ER model draws a box around each bin in the head image (what it is built for), and the
depth image gives the geometry inside each box: the points that stand above object height are the bin's walls, the
nearest row of them is the wall facing the robot. From that: how far away the bin starts, how wide it is, how high
its rim is. The planner then drops a fixed distance behind that wall, at the bin's middle.

Checked on saved frames (15 Nov): in the sim the three near walls measure x = 0.400 m (file 0.400), the rim 5.8 cm
(file 6.0), the middle bin's centre -3 mm, its width 20.4 cm (file 21).
"""
from __future__ import annotations

import json

import numpy as np

from . import er

PROMPT = ("This image is from the head camera of a humanoid robot looking down at a table. At the far side of the table "
          "stand three open boxes (bins) in a row. Return a bounding box around each of the three bins, labelled left, "
          "center and right as they appear in the image from left to right. Do not include the robot's hands or the small "
          "blocks. The boxes are in [ymin, xmin, ymax, xmax] format normalized to 0-1000.")
SCHEMA = {"type": "object", "required": ["items"], "properties": {"items": {"type": "array", "items": {
    "type": "object", "required": ["label", "box_2d"], "properties": {
        "label": {"type": "string", "enum": ["left", "center", "right"]},
        "box_2d": {"type": "array", "items": {"type": "integer"}}}}}}}
WALL_BAND_M = (0.05, 0.14)        # above the blocks (4.5 cm), below a hand or forearm held over the table
MIN_X_M = 0.34                    # the bins stand behind the blocks; anything tall nearer than this is an arm or a hand
MAX_FROM_FILE_M = 0.15            # a measurement further than this from the scene file's bin is not believed


def ask(rgb, timeout=3.5, tries=2):
    """ER's boxes: {"left" | "center" | "right": [ymin, xmin, ymax, xmax] 0..1000}, or None on any failure.
    The answer is checked: three boxes, every number inside 0..1000, each box with some size. (On a real frame the
    model once wrote 342220 for a coordinate; such an answer is asked for again once, not used.)"""
    import cv2
    if not er.available():
        return None
    try:
        ok, jpg = cv2.imencode(".jpg", np.ascontiguousarray(rgb[..., ::-1]), [cv2.IMWRITE_JPEG_QUALITY, 90])
    except Exception:  # noqa: BLE001
        return None
    for _ in range(tries):
        try:
            text, _s = er.create(jpg.tobytes(), PROMPT, SCHEMA, timeout=timeout)
            items = json.loads(text).get("items", [])
        except Exception:  # noqa: BLE001  (no answer: the caller keeps the bins it has)
            continue
        good = []
        for it in items:
            b = it.get("box_2d") or []
            if len(b) == 4 and all(0 <= float(v) <= 1000 for v in b) and b[2] - b[0] >= 30 and b[3] - b[1] >= 30:
                good.append([float(v) for v in b])
        if len(good) == 3 and len(items) == 3:
            good.sort(key=lambda b: b[1] + b[3])                      # left to right in the image, whatever the labels
            return dict(zip(("left", "center", "right"), good))
    return None


def measure(ctx, depth, boxes, min_spread_m=0.08):
    """-> {bin: {"near_wall_x", "y_lo", "y_hi", "rim_m", "cut": bool}} in the frame the planner works in, for the bins
    whose near wall could be found. `cut`: the bin runs out of the image, so its outer end was not seen."""
    cam = ctx.cal.cams[ctx.camera.cam]
    K, T = np.asarray(cam.K, float), np.asarray(cam.T_pelvis_optical, float)
    H, W = depth.shape
    v, u = np.mgrid[0:H, 0:W]
    z = depth.astype(np.float32) * cam.depth_scale
    P = np.stack([(u - K[0, 2]) / K[0, 0] * z, (v - K[1, 2]) / K[1, 1] * z, z], -1) @ T[:3, :3].T + T[:3, 3]
    h = P[..., 2] - ctx.scene.counter_z
    tall = (z > 0.15) & (z < 1.6) & (h > WALL_BAND_M[0]) & (h < WALL_BAND_M[1])
    out = {}
    for name, (y0, x0, y1, x1) in boxes.items():
        m = np.zeros((H, W), bool)
        m[int(y0 / 1000 * H):int(y1 / 1000 * H) + 1, int(x0 / 1000 * W):int(x1 / 1000 * W) + 1] = True
        pts, hh = P[m & tall], h[m & tall]
        keep = pts[:, 0] >= MIN_X_M                                    # nearer than this is the robot's own arms
        pts, hh = pts[keep], hh[keep]
        if len(pts) < 150:
            continue
        # the nearest 2.5 cm slice that is a wall: many points, spread sideways (a block or a finger is not)
        for x in np.arange(np.percentile(pts[:, 0], 1), np.percentile(pts[:, 0], 60), 0.01):
            near = (pts[:, 0] >= x) & (pts[:, 0] < x + 0.025)
            if near.sum() >= 120 and np.ptp(pts[near, 1]) >= min_spread_m:
                lo, hi = np.percentile(pts[near, 1], [3, 97])
                out[name] = {"near_wall_x": float(np.percentile(pts[near, 0], 10)), "y_lo": float(lo), "y_hi": float(hi),
                             "rim_m": float(np.percentile(hh[near], 92)), "cut": bool(x0 <= 8 or x1 >= 992)}
                break
    return out


def near_wall(ctx, depth):
    """Without the model's boxes: how far away the row of bins starts, from the depth image alone -> {"near_wall_x",
    "rim_m"} or None. The same test as in measure, over the whole image and for a row at least 30 cm wide."""
    whole = measure(ctx, depth, {"all": [0.0, 0.0, 1000.0, 1000.0]}, min_spread_m=0.30)
    f = whole.get("all")
    return None if f is None else {"near_wall_x": f["near_wall_x"], "rim_m": f["rim_m"]}


def apply_near_wall(ctx, f):
    """Only the distance is known: move the three bins of the scene file forward or back to it, sideways as they are."""
    b = ctx.scene.cfg["bins"]
    origin = np.asarray(ctx.scene.cfg["counter"]["mat_origin_in_pelvis"], float)
    x = f["near_wall_x"] + float(b["outer_size"][1]) / 2
    changed = {}
    for name, c in b["centers"].items():
        old = np.asarray(ctx.scene.mat_to_pelvis([c[0], c[1], 0.0]), float)[:2]
        if abs(x - old[0]) <= MAX_FROM_FILE_M and 0.03 <= f["rim_m"] <= 0.15:
            b["centers"][name] = [float(c[0]), float(x - origin[0])]
            changed[name] = [round(float(x), 3), round(float(old[1]), 3)]
    if changed:
        b["wall_height"] = float(np.clip(f["rim_m"], 0.04, 0.14))
        ctx.planner.__dict__.pop("_reach", None)
    return changed


def apply(ctx, found):
    """Put the measured bins into the scene the planner uses. -> {bin: [x, y] of its middle, pelvis frame} for the
    bins that were changed. A bin that was not found, or that measures implausibly far from the file's, is left."""
    b = ctx.scene.cfg["bins"]
    origin = np.asarray(ctx.scene.cfg["counter"]["mat_origin_in_pelvis"], float)
    depth_m = float(b["outer_size"][1])                               # along the robot's forward axis (not measured)
    widths = [f["y_hi"] - f["y_lo"] for f in found.values() if not f["cut"]]
    width = float(np.clip(np.median(widths), 0.15, 0.30)) if widths else float(b["outer_size"][0])
    changed, rims = {}, []
    for name, f in found.items():
        if f["cut"] or (f["y_hi"] - f["y_lo"]) < 0.8 * width:         # outer end not seen: inner end + half a bin
            mid = 0.5 * (f["y_lo"] + f["y_hi"])
            y = f["y_lo"] + width / 2 if mid > 0 else f["y_hi"] - width / 2
        else:
            y = 0.5 * (f["y_lo"] + f["y_hi"])
        x = f["near_wall_x"] + depth_m / 2
        old = np.asarray(ctx.scene.mat_to_pelvis([b["centers"][name][0], b["centers"][name][1], 0.0]), float)[:2]
        if np.linalg.norm(np.array([x, y]) - old) > MAX_FROM_FILE_M or not 0.03 <= f["rim_m"] <= 0.15:
            continue
        b["centers"][name] = [float(y - origin[1]), float(x - origin[0])]     # mat x runs along pelvis y, mat y along pelvis x
        changed[name] = [round(float(x), 3), round(float(y), 3)]
        rims.append(f["rim_m"])
    if changed:
        b["wall_height"] = float(np.clip(max(rims), 0.04, 0.14))      # the carry clears the highest rim seen
        if widths:
            b["outer_size"] = [width, depth_m]
        ctx.planner.__dict__.pop("_reach", None)                      # which arm reaches which bin: worked out for the old bins
    return changed
