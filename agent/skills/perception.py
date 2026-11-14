"""Perception and state tools. All positions are in the pelvis frame (x forward, y left, z up), metres.

    objs = detect_camera(ctx, obs)            # head RGB-D -> colour blobs with 3D positions (allowed in a submission)
    objs = detect_sim(ctx, scene_client)      # exact poses from the simulator (demonstration recording ONLY)
    state = joint_state(ctx, obs)

The camera detector is the organisers' classical one (HSV colour + depth). It gives colour and position; shape and
yaw come from `estimate_shape` (outline of the top face), which is a heuristic: check it on real images before
trusting it. An LLM/VLM can overwrite `WorldObject.shape` when it knows better.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from aaltonomous.baselines.classical.perception import Detection


@dataclass
class WorldObject:
    id: str                       # stable within one look: "<colour>_<n>" (camera) or the sim's object name
    color: str
    xyz: np.ndarray               # top-face centre
    shape: str | None = None      # cube | cylinder | triangle | None (unknown)
    yaw: float | None = None      # rad, orientation of the top face about z
    where: str = "counter"        # "counter" or a bin name
    source: str = "camera"
    extra: dict = field(default_factory=dict)

    def detection(self) -> Detection:
        """The planner's input type."""
        d = Detection(self.color, np.asarray(self.xyz, float), 1000, 1000, self.source, yaw=self.yaw)
        d.shape, d.name = self.shape, self.id
        return d

    def to_json(self):
        return {"id": self.id, "color": self.color, "shape": self.shape, "where": self.where,
                "position_m": [round(float(v), 3) for v in self.xyz],
                "yaw_deg": None if self.yaw is None else round(float(np.degrees(self.yaw)), 1)}


def estimate_shape(ctx, rgb, depth, obj):
    """Shape and yaw of one detected object from the outline of its colour blob in the head image, mapped onto the
    plane of its top face: 3 corners -> triangle, 4 -> cube, round -> cylinder. Returns (shape | None, yaw | None)."""
    import cv2
    uv = obj.extra.get("uv")
    if uv is None:
        return None, None
    masks = ctx.camera.masks(rgb)
    m = masks.get(obj.color)
    if m is None:
        return None, None
    n, lab = cv2.connectedComponents(m, connectivity=8)
    u0, v0 = int(round(uv[0])), int(round(uv[1]))
    h, w = m.shape
    if not (0 <= v0 < h and 0 <= u0 < w) or lab[v0, u0] == 0:
        return None, None
    blob = (lab == lab[v0, u0]).astype(np.uint8)
    cam = ctx.cal.cams[ctx.camera.cam]
    T = ctx.cal._T(ctx.camera.cam)
    R, o = T[:3, :3], T[:3, 3]
    vv, uu = np.nonzero(blob)
    rays = np.stack([(uu - cam.K[0, 2]) / cam.K[0, 0], (vv - cam.K[1, 2]) / cam.K[1, 1], np.ones(len(uu))], 1) @ R.T
    t = (obj.xyz[2] - o[2]) / rays[:, 2]                       # intersect with the top-face plane
    pts = (o + rays * t[:, None])[:, :2]
    # the side faces towards the camera are in the blob too: keep the part within one object size of the centre
    size = float(ctx.camera.size)
    pts = pts[np.linalg.norm(pts - obj.xyz[:2], axis=1) < 0.75 * size]
    if len(pts) < 30:
        return None, None
    hull = cv2.convexHull((pts * 1000).astype(np.float32))      # mm
    area, per = cv2.contourArea(hull), cv2.arcLength(hull, True)
    if area <= 0:
        return None, None
    poly = cv2.approxPolyDP(hull, 0.06 * per, True).reshape(-1, 2)
    circ = 4 * np.pi * area / per ** 2                          # 1 circle, 0.785 square, 0.605 triangle
    if len(poly) == 3 or circ < 0.70:
        shape = "triangle"
    elif circ > 0.87:
        shape = "cylinder"
    else:
        shape = "cube"
    yaw = None
    if shape == "cube" and len(poly) >= 2:
        e = max((poly[(i + 1) % len(poly)] - poly[i] for i in range(len(poly))), key=np.linalg.norm)
        yaw = float(np.arctan2(e[1], e[0]) % (np.pi / 2))
    elif shape == "triangle" and len(poly) == 3:
        c = poly.mean(0)
        apex = max(poly, key=lambda p: np.linalg.norm(p - c))
        yaw = float(np.arctan2(apex[1] - c[1], apex[0] - c[0]) - np.pi / 2)     # mesh frame: apex along +y
    return shape, yaw


def detect_camera(ctx, obs, shapes=True):
    """Objects on the counter seen by the head camera. Ids are numbered left to right per colour."""
    with ctx.lock:
        dets = ctx.camera.detect(obs["head_rgb"], obs["head_depth"])
    out, count = [], {}
    for d in sorted(dets, key=lambda d: -d.top[1]):               # +y = robot's left
        count[d.color] = count.get(d.color, 0) + 1
        o = WorldObject(f"{d.color}_{count[d.color]}", d.color, np.asarray(d.top, float), source="camera",
                        extra={"uv": tuple(d.uv), "n_px": int(d.n_px), "method": d.method})
        if shapes:
            try:
                o.shape, o.yaw = estimate_shape(ctx, obs["head_rgb"], obs["head_depth"], o)
            except Exception:
                o.shape, o.yaw = None, None
        out.append(o)
    return out


def detect_sim(ctx, client, include_binned=False):
    """Exact object poses from the simulator's scene service. Privileged: for recording demonstrations only."""
    sc, out = ctx.scene, []
    for o in client.state()["objects"]:
        if not o["active"]:
            continue
        p = np.asarray(o["pos"], float)
        m = sc.pelvis_to_mat(p)
        where = sc.bin_of(m) or ("counter" if sc.on_counter(m) else "off")
        if where != "counter" and not include_binned:
            continue
        if where == "counter" and p[2] > sc.counter_z + sc.object_height(o["shape"]) * 1.5:
            continue                                               # in a hand / in the air
        q = o.get("quat_xyzw") or [0, 0, 0, 1]
        yaw = float(np.arctan2(2 * (q[3] * q[2] + q[0] * q[1]), 1 - 2 * (q[1] ** 2 + q[2] ** 2)))
        top = p + np.array([0.0, 0.0, sc.object_height(o["shape"]) / 2])
        out.append(WorldObject(o["name"], o["color"], top, o["shape"], yaw, where, "sim"))
    return out


def joint_state(ctx, obs):
    """Arm and hand joint angles plus where each hand's grasp point is, and how closed each hand is (0 open, 1 closed)."""
    arm, hand = np.asarray(obs["arm_q"], float), np.asarray(obs["hand_q"], float)
    out = {"arm_q": arm.round(3).tolist(), "hand_q": hand.round(3).tolist()}
    with ctx.lock:
        for side in ("left", "right"):
            sl = ctx.kin.sl(side)
            p, axis = ctx.kin.grasp_pose(side, arm)
            a, b = ctx.post["open"][sl], ctx.post["closed"][sl]
            frac = float(np.clip(np.dot(hand[sl] - a, b - a) / max(np.dot(b - a, b - a), 1e-9), 0, 1))
            out[side] = {"grasp_point_m": p.round(3).tolist(), "palm_tilt_deg": round(float(np.degrees(np.arccos(
                np.clip(-axis[2], -1, 1)))), 1), "closed_fraction": round(frac, 2),
                "at_home": bool(np.abs(arm[sl] - ctx.home[sl]).max() < 0.08)}
    return out


# ------------------------------------------------------------------ for models that look at the images themselves
def encode_image(rgb, max_side=640, quality=85):
    """RGB uint8 image -> JPEG bytes (optionally downscaled), to send to a multimodal model unchanged in content."""
    import cv2
    img = np.ascontiguousarray(rgb[..., ::-1])
    h, w = img.shape[:2]
    if max(h, w) > max_side:
        s = max_side / max(h, w)
        img = cv2.resize(img, (int(round(w * s)), int(round(h * s))), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    return buf.tobytes()


def point_to_xyz(ctx, obs, point_yx, window_px=26):
    """Where a model POINTS in the head image -> a 3D grasp target in the pelvis frame.

    point_yx: [y, x] normalised to 0..1000 (the convention Gemini models use for pointing), on the head camera
    image. Uses the aligned head depth around that pixel: the highest surface near the point is taken as the
    object's top face and its centre is returned (so pointing anywhere on the object works). Without valid depth
    it falls back to the point's ray meeting the plane of an object's top face.
    Returns (xyz (3,), info dict)."""
    rgb, depth = obs["head_rgb"], obs.get("head_depth")
    h, w = rgb.shape[:2]
    v0 = float(np.clip(point_yx[0], 0, 1000)) / 1000.0 * (h - 1)
    u0 = float(np.clip(point_yx[1], 0, 1000)) / 1000.0 * (w - 1)
    cam = ctx.cal.cams[ctx.camera.cam]
    T = ctx.cal._T(ctx.camera.cam)
    R, o = T[:3, :3], T[:3, 3]
    K = cam.K
    top_z = ctx.scene.counter_z + ctx.scene.object_height()

    def ray(u, v):
        return np.array([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], 1.0]) @ R.T

    def on_plane():
        r = ray(u0, v0)
        t = (top_z - o[2]) / r[2]
        return o + t * r

    if depth is None:
        return on_plane(), {"method": "plane", "pixel": [round(u0), round(v0)]}
    ua, ub = int(max(0, u0 - window_px)), int(min(w, u0 + window_px + 1))
    va, vb = int(max(0, v0 - window_px)), int(min(h, v0 + window_px + 1))
    d = depth[va:vb, ua:ub].astype(float) * float(getattr(cam, "depth_scale", 0.001))
    vv, uu = np.mgrid[va:vb, ua:ub]
    ok = (d > 0.15) & (d < 2.0)
    if ok.sum() < 20:
        return on_plane(), {"method": "plane", "pixel": [round(u0), round(v0)]}
    pts = np.stack([(uu[ok] - K[0, 2]) / K[0, 0] * d[ok], (vv[ok] - K[1, 2]) / K[1, 1] * d[ok], d[ok]], 1) @ R.T + o
    guess = on_plane()
    near = np.linalg.norm(pts[:, :2] - guess[:2], axis=1) < 0.05          # ignore the bins / other objects
    lo, hi = ctx.scene.counter_z + 0.4 * ctx.scene.object_height(), ctx.scene.counter_z + 2.0 * ctx.scene.object_height()
    cand = pts[near & (pts[:, 2] > lo) & (pts[:, 2] < hi)]
    if len(cand) < 15:
        return guess, {"method": "plane", "pixel": [round(u0), round(v0)], "note": "nothing object-high at the point"}
    zt = np.percentile(cand[:, 2], 92)
    top = cand[cand[:, 2] > zt - 0.006]
    xyz = np.r_[top[:, :2].mean(0), zt]
    return xyz, {"method": "depth", "pixel": [round(u0), round(v0)], "top_points": int(len(top))}


def xyz_to_point(ctx, xyz, shape_hw=(480, 640)):
    """Inverse of point_to_xyz (pelvis point -> [y, x] normalised 0..1000 on the head image); for tests and overlays."""
    cam = ctx.cal.cams[ctx.camera.cam]
    T = ctx.cal._T(ctx.camera.cam)
    p = (np.asarray(xyz, float) - T[:3, 3]) @ T[:3, :3]
    u, v = cam.K[0, 0] * p[0] / p[2] + cam.K[0, 2], cam.K[1, 1] * p[1] / p[2] + cam.K[1, 2]
    return [float(v / (shape_hw[0] - 1) * 1000), float(u / (shape_hw[1] - 1) * 1000)]
