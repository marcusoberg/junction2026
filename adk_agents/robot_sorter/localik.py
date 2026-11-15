"""Local inverse kinematics: put a hand's grasp point somewhere with the least joint movement.

The organisers' planner asks for one particular hand pose (palm straight down within 35 deg, fingers in one of a few
directions) and searches the whole joint space for it; when that pose is out of reach it has no plan. On the real
robot the bins stand 8 cm further away than in the sim and the hand stopped short of them (15 Nov).

Measured against the organisers' solver (15 Nov, both arms, 8 targets, level and leaning 14 deg):
    - joint movement from the pose in front of the robot: 2.1-2.9 rad summed over the arm against 3.1-4.3 (a third
      less); from the parked pose the two are within 3 % (their solver starts from the present joints too);
    - reach: with the orientation free the grasp point gets 3-5 mm further, no more. What decides whether a bin can
      be reached is the arm's length: 1 cm more reach for every 4 cm the release point is raised, 4.5 cm more with
      the upper body leaning 6 deg. So the tools built on this raise the release point and come nearer to the bin's
      front wall when the wanted point is out of reach; the solver does not make the arm longer.

This solver asks for less:
    - the grasp point must reach the target (that is the task);
    - the hand keeps the orientation it has now as far as it can (a soft wish: it gives way when the target is
      otherwise out of reach, e.g. the hand tilts forward to get over a far bin);
    - of all joint configurations that do this, the one nearest to the present one is taken.
It starts from the present joints and only ever moves away from them as far as the target demands, so there are
no jumps between arm configurations and the move is short.

One bounded least-squares problem per target, 7 joints:
    minimise  |p(q) - target|^2 / pos_m^2  +  |rotation from the wanted to the reached orientation|^2 / ori_rad^2
              +  |q - q_now|^2 / move_rad^2          subject to the joint limits.
"""
from __future__ import annotations

import dataclasses

import numpy as np


@dataclasses.dataclass
class Reach:
    arm_q: np.ndarray          # (14,) the full arm vector with this arm's joints replaced
    pos_err_m: float           # how far the grasp point ends from the target
    turned_deg: float          # how far the hand had to turn from the wanted orientation
    moved_rad: float           # largest change of one joint
    ok: bool                   # the target is reached (pos_err_m within tol_m)


def hand_pose(kin, side, arm_q, hand_q=None):
    """(grasp point (3,), hand rotation (3,3)) in the pelvis frame. Call with the context's lock held."""
    kin._fk(np.asarray(arm_q, float), None if hand_q is None else np.asarray(hand_q, float))
    M = kin.data.oMf[kin.palm[side]]
    R = np.asarray(M.rotation).copy()
    return np.asarray(M.translation) + R @ kin.offset[side], R


def _log3(R):
    """Rotation matrix -> rotation vector."""
    c = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    a = float(np.arccos(c))
    if a < 1e-9:
        return np.zeros(3)
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2.0 * np.sin(a)) if np.sin(a) > 1e-6 else np.zeros(3)
    return w * a


def _about(axis, angle):
    """Rotation by `angle` about the unit vector `axis`."""
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K


def palm_down(kin, side, arm_q, finger_deg=None, fold=1):
    """The hand orientation for a pinch from above that is nearest to the one the hand has now: the present
    orientation turned the shortest way until the palm faces straight down, the fingers still pointing where they
    pointed. finger_deg: point the fingers this way instead (0 = away from the robot, +90 = to its left); with
    fold = 4 any of the four directions a quarter turn apart will do (the faces of a cube) and the one nearest to
    the present direction is taken. Call with the context's lock held."""
    R = hand_pose(kin, side, arm_q)[1]
    down = np.array([0.0, 0.0, -1.0])
    a = R @ kin.axis[side]
    v, c = np.cross(a, down), float(np.clip(a @ down, -1.0, 1.0))
    s = float(np.linalg.norm(v))
    if s > 1e-9:
        R = _about(v / s, np.arctan2(s, c)) @ R
    elif c < 0:                                              # palm straight up: half a turn about the finger axis
        R = _about(R[:, 0], np.pi) @ R
    if finger_deg is not None:
        now = np.arctan2(R[1, 0], R[0, 0])                   # where the fingers point, seen from above
        turn = np.radians(finger_deg) - now
        if fold > 1:
            turn = (turn + np.pi / fold) % (2 * np.pi / fold) - np.pi / fold
        else:
            turn = (turn + np.pi) % (2 * np.pi) - np.pi
        R = _about(np.array([0.0, 0.0, 1.0]), turn) @ R
    return R


def tilt_deg(kin, side, arm_q):
    """How far the palm is from facing straight down. Call with the context's lock held."""
    a = kin.grasp_pose(side, np.asarray(arm_q, float))[1]
    return float(np.degrees(np.arccos(np.clip(-a[2], -1.0, 1.0))))


def solve(kin, side, arm_q, target, R_want=None, pos_m=0.002, ori_rad=0.5, move_rad=2.0, tol_m=0.008):
    """arm_q: (14,) where the arms are now. target: (3,) for the grasp point, pelvis frame. R_want: the orientation
    to stay close to (default: the hand's present one). The three scales say what counts as much: 2 mm of position
    error weighs like 0.5 rad of turning or 2 rad of joint movement, so position wins, then orientation, then rest.
    Call with the context's lock held (it uses the shared kinematics data)."""
    from scipy.optimize import least_squares
    arm_q = np.asarray(arm_q, float)
    sl = kin.sl(side)
    q0 = np.clip(arm_q[sl], kin.lo[sl] + 1e-6, kin.hi[sl] - 1e-6)
    target = np.asarray(target, float)
    if R_want is None:
        R_want = hand_pose(kin, side, arm_q)[1]
    full = arm_q.copy()

    def residual(x):
        full[sl] = x
        p, R = hand_pose(kin, side, full)
        return np.r_[(p - target) / pos_m, _log3(R_want.T @ R) / ori_rad, (x - q0) / move_rad]

    sol = least_squares(residual, q0, bounds=(kin.lo[sl], kin.hi[sl]), method="trf", xtol=1e-6, ftol=1e-8, max_nfev=120)
    full[sl] = sol.x
    p, R = hand_pose(kin, side, full)
    err = float(np.linalg.norm(p - target))
    return Reach(full.copy(), err, float(np.degrees(np.linalg.norm(_log3(R_want.T @ R)))),
                 float(np.abs(sol.x - arm_q[sl]).max()), err <= tol_m)


def path(kin, side, arm_q, targets, R_want=None, step_m=0.04, **kw):
    """Through several targets in turn, each from where the last one ended, in steps of at most step_m so the hand
    travels in nearly straight lines. -> [Reach, ...] one per step; stops at the first step that cannot be reached
    (its .ok is False). The orientation wish stays the one from the start, so it does not drift along the way."""
    arm_q = np.asarray(arm_q, float)
    if R_want is None:
        R_want = hand_pose(kin, side, arm_q)[1]
    out, here = [], hand_pose(kin, side, arm_q)[0]
    for target in targets:
        target = np.asarray(target, float)
        n = max(1, int(np.ceil(np.linalg.norm(target - here) / step_m)))
        for k in range(1, n + 1):
            r = solve(kin, side, arm_q, here + (target - here) * k / n, R_want, **kw)
            out.append(r)
            if not r.ok:
                return out
            arm_q = r.arm_q
        here = target
    return out
