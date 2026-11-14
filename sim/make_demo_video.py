"""Turn one sim episode into a demo video: the head-camera footage on the left, the agent's loop on the right
(the instruction, every tool call Gemini made and what came back, the Decisions API labels), in sync.

    .venv/bin/python sim/make_demo_video.py sim_out/s1/nominal-sort_fixed-1001 [--episode 0] [--out demo.mp4] [--speed 2]

Needs a traced run (sim/modal_sim.py --action eval-adk or sweep): <run>/agent_trace/episode_NNN.jsonl and
<run>/runs/eval/*/episode_NNN.mp4. The trace's times are sim seconds; the video has 30 frames per sim second.
"""
import argparse
import json
import textwrap
from pathlib import Path

import cv2
import numpy as np

W_PANEL, H, FONT = 560, 480, cv2.FONT_HERSHEY_SIMPLEX
COL = {"call": (120, 220, 255), "ok": (140, 230, 140), "bad": (110, 130, 255), "info": (200, 200, 200), "head": (255, 255, 255)}


def events(trace):
    """-> instruction, [(t, kind, text)] in sim seconds."""
    out, instruction = [], ""
    for line in Path(trace).read_text().splitlines():
        r = json.loads(line)
        k, t = r["kind"], float(r.get("t", 0.0))
        if k == "reset":
            instruction = r.get("instruction", "")
        elif k == "call":
            args = ", ".join(f"{a}={v}" for a, v in r["args"].items())
            out.append((t, "call", f"> {r['tool']}({args})"))
        elif k == "result":
            res, tool = r["result"], r["tool"]
            if tool == "look":
                objs = res.get("objects_on_counter", [])
                out.append((t, "ok", f"  {len(objs)} objects on the counter"))
                for o in objs:
                    lab = o.get("gemini_label") or {}
                    out.append((t, "info", f"    {o['id']}: {lab.get('colour', '?')} {lab.get('shape', o.get('shape_guess') or '?')}"
                                           + (f" ({lab['colour_confidence']:.2f})" if lab else "")))
            elif tool == "ask_gemini":
                out.append((t, "ok", f"  -> {res.get('answer', res.get('score'))} (confidence {res.get('confidence')})"))
            else:
                status = res.get("status", "")
                kind = "ok" if status in ("completed", "ok", "episode ended") else "bad"
                extra = ""
                if res.get("grasp_check"):
                    extra = " | fingers: object in hand" if res["grasp_check"].get("object_in_hand") else " | fingers: hand is empty"
                if status == "failed":
                    extra = " | " + str(res.get("reason", ""))[:90]
                out.append((t, kind, f"  -> {status}{extra}"))
        elif k == "motion":
            out.append((float(r["t"]), "info", f"  [robot] {r['move']}"))
        elif k == "head_pose_fit":
            out.append((t, "info", f"  [camera] pose re-fitted from depth: {r.get('summary', '')[:60]}"))
        elif k == "decision_labels" and r.get("seconds"):
            out.append((t, "info", f"  [Gemini Decisions] labelled the objects in {r['seconds']} s"))
    return instruction, sorted(out, key=lambda e: e[0])


def panel(instruction, shown, t, model):
    img = np.full((H, W_PANEL, 3), 24, np.uint8)
    cv2.putText(img, f"ADK agent  |  {model}  |  t = {t:5.1f} s", (12, 22), FONT, 0.5, COL["head"], 1, cv2.LINE_AA)
    y = 44
    for ln in textwrap.wrap("Instruction: " + instruction, 74)[:3]:
        cv2.putText(img, ln, (12, y), FONT, 0.4, (180, 220, 255), 1, cv2.LINE_AA)
        y += 16
    cv2.line(img, (12, y), (W_PANEL - 12, y), (90, 90, 90), 1)
    lines = []
    for _, kind, text in shown:
        for j, ln in enumerate(textwrap.wrap(text, 76, subsequent_indent="      ")[:3]):
            lines.append((kind, ln))
    room = (H - y - 14) // 15
    for i, (kind, ln) in enumerate(lines[-room:]):
        cv2.putText(img, ln, (12, y + 18 + 15 * i), FONT, 0.38, COL[kind], 1, cv2.LINE_AA)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--episode", type=int, default=-1, help="trace episode number (default: the last one)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--speed", type=float, default=2.0, help="playback speed (the sim's footage is in sim time)")
    a = ap.parse_args()
    run = Path(a.run)
    traces = sorted((run / "agent_trace").glob("episode_*.jsonl"))
    videos = sorted(run.glob("runs/eval/*/episode_*.mp4"))
    trace = traces[a.episode]
    video = videos[a.episode if a.episode >= 0 else -1]
    first = json.loads(trace.read_text().splitlines()[0])
    instruction, evs = events(trace)
    cap = cv2.VideoCapture(str(video))
    out = Path(a.out or run / f"demo_{trace.stem}.mp4")
    wr = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (640 + W_PANEL, H))
    i, step = 0, max(1, int(round(a.speed)))
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % step == 0:
            t = i / 30.0
            wr.write(np.hstack([cv2.resize(frame, (640, H)), panel(instruction, [e for e in evs if e[0] <= t], t, first.get("model", ""))]))
        i += 1
    wr.release()
    print(f"{out}  ({i / 30.0:.0f} s of sim time at {a.speed:g}x, {len(evs)} agent events)")


if __name__ == "__main__":
    main()
