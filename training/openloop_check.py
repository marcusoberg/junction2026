"""Open-loop check of a checkpoint: feed frames of a prepared dataset through the SUBMISSION server's model path
(submission/server.py: same image handling, normalisation and joint order as in the sim / on the robot) and compare
the predicted 50-step action chunk with the actions recorded right after that frame.

Small error on the training data = serving matches training, and what the closed loop does wrong is the policy
(ambiguity, drift, too little data). Large error = a train/serve mismatch to fix before training any longer.

    modal run training/openloop_check.py --ckpt pi05-sim-v1a/checkpoints/002500 --dataset sim_fixed_v1a
    ... --samples 8      # the server's AALTO_SAMPLES: several predictions per observation, combined
"""
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent
REPO = next((p for p in (ROOT / "aaltonomous-humanoid", ROOT.parent / "third_party" / "aaltonomous-humanoid")
             if (p / "aaltonomous").is_dir()), ROOT / "aaltonomous-humanoid")   # own clone, or the team repo's submodule
FILES = ["aaltonomous/__init__.py", "aaltonomous/deploy/__init__.py", "aaltonomous/deploy/protocol.py",
         "aaltonomous/deploy/policy_server.py", "aaltonomous/interface/__init__.py", "aaltonomous/interface/safety.py",
         "aaltonomous/interface/joint_spec.yaml", "third_party/xr_teleoperate/assets/g1/g1_body29_hand14.urdf"]

app = modal.App("aalto-openloop")
image = (modal.Image.debian_slim(python_version="3.12").apt_install("ffmpeg", "libgl1", "libglib2.0-0", "libgomp1")
         .uv_pip_install("lerobot[pi,dataset]==0.7.0", "msgpack", "websockets>=12", "pin", "pyyaml"))
for f in FILES:
    image = image.add_local_file(REPO / f, f"/app/{f}")
image = image.add_local_file(ROOT / "submission/server.py", "/app/server.py")
runs = modal.Volume.from_name("aalto-runs")
data = modal.Volume.from_name("aalto-data")


@app.function(image=image, gpu="L40S", volumes={"/runs": runs, "/data": data}, timeout=3600, cpu=8, memory=32768)
def check(ckpt: str, dataset: str, episodes: int = 12, seed: int = 0, samples: int = 1):
    import json
    import os
    import sys

    import numpy as np
    sys.path.insert(0, "/app")
    os.chdir("/app")
    import server as S
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    root = Path("/data/train") / dataset
    ds = LeRobotDataset(f"local/{dataset}", root=root)
    side = [json.loads(line) for line in (root / "meta" / "aalto_episodes.jsonl").read_text().splitlines()]
    act = np.stack([np.asarray(a, np.float32) for a in ds.hf_dataset["action"]])
    ep_idx = np.asarray(ds.hf_dataset["episode_index"])
    model = S.Pi05Model(f"/runs/{ckpt}/pretrained_model", samples=samples)
    model.warm_up()
    cams = {k: S.obs_camera(k) for k in ds.meta.features if k.startswith("observation.images.")}
    print("dataset camera -> observation key:", cams)

    rng = np.random.default_rng(seed)
    rows = []
    for e in rng.choice(ds.num_episodes, min(episodes, ds.num_episodes), replace=False):
        idx = np.flatnonzero(ep_idx == e)
        a, n = int(idx[0]), len(idx)
        for where, k in (("start", 0), ("early", 45), ("middle", n // 2), ("late", max(n - 90, 0))):
            x = ds[a + k]
            obs = {name: (x[key].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8) for key, name in cams.items()}
            st = x["observation.state"].numpy()
            obs.update(arm_q=st[:14], hand_q=st[14:], t=k / 30)
            p1, p2 = model.infer(obs, x["task"]), model.infer(obs, x["task"])
            h = min(50, n - k)
            gt = act[a + k:a + k + h]
            err = np.abs(p1[:h] - gt)
            rows.append({"where": where, "episode": int(e), "arm": side[int(e)].get("arm"), "bin": side[int(e)].get("bin"),
                         "color": side[int(e)].get("color"),
                         "arm_err": err[:, :14].mean(1), "hand_err": err[:, 14:].mean(1),
                         "arm_max": float(err[:, :14].max()),
                         "spread": float(np.abs(p1 - p2)[:, :14].mean()),
                         "first_gap": float(np.abs(p1[0, :14] - st[:14]).max()), "gt_first_gap": float(np.abs(gt[0, :14] - st[:14]).max()),
                         "pred_speed": float(np.abs(np.diff(p1[:, :14], axis=0)).max() * 30),
                         "gt_speed": float(np.abs(np.diff(gt[:, :14], axis=0)).max() * 30) if h > 1 else 0.0,
                         # which arm moves in the prediction vs in the recording (left = joints 0-6)
                         "pred_left_move": float(np.abs(p1[-1, :7] - p1[0, :7]).max()), "pred_right_move": float(np.abs(p1[-1, 7:14] - p1[0, 7:14]).max()),
                         "gt_left_move": float(np.abs(gt[-1, :7] - gt[0, :7]).max()), "gt_right_move": float(np.abs(gt[-1, 7:14] - gt[0, 7:14]).max())})
    import time
    t0 = time.monotonic()
    for _ in range(5):
        model.infer(obs, x["task"])
    print(f"\nsamples per inference {samples}: {1e3 * (time.monotonic() - t0) / 5:.0f} ms per call")
    print(f"\n{ckpt} on {dataset}: {len(rows)} frames from {len(set(r['episode'] for r in rows))} episodes "
          f"(errors in rad, mean over joints)")
    print(f"{'frame':<7} {'arm err @1':>10} {'@10':>7} {'@25':>7} {'@50':>7} {'hand @25':>9} {'worst joint':>12} "
          f"{'2-sample spread':>16} {'pred/rec max speed':>19}")
    for where in ("start", "early", "middle", "late"):
        rs = [r for r in rows if r["where"] == where]
        at = lambda key, i: np.mean([r[key][min(i, len(r[key]) - 1)] for r in rs])
        print(f"{where:<7} {at('arm_err', 0):>10.4f} {at('arm_err', 9):>7.4f} {at('arm_err', 24):>7.4f} {at('arm_err', 49):>7.4f} "
              f"{at('hand_err', 24):>9.4f} {np.mean([r['arm_max'] for r in rs]):>12.3f} {np.mean([r['spread'] for r in rs]):>16.4f} "
              f"{np.mean([r['pred_speed'] for r in rs]):>9.2f} / {np.mean([r['gt_speed'] for r in rs]):.2f}")
    print("\nat the start frame (home pose): which arm the recording moves vs the prediction (max joint change over the chunk, rad)")
    for r in [r for r in rows if r["where"] == "start"]:
        print(f"  ep {r['episode']:>3} {str(r['color']):<6} -> {str(r['bin']):<6} recorded arm {str(r['arm']):<5} "
              f"rec L/R {r['gt_left_move']:.2f}/{r['gt_right_move']:.2f}   pred L/R {r['pred_left_move']:.2f}/{r['pred_right_move']:.2f}   "
              f"spread {r['spread']:.3f}  first-step gap pred {r['first_gap']:.3f} rec {r['gt_first_gap']:.3f}")


@app.local_entrypoint()
def main(ckpt: str, dataset: str, episodes: int = 12, samples: int = 1):
    check.remote(ckpt, dataset, episodes, samples=samples)
