"""Fine-tune pi0.5 on the Aaltonomous Humanoid demos with LeRobot 0.7.0, on Modal.

One-time setup
    pip install modal && modal setup
    modal secret create huggingface HF_TOKEN=hf_xxx
        The token's account must have accepted the licence of google/paligemma-3b-pt-224 on Hugging Face
        (pi0.5's tokenizer is gated) and must be able to read the dataset repo if it is private.

Typical session (run from the folder that contains training/)
    # 1. pull a data drop from Hugging Face onto the data volume; prints the LeRobot dataset folders it finds
    modal run training/modal_pi05.py --action fetch --repo-id <org>/<dataset> --name drop0
    # 2. make the training copy: failed episodes dropped, head depth + arm velocity removed (RGB + state + action left)
    modal run training/modal_pi05.py --action prepare --src drop0 --name drop0
    #    drops without success labels: choose episodes yourself and (optionally) rewrite the instruction
    #    modal run training/modal_pi05.py --action prepare --src g1_pick_cube --name pick_cube_clean \
    #        --keep 2,4,5,6,7,8,9 --task "Pick up the blue cube and put it in the left bin."
    # 3. 20 optimiser steps on a GPU: proves weights, tokenizer, data and memory all work before a long run
    modal run training/modal_pi05.py --action smoke --dataset drop0
    # 4. the real run, detached so it survives closing the laptop; re-running the same command resumes it
    modal run --detach training/modal_pi05.py --action train --dataset drop0 --run pi05-drop0 --steps 20000
    #    add --background (with --detach) so the run does not depend on this computer staying awake and online;
    #    without it, a closed laptop lid cancels the run (it resumes from the last checkpoint when re-run)
    # 5. list checkpoints, then download one for the submission image
    modal run training/modal_pi05.py --action ls --run pi05-drop0
    modal volume get aalto-runs pi05-drop0/checkpoints/020000/pretrained_model ./pi05-drop0-020000

Other actions
    merge     --names drop0,sim_a --name mix1          one dataset out of several prepared ones (real + sim)
    info      --dataset drop0                           episodes, frames, instructions of a prepared dataset
    A local dataset can be uploaded instead of fetched:  modal volume put aalto-data ./my_ds raw/my_ds

Modes (--mode)
    expert  (default) VLM and vision encoder frozen, only the action expert trains (693M of 4.1B parameters).
            Default: H100, batch 32.
    full    everything trains. Default: A100-80GB, batch 32. NOT measured: run the smoke action first.

Measured 2026-11-14 on Modal, expert mode, synthetic 3-camera data (the smoke action prints these numbers):
    L40S  batch 16   2.6 s/update, 21 GB peak   -> ~6 samples/s
    H100  batch 32   1.65 s/update, 29 GB peak  -> ~19 samples/s on the GPU, but only ~12 with 8 data workers
                     (video decoding was the bottleneck), hence the 20 workers / 24 CPUs default now
    Gradient checkpointing must stay on: without it batch 16 ran out of memory on the L40S and batch 32 on the H100.
    B200  batch 32   1.02 s/update, 26 GB peak  -> ~31 samples/s (1.6x the H100)
    B200  batch 32, --mode full (whole model trains)   1.46 s/update, 49 GB peak -> ~22 samples/s
    B200  batch 32, --no-grad-ckpt   0.58 s/update, 117 GB peak of 183 -> ~55 samples/s (2.9x the H100). Fastest
                     option measured:  --gpu B200 --no-grad-ckpt --workers 40   (more workers: 55 samples/s of video)
    So 10,000 steps at batch 32 is roughly 5-7.5 hours on an H100; 20,000 steps roughly 9.5-15 hours.

The dataset this expects is the organisers' LeRobot **v3.0** copy (the one with head depth), not the `_groot` copy.
"""
import json
import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path

import modal

LEROBOT = "lerobot[pi,dataset,training]==0.7.0"        # same version the organisers write their datasets with
BASE_MODEL = "lerobot/pi05_base"             # public, ~14.5 GB
DATA, RUNS, HF_HOME = Path("/data"), Path("/runs"), "/hf"
RAW, TRAIN = DATA / "raw", DATA / "train"
SIDECAR = "aalto_episodes.jsonl"             # organisers' per-episode labels (success, layout, notes, ...)

MODES = {
    #          flags for lerobot-train                                             default gpu     batch
    "expert": ({"train_expert_only": "true", "freeze_vision_encoder": "true"},     "H100",         32),
    "full":   ({"train_expert_only": "false", "freeze_vision_encoder": "false"},   "A100-80GB",    32),
}

app = modal.App("aalto-pi05")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg", "git", "libgl1", "libglib2.0-0")
    .uv_pip_install(LEROBOT)
    .env({"HF_HOME": HF_HOME, "HF_LEROBOT_HOME": str(DATA / "lerobot_home"), "TOKENIZERS_PARALLELISM": "false",
          "PYTHONUNBUFFERED": "1"})
)

data_vol = modal.Volume.from_name("aalto-data", create_if_missing=True)
runs_vol = modal.Volume.from_name("aalto-runs", create_if_missing=True)
hf_vol = modal.Volume.from_name("aalto-hf-cache", create_if_missing=True)
VOLUMES = {str(DATA): data_vol, str(RUNS): runs_vol, HF_HOME: hf_vol}
SECRETS = [modal.Secret.from_name("huggingface")]


# ---------------------------------------------------------------------------- plain helpers (also run locally in tests)
def sh(cmd: list[str]) -> None:
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def edit_dataset(src: Path, dst: Path, *operation: str) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    sh([sys.executable, "-m", "lerobot.scripts.lerobot_edit_dataset",
        "--repo_id", f"local/{src.name}", "--root", str(src),
        "--new_repo_id", f"local/{dst.name}", "--new_root", str(dst), *operation])


def find_datasets(root: Path) -> list[tuple[Path, str]]:
    """Every LeRobot dataset folder under root, with its format version."""
    out = []
    for info in sorted(root.rglob("meta/info.json")):
        out.append((info.parent.parent, json.loads(info.read_text()).get("codebase_version", "?")))
    return out


def read_sidecar(root: Path) -> list[dict]:
    p = root / "meta" / SIDECAR
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()] if p.exists() else []


def extra_features(info: dict) -> list[str]:
    """Features pi0.5 must not see: depth videos (it would take them as a 4th camera) and extra state vectors."""
    drop = []
    for key, ft in info["features"].items():
        is_depth = (ft.get("info") or {}).get("is_depth_map") or (ft["dtype"] == "video" and ft["shape"][-1] == 1)
        if is_depth or (key.startswith("observation.") and ft["dtype"] != "video" and key != "observation.state"):
            drop.append(key)
    return drop


def prepare_dataset(src: Path, dst: Path, keep_failures: bool = False, keep: str = "", task: str = "") -> dict:
    """src: a LeRobot v3.0 dataset as shipped. dst: the copy lerobot-train reads with no extra flags.
    keep: comma-separated episode indices to keep (our own labels, for drops that ship without success labels).
    task: replace every episode's instruction with this text."""
    info = json.loads((src / "meta" / "info.json").read_text())
    if not str(info.get("codebase_version", "")).startswith("v3"):
        raise SystemExit(f"{src} is LeRobot {info.get('codebase_version')}; this needs the v3.0 copy (not *_groot)")
    side = read_sidecar(src)
    if keep:
        wanted = {int(e) for e in keep.split(",")}
        unknown = sorted(wanted - set(range(info["total_episodes"])))
        if unknown:
            raise SystemExit(f"--keep names episodes that do not exist: {unknown}")
        failed = [e for e in range(info["total_episodes"]) if e not in wanted]
    else:
        failed = [] if keep_failures else [s["episode_index"] for s in side if s.get("success") is not True]
    if side and len(failed) == len(side):
        raise SystemExit("no successful episodes in this dataset (pass --keep-failures to train on everything)")
    drop = extra_features(info)
    tmp = dst.with_name(dst.name + "_tmp")
    cur = src
    # Order matters: in lerobot 0.7.0 delete_episodes fails on a dataset that remove_feature has rewritten.
    if failed:
        edit_dataset(cur, tmp, "--operation.type", "delete_episodes", "--operation.episode_indices", json.dumps(failed))
        cur = tmp
    if drop:
        edit_dataset(cur, dst, "--operation.type", "remove_feature", "--operation.feature_names", json.dumps(drop))
    elif cur == tmp:
        shutil.rmtree(dst, ignore_errors=True)
        tmp.rename(dst)
    else:
        shutil.rmtree(dst, ignore_errors=True)
        shutil.copytree(src, dst)
    shutil.rmtree(tmp, ignore_errors=True)
    if task:                                   # rewrites the task table in place
        sh([sys.executable, "-m", "lerobot.scripts.lerobot_edit_dataset", "--repo_id", f"local/{dst.name}",
            "--root", str(dst), "--operation.type", "modify_tasks", "--operation.new_task", task])
    kept_src = [e for e in range(info["total_episodes"]) if e not in failed]
    (dst / "meta" / "source_episodes.json").write_text(json.dumps({"source": str(src), "kept_source_episodes": kept_src}))
    kept = [s for s in side if s["episode_index"] not in failed]          # delete_episodes renumbers 0..n-1 in order
    if kept:
        with open(dst / "meta" / SIDECAR, "w") as f:
            for new_idx, s in enumerate(kept):
                f.write(json.dumps({**s, "episode_index": new_idx, "source_episode_index": s["episode_index"]}) + "\n")
    return {"dropped_episodes": failed, "dropped_features": drop, **describe(dst)}


def describe(root: Path) -> dict:
    info = json.loads((root / "meta" / "info.json").read_text())
    side = read_sidecar(root)
    tasks: dict[str, int] = {}
    for s in side:
        tasks[s.get("task") or "?"] = tasks.get(s.get("task") or "?", 0) + 1
    return {"root": str(root), "version": info.get("codebase_version"), "episodes": info["total_episodes"],
            "frames": info["total_frames"], "minutes": round(info["total_frames"] / info["fps"] / 60, 1),
            "features": [k for k in info["features"] if k.startswith(("observation", "action"))],
            "episodes_per_task": tasks or None}


def latest_checkpoint(run_dir: Path) -> Path | None:
    steps = [p for p in (run_dir / "checkpoints").glob("[0-9]*") if (p / "pretrained_model").is_dir()]
    return max(steps, key=lambda p: int(p.name)) if steps else None


def train_command(dataset_root: Path, run_dir: Path, mode: str, steps: int, batch_size: int, save_freq: int,
                  augment: bool, rtc_delay: int, episodes: str, seed: int, num_workers: int,
                  device: str = "cuda", fresh: bool = False, log_freq: int = 50, grad_ckpt: bool = True,
                  init: str = "") -> list[str]:
    exe = [sys.executable, "-m", "lerobot.scripts.lerobot_train"]
    if fresh:
        shutil.rmtree(run_dir, ignore_errors=True)
    ckpt = latest_checkpoint(run_dir)
    if ckpt is not None:                      # same command again = resume (also after a preemption retry)
        print(f"resuming {run_dir.name} from {ckpt.name}", flush=True)
        return exe + [f"--config_path={ckpt / 'pretrained_model' / 'train_config.json'}", "--resume=true",
                      f"--steps={steps}"]
    if run_dir.exists():                      # a run that died before its first checkpoint: start clean
        shutil.rmtree(run_dir)
    flags = {
        "dataset.repo_id": f"local/{dataset_root.name}", "dataset.root": str(dataset_root),
        "dataset.image_transforms.enable": str(augment).lower(),
        # init: start from one of our own checkpoints (<run>/checkpoints/<step>) instead of the base model
        "policy.type": "pi05",
        "policy.pretrained_path": str(RUNS / init / "pretrained_model") if init else BASE_MODEL,
        # --policy.pretrained_path loads weights only, so the chunking is stated here: 50 steps = 1.67 s at 30 Hz
        "policy.chunk_size": 50, "policy.n_action_steps": 50, "policy.rtc_training_max_delay": rtc_delay,
        "policy.dtype": "bfloat16", "policy.gradient_checkpointing": str(grad_ckpt).lower(),
        "policy.device": device, "policy.push_to_hub": "false",
        **{f"policy.{k}": v for k, v in MODES[mode][0].items()},
        "batch_size": batch_size, "steps": steps, "save_freq": save_freq, "log_freq": log_freq,
        "num_workers": num_workers, "seed": seed, "output_dir": str(run_dir), "job_name": run_dir.name,
    }
    if episodes:
        flags["dataset.episodes"] = json.dumps([int(e) for e in episodes.split(",")])
    return exe + [f"--{k}={v}" for k, v in flags.items()]


# ---------------------------------------------------------------------------- Modal functions
@app.function(image=image, volumes=VOLUMES, secrets=SECRETS, timeout=3 * 3600, cpu=4, memory=8192)
def fetch(repo_id: str, name: str, revision: str = ""):
    from huggingface_hub import snapshot_download

    dst = RAW / name
    snapshot_download(repo_id, repo_type="dataset", local_dir=str(dst), revision=revision or None)
    data_vol.commit()
    found = find_datasets(dst)
    print(f"\n{repo_id} -> {dst}")
    for root, version in found:
        rel = root.relative_to(RAW)
        note = "  <- use this as --src" if version.startswith("v3") else "  (GR00T / v2.1 copy: not for this app)"
        print(f"  {rel}   LeRobot {version}{note}")
    if not found:
        print("  no meta/info.json found: is this a LeRobot dataset repo?")


@app.function(image=image, volumes=VOLUMES, timeout=6 * 3600, cpu=8, memory=16384)
def prepare(src: str, name: str, keep_failures: bool = False, keep: str = "", task: str = ""):
    data_vol.reload()
    root = RAW / src
    if not root.exists():
        have = sorted(p.name for p in RAW.glob("*")) if RAW.exists() else []
        raise SystemExit(f"nothing at {root}: run --action fetch first (or `modal volume put aalto-data <dir> raw/{src}`). "
                         f"On the volume now: {have or 'nothing'}")
    if not (root / "meta" / "info.json").exists():
        v3 = [str(r.relative_to(RAW)) for r, v in find_datasets(root) if v.startswith("v3")]
        raise SystemExit(f"{root} is not a dataset folder. v3.0 datasets under it: {v3 or 'none'}")
    TRAIN.mkdir(parents=True, exist_ok=True)
    print(json.dumps(prepare_dataset(root, TRAIN / name, keep_failures, keep, task), indent=1))
    data_vol.commit()


@app.function(image=image, volumes=VOLUMES, timeout=6 * 3600, cpu=8, memory=16384)
def merge(names: str, name: str):
    data_vol.reload()
    roots = [TRAIN / n.strip() for n in names.split(",")]
    dst = TRAIN / name
    shutil.rmtree(dst, ignore_errors=True)
    sh([sys.executable, "-m", "lerobot.scripts.lerobot_edit_dataset", "--new_repo_id", f"local/{name}",
        "--new_root", str(dst), "--operation.type", "merge",
        "--operation.repo_ids", json.dumps([f"local/{r.name}" for r in roots]),
        "--operation.roots", json.dumps([str(r) for r in roots])])
    side, offset = [], 0                       # merge keeps input order: re-index the sidecars the same way
    for r in roots:
        rows = read_sidecar(r)
        side += [{**s, "episode_index": offset + s["episode_index"], "source_dataset": r.name} for s in rows]
        offset += json.loads((r / "meta" / "info.json").read_text())["total_episodes"]
    with open(dst / "meta" / SIDECAR, "w") as f:
        f.writelines(json.dumps(s) + "\n" for s in side)
    print(json.dumps(describe(dst), indent=1))
    data_vol.commit()


@app.function(image=image, volumes=VOLUMES, timeout=600)
def info(dataset: str = "", run: str = ""):
    data_vol.reload()
    runs_vol.reload()
    if dataset:
        d = describe(TRAIN / dataset)
        tasks = sorted({s.get("instruction") or "" for s in read_sidecar(TRAIN / dataset)})
        print(json.dumps({**d, "instructions": tasks[:20]}, indent=1))
    if run:
        ckpts = sorted((p.name for p in (RUNS / run / "checkpoints").glob("[0-9]*")), key=int)
        print(f"{run}: checkpoints {ckpts or 'none yet'}")
        for c in ckpts[-1:]:
            print(f"  modal volume get aalto-runs {run}/checkpoints/{c}/pretrained_model ./{run}-{c}")
    if not dataset and not run:
        print("prepared datasets:", sorted(p.name for p in TRAIN.glob("*") if (p / "meta").is_dir()))
        print("runs:", sorted(p.name for p in RUNS.glob("*") if p.is_dir()))


@app.function(image=image, volumes=VOLUMES, secrets=SECRETS, gpu="L40S", cpu=12, memory=65536,
              timeout=24 * 3600, retries=modal.Retries(max_retries=2, initial_delay=10.0))
def train(dataset: str, run: str, mode: str = "expert", steps: int = 20000, batch_size: int = 16,
          save_freq: int = 2500, augment: bool = True, rtc_delay: int = 0, episodes: str = "", seed: int = 1000,
          fresh: bool = False, log_freq: int = 50, grad_ckpt: bool = True, workers: int = 8, init: str = ""):
    data_vol.reload()
    runs_vol.reload()
    root = TRAIN / dataset
    if not (root / "meta" / "info.json").exists():
        raise SystemExit(f"no prepared dataset {dataset!r}; run --action prepare first")
    subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"])
    cmd = train_command(root, RUNS / run, mode, steps, batch_size, save_freq, augment, rtc_delay, episodes, seed,
                        num_workers=workers, fresh=fresh, log_freq=log_freq, grad_ckpt=grad_ckpt, init=init)

    stop = threading.Event()
    peak = {"mem": 0, "util": []}

    def keep_committing():                     # checkpoints reach the volume even if the container is preempted
        while not stop.wait(300):
            runs_vol.commit()

    def watch_gpu():                           # peak memory + mean utilisation: is there room for a bigger batch?
        while not stop.wait(5):
            out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True).stdout.strip().split(",")
            if len(out) == 2:
                peak["mem"] = max(peak["mem"], int(out[0]))
                peak["util"].append(int(out[1]))

    threading.Thread(target=keep_committing, daemon=True).start()
    threading.Thread(target=watch_gpu, daemon=True).start()
    try:
        sh(cmd)
    finally:
        stop.set()
        runs_vol.commit()
        hf_vol.commit()
        util = peak["util"][len(peak["util"]) // 3:] or [0]      # skip model loading
        print(f"GPU: peak memory {peak['mem']} MiB, mean utilisation {sum(util) / len(util):.0f} % (after start-up)")
    ckpt = latest_checkpoint(RUNS / run)
    print(f"done. latest checkpoint: {ckpt}")
    if ckpt:
        print(f"  modal volume get aalto-runs {run}/checkpoints/{ckpt.name}/pretrained_model ./{run}-{ckpt.name}")


# ---------------------------------------------------------------------------- CLI
@app.local_entrypoint()
def main(action: str, repo_id: str = "", name: str = "", src: str = "", names: str = "", dataset: str = "",
         run: str = "", mode: str = "expert", steps: int = 20000, batch_size: int = 0, gpu: str = "",
         save_freq: int = 2500, augment: bool = True, rtc_delay: int = 0, episodes: str = "", seed: int = 1000,
         keep_failures: bool = False, revision: str = "", grad_ckpt: bool = True, workers: int = 20,
         keep: str = "", task: str = "", init: str = "", background: bool = False):
    if action == "fetch":
        fetch.remote(repo_id, name or repo_id.split("/")[-1], revision)
    elif action == "prepare":
        prepare.remote(src, name or Path(src).name, keep_failures, keep, task)
    elif action == "merge":
        merge.remote(names, name)
    elif action in ("info", "ls"):
        info.remote(dataset, run)
    elif action in ("train", "smoke"):
        if mode not in MODES:
            raise SystemExit(f"--mode must be one of {sorted(MODES)}")
        _, default_gpu, default_batch = MODES[mode]
        if action == "smoke":                  # tiny run with the real batch size, so an OOM shows up here
            run, steps, save_freq = run or f"smoke-{mode}", 40, 40
        if not (dataset and run):
            raise SystemExit("--dataset and --run are required")
        # a smoke failure is a setup problem, not a preemption: do not pay for it three times
        fn = train.with_options(gpu=gpu or default_gpu, cpu=workers + 4, memory=max(65536, 2048 * workers + 16384),
                                **({"retries": 0} if action == "smoke" else {}))
        args = (dataset, run, mode, steps, batch_size or default_batch, save_freq, augment, rtc_delay, episodes, seed)
        kw = dict(fresh=action == "smoke", log_freq=10 if action == "smoke" else 50, grad_ckpt=grad_ckpt,
                  workers=workers, init=init)
        if background:                         # with `modal run --detach`: survives this computer sleeping or
            call = fn.spawn(*args, **kw)       # losing its network (a plain .remote() call is cancelled then)
            print(f"started {run} in the background (call {call.object_id}).\n"
                  f"  progress:    modal run training/modal_pi05.py --action ls --run {run}\n"
                  f"  logs / stop: Modal dashboard, app aalto-pi05")
        else:
            fn.remote(*args, **kw)
    else:
        raise SystemExit("--action: fetch | prepare | merge | info | ls | smoke | train")
