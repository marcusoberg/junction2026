"""The organisers' Isaac sim (Aalto-Sort-G129-Dex3-Joint) on Modal.

Built and run the way the organisers do it, so scores are comparable:
  * image   = their `aaltonomous/cloud/setup_instance.sh`, unmodified, same pinned versions, same step order as
              their `cloud/Dockerfile` (Isaac Sim 6.2, Isaac Lab 3.0, unitree_sim_isaaclab, CycloneDDS, ...).
              One forced deviation: their constraints file conflicts with Isaac Sim's own pins, so conflicting
              lines are dropped during the build (retry_setup.py; the list is printed at the end of the build)
  * running = only their CLI: `aalto sim`, `aalto check`, `aalto snap`, `aalto baseline`, `aalto eval ...`
              (`aalto eval` is `aaltonomous/sim/evaluate.py` with its default gate: 0 clips, >= 27 Hz, 120 s)
Known differences from their setup: Modal's sandbox instead of `docker run --network host --ipc host`, an L40S
instead of their RTX 5090, and the last build step skips their `verify` (one of its unit tests fails on the current
repo: test_scene.TestHeadCameraView, see ORGANIZER_ISSUES.md A1). The held-out sim variant is theirs alone.

By building/running this you accept the NVIDIA Omniverse / Isaac Sim licence (their scripts set
OMNI_KIT_ACCEPT_EULA=YES).

    modal run sim/modal_sim.py --action boot                      # start sim, scene checks, camera snapshots
    modal run sim/modal_sim.py --action baseline --trials 3       # their oracle baseline, rung 1
    modal run sim/modal_sim.py --action eval-model --ckpt pi05-pickcube-v1/checkpoints/002000 --trials 2
                                                   # scores + a head-camera video of every trial
    modal run sim/modal_sim.py --action eval-skills --trials 5 --seed 1000     # the skill policy in agent/ (camera)
    modal run sim/modal_sim.py --action eval-agent --trials 2 --seed 1000      # Gemini drives the skills (secret `gemini`)
    modal run sim/modal_sim.py --action eval-adk --trials 2 --seed 1000        # the ADK agent in adk_agents/ (key from its .env)
    modal run sim/modal_sim.py --action eval-adk --trials 5 --seed 2000 --variant "--colors real --dr wide" --dr wide
                                                   # ... on a varied scene: see sim/scene_variant.py
    modal run --detach sim/modal_sim.py --action sweep --tag s1 --sweep "nominal:sort_fixed:1000-1004, venue:sort_fixed:2000-2004"
                                                   # FAST: one app, every seed in its own sim at once, one table (run_sweep)
    modal run sim/modal_sim.py --action grasp-bench --bench "--shapes triangle --yaws=-60,-30,0,30,60" --tag gb1
                                                   # single-object grasp trials of our tools, scripted, no model calls
    modal run sim/modal_sim.py --action sh --script "aalto sim && aalto eval --policy oracle --task sort_language --trials 2"
Demonstrations for training: the oracle sorts, we record every tick (sim/gen_demos.py), then convert to a LeRobot
dataset in the format of the real-robot drop (sim/to_dataset.py). --shards sims run in parallel, one L40S each;
an oracle episode takes about 2.5 minutes at this sim's speed.
    modal run --detach sim/modal_sim.py --action datagen --name sim_fixed_v1 --shards 8 --episodes 12 --dr mix
    modal run sim/modal_sim.py --action convert --name sim_fixed_v1          # only redo the conversion
    modal run sim/modal_sim.py --action convert --name sim_fixed_v1 --tag sim_fixed_v1a   # ... into another dataset,
                                              # e.g. the episodes recorded so far while the sims are still running
    modal run training/modal_pi05.py --action prepare --src sim_fixed_v1 --name sim_fixed_v1     # then train on it
  --gen-args "--policy skills-random" records our skill library (agent/) with random object order and approach
  parameters instead of the oracle (more varied demonstrations, each pick labelled with its skill arguments).
  --dr train = the organisers' training randomisation in every sim, none = nominal scene, mix = 3 of 4 sims
  randomised. --mode segments (default) keeps one training episode per successful pick-and-place and leaves the
  oracle's failed grasps out; --mode episodes keeps whole episodes. --eval-task sort_language records rung-2 episodes. Seeds start at --seed (default 100000; the
  evaluator uses 1000+), shard k uses seed + 1000 k. Re-running the same command skips episodes already recorded.
  Raw episodes: aalto-data volume, sim_raw/<name>/. Dataset: aalto-data volume, raw/<name>.
Outputs (sim log, snapshots, runs/eval/...) land on the `aalto-sim-out` volume under <tag>/:
    mkdir -p sim_out && modal volume get aalto-sim-out <tag> sim_out     # the folder must exist first
"""
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
# The organisers' starter repo: a clone next to this folder, or the team repo's submodule (third_party/).
REPO = next((p for p in (HERE.parent / "aaltonomous-humanoid", HERE.parent.parent / "third_party" / "aaltonomous-humanoid")
             if (p / "aaltonomous").is_dir()), HERE.parent / "aaltonomous-humanoid")
APP_DIR = "/opt/aalto"
KITCACHE = "/root/miniforge3/envs/isaac/lib/python3.12/site-packages/isaacsim/kit/cache"
SETUP = f"cd {APP_DIR} && AALTO_NO_GPU=1 bash aaltonomous/cloud/setup_instance.sh"
RETRY = f"/usr/local/bin/python {APP_DIR}/retry_setup.py"
ENV = {"DEBIAN_FRONTEND": "noninteractive", "TZ": "Etc/UTC", "NVIDIA_VISIBLE_DEVICES": "all",
       "NVIDIA_DRIVER_CAPABILITIES": "all", "OMNI_KIT_ACCEPT_EULA": "YES", "OMNI_KIT_ALLOW_ROOT": "1",
       "PYTHONUNBUFFERED": "1", "HOME": "/root"}

app = modal.App("aalto-sim")

# Same packages and the same one-step-per-layer order as aaltonomous/cloud/Dockerfile.
image = (
    modal.Image.from_registry("ubuntu:24.04", add_python="3.12")
    .apt_install("ca-certificates", "curl", "git", "unzip", "cmake", "build-essential", "iproute2", "procps", "less",
                 "nano", "libturbojpeg", "libglu1-mesa", "libxt6", "libvulkan1", "libegl1", "libgl1", "libxrandr2",
                 "libxi6", "libsm6", "libice6", "libxkbcommon0", "libglib2.0-0t64", "libgomp1", "vulkan-tools")
    .env(ENV)
    .add_local_file(REPO / "aaltonomous/cloud/setup_instance.sh", f"{APP_DIR}/aaltonomous/cloud/setup_instance.sh", copy=True)
    .add_local_file(REPO / "aaltonomous/cloud/isaac-constraints.txt", f"{APP_DIR}/aaltonomous/cloud/isaac-constraints.txt", copy=True)
    .run_commands(f"{SETUP} prereq && {SETUP} conda")
    .run_commands(f"{SETUP} torch")
    .run_commands(f"{SETUP} isaacsim kit")
    # From here on each step goes through retry_setup.py: their constraints file conflicts with Isaac Sim's own
    # pins (click), so the unmodified script cannot finish a fresh install. See retry_setup.py.
    .add_local_file(HERE / "retry_setup.py", f"{APP_DIR}/retry_setup.py", copy=True)
    .run_commands(f"{RETRY} isaacsim rest")
    .run_commands(f"{RETRY} isaaclab")
    .run_commands(f"{RETRY} cyclonedds && {RETRY} repos")
    .run_commands(f"{RETRY} assets")
    .run_commands(f"cp {APP_DIR}/aaltonomous/cloud/isaac-constraints.txt /opt/constraints_used.txt")
    .add_local_dir(REPO / "aaltonomous", f"{APP_DIR}/aaltonomous", copy=True, ignore=["**/__pycache__", "**/*.pyc"])
    .run_commands(f"cp /opt/constraints_used.txt {APP_DIR}/aaltonomous/cloud/isaac-constraints.txt && {RETRY} aalto"
                  f" && ln -sf {APP_DIR}/aaltonomous/cloud/aalto.sh /usr/local/bin/aalto"
                  f" && (cat {APP_DIR}/constraints_removed.txt 2>/dev/null || true)"
                  f" && cp -a {KITCACHE} /opt/kitcache_seed 2>/dev/null || mkdir -p /opt/kitcache_seed")
    # Our policy server gets its own environment, with the same packages as submission/Dockerfile, so it runs
    # next to the sim the way a submission container would (the sim's own env is left exactly as they built it).
    .run_commands("/usr/local/bin/python -m venv /opt/policy && /opt/policy/bin/pip install --no-cache-dir "
                  "'lerobot[pi]==0.7.0' msgpack 'websockets>=12' pin pyyaml")
    # The ADK agent's environment, with the versions it was tested with. No pyzmq: without it the policy process
    # cannot reach the sim's scene service (ground-truth object poses), whatever the code does.
    .run_commands("/usr/local/bin/python -m venv /opt/adk && /opt/adk/bin/pip install --no-cache-dir google-adk==2.14.1 "
                  "numpy==2.2.6 scipy==1.15.2 opencv-python-headless==4.10.0.84 pyyaml msgpack 'websockets>=12' pin==3.3.1")
    .add_local_file(HERE.parent / "submission/server.py", "/opt/submission/server.py")
    .add_local_file(HERE / "gen_demos.py", "/opt/sim/gen_demos.py")
    .add_local_file(HERE / "scene_variant.py", "/opt/sim/scene_variant.py")
    .add_local_file(HERE / "grasp_bench.py", "/opt/sim/grasp_bench.py")
    .add_local_dir(HERE.parent / "agent", "/opt/agent_pkg/agent", ignore=["**/__pycache__", "**/*.pyc", "**/*.md"])
    # The ADK agent (adk_agents/robot_sorter). Its .env holds the API key and must not leave the laptop: the key
    # comes from the Modal secret `gemini` instead.
    .add_local_dir(HERE.parent / "adk_agents", "/opt/adk_agents",
                   ignore=["**/__pycache__", "**/*.pyc", "**/*.md", "**/.env", "**/.adk", "**/.DS_Store"])
)
# Converting recorded episodes to a LeRobot dataset needs no GPU and no sim: a small image of its own.
convert_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg", "libgl1", "libglib2.0-0")
    .uv_pip_install("lerobot[dataset]==0.7.0")
    .env({"PYTHONUNBUFFERED": "1"})
    .add_local_file(HERE / "to_dataset.py", "/opt/sim/to_dataset.py")
)

# Their evaluator, unchanged, except that its per-trial recorder also writes the head-camera video (the runner's
# own --video flag, which evaluate.py does not expose).
EVAL_WITH_VIDEO = ("python -c 'import ssl, sys; from aaltonomous.deploy import runner as R; Rec = R.Recorder; "
                   "R.Recorder = lambda path, **kw: Rec(path, video=True); "
                   "from aaltonomous.sim import evaluate as E; sys.exit(E.main(sys.argv[1:]))'")

kit_vol = modal.Volume.from_name("aalto-kitcache", create_if_missing=True)      # RTX shader cache (first start is slow)
out_vol = modal.Volume.from_name("aalto-sim-out", create_if_missing=True)
runs_vol = modal.Volume.from_name("aalto-runs", create_if_missing=True)          # training checkpoints (read here)
data_vol = modal.Volume.from_name("aalto-data", create_if_missing=True)          # datasets (the training app reads it)
SIM_RAW, SIM_DS, RAW = Path("/data/sim_raw"), Path("/data/sim_ds"), Path("/data/raw")
SHELL_PREFIX = (f"source /root/miniforge3/etc/profile.d/conda.sh && conda activate isaac && cd {APP_DIR} "
                f"&& export PYTHONPATH={APP_DIR}:/opt/agent_pkg && ")


def bash(script, check=False, timeout=None):
    print(f"\n$ {script}", flush=True)
    return subprocess.run(["bash", "-c", SHELL_PREFIX + script], check=check, timeout=timeout).returncode


def prepare_container():
    # Same job as their cloud/entrypoint.sh: make the NVIDIA Vulkan / EGL drivers discoverable.
    subprocess.run(["bash", f"{APP_DIR}/aaltonomous/cloud/entrypoint.sh", "true"], check=False)
    # Shader cache on a volume, seeded from the image the first time (what a Docker named volume does).
    vol = Path("/kitcache")
    if not any(vol.iterdir()):
        subprocess.run(f"cp -a /opt/kitcache_seed/. {vol}/", shell=True, check=False)
    if not Path(KITCACHE).is_symlink():
        shutil.rmtree(KITCACHE, ignore_errors=True)
        os.symlink(vol, KITCACHE)
    subprocess.run("nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader; nproc; free -g | sed -n 2p;"
                   " df -h /dev/shm | tail -1; (vulkaninfo --summary 2>&1 | grep -E 'deviceName|apiVersion|driverName|ERROR|error' | head -8)",
                   shell=True)


def collect(tag):
    dst = Path("/out") / tag
    dst.mkdir(parents=True, exist_ok=True)
    for src in ("/tmp/aalto_sim.log", "/tmp/policy.log", "/tmp/aalto_snap", "/tmp/agent_trace", f"{APP_DIR}/runs"):
        if Path(src).exists():
            subprocess.run(["cp", "-a", src, str(dst)], check=False)
    kit_vol.commit()
    out_vol.commit()
    data_vol.commit()
    print(f"\noutputs: mkdir -p sim_out && modal volume get aalto-sim-out {tag} sim_out")


@app.function(image=image, gpu="L40S", cpu=12, memory=49152, timeout=4 * 3600,
              volumes={"/kitcache": kit_vol, "/out": out_vol, "/runs": runs_vol, "/data": data_vol})
def session(script: str, tag: str, dr: str = "", task: str = "Aalto-Sort-G129-Dex3-Joint", return_eval: bool = False,
            return_path: str = ""):
    """Run a shell script inside the sim environment (the `aalto` CLI is on PATH), then save the outputs.
    return_eval: return {"rc", "trials"} with the evaluator's per-trial results instead of the exit code."""
    os.environ["TASK"] = task
    if dr and dr != "none":
        os.environ["AALTO_DR"] = dr          # e.g. train (aaltonomous/sim/dr/train.yaml)
    prepare_container()
    t0 = time.monotonic()
    over = threading.Event()

    def keep_committing():                   # recorded episodes reach the volume even if this container dies
        while not over.wait(300):
            data_vol.commit()
    threading.Thread(target=keep_committing, daemon=True).start()
    try:
        rc = bash(script)
        print(f"\nscript exit code {rc} after {time.monotonic() - t0:.0f} s")
        if rc:
            subprocess.run("echo '--- tail of /tmp/aalto_sim.log'; tail -60 /tmp/aalto_sim.log 2>/dev/null", shell=True)
    finally:
        over.set()
        bash("aalto stop", timeout=90)
        collect(tag)
    if return_path:
        p = Path(return_path)
        return {"rc": rc, "text": p.read_text() if p.exists() else ""}
    if return_eval:
        import json
        files = sorted(Path(f"{APP_DIR}/runs/eval").glob("*/eval.json"))
        return {"rc": rc, "trials": json.loads(files[-1].read_text()).get("trials", []) if files else []}
    return rc


def to_dataset(*args):
    cmd = ["python", "/opt/sim/to_dataset.py", *map(str, args)]
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


@app.function(image=convert_image, cpu=8, memory=16384, timeout=3 * 3600, volumes={"/data": data_vol})
def convert(name: str, shard: str, schema: str = "drop1", max_idle: int = 15, mode: str = "segments",
            out: str = ""):
    """One shard of raw episodes -> one LeRobot dataset (sim_ds/<name>/<shard>). Returns its episode count."""
    data_vol.reload()
    src, dst = SIM_RAW / name / shard, SIM_DS / (out or name) / shard
    if not list(src.glob("ep_*/meta.json")):
        print(f"{src}: no recorded episodes")
        return 0
    to_dataset(src, "--root", dst, "--repo-id", f"local/{shard}", "--schema", schema, "--max-idle", max_idle,
               "--mode", mode)
    if not (dst / "meta" / "info.json").exists():
        return 0
    data_vol.commit()
    return len(list(src.glob("ep_*/meta.json")))


@app.function(image=convert_image, cpu=8, memory=16384, timeout=3 * 3600, volumes={"/data": data_vol})
def merge(name: str, expect: int = 0):
    """All converted shards of <name> -> the dataset the training app reads: raw/<name> on the aalto-data volume."""
    data_vol.reload()
    shards = sorted(p.parent.parent for p in (SIM_DS / name).glob("*/meta/info.json"))
    print(f"merging {len(shards)} shards" + (f" (expected {expect})" if expect and expect != len(shards) else ""))
    if not shards:
        raise SystemExit(f"nothing converted under {SIM_DS / name}")
    to_dataset("--merge", *shards, "--root", RAW / name, "--repo-id", f"local/{name}")
    data_vol.commit()


@app.function(image=convert_image, cpu=1, timeout=8 * 3600, volumes={"/data": data_vol})
def datagen(name: str, shards: int = 1, episodes: int = 2, seed: int = 100000, eval_task: str = "sort_fixed",
            dr: str = "train", task: str = "Aalto-Sort-G129-Dex3-Joint", schema: str = "drop1", max_idle: int = 15,
            record: bool = True, mode: str = "segments", out: str = "", gen_args: str = ""):
    """Record `episodes` oracle episodes in each of `shards` parallel sims, convert each shard as soon as its sim
    is done, then merge. Runs on Modal (not in the local entrypoint) so `modal run --detach` sees it through."""
    from concurrent.futures import ThreadPoolExecutor

    def one(k):
        shard = f"shard_{k:02d}"
        variant = ("none" if k % 4 == 0 else "train") if dr == "mix" else dr
        if record:
            script = (f"aalto sim && python /opt/sim/gen_demos.py --task {eval_task} --seed {seed + 1000 * k} "
                      f"--episodes {episodes} --out {SIM_RAW / name / shard} {gen_args}")
            try:
                rc = session.remote(script, f"{name}-{shard}", variant, task)
                print(f"{shard} ({variant}): recording finished, exit code {rc}", flush=True)
            except Exception as e:           # a lost sim still leaves its finished episodes on the volume
                print(f"{shard} ({variant}): recording failed: {e!r}", flush=True)
        try:
            return convert.remote(name, shard, schema, max_idle, mode, out)
        except Exception as e:
            print(f"{shard}: conversion failed: {e!r}", flush=True)
            return 0

    if record:
        n = shards
    else:
        data_vol.reload()
        n = len(list((SIM_RAW / name).glob("shard_*")))
    with ThreadPoolExecutor(max(n, 1)) as pool:
        counts = list(pool.map(one, range(n)))
    print(f"episodes per shard: {counts}", flush=True)
    if not sum(counts):
        raise SystemExit("no episodes were recorded")
    name = out or name
    merge.remote(name, sum(c > 0 for c in counts))
    print(f"\ndataset: aalto-data volume, raw/{name}\n"
          f"next:    modal run training/modal_pi05.py --action prepare --src {name} --name {name}")


# Named scenes for --action sweep: (arguments for sim/scene_variant.py, the sim's randomisation config).
SCENES = {
    "nominal": ("", ""),
    "nominal-dr": ("", "train"),                                            # the organisers' training randomisation
    "venue": ("--colors venue", "train"),                                   # the real colours as the robot's camera sees them
    "venue-wide": ("--colors venue --placement wide", "train"),
    "venue-shape": ("--colors venue --placement wide --instructions shape", "train"),
    "pastel": ("--colors pastel --dr wide --placement wide", "wide"),
    "real-wide": ("--colors real --dr wide --placement wide", "wide"),      # orange / yellow can be ambiguous here
}


def run_sweep(spec, tag, task, video, server_env, adk_script_for):
    """One app, every (scene, task, seed) in its own sim container at the same time; one table at the end.

        modal run --detach sim/modal_sim.py --action sweep --tag s1 \\
            --sweep "nominal:sort_fixed:1000-1004, venue:sort_fixed:2000-2004, venue-shape:sort_language:4000-4004"

    Wall time is about one sim start plus the longest episode (8-10 min) however many seeds there are, up to the
    workspace's GPU limit. Outputs: aalto-sim-out volume, <tag>/<scene>-<task>-<seed>/ (traces, videos, logs)."""
    env_file = HERE.parent / "adk_agents" / "robot_sorter" / ".env"
    pairs = [line.split("=", 1) for line in env_file.read_text().splitlines()
             if "=" in line and not line.lstrip().startswith("#")]
    fn = session.with_options(secrets=[modal.Secret.from_dict({k.strip(): v.strip().strip("'\"") for k, v in pairs})])
    jobs = []
    for part in filter(None, (p.strip() for p in spec.split(","))):
        scene, eval_task, seeds = part.split(":")
        lo, _, hi = seeds.partition("-")
        variant, dr = SCENES[scene]
        for seed in range(int(lo), int(hi or lo) + 1):
            # always: with no arguments the script puts the nominal scene back (the container may be a reused one)
            script = f"python /opt/sim/scene_variant.py {variant} && " + adk_script_for(eval_task, 1, seed)
            jobs.append(((scene, eval_task, seed), (script, f"{tag}/{scene}-{eval_task}-{seed}", dr, task, True)))
    print(f"sweep {tag}: {len(jobs)} episodes, each in its own sim")
    rows, t0 = [], time.monotonic()
    for (scene, eval_task, seed), res in zip([j[0] for j in jobs], fn.starmap([j[1] for j in jobs], return_exceptions=True)):
        trial = (res.get("trials") or [{}])[0] if isinstance(res, dict) else {}
        out = [o["outcome"] for o in trial.get("outcomes", [])]
        rows.append((scene, eval_task, seed, out, trial.get("score"), trial.get("gate_ok"), "" if isinstance(res, dict) else repr(res)[:60]))
    print(f"\nsweep {tag} finished in {(time.monotonic() - t0) / 60:.1f} min\n")
    print(f"{'scene':14s}{'task':15s}{'seed':>6s}  correct  wrong  left  score  gate")
    totals = {}
    for scene, eval_task, seed, out, score, gate, err in rows:
        c, w, n = out.count("correct"), out.count("wrong") + out.count("off"), len(out)
        print(f"{scene:14s}{eval_task:15s}{seed:6d}  {c:3d}/{n:<3d}  {w:5d}  {n - c - w:4d}  {str(score):>5s}  {'ok' if gate else 'FAIL' if gate is False else '?'}  {err}")
        t = totals.setdefault((scene, eval_task), [0, 0, 0])
        t[0] += c; t[1] += w; t[2] += n
    print()
    for (scene, eval_task), (c, w, n) in totals.items():
        print(f"TOTAL {scene:14s}{eval_task:15s} {c}/{n} correct ({100 * c / max(n, 1):.0f} %), {w} wrong bin")
    print(f"\noutputs: mkdir -p sim_out && modal volume get aalto-sim-out {tag} sim_out")


@app.local_entrypoint()
def main(action: str = "boot", script: str = "", trials: int = 3, dr: str = "", tag: str = "",
         task: str = "Aalto-Sort-G129-Dex3-Joint", ckpt: str = "", eval_task: str = "sort_fixed", video: bool = True,
         name: str = "", shards: int = 1, episodes: int = 2, seed: int = 100000, schema: str = "drop1",
         max_idle: int = 15, mode: str = "segments", background: bool = False, server_env: str = "",
         gen_args: str = "", variant: str = "", sweep: str = "", bench: str = ""):
    # --background (with `modal run --detach`): start the job and return, so it does not depend on this computer
    # staying awake and online. Results land on the volumes as usual.
    if action in ("datagen", "convert"):
        if not name:
            raise SystemExit(f"--action {action} needs --name <dataset name>")
        args = (name, shards, episodes, seed, eval_task, dr or "train", task, schema, max_idle)
        kw = dict(record=action == "datagen", mode=mode, out=tag if action == "convert" else "", gen_args=gen_args)
        if background:
            print(f"started {action} {name} in the background (call {datagen.spawn(*args, **kw).object_id})")
        else:
            datagen.remote(*args, **kw)
        return
    # eval-model: our policy server on :8000 (as the submission container would be), scored by their evaluator
    # with its default gate. With --video every scored trial is also saved as a head-camera video
    # (<tag>/runs/eval/<run>/episode_NNN.mp4).
    # --server-env "AALTO_SAMPLES=8 AALTO_EXEC_TICKS=20": settings for our policy server (see submission/server.py)
    model = (f"({server_env} AALTO_CKPT=/runs/{ckpt}/pretrained_model PORT=8000 /opt/policy/bin/python /opt/submission/server.py "
             f"> /tmp/policy.log 2>&1 &); aalto sim && "
             f"for i in $(seq 150); do grep -q 'model ready\\|exiting\\|Error' /tmp/policy.log && break; sleep 2; done; "
             f"grep 'submission' /tmp/policy.log | tail -4; "
             + (EVAL_WITH_VIDEO if video else "aalto eval")
             + f" --policy ws://localhost:8000 --team model --task {eval_task} --trials {trials}; rc=$?; "
             + "grep 'episode end' /tmp/policy.log | tail -3; exit $rc")
    # eval-skills: our skill policy (agent/runtime/server.py) in the sim's own environment, scored the same way.
    #   --server-env "AGENT_PERCEPTION=sim" for exact poses instead of the head camera, "AGENT_CHOOSER=random", ...
    skills = (f"({server_env} PYTHONPATH={APP_DIR}:/opt/agent_pkg python -m agent.runtime.server > /tmp/policy.log 2>&1 &); "
              f"aalto sim && sleep 10; grep ' ready' /tmp/policy.log; "
              + (EVAL_WITH_VIDEO if video else "aalto eval")
              + f" --policy ws://localhost:8000 --team skills --task {eval_task} --trials {trials} --seed {seed}; rc=$?; "
              + "grep 'episode end\\|Traceback\\|Error' /tmp/policy.log | tail -8; exit $rc")
    # eval-agent: the Gemini-driven loop (agent/runtime/agent_policy.py). Needs a Modal secret named `gemini`
    # holding GEMINI_API_KEY. It runs in the policy venv (google-genai is installed there at start).
    agent_script = ("/opt/policy/bin/pip install -q google-genai 2>&1 | tail -1; "
                    f"(AGENT_MODE=gemini {server_env} PYTHONPATH={APP_DIR}:/opt/agent_pkg /opt/policy/bin/python "
                    f"-m agent.runtime.server > /tmp/policy.log 2>&1 &); aalto sim && sleep 15; grep ' ready\\|Error' /tmp/policy.log; "
                    + (EVAL_WITH_VIDEO if video else "aalto eval")
                    + f" --policy ws://localhost:8000 --team agent --task {eval_task} --trials {trials} --seed {seed}; rc=$?; "
                    + "grep 'agent t=\\|episode end\\|Traceback\\|Error' /tmp/policy.log | tail -60; exit $rc")
    # eval-adk: the Google ADK agent (adk_agents/robot_sorter/serve.py) as the policy server. The API key and the
    # model are taken from adk_agents/robot_sorter/.env on this computer when the run is started (the file itself is
    # not uploaded). ADK has an environment of its own in the image (/opt/adk), with the
    # versions the agent was tested with; nothing in the image or the sim's environment changes.
    #   --server-env "ROBOT_AGENT_MODEL=gemini-4-argon"   another model (default gemini-3.8-flash)
    # pyzmq is deliberately NOT installed there: without it the policy process cannot talk to the sim's scene
    # service (the only source of ground-truth object poses), whatever the code does.
    #   --variant "--colors real --dr wide --placement wide --instructions shape" --dr wide
    #       edit the container's scene before the sim starts (sim/scene_variant.py): other object colours, shades,
    #       placements, shape-based instructions. With "--dr wide" in the variant also pass --dr wide.
    def adk_script_for(eval_task, trials, seed):
        return (
                      # a container is reused for the next job of a sweep: no server, trace or results left over
                      f"for p in $(pgrep -f robot_sorter); do grep -q '^python' /proc/$p/comm 2>/dev/null && kill $p; done; rm -rf /tmp/agent_trace /tmp/policy.log {APP_DIR}/runs/eval; sleep 1; "
                      f"(ROBOT_TRACE_DIR=/tmp/agent_trace {server_env} PYTHONPATH={APP_DIR}:/opt/agent_pkg:/opt/adk_agents /opt/adk/bin/python "
                      f"-m robot_sorter.serve > /tmp/policy.log 2>&1 &); aalto sim && "
                      f"for i in $(seq 60); do grep -q ' ready\\|Traceback\\|Error' /tmp/policy.log && break; sleep 2; done; "
                      f"grep ' ready\\|Traceback\\|Error' /tmp/policy.log | tail -5; "
                      + (EVAL_WITH_VIDEO if video else "aalto eval")
                      + f" --policy ws://localhost:8000 --team adk --task {eval_task} --trials {trials} --seed {seed}; rc=$?; "
                      + "grep 'agent\\|robot t=\\|Traceback\\|Error' /tmp/policy.log | cut -c1-400 | tail -80; exit $rc")

    adk_script = adk_script_for(eval_task, trials, seed)
    if action == "sweep":
        return run_sweep(sweep, tag or f"sweep-{time.strftime('%m%d-%H%M')}", task, video, server_env, adk_script_for)
    # grasp-bench: single-object grasp trials of OUR tools, scripted agent, no model calls (sim/grasp_bench.py).
    #   --bench "--shapes triangle --yaws=-60,-30,0,30,60 --xs 0.30,0.45,0.60"
    bench_script = (f"for p in $(pgrep -f robot_sorter); do grep -q '^python' /proc/$p/comm 2>/dev/null && kill $p; done; rm -rf /tmp/agent_trace /tmp/policy.log; python /opt/sim/scene_variant.py {variant}; "
                    f"(ROBOT_AGENT_MODEL=bench ROBOT_TRACE_DIR=/tmp/agent_trace {server_env} PYTHONPATH={APP_DIR}:/opt/agent_pkg:/opt/adk_agents "
                    f"/opt/adk/bin/python -m robot_sorter.serve > /tmp/policy.log 2>&1 &); aalto sim && "
                    f"for i in $(seq 60); do grep -q ' ready\\|Traceback' /tmp/policy.log && break; sleep 2; done; "
                    f"python /opt/sim/grasp_bench.py {bench}; rc=$?; mkdir -p {APP_DIR}/runs && cp /tmp/grasp_bench.json {APP_DIR}/runs/ 2>/dev/null; "
                    f"grep 'Traceback' -A8 /tmp/policy.log | tail -20; exit $rc")
    if action == "grasp-bench" and sweep:
        # several settings of the agent side by side, one sim each, one app:
        #   --sweep "base: ; lift6: ROBOT_TRI_OFFSET_M=0,0,0.006 ; nofit: ROBOT_HEADPOSE=0 ROBOT_DETECTOR=colour"
        import json
        configs = [(c.split(":", 1)[0].strip(), c.split(":", 1)[1].strip()) for c in sweep.split(";") if ":" in c]
        tag = tag or f"gb-{time.strftime('%m%d-%H%M')}"
        jobs = [(bench_script.replace("(ROBOT_AGENT_MODEL=bench", f"({env} ROBOT_AGENT_MODEL=bench"), f"{tag}/{name}", dr, task,
                 False, "/tmp/grasp_bench.json") for name, env in configs]
        print(f"grasp bench {tag}: {len(jobs)} settings, one sim each: {[n for n, _ in configs]}")
        print(f"\n{'setting':12s}{'env':60s} sorted  first grasp held  grasps (held / made)")
        for (name, env), res in zip(configs, session.starmap(jobs, return_exceptions=True)):
            rows = json.loads(res["text"]) if isinstance(res, dict) and res.get("text") else []
            made = sum(len(r["attempts"]) for r in rows)
            held = sum(r["attempts"].count("completed") for r in rows)
            first = sum(1 for r in rows if r["attempts"][:1] == ["completed"])
            ok = sum(1 for r in rows if r["outcome"] == "correct")
            print(f"{name:12s}{env[:58]:60s}{ok:3d}/{len(rows):<3d} {first:8d}/{len(rows):<3d}       {held:3d} / {made}"
                  + ("" if isinstance(res, dict) else f"   {res!r}"[:80]))
        print(f"\noutputs: mkdir -p sim_out && modal volume get aalto-sim-out {tag} sim_out")
        return
    scripts = {
        "boot": "aalto sim && aalto check; aalto snap",
        "grasp-bench": bench_script,
        "eval-skills": skills,
        "eval-agent": agent_script,
        "eval-adk": adk_script,
        "baseline": f"aalto sim && aalto baseline --trials {trials}",
        "eval-model": model if ckpt else "",
        "sh": script,
    }
    if action not in scripts or not scripts[action]:
        raise SystemExit("--action boot | baseline | eval-model --ckpt <run>/checkpoints/<step> | eval-skills | eval-agent | eval-adk | sh --script '...' "
                         "| datagen --name <dataset> | convert --name <dataset>")
    if variant or action == "eval-adk":      # edit (or restore) the container's copy of the scene before anything reads it
        scripts[action] = f"python /opt/sim/scene_variant.py {variant} && " + scripts[action]
    tag = tag or f"{action}-{time.strftime('%m%d-%H%M%S')}"
    if action == "eval-adk":       # the key and model come from the agent's own .env, read here and never stored
        env_file = HERE.parent / "adk_agents" / "robot_sorter" / ".env"
        pairs = [line.split("=", 1) for line in env_file.read_text().splitlines()
                 if "=" in line and not line.lstrip().startswith("#")]
        fn = session.with_options(secrets=[modal.Secret.from_dict({k.strip(): v.strip().strip("'\"") for k, v in pairs})])
    else:
        fn = session.with_options(secrets=[modal.Secret.from_name("gemini")]) if action == "eval-agent" else session
    if background:
        call = fn.spawn(scripts[action], tag, dr, task)
        print(f"started {action} in the background (call {call.object_id}); when it is done:\n"
              f"  mkdir -p sim_out && modal volume get aalto-sim-out {tag} sim_out")
        return
    rc = fn.remote(scripts[action], tag, dr, task)
    print(f"tag {tag}: exit code {rc}")
