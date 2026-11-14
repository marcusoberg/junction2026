"""Run one step of the organisers' setup_instance.sh, working around conflicts in their constraints file.

Their aaltonomous/cloud/isaac-constraints.txt is a `pip freeze` of a working machine, not a consistent set: e.g. it
pins click==8.5.1 while isaacsim-kernel 6.2.0.0 requires click==8.3.3, so `setup_instance.sh isaacsim rest` fails
with ResolutionImpossible on a fresh install. When pip reports such a conflict we drop just that one constraint
line (pip then installs the version the package itself requires) and run the step again. Every line dropped is
recorded in /opt/aalto/constraints_removed.txt.

    python retry_setup.py isaacsim rest
"""
import re
import subprocess
import sys
from pathlib import Path

APP = Path("/opt/aalto")
CONSTRAINTS = APP / "aaltonomous/cloud/isaac-constraints.txt"
REMOVED = APP / "constraints_removed.txt"

for attempt in range(12):
    p = subprocess.Popen(["bash", "aaltonomous/cloud/setup_instance.sh", *sys.argv[1:]], cwd=APP,
                         env={**__import__("os").environ, "AALTO_NO_GPU": "1"},
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    out = []
    for line in p.stdout:
        sys.stdout.write(line)
        out.append(line)
    if p.wait() == 0:
        sys.exit(0)
    names = set(re.findall(r"The user requested \(constraint\) ([A-Za-z0-9_.\-]+)==", "".join(out[-400:])))
    if not names:
        sys.exit(p.returncode)
    norm = lambda s: re.sub(r"[-_.]+", "-", s.lower())
    keep, dropped = [], []
    for line in CONSTRAINTS.read_text().splitlines():
        (dropped if norm(line.split("==")[0].strip()) in {norm(n) for n in names} and "==" in line else keep).append(line)
    if not dropped:
        sys.exit(p.returncode)
    CONSTRAINTS.write_text("\n".join(keep) + "\n")
    with open(REMOVED, "a") as f:
        f.write(f"step {' '.join(sys.argv[1:])}: {dropped}\n")
    print(f"\n[retry_setup] constraint conflict: dropped {dropped}, retrying (attempt {attempt + 2})\n", flush=True)
sys.exit(1)
