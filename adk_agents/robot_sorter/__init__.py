"""ADK agent that drives the G1 through oracle-based skills. `adk web adk_agents` / `adk run adk_agents/robot_sorter`.

The skills live in the project's `agent/` package and the organisers' code in `aaltonomous-humanoid/`; both are
put on sys.path here so the ADK command line needs no PYTHONPATH. (In a container, set PYTHONPATH instead.)
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
for _p in (_ROOT, _ROOT / "aaltonomous-humanoid"):
    if (_p / "agent" / "skills").is_dir() or (_p / "aaltonomous").is_dir():
        if str(_p) not in sys.path:
            sys.path.append(str(_p))

try:  # `adk web` / `adk run` read .env themselves; this covers `python -m robot_sorter.serve` and the tests.
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).with_name(".env"), override=False)
except ImportError:
    pass

from . import agent as agent  # noqa: E402  (ADK looks for robot_sorter.agent.root_agent)
