# The ADK agent (adk_agents/robot_sorter) as a submission image. Follows the organisers' template
# aaltonomous/docker/submission.Dockerfile: same base image, policy server on port 8000, started by CMD, nothing
# downloaded at start. Differences: Python 3.11 (Google ADK needs >= 3.11; the base has 3.10) and our dependencies.
#
# Build from the PROJECT ROOT (the folder that holds aaltonomous-humanoid/, agent/ and adk_agents/):
#   docker buildx build --platform linux/amd64 -f submission/adk.Dockerfile -t <registry>/<name>:<tag> --load .
# Test:  docker run --rm -p 8000:8000 <image>
#        PYTHONPATH=aaltonomous-humanoid python -m aaltonomous.deploy.replay --synthetic 100 --policy ws://localhost:8000
#
# !! The image contains adk_agents/robot_sorter/.env, i.e. the Gemini API KEY. Anyone who can pull the image can read
# !! it. Keep the image PRIVATE; before making it public, use a key you are prepared to revoke.
# No GPU is needed (the organisers start it with --gpus all, which is harmless).
FROM nvidia/cuda:12.8.0-base-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 PIP_DISABLE_PIP_VERSION_CHECK=1
# Python 3.11 from the deadsnakes PPA (a plain apt install; nothing is fetched at start), in its own environment.
RUN apt-get update && apt-get install -y --no-install-recommends software-properties-common gpg-agent libgomp1 ca-certificates \
    && add-apt-repository -y ppa:deadsnakes/ppa && apt-get update \
    && apt-get install -y --no-install-recommends python3.11 python3.11-venv \
    && rm -rf /var/lib/apt/lists/* && python3.11 -m venv /opt/venv
RUN /opt/venv/bin/pip install --no-cache-dir google-adk==2.14.1 numpy==2.2.6 scipy==1.15.2 \
    opencv-python-headless==4.10.0.84 pyyaml msgpack "websockets>=12" pin==3.3.1

WORKDIR /app
# The robot model (inverse kinematics and the planner's safety pre-check) and the organisers' package.
COPY aaltonomous-humanoid/third_party/xr_teleoperate/assets/g1/g1_body29_hand14.urdf third_party/xr_teleoperate/assets/g1/
COPY aaltonomous-humanoid/aaltonomous aaltonomous
# Our skills and the agent (with its .env: API key and model name).
COPY agent agent
COPY adk_agents/robot_sorter adk_agents/robot_sorter
ENV PYTHONPATH=/app:/app/adk_agents PATH=/opt/venv/bin:$PATH
# Load-time check at build: imports, URDF, config, calibration and the planner (fails the build, not the run).
RUN python -c "import robot_sorter; from robot_sorter.robot import get_robot; r = get_robot(); c = r.ctx; \
print('planner ok; parked pose', 'ok' if r.aside is not None else 'MISSING'); \
from robot_sorter.agent import root_agent; print('agent', root_agent.model, len(root_agent.tools), 'tools')"

EXPOSE 8000
CMD ["python", "-m", "robot_sorter.serve"]
