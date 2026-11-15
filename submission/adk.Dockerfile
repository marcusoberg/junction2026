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
# calibrate / measure_scene (robot_sorter/selfcal.py) fuse their measurements with GTSAM.
RUN /opt/venv/bin/pip install --no-cache-dir gtsam==4.3.0 && /opt/venv/bin/pip uninstall -y pytest pluggy iniconfig 2>/dev/null; true

WORKDIR /app
# The robot model (inverse kinematics and the planner's safety pre-check) and the organisers' package.
COPY aaltonomous-humanoid/third_party/xr_teleoperate/assets/g1/g1_body29_hand14.urdf third_party/xr_teleoperate/assets/g1/
# The arm and hand meshes (25 MB): calibrate fits the head camera to the robot's own arms in the depth image.
COPY aaltonomous-humanoid/third_party/xr_teleoperate/assets/g1/meshes/left_shoulder* \
     aaltonomous-humanoid/third_party/xr_teleoperate/assets/g1/meshes/right_shoulder* \
     aaltonomous-humanoid/third_party/xr_teleoperate/assets/g1/meshes/left_elbow* \
     aaltonomous-humanoid/third_party/xr_teleoperate/assets/g1/meshes/right_elbow* \
     aaltonomous-humanoid/third_party/xr_teleoperate/assets/g1/meshes/left_wrist* \
     aaltonomous-humanoid/third_party/xr_teleoperate/assets/g1/meshes/right_wrist* \
     aaltonomous-humanoid/third_party/xr_teleoperate/assets/g1/meshes/left_hand* \
     aaltonomous-humanoid/third_party/xr_teleoperate/assets/g1/meshes/right_hand* \
     third_party/xr_teleoperate/assets/g1/meshes/
COPY aaltonomous-humanoid/aaltonomous aaltonomous
# Our skills and the agent (with its .env: API key and model name).
COPY agent agent
COPY adk_agents/robot_sorter adk_agents/robot_sorter
ENV PYTHONPATH=/app:/app/adk_agents PATH=/opt/venv/bin:$PATH
# Every tool call, check and camera frame of a run is written here (docker cp <container>:/app/trace . after a run):
# the run of 15 Nov on the real robot left only the console log to work from.
ENV ROBOT_TRACE_DIR=/app/trace
# Load-time check at build: imports, URDF, config, calibration and the planner (fails the build, not the run).
RUN python -c "import robot_sorter; from robot_sorter.robot import get_robot; r = get_robot(); c = r.ctx; \
print('planner ok; parked pose', 'ok' if r.aside is not None else 'MISSING'); \
from robot_sorter.agent import root_agent; print('agent', root_agent.model, len(root_agent.tools), 'tools'); \
import numpy as np, gtsam; from robot_sorter import selfcal, er; \
m = [selfcal.ArmModel(c.kin, lock=c.lock), selfcal.ArmModel(c.kin, lock=c.lock, hands=True)]; \
print('arm models', [len(x.links) for x in m], 'link meshes; gtsam ok; ER client', er.MODEL); \
a = r.act({'arm_q': np.asarray(c.home, float), 'hand_q': np.asarray(c.post['open'], float), 't': 0.0}); \
assert 'arm_tau_ff' not in a, 'the runner adds the gravity torque: the policy must not send it too'; \
print('action keys', sorted(a), '(no feed-forward torque from the policy: the runner adds it)')"
ARG REVISION=unknown
LABEL org.opencontainers.image.revision=$REVISION

EXPOSE 8000
CMD ["python", "-m", "robot_sorter.serve"]
