# Submission image: pi0.5 (LeRobot 0.7.0) behind the organisers' policy-server protocol on port 8000.
#
# Build from the folder that contains both submission/ and aaltonomous-humanoid/ (the build context):
#
#   1. put the checkpoint in submission/weights/ so that submission/weights/model.safetensors exists
#      (it is lerobot-train's .../checkpoints/<step>/pretrained_model folder, tokenizer/ included):
#        modal volume get aalto-runs <run>/checkpoints/<step>/pretrained_model ./ckpt && mv ./ckpt/* submission/weights/
#   2. docker build --platform linux/amd64 -f submission/Dockerfile -t <you>/aalto-policy:v1 .
#   3. docker run --rm --gpus all -p 8000:8000 <you>/aalto-policy:v1
#
# Plumbing-only image without weights (holds/waves, for testing the container contract):
#   docker build --platform linux/amd64 -f submission/Dockerfile --build-arg MODEL=stub -t aalto-policy:stub .
#
# Needs aaltonomous-humanoid/third_party/xr_teleoperate/assets/g1/g1_body29_hand14.urdf (the safety pre-check's FK).
FROM nvidia/cuda:12.8.0-base-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 UV_LINK_MODE=copy
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates libgomp1 libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.5.11 /uv /usr/local/bin/uv
RUN uv venv /opt/venv --python 3.12
ENV VIRTUAL_ENV=/opt/venv PATH=/opt/venv/bin:$PATH

# torch built against CUDA 12.8, as the organisers' GPU example does (RTX 5090). PyPI's default torch wheel now
# bundles CUDA 13 and needs a newer host driver; installing this one first keeps lerobot from pulling that in.
RUN uv pip install --no-cache "torch>=2.7,<2.16" "torchvision>=0.22,<0.30" \
        --index-url https://download.pytorch.org/whl/cu128
RUN uv pip install --no-cache "lerobot[pi]==0.7.0" msgpack "websockets>=12" pin pyyaml \
    && python -c "import torch; print('torch', torch.__version__, 'built for CUDA', torch.version.cuda)"

WORKDIR /app
COPY aaltonomous-humanoid/third_party/xr_teleoperate/assets/g1/g1_body29_hand14.urdf third_party/xr_teleoperate/assets/g1/
COPY aaltonomous-humanoid/aaltonomous/__init__.py aaltonomous/
COPY aaltonomous-humanoid/aaltonomous/deploy/__init__.py aaltonomous-humanoid/aaltonomous/deploy/protocol.py \
     aaltonomous-humanoid/aaltonomous/deploy/policy_server.py aaltonomous/deploy/
COPY aaltonomous-humanoid/aaltonomous/interface/__init__.py aaltonomous-humanoid/aaltonomous/interface/safety.py \
     aaltonomous-humanoid/aaltonomous/interface/joint_spec.yaml aaltonomous/interface/
COPY submission/server.py /app/server.py
COPY submission/weights/ /app/weights/

ARG MODEL=pi05
ENV PYTHONPATH=/app AALTO_MODEL=${MODEL} AALTO_CKPT=/app/weights HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

# Fail the build, not the run: imports, URDF + safety filter, and (for pi05) the checkpoint files are all present.
RUN python -c "import os, server; server.make_shadow_filter(0.9, 2.7).ws.positions([0.0] * 14); \
m = os.environ['AALTO_MODEL']; need = ['model.safetensors', 'config.json', 'policy_preprocessor.json', 'tokenizer']; \
missing = [f for f in need if not os.path.exists('/app/weights/' + f)] if m == 'pi05' else []; \
assert not missing, f'submission/weights is missing {missing}'; print('image check OK:', m)"

EXPOSE 8000
CMD ["python", "/app/server.py"]
