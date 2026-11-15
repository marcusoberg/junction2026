# The same image as adk.Dockerfile with two settings that are right for the Isaac SIM only (build adk.Dockerfile first):
#   docker buildx build --platform linux/amd64 -f submission/adk-sim.Dockerfile --build-arg BASE=<image>:<tag> -t <image>:<tag>-sim --load .
# ROBOT_HEAD_PP: the sim renders with the principal point at the image centre, not the calibration file's (325.0,
#   250.9): object positions are then 1.0 mm from the truth instead of 3.2 mm (98 objects, ORGANIZER_ISSUES A14).
# ROBOT_TRI_OFFSET_M: grasp offset for standing triangular prisms. Sim grasp bench, 24 prisms per setting: without
#   either 12 and 15 sorted (two runs), principal point alone 16, offset alone 19, both 20 (first grasp held 16 against 9-10).
# On the real robot the calibration file's principal point is the camera's own and the offset is untested: use the
# plain image there.
ARG BASE=maroberg/robot-sorter:gemini-er-1
FROM ${BASE}
ENV ROBOT_HEAD_PP=320,240 ROBOT_TRI_OFFSET_M=0.004,0.005,0.006
