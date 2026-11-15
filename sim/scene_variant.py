"""Scene variants for the Isaac sim, to test a policy on more than the nominal scene.

Run INSIDE the sim container before `aalto sim` (sim/modal_sim.py --variant "..." does this). It edits the
container's copy of the organisers' files in place; the checkout on the laptop is never touched, and the sim, the
evaluator and the scoring all read the same edited files, so they stay consistent with each other.

    python scene_variant.py --colors real --dr wide --placement wide --instructions shape
    python scene_variant.py                      # no arguments: back to the organisers' nominal scene

  --colors real        the foam blocks the organisers announced on Discord (14 Nov): yellow, purple, orange, with
                       their real sort_fixed mapping yellow -> left, purple -> center, orange -> right.
                       (orange objects in front of orange bins: on purpose, that is the real setup.)
  --colors pastel      washed-out shades of the same three (worn foam, strong light)
  --colors venue       the three colours as the robot's head camera recorded the real blocks (data drop 2). Use
                       this one to judge how the policy will do on the real colours. NOTE: with "--colors real"
                       and "--dr wide" orange and yellow are only 20 deg of hue apart and the +-11 deg jitter makes
                       some orange and yellow objects truly indistinguishable; wrong bins there are partly an
                       artefact of that scene, not of the policy.
  --dr wide            writes sim/dr/wide.yaml: the organisers' training randomisation with wider object shades
                       (hue / saturation / brightness), lighting, camera noise and camera pose error. Then start
                       the sim with AALTO_DR=wide (modal_sim.py: --dr wide).
  --placement wide     objects anywhere between the robot and the bins (not only the nominal 7 cm strip), any yaw
  --instructions shape sort_language instructions that name shapes, or colour and shape together, as the
                       organisers said the finals may ("colour, shape or both"). These sentences are ours.

What a variant does NOT change: object size, the bins, the counter, the camera. Prisms always stand on a
triangular face (the sim has no lying prisms).
"""
import argparse
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = next((p for p in (Path("/opt/aalto"), ROOT.parent / "aaltonomous-humanoid") if (p / "aaltonomous").is_dir()), None)

COLORS = {
    "real": {"yellow": [235, 200, 20], "purple": [125, 70, 175], "orange": [245, 135, 25]},
    "pastel": {"yellow": [240, 220, 110], "purple": [165, 130, 200], "orange": [245, 170, 100]},
    # As the robot's own head camera sees the real blocks: measured by our teammates on 5 frames of data drop 2
    # (bot-icelli docs/research/aalto-discord-notes.md; OpenCV hue / saturation): orange 11-13 / 150-220,
    # yellow 34-46 / 72-88 (a pale green-yellow), purple ~110 / 140-150 (looks blue).
    "venue": {"yellow": [194, 217, 150], "purple": [82, 119, 191], "orange": [230, 129, 62]},
}
FIXED = {"yellow": "left", "purple": "center", "orange": "right"}      # Discord, 14 Nov 15:07

DR_WIDE = """# Wider than train.yaml where it matters for telling objects apart; written by sim/scene_variant.py.
name: wide
lighting:
  key_intensity_scale: [0.5, 1.6]
  key_pos_jitter: [0.5, 0.5, 0.3]
  key_rgb_jitter: 0.12
  dome_intensity_scale: [0.4, 1.6]
  dome_rgb_jitter: 0.08
textures:
  mat_scale: [0.75, 1.15]
  mat_rgb_jitter: 0.08
  mat_bias: [-0.04, 0.04]
  counter_rgb_jitter: 0.25
  bin_rgb_jitter: 0.10
objects:
  hue_jitter: 0.03                      # turns (0.03 = 11 deg)
  sat_scale: [0.6, 1.2]
  val_scale: [0.6, 1.2]
  mass_scale: [0.7, 1.5]
  friction_scale: [0.75, 1.1]
pose:
  xy_jitter: 0.005
  yaw_jitter_deg: 5
camera:
  rgb_noise_std: 4.0
  rgb_gain: [0.85, 1.15]
  depth_noise_mm: 1.5
  depth_noise_rel: 0.002
  depth_hole_frac: 0.02
  pose_jitter_m: 0.01
  pose_jitter_deg: 1.0
"""

SHAPE_INSTRUCTIONS = '''

# ---- added by sim/scene_variant.py --instructions shape: instructions by shape, or by colour and shape -------------
_SHAPE_WORD = {"cube": "cubes", "cylinder": "cylinders", "triangle": "triangular prisms"}
_SHAPE_TEMPLATES = (
    "Put the {s0} in the {b0} bin, the {s1} in the {b1} bin and the {s2} in the {b2} bin.",
    "Sort by shape: {s0} go to the {b0} bin, {s1} to the {b1} bin, {s2} to the {b2} bin.",
    "The {b0} bin is for {s0}, the {b1} bin for {s1} and the {b2} bin for {s2}. Colour does not matter.",
)
_MIXED_TEMPLATES = (
    "Put all {c0} objects in the {b0} bin. Of the others, the {s0} go in the {b1} bin and everything else in the {b2} bin.",
    "{C0} objects belong in the {b0} bin whatever their shape; other {s0} go to the {b1} bin; the rest go to the {b2} bin.",
)
_colour_instruction = make_instruction


def make_instruction(task, rng):
    if task != "sort_language":
        return _colour_instruction(task, rng)
    kind = int(rng.integers(3))
    if kind == 0:
        return _colour_instruction(task, rng)
    shapes, bins = [str(s) for s in rng.permutation(SHAPES)], [str(b) for b in rng.permutation(BINS)]
    words = {f"s{i}": _SHAPE_WORD[s] for i, s in enumerate(shapes)}
    words.update({f"b{i}": b for i, b in enumerate(bins)})
    if kind == 1:
        text = _SHAPE_TEMPLATES[int(rng.integers(len(_SHAPE_TEMPLATES)))].format(**words)
        return text, dict(zip(shapes, bins))
    colour = str(COLORS[int(rng.integers(len(COLORS)))])
    words.update(c0=colour, C0=colour.capitalize())
    text = _MIXED_TEMPLATES[int(rng.integers(len(_MIXED_TEMPLATES)))].format(**words)
    return text, {colour: bins[0], shapes[0]: bins[1], "default": bins[2]}      # most specific key wins (resolve_bin)
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--colors", choices=sorted(COLORS), default=None)
    ap.add_argument("--dr", choices=["wide"], default=None)
    ap.add_argument("--placement", choices=["wide"], default=None)
    ap.add_argument("--instructions", choices=["shape"], default=None)
    ap.add_argument("--table", default=None, help="dx,dy,dz in metres: move the table (counter, mat, bins, objects) "
                    "relative to the robot, e.g. 0,0,0.03 = table 3 cm higher. The policy can keep the original "
                    "file: ROBOT_SCENE_FILE=<scene.yaml>.orig")
    a = ap.parse_args()
    pkg = REPO / "aaltonomous"
    scene, tasks = pkg / "scene" / "scene.yaml", pkg / "interface" / "tasks.py"
    # Always start from the organisers' files as they were: a sim container is reused for the next job of a sweep,
    # and a variant applied on top of another variant (or left behind for a nominal job) would be a different scene.
    for f in (scene, tasks):
        orig = f.with_name(f.name + ".orig")
        if not orig.exists():
            orig.write_text(f.read_text())
    s, t = (scene.with_name(scene.name + ".orig")).read_text(), (tasks.with_name(tasks.name + ".orig")).read_text()
    if a.colors:
        cols = COLORS[a.colors]
        block = "".join(f"    {name}: {rgb}\n" for name, rgb in cols.items())
        s, n = re.subn(r"(  colors:[^\n]*\n)(?:    \w+: \[[^\n]*\n)+", lambda m: m.group(1) + block, s, count=1)
        assert n == 1, "objects.colors block not found in scene.yaml"
        names = list(cols)
        sentence = ("Put the {0} objects in the {1} bin, the {2} objects in the {3} bin and the {4} objects in the "
                    "{5} bin.").format(names[0], FIXED[names[0]], names[1], FIXED[names[1]], names[2], FIXED[names[2]])
        t += (f"\n\n# ---- added by sim/scene_variant.py --colors {a.colors} ----\n"
              f"COLORS = {tuple(names)!r}\nFIXED_MAPPING = {dict(FIXED)!r}\nFIXED_INSTRUCTION = {sentence!r}\n")
    if a.placement == "wide":
        for key, value in (("region_x", "[0.24, 0.66]"), ("region_y", "[0.12, 0.23]"), ("yaw_range_deg", "[-90, 90]")):
            s, n = re.subn(rf"(\n  {key}: )\[[^\]]*\]", lambda m: m.group(1) + value, s, count=1)
            assert n == 1, f"layout.{key} not found in scene.yaml"
    if a.instructions == "shape":
        t += SHAPE_INSTRUCTIONS
    if a.table:
        d = [float(v) for v in a.table.split(",")]

        def moved(m):
            o = [float(v) for v in m.group(2).split(",")]
            return m.group(1) + "[" + ", ".join(f"{o[k] + d[k]:.4f}" for k in range(3)) + "]"
        s, n = re.subn(r"(\n  mat_origin_in_pelvis: )\[([^\]]*)\]", moved, s, count=1)
        assert n == 1, "counter.mat_origin_in_pelvis not found in scene.yaml"
    if a.dr == "wide":
        (pkg / "sim" / "dr" / "wide.yaml").write_text(DR_WIDE)
    scene.write_text(s)
    tasks.write_text(t)
    print(f"scene variant applied in {REPO}: colors={a.colors} dr={a.dr} placement={a.placement} "
          f"instructions={a.instructions} table={a.table}")


if __name__ == "__main__":
    main()
