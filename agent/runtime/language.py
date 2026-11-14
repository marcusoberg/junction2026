"""Instruction -> colour-to-bin mapping without any model: the organisers' training templates exactly, then a
clause-by-clause reading that also covers re-worded sentences ("the left bin is for red", "red goes left").
Returns None when a sentence cannot be read with confidence; the caller then asks a model or keeps the default."""
from __future__ import annotations

import re

from aaltonomous.interface import tasks

BIN_WORDS = {"left": "left", "center": "center", "centre": "center", "middle": "center", "right": "right"}
SHAPE_WORDS = {"cube": "cube", "cubes": "cube", "box": "cube", "boxes": "cube", "block": "cube", "blocks": "cube",
               "cylinder": "cylinder", "cylinders": "cylinder", "triangle": "triangle", "triangles": "triangle",
               "prism": "triangle", "prisms": "triangle"}


def parse_mapping(text):
    """-> {"red": "left", ...} (keys may also be "red cube" or "cube" when the sentence names shapes), or None."""
    if not text:
        return None
    if text == tasks.FIXED_INSTRUCTION:
        return dict(tasks.FIXED_MAPPING)
    try:
        from aaltonomous.sim.oracle import parse_instruction
        m = parse_instruction(text)
        if m:
            return m
    except Exception:
        pass
    out = {}
    for clause in re.split(r"[,.;:]|\band\b", text.lower()):
        words = re.findall(r"[a-z]+", clause)
        colors = [w for w in words if w in tasks.COLORS]
        bins = [BIN_WORDS[w] for w in words if w in BIN_WORDS]
        shapes = [SHAPE_WORDS[w] for w in words if w in SHAPE_WORDS]
        if len(bins) != 1 or len(colors) > 1 or len(set(shapes)) > 1 or not (colors or shapes):
            continue
        key = " ".join(colors + sorted(set(shapes)))
        if key in out and out[key] != bins[0]:
            return None                              # contradictory reading
        out[key] = bins[0]
    colour_keys = [k for k in out if k in tasks.COLORS]
    if len(colour_keys) == len(tasks.COLORS) or (out and any(" " in k or k in SHAPE_WORDS.values() for k in out)):
        return out
    return None
