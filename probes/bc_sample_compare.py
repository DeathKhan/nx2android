#!/usr/bin/env python3
"""Compare fixed-coordinate native samples without imposing cross-API bit identity."""
import argparse
import hashlib
import json
import math
import struct
from pathlib import Path


def compare(manifest, reference, candidate):
    count = manifest["point_count"]
    if len(reference) != count * 16 or len(candidate) != count * 16:
        raise ValueError("sample byte length does not match manifest")
    a = struct.unpack(f"<{count * 4}f", reference)
    b = struct.unpack(f"<{count * 4}f", candidate)
    au = struct.unpack(f"<{count * 4}I", reference)
    bu = struct.unpack(f"<{count * 4}I", candidate)
    rows = []
    groups = {}
    for point in manifest["points"]:
        i = point["index"]
        av, bv = a[i * 4:i * 4 + 4], b[i * 4:i * 4 + 4]
        finite = all(math.isfinite(x) for x in av + bv)
        absolute = max(abs(x - y) for x, y in zip(av, bv)) if finite else None
        relative = max(abs(x - y) / max(abs(x), abs(y), 1e-30)
                       for x, y in zip(av, bv)) if finite else None
        row = dict(point, reference=list(av), candidate=list(bv),
                   finite=finite, max_absolute=absolute, max_relative=relative,
                   bit_differences=sum(x != y for x, y in zip(au[i * 4:i * 4 + 4], bu[i * 4:i * 4 + 4])))
        rows.append(row)
        for key in ("all", f"layer:{point['layer']}", f"kind:{point['kind']}"):
            g = groups.setdefault(key, {"points": 0, "bit_different_components": 0,
                                       "nonfinite_points": 0, "max_absolute": 0.0, "max_relative": 0.0})
            g["points"] += 1
            g["bit_different_components"] += row["bit_differences"]
            g["nonfinite_points"] += not finite
            if finite:
                g["max_absolute"] = max(g["max_absolute"], absolute)
                g["max_relative"] = max(g["max_relative"], relative)
    return {"schema": "bc-sample-comparison-v1", "input_sha256": manifest["sha256"],
            "reference_sha256": hashlib.sha256(reference).hexdigest(),
            "candidate_sha256": hashlib.sha256(candidate).hexdigest(), "groups": groups,
            "worst_points": sorted(rows, key=lambda r: r["max_absolute"] if r["finite"] else math.inf, reverse=True)[:12],
            "scope": "Native fixed-input LOD0 filtering comparison. Numeric differences alone do not establish a game defect or image quality."}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("manifest", type=Path)
    p.add_argument("reference", type=Path)
    p.add_argument("candidate", type=Path)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    result = compare(json.loads(args.manifest.read_text()), args.reference.read_bytes(), args.candidate.read_bytes())
    # JSON retains standards compliance even for a broken driver's nonfinite result.
    def clean(v):
        if isinstance(v, float) and not math.isfinite(v):
            return str(v)
        if isinstance(v, list):
            return [clean(x) for x in v]
        if isinstance(v, dict):
            return {k: clean(x) for k, x in v.items()}
        return v
    args.out.write_text(json.dumps(clean(result), indent=2, allow_nan=False) + "\n")
    print(json.dumps(result["groups"], indent=2))


if __name__ == "__main__":
    main()
