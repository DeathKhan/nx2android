#!/usr/bin/env python3
"""Audit BC6H center samples against the renderer's actual CPU decoder.

Build the decoder on Arch with:
cc -std=c11 -O2 -shared -fPIC nxnative/dm/gfx/tex/nvn_bc.c -o dist/bc-sampling/libnvn_bc.so
This models a small neighboring-texel contribution; it does not identify the
driver's exact filtering algorithm or assert a Vulkan conformance defect.
"""
import argparse
import ctypes
import json
import struct
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", type=Path)
    p.add_argument("reference", type=Path)
    p.add_argument("candidate", type=Path)
    p.add_argument("decoder", type=Path)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    data = a.input.read_bytes()
    header = struct.unpack_from("<8s8I2Q8x", data)
    magic, version, w, h, layers, vkfmt, _, n, _, block_bytes, point_bytes = header
    if magic != b"BCSAMP01" or version != 1 or vkfmt != 143 or len(data) != 64 + block_bytes + point_bytes:
        raise ValueError("expected complete BC6H unsigned input")
    manifest = json.loads(Path(str(a.input) + ".json").read_text())
    ref = struct.unpack(f"<{n * 4}f", a.reference.read_bytes())
    candidate = struct.unpack(f"<{n * 4}f", a.candidate.read_bytes())
    lib = ctypes.CDLL(str(a.decoder.resolve()))
    fn = lib.nvn_bc_decode
    fn.argtypes = [ctypes.c_int, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
                   ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
                   ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t]
    fn.restype = ctypes.c_int
    cache = {}

    def texel(x, y, z):
        key = (x // 4, y // 4, z)
        if key not in cache:
            off = 64 + ((z * ((h + 3) // 4) + y // 4) * ((w + 3) // 4) + x // 4) * 16
            src = ctypes.create_string_buffer(data[off:off + 16])
            dst = ctypes.create_string_buffer(128)
            if fn(0x50, 4, 4, 1, src, 16, 16, 16, dst, 128, 32, 128):
                raise ValueError("CPU block decode refused")
            cache[key] = struct.unpack("<64e", dst.raw)
        i = ((y % 4) * 4 + x % 4) * 4
        return cache[key][i:i + 3]

    half = lambda x: struct.unpack("<e", struct.pack("<e", x))[0]
    rows = []
    for point in manifest["points"]:
        if point["kind"] != "texel_center":
            continue
        x = round(point["uv"][0] * w - 0.5)
        y = round(point["uv"][1] * h - 0.5)
        z = point["layer"]
        if not (0 <= x < w and 0 <= y < h and 0 <= z < layers):
            raise ValueError("center outside image")
        i = point["index"] * 4
        center, left = texel(x, y, z), texel(max(x - 1, 0), y, z)
        rows.append(dict(point, texel=[x, y, z], reference=ref[i:i + 3],
                         candidate=candidate[i:i + 3], cpu_center=center, cpu_left=left,
                         normalized_x_position=point["uv"][0] * w - 0.5,
                         model_left_1_over_256=[half((255 * c + l) / 256) for c, l in zip(center, left)]))
    result = {"schema": "bc6h-center-audit-v1", "input_sha256": manifest["sha256"],
              "centers": len(rows),
              "cpu_reference_exact": sum(r["cpu_center"] == r["reference"] for r in rows),
              "rows": rows,
              "scope": "CPU decode and neighboring-texel model only; native integer texel fetch has not been measured."}
    a.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ["centers", "cpu_reference_exact"]}))


if __name__ == "__main__":
    main()
