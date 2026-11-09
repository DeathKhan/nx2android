#!/usr/bin/env python3
"""Diagnostic v2: live output modes layered on the audited cca lighting gates.

The v1 artifact is never overwritten. Bits0..3 retain v1 contribution gates;
bits8..11 select output only. All original samples/math/side effects still run.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import cca_lighting_gate as gate
import spv_intermediate as spv

MODES = {
    0: "Original output with optional low-bit contribution gates",
    1: "Selected-fragment coverage tag (0.875,0.125,0.875,1)",
    2: "Raw A+24 BC7 array layer0 sample (RGBA)",
    3: "Raw A+16 BC6H array layer0 R replicated to RGB, alpha1",
    4: "Location13 input XY, then0,1",
    5: "Final clamped direct RGB before its low-bit output gate",
    6: "Final clamped indirect RGB before its low-bit output gate",
}


def instrument(data):
    gated, meta = gate.instrument(data)
    h, ins = spv.decode(gated)
    original_bound = h[3]
    decl, init = [], []
    def fresh():
        r = h[3]
        h[3] += 1
        return r
    def constant(t, bits):
        for op, a in ins + decl:
            if op == 43 and a[0] == t and a[2:] == [bits]:
                return a[1]
        r = fresh()
        decl.append((43, [t, r, bits]))
        return r
    zero, one = 716, 723
    private4, private1 = fresh(), fresh()
    decl += [(32, [private4, 6, 5]), (32, [private1, 6, 4])]
    bool4 = next((a[0] for op, a in ins if op == 23 and a[1:] == [121, 4]), None)
    if bool4 is None:
        bool4 = fresh()
        decl.append((23, [bool4, 121, 4]))
    zero4 = fresh()
    decl.append((44, [5, zero4, zero, zero, zero, zero]))
    tag_hi, tag_lo = constant(4, 0x3F600000), constant(4, 0x3E000000)
    tag = fresh()
    decl.append((44, [5, tag, tag_hi, tag_lo, tag_hi, one]))
    mask15 = constant(6, 15)
    mode_consts = {m: constant(6, m) for m in range(1, 7)}
    variables = []
    def variable(pointer, initial):
        r = fresh()
        variables.append((59, [pointer, r, 6]))
        init.append((62, [r, initial]))
        return r
    rgb, scalar, uv = [variable(private4, zero4) for _ in range(3)]
    direct = [variable(private1, zero) for _ in range(3)]
    indirect = [variable(private1, zero) for _ in range(3)]
    captures = {}
    captures[1484] = [(62, [rgb, 1484])]
    scalar_r, scalar_vec = fresh(), fresh()
    captures[1521] = [(81, [4, scalar_r, 1521, 0]),
                      (80, [5, scalar_vec, scalar_r, scalar_r, scalar_r, one]),
                      (62, [scalar, scalar_vec])]
    for target, var in zip(gate.DIRECT + gate.INDIRECT, direct + indirect):
        raw = meta["gates"][target]["raw_result"]
        captures[raw] = [(62, [var, raw])]
    # Read only the two input components already used by this shader.
    uv_values = []
    for component in (691, 705):
        ptr, value = fresh(), fresh()
        init += [(65, [692, ptr, 80, component]), (61, [4, value, ptr])]
        uv_values.append(value)
    uv_value = fresh()
    init += [(80, [5, uv_value, *uv_values, zero, one]), (62, [uv, uv_value])]
    mask_value = meta["prefix"][1][1][1]
    shifted, mode = fresh(), fresh()
    selector = [(194, [6, shifted, mask_value, 785]), (199, [6, mode, shifted, mask15])]
    output = []
    before_return = []
    original_output = fresh()
    before_return.append((61, [5, original_output, 91]))
    candidates = {1: tag}
    for m, var in ((2, rgb), (3, scalar), (4, uv)):
        r = fresh()
        before_return.append((61, [5, r, var]))
        candidates[m] = r
    for m, vars_ in ((5, direct), (6, indirect)):
        channels = []
        for var in vars_:
            r = fresh()
            before_return.append((61, [4, r, var]))
            channels.append(r)
        value = fresh()
        before_return.append((80, [5, value, *channels, one]))
        candidates[m] = value
    selected = original_output
    for m in range(1, 7):
        condition, condition4, value = fresh(), fresh(), fresh()
        before_return += [(170, [121, condition, mode, mode_consts[m]]),
                          (80, [bool4, condition4, condition, condition, condition, condition]),
                          (169, [5, value, condition4, candidates[m], selected])]
        selected = value
    before_return.append((62, [91, selected]))
    declared = initialized = False
    capture_count = 0
    function = None
    first_label = False
    for op, old in ins:
        a = list(old)
        if op == 54:
            if not declared:
                output += decl + variables
                declared = True
            function = a[1]
        if op == 15 and h[1] >= 0x10400:
            a += [v[1][1] for v in variables]
        if function == 94 and op == 248 and not initialized:
            first_label = True
        if first_label and not initialized and op not in (248, 59):
            output += init
            initialized = True
        if op == 253 and function == 94:
            output += before_return
        output.append((op, a))
        # Mask SSA is defined by the second v1-prefix instruction.
        if op == 81 and a[1] == mask_value:
            output += selector
        result = a[1] if op in (12, 87) else None
        if result in captures:
            output += captures[result]
            capture_count += 1
        if op == 56:
            function = None
    assert initialized and capture_count == 8
    meta.update({"schema": "cca-live-lighting-modes-v2", "output_modes": MODES,
                 "output_mode_encoding": "(uint32 control >> 8) & 15; modes7..15 retain original output",
                 "identity_ack": 16, "mode0_note": "Original output bits remain selected; low4 bits optionally apply v1 gates.",
                 "source_targets": {"rgb_layer0": 1484, "scalar_layer0": 1521, "uv_location": 13,
                                    "direct": gate.DIRECT, "indirect": gate.INDIRECT},
                 "coverage_limits": "Depth, blending, write masks, other shaders and later draws still apply. Coverage identifies only surviving selected fragments. R11 clips negative values and omits alpha.",
                 "branch_not_executed": "Sample capture remains float4 zero for this invocation.",
                 "combined_modes": "Low contribution gates still affect upstream values in modes5/6; use controls0x500/0x600 alone for original branches.",
                 "v2_proof": {"declarations": decl + variables, "initialization": init,
                              "selector": selector, "captures": captures, "before_return": before_return,
                              "gated_bound": original_bound, "mode_id": mode,
                              "original_output_id": original_output, "candidates": candidates}})
    return spv.encode(h, output), meta


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--origin", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    result, meta = instrument(a.source.read_bytes())
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_bytes(result)
    subprocess.run(["spirv-val", "--target-env", "vulkan1.1", str(a.out)], check=True)
    meta["origin"] = json.loads(a.origin.read_text())
    meta["output_sha256"] = hashlib.sha256(result).hexdigest()
    meta["tool_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    Path(str(a.out) + ".json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps({"sha256": meta["output_sha256"], "bytes": len(result)}))


if __name__ == "__main__":
    main()
