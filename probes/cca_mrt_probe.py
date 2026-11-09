#!/usr/bin/env python3
"""Audited CCA diagnostic MRT: preserve original output0 and all original operations.

Support member7 at byte1376 selects one raw mode1..12. Output1 is float4;
the host must supply its isolated RGBA32F attachment for the selected draw.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import cca_lighting_gate as gate
import spv_intermediate as spv

MODES = {
    1: "Original final RGBA before R11G11B10F conversion",
    2: "Actual baseUV.xy and lightmapUV.xy after original reciprocal-w reconstruction",
    3: "Original implicit-LOD normal ASTC sample RGBA (guest texture1)",
    4: "Original implicit-LOD base-color ASTC sample RGBA (guest texture0)",
    5: "Original A+24 BC7 layer0 sample RGBA",
    6: "Original A+16 BC6H layer0 sample RGBA",
    7: "Original A+24 BC7 layer1 sample RGBA",
    8: "Original A+16 BC6H layer1 sample RGBA",
    9: "Original A+24 BC7 layer2 sample RGBA",
    10: "Original A+16 BC6H layer2 sample RGBA",
    11: "Original final direct RGB, alpha1",
    12: "Original final indirect RGB, alpha1",
}
SAMPLES = {3: 1275, 4: 2272, 5: 1484, 6: 1521,
           7: 1387, 8: 1355, 9: 2312, 10: 2351}
SCALARS = {2: [1219, 1226, 1212, 1163],
           11: gate.DIRECT, 12: gate.INDIRECT}


LAYER_COORDS = [1349, 1383, 2308, 2347]
RESULT_OPS = (12, 61, 111, 133, 87)


def instrument(data, mode2="uv"):
    if mode2 not in ("uv", "layer-coordinates"):
        raise ValueError("unsupported mode2 observable")
    scalars = {k: list(v) for k, v in SCALARS.items()}
    modes = dict(MODES)
    if mode2 == "layer-coordinates":
        scalars[2] = list(LAYER_COORDS)
        modes[2] = "Original converted layer operands: BC6H1, BC7_1, BC7_2, BC6H2; expected1,1,2,2"
    if hashlib.sha256(data).hexdigest() != gate.SOURCE_SHA:
        raise ValueError("requires exact audited runtime-normalized source SHA")
    h, ins = spv.decode(data)
    original_bound = h[3]
    assert (59, [89, 91, 3]) in ins and (32, [89, 3, 5]) in ins
    assert (71, [91, 30, 0]) in ins
    assert (72, [10, 7, 35, 1376]) in ins
    decl, variables, init, captures, annotations = [], [], [], {}, []
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
    private4, private1 = fresh(), fresh()
    decl += [(32, [private4, 6, 5]), (32, [private1, 6, 4])]
    bool4 = next((a[0] for op, a in ins if op == 23 and a[1:] == [121, 4]), None)
    if bool4 is None:
        bool4 = fresh()
        decl.append((23, [bool4, 121, 4]))
    zero4 = fresh()
    decl.append((44, [5, zero4, 716, 716, 716, 716]))
    output = fresh()
    variables.append((59, [89, output, 3]))
    annotations.append((71, [output, 30, 1]))
    candidates = {}
    for mode, target in SAMPLES.items():
        r = fresh()
        variables.append((59, [private4, r, 6]))
        init.append((62, [r, zero4]))
        candidates[mode] = [r]
        captures[target] = [(62, [r, target])]
    for mode, targets in scalars.items():
        candidates[mode] = []
        for target in targets:
            r = fresh()
            variables.append((59, [private1, r, 6]))
            init.append((62, [r, 716]))
            candidates[mode].append(r)
            captures[target] = [(62, [r, target])]
    # Existing struct load avoids adding a Uniform pointer or resource declaration.
    support, mode_value = fresh(), fresh()
    init += [(61, [10, support, 13]), (81, [6, mode_value, support, 7])]
    before_return = []
    original_output = fresh()
    before_return.append((61, [5, original_output, 91]))
    values = {1: original_output}
    for mode, vars_ in candidates.items():
        channels = []
        for var in vars_:
            value = fresh()
            before_return.append((61, [5 if mode in SAMPLES else 4, value, var]))
            channels.append(value)
        if mode in SAMPLES:
            values[mode] = channels[0]
        else:
            if len(channels) == 3:
                channels.append(723)
            result = fresh()
            before_return.append((80, [5, result, *channels]))
            values[mode] = result
    selected = original_output
    for mode in range(2, 13):
        mode_const = constant(6, mode)
        condition, condition4, value = fresh(), fresh(), fresh()
        before_return += [(170, [121, condition, mode_value, mode_const]),
                          (80, [bool4, condition4, condition, condition, condition, condition]),
                          (169, [5, value, condition4, values[mode], selected])]
        selected = value
    before_return.append((62, [output, selected]))
    result, declared, initialized, annotated = [], False, False, False
    function, first_label, captured = None, False, set()
    for op, old in ins:
        a = list(old)
        if not annotated and op in range(19, 40):
            result += annotations
            annotated = True
        if op == 54:
            if not declared:
                result += decl + variables
                declared = True
            function = a[1]
        if op == 15:
            a.append(output)
            if h[1] >= 0x10400:
                a += [v[1][1] for v in variables if v[1][2] == 6]
        if function == 94 and op == 248 and not initialized:
            first_label = True
        if first_label and not initialized and op not in (248, 59):
            result += init
            initialized = True
        if op == 253 and function == 94:
            result += before_return
        result.append((op, a))
        target = a[1] if op in RESULT_OPS else None
        if target in captures:
            result += captures[target]
            captured.add(target)
        if op == 56:
            function = None
    assert captured == set(captures) and declared and initialized and annotated
    meta = {"schema": "cca-mrt-probe-v1", "diagnostic_only": True,
            "source_sha256": gate.SOURCE_SHA, "key": "cca3264814bd7187",
            "mode_encoding": "raw signed-int support member7 at byte1376; 1..12; unknown selects finalRGBA",
            "modes": modes, "mode2": mode2, "sample_ids": SAMPLES, "scalar_ids": scalars,
            "output": {"location": 1, "type": "float32x4", "variable": output},
            "original_output": "Location0 is never changed by inserted instructions.",
            "sample_capture": "Immediately after original result; no new sampling, arithmetic, derivatives or branch.",
            "branch_not_executed": "Private capture initialized to zero per invocation.",
            "limits": "Only fragments surviving original shader/depth produce MRT values. Extra attachment can perturb GPU scheduling; compare original attachment0 against baseline.",
            "proof": {"source_bound": original_bound, "annotations": annotations,
                      "declarations": decl + variables, "initialization": init,
                      "captures": captures, "before_return": before_return,
                      "mode_id": mode_value, "output_id": output,
                      "original_output_id": original_output, "candidates": candidates}}
    if mode2 == "layer-coordinates":
        meta["mode2_fields"] = [
            {"channel": channel, "ssa": target, "sample_ssa": sample,
             "expected": expected, "operation": "OpConvertSToF"}
            for channel, target, sample, expected in
            zip("xyzw", LAYER_COORDS, (1355, 1387, 2312, 2351), (1, 1, 2, 2))
        ]
        meta["mode2_expected_scope"] = "Captured CCA396 nonzero A+16/A+24 handles; untaken branch remains zero."
    return spv.encode(h, result), meta


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--origin", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--mode2", choices=("uv", "layer-coordinates"), default="uv")
    a = p.parse_args()
    data, meta = instrument(a.source.read_bytes(), a.mode2)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_bytes(data)
    subprocess.run(["spirv-val", "--target-env", "vulkan1.1", str(a.out)], check=True)
    meta["origin"] = json.loads(a.origin.read_text())
    meta["output_sha256"] = hashlib.sha256(data).hexdigest()
    meta["tool_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    Path(str(a.out) + ".json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps({"sha256": meta["output_sha256"], "bytes": len(data)}))


if __name__ == "__main__":
    main()
