#!/usr/bin/env python3
"""Diagnostic live contribution gates for one audited runtime-normalized cca shader.

This is not a production shader transformation. Original instructions execute in
the same order. A select after each chosen result preserves its bits when off.
The control is existing support-buffer member7 (int32 byte1376), no new resource.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import spv_intermediate as spv

SOURCE_SHA = "8dcfae22d0d8f5afed3d5a4c1795d4e89f36049a52a555168ba521dabbfd56dc"
SH_FIELDS = dict(zip([169, 171, 173, 175, 177, 179, 185, 187, 189, 197, 199, 201],
                     [48, 52, 56, 64, 68, 72, 80, 84, 88, 96, 100, 104]))
DIRECT = [3138, 3172, 3206]
INDIRECT = [3162, 3196, 3230]
LABELS = {
    1: "Suppress twelve probe SH coefficient values after their storage reads",
    2: "Remove baked-lightmap additive terms, retaining each existing SH addend",
    4: "Suppress final clamped direct-light RGB contribution",
    8: "Suppress final clamped indirect-light RGB contribution",
}


def mapping(data):
    if hashlib.sha256(data).hexdigest() != SOURCE_SHA:
        raise ValueError("requires the exact audited runtime source; do not apply IDs to another module")
    h, ins = spv.decode(data)
    constants = {a[1]: a[2] for op, a in ins if op == 43 and len(a) == 3}
    definitions = {a[1]: (op, a) for op, a in ins if op in (12, 61, 65, 124, 129, 133)}
    assert (72, [10, 7, 35, 1376]) in ins  # member7 Offset1376
    assert (30, [10, 1, 3, 5, 5, 6, 8, 9, 6]) in ins
    assert (59, [11, 13, 2]) in ins
    gates = {}
    for private, offset in SH_FIELDS.items():
        values = [a[1] for op, a in ins if op == 62 and a[0] == private]
        assert len(values) == 1
        result = values[0]
        op, a = definitions[result]
        assert op == 124 and a[0] == 4  # loaded uint coefficient reinterpreted as float
        load_op, load = definitions[a[2]]
        ptr_op, pointer = definitions[load[2]]
        assert load_op == 61 and ptr_op == 65 and pointer[2] == 36
        gates[result] = {"mask": 1, "replacement": 716, "role": "probe_sh",
                         "guest_ssbo_reg": 0, "byte_offset": offset,
                         "private_destination": private, "original": [op, a]}
    pointers = {a[1] for op, a in ins if op == 65 and a[2] == 19
                and [constants.get(i) for i in a[3:]] == [0, 56, 0]}
    gains = {a[1] for op, a in ins if op == 61 and a[2] in pointers}
    for op, a in ins:
        if op == 12 and len(a) == 7 and a[3] == 50 and a[5] in gains:
            gates[a[1]] = {"mask": 2, "replacement": a[6], "role": "baked_addend",
                           "decoded_baked_value": a[4], "gain": "guest CBUF3[56].x",
                           "retained_sh_addend": a[6], "original": [op, a]}
    assert sum(g["mask"] == 2 for g in gates.values()) == 12
    for mask, results in ((4, DIRECT), (8, INDIRECT)):
        for channel, result in zip("RGB", results):
            op, a = definitions[result]
            assert op == 12 and a[:4] == [4, result, 766, 80] and a[4] == 716
            gates[result] = {"mask": mask, "replacement": 716,
                             "role": "final_direct" if mask == 4 else "final_indirect",
                             "channel": channel, "original": [op, a]}
    assert len(gates) == 30
    assert not any(op == 71 and a[0] in gates for op, a in ins)
    # Each output is exactly a sum of the independently gated clamped terms.
    for direct, indirect in zip(DIRECT, INDIRECT):
        assert any(op == 129 and a[2:] == [direct, indirect] for op, a in ins)
    return h, ins, gates


def instrument(data):
    h, ins, gates = mapping(data)
    oldbound = h[3]
    def fresh():
        value = h[3]
        h[3] += 1
        return value
    # Loading the existing fixed-size struct and extracting member7 avoids adding
    # a Uniform pointer type; the strict runtime layout guard stays unchanged.
    support_value, mask_value = fresh(), fresh()
    prefix = [(61, [10, support_value, 13]), (81, [6, mask_value, support_value, 7])]
    conditions = {}
    bit_constants = {1: 705, 2: 701, 4: 749, 8: 785}
    for mask, const in bit_constants.items():
        masked, condition = fresh(), fresh()
        prefix += [(199, [6, masked, mask_value, const]), (171, [121, condition, masked, 691])]
        conditions[mask] = condition
    output = []
    inserted = False
    in_main = False
    entry_block = False
    for op, args in ins:
        a = list(args)
        if op == 54:
            in_main = a[1] == 94
        if in_main and op == 248 and not inserted:
            entry_block = True
            output.append((op, a))
            continue
        if entry_block and not inserted and op != 59:
            output.extend(prefix)
            inserted = True
        result = a[1] if len(a) > 1 and op in (12, 124) else None
        if result in gates:
            gate = gates[result]
            assert [op, args] == gate["original"]
            raw = fresh()
            a[1] = raw
            output.append((op, a))
            output.append((169, [4, result, conditions[gate["mask"]], gate["replacement"], raw]))
            gate["raw_result"] = raw
            gate["condition"] = conditions[gate["mask"]]
        else:
            output.append((op, a))
    assert inserted
    metadata = {"schema": "cca-live-lighting-gates-v1", "diagnostic_only": True,
                "source_sha256": SOURCE_SHA, "key": "cca3264814bd7187",
                "support": {"variable": 13, "struct_type": 10, "member": 7,
                            "byte_offset": 1376, "encoding": "int32 bitmask"},
                "labels": LABELS, "gates": gates, "prefix": prefix,
                "old_bound": oldbound, "new_bound": h[3],
                "mask_zero": "All selects take the exact original SSA bits; no original FP opcode, operand, or instruction order changes.",
                "scope": "Mask2 removes additive baked coefficients, not their normal/visibility modulation of other terms. Mask8 is the complete final indirect branch. Only this fragment key is affected."}
    return spv.encode(h, output), metadata


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
    print(json.dumps({"sha256": meta["output_sha256"], "gates": len(meta["gates"]), "bytes": len(result)}))


if __name__ == "__main__":
    main()
