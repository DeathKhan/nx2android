#!/usr/bin/env python3
"""Exact paired mesh-word observer. Diagnostic only: original color0 is unchanged.

VS c8ab0e758c17df4a -> FS 96dbc0392691b437. A new Flat uint location12
carries the original uint mesh word4. Output1 is RGBA32_UINT:
(sideband, immediate FS input bits, recovered original local bits, 0x43415252).
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import spv_intermediate as spv

VS_KEY = "c8ab0e758c17df4a"
FS_KEY = "96dbc0392691b437"
SIDEBAND_LOCATION = 12
OUTPUT_LOCATION = 1
MARKER = 0x43415252
RESULT_OPS = {43, 44, 59, 61, 65, 80, 124, 128, 132}
TYPE_OPS = set(range(19, 40))


def require(ok, message):
    if not ok:
        raise ValueError(message)


class Module:
    def __init__(self, data, expected_sha, model):
        require(bool(re.fullmatch("[0-9a-f]{64}", expected_sha)), "explicit source SHA256 required")
        require(hashlib.sha256(data).hexdigest() == expected_sha, "source SHA256 mismatch")
        self.data, self.sha = data, expected_sha
        self.header, self.ins = spv.decode(data)
        self.old_bound = self.header[3]
        self.defs, self.at, self.functions, self.decorations = {}, {}, {}, {}
        function = None
        for i, (op, a) in enumerate(self.ins):
            if op == 54:
                function = a[1]
            self.functions[i] = function
            if op in TYPE_OPS:
                result = a[0]
            elif op in RESULT_OPS:
                result = a[1]
            else:
                result = None
            if result is not None:
                require(result not in self.defs, "duplicate result ID")
                self.defs[result] = (op, a)
                self.at[result] = i
            if op == 71:
                self.decorations.setdefault(a[0], []).append(a[1:])
            if op == 56:
                function = None
        entries = [(i, a) for i, (op, a) in enumerate(self.ins) if op == 15]
        require(len(entries) == 1 and entries[0][1][0] == model, "requires one exact shader stage entry")
        self.entry_index, entry = entries[0]
        self.main = entry[1]
        self.annotations, self.declarations, self.before, self.after = [], [], {}, {}
        self.interfaces = []
        self.uint = self.find_type(21, [32, 0])
        self.float = self.find_type(22, [32])
        require(self.uint is not None and self.float is not None, "requires uint32 and float32")
        self.float4 = self.find_type(23, [self.float, 4])
        require(self.float4 is not None, "requires float4")
        require(not any(d[0] in (31, 32) for ds in self.decorations.values() for d in ds),
                "component/index-qualified IO is outside this diagnostic")

    def find_type(self, opcode, rest):
        return next((a[0] for op, a in self.ins + self.declarations if op == opcode and a[1:] == rest), None)

    def fresh(self):
        result = self.header[3]
        self.header[3] += 1
        return result

    def definition(self, value, opcode=None):
        require(value in self.defs, "missing definition")
        op, a = self.defs[value]
        require(opcode is None or op == opcode, "unexpected definition opcode")
        return a

    def constant_bits(self, value):
        a = self.definition(value, 43)
        require(len(a) == 3, "requires scalar32 constant")
        return a[2]

    def type(self, opcode, *rest):
        found = self.find_type(opcode, list(rest))
        if found is not None:
            return found
        result = self.fresh()
        self.declarations.append((opcode, [result, *rest]))
        return result

    def constant(self, type_, bits):
        for op, a in self.ins + self.declarations:
            if op == 43 and a[0] == type_ and a[2:] == [bits]:
                return a[1]
        result = self.fresh()
        self.declarations.append((43, [type_, result, bits]))
        return result

    def variable(self, type_, storage):
        pointer = self.type(32, storage, type_)
        result = self.fresh()
        self.declarations.append((59, [pointer, result, storage]))
        return result

    def location(self, storage, location):
        found = [a[1] for op, a in self.ins if op == 59 and a[2] == storage
                 and [30, location] in self.decorations.get(a[1], [])]
        require(len(found) == 1, "requires unique location")
        value = found[0]
        pointer = self.definition(self.definition(value, 59)[0], 32)
        require(pointer[1] == storage, "storage class mismatch")
        return value, pointer[2]

    def vacant(self, location):
        require(not any([30, location] in ds for ds in self.decorations.values()),
                "diagnostic location already occupied")

    def uses(self, value):
        """Conservative operand uses inside functions; unknown uses decline."""
        found = []
        for i, (op, a) in enumerate(self.ins):
            if self.functions.get(i) is None:
                continue
            # ExtInst's instruction number and composite selectors are literals,
            # not IDs. Runtime NClamp is81, also the actual input variable ID.
            if op == 12:
                operands = a[2:3] + a[4:]
            elif op == 81:
                operands = a[2:3]
            elif op in (79, 82):
                operands = a[2:4]
            else:
                operands = a[2:] if op in RESULT_OPS else a
            if value in operands:
                found.append((i, op, a))
        return found

    def new_interface(self, type_, storage, location, flat=False):
        value = self.variable(type_, storage)
        self.annotations.append((71, [value, 30, location]))
        if flat:
            self.annotations.append((71, [value, 14]))
        self.interfaces.append(value)
        return value

    def emit_result(self, code, opcode, type_, *args):
        result = self.fresh()
        code.append((opcode, [type_, result, *args]))
        return result

    def finish(self):
        result, added, entry_changes = [], [], {}
        annotated = declared = False
        def insert(code):
            for instruction in code:
                added.append(len(result))
                result.append(instruction)
        for i, (op, old) in enumerate(self.ins):
            if not annotated and op in TYPE_OPS:
                insert(self.annotations)
                annotated = True
            if op == 54 and not declared:
                insert(self.declarations)
                declared = True
            insert(self.before.get(i, []))
            a = list(old)
            if i == self.entry_index:
                entry_changes[str(len(result))] = list(old)
                a += self.interfaces
                if self.header[1] >= 0x10400:
                    a += [d[1] for oo, d in self.declarations if oo == 59 and d[2] == 6]
            result.append((op, a))
            insert(self.after.get(i, []))
        require(annotated and declared, "missing declaration/function sections")
        proof = {"source_bound": self.old_bound, "inserted_instruction_indices": added,
                 "entry_original_operands": entry_changes}
        out = spv.encode(self.header, result)
        require(restore(out, proof) == self.data, "original instruction restoration failed")
        return out, proof


def restore(data, proof):
    header, instructions = spv.decode(data)
    skipped = set(proof["inserted_instruction_indices"])
    restored = []
    for i, (op, a) in enumerate(instructions):
        if i in skipped:
            continue
        a = proof["entry_original_operands"].get(str(i), a)
        restored.append((op, a))
    header[3] = proof["source_bound"]
    return spv.encode(header, restored)


def audit_vertex(m):
    out, out_type = m.location(3, 0)
    require(out_type == m.float4, "VS out0 must be float4")
    m.vacant(SIDEBAND_LOCATION)
    matches = []
    for i, op, a in m.uses(out):
        require(op == 62 and a[0] == out, "VS output pointer escapes")
        value = a[1]
        if m.defs.get(value, (None,))[0] != 80:
            continue  # runtime output initialization is retained
        construct = m.definition(value, 80)
        if len(construct) != 6 or construct[0] != m.float4:
            continue
        bitcast = m.defs.get(construct[2])
        if not bitcast or bitcast[0] != 124:
            continue
        require(bitcast[1][0] == m.float, "VS carrier bitcast must produce float32")
        raw = bitcast[1][2]
        load = m.definition(raw, 61)
        require(len(load) == 3 and load[0] == m.uint, "VS raw source must be uint load")
        chain = m.definition(load[2], 65)
        require(len(chain) == 5 and m.constant_bits(chain[3]) == 0, "VS source is not runtime-array word")
        address = m.definition(chain[4], 128)
        require(m.constant_bits(address[3]) == 4, "VS source is not word4")
        base = m.definition(address[2], 132)
        require(m.constant_bits(base[3]) == 21, "VS vertex stride is not21 words")
        vertex_bits = m.definition(base[2], 124)
        vertex_load = m.definition(vertex_bits[2], 61)
        require([11, 42] in m.decorations.get(vertex_load[2], []), "VS index is not VertexIndex")
        require([m.constant_bits(v) for v in construct[3:]] == [0, 0, 0x3f800000],
                "VS remaining output components changed")
        require(m.functions[m.at[raw]] == m.main, "VS source must be in main")
        matches.append(raw)
    require(len(matches) == 1, "requires exactly one word4 out0 store")
    return out, matches[0]


def audit_fragment(m):
    inp, type_ = m.location(1, 0)
    out, out_type = m.location(3, 0)
    require(type_ == m.float4 and out_type == m.float4, "FS input0/output0 must be float4")
    require([14] in m.decorations.get(inp, []), "FS input0 must be Flat")
    m.vacant(SIDEBAND_LOCATION)
    # Location1 may be a vertex INPUT in this shader. Only output1 must be vacant.
    require(not any(op == 59 and a[2] == 3 and [30, OUTPUT_LOCATION] in m.decorations.get(a[1], [])
                    for op, a in m.ins), "FS diagnostic output occupied")
    uses = m.uses(inp)
    require(len(uses) == 1 and uses[0][1] == 65, "FS input0 needs one scalar access")
    chain = uses[0][2]
    require(len(chain) == 4 and m.constant_bits(chain[3]) == 0, "FS input component is not x")
    loads = m.uses(chain[1])
    require(len(loads) == 1 and loads[0][1] == 61, "FS input pointer escapes")
    immediate = loads[0][2][1]
    require(loads[0][2][0] == m.float and len(loads[0][2]) == 3, "FS input load shape")
    stores = m.uses(immediate)
    require(len(stores) == 1 and stores[0][1] == 62, "FS immediate input has other uses")
    local = stores[0][2][0]
    localvar = m.definition(local, 59)
    require(localvar[2] == 7 and m.definition(localvar[0], 32)[2] == m.float, "FS local must be Function float")
    local_uses = m.uses(local)
    require(sum(op == 62 and a[0] == local for _, op, a in local_uses) == 1, "FS local has additional stores")
    recoveries = []
    for i, op, a in local_uses:
        if op == 62:
            continue
        require(op == 61 and a[0] == m.float and len(a) == 3, "FS local escapes")
        bitcasts = m.uses(a[1])
        require(len(bitcasts) == 1 and bitcasts[0][1] == 124, "FS local load is not raw recovered bits")
        recovery = bitcasts[0][2]
        require(m.definition(recovery[0], 21)[1:] == [32, 1], "FS recovery must be signed32 bitcast")
        recoveries.append((bitcasts[0][0], recovery[1]))
    require(len(recoveries) == 2, "requires exactly two recovered local reads")
    recoveries.sort()
    require(all(m.functions[m.at[v]] == m.main for v in [immediate, recoveries[0][1]]),
            "FS observations must be in main")
    return inp, out, local, immediate, recoveries[0][1]


def instrument_pair(vertex, fragment, vertex_sha, fragment_sha):
    vs, fs = Module(vertex, vertex_sha, 0), Module(fragment, fragment_sha, 4)
    vs_out, raw = audit_vertex(vs)
    fs_in, fs_out, local, immediate, recovered = audit_fragment(fs)
    side_vs = vs.new_interface(vs.uint, 3, SIDEBAND_LOCATION, True)
    vs.after[vs.at[raw]] = [(62, [side_vs, raw])]
    side_fs = fs.new_interface(fs.uint, 1, SIDEBAND_LOCATION, True)
    uint4 = fs.type(23, fs.uint, 4)
    debug = fs.new_interface(uint4, 3, OUTPUT_LOCATION)
    immediate_var = fs.variable(fs.uint, 6)
    debug_var = fs.variable(uint4, 6)
    zero = fs.constant(fs.uint, 0)
    zero4 = fs.fresh()
    fs.declarations.append((44, [uint4, zero4, zero, zero, zero, zero]))
    marker = fs.constant(fs.uint, MARKER)
    first_label = next(i for i, (op, a) in enumerate(fs.ins) if op == 248 and fs.functions[i] == fs.main)
    start = first_label + 1
    while fs.ins[start][0] in (59, 8, 317):
        start += 1
    fs.before[start] = [(62, [immediate_var, zero]), (62, [debug_var, zero4])]
    code = []
    bits = fs.emit_result(code, 124, fs.uint, immediate)
    code.append((62, [immediate_var, bits]))
    fs.after[fs.at[immediate]] = code
    code = []
    side = fs.emit_result(code, 61, fs.uint, side_fs)
    immediate_bits = fs.emit_result(code, 61, fs.uint, immediate_var)
    original_bits = fs.emit_result(code, 124, fs.uint, recovered)
    value = fs.emit_result(code, 80, uint4, side, immediate_bits, original_bits, marker)
    code.append((62, [debug_var, value]))
    fs.after[fs.at[recovered]] = code
    returns = 0
    for i, (op, a) in enumerate(fs.ins):
        if op == 253 and fs.functions[i] == fs.main:
            code = []
            value = fs.emit_result(code, 61, uint4, debug_var)
            code.append((62, [debug, value]))
            fs.before.setdefault(i, []).extend(code)
            returns += 1
    require(returns > 0, "FS has no main return")
    vertex_out, vertex_proof = vs.finish()
    fragment_out, fragment_proof = fs.finish()
    manifest = {"schema": "flat-carrier-observer-uint-v1", "diagnostic_only": True,
                "interface": {"location": SIDEBAND_LOCATION, "type": "uint32", "interpolation": "Flat"},
                "output": {"location": OUTPUT_LOCATION, "type": "uint32x4", "format": "VK_FORMAT_R32G32B32A32_UINT",
                           "channels": ["VS native word4", "immediate original FS input0.x bits",
                                        "original recovered local bits", "coverage marker"],
                           "marker": MARKER},
                "vertex": {"key": VS_KEY, "source_sha256": vertex_sha, "output_sha256": hashlib.sha256(vertex_out).hexdigest(),
                           "source_word_id": raw, "sideband_id": side_vs, "original_output_id": vs_out, "proof": vertex_proof},
                "fragment": {"key": FS_KEY, "source_sha256": fragment_sha, "output_sha256": hashlib.sha256(fragment_out).hexdigest(),
                             "input_id": fs_in, "local_id": local, "immediate_id": immediate, "recovered_id": recovered,
                             "sideband_id": side_fs, "debug_output_id": debug, "original_output_id": fs_out,
                             "proof": fragment_proof},
                "preservation": "All original instructions remain byte-identical/in original order; only entrypoint interfaces are extended. Removing indexed additions restores exact source bytes. No new FP arithmetic, sampling, derivatives, branches, or descriptor resources.",
                "limits": "Requires atomic paired activation and isolated UINT MRT. Extra IO may change driver scheduling/optimization; retain original color0 control. Discarded fragments do not write debug, and an unexecuted recovery writes marker0."}
    return vertex_out, fragment_out, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vertex", required=True, type=Path)
    parser.add_argument("--fragment", required=True, type=Path)
    parser.add_argument("--origin", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    origin = json.loads(args.origin.read_text())
    require(origin["vertex"]["key"] == VS_KEY and origin["fragment"]["key"] == FS_KEY, "wrong shader pair")
    require(origin.get("runtime_normalized") is True, "requires exact runtime-normalized origin")
    vertex, fragment, manifest = instrument_pair(args.vertex.read_bytes(), args.fragment.read_bytes(),
        origin["vertex"]["sha256"], origin["fragment"]["sha256"])
    args.out.mkdir(parents=True, exist_ok=True)
    for key, data in ((VS_KEY, vertex), (FS_KEY, fragment)):
        path = args.out / (key + ".spv")
        require(not path.exists(), "refuse to overwrite an observer artifact")
        path.write_bytes(data)
        subprocess.run(["spirv-val", "--target-env", "vulkan1.1", str(path)], check=True)
    manifest["origin"] = origin
    manifest["tool_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({s: manifest[s]["output_sha256"] for s in ("vertex", "fragment")}))


if __name__ == "__main__":
    main()

