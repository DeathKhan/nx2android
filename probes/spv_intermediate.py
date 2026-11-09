#!/usr/bin/env python3
"""Offline, diagnostic-only fragment SPIR-V intermediate-output probes.

extract writes the exact DMSV module/reflection and their origin manifest.
probe preserves original instructions and captures one float SSA result, or
selected components of one fragment input, into an existing location-0 vec4.
It adds no descriptors or sampling. Never use these outputs as production shaders.
"""
import argparse
import hashlib
import json
import mmap
from pathlib import Path
import re
import struct
import subprocess
import sys

MAGIC = 0x07230203
FLOAT, VECTOR, POINTER, CONSTANT, COMPOSITE, VARIABLE = 22, 23, 32, 43, 44, 59
FUNCTION, END, LOAD, STORE, ACCESS = 54, 56, 61, 62, 65
PHI, LABEL, RETURN = 245, 248, 253

def sha(b):
    return hashlib.sha256(b).hexdigest()

def decode(data):
    if len(data) < 20 or len(data) % 4:
        raise ValueError("not a whole SPIR-V module")
    w = list(struct.unpack("<%dI" % (len(data)//4), data))
    if w[0] != MAGIC:
        raise ValueError("not little-endian SPIR-V")
    ins = []
    at = 5
    while at < len(w):
        n, op = w[at] >> 16, w[at] & 65535
        if not n or at+n > len(w):
            raise ValueError("malformed instruction at word %d" % at)
        ins.append((op, w[at+1:at+n]))
        at += n
    return w[:5], ins

def encode(header, ins):
    w = list(header)
    for op, a in ins:
        w += [(len(a)+1) << 16 | op] + list(a)
    return struct.pack("<%dI" % len(w), *w)

def string_words(s):
    b = s.encode()+b"\0"
    b += b"\0"*((-len(b)) % 4)
    return list(struct.unpack("<%dI" % (len(b)//4), b))

def entry_string(a):
    raw = struct.pack("<%dI" % (len(a)-2), *a[2:])
    end = raw.index(0)
    return raw[:end].decode(), 2+(end+4)//4

def module_info(ins):
    types, vars_, locs, index = {}, {}, {}, {}
    eps = []
    for op, a in ins:
        if op in range(19, 40):
            types[a[0]] = (op, a[1:])
        elif op == VARIABLE:
            vars_[a[1]] = (a[0], a[2])
        elif op == 71 and len(a) >= 3:
            if a[1] == 30: locs[a[0]] = a[2]
            if a[1] == 32: index[a[0]] = a[2]
        elif op == 15:
            name, off = entry_string(a)
            eps.append((a[0], a[1], name, a[off:]))
    return types, vars_, locs, index, eps

def instrument(data, *, target=None, input_location=None, components="xy", entry="main"):
    if (target is None) == (input_location is None):
        raise ValueError("select exactly one SSA result or input location")
    h, ins = decode(data)
    types, vars_, locs, indexes, eps = module_info(ins)
    ep = [e for e in eps if e[0] == 4 and e[2] == entry]
    if len(ep) != 1 or len(eps) != 1:
        raise ValueError("requires one selected Fragment entrypoint")
    main = ep[0][1]
    outs = [v for v, loc in locs.items() if loc == 0 and
            v in vars_ and vars_[v][1] == 3 and indexes.get(v, 0) == 0]
    if len(outs) != 1:
        raise ValueError("requires one ordinary location-0 Output variable")
    out = outs[0]
    ptr_op, ptr = types[vars_[out][0]]
    if ptr_op != POINTER or ptr[0] != 3:
        raise ValueError("unexpected output pointer")
    v4 = ptr[1]
    if types[v4][0] != VECTOR or types[v4][1][1] != 4:
        raise ValueError("output must be float32 vec4")
    f32 = types[v4][1][0]
    if types[f32] != (FLOAT, [32]):
        raise ValueError("output must be float32 vec4")

    nextid = h[3]
    def alloc():
        nonlocal nextid
        r = nextid
        nextid += 1
        return r
    decl = []
    def add(op, a):
        decl.append((op, a))
    def constant(bits):
        for op, a in ins:
            if op == CONSTANT and a[0] == f32 and a[2:] == [bits]:
                return a[1]
        r = alloc()
        add(CONSTANT, [f32, r, bits])
        return r
    zero, one = constant(0), constant(0x3f800000)
    z4 = alloc()
    add(COMPOSITE, [v4, z4, zero, zero, zero, zero])
    pp = next((t for t, x in types.items() if x == (POINTER, [6, v4])), None)
    if pp is None:
        pp = alloc()
        add(POINTER, [pp, 6, v4])  # Private, per invocation
    debug = alloc()
    # The global variable follows all added declarations (including input types).
    captures, selected = {}, {}
    target_index = None
    function = None
    first_main_label = None
    main_returns = 0
    for i, (op, a) in enumerate(ins):
        if op == FUNCTION: function = a[1]
        if op == LABEL and function == main and first_main_label is None:
            first_main_label = i
        if op == RETURN and function == main: main_returns += 1
        if target is not None and function is not None and len(a) >= 2 and a[0] in types and a[1] == target:
            if target_index is not None:
                raise ValueError("duplicate selected result ID")
            if op in (FUNCTION, 55, VARIABLE):
                raise ValueError("select a float value definition, not a function/pointer")
            t = a[0]
            if t == f32:
                count = 1
            elif types[t][0] == VECTOR and types[t][1][0] == f32 and 2 <= types[t][1][1] <= 4:
                count = types[t][1][1]
            else:
                raise ValueError("selected result must be float32 scalar/vec2/vec3/vec4")
            seq = []
            if count == 4:
                value = target
            else:
                lanes = [target] if count == 1 else []
                for c in range(count if count > 1 else 0):
                    r = alloc()
                    seq.append((81, [f32, r, target, c]))
                    lanes.append(r)
                lanes += [zero]*(3-len(lanes)) + [one]
                value = alloc()
                seq.append((80, [v4, value] + lanes))
            seq.append((STORE, [debug, value]))
            after = i
            # OpPhi instructions must stay contiguous at the start of a block.
            if op == PHI:
                while after+1 < len(ins) and ins[after+1][0] in (PHI, 8, 317):
                    after += 1
            captures[after] = seq
            target_index = i
            selected = {"kind": "ssa", "result_id": target, "opcode": op,
                        "type_id": t, "components": count, "function_id": function,
                        "capture_after_instruction": after}
        if op == END: function = None
    if first_main_label is None or not main_returns:
        raise ValueError("main must have a block and ordinary return")
    if target is not None and target_index is None:
        raise ValueError("selected SSA value definition not found in a function")

    init = [(STORE, [debug, z4])]
    if input_location is not None:
        if not components or len(components) > 4 or any(c not in "xyzw" for c in components):
            raise ValueError("input components must contain one to four xyzw letters")
        inputs = [v for v, loc in locs.items() if loc == input_location and
                  v in vars_ and vars_[v][1] == 1]
        if len(inputs) != 1:
            raise ValueError("requires one Input variable at selected location")
        inp = inputs[0]
        ptype = types[vars_[inp][0]]
        itype = ptype[1][1]
        if ptype[0] != POINTER or types[itype][0] != VECTOR or types[itype][1][0] != f32:
            raise ValueError("selected input must be a float32 vector")
        ncomp = types[itype][1][1]
        if any("xyzw".index(c) >= ncomp for c in components):
            raise ValueError("component outside selected input vector")
        it = next((t for t, x in types.items() if x == (21, [32, 0])), None)
        if it is None:
            it = alloc()
            add(21, [it, 32, 0])
        pi = next((t for t, x in types.items() if x == (POINTER, [1, f32])), None)
        if pi is None:
            pi = alloc()
            add(POINTER, [pi, 1, f32])
        lanes = []
        for c in components:
            ci, ptrid, value = alloc(), alloc(), alloc()
            add(CONSTANT, [it, ci, "xyzw".index(c)])
            init += [(ACCESS, [pi, ptrid, inp, ci]), (LOAD, [f32, value, ptrid])]
            lanes.append(value)
        if len(lanes) < 4:
            lanes += [zero]*(3-len(lanes)) + [one]
        value = alloc()
        init += [(80, [v4, value] + lanes), (STORE, [debug, value])]
        selected = {"kind": "input", "location": input_location,
                    "variable_id": inp, "components": components}
    add(VARIABLE, [pp, debug, 6])
    # Main's local declarations must precede initialization/capture.
    init_after = first_main_label
    while init_after+1 < len(ins) and ins[init_after+1][0] in (VARIABLE, 8, 317):
        init_after += 1
    result = []
    declared = False
    function = None
    for i, (op, original) in enumerate(ins):
        a = list(original)
        if op == FUNCTION and not declared:
            result.extend(decl)
            declared = True
        if op == 15 and h[1] >= 0x00010400:
            a.append(debug)  # SPIR-V1.4+ requires used Private globals in interface.
        if op == FUNCTION: function = a[1]
        if op == RETURN and function == main:
            value = alloc()
            result += [(LOAD, [v4, value, debug]), (STORE, [out, value])]
        result.append((op, a))
        if i == init_after: result.extend(init)
        result.extend(captures.get(i, []))
        if op == END: function = None
    h[3] = nextid
    return encode(h, result), {
        "selected": selected, "entrypoint_id": main, "output_variable_id": out,
        "private_debug_id": debug, "main_returns_overridden": main_returns,
        "unexecuted_value": [0, 0, 0, 0],
        "discard_behavior": "original kills/termination remain; discarded fragments write no diagnostic",
        "execution_note": "Original instructions stay in order; compiler optimization and timing can change. Original side effects remain, including any feedback writes.",
        "attachment_limit": "Existing attachment format, blending, masks and depth tests still apply. R11G11B10F clips negative values and has no alpha channel."
    }

def validate(path, env):
    r = subprocess.run(["spirv-val", "--target-env", env, str(path)], capture_output=True, text=True)
    msg = r.stdout+r.stderr
    # Inserting instructions changes diagnostic line numbers, not original IDs.
    normalized = re.sub(r"line \d+", "line N", msg)
    return {"exit_code": r.returncode, "message": msg, "normalized": normalized}

def save_json(path, obj):
    path.write_text(json.dumps(obj, indent=2)+"\n")

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    e = sub.add_parser("extract")
    e.add_argument("--bundle", type=Path, required=True)
    e.add_argument("--key", required=True)
    e.add_argument("--stage", type=int, default=1)
    e.add_argument("--out", type=Path, required=True)
    e.add_argument("--target-env", default="vulkan1.1")
    rt = sub.add_parser("adopt-runtime", help="record an exact host runtime dump; do not emulate its rewrites")
    rt.add_argument("--source", type=Path, required=True)
    rt.add_argument("--metadata", type=Path, required=True)
    rt.add_argument("--bundle-origin", type=Path, required=True)
    rt.add_argument("--out", type=Path, required=True)
    rt.add_argument("--target-env", default="vulkan1.1")
    x = sub.add_parser("probe")
    x.add_argument("--source", type=Path, required=True)
    x.add_argument("--origin", type=Path, required=True)
    x.add_argument("--out", type=Path, required=True)
    pick = x.add_mutually_exclusive_group(required=True)
    pick.add_argument("--target", type=lambda s:int(s.lstrip("%"),0))
    pick.add_argument("--input-location", type=int)
    x.add_argument("--components", default="xy")
    x.add_argument("--entry", default="main")
    x.add_argument("--label", required=True)
    x.add_argument("--target-env", default="vulkan1.1")
    a = p.parse_args(argv)
    if a.command == "extract":
        a.out.mkdir(parents=True, exist_ok=True)
        key = int(a.key, 16)
        with a.bundle.open("rb") as f, mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ) as b:
            if len(b)<12 or b[:4]!=b"DMSV": raise ValueError("not DMSV")
            version,n=struct.unpack_from("<II",b,4)
            if version not in (1,2) or 12+32*n>len(b): raise ValueError("invalid DMSV table")
            found=[]
            for i in range(n):
                ent=struct.unpack_from("<Q6I",b,12+32*i)
                if ent[0]==key and ent[1]==a.stage: found.append(ent)
            if len(found)!=1: raise ValueError("key/stage not unique in bundle")
            ent=found[0]; _,_,so,sl,ro,rl,raw=ent
            if so+sl>len(b) or ro+rl>len(b): raise ValueError("entry exceeds bundle")
            spv=bytes(b[so:so+sl]); refl=bytes(b[ro:ro+rl])
            if raw:
                spv=subprocess.run(["zstd","-d","-q","-c"],input=spv,capture_output=True,check=True).stdout
                if len(spv)!=raw: raise ValueError("decompressed size mismatch")
            bundle_sha=sha(b)
        decode(spv)
        source=a.out/"original.spv"
        source.write_bytes(spv)
        (a.out/"reflection.bin").write_bytes(refl)
        subprocess.run(["spirv-dis","--raw-id",str(source),"-o",str(a.out/"original.spvasm")],check=True)
        obj={"schema":"spv-intermediate-origin-v1","diagnostic_only":True,
             "bundle":str(a.bundle.resolve()),"bundle_sha256":bundle_sha,
             "bundle_version":version,"key":"%016x"%key,"stage":a.stage,
             "entry":list(ent),"source_sha256":sha(spv),"reflection_sha256":sha(refl),
             "source_bytes":len(spv),"validation":validate(source,a.target_env)}
        save_json(a.out/"origin.json",obj)
        print(json.dumps(obj,indent=2))
        return 0
    if a.command == "adopt-runtime":
        data=a.source.read_bytes();header,ins=decode(data)
        metadata=json.loads(a.metadata.read_text())
        bundle=json.loads(a.bundle_origin.read_text())
        if metadata.get("schema")!="vk-runtime-spv-v1" or metadata.get("key")!=bundle.get("key") or metadata.get("stage")!=bundle.get("stage"):
            raise ValueError("runtime metadata does not match bundle key/stage")
        if metadata.get("word_count")*4!=len(data) or not metadata.get("requires_alt_runtime"):
            raise ValueError("runtime metadata size/mode mismatch")
        a.out.mkdir(parents=True,exist_ok=True)
        source=a.out/"original.spv";source.write_bytes(data)
        obj={"schema":"spv-intermediate-runtime-origin-v1","diagnostic_only":True,
             "key":bundle["key"],"stage":bundle["stage"],"source_sha256":sha(data),
             "source_bytes":len(data),"bundle_origin":bundle,"runtime_metadata":metadata,
             "requires_alt_runtime":True,
             "validation":validate(source,a.target_env),
             "provenance_limit":"Bundle key/stage association is recorded by the operator; runtime ALT verifies the exact current preamble before loading."}
        save_json(a.out/"origin.json",obj)
        subprocess.run(["spirv-dis","--raw-id",str(source),"-o",str(a.out/"original.spvasm")],check=True)
        print(json.dumps(obj,indent=2));return 0
    source=a.source.read_bytes()
    origin=json.loads(a.origin.read_text())
    if origin.get("source_sha256")!=sha(source):
        raise ValueError("source does not match origin manifest")
    out,meta=instrument(source,target=a.target,input_location=a.input_location,
                        components=a.components,entry=a.entry)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_bytes(out)
    baseline=validate(a.source,a.target_env); check=validate(a.out,a.target_env)
    same=baseline["exit_code"]==check["exit_code"] and baseline["normalized"]==check["normalized"]
    meta.update({"schema":"spv-intermediate-probe-v1","diagnostic_only":True,
        "label":a.label,"origin":origin,"probe_sha256":sha(out),
        "tool_sha256":sha(Path(__file__).read_bytes()),"target_env":a.target_env,
        "baseline_validation":baseline,"probe_validation":check,
        "same_validation_result":same,
        "validation_limit":"A pre-existing validator failure can hide later errors; matching first failure alone is not full validation."})
    save_json(Path(str(a.out)+".json"),meta)
    subprocess.run(["spirv-dis","--raw-id",str(a.out),"-o",str(a.out)+".spvasm"],check=True)
    print(json.dumps({"out":str(a.out),"sha256":sha(out),"selected":meta["selected"],
                      "validator":check["exit_code"],"same_baseline":same}))
    return 0 if same else 2

if __name__=="__main__":
    try: sys.exit(main())
    except (ValueError, OSError, subprocess.CalledProcessError) as e:
        print(str(e),file=sys.stderr);sys.exit(1)

