#!/usr/bin/env python3
"""inventory.py <shaders.dmsv>: what a game's translated shaders ask of the GPU, per NVN stage: instruction counts, workgroup sizes, shared (Workgroup)
memory, private arrays (Function/Private storage: local memory, register-pressure and scratch proxy), subgroup operations, 16/64-bit types, capabilities."""
import struct, sys, collections, statistics
from compression import zstd

MAGIC, HDR, ENT = b'DMSV', struct.Struct('<4sII'), struct.Struct('<QIIIIII')
STAGE = {0: "vertex", 1: "fragment", 2: "geometry", 3: "tess-ctrl", 4: "tess-eval", 5: "compute", 6: "task", 7: "mesh"}
CAPS = {9: "Float16", 10: "Float64", 11: "Int64", 22: "Int16", 39: "Int8", 61: "GroupNonUniform", 62: "GNU-Vote", 63: "GNU-Arith", 64: "GNU-Ballot",
        65: "GNU-Shuffle", 66: "GNU-ShuffleRel", 67: "GNU-Clustered", 68: "GNU-Quad", 4433: "StorageBuffer16", 4448: "StorageBuffer8",
        5345: "PhysicalStorageBufferAddresses", 4435: "StorageUniform16", 5568: "SubgroupBallotKHR", 12: "Int64Atomics", 5009: "FragmentShaderInterlock?",
        1: "Shader", 56: "StorageImageWriteWithoutFormat", 55: "StorageImageReadWithoutFormat", 5282: "MeshShadingEXT", 4423: "DrawParameters"}

def parse(spv):
    w = struct.unpack_from("<%dI" % (len(spv) // 4), spv)
    i = 5; ninstr = 0; caps = set(); local = None; types = {}; consts = {}; ptrs = {}; vars_ = []; gnu = collections.Counter()
    while i < len(w):
        op = w[i] & 0xffff; n = w[i] >> 16
        if n == 0: break
        a = w[i + 1:i + n]; ninstr += 1
        if op == 17: caps.add(a[0])                                   # OpCapability
        elif op == 16 and a[1] == 17: local = tuple(a[2:5])          # OpExecutionMode LocalSize
        elif op == 21: types[a[0]] = ("int", a[1] // 8)              # OpTypeInt
        elif op == 22: types[a[0]] = ("float", a[1] // 8)            # OpTypeFloat
        elif op == 20: types[a[0]] = ("bool", 4)
        elif op == 23: types[a[0]] = ("vec", a[1], a[2])             # OpTypeVector
        elif op == 24: types[a[0]] = ("mat", a[1], a[2])
        elif op == 28: types[a[0]] = ("arr", a[1], a[2])             # OpTypeArray elem, length id
        elif op == 30: types[a[0]] = ("struct", tuple(a[1:]))
        elif op == 43: consts[a[1]] = a[2]                            # OpConstant type id value
        elif op == 32: ptrs[a[0]] = (a[1], a[2])                      # OpTypePointer storage class, type
        elif op == 59: vars_.append((a[0], a[1], a[2]))               # OpVariable type id storage
        elif 333 <= op <= 366: gnu[op] += 1                           # OpGroupNonUniform*
        i += n
    def size(t, depth=0):
        if depth > 20 or t not in types: return 0
        k = types[t]
        if k[0] in ("int", "float", "bool"): return k[1]
        if k[0] == "vec": return size(k[1], depth + 1) * k[2]
        if k[0] == "mat": return size(k[1], depth + 1) * k[2]
        if k[0] == "arr": return size(k[1], depth + 1) * consts.get(k[2], 1)
        if k[0] == "struct": return sum(size(m, depth + 1) for m in k[1])
        return 0
    shared = sum(size(ptrs[t][1]) for t, _, sc in vars_ if sc == 4 and t in ptrs)                       # Workgroup
    private = sum(size(ptrs[t][1]) for t, _, sc in vars_ if sc in (6, 7) and t in ptrs and types.get(ptrs[t][1], ("",))[0] == "arr")   # Private/Function arrays
    return ninstr, caps, local, shared, private, gnu

def main():
    data = open(sys.argv[1], "rb").read()
    magic, ver, count = HDR.unpack_from(data, 0); assert magic == MAGIC
    per = collections.defaultdict(list); z = zstd
    for k in range(count):
        key, stage, off, ln, roff, rlen, sz = ENT.unpack_from(data, HDR.size + k * ENT.size)
        spv = data[off:off + ln]
        if sz: spv = z.decompress(spv)
        try: per[stage].append((key,) + parse(spv))
        except Exception as e: pass
    for st in sorted(per):
        L = per[st]; ni = [x[1] for x in L]
        print(f"== {STAGE.get(st, st)}: {len(L)} shaders; SPIR-V instructions median {statistics.median(ni):.0f}, p95 {sorted(ni)[int(len(ni)*.95)]}, max {max(ni)}")
        capc = collections.Counter(c for x in L for c in x[2]); print("   capabilities:", ", ".join(f"{CAPS.get(c, c)} {n}" for c, n in capc.most_common(14)))
        gn = sum(1 for x in L if x[6]); print(f"   with subgroup (warp) operations: {gn}" + (" — ops: " + ", ".join(f"{o}:{n}" for o, n in sum((x[6] for x in L), collections.Counter()).most_common(8)) if gn else ""))
        pv = [x[5] for x in L if x[5]]; print(f"   with private arrays: {len(pv)}" + (f"; bytes median {statistics.median(pv):.0f}, max {max(pv)}" if pv else ""))
        if st in (5, 6, 7):
            sh = [x[4] for x in L]; ls = collections.Counter(x[3] for x in L)
            print(f"   shared memory: none {sum(1 for s in sh if not s)}, >16K {sum(1 for s in sh if s > 16384)}, >32K {sum(1 for s in sh if s > 32768)}, max {max(sh)}")
            print("   workgroup sizes:", ", ".join(f"{'x'.join(map(str, l)) if l else '?'} {n}" for l, n in ls.most_common(8)))
        big = sorted(L, key=lambda x: -x[1])[:3]; print("   largest:", ", ".join(f"{x[0]:016x} ({x[1]} instr)" for x in big))
main()
