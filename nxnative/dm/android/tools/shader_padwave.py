#!/usr/bin/env python3
"""32-lane compute workgroups padded to 64 lanes for a GPU whose waves are 64 lanes (the Odin 3's stock Adreno 830 driver, 512.800.72).

Ocarina of Time (Switch 2) runs an NDVE picture-enhancement network over each decoded movie picture: 22 compute kernels of up to 1.8 MB of SPIR-V, 17 of them LocalSize 32xNx1
with hundreds of OpGroupNonUniformShuffle and OpControlBarrier (Ampere warp code, translated with Ryujinx's 64-lane warp emulation: the warp lane is the subgroup lane & 31,
a shuffle's index carries the lane's upper bit so it stays inside its own 32). Qualcomm's compiler fails them (vkCreateComputePipelines → VK_ERROR_UNKNOWN, or std::bad_alloc):
the same code with LocalSize 64 1 1 compiles, the pack's 16x8 kernels without shuffles compile, and other games' 128-lane kernels with shuffles compile. So the failure is a
32-lane workgroup with wave-wide operations on a 64-lane wave. pad() makes the workgroup 64xNx1 with the same dispatch geometry, in a form that keeps the 32-lane semantics:

  LocalSize 32 N 1 → 64 N 1. Lanes with LocalInvocationId.x >= 32 are padding (one real and one padding half of each 64-lane subgroup; a workgroup of N rows is N subgroups).
  Padding lane x mirrors real lane x - 32: every read of LocalInvocationId.x is masked & 31 (and a LocalInvocationIndex is recomputed from the masked id), so the padding lane
  computes the same addresses, the same shared-memory slots and the same branch conditions as its real lane, and reaches every OpControlBarrier with it (control flow is otherwise
  untouched, so the barriers stay convergent). Its shuffles read the padding half, whose values mirror the real half (the translator's index already keeps a shuffle inside the
  lane's own 32: index | (SubgroupLocalInvocationId & ~31)). Reads of shared arrays sized for 32 lanes stay in bounds for the same reason.
  Its side effects are predicated off: every OpStore and OpCopyMemory to Workgroup, Uniform (the translator's storage buffers), StorageBuffer, PhysicalStorageBuffer or Image
  memory, every OpImageWrite and every OpAtomic* except OpAtomicLoad runs under `if (LocalInvocationId.x < 32)`, a selection construct of its own (SPIR-V has no predicated store;
  a run of consecutive side effects shares one). An atomic's result reaches the merge block through an OpPhi (OpUndef on the padding side). Function-storage and Private stores
  are per-invocation and stay as they are. A helper function with side effects gets the predicate at its entry from the built-in (a global), so a function called 640 times is
  predicated once in its body, not at each call. A block whose OpSelectionMerge precedes a side effect keeps the merge with its branch at the end of its last piece; a loop header
  (OpLoopMerge) with a side effect is split into the header (its OpPhis, the OpLoopMerge, an OpBranch) and a body block that holds the rest, so the back edge still reaches the header.
  OpPhi operands that named a split block now name its last piece. A WorkgroupSize built-in constant, if any, gets x = 64 too (it overrides the execution mode).
  Not changed: dispatch group counts (the game's; NumWorkgroups and WorkgroupId are as before), bindings, descriptor sets, SubgroupLocalInvocationId (the translator's warp code
  masks it itself), subgroup operations. Modules with OpGroupNonUniformBallot / Vote / Elect / Broadcast would see 64 lanes where the warp code expects a 32-lane ballot word per
  half, and the translator's 64-lane emulation already picks the word by lane >> 5 (shader_vk.py SUBGROUP); nothing here changes that, pad() only notes them (info()).
  Bails (ValueError) on: a LocalSize whose x is not 32, a LocalInvocationId read through a non-constant index, a LocalInvocationId or LocalInvocationIndex passed by pointer, a
  loop header holding a side effect whose terminator is conditional (the body block would need a merge of its own).

    pad(spv, pad_to=64, mode='branch') → the padded module's bytes, and a dict of what was done: {'stores': {kind: n}, 'atomics': n, 'image_writes': n, 'functions': n predicated, 'lid_loads': n,
                           'barriers': n, 'shuffles': n, 'local': [64, N, 1]}
    mode 'select' (branchless padding): no selection construct for a Workgroup store or an image write. Each Workgroup array grows by one spare element and a padding
      lane's store index is OpSelect'ed to it (the lane's own slot otherwise: the access chain's one index; Qualcomm's compiler ran out of heap on the ~280 small
      blocks mode 'branch' adds to a 1.4 MB kernel, where the plain LocalSize-64 probe compiled); an image write's coordinate is OpSelect'ed to -1 (out of bounds:
      Vulkan drops the store). Only a BDA / storage-buffer store and an atomic keep a branch (one or two blocks a module here). A Workgroup store through anything
      but a one-index access chain on a Workgroup variable falls back to the branch.
    shuffle_xor(spv) → (module, n) with every OpGroupNonUniformShuffle whose index is provably `lane ^ constant` (lane = SubgroupLocalInvocationId & 31 or the masked
                       LocalInvocationId.x, the warp base ORed in or not) rewritten to OpGroupNonUniformShuffleXor, and how many were; the rest keep their OpGroupNonUniformShuffle
    info(spv) → {'local': [x, y, z], 'shuffles': n, 'barriers': n, 'ballot_votes': [op names]}: what a module holds, for shader_vk.py to pick the kernels the pass is for
    wants_pad(spv, x=32) → True for a compute module with LocalSize x == 32 (any y, z 1) that uses a subgroup operation

shader_vk.py applies pad() when the game profile says `android_pad_wave 32` (dm/games/<id>/game.tsv); the bundle's reflection header then carries the padded local size
(gfx_vk_shader.c reads it as the kernel's; gfx_vk_core.c vk_sg_pin only checks the workgroup fits the pinned subgroup count). The command line pads one file: shader_padwave.py in.spv out.spv
[--xor] [--to 64]."""
import collections, struct, sys

OP_NAME, OP_STRING, OP_EXT_IMPORT, OP_ENTRY, OP_EXEC_MODE, OP_TYPE_BOOL, OP_TYPE_INT, OP_TYPE_VECTOR, OP_TYPE_PTR, OP_TYPE_FN = 5, 7, 11, 15, 16, 20, 21, 23, 32, 33
OP_CONST, OP_CONST_COMPOSITE, OP_FUNCTION, OP_FN_PARAM, OP_FUNCTION_END, OP_CALL, OP_VARIABLE, OP_LOAD, OP_STORE, OP_COPY_MEMORY, OP_COPY_MEMORY_SIZED = 43, 44, 54, 55, 56, 57, 59, 61, 62, 63, 64
OP_ACCESS, OP_IN_ACCESS, OP_PTR_ACCESS, OP_DECORATE, OP_DECORATION_GROUP, OP_EXTRACT, OP_INSERT, OP_IMAGE_WRITE, OP_UNDEF = 65, 66, 67, 71, 73, 81, 82, 99, 1
OP_BITCAST, OP_IADD, OP_IMUL, OP_ULESS, OP_AND, OP_OR, OP_XOR, OP_SHL, OP_PHI, OP_LOOP_MERGE, OP_SEL_MERGE, OP_LABEL, OP_BRANCH, OP_BRANCH_COND, OP_SWITCH = 124, 128, 132, 176, 199, 197, 198, 196, 245, 246, 247, 248, 249, 250, 251
OP_SELECT = 169
OP_KILL, OP_RETURN, OP_RETURN_VALUE, OP_UNREACHABLE, OP_BARRIER = 252, 253, 254, 255, 224
OP_ATOMIC_LOAD, OP_ATOMIC_STORE, OP_ATOMIC_FLAG_TEST, OP_ATOMIC_FLAG_CLEAR = 227, 228, 318, 319
OP_SHUFFLE, OP_SHUFFLE_XOR = 345, 346
TERMINATORS = {OP_BRANCH, OP_BRANCH_COND, OP_SWITCH, OP_KILL, OP_RETURN, OP_RETURN_VALUE, OP_UNREACHABLE, 4416}
ATOMICS = set(range(OP_ATOMIC_STORE, 243)) | {OP_ATOMIC_FLAG_TEST, OP_ATOMIC_FLAG_CLEAR}   # every atomic but OpAtomicLoad (227): a load has no side effect
NO_RESULT = {0, 2, 3, 4, 5, 6, 8, 10, 14, 15, 16, 17, 56, 62, 63, 64, 71, 72, 74, 75, 99, 218, 219, 220, 221, 224, 225, 228, 246, 247, 249, 250, 251, 252, 253, 254, 255, 317, 319, 330, 331, 4416, 5380}
RESULT_ONLY = set(range(19, 40)) | {OP_STRING, OP_EXT_IMPORT, OP_DECORATION_GROUP, OP_LABEL, 322, 4456, 5281, 6086}   # a result without a result type (types, labels, strings)
# group operations whose meaning changes with the lane count: Elect, All, Any, AllEqual, Broadcast, BroadcastFirst, Ballot, InverseBallot, BallotBitExtract, BallotBitCount,
# BallotFindLSB, BallotFindMSB; the reductions and scans 349-364; QuadBroadcast / QuadSwap. Shuffles (345-348) read a lane the index names and are fine with mirrored halves
BALLOT_VOTE = {333: 'Elect', 334: 'All', 335: 'Any', 336: 'AllEqual', 337: 'Broadcast', 338: 'BroadcastFirst', 339: 'Ballot', 340: 'InverseBallot', 341: 'BallotBitExtract',
               342: 'BallotBitCount', 343: 'BallotFindLSB', 344: 'BallotFindMSB', 365: 'QuadBroadcast', 366: 'QuadSwap', **{o: f'GroupNonUniform#{o}' for o in range(349, 365)}}
SUBGROUP_OPS = set(BALLOT_VOTE) | {345, 346, 347, 348}
SC_UNIFORM, SC_WORKGROUP, SC_CROSS_WG, SC_PRIVATE, SC_FUNCTION, SC_IMAGE, SC_STORAGE_BUFFER, SC_PHYSICAL = 2, 4, 5, 6, 7, 11, 12, 5349
SE_KIND = {SC_WORKGROUP: 'workgroup', SC_UNIFORM: 'storage', SC_STORAGE_BUFFER: 'storage', SC_CROSS_WG: 'storage', SC_PHYSICAL: 'bda', SC_IMAGE: 'image'}
BI_LOCAL_ID, BI_LOCAL_INDEX, BI_WORKGROUP_SIZE, BI_SUBGROUP_LANE, DEC_BUILTIN, EM_LOCAL_SIZE = 27, 29, 25, 41, 11, 17


class Ins:
    __slots__ = ('op', 'a')
    def __init__(self, op, a): self.op, self.a = op, list(a)
    @property
    def result(self):
        if self.op in NO_RESULT: return None
        return self.a[0] if self.op in RESULT_ONLY else (self.a[1] if len(self.a) > 1 else None)
    @property
    def type(self): return None if self.op in NO_RESULT or self.op in RESULT_ONLY else self.a[0]


def parse(spv):
    if len(spv) < 20 or len(spv) % 4 or struct.unpack_from('<I', spv)[0] != 0x07230203: raise ValueError('not a SPIR-V module')
    w = struct.unpack(f'<{len(spv) // 4}I', spv); hdr = list(w[:5]); ins = []; i = 5
    while i < len(w):
        n = w[i] >> 16
        if not n or i + n > len(w): raise ValueError('a SPIR-V instruction runs past the module')
        ins.append(Ins(w[i] & 0xffff, w[i + 1:i + n])); i += n
    return hdr, ins


def emit(hdr, ins, bound):
    w = hdr[:]; w[3] = bound
    for x in ins: w.append((len(x.a) + 1) << 16 | x.op); w.extend(x.a)
    return struct.pack(f'<{len(w)}I', *w)


class _Mod:
    """The module's global facts: result → instruction, pointer storage classes, built-ins, the entry's LocalSize; new ids and global instructions (types, constants)"""
    def __init__(self, spv):
        self.hdr, self.ins = parse(spv); self.bound = self.hdr[3]
        self.by_id = {}
        for x in self.ins:
            r = x.result
            if r is not None: self.by_id[r] = x
        self.first_fn = next((i for i, x in enumerate(self.ins) if x.op == OP_FUNCTION), len(self.ins))
        self.builtin = {}   # built-in number → variable (or constant) id
        for x in self.ins[:self.first_fn]:
            if x.op == OP_DECORATE and len(x.a) == 3 and x.a[1] == DEC_BUILTIN: self.builtin[x.a[2]] = x.a[0]
        self.entry = next((x for x in self.ins if x.op == OP_ENTRY), None)
        self.exec_mode = next((x for x in self.ins if x.op == OP_EXEC_MODE and len(x.a) == 5 and x.a[1] == EM_LOCAL_SIZE), None)
        self.new_globals = []
        self._const = {}

    def id(self): self.bound += 1; return self.bound - 1

    def storage_class(self, ptr_id):
        x = self.by_id.get(ptr_id); t = self.by_id.get(x.type) if x is not None and x.type is not None else None
        return t.a[1] if t is not None and t.op == OP_TYPE_PTR else None

    def type_of(self, vid):
        x = self.by_id.get(vid); return x.type if x is not None else None

    def const_value(self, vid):   # an integer constant's value, else None
        x = self.by_id.get(vid)
        return x.a[2] if x is not None and x.op == OP_CONST and len(x.a) == 3 else None

    def find_type(self, op, *operands):
        for x in self.ins[:self.first_fn] + self.new_globals:
            if x.op == op and x.a[1:] == list(operands): return x.a[0]
        t = self.id(); self.new_globals.append(Ins(op, [t] + list(operands))); return t

    def uint(self): return self.find_type(OP_TYPE_INT, 32, 0)
    def bool(self): return self.find_type(OP_TYPE_BOOL)
    def v3uint(self): return self.find_type(OP_TYPE_VECTOR, self.uint(), 3)

    def const(self, ty, v):
        k = (ty, v)
        if k not in self._const:
            c = next((x.a[1] for x in self.ins[:self.first_fn] + self.new_globals if x.op == OP_CONST and x.a[0] == ty and x.a[2:] == [v]), None)
            if c is None: c = self.id(); self.new_globals.append(Ins(OP_CONST, [ty, c, v]))
            self._const[k] = c
        return self._const[k]

    def undef(self, ty): return self.const_like(OP_UNDEF, ty)
    def const_composite(self, ty, parts):
        k = (OP_CONST_COMPOSITE, ty, tuple(parts))
        if k not in self._const: c = self.id(); self.new_globals.append(Ins(OP_CONST_COMPOSITE, [ty, c] + list(parts))); self._const[k] = c
        return self._const[k]
    def const_like(self, op, ty):
        k = (op, ty)
        if k not in self._const: u = self.id(); self.new_globals.append(Ins(op, [ty, u])); self._const[k] = u
        return self._const[k]

    def functions(self):   # [(start, end)] index ranges of the functions, OpFunction .. OpFunctionEnd inclusive
        out, s = [], None
        for i, x in enumerate(self.ins):
            if x.op == OP_FUNCTION: s = i
            elif x.op == OP_FUNCTION_END: out.append((s, i)); s = None
        return out

    def bytes(self, ins=None):
        ins = self.ins if ins is None else ins
        first_fn = next((i for i, x in enumerate(ins) if x.op == OP_FUNCTION), len(ins))
        return emit(self.hdr, ins[:first_fn] + self.new_globals + ins[first_fn:], self.bound)


def _mask_x(m, x_id, ty, pad_to):   # x & (pad_to / 2 - 1), in x's integer type
    r = m.id(); return r, Ins(OP_AND, [ty, r, x_id, m.const(ty, pad_to // 2 - 1)])


def info(spv):
    """What a module holds: {'local': [x, y, z] or None, 'shuffles', 'barriers', 'subgroup_ops', 'ballot_votes': [names]}"""
    _, ins = parse(spv); c = collections.Counter(x.op for x in ins)
    local = next((x.a[2:5] for x in ins if x.op == OP_EXEC_MODE and len(x.a) == 5 and x.a[1] == EM_LOCAL_SIZE), None)
    return {'local': local, 'shuffles': sum(c[o] for o in (345, 346, 347, 348)), 'barriers': c[OP_BARRIER], 'subgroup_ops': sum(c[o] for o in SUBGROUP_OPS),
            'ballot_votes': sorted({BALLOT_VOTE[o] for o in c if o in BALLOT_VOTE})}


def wants_pad(spv, x=32):
    """A compute module with LocalSize x == x (z 1) that uses a subgroup operation: what the pass is for"""
    i = info(spv)
    return bool(i['local']) and i['local'][0] == x and i['local'][2] == 1 and i['subgroup_ops'] > 0


def pad(spv, pad_to=64, mode='branch'):
    if mode not in ('branch', 'select'): raise ValueError(f'mode {mode!r}: branch or select')
    m = _Mod(spv)
    if m.exec_mode is None: raise ValueError('no LocalSize execution mode (not a compute kernel)')
    x, y, z = m.exec_mode.a[2:5]
    if x != pad_to // 2 or z != 1: raise ValueError(f'LocalSize {x}x{y}x{z}: the pass pads {pad_to // 2}xNx1 to {pad_to}xNx1')
    lid, lidx, wgs = m.builtin.get(BI_LOCAL_ID), m.builtin.get(BI_LOCAL_INDEX), m.builtin.get(BI_WORKGROUP_SIZE)
    if lid is None and (lidx is not None or any(x.op in ATOMICS or x.op in (OP_STORE, OP_COPY_MEMORY, OP_IMAGE_WRITE) for x in m.ins[m.first_fn:])):
        # the predicate needs LocalInvocationId: declare it (an Input variable the entry point lists)
        pt = m.find_type(OP_TYPE_PTR, 1, m.v3uint()); lid = m.id(); m.new_globals.append(Ins(OP_VARIABLE, [pt, lid, 1]))
        at = next(i for i, x_ in enumerate(m.ins) if x_.op in RESULT_ONLY and x_.op != OP_EXT_IMPORT and x_.op != OP_STRING or x_.op in (OP_VARIABLE, OP_CONST))   # before the first type
        m.ins.insert(at, Ins(OP_DECORATE, [lid, DEC_BUILTIN, BI_LOCAL_ID])); m.first_fn += 1; m.entry.a.append(lid); m.by_id[lid] = m.new_globals[-1]
    uint, bool_, v3 = m.uint(), m.bool(), m.v3uint()
    # LocalInvocationId.x reached through an access chain with a constant index: index 0 is x (masked), 1 / 2 are y / z (as they are); a run-time index bails
    x_chains, lid_ptr_users = set(), []
    for x_ in m.ins[m.first_fn:]:
        if x_.op in (OP_ACCESS, OP_IN_ACCESS) and len(x_.a) >= 4 and x_.a[2] == lid:
            if len(x_.a) != 4: raise ValueError('LocalInvocationId indexed twice')
            v = m.const_value(x_.a[3])
            if v is None: raise ValueError('LocalInvocationId read through a run-time index')
            if v == 0: x_chains.add(x_.a[1])
        elif lid is not None and ((x_.op == OP_CALL and lid in x_.a[3:]) or (x_.op == OP_STORE and x_.a[1] == lid) or (x_.op in (OP_PHI, 169, 83, OP_BITCAST, OP_PTR_ACCESS) and lid in x_.a[2:])):
            lid_ptr_users.append(x_.op)   # the variable itself handed on (a call argument, a stored pointer, OpSelect / OpCopyObject / OpBitcast / OpPtrAccessChain of it): its reads could not be masked
        if lidx is not None and x_.op not in (OP_LOAD, OP_NAME, OP_DECORATE, OP_ENTRY) and lidx in x_.a[2:]: raise ValueError('LocalInvocationIndex used other than by OpLoad')
    if lid_ptr_users: raise ValueError(f'LocalInvocationId passed by pointer ({", ".join(str(o) for o in lid_ptr_users)})')
    if wgs is not None:   # the WorkgroupSize constant overrides the execution mode: its x becomes pad_to
        c = m.by_id[wgs]
        if c.op != OP_CONST_COMPOSITE or len(c.a) != 5: raise ValueError('WorkgroupSize is not a 3-component constant composite')
        c.a[2] = m.const(m.type_of(c.a[2]), pad_to)
    m.exec_mode.a[2] = pad_to
    stats = {'stores': collections.Counter(), 'atomics': 0, 'image_writes': 0, 'functions': 0, 'lid_loads': 0, 'barriers': sum(1 for x_ in m.ins if x_.op == OP_BARRIER),
             'shuffles': sum(1 for x_ in m.ins if x_.op in (345, 346, 347, 348)), 'local': [pad_to, y, z], 'split_loop_headers': 0, 'selections': 0, 'selects': 0, 'mode': mode}
    # mode select: the Workgroup arrays grow by one spare slot (variable → its index); the access chains whose one index a padding lane's store redirects there
    spare, redirect = {}, {}
    if mode == 'select':
        for x_ in m.ins[:m.first_fn]:
            if x_.op == OP_VARIABLE and x_.a[2] == SC_WORKGROUP:
                arr = m.by_id.get(m.by_id[x_.a[0]].a[2])
                if arr is None or arr.op != 28 or m.const_value(arr.a[2]) is None: continue   # not a sized array: its stores keep the branch
                n_ = m.const_value(arr.a[2]); spare[x_.a[1]] = (n_, arr)
        for v, (n_, arr) in spare.items():   # (the type may be shared by another variable: it grows too, harmlessly); the new length constant goes right before the array type
            ct = m.type_of(arr.a[2]); c = m.id(); at = m.ins.index(arr); m.ins.insert(at, Ins(OP_CONST, [ct, c, n_ + 1])); m.first_fn += 1; m.by_id[c] = m.ins[at]; arr.a[2] = c
        for x_ in m.ins[m.first_fn:]:
            if x_.op == OP_STORE and m.storage_class(x_.a[0]) == SC_WORKGROUP:
                ch = m.by_id.get(x_.a[0])
                if ch is not None and ch.op in (OP_ACCESS, OP_IN_ACCESS) and len(ch.a) == 4 and ch.a[2] in spare: redirect[ch.a[1]] = spare[ch.a[2]][0]
    c32 = m.const(uint, pad_to // 2)

    def selectable(x_):   # mode select: a side effect made harmless by an OpSelect instead of a branch
        return mode == 'select' and ((x_.op == OP_STORE and x_.a[0] in redirect) or x_.op == OP_IMAGE_WRITE)

    def side_effect(x_):   # the kind of side effect an instruction is (one that needs the branch), or None
        if selectable(x_): return None
        if x_.op == OP_STORE or x_.op == OP_COPY_MEMORY or x_.op == OP_COPY_MEMORY_SIZED: return SE_KIND.get(m.storage_class(x_.a[0]))
        if x_.op == OP_IMAGE_WRITE: return 'image'
        if x_.op in ATOMICS:
            sc = m.storage_class(x_.a[0] if x_.op in (OP_ATOMIC_STORE, OP_ATOMIC_FLAG_CLEAR) else x_.a[2])
            return 'atomic' if sc not in (SC_FUNCTION, SC_PRIVATE) else None
        return None

    def rewrite_lid_load(x_):   # the load of LocalInvocationId (whole or .x): the same result id, from the masked value
        stats['lid_loads'] += 1
        raw = m.id(); ty = x_.a[0]
        if x_.a[2] == lid:   # the vec3: x & 31 back in
            xr = m.id(); masked, and_ = _mask_x(m, xr, uint, pad_to)
            return [Ins(OP_LOAD, [ty, raw, lid]), Ins(OP_EXTRACT, [uint, xr, raw, 0]), and_, Ins(OP_INSERT, [ty, x_.a[1], masked, raw, 0])]
        if x_.a[2] in x_chains:
            masked, and_ = _mask_x(m, raw, ty, pad_to); and_.a[1] = x_.a[1]
            return [Ins(OP_LOAD, x_.a[:1] + [raw] + x_.a[2:]), and_]
        if lidx is not None and x_.a[2] == lidx:   # (x & 31) + y * 32 + z * 32 * N: the index of the 32xNx1 group
            raw = m.id(); xr, yr, zr = m.id(), m.id(), m.id(); xm, and_ = _mask_x(m, xr, uint, pad_to); ym, zm, s1 = m.id(), m.id(), m.id()
            return [Ins(OP_LOAD, [v3, raw, lid]), Ins(OP_EXTRACT, [uint, xr, raw, 0]), Ins(OP_EXTRACT, [uint, yr, raw, 1]), Ins(OP_EXTRACT, [uint, zr, raw, 2]), and_,
                    Ins(OP_IMUL, [uint, ym, yr, c32]), Ins(OP_IMUL, [uint, zm, zr, m.const(uint, pad_to // 2 * y)]), Ins(OP_IADD, [uint, s1, xm, ym]), Ins(OP_IADD, [uint, x_.a[1], s1, zm])]
        return None

    out = m.ins[:m.first_fn]
    for s, e in m.functions():
        fn = m.ins[s:e + 1]
        has_se = any(side_effect(x_) or selectable(x_) for x_ in fn)
        reads_lid = any(x_.op == OP_LOAD and (x_.a[2] == lid or x_.a[2] in x_chains or x_.a[2] == lidx) for x_ in fn)
        if not has_se and not reads_lid: out += fn; continue
        first_label = next(i for i, x_ in enumerate(fn) if x_.op == OP_LABEL)
        head, body = fn[:first_label], fn[first_label:-1]   # OpFunction + parameters; the blocks; (OpFunctionEnd)
        blocks, cur = [], None
        for x_ in body:
            if x_.op == OP_LABEL: cur = [x_]; blocks.append(cur)
            else: cur.append(x_)
        real = None
        if has_se: stats['functions'] += 1; real = m.id()
        final_label, new_blocks, own_phis = {}, [], set()   # own_phis: the merge blocks' OpPhis, whose parents are already the split pieces
        for bi, blk in enumerate(blocks):
            label, term = blk[0], blk[-1]
            if term.op not in TERMINATORS: raise ValueError(f'block %{label.a[0]} does not end in a terminator')
            mid = blk[1:-1]; merge = mid.pop() if mid and mid[-1].op in (OP_LOOP_MERGE, OP_SEL_MERGE) else None
            # the block's instructions, with LocalInvocationId loads rewritten
            body2 = []
            for x_ in mid:
                r = rewrite_lid_load(x_) if x_.op == OP_LOAD and len(x_.a) >= 3 and (x_.a[2] == lid or x_.a[2] in x_chains or x_.a[2] == lidx) else None
                body2 += r if r else [x_]
            if bi == 0 and has_se:   # the predicate: after the function's leading OpVariables, before anything else
                nv = 0
                while nv < len(body2) and body2[nv].op == OP_VARIABLE: nv += 1
                raw, xr = m.id(), m.id()
                body2[nv:nv] = [Ins(OP_LOAD, [v3, raw, lid]), Ins(OP_EXTRACT, [uint, xr, raw, 0]), Ins(OP_ULESS, [bool_, real, xr, c32])]
            cur = [label]
            if merge is not None and merge.op == OP_LOOP_MERGE and any(side_effect(x_) for x_ in body2):
                # a loop header with a side effect: the header keeps its OpPhis, the OpLoopMerge and a branch to a new body block that holds the rest
                if term.op != OP_BRANCH: raise ValueError(f'loop header %{label.a[0]} holds a side effect and ends in a conditional branch')
                np = 0
                while np < len(body2) and body2[np].op == OP_PHI: np += 1
                b = m.id(); cur += body2[:np] + [merge, Ins(OP_BRANCH, [b])]; new_blocks.append(cur); cur = [Ins(OP_LABEL, [b])]; body2 = body2[np:]; merge = None
                stats['split_loop_headers'] += 1
            i = 0
            while i < len(body2):
                x_ = body2[i]
                if mode == 'select' and x_.op in (OP_ACCESS, OP_IN_ACCESS) and x_.a[1] in redirect:   # the lane's slot, or the spare one for a padding lane
                    it = m.type_of(x_.a[3]); sel = m.id(); cur.append(Ins(OP_SELECT, [it, sel, real, x_.a[3], m.const(it, redirect[x_.a[1]])])); x_.a[3] = sel; stats['selects'] += 1
                    stats['stores']['workgroup'] += sum(1 for y_ in fn if y_.op == OP_STORE and y_.a[0] == x_.a[1]); cur.append(x_); i += 1; continue
                if selectable(x_) and x_.op == OP_IMAGE_WRITE:   # the coordinate, or -1 (out of bounds, dropped) for a padding lane
                    ct = m.type_of(x_.a[1]); tv = m.by_id[ct]; n_ = tv.a[2] if tv.op == OP_TYPE_VECTOR else 1; et = tv.a[1] if tv.op == OP_TYPE_VECTOR else ct
                    neg = m.const(et, 0xffffffff)
                    if n_ > 1:
                        neg = m.const_composite(ct, [neg] * n_); vb = m.find_type(OP_TYPE_VECTOR, bool_, n_); cv = m.id(); cur.append(Ins(80, [vb, cv] + [real] * n_))   # OpCompositeConstruct
                    else: cv = real
                    sel = m.id(); cur.append(Ins(OP_SELECT, [ct, sel, cv, x_.a[1], neg])); x_.a[1] = sel; stats['selects'] += 1; stats['image_writes'] += 1; cur.append(x_); i += 1; continue
                if not side_effect(x_): cur.append(x_); i += 1; continue
                run = []
                while i < len(body2) and side_effect(body2[i]): run.append(body2[i]); i += 1
                then, mrg = m.id(), m.id(); phis = []
                cur += [Ins(OP_SEL_MERGE, [mrg, 0]), Ins(OP_BRANCH_COND, [real, then, mrg])]; new_blocks.append(cur)
                tb = [Ins(OP_LABEL, [then])]
                for se in run:
                    k = side_effect(se)
                    if se.op == OP_IMAGE_WRITE: stats['image_writes'] += 1
                    elif se.op in ATOMICS: stats['atomics'] += 1
                    else: stats['stores'][k] += 1
                    r = se.result
                    if r is not None:   # an atomic's value: a new id inside, the old one from an OpPhi at the merge
                        nid = m.id(); se.a[1] = nid; phis.append(Ins(OP_PHI, [se.type, r, nid, then, m.undef(se.type), cur[0].a[0]])); own_phis.add(r)
                    tb.append(se)
                tb.append(Ins(OP_BRANCH, [mrg])); new_blocks.append(tb)
                cur = [Ins(OP_LABEL, [mrg])] + phis; stats['selections'] += 1
            if merge is not None: cur.append(merge)
            cur.append(term); new_blocks.append(cur); final_label[label.a[0]] = cur[0].a[0]
        for blk in new_blocks:   # OpPhi parents that named a split block: its last piece now
            for x_ in blk:
                if x_.op == OP_PHI and x_.a[1] not in own_phis:
                    for j in range(3, len(x_.a), 2): x_.a[j] = final_label.get(x_.a[j], x_.a[j])
        out += head
        for blk in new_blocks: out += blk
        out.append(fn[-1])
    stats['stores'] = dict(stats['stores'])
    return m.bytes(out), stats


def _lane_expr(m, vid, local_defs, depth=0):
    """Is value vid the warp lane (SubgroupLocalInvocationId & 31, or LocalInvocationId.x), possibly through bitcasts and same-block Function-variable loads? → 'lane';
    an integer constant → ('c', value); 'lane' ^ constant → ('xor', value); else None"""
    if depth > 24: return None
    x = m.by_id.get(vid)
    if x is None: return None
    if x.op == OP_CONST and len(x.a) == 3: return ('c', x.a[2] & 0xffffffff)
    if x.op == OP_BITCAST: return _lane_expr(m, x.a[2], local_defs, depth + 1)
    if x.op == OP_LOAD:
        if x.a[2] == m.builtin.get(BI_SUBGROUP_LANE): return 'sgid'
        if x.a[2] in local_defs: return _lane_expr(m, local_defs[x.a[2]], local_defs, depth + 1)
        return None
    if x.op == OP_AND:
        a, b = _lane_expr(m, x.a[2], local_defs, depth + 1), _lane_expr(m, x.a[3], local_defs, depth + 1)
        for p, q in ((a, b), (b, a)):
            if p == 'sgid' and q == ('c', 31): return 'lane'
            if p == 'sgid' and q == ('c', 0xffffffe0): return 'base'
        return None
    if x.op == OP_XOR:
        a, b = _lane_expr(m, x.a[2], local_defs, depth + 1), _lane_expr(m, x.a[3], local_defs, depth + 1)
        for p, q in ((a, b), (b, a)):
            if p == 'lane' and isinstance(q, tuple) and q[0] == 'c' and q[1] < 32: return ('xor', q[1])
        return None
    if x.op == OP_IADD:
        a, b = _lane_expr(m, x.a[2], local_defs, depth + 1), _lane_expr(m, x.a[3], local_defs, depth + 1)
        for p, q in ((a, b), (b, a)):
            if q == ('c', 0): return p
        return None
    if x.op == OP_OR:   # (lane ^ c) | base: the translator's 64-lane form, the base is the lane's own upper bit
        a, b = _lane_expr(m, x.a[2], local_defs, depth + 1), _lane_expr(m, x.a[3], local_defs, depth + 1)
        for p, q in ((a, b), (b, a)):
            if q == 'base' and isinstance(p, tuple) and p[0] == 'xor': return p
        return None
    return None


def shuffle_xor(spv):
    """Every OpGroupNonUniformShuffle whose index is provably lane ^ constant (constant < 32) as OpGroupNonUniformShuffleXor with that constant: (module bytes, count rewritten,
    count of shuffles). A Function-variable load counts as the value of the last OpStore to it in the same block, provided no OpFunctionCall lies between (a call may take the variable)"""
    m = _Mod(spv); n, total = 0, 0; uint = m.uint()
    local_defs = {}
    for x in m.ins[m.first_fn:]:
        if x.op == OP_LABEL or x.op == OP_CALL: local_defs = {}
        elif x.op == OP_STORE and m.storage_class(x.a[0]) == SC_FUNCTION: local_defs[x.a[0]] = x.a[1]
        elif x.op == OP_SHUFFLE:
            total += 1; e = _lane_expr(m, x.a[4], local_defs)   # operands: type, result, scope, value, index
            if isinstance(e, tuple) and e[0] == 'xor': x.op = OP_SHUFFLE_XOR; x.a[4] = m.const(uint, e[1]); n += 1
    return m.bytes(), n, total


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument('inp'); ap.add_argument('out'); ap.add_argument('--to', type=int, default=64); ap.add_argument('--xor', action='store_true'); ap.add_argument('--mode', default='branch', choices=('branch', 'select')); ap.add_argument('--info', action='store_true')
    a = ap.parse_args(); b = open(a.inp, 'rb').read()
    if a.info: print(info(b)); sys.exit()
    if a.xor: b, n, t = shuffle_xor(b); print(f'{n} of {t} shuffles → ShuffleXor')
    else: b, st = pad(b, a.to, a.mode); print(st)
    open(a.out, 'wb').write(b)
