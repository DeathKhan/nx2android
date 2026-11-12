#!/usr/bin/env python3
"""NVN2 task and mesh shaders drawn without VK_EXT_mesh_shader (the Odin 3's stock Adreno 830 driver, 512.800.72): the classic emulation, compute dispatches plus an indexed indirect draw.

The translator turns a task or mesh shader into a compute kernel whose stage memory is one Workgroup array (dmsh_meshstage.py has the layout; InstEmitAmpereMem.cs AIsberd / AStageOut;
job stage_memory): word 0 the workgroup's number, INPUT the task's payload, then the outputs (a task's payload, or a mesh shader's attribute blocks of 32 vertices and its primitive bytes).
On Metal dmsh_meshstage.to_stage wraps the kernel's MSL into object and mesh functions; here wrap() wraps its SPIR-V into a compute kernel the Vulkan backend dispatches
(gfx_vk_exec.c mesh_draw): the kernel's main becomes a plain function, and a new main zeroes the stage memory, fills its inputs from buffers, calls it, and copies its outputs out:

  task kernel     word 0 = ctl[0] (the draw's first workgroup) + the workgroup's id; after the body its payload (PAY words, from OUT) goes to taskout[wg * PAY ..]
  mesh kernel     with a task: launch[16 + 2 wg] = the task that launched it and launch[17 + 2 wg] = its number among that task's launches (k), its payload is taskout[task * PAY ..]
                  copied to INPUT, word 0 = k and INPUT word 0 = k (as the hardware leaves it; the Metal wrapper does the same); without one: word 0 = ctl[0] + wg.
                  After the body: every vertex v < max_v of the workgroup → vtx[(wg * max_v + v) * NCOMP + c] = component c (the SPH's OMAP order, dmsh_meshstage.sph_outputs);
                  every index n < max_p * verts → idx[wg * max_p * verts + n] = wg * max_v + (n < count * verts ? the primitive byte : 0), count = the byte at PRIM + 3, so a
                  primitive the shader did not emit is a degenerate one of vertex 0 (the draw is a fixed max_p primitives a workgroup, in workgroup order, as the hardware rasterizes them)
                  A mesh shader whose max_v is whole blocks of 32 (blocks()) keeps the stage memory's attribute blocks in vtx instead: vertex v's component c is at
                  vtx[wg * max_v * NCOMP + (v / 32) * NCOMP * 32 + c * 32 + v % 32], the same size, and the pass-through vertex shader reads that
Direct stores and loads (the usual case; the idea of droidplay-core's #411, on this wrapper): the translator writes every attribute store and payload read with its area's offset as a
constant in the address ((register + C) >> 2, InstEmitAmpereMem.cs AStageOut and AIsberd), so _areas tells the areas apart and _rewrite sends a mesh kernel's attribute stores straight
to vtx (the block layout's word), a task kernel's payload stores straight to taskout (zeroed first), and a launched mesh kernel's payload reads straight to the task's payload in
taskout (word 0 reads k). The Workgroup array then holds the header and the primitive bytes alone, and nothing but the indices is copied out after the body: a kernel's GPU time on an
Adreno 830 follows its shared memory. A kernel whose addresses do not all sort, or that reads its outputs back or writes its input, keeps the copy form above (_direct_ok)
The buffers are storage buffers the wrapper adds at bindings above the kernel's own, named for the backend's reflection (shader_vk.py classify → gfx_vk.h RK_MESH_*):
  dm_mesh_ctl      {first workgroup, workgroup count, ...} (gfx_vk_exec.c mesh_draw writes it, or dm/android/shaders/meshprep.comp for an indirect draw)
  dm_mesh_taskout  the task payloads, PAY words a task workgroup
  dm_mesh_launch   16 header words (the mesh dispatch's and the draw's indirect arguments, dm/android/shaders/meshlaunch.comp writes them), then {task, k} a mesh workgroup
  dm_mesh_vtx      the vertices, NCOMP words each;  dm_mesh_idx  the 32-bit indices
A task or mesh entry keeps its stage (6 / 7) in the bundle; its header words 2-4 carry facts() instead of a compute shader's local size (the backend decodes them: gfx_vk_shader.c).
vertex_shader() makes the pass-through vertex shader of a mesh shader (bundle key mesh_vs_key(mesh key), stage 0): it reads vertex gl_VertexIndex's NCOMP words from dm_mesh_vtx
and writes gl_Position and a vec4 at each generic location the mesh shader's OMAP names (a component the mesh shader does not write is 0, w 1, as a vertex output starts).
Not supported (ValueError, the shader is noted and left out): a point size output, per-primitive outputs, a primitive of other than 1, 2 or 3 vertices, a kernel without exactly one
Workgroup array of the stage memory's size."""
import bisect
import collections
import struct
import dmsh_meshstage as ms

MESH_VS_SALT = 0x2545F4914F6CDD1D   # the pass-through vertex shader's key: the mesh key XOR this (not a multiple of shader_vk.py's gs_variant_key salt, nor shader_pvgs.PV_GS_SALT)
def mesh_vs_key(ms_key): return ms_key ^ MESH_VS_SALT

CTL, TASKOUT, LAUNCH, VTX, IDX = 'dm_mesh_ctl', 'dm_mesh_taskout', 'dm_mesh_launch', 'dm_mesh_vtx', 'dm_mesh_idx'
LAUNCH_HDR = 16   # words before the launch entries (gfx_vk_exec.c mesh_draw, dm/android/shaders/meshlaunch.comp)
BLOCK = ms.BLOCK

# opcodes
OP_NAME, OP_ENTRY, OP_EXEC_MODE, OP_TYPE_VOID, OP_TYPE_BOOL, OP_TYPE_INT, OP_TYPE_FLOAT, OP_TYPE_VECTOR, OP_TYPE_ARRAY, OP_TYPE_RARRAY, OP_TYPE_STRUCT, OP_TYPE_PTR, OP_TYPE_FN = 5, 15, 16, 19, 20, 21, 22, 23, 28, 29, 30, 32, 33
OP_CONST, OP_FUNCTION, OP_FUNCTION_END, OP_CALL, OP_VARIABLE, OP_LOAD, OP_STORE, OP_ACCESS, OP_DECORATE, OP_MEMBER_DECORATE, OP_CONSTRUCT, OP_EXTRACT = 43, 54, 56, 57, 59, 61, 62, 65, 71, 72, 80, 81
OP_BITCAST, OP_IADD, OP_IMUL, OP_SELECT, OP_IEQUAL, OP_ULESS, OP_SHR, OP_SHL, OP_AND, OP_BARRIER, OP_LOOP_MERGE, OP_SEL_MERGE, OP_LABEL, OP_BRANCH, OP_BRANCH_COND, OP_RETURN, OP_PHI = 124, 128, 132, 169, 170, 176, 194, 196, 199, 224, 246, 247, 248, 249, 250, 253, 245
OP_EXT_INST, OP_EXT_IMPORT, OP_CAPABILITY, OP_MEMORY_MODEL = 12, 11, 17, 14
DEC_BLOCK, DEC_BUFFER_BLOCK, DEC_ARRAY_STRIDE, DEC_BUILTIN, DEC_LOCATION, DEC_BINDING, DEC_SET, DEC_OFFSET = 2, 3, 6, 11, 30, 33, 34, 35
BI_POSITION, BI_WORKGROUP_ID, BI_LOCAL_INDEX, BI_VERTEX_INDEX = 0, 26, 29, 42
SC_INPUT, SC_UNIFORM, SC_OUTPUT, SC_WORKGROUP, SC_FUNCTION = 1, 2, 3, 4, 7
OP_CONST_NULL = 46
OP_PARAM, OP_INBOUNDS, OP_COPY, OP_ISUB = 55, 66, 83, 130
SC_PRIVATE = 6
SEM_WORKGROUP, SEM_WORKGROUP_UNIFORM = 0x108, 0x148   # AcquireRelease | WorkgroupMemory; with UniformMemory too: a barrier that orders the buffer stores
DEC_NO_CONTRACTION = 42
FLOAT_ARITH = (127, 129, 131, 133, 136, 140, 141, 142, 148)   # FNegate FAdd FSub FMul FDiv FRem FMod VectorTimesScalar Dot
GLSL_UMIN = 38
TYPE_OPS = set(range(19, 40)) | {41, 42, 43, 44, 45, 46}


def _string(s):
    b = s.encode() + b'\0'; b += b'\0' * (-len(b) % 4)
    return list(struct.unpack(f'<{len(b) // 4}I', b))


class _Mod:
    """A SPIR-V module as a list of (opcode, operands), with new ids, deduplicated scalar / pointer types and constants, and three insertion points"""
    def __init__(self, spv=None, version=0x10300):
        if spv is None: self.hdr, self.ins = [0x07230203, version, 0, 1, 0], []
        else:
            if len(spv) < 20 or len(spv) % 4 or struct.unpack_from('<I', spv)[0] != 0x07230203: raise ValueError('not a SPIR-V module')
            w = struct.unpack(f'<{len(spv) // 4}I', spv); self.hdr = list(w[:5]); self.ins = []; i = 5
            while i < len(w):
                n = w[i] >> 16
                if not n or i + n > len(w): raise ValueError('a SPIR-V instruction runs past the module')
                self.ins.append((w[i] & 0xffff, list(w[i + 1:i + n]))); i += n
        self.bound = self.hdr[3]
        self.new_dec, self.new_types, self.new_fn = [], [], []   # inserted before the first type, before the first function, at the end
        self._const_cache = {}

    def id(self): self.bound += 1; return self.bound - 1

    def _find(self, op, operands):
        for o, a in self.ins:
            if o == op and a[1:] == list(operands): return a[0]
        for o, a in self.new_types:
            if o == op and a[1:] == list(operands): return a[0]
        return None

    def type(self, op, *operands):
        """A non-aggregate type (scalar, vector, pointer, function): the module's, or a new one (SPIR-V forbids two of the same form)"""
        t = self._find(op, operands)
        if t is None: t = self.id(); self.new_types.append((op, [t] + list(operands)))
        return t

    def new_type(self, op, *operands):   # an aggregate (array, struct): always its own
        t = self.id(); self.new_types.append((op, [t] + list(operands))); return t

    def const(self, ty, v):
        k = (ty, v)
        if k in self._const_cache: return self._const_cache[k]
        c = self._find(OP_CONST, [ty, v])
        if c is None: c = self.id(); self.new_types.append((OP_CONST, [ty, c, v]))
        self._const_cache[k] = c; return c

    def var(self, ptr_ty, sc, name=None, decorations=()):
        v = self.id(); self.new_types.append((OP_VARIABLE, [ptr_ty, v, sc]))
        if name: self.new_dec.append((OP_NAME, [v] + _string(name)))
        for d in decorations: self.new_dec.append((OP_DECORATE, [v] + list(d)))
        return v

    def decorate(self, target, *d): self.new_dec.append((OP_DECORATE, [target] + list(d)))
    def member_decorate(self, target, m, *d): self.new_dec.append((OP_MEMBER_DECORATE, [target, m] + list(d)))

    def ssbo(self, uint, rarr, binding, name):
        """A storage buffer of uint words at set 0 / binding (Uniform + BufferBlock, the translator's form): (its variable, the pointer type of a word)"""
        st = self.new_type(OP_TYPE_STRUCT, rarr); self.decorate(st, DEC_BUFFER_BLOCK); self.member_decorate(st, 0, DEC_OFFSET, 0)
        pt = self.type(OP_TYPE_PTR, SC_UNIFORM, st)
        v = self.var(pt, SC_UNIFORM, name, [(DEC_SET, 0), (DEC_BINDING, binding)])
        return v, self.type(OP_TYPE_PTR, SC_UNIFORM, uint)

    def bytes(self):
        names = [x for x in self.new_dec if x[0] == OP_NAME]; decs = [x for x in self.new_dec if x[0] != OP_NAME]
        first_type = next((i for i, (o, _) in enumerate(self.ins) if o in TYPE_OPS), len(self.ins))
        first_dec = next((i for i, (o, _) in enumerate(self.ins) if o in (OP_DECORATE, OP_MEMBER_DECORATE)), first_type)
        first_fn = next((i for i, (o, _) in enumerate(self.ins) if o == OP_FUNCTION), len(self.ins))
        out = self.ins[:first_dec] + names + self.ins[first_dec:first_type] + decs + self.ins[first_type:first_fn] + self.new_types + self.ins[first_fn:] + self.new_fn
        w = self.hdr[:]; w[3] = self.bound
        for o, a in out: w.append((len(a) + 1) << 16 | o); w.extend(a)
        return struct.pack(f'<{len(w)}I', *w)


class _Fn:
    """A function being emitted: straight-line code, if (lid == 0) blocks and for (i = start; i < n; i += step) loops in structured form"""
    def __init__(self, m, uint, bool_):
        self.m, self.uint, self.bool = m, uint, bool_; self.code = []; self.label = None
    def emit(self, op, *a): self.code.append((op, list(a)))
    def val(self, op, ty, *a): r = self.m.id(); self.emit(op, ty, r, *a); return r
    def block(self, l=None): l = l if l is not None else self.m.id(); self.emit(OP_LABEL, l); self.label = l; return l
    def c(self, v): return self.m.const(self.uint, v)
    def load(self, ty, ptr): return self.val(OP_LOAD, ty, ptr)
    def store(self, ptr, v): self.emit(OP_STORE, ptr, v)
    def chain(self, pty, base, *idx): return self.val(OP_ACCESS, pty, base, *idx)
    def add(self, a, b): return self.val(OP_IADD, self.uint, a, b)
    def mul(self, a, b): return self.val(OP_IMUL, self.uint, a, b)
    def shr(self, a, b): return self.val(OP_SHR, self.uint, a, b)
    def shl(self, a, b): return self.val(OP_SHL, self.uint, a, b)
    def band(self, a, b): return self.val(OP_AND, self.uint, a, b)
    def barrier(self, sem=SEM_WORKGROUP): self.emit(OP_BARRIER, self.c(2), self.c(2), self.c(sem))   # Workgroup, Workgroup

    def if_zero(self, v, body):
        """if (v == 0) body()"""
        cond = self.val(OP_IEQUAL, self.bool, v, self.c(0)); then, merge = self.m.id(), self.m.id()
        self.emit(OP_SEL_MERGE, merge, 0); self.emit(OP_BRANCH_COND, cond, then, merge)
        self.block(then); body(); self.emit(OP_BRANCH, merge); self.block(merge)

    def loop(self, start, n, step, body):
        """for (i = start; i < n; i += step) body(i): n and step are ids"""
        pre = self.label; hdr, chk, bd, cont, merge = (self.m.id() for _ in range(5)); i, inext = self.m.id(), self.m.id()
        self.emit(OP_BRANCH, hdr); self.block(hdr)
        self.emit(OP_PHI, self.uint, i, start, pre, inext, cont)
        self.emit(OP_LOOP_MERGE, merge, cont, 0); self.emit(OP_BRANCH, chk); self.block(chk)
        cond = self.val(OP_ULESS, self.bool, i, n); self.emit(OP_BRANCH_COND, cond, bd, merge)
        self.block(bd); body(i); self.emit(OP_BRANCH, cont)
        self.block(cont); self.emit(OP_IADD, self.uint, inext, i, step); self.emit(OP_BRANCH, hdr)
        self.block(merge)


def _check(i):
    if i['kind'] not in ('ts', 'ms'): raise ValueError('not a task or mesh shader')
    if i['kind'] == 'ms':
        if i.get('verts', 3) not in ms.TOPOLOGY: raise ValueError(f"a primitive of {i.get('verts')} vertices (1, 2 or 3 are drawn)")
        if any(n == 'psiz' for n, _ in i['vertex']): raise ValueError('a point size output (not drawn yet)')
        if i['primitive']: raise ValueError('per-primitive outputs (not drawn yet)')
        if None not in {n for n, _ in i['vertex']}: raise ValueError('its SPH enables no position component')
        if i['max_v'] > 256 or i['max_p'] > 512: raise ValueError(f"{i['max_v']} vertices / {i['max_p']} primitives a workgroup (the backend draws up to 256 / 512)")


def facts(i):
    """The three header words of a task or mesh entry (the backend reads them in place of a compute shader's local size: gfx_vk_shader.c mesh_facts):
    word 0: threads (bits 0-9) | payload words (10-19) | has a task (20) | vertices a primitive (21-23) | kind 1 task 2 mesh (24-25);  word 1: max vertices | max primitives << 16;  word 2: vertex components"""
    kind = 1 if i['kind'] == 'ts' else 2; pay = (i['payload'] + 3) >> 2
    w0 = (i['threads'] & 0x3ff) | (pay & 0x3ff) << 10 | (1 if i.get('task') else 0) << 20 | (i.get('verts', 3) & 7) << 21 | kind << 24
    return [w0, (i.get('max_v', 0) & 0xffff) | (i.get('max_p', 0) & 0xffff) << 16, len(i.get('vertex', ()))]


def init_locals(m):
    """Every Function variable of the kernel without an initializer gets its type's null constant → how many. The translator's locals are the shader's registers, which it
    leaves undefined until their first store; a path that reads one before any store (Ocarina of Time's grass: task 99280b7b8aae78df, mesh 65a079a285d693a0 and
    eb021183bf9b0092) got whatever the Adreno compiler left there, and the grass came out solid white. Zero made the three kernels deterministic and the grass green"""
    ptr = {a[0]: a[2] for o, a in m.ins if o == OP_TYPE_PTR and a[1] == SC_FUNCTION}
    null = {a[0]: a[1] for o, a in m.ins if o == OP_CONST_NULL}; n = 0
    for o, a in m.ins:
        if o != OP_VARIABLE or len(a) != 3 or a[2] != SC_FUNCTION or a[0] not in ptr: continue
        t = ptr[a[0]]
        if t not in null: null[t] = m.id(); m.new_types.append((OP_CONST_NULL, [t, null[t]]))
        a.append(null[t]); n += 1
    return n


def no_contraction(m):
    """Every float arithmetic result of the kernel is decorated NoContraction → how many. A game draws one geometry with two mesh shaders, a depth pre-pass and a
    colour pass that tests depth EQUAL, so both must compute a vertex's position to the bit; a rasterising pipeline asks for that with an invariant position, a
    compute kernel has no such output. The translator writes the shader's own fused multiply-adds as Fma, and what it left as a multiply and an add the Adreno
    compiler fused in one kernel and not in the other: Ocarina of Time's grass (65a079a285d693a0 then eb021183bf9b0092) lost a few blades of the colour pass to
    the depth test, which the lighting showed as saturated white specks (1-6 speck cells in each of 5 runs; 0 in 6 screenshots of 2 runs with this; GPU time alike)"""
    ids = [a[1] for o, a in m.ins if o in FLOAT_ARITH]
    for x in ids: m.decorate(x, DEC_NO_CONTRACTION)
    return len(ids)


def blocks(i):
    """Whether a mesh shader's vertices lie in vtx as the stage memory's attribute blocks (the docstring): its max_v is whole blocks, so workgroups' blocks lie end to end"""
    return i['kind'] == 'ms' and i['max_v'] > 0 and i['max_v'] % BLOCK == 0 and not i['primitive']   # per-primitive blocks follow the vertices' in the output area: the old layout


def _areas(ins, mem, lay, words):
    """Every access to the stage memory, by the area its address names: {access chain id: (kind, area, dyn, c)} with kind 'load' / 'store' / 'atomic', area 'hdr' / 'in' /
    'out' / 'prim' or None when the address is not of the translator's form ((register + C) >> 2 or a constant), dyn the register's id (None for a constant address) and c
    the constant's bytes. A helper that takes a byte address (the translator's dmsh_shared_store8, the primitive area's) is classified by its call sites, all of which must
    agree. ins: the module as [opcode, operands...] lists (droidplay-core's spv_meshemu._areas, which this follows)"""
    inp_b, out_b, prim_b = lay[1], lay[2], lay[3]
    cval = {x[2]: x[3] for x in ins if x[0] == OP_CONST and len(x) == 4}
    defs = {x[2]: (i, x) for i, x in enumerate(ins) if len(x) > 2 and x[0] in (OP_LOAD, OP_BITCAST, OP_IADD, OP_SHR, OP_COPY, OP_PARAM)}
    fns, cur, fn_of, params = [], None, {}, set()
    for i, x in enumerate(ins):
        if x[0] == OP_FUNCTION: cur = [i, x[2], []]
        elif x[0] == OP_PARAM: cur[2].append(x[2]); params.add(x[2])
        elif x[0] == OP_FUNCTION_END: fns.append((cur[0], i, cur[1], cur[2])); cur = None
    for s, e, fid, _ in fns:
        for i in range(s, e + 1): fn_of[i] = fid
    stores = collections.defaultdict(list)
    for i, x in enumerate(ins):
        if x[0] == OP_STORE: stores[x[1]].append((i, x[2]))
    labels = [i for i, x in enumerate(ins) if x[0] == OP_LABEL]
    def block_of(i): return labels[bisect.bisect_right(labels, i) - 1]
    def last_store(var, i):   # the local's value at i: its last store in the block, or its one store in the function (the translator's registers are stored before use)
        prev = [s for s in stores.get(var, []) if s[0] < i and block_of(s[0]) == block_of(i)]
        if not prev and len(stores.get(var, [])) == 1 and fn_of.get(stores[var][0][0]) == fn_of.get(i) and stores[var][0][0] < i: prev = stores[var]
        return prev[-1][1] if prev else None
    def value(v):   # v as (dyn, C bytes, parameter): through bitcasts, copies, locals and additions of constants
        for _ in range(16):
            if v in cval: return None, cval[v], None
            if v in params: return None, 0, v
            d = defs.get(v)
            if d is None: return v, 0, None
            i, x = d
            if x[0] in (OP_BITCAST, OP_COPY): v = x[3]; continue
            if x[0] == OP_PARAM: return None, 0, x[2]
            if x[0] == OP_LOAD:
                if x[3] in params: return v, 0, x[3]   # a helper's parameter: its value here, and the parameter
                w = last_store(x[3], i)
                if w is None: return v, 0, None
                v = w; continue
            if x[0] == OP_IADD:
                a, b = x[3], x[4]
                if b in cval: dv, c, p = value(a); return dv, c + cval[b], p
                if a in cval: dv, c, p = value(b); return dv, c + cval[a], p
                return v, 0, None
            return v, 0, None
        return v, 0, None
    def index(idx):   # the word index → (dyn, C bytes, parameter), or None when its form is not (X + C) >> 2, a constant or a parameter's
        if idx in cval: return None, cval[idx] * 4, None
        d = defs.get(idx)
        for _ in range(8):
            if d is None: return None
            i, x = d
            if x[0] in (OP_BITCAST, OP_COPY): d = defs.get(x[3]); continue
            if x[0] == OP_LOAD:
                if x[3] in params: return None, 0, x[3]
                w = last_store(x[3], i)
                if w is None: return None
                d = defs.get(w); continue
            if x[0] == OP_SHR and x[4] in cval and cval[x[4]] == 2: return value(x[3])
            if x[0] == OP_PARAM: return None, 0, x[2]
            return None
        return None
    def area(c): return 'hdr' if c < inp_b else 'in' if c < out_b else 'out' if c < prim_b else 'prim' if c < words * 4 else None
    param_area = {}
    for s, e, fid, ps in fns:
        calls = [(i, x) for i, x in enumerate(ins) if x[0] == OP_CALL and x[3] == fid]
        for n, p in enumerate(ps):
            found = set()
            for i, x in calls:
                w = last_store(x[4 + n], i)
                dv, c, q = value(w) if w is not None else (None, 0, p)
                found.add(area(c) if q is None and (dv is None or c) else None)   # a register alone names no area
            param_area[p] = next(iter(found)) if len(found) == 1 else None
    uses = {}   # an access chain's one use; our translator computes a store's value between the chain and the store, so the use may lie far below
    for y in ins:
        q = y[3] if y[0] == OP_LOAD or 227 <= y[0] <= 241 else y[1] if y[0] == OP_STORE else None
        if q is not None: uses[q] = None if q in uses else y   # used twice: not the translator's form
    out = {}
    for i, x in enumerate(ins):
        if x[0] in (OP_ACCESS, OP_INBOUNDS) and x[3] == mem and len(x) == 5:
            use = uses.get(x[2])
            kind = 'load' if use and use[0] == OP_LOAD else 'store' if use and use[0] == OP_STORE else 'atomic' if use else None
            r = index(x[4])
            if r is None or kind is None: out[x[2]] = (kind, None, None, 0); continue
            dv, c, p = r
            if p is not None and not c:   # a helper's whole address (dmsh_shared_store8): its area is its callers'; only the header's and the primitive bytes' stay in shared memory, where the address needs no rewriting
                a = param_area.get(p); out[x[2]] = (kind, a if a in ('hdr', 'prim') else None, None, 0); continue
            out[x[2]] = (kind, area(c) if dv is None or c else None, dv, c)   # a register alone names no area
    return out


def _direct_ok(areas, i):
    """Whether the kernel can store its outputs and read its payload in place: every address sorts, nothing is read back from the output area or written into the input's,
    and a mesh shader's vertices are whole blocks"""
    if any(a is None for _, a, _, _ in areas.values()): return False
    if any(k in ('load', 'atomic') and a == 'out' or k in ('store', 'atomic') and a == 'in' for k, a, _, _ in areas.values()): return False
    return i['kind'] == 'ts' or blocks(i)


_TYPED = (OP_LOAD, OP_BITCAST, OP_IADD, OP_ISUB, OP_IMUL, OP_SHR, OP_SHL, OP_AND, OP_SELECT, OP_EXTRACT, OP_ACCESS, OP_INBOUNDS, OP_PARAM, OP_COPY, OP_EXT_INST, OP_CONST, OP_PHI, OP_CALL,
          126, 134, 135, 137, 138, 139, 195, 197, 198, 200)   # ... SNegate UDiv SDiv UMod SRem SMod ShiftRightArithmetic BitwiseOr BitwiseXor Not


def _rewrite(m, k):
    """The kernel's functions rewritten for the direct stores and loads (the docstring), by the context k → counts by kind. Its stage memory's access chains into the input
    and output areas go, and the load or store that used each becomes one of taskout or vtx; the primitive area's move down by k['shift'] words"""
    uint, bool_, p_u, mem, areas = k['uint'], k['bool'], k['p_u'], k['mem'], k['areas']
    rtype = {a[1]: a[0] for o, a in m.ins if o in _TYPED and len(a) > 1}
    out, pending, n, infn = [], {}, collections.Counter(), False
    c = lambda v: m.const(uint, v & 0xffffffff)
    def val(op, ty, *a): r = m.id(); out.append((op, [ty, r] + list(a))); return r
    def as_uint(v): return v if rtype.get(v) == uint else val(OP_BITCAST, uint, v)
    for o, a in m.ins:
        if o == OP_FUNCTION: infn = True
        if not infn: out.append((o, a)); continue
        if o == OP_FUNCTION_END:
            if pending: raise ValueError(f'{len(pending)} stage memory accesses with no load or store after them')
            infn = False; out.append((o, a)); continue
        if o in (OP_ACCESS, OP_INBOUNDS) and a[2] == mem and len(a) == 4:
            kind, ar, dyn, cb = areas[a[1]]
            if ar == 'in': pending[a[1]] = ('in', a[3], cb); n['in'] += 1; continue
            if ar == 'out': pending[a[1]] = ('out', dyn, cb); n['out'] += 1; continue
            if ar == 'prim' and k['shift']: out.append((o, a[:3] + [val(OP_ISUB, uint, as_uint(a[3]), c(k['shift']))])); n['prim'] += 1; continue
            out.append((o, a)); n['hdr'] += 1; continue
        if o == OP_LOAD and a[2] in pending:   # a payload word: the task's in taskout, word 0 being k; a task kernel's or a mesh kernel's without a task reads 0, as its stage memory held
            _, idx, cb = pending.pop(a[2])
            if k['task_v'] is None: out.append((OP_COPY, [a[0], a[1], c(0)])); continue
            w = val(OP_ISUB, uint, as_uint(idx), c(k['inp'])); at = val(OP_IADD, uint, val(OP_LOAD, uint, k['task_v']), w)
            p = val(OP_ACCESS, p_u, k['taskout'], c(0), at)
            if cb - k['inp'] * 4 >= 4: out.append((OP_LOAD, [a[0], a[1], p]))   # never word 0
            else:
                g = val(OP_LOAD, uint, p); kk = val(OP_LOAD, uint, k['k_v']); z = val(OP_IEQUAL, bool_, w, c(0)); out.append((OP_SELECT, [a[0], a[1], z, kk, g]))
            continue
        if o == OP_STORE and a[0] in pending:   # an output word: at the word the stage memory had past its output area, in the workgroup's part of vtx (a task's of taskout)
            _, dyn, cb = pending.pop(a[0]); off = c(cb - k['out_b'])
            at = off if dyn is None else val(OP_IADD, uint, as_uint(dyn), off)
            at = val(OP_IADD, uint, val(OP_LOAD, uint, k['pbase_v']), val(OP_SHR, uint, at, c(2)))
            out.append((OP_STORE, [val(OP_ACCESS, p_u, k['dst'], c(0), at), a[1]])); continue
        out.append((o, a))
    m.ins = out
    return n


def wrap(spv, i, lay, size, direct=None, stats=None):
    """The translated kernel (job stage_memory = lay, shared = size bytes; dmsh_meshstage.memory) wrapped as described above → SPIR-V. direct: None for _direct_ok's
    answer, False for the copy form (a test); stats: a dict that gets 'direct' and the shared array's words"""
    _check(i)
    m = _Mod(spv); words = size // 4
    init_locals(m)
    if i['kind'] == 'ms': no_contraction(m)
    # the kernel's entry point and its stage memory
    ep = next((a for o, a in m.ins if o == OP_ENTRY), None)
    if not ep or ep[0] != 5: raise ValueError('no GLCompute entry point')
    old_main = ep[1]
    uint = m.type(OP_TYPE_INT, 32, 0); bool_ = m.type(OP_TYPE_BOOL); void = m.type(OP_TYPE_VOID)
    arr_ty = {a[0]: (a[1], a[2]) for o, a in m.ins if o == OP_TYPE_ARRAY}; consts = {a[1]: a[2] for o, a in m.ins if o == OP_CONST and a[0] == uint}
    ptr_ty = {a[0]: (a[1], a[2]) for o, a in m.ins if o == OP_TYPE_PTR}
    mems = [a[1] for o, a in m.ins if o == OP_VARIABLE and a[2] == SC_WORKGROUP]
    mems = [v for v in mems for (sc, t) in [ptr_ty.get(next(a[0] for o, a in m.ins if o == OP_VARIABLE and a[1] == v), (0, 0))] if t in arr_ty and arr_ty[t][0] == uint and consts.get(arr_ty[t][1]) == words]
    if len(mems) != 1: raise ValueError(f'the kernel has {len(mems)} Workgroup arrays of {words} words (one stage memory expected)')
    mem = mems[0]
    # the stage memory's words
    wg_w, inp_w, out_w, prim = lay[0] >> 2, lay[1] >> 2, lay[2] >> 2, lay[3]
    thr, pay = i['threads'], (i['payload'] + 3) >> 2
    # types the wrapper uses (the module's where it has them)
    v3 = m.type(OP_TYPE_VECTOR, uint, 3)
    p_in_uint, p_in_v3 = m.type(OP_TYPE_PTR, SC_INPUT, uint), m.type(OP_TYPE_PTR, SC_INPUT, v3)
    p_wg_uint = m.type(OP_TYPE_PTR, SC_WORKGROUP, uint)
    rarr = next((a[0] for o, a in m.ins if o == OP_TYPE_RARRAY and a[1] == uint and any(oo == OP_DECORATE and aa[0] == a[0] and aa[1] == DEC_ARRAY_STRIDE and aa[2] == 4 for oo, aa in m.ins)), None)
    if rarr is None: rarr = m.new_type(OP_TYPE_RARRAY, uint); m.decorate(rarr, DEC_ARRAY_STRIDE, 4)
    fn_void = m.type(OP_TYPE_FN, void)
    # built-in inputs: the module's variables where it has them, else new ones in the interface
    builtin = {a[2]: a[0] for o, a in m.ins if o == OP_DECORATE and len(a) == 3 and a[1] == DEC_BUILTIN}
    def bi(b, pty, name):
        if b in builtin: return builtin[b]
        v = m.var(pty, SC_INPUT, name, [(DEC_BUILTIN, b)]); ep.append(v); return v
    v_lid, v_wg = bi(BI_LOCAL_INDEX, p_in_uint, 'dm_lid'), bi(BI_WORKGROUP_ID, p_in_v3, 'dm_wg')
    # the buffers, above the kernel's bindings
    nb = 1 + max([a[2] for o, a in m.ins if o == OP_DECORATE and len(a) == 3 and a[1] == DEC_BINDING] + [-1])
    ctl, p_u = m.ssbo(uint, rarr, nb, CTL); taskout, _ = m.ssbo(uint, rarr, nb + 1, TASKOUT)
    if i['kind'] == 'ms': launch, _ = m.ssbo(uint, rarr, nb + 2, LAUNCH); vtx, _ = m.ssbo(uint, rarr, nb + 3, VTX); idx, _ = m.ssbo(uint, rarr, nb + 4, IDX)
    # the wrapper
    # the direct form: the body's stores and loads in the buffers, the shared array the header and the primitive bytes (moved down by shift words) alone
    launched = i['kind'] == 'ms' and bool(i['task'])
    areas = _areas([[o] + a for o, a in m.ins], mem, lay, words)
    if direct is None: direct = _direct_ok(areas, i)
    tail_w = prim >> 2 if i['kind'] == 'ms' else lay[3] >> 2
    shift, nwords = (tail_w - inp_w, inp_w + words - tail_w) if direct else (0, words)
    area_w = (lay[3] - lay[2]) >> 2   # a mesh workgroup's attribute blocks
    pbase_v = task_v = k_v = None
    if direct:   # what the body's stores and loads need: a Private word each, set before the body runs
        p_pv = m.type(OP_TYPE_PTR, SC_PRIVATE, uint); priv = [m.var(p_pv, SC_PRIVATE, 'dm_pbase')]; pbase_v = priv[0]
        if launched: task_v, k_v = m.var(p_pv, SC_PRIVATE, 'dm_task'), m.var(p_pv, SC_PRIVATE, 'dm_k'); priv += [task_v, k_v]
        if m.hdr[1] >= 0x10400: ep.extend(priv)   # 1.4 lists every global the entry point uses
        _rewrite(m, dict(uint=uint, bool=bool_, p_u=p_u, mem=mem, areas=areas, shift=shift, inp=inp_w, out_b=lay[2], taskout=taskout, dst=vtx if i['kind'] == 'ms' else taskout,
                         pbase_v=pbase_v, task_v=task_v, k_v=k_v))
        mem_arr = next(ptr_ty[a[0]][1] for o, a in m.ins if o == OP_VARIABLE and a[1] == mem)   # the shared array's size, by a constant of its own
        at = next(n for n, (o, a) in enumerate(m.ins) if o == OP_TYPE_ARRAY and a[0] == mem_arr)
        nc = m.id(); m.ins.insert(at, (OP_CONST, [uint, nc, nwords])); m.ins[at + 1][1][2] = nc
    if stats is not None: stats.update(direct=int(bool(direct)), words=nwords)
    f = _Fn(m, uint, bool_); new_main = m.id()
    f.emit(OP_FUNCTION, void, new_main, 0, fn_void); f.block()
    c = f.c
    lid = f.load(uint, v_lid); wg = f.val(OP_EXTRACT, uint, f.load(v3, v_wg), 0)
    f.loop(lid, c(nwords), c(thr), lambda k: f.store(f.chain(p_wg_uint, mem, k), c(0)))   # the stage memory starts at zero
    f.barrier()
    if i['kind'] == 'ts' or not i['task']:
        def first(): f.store(f.chain(p_wg_uint, mem, c(wg_w)), f.add(f.load(uint, f.chain(p_u, ctl, c(0), c(0))), wg))   # word 0 = the draw's first workgroup + this one
        f.if_zero(lid, first)
    elif direct:   # the task's payload is read in place: every lane keeps where it is and k, lane 0 sets word 0
        e = f.shl(wg, c(1)); task = f.load(uint, f.chain(p_u, launch, c(0), f.add(e, c(LAUNCH_HDR)))); k = f.load(uint, f.chain(p_u, launch, c(0), f.add(e, c(LAUNCH_HDR + 1))))
        f.store(task_v, f.mul(task, c(pay))); f.store(k_v, k)
        f.if_zero(lid, lambda: f.store(f.chain(p_wg_uint, mem, c(wg_w)), k))
    else:
        e = f.shl(wg, c(1)); task = f.load(uint, f.chain(p_u, launch, c(0), f.add(e, c(LAUNCH_HDR)))); k = f.load(uint, f.chain(p_u, launch, c(0), f.add(e, c(LAUNCH_HDR + 1))))
        base = f.mul(task, c(pay))
        f.loop(lid, c(pay), c(thr), lambda n: f.store(f.chain(p_wg_uint, mem, f.add(c(inp_w), n)), f.load(uint, f.chain(p_u, taskout, c(0), f.add(base, n)))))   # INPUT = the task's payload
        f.barrier()
        def own(): f.store(f.chain(p_wg_uint, mem, c(wg_w)), k); f.store(f.chain(p_wg_uint, mem, c(inp_w)), k)   # word 0 and the payload's word 0: its number among the task's launches
        f.if_zero(lid, own)
    if direct: f.store(pbase_v, f.mul(wg, c(pay if i['kind'] == 'ts' else area_w)))   # the workgroup's attribute blocks in vtx (a task's payload in taskout)
    if direct and i['kind'] == 'ts':   # the payload zero first, as the stage memory was: a task may leave words unwritten (its launch count among them), which meshlaunch.comp and the mesh kernel read
        base = f.mul(wg, c(pay))
        f.loop(lid, c(pay), c(thr), lambda n: f.store(f.chain(p_u, taskout, c(0), f.add(base, n)), c(0)))
        f.barrier(SEM_WORKGROUP_UNIFORM)   # the zeros land before the body's stores
    else: f.barrier()
    f.emit(OP_CALL, void, m.id(), old_main); f.barrier()
    if i['kind'] == 'ts':
        if not direct:
            base = f.mul(wg, c(pay))
            f.loop(lid, c(pay), c(thr), lambda n: f.store(f.chain(p_u, taskout, c(0), f.add(base, n)), f.load(uint, f.chain(p_wg_uint, mem, f.add(c(out_w), n)))))
    else:
        ncomp, max_v, max_p, verts = len(i['vertex']), i['max_v'], i['max_p'], i.get('verts', 3); vstride = ncomp * BLOCK
        glsl = next((a[0] for o, a in m.ins if o == OP_EXT_IMPORT), None)
        if glsl is None: raise ValueError('the kernel imports no GLSL.std.450 (UMin)')
        prim -= shift * 4   # the primitive bytes in the shared array
        np_ = f.val(OP_EXT_INST, uint, glsl, GLSL_UMIN, f.shr(f.load(uint, f.chain(p_wg_uint, mem, c(prim >> 2))), c(24)), c(max_p))   # the count byte at PRIM + 3
        vbase = f.mul(wg, c(max_v))
        def vertex(v):
            dst = f.mul(f.add(vbase, v), c(ncomp)); src = f.add(f.add(c(out_w), f.mul(f.shr(v, c(5)), c(vstride))), f.band(v, c(BLOCK - 1)))
            for k in range(ncomp): f.store(f.chain(p_u, vtx, c(0), f.add(dst, c(k))), f.load(uint, f.chain(p_wg_uint, mem, f.add(src, c(k * BLOCK)))))
        if direct: pass   # the body stored them
        elif blocks(i):   # the attribute blocks as they are
            abase = f.mul(wg, c(area_w))
            f.loop(lid, c(area_w), c(thr), lambda n: f.store(f.chain(p_u, vtx, c(0), f.add(abase, n)), f.load(uint, f.chain(p_wg_uint, mem, f.add(c(out_w), n)))))
        else: f.loop(lid, c(max_v), c(thr), vertex)
        nidx = f.mul(np_, c(verts)); ibase = f.mul(wg, c(max_p * verts))
        def index(n):
            at = f.add(c(prim + 4), n); word = f.load(uint, f.chain(p_wg_uint, mem, f.shr(at, c(2))))
            byte = f.band(f.shr(word, f.shl(f.band(at, c(3)), c(3))), c(0xff))
            live = f.val(OP_ULESS, bool_, n, nidx); sel = f.val(OP_SELECT, uint, live, byte, c(0))
            f.store(f.chain(p_u, idx, c(0), f.add(ibase, n)), f.add(vbase, sel))
        f.loop(lid, c(max_p * verts), c(thr), index)
    f.emit(OP_RETURN); f.emit(OP_FUNCTION_END)
    m.new_fn = f.code
    # the entry point and execution modes name the wrapper; the old main stays a function
    ep[1] = new_main
    for o, a in m.ins:
        if o == OP_EXEC_MODE and a[0] == old_main: a[0] = new_main
    m.new_dec.append((OP_NAME, [new_main] + _string('main'))); m.new_dec.append((OP_NAME, [old_main] + _string('dm_stage_body')))
    return m.bytes()


def vertex_shader(i):
    """The pass-through vertex shader of a mesh shader (see the module's docstring) → SPIR-V 1.3"""
    _check(i)
    comps = i['vertex']; ncomp = len(comps)
    m = _Mod()
    m.ins += [(OP_CAPABILITY, [1]), (OP_MEMORY_MODEL, [0, 1])]
    void = m.type(OP_TYPE_VOID); uint = m.type(OP_TYPE_INT, 32, 0); int_ = m.type(OP_TYPE_INT, 32, 1); float_ = m.type(OP_TYPE_FLOAT, 32); bool_ = m.type(OP_TYPE_BOOL)
    v4 = m.type(OP_TYPE_VECTOR, float_, 4); fn_void = m.type(OP_TYPE_FN, void)
    rarr = m.new_type(OP_TYPE_RARRAY, uint); m.decorate(rarr, DEC_ARRAY_STRIDE, 4)
    vtx, p_u = m.ssbo(uint, rarr, 0, VTX)
    p_in_int = m.type(OP_TYPE_PTR, SC_INPUT, int_); v_vi = m.var(p_in_int, SC_INPUT, 'gl_VertexIndex', [(DEC_BUILTIN, BI_VERTEX_INDEX)])
    pv = m.new_type(OP_TYPE_STRUCT, v4); m.decorate(pv, DEC_BLOCK); m.member_decorate(pv, 0, DEC_BUILTIN, BI_POSITION)
    p_out_pv = m.type(OP_TYPE_PTR, SC_OUTPUT, pv); v_pv = m.var(p_out_pv, SC_OUTPUT, 'gl_PerVertex')
    p_out_v4 = m.type(OP_TYPE_PTR, SC_OUTPUT, v4)
    locs = sorted({n for n, _ in comps if isinstance(n, int)})
    outs = {n: m.var(p_out_v4, SC_OUTPUT, f'out{n}', [(DEC_LOCATION, n)]) for n in locs}
    main = m.id(); m.ins.append((OP_ENTRY, [0, main] + _string('main') + [v_vi, v_pv] + [outs[n] for n in locs]))
    m.new_dec.append((OP_NAME, [main] + _string('main')))
    f = _Fn(m, uint, bool_); f.emit(OP_FUNCTION, void, main, 0, fn_void); f.block(); c = f.c
    vid = f.val(OP_BITCAST, uint, f.load(int_, v_vi))
    if blocks(i):   # component k of vertex v: (v / 32) * ncomp * 32 + k * 32 + v % 32, the blocks of every workgroup end to end
        base = f.add(f.mul(f.shr(vid, c(5)), c(ncomp * BLOCK)), f.band(vid, c(BLOCK - 1))); step = BLOCK
    else: base = f.mul(vid, c(ncomp)); step = 1
    f0, f1 = m.const(float_, 0), m.const(float_, 0x3f800000)
    def value(n):
        parts = []
        for k in range(4):
            if (n, k) in comps: parts.append(f.val(OP_BITCAST, float_, f.load(uint, f.chain(p_u, vtx, c(0), f.add(base, c(comps.index((n, k)) * step))))))
            else: parts.append(f1 if k == 3 else f0)
        return f.val(OP_CONSTRUCT, v4, *parts)
    f.store(f.chain(p_out_v4, v_pv, m.const(int_, 0)), value(None))
    for n in locs: f.store(outs[n], value(n))
    f.emit(OP_RETURN); f.emit(OP_FUNCTION_END); m.new_fn = f.code
    return m.bytes()
