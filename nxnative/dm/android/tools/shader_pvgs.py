#!/usr/bin/env python3
"""The pass-through geometry shader a fragment shader with per-vertex inputs needs on a device without VK_KHR_fragment_shader_barycentric (shader_vk.py PER_VERTEX = gs).

Ampere's LDTRAM reads a fragment input at each of the primitive's three vertices (Nintendo Switch 2 Welcome Tour's wireframe pixel shaders, 27 of them: the triangle's edge
lengths). With the extension the translator declares the input as one array of the three vertices, PerVertexKHR; without it (the Odin 3's stock Adreno 830 driver, 512.800.72)
the translator reads three flat inputs per attribute instead (job per_vertex = gs; Ryujinx.Graphics.Shader AttributeUsage.GetPerVertexFlatLocation): the attribute's own
location holds the value at vertex 0, two free locations above every input the shader reads hold vertices 1 and 2 (the reflection's "per_vertex_gs": source location → the three).
This module writes the geometry shader between the vertex and fragment stages that feeds them: for each primitive it emits the primitive's vertices again with gl_Position and
every ordinary varying the fragment shader reads copied through, and the three flat outputs set to the attribute's value at the primitive's vertices 0, 1 and 2 (the same on each
emitted vertex, so the provoking vertex does not matter). The vertex shader is untouched; its outputs the fragment shader does not read are dropped here (Vulkan allows an output
with no input), and an input it does not write is padded by the backend as for any pair (gfx_vk_shader.c spv_add_outputs, from the geometry shader's in_locs).

One geometry shader per fragment shader, under pv_gs_key(fragment key) with stage 2 in the bundle, for input triangles (lists, strips and fans all reach a geometry shader as
triangles); the other input primitives under shader_vk.py gs_variant_key as a game's own geometry shaders, so the backend's vk_shader_gs_topo picks the draw's. A line or point
primitive has fewer than three vertices: the missing ones repeat the last (the fragment shader's differences are then 0, as a wireframe of a line has no third edge).
Only gl_Position passes through among the built-ins: a vertex shader's clip or cull distances, point size, layer or viewport index are not forwarded (see the report's limits).

    pv_geometry(fragment SPIR-V, per_vertex_gs map, topology) → the geometry shader's SPIR-V, or ValueError when the fragment shader's inputs are not all vec4 floats
    fragment_inputs(fragment SPIR-V) → {location: variable id} of its user inputs (Input variables decorated Location)

Topology numbers are Ryujinx's InputTopology, as shader_vk.py's gs_input jobs and gs_variant_key use them: 0 points, 1 lines, 2 lines with adjacency, 3 triangles, 4 triangles with adjacency."""
import struct

# per input topology: OpExecutionMode input (InputPoints 19, InputLines 20, InputLinesAdjacency 21, Triangles 22, InputTrianglesAdjacency 23), output mode (OutputPoints 27,
# OutputLineStrip 28, OutputTriangleStrip 29), the input array length, and which inputs are the primitive's own vertices (adjacency: the odd ones are the neighbours)
TOPO = {0: (19, 27, 1, [0]), 1: (20, 28, 2, [0, 1]), 2: (21, 28, 4, [1, 2]), 3: (22, 29, 3, [0, 1, 2]), 4: (23, 29, 6, [0, 2, 4])}
TRIANGLES = 3
# the bundle key of a fragment shader's generated geometry shader (its triangles form; the others under shader_vk.py gs_variant_key of this): the fragment key XOR this salt
# (2^64 / phi^2; gs_variant_key uses multiples of 2^64 / phi, so the two never meet). The backend derives the same (gfx_vk_shader.c vk_shader_pv_gs)
PV_GS_SALT = 0x61C8864680B583EB
def pv_gs_key(fs_key): return fs_key ^ PV_GS_SALT

OP_NAME, OP_DECORATE, OP_VARIABLE, OP_TYPE_POINTER, OP_TYPE_FLOAT, OP_TYPE_VECTOR = 5, 71, 59, 32, 22, 23
DEC_LOCATION, DEC_COMPONENT, SC_INPUT = 30, 31, 1

def _words(spv):
    if len(spv) < 20 or len(spv) % 4 or struct.unpack_from('<I', spv)[0] != 0x07230203: raise ValueError('not a SPIR-V module')
    w = struct.unpack(f'<{len(spv) // 4}I', spv); i = 5; out = []
    while i < len(w):
        n = w[i] >> 16
        if not n or i + n > len(w): raise ValueError('a SPIR-V instruction runs past the module')
        out.append((w[i] & 0xffff, w[i + 1:i + n])); i += n
    return out

def fragment_inputs(spv):
    """A fragment shader's user inputs: {location: variable id}. Every one must be a vec4 of 32-bit floats without a Component decoration (what the translator declares for a
    user attribute; the generated geometry shader copies them as such), else ValueError"""
    ins = _words(spv); types, deco, vars_ = {}, {}, []
    for op, a in ins:
        if op == OP_TYPE_FLOAT: types[a[0]] = ('f', a[1])
        elif op == OP_TYPE_VECTOR: types[a[0]] = ('v', a[1], a[2])
        elif op == OP_TYPE_POINTER: types[a[0]] = ('p', a[1], a[2])
        elif op == OP_DECORATE and len(a) >= 2: deco.setdefault(a[0], {})[a[1]] = a[2] if len(a) > 2 else 1
        elif op == OP_VARIABLE and a[2] == SC_INPUT: vars_.append((a[1], a[0]))
    out = {}
    for vid, pt in vars_:
        d = deco.get(vid, {})
        if DEC_LOCATION not in d: continue   # a built-in
        loc = d[DEC_LOCATION]; t = types.get(types.get(pt, (None, None, None))[2])
        if DEC_COMPONENT in d or not t or t[0] != 'v' or t[2] != 4 or types.get(t[1]) != ('f', 32):
            raise ValueError(f'fragment input at location {loc} is not a vec4 of floats (its per-vertex reads need a pass-through geometry shader of vec4 inputs)')
        if loc in out: raise ValueError(f'two fragment inputs at location {loc}')
        out[loc] = vid
    return out

class _Asm:
    def __init__(self): self.bound = 1; self.pre, self.body = [], []   # instructions before the function, and the function's
    def id(self): self.bound += 1; return self.bound - 1
    def emit(self, where, op, *a): where.append(((len(a) + 1) << 16) | op); where.extend(a)
    @staticmethod
    def string(s): b = s.encode() + b'\0'; b += b'\0' * (-len(b) % 4); return struct.unpack(f'<{len(b) // 4}I', b)

def pv_geometry(fs_spv, pv_map, topo=TRIANGLES):
    """The pass-through geometry shader for a fragment shader whose per-vertex reads are flat inputs: pv_map = the reflection's per_vertex_gs, {source location: [location of the value
    at vertex 0, 1, 2]} (keys as int or str). Inputs: the fragment shader's ordinary varyings and the sources, arrays over the input primitive; outputs: the ordinary varyings and the
    flat locations. topo: Ryujinx's InputTopology (TOPO). → SPIR-V 1.3 bytes"""
    if topo not in TOPO: raise ValueError(f'input topology {topo} not known')
    in_mode, out_mode, n_in, prim = TOPO[topo]
    pv = {int(k): [int(x) for x in v] for k, v in pv_map.items()}
    if not pv or any(len(v) != 3 for v in pv.values()): raise ValueError('per_vertex_gs: each source needs three locations')
    fs_in = fragment_inputs(fs_spv)
    targets = {t for v in pv.values() for t in v}
    if len(targets) != 3 * len(pv): raise ValueError('per_vertex_gs: a location stands for two vertices')
    for s, v in pv.items():
        if v[0] != s: raise ValueError(f'per_vertex_gs: vertex 0 of attribute {s} is not at its own location')
        for t in v:
            if t not in fs_in: raise ValueError(f'per_vertex_gs: location {t} is not an input of the fragment shader')
    ordinary = sorted(set(fs_in) - targets)
    gs_in = sorted(set(ordinary) | set(pv))        # read from the vertices
    gs_out = sorted(set(ordinary) | targets)       # written per emitted vertex
    if any(l > 31 for l in gs_in + gs_out): raise ValueError('a location past 31')

    a = _Asm(); P, B, E = a.pre, a.body, a.emit
    # ids first, so the entry point can list the interface before the declarations are written
    main, fn_label = a.id(), a.id()
    void, fn_t, f32, v4, u32, i32 = (a.id() for _ in range(6))
    c_u8, c_u1, c_un = a.id(), a.id(), a.id()
    arr_f8, arr_f1, pv_t, arr_pv, p_in_arr_pv, gl_in, p_out_pv, gl_out = (a.id() for _ in range(8))
    arr_v4, p_in_arr_v4, p_in_v4, p_out_v4 = (a.id() for _ in range(4))
    c_i = {k: a.id() for k in sorted({0, *range(n_in)})}
    in_var = {l: a.id() for l in gs_in}; out_var = {l: a.id() for l in gs_out}
    E(P, 17, 1); E(P, 17, 2)                                   # OpCapability Shader, Geometry
    E(P, 14, 0, 1)                                             # OpMemoryModel Logical GLSL450
    E(P, 15, 3, main, *a.string('main'), gl_in, gl_out, *in_var.values(), *out_var.values())   # OpEntryPoint Geometry
    E(P, 16, main, in_mode); E(P, 16, main, 0, 1); E(P, 16, main, out_mode); E(P, 16, main, 26, len(prim))   # input, Invocations 1, output, OutputVertices
    E(P, OP_NAME, main, *a.string('main')); E(P, OP_NAME, pv_t, *a.string('gl_PerVertex')); E(P, OP_NAME, gl_in, *a.string('gl_in'))
    for l, v in in_var.items(): E(P, OP_NAME, v, *a.string(f'in_{l}'))
    stands_for = {t: (s, k) for s, ts in pv.items() for k, t in enumerate(ts)}   # flat location → (source attribute, vertex)
    for l, v in out_var.items(): E(P, OP_NAME, v, *a.string(f'pv{stands_for[l][0]}_v{stands_for[l][1]}' if l in targets else f'out_{l}'))
    E(P, OP_DECORATE, pv_t, 2)                                 # Block
    for m, b in enumerate((0, 1, 3, 4)): E(P, 72, pv_t, m, 11, b)   # OpMemberDecorate BuiltIn Position, PointSize, ClipDistance, CullDistance (the translator's block)
    for l, v in in_var.items(): E(P, OP_DECORATE, v, DEC_LOCATION, l)
    for l, v in out_var.items(): E(P, OP_DECORATE, v, DEC_LOCATION, l)
    E(P, 19, void); E(P, 33, fn_t, void); E(P, OP_TYPE_FLOAT, f32, 32); E(P, OP_TYPE_VECTOR, v4, f32, 4); E(P, 21, u32, 32, 0); E(P, 21, i32, 32, 1)
    E(P, 43, u32, c_u8, 8); E(P, 43, u32, c_u1, 1); E(P, 43, u32, c_un, n_in)
    E(P, 28, arr_f8, f32, c_u8); E(P, 28, arr_f1, f32, c_u1); E(P, 30, pv_t, v4, f32, arr_f8, arr_f1)
    E(P, 28, arr_pv, pv_t, c_un); E(P, OP_TYPE_POINTER, p_in_arr_pv, SC_INPUT, arr_pv); E(P, OP_VARIABLE, p_in_arr_pv, gl_in, SC_INPUT)
    E(P, OP_TYPE_POINTER, p_out_pv, 3, pv_t); E(P, OP_VARIABLE, p_out_pv, gl_out, 3)
    E(P, 28, arr_v4, v4, c_un); E(P, OP_TYPE_POINTER, p_in_arr_v4, SC_INPUT, arr_v4); E(P, OP_TYPE_POINTER, p_in_v4, SC_INPUT, v4); E(P, OP_TYPE_POINTER, p_out_v4, 3, v4)
    for k, c in c_i.items(): E(P, 43, i32, c, k)
    for v in in_var.values(): E(P, OP_VARIABLE, p_in_arr_v4, v, SC_INPUT)
    for v in out_var.values(): E(P, OP_VARIABLE, p_out_v4, v, 3)
    E(B, 54, void, main, 0, fn_t); E(B, 248, fn_label)         # OpFunction, OpLabel
    def copy(src_chain, dst_chain):
        p, val, q = a.id(), a.id(), None
        E(B, 65, p_in_v4, p, *src_chain); E(B, 61, v4, val, p)   # OpAccessChain, OpLoad
        if len(dst_chain) > 1: q = a.id(); E(B, 65, p_out_v4, q, *dst_chain)
        E(B, 62, q if q else dst_chain[0], val)                 # OpStore
    for idx in prim:
        copy((gl_in, c_i[idx], c_i[0]), (gl_out, c_i[0]))
        for l in ordinary: copy((in_var[l], c_i[idx]), (out_var[l],))
        for s, t in pv.items():
            for k in range(3): copy((in_var[s], c_i[prim[min(k, len(prim) - 1)]]), (out_var[t[k]],))
        E(B, 218)                                               # OpEmitVertex
    E(B, 219); E(B, 253); E(B, 56)                              # OpEndPrimitive, OpReturn, OpFunctionEnd
    words = [0x07230203, 0x00010300, 0, a.bound, 0] + P + B
    return struct.pack(f'<{len(words)}I', *words)
