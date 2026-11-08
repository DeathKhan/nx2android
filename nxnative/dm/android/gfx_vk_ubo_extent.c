#include "gfx_vk_ubo_extent.h"
#include <stdlib.h>
#include <string.h>

enum { OP_NAME = 5, OP_MEMBER_NAME = 6, OP_ENTRY_POINT = 15, OP_TYPE_INT = 21, OP_TYPE_FLOAT = 22, OP_TYPE_VECTOR = 23,
       OP_TYPE_ARRAY = 28, OP_TYPE_STRUCT = 30, OP_TYPE_POINTER = 32, OP_CONSTANT = 43, OP_VARIABLE = 59, OP_LOAD = 61,
       OP_ACCESS_CHAIN = 65, OP_IN_BOUNDS_ACCESS_CHAIN = 66, OP_DECORATE = 71, OP_MEMBER_DECORATE = 72 };
enum { F_BLOCK = 1, F_STRIDE = 2, F_OFFSET = 4, F_SET = 8, F_BINDING = 16, F_BAD = 32 };
typedef struct { uint32_t def, set, binding, stride, offset, flags; } id_info;

static const uint32_t *definition(const uint32_t *w, const id_info *ids, uint32_t bound, uint32_t id, uint32_t op, uint32_t count) {
    if (!id || id >= bound || !ids[id].def) return NULL;
    const uint32_t *p = w + ids[id].def;
    return (p[0] & 0xffff) == op && p[0] >> 16 == count ? p : NULL;
}
static int constant_u32(const uint32_t *w, const id_info *ids, uint32_t bound, uint32_t id, uint32_t *value) {
    const uint32_t *c = definition(w, ids, bound, id, OP_CONSTANT, 4);
    const uint32_t *t = c ? definition(w, ids, bound, c[1], OP_TYPE_INT, 4) : NULL;
    if (!t || t[2] != 32 || (t[3] && (c[3] & 0x80000000u))) return 0;
    *value = c[3]; return 1;
}
static int pointer_to(const uint32_t *w, const id_info *ids, uint32_t bound, uint32_t id, uint32_t pointee) {
    const uint32_t *p = definition(w, ids, bound, id, OP_TYPE_POINTER, 4);
    return p && p[2] == 2 && p[3] == pointee;   // Uniform storage
}
static uint32_t read_extent(const uint32_t *w, size_t nw, const id_info *ids, uint32_t bound, uint32_t root, uint32_t *pointers) {
    const uint32_t *var = definition(w, ids, bound, root, OP_VARIABLE, 4);
    const uint32_t *ptr = var ? definition(w, ids, bound, var[1], OP_TYPE_POINTER, 4) : NULL;
    const uint32_t *block = ptr ? definition(w, ids, bound, ptr[3], OP_TYPE_STRUCT, 3) : NULL;
    if (!var || var[3] != 2 || !ptr || ptr[2] != 2 || !block) return 0;
    const id_info *bd = &ids[ptr[3]];
    if ((bd->flags & (F_BLOCK | F_OFFSET | F_BAD)) != (F_BLOCK | F_OFFSET) || bd->offset != 0) return 0;
    const uint32_t *array = definition(w, ids, bound, block[2], OP_TYPE_ARRAY, 4);
    if (!array || !(ids[block[2]].flags & F_STRIDE) || (ids[block[2]].flags & F_BAD) || ids[block[2]].stride != 16) return 0;
    const uint32_t *vec = definition(w, ids, bound, array[2], OP_TYPE_VECTOR, 4);
    if (!vec || vec[3] != 4) return 0;
    const uint32_t *scalar = definition(w, ids, bound, vec[2], OP_TYPE_FLOAT, 3);
    if (!scalar) scalar = definition(w, ids, bound, vec[2], OP_TYPE_INT, 4);
    if (!scalar || scalar[2] != 32) return 0;
    uint32_t length;
    if (!constant_u32(w, ids, bound, array[3], &length) || !length) return 0;

    memset(pointers, 0, (size_t)bound * sizeof *pointers);
    pointers[root] = UINT32_MAX;   // a whole-block load is deliberately unsupported
    // First collect direct, constant access chains. No pointer propagation is accepted; an alias is caught by the use audit below.
    for (size_t at = 5; at < nw; at += w[at] >> 16) {
        const uint32_t *p = w + at; uint32_t n = p[0] >> 16, op = p[0] & 0xffff;
        if ((op != OP_ACCESS_CHAIN && op != OP_IN_BOUNDS_ACCESS_CHAIN) || n < 4 || p[3] != root) continue;
        uint32_t member, row, component = 0;
        if ((n != 6 && n != 7) || !constant_u32(w, ids, bound, p[4], &member) || member ||
            !constant_u32(w, ids, bound, p[5], &row) || row >= length || row >= 4096 ||
            (n == 7 && (!constant_u32(w, ids, bound, p[6], &component) || component >= 4)) ||
            !pointer_to(w, ids, bound, p[1], n == 6 ? array[2] : vec[2]) || !p[2] || p[2] >= bound) return 0;
        pointers[p[2]] = (row + 1) * 16;   // complete vec4, matching the host's uniform-copy granularity
    }

    uint32_t high = 0;
    for (size_t at = 5; at < nw; at += w[at] >> 16) {
        const uint32_t *p = w + at; uint32_t n = p[0] >> 16, op = p[0] & 0xffff;
        // These are declarations/debug metadata, never reads or pointer escapes. Entry-point interfaces can list Uniform variables in SPIR-V 1.4+.
        if (op == OP_NAME || op == OP_MEMBER_NAME || op == OP_ENTRY_POINT || op == OP_DECORATE || op == OP_MEMBER_DECORATE ||
            op == 17 || op == OP_CONSTANT) continue;   // capability enums and scalar constant values are literals, not pointer IDs
        if (op == OP_VARIABLE && n == 4 && p[2] == root) continue;
        if ((op == OP_ACCESS_CHAIN || op == OP_IN_BOUNDS_ACCESS_CHAIN) && n >= 4 && p[3] == root) continue;   // validated above
        if (op == OP_LOAD && n >= 4 && p[3] < bound && pointers[p[3]]) {
            // Only a load of the access chain's exact element type is understood; optional memory operands also stay conservative.
            const uint32_t *ac = w + ids[p[3]].def;
            const uint32_t *pt = definition(w, ids, bound, ac[1], OP_TYPE_POINTER, 4);
            if (n != 4 || p[3] == root || !pt || p[1] != pt[3]) return 0;
            if (pointers[p[3]] > high) high = pointers[p[3]];
            continue;
        }
        // Treat every operand of any other instruction as a possible ID. Literal/ID coincidences can only refuse an optimization.
        // This catches dynamic/secondary AccessChains, CopyObject, Phi/Select, function arguments, stores, pointer conversions and unknown extensions.
        for (uint32_t j = 1; j < n; j++) {
            if (op == 12 && j == 4) continue;   // OpExtInst's instruction number is a literal; every actual argument still counts
            if (p[j] < bound && pointers[p[j]]) return 0;
        }
    }
    return high < 65536 ? high : 0;
}
void vk_spv_ubo_extents(const uint32_t *w, size_t nw, uint32_t set, uint32_t *bytes, size_t nb) {
    if (!bytes || nb > 65536) return;
    memset(bytes, 0, nb * sizeof *bytes);
    if (!w || nw < 5 || w[0] != 0x07230203 || nw > UINT32_MAX || !w[3] || w[3] > (1u << 20)) return;
    uint32_t bound = w[3];
    id_info *ids = calloc(bound, sizeof *ids);
    uint32_t *pointers = calloc(bound, sizeof *pointers);
    uint8_t *seen = calloc(nb ? nb : 1, 1);
    if (!ids || !pointers || !seen) goto done;
    // Validate all framing first: even a trailing malformed instruction makes every result unknown.
    for (size_t at = 5; at < nw;) {
        const uint32_t *p = w + at; uint32_t n = p[0] >> 16, op = p[0] & 0xffff;
        if (!n || n > nw - at) goto done;
        if (op == 73 || op == 74 || op == 75) goto done;   // decoration groups can hide another variable aliasing the same descriptor
        uint32_t id = 0;
        if (op == OP_TYPE_INT || op == OP_TYPE_FLOAT || op == OP_TYPE_VECTOR || op == OP_TYPE_ARRAY || op == OP_TYPE_STRUCT || op == OP_TYPE_POINTER) {
            if (n < 2) goto done;
            id = p[1];
        } else if (op == OP_CONSTANT || op == OP_VARIABLE || op == OP_ACCESS_CHAIN || op == OP_IN_BOUNDS_ACCESS_CHAIN) {
            if (n < 3) goto done;
            id = p[2];
        }
        if (id) { if (id >= bound || ids[id].def) goto done; ids[id].def = (uint32_t)at; }
        else if (op == OP_TYPE_INT || op == OP_TYPE_FLOAT || op == OP_TYPE_VECTOR || op == OP_TYPE_ARRAY || op == OP_TYPE_STRUCT || op == OP_TYPE_POINTER ||
                 op == OP_CONSTANT || op == OP_VARIABLE || op == OP_ACCESS_CHAIN || op == OP_IN_BOUNDS_ACCESS_CHAIN) goto done;
        if (op == OP_DECORATE && n >= 3) {
            if (!p[1] || p[1] >= bound) goto done;
            id_info *d = &ids[p[1]];
            if (p[2] == 2) d->flags |= F_BLOCK;
            else if (p[2] == 3) d->flags |= F_BAD;   // BufferBlock is a storage buffer
            else if (p[2] == 6 || p[2] == 33 || p[2] == 34) {
                if (n != 4) goto done;
                uint32_t flag = p[2] == 6 ? F_STRIDE : p[2] == 33 ? F_BINDING : F_SET;
                if (d->flags & flag) goto done;
                d->flags |= flag;
                if (p[2] == 6) d->stride = p[3]; else if (p[2] == 33) d->binding = p[3]; else d->set = p[3];
            }
        } else if (op == OP_MEMBER_DECORATE && n >= 4) {
            if (!p[1] || p[1] >= bound) goto done;
            id_info *d = &ids[p[1]];
            if (p[3] == 35) {
                if (n != 5) goto done;
                if (p[2] || (d->flags & F_OFFSET)) d->flags |= F_BAD;
                d->flags |= F_OFFSET; d->offset = p[4];
            } else if (p[3] != 0) d->flags |= F_BAD;   // only Offset and RelaxedPrecision on the one vec4-array member
        }
        at += n;
    }
    for (uint32_t id = 1; id < bound; id++) {
        const id_info *d = &ids[id];
        if ((d->flags & (F_SET | F_BINDING)) != (F_SET | F_BINDING) || d->set != set || d->binding >= nb) continue;
        if (seen[d->binding]) { bytes[d->binding] = 0; continue; }   // aliased variables for one descriptor are not analyzed
        seen[d->binding] = 1;
        if (!(d->flags & F_BAD)) bytes[d->binding] = read_extent(w, nw, ids, bound, id, pointers);
    }
done:
    free(seen); free(pointers); free(ids);
}
