#ifndef DM_GFX_VK_GS_CLAIM_RUNTIME_H
#define DM_GFX_VK_GS_CLAIM_RUNTIME_H
#include <stdint.h>
#define GS_CLAIM_MAX_SLOTS (1u << 20)
#define GS_CLAIM_BUDGET (16ull << 20)
enum { GS_CLAIM_OK, GS_CLAIM_EMPTY, GS_CLAIM_INDIRECT, GS_CLAIM_INSTANCES,
       GS_CLAIM_TOPOLOGY, GS_CLAIM_SIZE, GS_CLAIM_LIMIT, GS_CLAIM_MEMORY };
/* Conservative primitive bound; PrimitiveId resets for each instance. The caller
 * uses disjoint coherent scratch until submission completion and charges worst-case
 * alignment padding against a bounded per-submission allocation budget. */
static inline uint64_t gs_claim_size(uint32_t count, uint32_t instances, uint32_t invocations,
    int prim, int indirect, uint64_t alignment, uint64_t remaining, uint64_t storage_limit, int *reason) {
    *reason = GS_CLAIM_OK;
    if (!count || !invocations) { *reason = GS_CLAIM_EMPTY; return 0; }
    if (indirect) { *reason = GS_CLAIM_INDIRECT; return 0; }
    if (instances > 1) { *reason = GS_CLAIM_INSTANCES; return 0; }
    if (prim != 4) { *reason = GS_CLAIM_TOPOLOGY; return 0; }
    uint64_t slots = (uint64_t)count * invocations;
    if (slots > GS_CLAIM_MAX_SLOTS) { *reason = GS_CLAIM_SIZE; return 0; }
    uint64_t size = 4 + slots * 4;
    if (size > storage_limit) { *reason = GS_CLAIM_LIMIT; return 0; }
    if (!alignment || size > remaining || alignment - 1 > remaining - size) { *reason = GS_CLAIM_MEMORY; return 0; }
    return size;
}
/* Dynamic descriptors omit the offset from their cache key, but retain range.
 * Bucket modest ranges so many draws can share a descriptor set. Allocate/zero
 * the whole range; never expose a following draw's storage as padding. Large
 * draws and constrained remaining budgets retain the already-validated exact size. */
static inline uint64_t gs_claim_range(uint64_t exact, uint64_t alignment, uint64_t remaining,
    uint64_t storage_limit, int bucketed) {
    if (!exact || !bucketed || exact > 65536 || !alignment) return exact;
    uint64_t range = 256;
    while (range < exact) range <<= 1;
    if (range > storage_limit || range > remaining || alignment - 1 > remaining - range) return exact;
    return range;
}
#endif
