// The video decoder's pictures as other parts of the host see them (hle_codec_video.c)
#pragma once
#include <stdint.h>
// The latest NV12 picture a decoder wrote into a pool's memory (pool_native_write; the CPU path): its guest address (Y rows of stride w at addr, the UV plane at addr + w x h),
// size, a count bumped at every write, and its colour matrix (DM_CODEC_CS_*, or 0 for BT.601 limited); 0 when none yet, or its pool is gone. Any thread.
// dm/android/movie_fill_vk.c converts it on the GPU in place of a movie post-processing stage the device cannot run
#define DM_CODEC_CS_BT709 1   // BT.709 matrix (else BT.601)
#define DM_CODEC_CS_FULL 2    // full range (else limited: Y 16..235, C 16..240)
int dm_codec_last_nv12(uint64_t *addr, uint32_t *w, uint32_t *h, uint64_t *gen, int *cs);
// Copy the exact selected generation into caller-owned memory; 0 if changed, forgotten, invalid, or capacity is too small.
// Pixels and metadata are protected against decoder writes/finalization by one lock. No borrowed pointer escapes this API.
int dm_codec_copy_last_nv12(uint64_t expected_gen, void *dst, uint64_t capacity);
