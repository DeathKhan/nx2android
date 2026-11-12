// Memory pools backed by a file (DM_VK_POOL_FILEBACK; gfx_vk_fileback.c). The caller (gfx_vk_core.c) holds the pool's writers off around a swap and a give-back
#pragma once
#include <stddef.h>
#include <stdint.h>

int vk_fb_on(void);                                                       // DM_VK_POOL_FILEBACK=texture (read once)
const char *vk_fb_refuse(int flags, uint64_t size);                         // why a pool with these NVN flags and size is not one this backs (NULL: it is), and the pool's line logged
// [lo, hi) of the pool at owner moved into the file: its bytes copied there (pages never touched or all zero left as holes) and the file mapped over it, writable. The range is made
// read-only for the copy, so a writer must be waiting on it (nvn_dirty_freeze) and no page of it protected (nvn_dirty_hold). 0: moved; -1: it stays anonymous (logged)
int vk_fb_take(uintptr_t lo, uintptr_t hi, uintptr_t owner, int flags);
int vk_fb_has(uintptr_t lo, uintptr_t hi);                                  // some page of [lo, hi) is the file's: a copy out of it takes every page (a dropped page reads as not present)
void vk_fb_moved(uintptr_t lo, uintptr_t hi);                               // [lo, hi) mapped over by something else (a zero-copy pool's buffer): its file blocks given back
// The pieces of [lo, hi) the pool at owner moved into the file go back to anonymous memory with their bytes, their blocks given back (the pool's Finalize). Writers as for
// vk_fb_take. Returns the bytes given back
uint64_t vk_fb_restore(uintptr_t lo, uintptr_t hi, uintptr_t owner);
#ifdef FB_TEST   // dm/android/tests/t_pool_fileback.c: the switch's values without the environment
void vk_fb_test_config(const char *dir, uint64_t min_bytes, uint64_t cap_bytes);
uint64_t vk_fb_test_in(void);                                               // bytes in the file now
#endif
