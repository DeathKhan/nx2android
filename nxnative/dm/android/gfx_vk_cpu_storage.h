// Storage the GPU writes and the uncompressed textures taken from storage the CPU wrote (gfx_vk_cpu_storage.c): the bookkeeping half of nvn_mtl.m's
// gws_span / cpu_unc rules, kept apart from Vulkan so dm/android/tests/t_cpu_storage.c can run it with a fake dirty-page tracker
#pragma once
#include <stddef.h>
#include <stdint.h>
void vcs_gpu_written(uintptr_t lo, uintptr_t hi);         // a GPU write into [lo, hi) is recorded (mtl_tex_gpu_written)
int vcs_owned(uintptr_t lo, uintptr_t hi);                // the GPU wrote some of [lo, hi) and the CPU has not rewritten it since: the storage is not the content
void vcs_unc_take(uint64_t serial, uintptr_t lo, uintptr_t hi);   // an uncompressed texture is taken from its storage: the GPU's next write there makes it a GPU texture again
int vcs_unc_taken(uint64_t serial);                       // (tests)
// The uncompressed textures taken from storage the GPU has written since: each one's serial to drop (it leaves the CPU-written set), at most when the GPU wrote anew
void vcs_unc_drop(void (*drop)(uint64_t serial));
void vcs_range_gone(uintptr_t lo, uintptr_t hi);          // a pool (chunk) goes: its spans and taken textures with it
size_t vcs_spans(void);                                   // (tests)
