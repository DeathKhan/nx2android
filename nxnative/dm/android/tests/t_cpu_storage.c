// dm/android/gfx_vk_cpu_storage.c with a fake dirty-page tracker: the storage-span rules of nvn_mtl.m (dm/gfx/tests/t_cpu_storage.m's cases that do not need a GPU)
#include "../gfx_vk_cpu_storage.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
static unsigned checks, failed;
#define CHECK(label, c) do { checks++; if (!(c)) { failed++; fprintf(stderr, "FAIL: %s\n", label); } } while (0)
// fake tracker: an arena of NPG pages at a fake address; a CPU write stamps its pages with a new generation
enum { NPG = 64 };
static uintptr_t BASE; static size_t PG; static uint64_t g_gen = 1, g_page[NPG];
uint64_t nvn_dirty_now(void) { return g_gen; }
const void *nvn_dirty_span(const void *a, size_t n) { (void)n; return (uintptr_t)a >= BASE && (uintptr_t)a < BASE + NPG * PG ? (const void *)1 : NULL; }
void nvn_dirty_protect(const void *a, size_t n) { (void)a; (void)n; }
int nvn_dirty_since_cached(const void **span, const void *a, size_t n, uint64_t stamp) {
    (void)span; for (uintptr_t p = (uintptr_t)a; p < (uintptr_t)a + n; p += PG) if (g_page[(p - BASE) / PG] > stamp) return 1; return 0;
}
static void cpu_write(size_t pg0, size_t npg) { g_gen++; for (size_t i = pg0; i < pg0 + npg; i++) g_page[i] = g_gen; }
static uintptr_t at(size_t pg) { return BASE + pg * PG; }
static uint64_t dropped[8]; static int ndropped;
static void drop(uint64_t s) { dropped[ndropped++] = s; }
int main(void) {
    PG = (size_t)getpagesize(); BASE = (uintptr_t)1 << 40;
    // a render target the GPU writes, pages 0-3
    vcs_gpu_written(at(0), at(4));
    CHECK("over storage the GPU writes it is not taken from storage", vcs_owned(at(0), at(4)));
    CHECK("beside storage the GPU writes, in the same pool, it is taken from storage", !vcs_owned(at(8), at(12)));
    // the CPU rewrites the GPU-written storage: taken, and the GPU's next write there makes it a GPU texture again
    cpu_write(0, 4);
    CHECK("over storage the GPU wrote, then the CPU rewrote, it is taken from storage", !vcs_owned(at(0), at(4)));
    vcs_unc_take(7, at(0), at(4));
    ndropped = 0; vcs_unc_drop(drop);   // (the take itself is no GPU write)
    vcs_gpu_written(at(0), at(4));
    ndropped = 0; vcs_unc_drop(drop);
    CHECK("and the GPU's next write there makes it a GPU texture", ndropped == 1 && dropped[0] == 7 && !vcs_unc_taken(7));
    CHECK("which then is not taken from storage again", vcs_owned(at(0), at(4)));
    // a GPU write over part of a span leaves the CPU's rewrite of the other part seen
    vcs_gpu_written(at(16), at(20)); cpu_write(18, 2); vcs_gpu_written(at(16), at(18));
    CHECK("a GPU write over another part of the storage leaves the CPU's rewrite seen", !vcs_owned(at(18), at(20)));
    CHECK("and the part it wrote is the GPU's", vcs_owned(at(16), at(18)));
    // mtl_tex_cpu_rewritten's question: GPU_NOW while the GPU wrote after the CPU, uploaded at the CPU's next rewrite
    vcs_gpu_written(at(24), at(28)); cpu_write(24, 4);
    CHECK("a rewrite of storage the GPU wrote before is the CPU's", !vcs_owned(at(24), at(28)));
    vcs_gpu_written(at(24), at(28));
    CHECK("a rewrite the GPU wrote over afterwards is the GPU's until the CPU rewrites it", vcs_owned(at(24), at(28)));
    cpu_write(24, 4);
    CHECK("and the CPU's next rewrite is the CPU's", !vcs_owned(at(24), at(28)));
    // only whole pages count: a write into a span's shared edge page is not the span's
    vcs_gpu_written(at(32) + PG / 2, at(36) + PG / 2); cpu_write(32, 1); cpu_write(36, 1);
    CHECK("writes in the pages it shares with its neighbours leave it the GPU's", vcs_owned(at(32) + PG / 2, at(36) + PG / 2));
    cpu_write(33, 1);
    CHECK("a write of a page wholly inside it is the CPU's", !vcs_owned(at(33), at(34)));
    // a pool going takes its spans and taken textures with it
    size_t before = vcs_spans(); vcs_unc_take(9, at(40), at(41)); vcs_gpu_written(at(40), at(44));
    vcs_range_gone(at(40), at(48));
    CHECK("a pool's spans go with it", vcs_spans() == before && !vcs_owned(at(40), at(44)) && !vcs_unc_taken(9));
    printf("cpu storage: %u checks, %u failed\n", checks, failed);
    return failed != 0;
}
