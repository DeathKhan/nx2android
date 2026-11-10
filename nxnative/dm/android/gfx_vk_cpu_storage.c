// Port of nvn_mtl.m's GPU-written storage spans and taken uncompressed textures (upstream #1390 / e2b76d4e via #1604, #1558) for the Vulkan backend (gfx_vk_tex.c holds the
// texture rules: tex_storage_ok, cpu_content_tex's uncompressed case, mtl_tex_cpu_rewritten).
// Storage the GPU writes, by address range: a render target, a texture clear, a copy into a texture or a shader image recorded over it. An uncompressed 2D texture is taken from
// its storage at first use unless the GPU wrote that storage and the CPU has not written it since. Ocarina of Time draws its terrain tiles' normal maps into the pool that also holds
// their height (260x260 R32F) and blend / hole (260x260 RGBA8, two layers) maps, which the CPU fills before nvnTextureInitialize with no flush: a per-pool rule left every tile
// streamed in after the first normal map as zeros (holes in the ground). A span keeps the dirty-page generation of its last GPU write (stamp) and has its whole pages write-protected,
// so a CPU write into it afterwards is seen. Only whole pages count: a span's first and last pages are shared with whatever lies beside it. Spans do not overlap: a GPU write over part
// of one cuts it, and the parts outside keep their own stamp
#include "gfx_vk_cpu_storage.h"
#include <pthread.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
uint64_t nvn_dirty_now(void);
const void *nvn_dirty_span(const void *addr, size_t len);
void nvn_dirty_protect(const void *addr, size_t len);
int nvn_dirty_since_cached(const void **span, const void *addr, size_t len, uint64_t stamp);
#ifdef VCS_TEST   // dm/android/tests/t_cpu_storage.c, without the host
#include <stdio.h>
#define die(...) do { fprintf(stderr, __VA_ARGS__); abort(); } while (0)
#else
#include "dm.h"   // die (a macro: LOG then abort)
#endif

typedef struct { uintptr_t lo, hi; uint64_t stamp, seen; const void *rg; int taken; } gws_span;   // seen: dirty generation at the last look; rg: the tracked range of its whole pages
static gws_span *g_gws; static size_t g_ngws, g_cgws; static pthread_mutex_t g_gws_mu = PTHREAD_MUTEX_INITIALIZER;
static _Atomic int g_gws_new;   // the GPU wrote storage anew, or wrote storage a texture was taken from, since vcs_unc_drop last looked
#define MAXU(a, b) ((a) > (b) ? (a) : (b))
#define MINU(a, b) ((a) < (b) ? (a) : (b))
static void whole_pages(uintptr_t lo, uintptr_t hi, uintptr_t *a, uintptr_t *e) { uintptr_t pg = (uintptr_t)getpagesize(); *a = (lo + pg - 1) & ~(pg - 1); *e = hi & ~(pg - 1); }
static size_t gws_find(uintptr_t a) {   // the first span that ends after a; g_gws_mu held
    size_t l = 0, h = g_ngws;
    while (l < h) { size_t m = (l + h) / 2; if (g_gws[m].hi <= a) l = m + 1; else h = m; }
    return l;
}
static void gws_stamp(gws_span *s) {   // the stamp first, then the protection (nvn_dirty_now's rule); g_gws_mu held
    uintptr_t a, e; whole_pages(s->lo, s->hi, &a, &e);
    s->stamp = s->seen = nvn_dirty_now();
    s->rg = e > a ? nvn_dirty_span((const void *)a, e - a) : NULL;
    if (s->rg) nvn_dirty_protect((const void *)a, e - a);
}
// A span whose overlap with [lo, hi) holds no whole page of it, and which does not cover it, is a sliver of something beside it and does not decide (upstream nxnative
// c06d1e79a): gws_cpu_since cannot answer for it, and taking its silence as the GPU's made Ocarina of Time's CPU-filled terrain height / hole maps GPU textures (tiles
// vanished or showed another tile). A span that covers the whole texture keeps the GPU's
static int gws_sliver(const gws_span *s, uintptr_t lo, uintptr_t hi) {
    if (s->lo <= lo && s->hi >= hi) return 0;
    uintptr_t a, e; whole_pages(MAXU(lo, s->lo), MINU(hi, s->hi), &a, &e);
    return e <= a;
}
static int gws_cpu_since(gws_span *s, uintptr_t lo, uintptr_t hi) {   // the CPU wrote a whole page of [lo, hi) within the span after its stamp; g_gws_mu held
    uintptr_t a, e; whole_pages(MAXU(lo, s->lo), MINU(hi, s->hi), &a, &e);
    if (e <= a || !s->rg) return 0;
    return nvn_dirty_since_cached(&s->rg, (const void *)a, e - a, s->stamp);
}
static void gws_rewritten(gws_span *s) {   // the GPU writes a span again: a CPU write since its stamp is overwritten now; g_gws_mu held
    uint64_t now = nvn_dirty_now();
    if (now == s->seen) return;
    s->seen = now;
    if (!gws_cpu_since(s, s->lo, s->hi)) return;
    gws_stamp(s);
    if (s->taken) { s->taken = 0; atomic_store(&g_gws_new, 1); }
}
static void gws_room(size_t more) {
    if (g_ngws + more <= g_cgws) return;
    size_t cap = g_cgws ? 2 * g_cgws : 256; while (cap < g_ngws + more) cap *= 2;
    gws_span *g = realloc(g_gws, cap * sizeof *g_gws); if (!g) die("GPU-written storage spans: out of memory");
    g_gws = g; g_cgws = cap;
}
static void gws_set(uintptr_t lo, uintptr_t hi) {   // [lo, hi) becomes one span, stamped now; the spans it overlaps keep their parts outside it, with their own stamps; g_gws_mu held
    size_t i = gws_find(lo), j = i;
    while (j < g_ngws && g_gws[j].lo < hi) j++;
    int nl = i < j && g_gws[i].lo < lo, nr = i < j && g_gws[j - 1].hi > hi;
    gws_span left = nl ? g_gws[i] : (gws_span){0}, right = nr ? g_gws[j - 1] : (gws_span){0};
    left.hi = lo; right.lo = hi;
    size_t n = (size_t)nl + 1 + (size_t)nr;
    gws_room(n);
    memmove(g_gws + i + n, g_gws + j, (g_ngws - j) * sizeof *g_gws);
    g_ngws = g_ngws - (j - i) + n;
    if (nl) g_gws[i++] = left;
    g_gws[i] = (gws_span){.lo = lo, .hi = hi}; gws_stamp(&g_gws[i]);
    if (nr) g_gws[i + 1] = right;
    atomic_store(&g_gws_new, 1);
}
void vcs_gpu_written(uintptr_t lo, uintptr_t hi) {
    if (hi <= lo) return;
    pthread_mutex_lock(&g_gws_mu);
    size_t i = gws_find(lo);
    if (i < g_ngws && g_gws[i].lo == lo && g_gws[i].hi == hi) gws_rewritten(&g_gws[i]); else gws_set(lo, hi);
    pthread_mutex_unlock(&g_gws_mu);
}
int vcs_owned(uintptr_t lo, uintptr_t hi) {
    int owned = 0;
    pthread_mutex_lock(&g_gws_mu);
    for (size_t i = gws_find(lo); i < g_ngws && g_gws[i].lo < hi && !owned; i++) owned = !gws_sliver(&g_gws[i], lo, hi) && !gws_cpu_since(&g_gws[i], lo, hi);
    pthread_mutex_unlock(&g_gws_mu);
    return owned;
}
static void gws_take(uintptr_t lo, uintptr_t hi) {
    pthread_mutex_lock(&g_gws_mu);
    for (size_t i = gws_find(lo); i < g_ngws && g_gws[i].lo < hi; i++) g_gws[i].taken = 1;
    pthread_mutex_unlock(&g_gws_mu);
}
size_t vcs_spans(void) { pthread_mutex_lock(&g_gws_mu); size_t n = g_ngws; pthread_mutex_unlock(&g_gws_mu); return n; }

// The uncompressed textures taken from storage, by serial (a list: it is walked only when the GPU wrote anew)
typedef struct { uint64_t serial; uintptr_t lo, hi; } unc_ent;
static unc_ent *g_unc; static size_t g_nunc, g_cunc; static pthread_mutex_t g_unc_mu = PTHREAD_MUTEX_INITIALIZER;
void vcs_unc_take(uint64_t serial, uintptr_t lo, uintptr_t hi) {
    pthread_mutex_lock(&g_unc_mu);
    size_t i = 0; while (i < g_nunc && g_unc[i].serial != serial) i++;
    if (i == g_nunc) {
        if (g_nunc == g_cunc) { size_t c = g_cunc ? 2 * g_cunc : 64; unc_ent *g = realloc(g_unc, c * sizeof *g); if (!g) die("taken textures: out of memory"); g_unc = g; g_cunc = c; }
        g_nunc++;
    }
    g_unc[i] = (unc_ent){serial, lo, hi};
    pthread_mutex_unlock(&g_unc_mu);
    gws_take(lo, hi);
}
int vcs_unc_taken(uint64_t serial) {
    pthread_mutex_lock(&g_unc_mu); int r = 0; for (size_t i = 0; i < g_nunc && !r; i++) r = g_unc[i].serial == serial; pthread_mutex_unlock(&g_unc_mu); return r;
}
void vcs_unc_drop(void (*drop)(uint64_t serial)) {
    if (!atomic_load_explicit(&g_gws_new, memory_order_relaxed) || !atomic_exchange(&g_gws_new, 0)) return;
    pthread_mutex_lock(&g_unc_mu);
    for (size_t i = 0; i < g_nunc;) {
        if (!vcs_owned(g_unc[i].lo, g_unc[i].hi)) { i++; continue; }
        uint64_t s = g_unc[i].serial; g_unc[i] = g_unc[--g_nunc];
        drop(s);
    }
    pthread_mutex_unlock(&g_unc_mu);
}
void vcs_range_gone(uintptr_t lo, uintptr_t hi) {
    pthread_mutex_lock(&g_gws_mu);
    size_t i = gws_find(lo), j = i;
    while (j < g_ngws && g_gws[j].lo < hi) j++;
    // every span meeting the range goes (a pool's spans lie inside it), as nvn_mtl.m gws_drop_pool
    memmove(g_gws + i, g_gws + j, (g_ngws - j) * sizeof *g_gws); g_ngws -= j - i;
    pthread_mutex_unlock(&g_gws_mu);
    pthread_mutex_lock(&g_unc_mu);
    for (size_t k = 0; k < g_nunc;) if (g_unc[k].lo >= lo && g_unc[k].hi <= hi) g_unc[k] = g_unc[--g_nunc]; else k++;
    pthread_mutex_unlock(&g_unc_mu);
}
