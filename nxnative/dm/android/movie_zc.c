// Zero-copy movie pictures: the parts with no Android or Vulkan in them (see movie_zc.h). Picture slots map guest memory to the frame shown there; frames are reference counted and
// given back to the codec (zc_frame_destroy) only once every reference went and the GPU has completed the last submission that read them (zc_fence_done).
// Built into the Android host (dm/android/build.py takes dm/android/*.c) and into dm/android/tests/t_movie_zc.c on a Mac or Linux host
#include "movie_zc.h"
#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#ifdef DM_ANDROID
#include "dm.h"
#define ZLOG(...) LOG("[movie] " __VA_ARGS__)
#else
#include <stdio.h>
#define ZLOG(...) do { fprintf(stderr, "[movie] " __VA_ARGS__); fputc('\n', stderr); } while (0)
#endif

static pthread_mutex_t g_mu = PTHREAD_MUTEX_INITIALIZER;   // frames' references, the slots, the pending list
static zc_frame *g_pending;   // last reference gone, the GPU still reading
static int g_live;

__attribute__((weak)) uint64_t zc_fence_done(void) { return ~0ull; }   // (the Vulkan side's: the highest completed submission fence)
__attribute__((weak)) void zc_frame_destroy(zc_frame *f) { free(f); }

void zc_frame_ref(zc_frame *f) { if (!f) return; pthread_mutex_lock(&g_mu); f->refs++; pthread_mutex_unlock(&g_mu); }
static void reap_locked(void) {
    uint64_t done = zc_fence_done(); zc_frame **pp = &g_pending;
    while (*pp) { zc_frame *f = *pp; if (f->fence <= done) { *pp = f->next; g_live--; zc_frame_destroy(f); } else pp = &f->next; }
}
static void unref_locked(zc_frame *f) {
    if (--f->refs > 0) return;
    if (f->fence > zc_fence_done()) { f->next = g_pending; g_pending = f; return; }   // given back by zc_reap once the GPU has read it
    g_live--; zc_frame_destroy(f);
}
void zc_frame_unref(zc_frame *f) { if (!f) return; pthread_mutex_lock(&g_mu); unref_locked(f); reap_locked(); pthread_mutex_unlock(&g_mu); }
void zc_reap(void) { pthread_mutex_lock(&g_mu); reap_locked(); pthread_mutex_unlock(&g_mu); }
int zc_frame_w(const zc_frame *f) { return f->w; }
int zc_frame_h(const zc_frame *f) { return f->h; }
int64_t zc_frame_ts(const zc_frame *f) { return f->ts_ns; }
int zc_frame_crop(const zc_frame *f, int *x, int *y, int *w, int *h) {
    int has = f->cw > 0 && f->ch > 0;
    *x = has ? f->cx : 0; *y = has ? f->cy : 0; *w = has ? f->cw : f->w; *h = has ? f->ch : f->h;
    return has;
}
void zc_frame_used(zc_frame *f, uint64_t fence) { pthread_mutex_lock(&g_mu); if (fence > f->fence) f->fence = fence; pthread_mutex_unlock(&g_mu); }
int zc_frames_live(void) { pthread_mutex_lock(&g_mu); int n = g_live; pthread_mutex_unlock(&g_mu); return n; }
// a frame the Android side (or a test) made: one reference, counted live
zc_frame *zc_frame_new(void) { zc_frame *f = calloc(1, sizeof *f); if (!f) return NULL; f->refs = 1; pthread_mutex_lock(&g_mu); g_live++; pthread_mutex_unlock(&g_mu); return f; }

// ---- slots ----
#define ZC_SLOTS 64
static struct { uintptr_t lo, hi; zc_frame *f; uint64_t gen; } g_slot[ZC_SLOTS];
static int g_nslot;
static int slot_at(uintptr_t lo, uintptr_t hi) { for (int i = 0; i < ZC_SLOTS; i++) if (g_slot[i].hi && g_slot[i].lo == lo && g_slot[i].hi == hi) return i; return -1; }
void zc_slot_set(const void *mem, uint64_t size, zc_frame *f) {
    uintptr_t lo = (uintptr_t)mem, hi = lo + size;
    pthread_mutex_lock(&g_mu);
    int i = slot_at(lo, hi);
    if (i < 0) {   // a new slot: one inside or around it (a pool made again at another size) goes first
        for (int k = 0; k < ZC_SLOTS; k++) if (g_slot[k].hi && g_slot[k].lo < hi && lo < g_slot[k].hi) { if (g_slot[k].f) unref_locked(g_slot[k].f); memset(&g_slot[k], 0, sizeof g_slot[k]); g_nslot--; }
        for (int k = 0; k < ZC_SLOTS && i < 0; k++) if (!g_slot[k].hi) i = k;
        if (i < 0) { static int once; if (!once++) ZLOG("[RED] zero-copy: more than %d picture slots", ZC_SLOTS); pthread_mutex_unlock(&g_mu); return; }
        g_slot[i].lo = lo; g_slot[i].hi = hi; g_slot[i].f = NULL; g_slot[i].gen = 0; g_nslot++;
    }
    if (f) f->refs++;
    if (g_slot[i].f) unref_locked(g_slot[i].f);
    g_slot[i].f = f; g_slot[i].gen++;
    reap_locked();
    pthread_mutex_unlock(&g_mu);
}
void zc_slot_clear(const void *mem, uint64_t size) {
    uintptr_t lo = (uintptr_t)mem, hi = lo + size;
    pthread_mutex_lock(&g_mu);
    for (int k = 0; k < ZC_SLOTS; k++) if (g_slot[k].hi && g_slot[k].lo >= lo && g_slot[k].hi <= hi) { if (g_slot[k].f) unref_locked(g_slot[k].f); memset(&g_slot[k], 0, sizeof g_slot[k]); g_nslot--; }
    reap_locked();
    pthread_mutex_unlock(&g_mu);
}
int zc_slot_find(const void *mem, uint64_t len, zc_frame **f, uint64_t *gen) {
    uintptr_t lo = (uintptr_t)mem, hi = lo + (len ? len : 1); int ok = 0;
    if (f) *f = NULL;
    if (gen) *gen = 0;
    if (!__atomic_load_n(&g_nslot, __ATOMIC_RELAXED)) return 0;   // (no movie: the lookup of every texture costs nothing)
    pthread_mutex_lock(&g_mu);
    for (int k = 0; k < ZC_SLOTS; k++) if (g_slot[k].hi && lo >= g_slot[k].lo && hi <= g_slot[k].hi) {
        ok = 1; if (f && g_slot[k].f) { *f = g_slot[k].f; (*f)->refs++; } if (gen) *gen = g_slot[k].gen; break; }
    pthread_mutex_unlock(&g_mu);
    return ok;
}
int zc_slots_live(void) { pthread_mutex_lock(&g_mu); int n = g_nslot; pthread_mutex_unlock(&g_mu); return n; }

// ---- timestamp map ----
void zc_tsmap_push(zc_tsmap *m, int64_t codec_us, int64_t game_us) {
    if (m->n == 64) { memmove(m->e, m->e + 1, 63 * sizeof m->e[0]); m->n = 63; }
    m->e[m->n].c = codec_us; m->e[m->n].g = game_us; m->n++;
}
int zc_tsmap_take(zc_tsmap *m, int64_t codec_us, int64_t *game_us) {
    for (int i = 0; i < m->n; i++) if (m->e[i].c == codec_us) { *game_us = m->e[i].g; memmove(m->e + i, m->e + i + 1, (size_t)(m->n - i - 1) * sizeof m->e[0]); m->n--; return 1; }
    return 0;
}
