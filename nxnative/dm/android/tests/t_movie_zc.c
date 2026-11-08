// Zero-copy movie pictures without Android or Vulkan (dm/android/movie_zc.c): frame references, picture slots, the release deferred until the GPU's fence, the timestamp map.
// make -C dm/android/tests (clang or gcc on a Mac or Linux host; no game, no device)
#include "../movie_zc.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

zc_frame *zc_frame_new(void);   // movie_zc.c
static uint64_t g_done;                 // the GPU's highest completed fence, as the test sets it
static int g_destroyed, g_last_id;
uint64_t zc_fence_done(void) { return g_done; }
void zc_frame_destroy(zc_frame *f) { g_destroyed++; g_last_id = (int)f->id; free(f); }
static zc_frame *mk(int id, int w, int h) { zc_frame *f = zc_frame_new(); f->id = (uint64_t)id; f->w = w; f->h = h; f->ts_ns = id * 16666667ll; return f; }

static uint8_t pool[4][0x4000];   // four picture slots of a pool, as hle_codec_video.c lays them out (native_size, 16 KB each)

static void t_refs(void) {
    zc_frame *f = mk(1, 64, 64);
    assert(zc_frames_live() == 1);
    zc_frame_ref(f); zc_frame_unref(f); assert(g_destroyed == 0 && zc_frames_live() == 1);
    zc_frame_unref(f); assert(g_destroyed == 1 && g_last_id == 1 && zc_frames_live() == 0);   // never read by the GPU: given back at once
    int x, y, w, h; zc_frame *g = mk(2, 3840, 2176); g->cx = 0; g->cy = 0; g->cw = 3840; g->ch = 2160;
    assert(zc_frame_crop(g, &x, &y, &w, &h) == 1 && w == 3840 && h == 2160);
    g->cw = 0; assert(zc_frame_crop(g, &x, &y, &w, &h) == 0 && w == 3840 && h == 2176);   // no crop: the whole buffer
    zc_frame_unref(g); assert(g_destroyed == 2);
    puts("refs ok");
}
static void t_slots(void) {
    g_destroyed = 0;
    zc_frame *f1 = mk(1, 8, 8), *f2 = mk(2, 8, 8), *got; uint64_t gen;
    assert(zc_slot_find(pool[0], 0x4000, &got, &gen) == 0);   // no slot yet: every other texture's lookup says no
    zc_slot_set(pool[0], 0x4000, f1);                           // the slot shows f1 (its own reference)
    assert(zc_slots_live() == 1);
    assert(zc_slot_find(pool[0], 0x4000, &got, &gen) == 1 && got == f1 && gen == 1); zc_frame_unref(got);
    assert(zc_slot_find(pool[0] + 64, 64, &got, &gen) == 1 && got == f1); zc_frame_unref(got);   // a texture inside the slot
    assert(zc_slot_find(pool[0] + 0x3ff0, 64, &got, &gen) == 0);                             // one crossing its end: not a slot's
    zc_frame_unref(f1);                                          // the decoder's reference goes: the slot still holds it
    assert(g_destroyed == 0);
    zc_slot_set(pool[0], 0x4000, f2); zc_frame_unref(f2);        // the next picture: f1 goes back
    assert(g_destroyed == 1 && g_last_id == 1);
    assert(zc_slot_find(pool[0], 0x4000, &got, &gen) == 1 && got == f2 && gen == 2); zc_frame_unref(got);
    zc_slot_set(pool[0], 0x4000, NULL);                          // shows nothing: a lookup finds the slot and no frame
    assert(g_destroyed == 2 && zc_slot_find(pool[0], 0x4000, &got, &gen) == 1 && got == NULL && gen == 3);
    zc_frame *f3 = mk(3, 8, 8); zc_slot_set(pool[1], 0x4000, f3); zc_frame_unref(f3);
    zc_slot_set(pool[0], 0x8000, NULL);                          // a pool made again over the first two slots: both go (f3 back)
    assert(g_destroyed == 3 && g_last_id == 3 && zc_slots_live() == 1);
    zc_slot_clear(pool[0], sizeof pool); assert(zc_slots_live() == 0);
    assert(zc_slot_find(pool[0], 0x4000, &got, &gen) == 0 && zc_frames_live() == 0);
    puts("slots ok");
}
static void t_fence(void) {
    g_destroyed = 0; g_done = 10;
    zc_frame *f = mk(7, 8, 8);
    zc_slot_set(pool[2], 0x4000, f); zc_frame_unref(f);
    zc_frame *got; uint64_t gen; assert(zc_slot_find(pool[2], 0x4000, &got, &gen) == 1 && got == f);
    zc_frame_used(got, 12); zc_frame_unref(got);                 // the GPU reads it in submission 12
    zc_frame *n = mk(8, 8, 8); zc_slot_set(pool[2], 0x4000, n); zc_frame_unref(n);   // replaced while submission 12 is not done
    assert(g_destroyed == 0 && zc_frames_live() == 2);
    g_done = 11; zc_reap(); assert(g_destroyed == 0);
    g_done = 12; zc_reap(); assert(g_destroyed == 1 && g_last_id == 7 && zc_frames_live() == 1);   // given back once the fence completed
    zc_frame_used(n, 13); zc_frame_used(n, 12); assert(n->fence == 13);                           // the highest fence stands
    zc_slot_clear(pool[2], 0x4000); assert(g_destroyed == 1);
    g_done = 13; zc_slot_set(pool[3], 0x4000, NULL);             // any slot change reaps
    assert(g_destroyed == 2 && g_last_id == 8 && zc_frames_live() == 0);
    zc_slot_clear(pool[0], sizeof pool);
    puts("fence ok");
}
static void t_stream(void) {   // a 4K movie's flow: the codec's frames shown one after another in one slot, each read by the GPU, with the game two submissions behind
    g_destroyed = 0; g_done = 100;
    for (int i = 0; i < 200; i++) {
        zc_frame *f = mk(100 + i, 3840, 2160); zc_slot_set(pool[0], 0x4000, f); zc_frame_unref(f);
        zc_frame *got; uint64_t gen; assert(zc_slot_find(pool[0], 0x4000, &got, &gen) == 1 && got == f && gen == (uint64_t)i + 1);
        zc_frame_used(got, 101 + i); zc_frame_unref(got);
        g_done = 99 + i; zc_reap();
        assert(zc_frames_live() <= 3);   // shown, read by the submission in flight, and the one before
    }
    zc_slot_clear(pool[0], 0x4000); g_done = 400; zc_reap();
    assert(zc_frames_live() == 0 && g_destroyed == 200);
    puts("stream ok");
}
static void t_tsmap(void) {
    zc_tsmap m; memset(&m, 0, sizeof m); int64_t g;
    for (int i = 0; i < 70; i++) zc_tsmap_push(&m, 1000 + i, 5000 + i);   // 70 rendered pictures: the oldest 6 are gone
    assert(m.n == 64 && zc_tsmap_take(&m, 1000, &g) == 0 && zc_tsmap_take(&m, 1006, &g) == 1 && g == 5006 && m.n == 63);
    assert(zc_tsmap_take(&m, 1069, &g) == 1 && g == 5069 && zc_tsmap_take(&m, 1069, &g) == 0);   // out of order, taken once
    assert(zc_tsmap_take(&m, 1030, &g) == 1 && g == 5030 && m.n == 61);
    puts("tsmap ok");
}
int main(void) { t_refs(); t_slots(); t_fence(); t_stream(); t_tsmap(); puts("t_movie_zc: all ok"); return 0; }
