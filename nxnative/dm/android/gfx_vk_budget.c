// Android port, Vulkan backend: the texture memory budget. The images (gfx_vk_tex.c) are kept per storage key until their pool goes or a newer texture takes
// their memory (evict_overlaps); a game whose textures are all live at once for a while keeps them all as device memory. Nintendo Switch 2 Welcome Tour's first
// scene made 3.5 GB of images on top of its memory pools, and Android's low-memory killer ended it. Here the images' device memory (vkimg_t.vkbytes) is held under
// a budget: before a new image is made (vk_tex, under the image lock, inside a submission), while the live images are over it, the least recently drawn go first
// (img_drop through vk_img_evict: the memory is freed once the GPU is past the submission), down to BUDGET_LOW of the budget.
//   Never dropped: an image looked up in the last KEEP presents or drawn to as an attachment in them (it may be in this submission's descriptor sets or passes), and
// a GPU-written image (a render target, a storage image, a copy destination: its contents exist nowhere else) idle for fewer than KEEP_GPU presents. Packaged
// textures and never-written ones come back from the guest's memory when drawn again (pkg_upload); a GPU-written one dropped and drawn again comes back empty.
//   DM_VK_IMAGE_BUDGET_MB=<n>: the budget (0: none). Default from the device's memory (/proc/meminfo MemTotal): 2048 MB from 12 GB, 1536 from 8 GB, else 1024.
#include "gfx_vk.h"
#include <stdio.h>
#include <string.h>

extern uint64_t g_tex_frame;   // presents so far (gfx_vk_tex.c)
enum { KEEP = 60, KEEP_GPU = 600, BUDGET_LOW_PCT = 90, MAX_PER_CALL = 1024 };
static uint64_t g_budget = ~0ull, g_evicted, g_evicted_bytes, g_sweeps;
static uint64_t mem_total_kb(void) {
    FILE *f = fopen("/proc/meminfo", "r"); if (!f) return 0;
    char line[256]; uint64_t kb = 0;
    while (fgets(line, sizeof line, f)) if (sscanf(line, "MemTotal: %llu kB", (unsigned long long *)&kb) == 1) break;
    fclose(f); return kb;
}
static uint64_t budget(void) {
    if (g_budget != ~0ull) return g_budget;
    const char *e = getenv("DM_VK_IMAGE_BUDGET_MB"); uint64_t mb;
    if (e) mb = (uint64_t)atoll(e);
    else { uint64_t gb = mem_total_kb() >> 20; mb = gb >= 12 ? 2048 : gb >= 8 ? 1536 : 1024; }
    g_budget = mb << 20;
    VKLOG("texture budget: %llu MB of image memory (DM_VK_IMAGE_BUDGET_MB; 0 = none); the least recently drawn images go beyond it", (unsigned long long)mb);
    return g_budget;
}
typedef struct { uint64_t age; int tier; vk_img_ref r; } cand;
static int cand_cmp(const void *a, const void *b) {   // re-creatable images first, then the oldest
    const cand *x = a, *y = b; if (x->tier != y->tier) return x->tier - y->tier; return x->age > y->age ? -1 : x->age < y->age;
}
void vk_img_budget_make_room(uint64_t incoming) {
    uint64_t b = budget(); if (!b) return;
    uint64_t have = vk_img_vkbytes(); if (have + incoming <= b) return;
    uint64_t target = b * BUDGET_LOW_PCT / 100; if (target > incoming) target -= incoming; else target = 0;
    static vk_img_ref *list; static size_t cap; static cand *cs; static size_t ccap;
    size_t n = vk_img_list(&list, &cap), nc = 0;
    if (ccap < n) { ccap = n; cs = realloc(cs, ccap * sizeof *cs); if (!cs) { ccap = 0; return; } }
    uint64_t now = g_tex_frame;
    for (size_t i = 0; i < n; i++) {
        vkimg_t *im = list[i].im;
        uint64_t idle = now - im->used; if (idle < KEEP) continue;
        if (im->rt_last && now + 1 - im->rt_last < KEEP) continue;   // drawn to in the last KEEP presents (rt_last = 1 + the present)
        int gpu = im->gpu_w || (im->rt_last && !im->packaged);
        if (gpu && idle < KEEP_GPU) continue;
        if (!im->vkbytes) continue;
        cs[nc++] = (cand){idle, gpu ? 1 : 0, list[i]};
    }
    if (!nc) { static int said; if (!said++) VKLOG("texture budget: %.0f MB of images over the %llu MB budget, none old enough to drop (looked up or drawn in the last %d presents)", have / 1048576.0, (unsigned long long)(b >> 20), KEEP); return; }
    qsort(cs, nc, sizeof *cs, cand_cmp);
    uint64_t before = have; int dropped = 0;
    for (size_t i = 0; i < nc && have > target && dropped < MAX_PER_CALL; i++) {
        uint64_t sz = cs[i].r.im->vkbytes; vk_img_evict(&cs[i].r);
        have = vk_img_vkbytes(); dropped++; g_evicted++; g_evicted_bytes += sz;
    }
    g_sweeps++;
    static int nl; if (nl++ < 12 || g_sweeps % 100 == 0) VKLOG("texture budget: %d images dropped (%.0f MB; %.0f → %.0f MB of %llu, incoming %.1f MB); %llu dropped so far in %llu sweeps", dropped, (before - have) / 1048576.0, before / 1048576.0, have / 1048576.0, (unsigned long long)(b >> 20), incoming / 1048576.0, (unsigned long long)g_evicted, (unsigned long long)g_sweeps);
}
void vk_img_budget_stats(uint64_t *bu, uint64_t *evicted, uint64_t *evicted_bytes, uint64_t *sweeps) { *bu = budget(); *evicted = g_evicted; *evicted_bytes = g_evicted_bytes; *sweeps = g_sweeps; }
