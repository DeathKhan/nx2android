// DM_NVN_CAPTURE=<frame>:1:<path> on the Vulkan backend: the counterpart of dm/gfx/nvn/nvn_capture.inc (nvn_mtl.m), the same file format (nvn_capture.h), replayed by
// dm/android/tools/nvn_replay_vk.c on the device. Included at the end of gfx_vk_tex.c: it reads that file's registries (g_treg, g_sreg) and image tables (g_by_key,
// g_by_serial), and the pool list mtl_pool_add / mtl_pool_del keep for it (vk_cap_pool_note, gfx_vk_core.c).
//
// What differs from the Metal capture: the contents of the images the GPU wrote (vkimg_t gpu_w: attachments, storage images, copy and clear destinations) are read back
// through a command buffer of the capture's own (outside the replay's submissions, after mtl_finish), with their VkFormat as the chunk's pixfmt; the program interface
// hashes are 0 (the replayer checks instead that the bundle has every stage's shader); textures are never "live" (mtl_tex_init is a no-op here).
// A capture needs the submissions replayed in order on the thread that presents: DM_NVN_ASYNC=0 and DM_NVN_LOGICAL_QUEUES=0 (it refuses otherwise).
#include "nvn_capture.h"
#include "dmsh_chain.h"
#include <CommonCrypto/CommonDigest.h>
#include <sys/param.h>

static int g_cap_frame = -1;            // -1: not parsed yet, 0: off
static char g_cap_path[1024];
static int g_cap_active, g_cap_done, g_cap_failed;
static cap_queue_state g_cap_queues;
void nvn_cap_queue(uint64_t serial, int waits) { cap_queue_note(&g_cap_queues, serial, waits); }
static uint32_t g_cap_submit;
static uint64_t g_cap_presents;
static cap_writer g_cap_w;
static uint64_t g_cap_ns, g_cap_unmapped, g_cap_segs, g_cap_tex_bytes, g_cap_tex_n, g_cap_tex_skip;
typedef struct { uint64_t lo, size; uint32_t kind, pending; uint64_t *hash; } cap_region;
static cap_region *g_cap_reg; static uint32_t g_cap_nreg;
static nvn_hmap g_cap_tex = {.klen = sizeof(tex_t)}, g_cap_prog = {.klen = 16}, g_cap_dpol = {.klen = 8 + sizeof(dpool_t)}, g_cap_sync = {.klen = 8}, g_cap_pool = {.klen = 8 + sizeof(pool_t)};
static nvn_hmap g_cap_seg = {.klen = 8}, g_cap_read = {.klen = 8}, g_cap_vimg = {.klen = 8};
static int g_cap_no_read;
static uint32_t g_cap_ntex, g_cap_nprog, g_cap_ndpol, g_cap_nsync, g_cap_npool;
static cap_buf g_cap_tobj, g_cap_pobj, g_cap_progb, g_cap_dpolb, g_cap_kind;
typedef struct { uint8_t ok, has_v; tex_t t; view_t v; } cap_treg0;
static cap_treg0 *g_cap_treg0; static int g_cap_ntreg0; static samp_par *g_cap_sreg0; static uint8_t *g_cap_sreg0_ok; static int g_cap_nsreg0;

// The pools as hle_nvn.c gave them to mtl_pool_add (the replayer gives the same ones to its own mtl_pool_add): kept from the start, a capture may begin at any frame
typedef struct { uint64_t lo, hi; } cap_pent;
static cap_pent *g_cap_pl; static int g_cap_npl, g_cap_plcap; static pthread_mutex_t g_cap_pmu = PTHREAD_MUTEX_INITIALIZER;
void vk_cap_pool_note(const void *mem, uint64_t size, int add) {
    uint64_t lo = (uint64_t)(uintptr_t)mem;
    pthread_mutex_lock(&g_cap_pmu);
    if (add) { if (g_cap_npl == g_cap_plcap) { g_cap_plcap = g_cap_plcap ? 2 * g_cap_plcap : 64; g_cap_pl = realloc(g_cap_pl, (size_t)g_cap_plcap * sizeof *g_cap_pl); }
        g_cap_pl[g_cap_npl++] = (cap_pent){lo, lo + size}; }
    else for (int i = 0; i < g_cap_npl; i++) if (g_cap_pl[i].lo == lo) { g_cap_pl[i] = g_cap_pl[--g_cap_npl]; break; }
    pthread_mutex_unlock(&g_cap_pmu);
}

// DM_NVN_CAPTURE=<frame>:1:<path>, frame = the present number (1 = the first), or +n: n presents after the setting is seen. Read again every 30 presents until it is
// set, so a DM_CTL "env" line arms it while the game runs (the frame to capture is rarely known at launch)
static int nvn_cap_enabled(void) {
    if (g_cap_done) return 0;
    if (g_cap_frame > 0) return 1;
    static uint64_t looked = ~0ull;
    if (looked != ~0ull && g_cap_presents - looked < 30) return 0;
    looked = g_cap_presents;
    const char *e = getenv("DM_NVN_CAPTURE"); int f = 0, n = 0, used = 0, rel = e && *e == '+';
    if (!e || !*e) return 0;
    if (sscanf(e + rel, "%d:%d:%n", &f, &n, &used) < 2 || !used || f < 1 || !e[rel + used]) { LOG("[RED] DM_NVN_CAPTURE=%s: expected <frame>:1:<path> or +<n>:1:<path>; capture off", e); g_cap_done = 1; return 0; }
    const char *a = getenv("DM_NVN_ASYNC"), *q = getenv("DM_NVN_LOGICAL_QUEUES");
    if ((a && atoi(a)) || (q && atoi(q))) { LOG("[RED] DM_NVN_CAPTURE needs DM_NVN_ASYNC=0 and DM_NVN_LOGICAL_QUEUES=0 from the start (the frame's submissions replayed in order before its present): capture off"); g_cap_done = 1; return 0; }
    if (n != 1) LOG("[RED] DM_NVN_CAPTURE: %d frames asked, step 1 captures one; capturing frame %d only", n, f);
    strlcpy(g_cap_path, e + rel + used, sizeof g_cap_path); g_cap_frame = rel ? (int)g_cap_presents + f : f;
    if ((uint64_t)g_cap_frame <= g_cap_presents) { LOG("[RED] DM_NVN_CAPTURE: frame %d is past (present %llu now); capture off", g_cap_frame, (unsigned long long)g_cap_presents); g_cap_done = 1; return 0; }
    LOG("DM_NVN_CAPTURE: capturing frame %d (present %llu now) into %s", g_cap_frame, (unsigned long long)g_cap_presents, g_cap_path);
    return 1;
}
static uint64_t cap_hash_block(const uint8_t *p, size_t n) {
    uint64_t h = 0x9E3779B97F4A7C15ull ^ n, v; size_t i = 0;
    for (; i + 8 <= n; i += 8) { memcpy(&v, p + i, 8); h = (h ^ v) * 0xff51afd7ed558ccdull; h ^= h >> 32; }
    for (; i < n; i++) h = (h ^ p[i]) * 0x100000001b3ull;
    return h;
}
static void cap_fail(const char *why) { if (!g_cap_failed++) LOG("[RED] DM_NVN_CAPTURE: %s: the capture is incomplete", why); }
static int cap_reject_queues(void) {
    const char *why = cap_queue_unsupported(&g_cap_queues); if (!why) return 0;
    cap_fail(why);
    if (!g_cap_w.f && !cap_open(&g_cap_w, g_cap_path)) LOG("[RED] DM_NVN_CAPTURE: cannot write failure marker to %s", g_cap_path);
    if (g_cap_w.f) { cap_put(&g_cap_w, CAP_FAIL, g_cap_submit, why, strlen(why), 0); cap_close(&g_cap_w); }
    g_cap_active = 0; g_cap_done = 1; return 1;
}

// ---------- Regions and the address encoding (as nvn_capture.inc) ----------
static int cap_reg_find(uint64_t a) {
    int lo = 0, hi = (int)g_cap_nreg - 1;
    while (lo <= hi) { int m = (lo + hi) / 2; if (a < g_cap_reg[m].lo) hi = m - 1; else if (a >= g_cap_reg[m].lo + g_cap_reg[m].size) lo = m + 1; else return m; }
    return -1;
}
static void cap_reg_hash(cap_region *r, const uint8_t *p) {
    size_t nb = (size_t)((r->size + 4095) / 4096); r->hash = realloc(r->hash, nb * 8);
    for (size_t b = 0; b < nb; b++) { uint64_t o = (uint64_t)b * 4096; r->hash[b] = cap_hash_block(p + o, (size_t)MIN(4096, r->size - o)); }
}
static uint32_t *g_cap_rid; static uint32_t g_cap_nrid;
static int cap_reg_add(uint64_t lo, uint64_t size, uint32_t kind) {
    int pos = 0; while (pos < (int)g_cap_nreg && g_cap_reg[pos].lo < lo) pos++;
    g_cap_reg = realloc(g_cap_reg, (g_cap_nreg + 1) * sizeof *g_cap_reg); g_cap_rid = realloc(g_cap_rid, (g_cap_nreg + 1) * sizeof *g_cap_rid);
    memmove(g_cap_reg + pos + 1, g_cap_reg + pos, (g_cap_nreg - pos) * sizeof *g_cap_reg); memmove(g_cap_rid + pos + 1, g_cap_rid + pos, (g_cap_nreg - pos) * sizeof *g_cap_rid);
    g_cap_reg[pos] = (cap_region){lo, size, kind, 1, NULL}; g_cap_rid[pos] = g_cap_nrid++; g_cap_nreg++;
    cap_reg_hash(&g_cap_reg[pos], (const uint8_t *)(uintptr_t)lo);
    return pos;
}
static uint32_t g_cap_cur_submit;
static char g_cap_where[96]; static const uint8_t *g_cap_where_base;
#define CAP_WHERE(base, ...) (snprintf(g_cap_where, sizeof g_cap_where, __VA_ARGS__), g_cap_where_base = (const uint8_t *)(base))
static const char *cap_where(const uint64_t *field) {
    static char b[128]; uint64_t off = (uint64_t)((const uint8_t *)field - g_cap_where_base);
    if (g_cap_where_base && off < (64u << 20)) snprintf(b, sizeof b, "%s, field +%llu", g_cap_where, (unsigned long long)off); else strlcpy(b, g_cap_where, sizeof b);
    return b;
}
int nvn_host_readable(uint64_t addr, uint64_t len);
static int cap_enc_addr(void *ctx, uint64_t *field, uint64_t len, int writable) {
    (void)ctx; (void)writable;
    uint64_t v = *field; if (!v) return 1;
    int i = cap_reg_find(v);
    if (i < 0 && len && len <= (1u << 20) && nvn_host_readable(v, len)) {
        for (uint32_t k = 0; k < g_cap_nreg; k++) if (g_cap_reg[k].lo < v + len && v < g_cap_reg[k].lo + g_cap_reg[k].size) { cap_fail("a record's memory straddles a region");
            LOG("[RED] DM_NVN_CAPTURE: %#llx (%llu bytes) straddles region %#llx (%llu bytes): %s", (unsigned long long)v, (unsigned long long)len, (unsigned long long)g_cap_reg[k].lo,
                (unsigned long long)g_cap_reg[k].size, cap_where(field)); return 1; }
        i = cap_reg_add(v, len, 1);
    }
    if (i < 0) { if (g_cap_unmapped++ < 8) LOG("[RED] DM_NVN_CAPTURE: a record names %#llx (%llu bytes), outside every pool: kept as it is: %s", (unsigned long long)v, (unsigned long long)len, cap_where(field)); return 1; }
    *field = cap_addr(g_cap_rid[i], v - g_cap_reg[i].lo);
    return 1;
}

// ---------- Objects ----------
static int cap_enc_obj(void *ctx, int kind, uint64_t *field, uint8_t *payload, uint32_t size);
static const cap_ops g_cap_enc = {cap_enc_addr, cap_enc_obj};
static void cap_tex_read(const tex_t *t, int rec);
static uint64_t cap_pool_obj(const pool_t *p) {
    if (!p) return 0;
    uint8_t k[8 + sizeof(pool_t)]; memcpy(k, &p, 8); memcpy(k + 8, p, sizeof *p);
    void *have = nvn_hm_get(&g_cap_pool, k); if (have) return cap_obj(CAP_K_POOL, (uint64_t)(uintptr_t)have - 1);
    uint32_t idx = g_cap_npool++; nvn_hm_put(&g_cap_pool, k, (void *)(uintptr_t)(idx + 1));
    cap_pobj o = {idx, p->flags, (uint64_t)(uintptr_t)p->mem, p->size, p->vmode, 0};
    if (p->vmode == 2) cap_fail("a sparse virtual pool (step 1 does not capture page tables)");
    cap_enc_addr(NULL, &o.mem, 0, 0); cap_buf_put(&g_cap_pobj, &o, sizeof o);
    return cap_obj(CAP_K_POOL, idx);
}
static void cap_tex_enc(tex_t *t, const char *what) {
    char sv[sizeof g_cap_where]; const uint8_t *sb = g_cap_where_base; memcpy(sv, g_cap_where, sizeof sv);
    CAP_WHERE(t, "%s ser=%llu%s%s", what, (unsigned long long)t->serial, sb ? " in " : "", sb ? sv : "");
    cap_tex_fields(t, &g_cap_enc, NULL);
    memcpy(g_cap_where, sv, sizeof sv); g_cap_where_base = sb;
}
static uint64_t cap_tex_obj(const tex_t *src) {
    tex_t t = *src;
    void *have = nvn_hm_get(&g_cap_tex, &t); if (have) return cap_obj(CAP_K_TEX, (uint64_t)(uintptr_t)have - 1);
    uint32_t idx = g_cap_ntex++; nvn_hm_put(&g_cap_tex, &t, (void *)(uintptr_t)(idx + 1));
    cap_tex_read(&t, 1);
    cap_tex_enc(&t, "texture object");
    uint32_t hd[2] = {idx, 0}; cap_buf_put(&g_cap_tobj, hd, 8); cap_buf_put(&g_cap_tobj, &t, sizeof t);
    return cap_obj(CAP_K_TEX, idx);
}
static const uint8_t *g_cap_orig;
static uint64_t cap_seg_get(uint64_t addr);
int nvn_dpool_base(const void *pool, int tex);
static int cap_enc_obj(void *ctx, int kind, uint64_t *field, uint8_t *payload, uint32_t size) {
    (void)ctx; uint64_t v = *field; if (!v && kind != CAP_K_SEG) return 1;
    switch (kind) {
        case CAP_K_SEG: *field = cap_obj(CAP_K_SEG, cap_seg_get(v)); return 1;
        case CAP_K_POOL: *field = cap_pool_obj((const pool_t *)(uintptr_t)v); return 1;
        case CAP_K_TEX:
            if (g_cap_orig && v >= (uint64_t)(uintptr_t)g_cap_orig && v + sizeof(tex_t) <= (uint64_t)(uintptr_t)g_cap_orig + size) {
                uint64_t off = v - (uint64_t)(uintptr_t)g_cap_orig; tex_t *in = (tex_t *)(payload + off);
                cap_tex_read(in, 1); cap_tex_enc(in, "texture snapshot"); *field = cap_obj(CAP_K_TEXIN, off); return 1;
            }
            *field = cap_tex_obj((const tex_t *)(uintptr_t)v); return 1;
        case CAP_K_PROG: {
            const prog_t *p = (const prog_t *)(uintptr_t)v; uint64_t k[2] = {v, p->serial};
            void *have = nvn_hm_get(&g_cap_prog, k); if (have) { *field = cap_obj(CAP_K_PROG, (uint64_t)(uintptr_t)have - 1); return 1; }
            uint32_t idx = g_cap_nprog++; nvn_hm_put(&g_cap_prog, k, (void *)(uintptr_t)(idx + 1));
            cap_prog o = {idx, p->nstage, p->serial, {{0}}, 0, {0}, {0}};
            for (unsigned s = 0; s < p->nstage && s < 6; s++) if (prog_key(p, s, o.key[s])) o.ok |= (uint8_t)(1u << s);
            cap_buf_put(&g_cap_progb, &o, sizeof o); *field = cap_obj(CAP_K_PROG, idx); return 1;
        }
        case CAP_K_DPOOL: {
            const dpool_t *d = (const dpool_t *)(uintptr_t)v; uint8_t k[8 + sizeof(dpool_t)]; memcpy(k, &v, 8); memcpy(k + 8, d, sizeof *d);
            void *have = nvn_hm_get(&g_cap_dpol, k); if (have) { *field = cap_obj(CAP_K_DPOOL, (uint64_t)(uintptr_t)have - 1); return 1; }
            uint32_t idx = g_cap_ndpol++; nvn_hm_put(&g_cap_dpol, k, (void *)(uintptr_t)(idx + 1));
            cap_dpol o = {idx, d->n, cap_pool_obj(d->pool), d->off, nvn_dpool_base(d, 1), nvn_dpool_base(d, 0)};
            cap_buf_put(&g_cap_dpolb, &o, sizeof o); *field = cap_obj(CAP_K_DPOOL, idx); return 1;
        }
        case CAP_K_SYNC: {
            void *have = nvn_hm_get(&g_cap_sync, &v); uint32_t idx = have ? (uint32_t)(uintptr_t)have - 1 : g_cap_nsync++;
            if (!have) nvn_hm_put(&g_cap_sync, &v, (void *)(uintptr_t)(idx + 1));
            *field = cap_obj(CAP_K_SYNC, idx); return 1;
        }
    }
    return 0;
}

// ---------- Segments ----------
static uint64_t *g_cap_segq; static uint32_t g_cap_nseg, g_cap_segq_cap;
static uint64_t cap_seg_get(uint64_t addr) {
    void *have = nvn_hm_get(&g_cap_seg, &addr); if (have) return (uint64_t)(uintptr_t)have - 1;
    uint32_t idx = g_cap_nseg++; nvn_hm_put(&g_cap_seg, &addr, (void *)(uintptr_t)(idx + 1));
    if (g_cap_nseg > g_cap_segq_cap) { g_cap_segq_cap = g_cap_segq_cap ? g_cap_segq_cap * 2 : 256; g_cap_segq = realloc(g_cap_segq, g_cap_segq_cap * 8); }
    g_cap_segq[idx] = addr; return idx;
}
static void cap_seg_write(uint32_t idx) {
    const uint8_t *p = (const uint8_t *)(uintptr_t)g_cap_segq[idx]; size_t n = 0;
    for (;;) {
        op_hdr h; memcpy(&h, p + n, 8);
        if (h.size < 8 || (h.size & 7) || h.size > (64u << 20) || h.op >= OP_COUNT_) { cap_fail("a corrupt record in a submitted segment"); LOG("[RED] DM_NVN_CAPTURE: segment %p offset %zu: op %u size %u", p, n, h.op, h.size); break; }
        n += h.size;
        if (h.op == OP_END || h.op == OP_JUMP) break;
        if (n > (256u << 20)) { cap_fail("a segment longer than 256 MiB without END"); break; }
    }
    uint8_t *b = malloc(8 + n); uint32_t hd[2] = {idx, (uint32_t)n}; memcpy(b, hd, 8); memcpy(b + 8, p, n);
    for (size_t o = 0; o < n;) {
        op_hdr h; memcpy(&h, b + 8 + o, 8);
        g_cap_orig = p + o + 8; CAP_WHERE(b + 8 + o + 8, "op %u (%u bytes) at +%zu of segment %u", h.op, h.size, o, idx);
        const char *unsupported = cap_record_unsupported(h.op);
        if (unsupported) { cap_fail(unsupported); cap_put(&g_cap_w, CAP_FAIL, g_cap_cur_submit, unsupported, strlen(unsupported), 0); }
        if (!cap_rec_fields(h.op, b + 8 + o + 8, h.size, &g_cap_enc, NULL)) { cap_fail("a record the capture walk does not know"); LOG("[RED] DM_NVN_CAPTURE: op %u size %u not walked", h.op, h.size); }
        o += h.size;
    }
    g_cap_orig = NULL; CAP_WHERE(NULL, "-");
    cap_put(&g_cap_w, CAP_SEGM, g_cap_cur_submit, b, 8 + n, 1); free(b); g_cap_segs++;
}

// ---------- Image contents: read back / filled through a command buffer of the capture's own ----------
// Bytes per block of a plane of a Vulkan format and its block size; 0 = not read back (multi-planar, unknown)
int vk_cap_fmt_block(VkFormat f, int plane, uint32_t *bw, uint32_t *bh) {
    *bw = *bh = 1;
    switch (f) {
    case VK_FORMAT_D32_SFLOAT_S8_UINT: return plane ? 1 : 4;
    case VK_FORMAT_D24_UNORM_S8_UINT: return plane ? 1 : 4;
    case VK_FORMAT_D16_UNORM_S8_UINT: return plane ? 1 : 2;
    case VK_FORMAT_D32_SFLOAT: case VK_FORMAT_X8_D24_UNORM_PACK32: return plane ? 0 : 4;
    case VK_FORMAT_D16_UNORM: return plane ? 0 : 2;
    case VK_FORMAT_S8_UINT: return plane ? 0 : 1;
    default: break;
    }
    if (plane) return 0;
    if (f >= VK_FORMAT_BC1_RGB_UNORM_BLOCK && f <= VK_FORMAT_BC7_SRGB_BLOCK) { *bw = *bh = 4; return f <= VK_FORMAT_BC1_RGBA_SRGB_BLOCK || f == VK_FORMAT_BC4_UNORM_BLOCK || f == VK_FORMAT_BC4_SNORM_BLOCK ? 8 : 16; }
    if (f >= VK_FORMAT_ASTC_4x4_UNORM_BLOCK && f <= VK_FORMAT_ASTC_12x12_SRGB_BLOCK) {
        static const uint8_t d[14][2] = {{4,4},{5,4},{5,5},{6,5},{6,6},{8,5},{8,6},{8,8},{10,5},{10,6},{10,8},{10,10},{12,10},{12,12}};
        int i = (f - VK_FORMAT_ASTC_4x4_UNORM_BLOCK) / 2; *bw = d[i][0]; *bh = d[i][1]; return 16; }
    switch (f) {
    case VK_FORMAT_R8_UNORM: case VK_FORMAT_R8_SNORM: case VK_FORMAT_R8_UINT: case VK_FORMAT_R8_SINT: case VK_FORMAT_R8_SRGB: return 1;
    case VK_FORMAT_R8G8_UNORM: case VK_FORMAT_R8G8_SNORM: case VK_FORMAT_R8G8_UINT: case VK_FORMAT_R8G8_SINT: case VK_FORMAT_R16_UNORM: case VK_FORMAT_R16_SNORM:
    case VK_FORMAT_R16_UINT: case VK_FORMAT_R16_SINT: case VK_FORMAT_R16_SFLOAT: case VK_FORMAT_R5G6B5_UNORM_PACK16: case VK_FORMAT_B5G6R5_UNORM_PACK16:
    case VK_FORMAT_R5G5B5A1_UNORM_PACK16: case VK_FORMAT_B5G5R5A1_UNORM_PACK16: case VK_FORMAT_A1R5G5B5_UNORM_PACK16: case VK_FORMAT_R4G4B4A4_UNORM_PACK16: case VK_FORMAT_B4G4R4A4_UNORM_PACK16: return 2;
    case VK_FORMAT_R16G16B16A16_UNORM: case VK_FORMAT_R16G16B16A16_SNORM: case VK_FORMAT_R16G16B16A16_UINT: case VK_FORMAT_R16G16B16A16_SINT: case VK_FORMAT_R16G16B16A16_SFLOAT:
    case VK_FORMAT_R32G32_UINT: case VK_FORMAT_R32G32_SINT: case VK_FORMAT_R32G32_SFLOAT: return 8;
    case VK_FORMAT_R32G32B32A32_UINT: case VK_FORMAT_R32G32B32A32_SINT: case VK_FORMAT_R32G32B32A32_SFLOAT: return 16;
    case VK_FORMAT_R32G32B32_UINT: case VK_FORMAT_R32G32B32_SINT: case VK_FORMAT_R32G32B32_SFLOAT: return 12;
    default: break;
    }
    if ((f >= VK_FORMAT_R8G8B8A8_UNORM && f <= VK_FORMAT_A2B10G10R10_SINT_PACK32) || (f >= VK_FORMAT_R16G16_UNORM && f <= VK_FORMAT_R16G16_SFLOAT)
        || (f >= VK_FORMAT_R32_UINT && f <= VK_FORMAT_R32_SFLOAT) || f == VK_FORMAT_B10G11R11_UFLOAT_PACK32 || f == VK_FORMAT_E5B9G9R9_UFLOAT_PACK32) return 4;
    return 0;
}
static VkImageAspectFlags cap_plane_aspect(const vkimg_t *im, int plane) {
    if (!(im->aspect & (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT))) return VK_IMAGE_ASPECT_COLOR_BIT;
    if (plane) return VK_IMAGE_ASPECT_STENCIL_BIT;
    return im->aspect & VK_IMAGE_ASPECT_DEPTH_BIT ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_STENCIL_BIT;
}
// Every level, layer and plane of an image, packed as the Metal capture packs them (plane, then level, then layer); *total: their bytes. 0 = a format not read back
static int cap_img_subs(const vkimg_t *im, cap_tsub **subs, uint32_t *nsub, uint64_t *total) {
    uint32_t bw, bh, np = cap_plane_aspect(im, 0) == VK_IMAGE_ASPECT_DEPTH_BIT && (im->aspect & VK_IMAGE_ASPECT_STENCIL_BIT) ? 2 : 1;
    if (!vk_cap_fmt_block(im->fmt, 0, &bw, &bh)) return 0;
    *nsub = 0; *total = 0; *subs = calloc((size_t)im->levels * im->layers * np, sizeof **subs);
    for (uint32_t pl = 0; pl < np; pl++) for (uint32_t lv = 0; lv < im->levels; lv++) for (uint32_t sl = 0; sl < im->layers; sl++) {
        uint32_t w = MAX(im->w >> lv, 1u), h = MAX(im->h >> lv, 1u), d = MAX(im->d >> lv, 1u);
        uint64_t bpb = (uint64_t)vk_cap_fmt_block(im->fmt, (int)pl, &bw, &bh), row = (uint64_t)((w + bw - 1) / bw) * bpb, img = row * ((h + bh - 1) / bh);
        (*subs)[(*nsub)++] = (cap_tsub){lv, sl, w, h, d, pl, row, img, img * d}; *total += img * d;
    }
    return 1;
}
// One command buffer, submitted and waited for: dir 0 image → buffer (read back), 1 buffer → image (fill). The image ends in GENERAL
static int cap_img_copy(vkimg_t *im, const cap_tsub *subs, uint32_t nsub, uint64_t total, void *bytes, int dir) {
    static VkCommandPool cp; static VkFence fe;
    if (!cp) { VkCommandPoolCreateInfo pi = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, NULL, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, V.qf};
        VkFenceCreateInfo fi = {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        if (vkCreateCommandPool(V.dev, &pi, NULL, &cp) || vkCreateFence(V.dev, &fi, NULL, &fe)) return 0; }
    VkBufferCreateInfo bi = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, NULL, 0, MAX(total, 16u), VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_SHARING_MODE_EXCLUSIVE};
    VkBuffer b; if (vkCreateBuffer(V.dev, &bi, NULL, &b)) return 0;
    VkMemoryRequirements mr; vkGetBufferMemoryRequirements(V.dev, b, &mr);
    VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL, mr.size, vk_mem_type(mr.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 0)};
    VkDeviceMemory m; void *p = NULL; vk_mem_tag = "capture";
    if (vkAllocateMemory(V.dev, &ai, NULL, &m)) { vkDestroyBuffer(V.dev, b, NULL); return 0; }
    vkBindBufferMemory(V.dev, b, m, 0); vkMapMemory(V.dev, m, 0, VK_WHOLE_SIZE, 0, &p);
    if (dir) memcpy(p, bytes, (size_t)total);
    VkCommandBuffer cb; VkCommandBufferAllocateInfo ci = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, NULL, cp, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1};
    vkAllocateCommandBuffers(V.dev, &ci, &cb);
    VkCommandBufferBeginInfo be = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, NULL, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
    vkBeginCommandBuffer(cb, &be);
    VkImageMemoryBarrier ib = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, NULL, VK_ACCESS_MEMORY_WRITE_BIT, VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
        im->layout == VK_IMAGE_LAYOUT_GENERAL ? VK_IMAGE_LAYOUT_GENERAL : VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL, VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, im->img,
        {im->aspect, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS}};
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, NULL, 0, NULL, 1, &ib);
    uint64_t off = 0;
    for (uint32_t i = 0; i < nsub; i++) { const cap_tsub *s = &subs[i]; uint32_t bw, bh; vk_cap_fmt_block(im->fmt, (int)s->plane, &bw, &bh);
        VkBufferImageCopy c = {off, (uint32_t)((s->row / MAX(vk_cap_fmt_block(im->fmt, (int)s->plane, &bw, &bh), 1)) * bw), (uint32_t)((s->img / MAX(s->row, 1)) * bh),
            {cap_plane_aspect(im, (int)s->plane), s->level, im->d > 1 ? 0 : s->slice, 1}, {0, 0, 0}, {s->w, s->h, s->d}};
        if (dir) vkCmdCopyBufferToImage(cb, b, im->img, VK_IMAGE_LAYOUT_GENERAL, 1, &c); else vkCmdCopyImageToBuffer(cb, im->img, VK_IMAGE_LAYOUT_GENERAL, b, 1, &c);
        off += s->bytes; }
    VkMemoryBarrier mb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, NULL, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT | VK_ACCESS_MEMORY_READ_BIT};
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT | VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
    vkEndCommandBuffer(cb);
    VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO, NULL, 0, NULL, NULL, 1, &cb};
    extern pthread_mutex_t g_vk_queue_mu;
    pthread_mutex_lock(&g_vk_queue_mu); VkResult r = vkQueueSubmit(V.q, 1, &si, fe); pthread_mutex_unlock(&g_vk_queue_mu);
    if (!r) r = vkWaitForFences(V.dev, 1, &fe, VK_TRUE, ~0ull);
    vkResetFences(V.dev, 1, &fe);
    if (!r) { im->layout = VK_IMAGE_LAYOUT_GENERAL; if (!dir) memcpy(bytes, p, (size_t)total); }
    vkFreeCommandBuffers(V.dev, cp, 1, &cb); vkUnmapMemory(V.dev, m); vkDestroyBuffer(V.dev, b, NULL); vkFreeMemory(V.dev, m, NULL);
    return r == VK_SUCCESS;
}
// The image a texture has now, without making one
static vkimg_t *cap_img_peek(const tex_t *t) {
    const vkfmt_t *f = vk_fmt(t->d.format); if (!f || t->d.target == NVN_TEXTURE_TARGET_BUFFER) return NULL;
    akey k; tex_storage_key(&k, t, f);
    pthread_mutex_lock(&g_img_mu); vkimg_t *im = tex_image_find(t, &k, NULL); pthread_mutex_unlock(&g_img_mu);
    return im;
}
static void cap_tex_read(const tex_t *t, int rec) {   // the contents of a texture the frame can reach, if the GPU wrote them (the replay makes the others from the captured pools)
    (void)rec;
    if (g_cap_no_read || !t || !t->serial || t->magic != MAGIC('T', 'E', 'X', 0)) return;
    if (nvn_hm_get(&g_cap_read, &t->serial)) return;
    nvn_hm_put(&g_cap_read, &t->serial, (void *)1);
    pthread_mutex_lock(&g_cw_mu); int cpu = nvn_hm_get(&g_cpu_written, &t->serial) != NULL; pthread_mutex_unlock(&g_cw_mu);
    if (cpu) { cap_tknd k = {*t, CAP_TK_CPU, 0}; cap_tex_enc(&k.t, "texture kind"); cap_buf_put(&g_cap_kind, &k, sizeof k); }
    vkimg_t *im = cap_img_peek(t);
    if (!im || !im->gpu_w || im->orphan || im->layout != VK_IMAGE_LAYOUT_GENERAL) return;
    uint64_t key = (uint64_t)(uintptr_t)im; if (nvn_hm_get(&g_cap_vimg, &key)) return;
    nvn_hm_put(&g_cap_vimg, &key, (void *)1);
    cap_tsub *subs = NULL; uint32_t nsub = 0; uint64_t total = 0;
    if (im->samples > 1 || !cap_img_subs(im, &subs, &nsub, &total)) {
        if (g_cap_tex_skip++ < 8) LOG("[RED] DM_NVN_CAPTURE: texture ser=%llu %ux%u format %d samples %u: content not captured (%s)", (unsigned long long)t->serial, im->w, im->h, im->fmt, im->samples,
                                      im->samples > 1 ? "multisampled" : "format the capture cannot read back");
        free(subs); return;
    }
    cap_buf c = {0};
    cap_tpix h = {*t, (uint32_t)im->fmt, (uint32_t)im->vtype, im->w, im->h, im->d, im->levels, im->layers, im->samples, nsub, 0};
    cap_tex_enc(&h.t, "texture contents");
    cap_buf_put(&c, &h, sizeof h); cap_buf_put(&c, subs, nsub * sizeof *subs);
    size_t at = c.n;   // the image's bytes go straight after the header and the subresource table
    c.p = realloc(c.p, at + (size_t)total + 1); c.n = at + (size_t)total; c.cap = c.n + 1;
    if (!cap_img_copy(im, subs, nsub, total, c.p + at, 0)) { cap_fail("an image could not be read back"); cap_buf_free(&c); free(subs); return; }
    cap_put(&g_cap_w, CAP_TPIX, g_cap_cur_submit, c.p, c.n, 1);
    g_cap_tex_bytes += total; g_cap_tex_n++;
    cap_buf_free(&c); free(subs);
}
// The replayer's side of TPIX: the image made as the replay makes it (vk_tex, inside a submission), then filled
int vk_cap_fill(const tex_t *t, const uint8_t *d, size_t n, char *why, size_t nwhy) {
    const cap_tpix *h = (const cap_tpix *)d; const cap_tsub *subs = (const cap_tsub *)(d + sizeof *h); const uint8_t *px = (const uint8_t *)(subs + h->nsub);
    vkimg_t *im = cap_img_peek(t);
    if (!im) { snprintf(why, nwhy, "the replay layer made no image for it"); return 0; }
    if ((uint32_t)im->fmt != h->pixfmt || im->w != h->width || im->h != h->height || im->levels != h->levels || im->layers != h->slices) {
        snprintf(why, nwhy, "captured as %ux%u format %u, %u levels, %u layers; this build makes it %ux%u format %d, %u levels, %u layers", h->width, h->height, h->pixfmt, h->levels, h->slices,
                 im->w, im->h, im->fmt, im->levels, im->layers); return 0; }
    uint64_t total = 0; for (uint32_t i = 0; i < h->nsub; i++) total += subs[i].bytes;
    if ((size_t)(px - d) + total > n) { snprintf(why, nwhy, "short chunk"); return 0; }
    if (!cap_img_copy(im, subs, h->nsub, total, (void *)px, 1)) { snprintf(why, nwhy, "the copy into it failed"); return 0; }
    im->gpu_w = 1; __atomic_store_n(&im->looked, 1, __ATOMIC_RELEASE);
    return 1;
}
// The presented texture, level 0 layer 0, tightly packed (both the capture and the replayer read it this way); *fmt its VkFormat
int vk_cap_read_tex(const tex_t *t, cap_buf *out, uint32_t *w, uint32_t *h, uint32_t *fmt) {
    vkimg_t *im = t ? cap_img_peek(t) : NULL; if (!im) return 0;
    uint32_t bw, bh; int bpb = vk_cap_fmt_block(im->fmt, 0, &bw, &bh); if (!bpb || bw != 1) return 0;
    cap_tsub s = {0, 0, im->w, im->h, 1, 0, (uint64_t)im->w * bpb, (uint64_t)im->w * bpb * im->h, (uint64_t)im->w * bpb * im->h};
    size_t at = out->n; out->p = realloc(out->p, at + (size_t)s.bytes + 1); out->n = at + (size_t)s.bytes; out->cap = out->n + 1;
    if (!cap_img_copy(im, &s, 1, s.bytes, out->p + at, 0)) { out->n = at; return 0; }
    *w = im->w; *h = im->h; *fmt = (uint32_t)im->fmt; return 1;
}

// ---------- Registries ----------
static void cap_registries(uint32_t submit) {
    cap_buf tb = {0}, sb = {0}; tex_t *reads = NULL; int nreads = 0;
    pthread_mutex_lock(&g_rmu);
    if (g_cap_ntreg0 < g_ntreg) { g_cap_treg0 = realloc(g_cap_treg0, g_ntreg * sizeof *g_cap_treg0); memset(g_cap_treg0 + g_cap_ntreg0, 0, (g_ntreg - g_cap_ntreg0) * sizeof *g_cap_treg0); g_cap_ntreg0 = g_ntreg; }
    reads = malloc((size_t)MAX(g_ntreg, 1) * sizeof *reads);
    for (int i = 0; i < g_ntreg; i++) {
        const treg_ver *r = &g_treg[i].cur; cap_treg0 *o = &g_cap_treg0[i];
        if (r->ok == o->ok && (!r->ok || (r->has_v == o->has_v && !memcmp(&r->t, &o->t, sizeof r->t) && (!r->has_v || !memcmp(&r->v, &o->v, sizeof r->v))))) continue;
        cap_treg e = {i, r->ok, r->has_v, 0, 0, r->t, r->v}; o->ok = r->ok; o->has_v = r->has_v; o->t = r->t; o->v = r->v;
        if (r->ok) reads[nreads++] = r->t;
        cap_buf_put(&tb, &e, sizeof e);
    }
    if (g_cap_nsreg0 < g_nsreg) { g_cap_sreg0 = realloc(g_cap_sreg0, g_nsreg * sizeof *g_cap_sreg0); g_cap_sreg0_ok = realloc(g_cap_sreg0_ok, g_nsreg); memset(g_cap_sreg0_ok + g_cap_nsreg0, 0, g_nsreg - g_cap_nsreg0); g_cap_nsreg0 = g_nsreg; }
    for (int i = 0; i < g_nsreg; i++) {
        if (!g_sreg_ok[i] || (g_cap_sreg0_ok[i] && !memcmp(&g_sreg[i], &g_cap_sreg0[i], sizeof g_sreg[i]))) continue;
        cap_sreg e = {i, 0, g_sreg[i]}; g_cap_sreg0[i] = g_sreg[i]; g_cap_sreg0_ok[i] = 1; cap_buf_put(&sb, &e, sizeof e);
    }
    pthread_mutex_unlock(&g_rmu);
    for (size_t o = 0; o < tb.n; o += sizeof(cap_treg)) { cap_treg *e = (cap_treg *)(tb.p + o); if (e->ok) { char w[32]; snprintf(w, sizeof w, "texture registry entry %d", e->tid); cap_tex_enc(&e->t, w); } }
    for (int i = 0; i < nreads; i++) cap_tex_read(&reads[i], 0);
    free(reads);
    if (tb.n) cap_put(&g_cap_w, CAP_TREG, submit, tb.p, tb.n, 1);
    if (sb.n) cap_put(&g_cap_w, CAP_SREG, submit, sb.p, sb.n, 1);
    cap_buf_free(&tb); cap_buf_free(&sb);
}

// ---------- The frame ----------
extern char **environ;
static void cap_head(void) {
    cap_buf h = {0}; char line[2048];
#define KV(...) do { int n_ = snprintf(line, sizeof line, __VA_ARGS__); cap_buf_put(&h, line, (size_t)MIN(n_, (int)sizeof line - 1)); } while (0)
    KV("format\t%u\n", CAP_VERSION); KV("iface\t0\n"); KV("backend\tvulkan\n"); KV("device\t%s\n", V.props.deviceName); KV("chain\t%s\n", DMSH_CHAIN); KV("commit\tunknown\n");
    KV("title\t%s\n", DM_GAME_ID); KV("build\t%s\n", dm_main_build());
    KV("frame\t%d\n", g_cap_frame); KV("frames\t1\n"); KV("mtl_frame\t%llu\n", (unsigned long long)g_tex_frame);
    for (char **e = environ; *e; e++) if (!strncmp(*e, "DM_", 3) && strncmp(*e, "DM_NVN_CAPTURE=", 15)) KV("env\t%s\n", *e);
#undef KV
    cap_put(&g_cap_w, CAP_HEAD, 0, h.p, h.n, 0); cap_buf_free(&h);
    const char *lay = cap_layout(); cap_put(&g_cap_w, CAP_LAYT, 0, lay, strlen(lay), 0);
    extern int vk_cap_origin_ll(void), vk_cap_depth_zero(void);
    snprintf(line, sizeof line, "tex_pool_n\t%d\norigin_ll\t%d\nqueue_origin_ll\t%d\ndepth_zero\t%d\n", 0, vk_cap_origin_ll(), vk_cap_origin_ll(), vk_cap_depth_zero());
    cap_put(&g_cap_w, CAP_STAT, 0, line, strlen(line), 0);
}
static int cap_pent_cmp(const void *a, const void *b) { uint64_t x = ((const cap_pent *)a)->lo, y = ((const cap_pent *)b)->lo; return x < y ? -1 : x > y; }
static void cap_pools(uint32_t submit) {
    pthread_mutex_lock(&g_cap_pmu);
    int n = g_cap_npl; cap_pent *ps = malloc((size_t)MAX(n, 1) * sizeof *ps); memcpy(ps, g_cap_pl, (size_t)n * sizeof *ps);
    pthread_mutex_unlock(&g_cap_pmu);
    qsort(ps, (size_t)n, sizeof *ps, cap_pent_cmp);
    for (int i = 0; i < n;) {
        uint64_t lo = ps[i].lo, hi = ps[i].hi; int j = i + 1;
        while (j < n && ps[j].lo < hi) { hi = MAX(hi, ps[j].hi); j++; }
        int k = cap_reg_find(lo);
        if (k < 0 || g_cap_reg[k].lo + g_cap_reg[k].size < hi) {
            if (k >= 0 || cap_reg_find(hi - 1) >= 0) cap_fail("a pool grew over an existing region during the frame");
            else cap_reg_add(lo, hi - lo, 0);
        }
        i = j;
    }
    cap_buf b = {0};
    for (int i = 0; i < n; i++) { cap_padd a = {ps[i].lo, ps[i].hi - ps[i].lo}; cap_enc_addr(NULL, &a.mem, 0, 0); cap_buf_put(&b, &a, sizeof a); }
    cap_put(&g_cap_w, CAP_PADD, submit, b.p, b.n, 1); cap_buf_free(&b); free(ps);
}
static uint64_t cap_reg_write(uint32_t submit) {
    uint64_t moved = 0;
    for (uint32_t i = 0; i < g_cap_nreg; i++) {
        cap_region *r = &g_cap_reg[i]; if (!r->pending) continue;
        size_t nb = (size_t)((r->size + 4095) / 4096);
        for (size_t k = 0; k < nb; k++) { uint64_t o = (uint64_t)k * 4096; size_t len = (size_t)MIN(4096, r->size - o); if (cap_hash_block((const uint8_t *)(uintptr_t)r->lo + o, len) != r->hash[k]) moved += len; }
        cap_regn hh = {g_cap_rid[i], r->kind, r->lo, r->size}; r->pending = 0;
        uint8_t *b = malloc(sizeof hh + r->size); if (!b) { cap_fail("out of memory for a region"); continue; }
        memcpy(b, &hh, sizeof hh); memcpy(b + sizeof hh, (const void *)(uintptr_t)r->lo, (size_t)r->size);
        cap_reg_hash(r, b + sizeof hh);
        cap_put(&g_cap_w, CAP_REGN, submit, b, sizeof hh + r->size, 1); free(b);
    }
    return moved;
}
static void cap_mem(uint32_t submit) {
    uint64_t moved = cap_reg_write(submit);
    if (moved) LOG("DM_NVN_CAPTURE: submit %u: %llu bytes of the pools changed while it was captured (written as they are at its end)", submit, (unsigned long long)moved);
    cap_buf b = {0};
    for (uint32_t i = 0; i < g_cap_nreg; i++) {
        cap_region *r = &g_cap_reg[i]; size_t nb = (size_t)((r->size + 4095) / 4096);
        for (size_t k = 0; k < nb;) {
            uint64_t o = (uint64_t)k * 4096; size_t len = (size_t)MIN(4096, r->size - o);
            uint64_t h = cap_hash_block((const uint8_t *)(uintptr_t)r->lo + o, len);
            if (h == r->hash[k]) { k++; continue; }
            size_t k1 = k; uint64_t end = o;
            while (k1 < nb) { uint64_t o1 = (uint64_t)k1 * 4096; size_t l1 = (size_t)MIN(4096, r->size - o1); uint64_t h1 = cap_hash_block((const uint8_t *)(uintptr_t)r->lo + o1, l1);
                if (h1 == r->hash[k1]) break; r->hash[k1] = h1; end = o1 + l1; k1++; }
            struct { uint32_t region, pad; uint64_t off, len; } d = {g_cap_rid[i], 0, o, end - o};
            cap_buf_put(&b, &d, sizeof d); cap_buf_put(&b, (const uint8_t *)(uintptr_t)r->lo + o, (size_t)(end - o));
            k = k1;
        }
    }
    if (b.n) cap_put(&g_cap_w, CAP_PDIF, submit, b.p, b.n, 1);
    cap_buf_free(&b);
}
static void cap_flush_objs(uint32_t submit) {
    if (g_cap_pobj.n) cap_put(&g_cap_w, CAP_POBJ, submit, g_cap_pobj.p, g_cap_pobj.n, 1);
    if (g_cap_tobj.n) cap_put(&g_cap_w, CAP_TOBJ, submit, g_cap_tobj.p, g_cap_tobj.n, 1);
    if (g_cap_progb.n) cap_put(&g_cap_w, CAP_PROG, submit, g_cap_progb.p, g_cap_progb.n, 1);
    if (g_cap_dpolb.n) cap_put(&g_cap_w, CAP_DPOL, submit, g_cap_dpolb.p, g_cap_dpolb.n, 1);
    if (g_cap_kind.n) cap_put(&g_cap_w, CAP_TKND, submit, g_cap_kind.p, g_cap_kind.n, 1);
    g_cap_pobj.n = g_cap_tobj.n = g_cap_progb.n = g_cap_dpolb.n = g_cap_kind.n = 0;
}
// Hook: submit_now (hle_nvn.c), under the submit lock, before the submission is replayed
void nvn_cap_submit(int n, const uint64_t *h) {
    if (!nvn_cap_enabled() || cap_reject_queues() || g_cap_presents + 1 != (uint64_t)g_cap_frame) return;
    uint64_t t0 = now_ns();
    mtl_finish();
    g_cap_cur_submit = ++g_cap_submit;
    if (!g_cap_active) {
        if (!cap_open(&g_cap_w, g_cap_path)) { LOG("[RED] DM_NVN_CAPTURE: cannot write %s: capture off", g_cap_path); g_cap_done = 1; return; }
        g_cap_active = 1;
        cap_head();
    }
    cap_pools(g_cap_cur_submit);
    uint64_t t1 = now_ns(); cap_registries(g_cap_cur_submit);
    uint64_t t2 = now_ns(); nvn_hm_clear(&g_cap_seg, NULL); g_cap_nseg = 0;
    uint32_t *top = malloc((size_t)MAX(n, 1) * 4);
    for (int i = 0; i < n; i++) top[i] = (uint32_t)cap_seg_get(h[i]);
    for (uint32_t i = 0; i < g_cap_nseg; i++) cap_seg_write(i);
    cap_flush_objs(g_cap_cur_submit);
    uint64_t t3 = now_ns(); cap_mem(g_cap_cur_submit);
    LOG("DM_NVN_CAPTURE: submit %u: registries and their textures %.1f s, records and their textures %.1f s, pool bytes %.1f s", g_cap_cur_submit, (t2 - t1) / 1e9, (t3 - t2) / 1e9, (now_ns() - t3) / 1e9);
    if (g_cap_nsync) { uint32_t ns = g_cap_nsync; cap_put(&g_cap_w, CAP_SYNC, g_cap_cur_submit, &ns, 4, 0); }
    cap_buf s = {0}; uint32_t nn = (uint32_t)n; cap_buf_put(&s, &nn, 4); cap_buf_put(&s, top, (size_t)n * 4);
    cap_put(&g_cap_w, CAP_SUBM, g_cap_cur_submit, s.p, s.n, 0); cap_buf_free(&s); free(top);
    g_cap_ns += now_ns() - t0;
}
void vk_cap_sha256(const void *p, size_t n, uint8_t md[32]) { CC_SHA256_CTX c; CC_SHA256_Init(&c); CC_SHA256_Update(&c, p, (CC_LONG)n); CC_SHA256_Final(md, &c); }
static void cap_hex(const uint8_t *md, char out[65]) { for (int i = 0; i < 32; i++) snprintf(out + 2 * i, 3, "%02x", md[i]); }
// Hook: nvnQueuePresentTexture, before the present is queued
void nvn_cap_present(const tex_t *t, const int crop[4]) {
    g_cap_presents++;   // counted whether or not a capture is armed: the frame numbers are presents since the start
    if (!nvn_cap_enabled() || cap_reject_queues()) return;
    if (!g_cap_active || g_cap_presents != (uint64_t)g_cap_frame) {
        if (g_cap_presents >= (uint64_t)g_cap_frame && !g_cap_active) { LOG("[RED] DM_NVN_CAPTURE: frame %d presented without a submit: nothing captured", g_cap_frame); g_cap_done = 1; }
        return;
    }
    uint64_t t0 = now_ns();
    cap_pres p = {0}; cap_buf px = {0};
    g_cap_cur_submit = g_cap_submit + 1; g_cap_no_read = 1;
    mtl_finish();
    if (t) { p.tex = cap_tex_obj(t); cap_flush_objs(g_cap_cur_submit); cap_reg_write(g_cap_cur_submit); }
    if (crop) memcpy(p.crop, crop, sizeof p.crop);
    p.frame = (uint32_t)g_cap_frame;
    if (!t || !vk_cap_read_tex(t, &px, &p.width, &p.height, &p.pixfmt)) cap_fail("the presented texture could not be read back");
    else { vk_cap_sha256(px.p, px.n, p.sha256); p.bytes = px.n; }
    g_cap_ns += now_ns() - t0; p.capture_ms = g_cap_ns / 1e6;
    cap_put(&g_cap_w, CAP_PRES, g_cap_cur_submit, &p, sizeof p, 0);
    if (px.n) cap_put(&g_cap_w, CAP_LPIX, g_cap_cur_submit, px.p, px.n, 1);
    cap_buf_free(&px);
    {
        cap_buf hb = {0};
        for (uint32_t i = 0; i < g_cap_nreg; i++) {
            cap_region *r = &g_cap_reg[i]; uint32_t nb = (uint32_t)((r->size + 4095) / 4096), hd[2] = {g_cap_rid[i], nb}; cap_buf_put(&hb, hd, 8);
            for (uint32_t k = 0; k < nb; k++) { uint64_t o = (uint64_t)k * 4096, hh = cap_hash_block((const uint8_t *)(uintptr_t)r->lo + o, (size_t)MIN(4096, r->size - o)); cap_buf_put(&hb, &hh, 8); }
        }
        cap_put(&g_cap_w, CAP_PHSH, g_cap_cur_submit, hb.p, hb.n, 1); cap_buf_free(&hb);
    }
    uint64_t raw = g_cap_w.raw, stored = g_cap_w.stored;
    int ok = cap_close(&g_cap_w); char hex[65]; cap_hex(p.sha256, hex);
    LOG("%sDM_NVN_CAPTURE: frame %d: %u submits, %llu segments, %u regions, %u textures (%llu read back, %.1f MB; %llu not captured), %llu addresses outside the pools; "
        "%.1f MB (%.1f MB raw) in %.0f ms -> %s%s; live digest %s (%ux%u format %u)",
        g_cap_failed || !ok ? "[RED] " : "", g_cap_frame, g_cap_submit, (unsigned long long)g_cap_segs, g_cap_nreg, g_cap_ntex, (unsigned long long)g_cap_tex_n, g_cap_tex_bytes / 1048576.0,
        (unsigned long long)g_cap_tex_skip, (unsigned long long)g_cap_unmapped, stored / 1048576.0, raw / 1048576.0, p.capture_ms, g_cap_path, ok ? "" : " (write failed)", hex, p.width, p.height, p.pixfmt);
    g_cap_active = 0; g_cap_done = 1;
}
