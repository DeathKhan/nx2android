// nvn_replay_vk: replays a frame captured with DM_NVN_CAPTURE (dm/android/gfx_vk_capture.inc, or the Metal host's nvn_capture.inc) against this tree's Vulkan replay
// layer, on the device, and compares the presented frame with the live one. The counterpart of dm/gfx/tests/nvn_replay.m: hle_nvn.c is included in this translation unit
// (its side tables: program keys, descriptor-pool bases), the rest of libdmhost's objects are linked as they are (build.py --replay).
//   build.py --game <id> --replay                        → <out>/nvn_replay_vk
//   DM_VK_SHADERS=<bundle> nvn_replay_vk <capture> [--warm n] [--out prefix] [--max-pixels n]
// --warm n: n replays in one process, each after the one before (the replay layer keeps its state, the pools go back to the capture's): what a frame-to-frame feedback
// (an exposure, a probe cube) does over n frames. --out prefix: <prefix><n>.ppm per replay (8-bit colour presents) and <prefix><n>.raw (the texels as read back).
// The live run's DM_* settings come from the capture; one set here wins (DM_VK_PASSDUMP and the other diagnostics work as in the app).
// Exit status: 0 the replayed frame equals the live one (within --max-pixels), 1 it differs, 2 the capture cannot be replayed by this build.
#include "../../gfx/nvn/hle_nvn.c"
#include "gfx_vk.h"
#include "dmsh_chain.h"
#include <sys/mman.h>
#include <sys/param.h>

void crash_install(void);
int vk_cap_fill(const tex_t *t, const uint8_t *d, size_t n, char *why, size_t nwhy);
int vk_cap_read_tex(const tex_t *t, cap_buf *out, uint32_t *w, uint32_t *h, uint32_t *fmt);
void vk_cap_sha256(const void *p, size_t n, uint8_t md[32]);

#define RFAIL(...) do { fprintf(stderr, "[RED] nvn_replay_vk: " __VA_ARGS__); fputc('\n', stderr); exit(2); } while (0)
typedef struct { cap_chunk c; uint8_t *d; } rp_chunk;
static rp_chunk *rp_ch; static size_t rp_nch;
typedef struct { uint64_t old_lo, size; uint32_t kind; uint8_t *mem, *orig; } rp_region;
static rp_region *rp_rg; static uint32_t rp_nrg;
static pool_t *rp_po; static uint32_t rp_npo;
static tex_t *rp_to; static uint32_t rp_nto;
static prog_t *rp_pr; static cap_prog *rp_prc; static uint32_t rp_npr;
static dpool_t *rp_dp; static uint32_t rp_ndp;
static sync_t *rp_sy; static uint32_t rp_nsy;
static uint8_t **rp_seg; static uint32_t rp_nseg;
static uint64_t rp_bad;

static int rp_dec_addr(void *ctx, uint64_t *field, uint64_t len, int writable) {
    (void)ctx; (void)writable;
    uint64_t v = *field; if (!cap_is_addr(v)) return 1;
    uint32_t r = cap_addr_region(v); uint64_t off = cap_addr_off(v);
    if (r >= rp_nrg || !rp_rg[r].mem || off > rp_rg[r].size || (len && off + len > rp_rg[r].size && rp_rg[r].kind)) { rp_bad++; *field = 0; return 0; }
    *field = (uint64_t)(uintptr_t)rp_rg[r].mem + off; return 1;
}
static int rp_dec_obj(void *ctx, int kind, uint64_t *field, uint8_t *payload, uint32_t size);
static const cap_ops rp_dec = {rp_dec_addr, rp_dec_obj};
static int rp_dec_obj(void *ctx, int kind, uint64_t *field, uint8_t *payload, uint32_t size) {
    (void)ctx; uint64_t v = *field; if (!v) return 1;
    int k = cap_obj_kind(v); uint64_t i = cap_obj_index(v);
    if (k != kind && !(kind == CAP_K_TEX && k == CAP_K_TEXIN)) { rp_bad++; *field = 0; return 0; }
    void *p = NULL;
    switch (k) {
        case CAP_K_TEX: p = i < rp_nto ? &rp_to[i] : NULL; break;
        case CAP_K_TEXIN: if (payload && i + sizeof(tex_t) <= size) { p = payload + i; cap_tex_fields(p, &rp_dec, NULL); } break;
        case CAP_K_PROG: p = i < rp_npr ? &rp_pr[i] : NULL; break;
        case CAP_K_DPOOL: p = i < rp_ndp ? &rp_dp[i] : NULL; break;
        case CAP_K_SYNC: p = i < rp_nsy ? &rp_sy[i] : NULL; break;
        case CAP_K_POOL: p = i < rp_npo ? &rp_po[i] : NULL; break;
        case CAP_K_SEG: p = i < rp_nseg ? rp_seg[i] : NULL; break;
    }
    if (!p) { rp_bad++; *field = 0; return 0; }
    *field = (uint64_t)(uintptr_t)p; return 1;
}
static const char *rp_head_get(const char *head, size_t n, const char *key, char *out, size_t on) {
    size_t kl = strlen(key);
    for (size_t i = 0; i < n;) {
        const char *e = memchr(head + i, '\n', n - i); size_t ln = e ? (size_t)(e - head - i) : n - i;
        if (ln > kl && !memcmp(head + i, key, kl) && head[i + kl] == '\t') { size_t vl = MIN(ln - kl - 1, on - 1); memcpy(out, head + i + kl + 1, vl); out[vl] = 0; return out; }
        i += ln + 1;
    }
    return NULL;
}
static void rp_apply_env(const char *head, size_t n) {   // the live run's DM_* settings (one given to the replayer wins); not the ones that only concern the app or the submit path
    static const char *const skip[] = {"DM_NVN_SHOT", "DM_NVN_TSV", "DM_KEYS", "DM_CTL", "DM_WINDOW", "DM_NVN_ASYNC", "DM_NVN_LOGICAL_QUEUES", "DM_ROOT", "DM_VK_SHADERS", "DM_SHARED_FONT"};
    for (size_t i = 0; i < n;) {
        const char *e = memchr(head + i, '\n', n - i); size_t ln = e ? (size_t)(e - head - i) : n - i;
        if (ln > 4 && !memcmp(head + i, "env\t", 4)) {
            char kv[4096]; size_t l = MIN(ln - 4, sizeof kv - 1); memcpy(kv, head + i + 4, l); kv[l] = 0;
            char *eq = strchr(kv, '='); if (eq) { *eq = 0; int sk = 0; for (unsigned k = 0; k < sizeof skip / sizeof *skip; k++) sk |= !strncmp(kv, skip[k], strlen(skip[k]));
                if (!sk) setenv(kv, eq + 1, 0); }
        }
        i += ln + 1;
    }
}
static void rp_map_region(uint32_t idx, const cap_regn *h, const uint8_t *bytes) {   // a fresh mapping at the same offset within 64 KiB as the live one
    if (idx >= rp_nrg) { rp_rg = realloc(rp_rg, (idx + 1) * sizeof *rp_rg); memset(rp_rg + rp_nrg, 0, (idx + 1 - rp_nrg) * sizeof *rp_rg); rp_nrg = idx + 1; }
    size_t lead = (size_t)(h->old_lo & 0xffff), len = (size_t)h->size + lead + 0x30000;
    uint8_t *m = mmap(NULL, len, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
    if (m == MAP_FAILED) RFAIL("cannot map region %u (%llu bytes)", idx, (unsigned long long)h->size);
    uint8_t *base = (uint8_t *)(((uintptr_t)m + 0xffff) & ~(uintptr_t)0xffff) + lead;
    memcpy(base, bytes, (size_t)h->size);
    rp_rg[idx] = (rp_region){h->old_lo, h->size, h->kind, base, (uint8_t *)bytes};
}
static int rp_pools_added;
static void rp_add_pools(const uint8_t *d, size_t n) {
    static nvn_hmap seen = {.klen = 8};
    for (size_t o = 0; o + sizeof(cap_padd) <= n; o += sizeof(cap_padd)) {
        cap_padd a; memcpy(&a, d + o, sizeof a);
        if (!rp_dec_addr(NULL, &a.mem, 0, 0) || !a.mem) RFAIL("a pool lies in no captured region");
        if (nvn_hm_get(&seen, &a.mem)) continue; nvn_hm_put(&seen, &a.mem, (void *)1);
        mtl_pool_add((void *)(uintptr_t)a.mem, a.size); mtl_pool_addressed((void *)(uintptr_t)a.mem, a.size); nvn_dirty_add_range((void *)(uintptr_t)a.mem, a.size); rp_pools_added++;
    }
}
// The images of this submit's TPIX chunks: made (and given their first use) inside a submission of their own, as the replay makes them, then filled
static int rp_fill_textures(uint32_t s, int metal) {
    int n = 0;
    for (int pass = 0; pass < 2; pass++) {
        if (pass == 0) mtl_sub_begin();
        for (size_t i = 0; i < rp_nch; i++) {
            cap_chunk *c = &rp_ch[i].c; if (c->submit != s || c->tag != CAP_TPIX) continue;
            cap_tpix h; if (c->raw < sizeof h) RFAIL("short texture chunk");
            memcpy(&h, rp_ch[i].d, sizeof h); cap_tex_fields(&h.t, &rp_dec, NULL);
            if (pass == 0) { vk_tex(&h.t); continue; }
            if (metal) continue;   // Metal pixel formats: the images start empty (the frame's own passes fill most of them; --warm replays fill the rest)
            char why[256];
            if (!vk_cap_fill(&h.t, rp_ch[i].d, c->raw, why, sizeof why)) fprintf(stderr, "[RED] nvn_replay_vk: texture ser=%llu %ux%u: %s; it starts empty\n", (unsigned long long)h.t.serial, h.width, h.height, why);
            else n++;
        }
        if (pass == 0) { mtl_sub_commit(); mtl_finish(); }
    }
    return n;
}
// 8-bit colour of the common present formats, for a picture to look at (the comparison is on the raw texels)
static int rp_ppm(const char *path, const uint8_t *px, uint32_t w, uint32_t h, uint32_t fmt) {
    int bgr = fmt == VK_FORMAT_B8G8R8A8_UNORM || fmt == VK_FORMAT_B8G8R8A8_SRGB, rgba = fmt == VK_FORMAT_R8G8B8A8_UNORM || fmt == VK_FORMAT_R8G8B8A8_SRGB || fmt == VK_FORMAT_A8B8G8R8_UNORM_PACK32
        || fmt == VK_FORMAT_A8B8G8R8_SRGB_PACK32, a2b10 = fmt == VK_FORMAT_A2B10G10R10_UNORM_PACK32;
    if (!bgr && !rgba && !a2b10) return 0;
    FILE *f = fopen(path, "wb"); if (!f) return 0;
    fprintf(f, "P6\n%u %u\n255\n", w, h);
    uint8_t *row = malloc((size_t)w * 3);
    for (uint32_t y = 0; y < h; y++) { for (uint32_t x = 0; x < w; x++) { const uint8_t *p = px + ((size_t)y * w + x) * 4; uint8_t *o = row + (size_t)x * 3;
            if (a2b10) { uint32_t v; memcpy(&v, p, 4); o[0] = (uint8_t)((v & 1023) >> 2); o[1] = (uint8_t)((v >> 10 & 1023) >> 2); o[2] = (uint8_t)((v >> 20 & 1023) >> 2); }
            else if (bgr) { o[0] = p[2]; o[1] = p[1]; o[2] = p[0]; } else { o[0] = p[0]; o[1] = p[1]; o[2] = p[2]; } }
        fwrite(row, 1, (size_t)w * 3, f); }
    free(row); fclose(f); return 1;
}
static void rp_hex(const uint8_t *md, char out[65]) { for (int i = 0; i < 32; i++) snprintf(out + 2 * i, 3, "%02x", md[i]); }

int main(int argc, char **argv) {
    const char *path = NULL, *outp = NULL; int warm = 1; uint64_t maxpx = 0;
    setvbuf(stdout, NULL, _IOLBF, 0); setvbuf(stderr, NULL, _IOLBF, 0); crash_install();
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--warm") && i + 1 < argc) warm = MAX(1, atoi(argv[++i]));
        else if (!strcmp(argv[i], "--out") && i + 1 < argc) outp = argv[++i];
        else if (!strcmp(argv[i], "--max-pixels") && i + 1 < argc) maxpx = strtoull(argv[++i], NULL, 10);
        else if (argv[i][0] != '-' && !path) path = argv[i];
        else { fprintf(stderr, "usage: %s <capture> [--warm n] [--out prefix] [--max-pixels n]\n", argv[0]); return 2; }
    }
    if (!path) { fprintf(stderr, "usage: %s <capture> [--warm n] [--out prefix] [--max-pixels n]\n", argv[0]); return 2; }
    uint64_t tl0 = now_ns();
    nvn_serial_floor(1ull << 61);   // the replay layer's own serials stay clear of the captured ones (nvn_replay.m)
    cap_reader rd; int lr = cap_load(&rd, path);
    if (lr <= 0) RFAIL("%s: %s", path, lr < 0 ? "not a capture of format 1" : "cannot read");
    for (;;) { cap_chunk c; uint8_t *d; int r = cap_next(&rd, &c, &d); if (r < 0) RFAIL("%s: corrupt chunk at byte %zu", path, rd.pos); if (!r) break;
        if (c.tag == CAP_FAIL) RFAIL("%s: capture refused: %.*s", path, (int)MIN(c.raw, 1024), (const char *)d);
        rp_ch = realloc(rp_ch, (rp_nch + 1) * sizeof *rp_ch); rp_ch[rp_nch++] = (rp_chunk){c, d}; }
    cap_unload(&rd);
    char *head = NULL; size_t hn = 0; const char *lay = NULL; size_t ln = 0; const char *stat = ""; size_t sn = 0;
    for (size_t i = 0; i < rp_nch; i++) { if (rp_ch[i].c.tag == CAP_HEAD) { head = (char *)rp_ch[i].d; hn = rp_ch[i].c.raw; } else if (rp_ch[i].c.tag == CAP_LAYT) { lay = (const char *)rp_ch[i].d; ln = rp_ch[i].c.raw; }
        else if (rp_ch[i].c.tag == CAP_STAT) { stat = (const char *)rp_ch[i].d; sn = rp_ch[i].c.raw; } }
    if (!head || !lay) RFAIL("%s: no header", path);
    char title[64] = "", chain[64] = "?", frame[32] = "?", backend[32] = "metal", device[256] = "?", v[64];
    rp_head_get(head, hn, "title", title, sizeof title); rp_head_get(head, hn, "chain", chain, sizeof chain); rp_head_get(head, hn, "frame", frame, sizeof frame);
    rp_head_get(head, hn, "backend", backend, sizeof backend); rp_head_get(head, hn, "device", device, sizeof device);
    uint64_t live_frame = rp_head_get(head, hn, "mtl_frame", v, sizeof v) ? strtoull(v, NULL, 10) : 0;
    int metal = strcmp(backend, "vulkan") != 0;
    if (strcmp(title, DM_GAME_ID)) RFAIL("the capture is of %s, this replayer is built for %s", title, DM_GAME_ID);
    if (!cap_layout_matches(lay, ln)) RFAIL("the capture's record layout differs from this build's (nvn_obj.h changed since the capture): capture again");
    printf("capture %s: %s frame %s, %s (%s), chain %s (this build: chain %s)%s\n", path, title, frame, backend, device, chain, DMSH_CHAIN,
           metal ? "; Metal texture contents are not restored" : "");
    rp_apply_env(head, hn);
    mtl_init();
    if (rp_head_get(stat, sn, "origin_ll", v, sizeof v)) mtl_set_origin_lower_left(atoi(v));
    if (rp_head_get(stat, sn, "depth_zero", v, sizeof v)) mtl_set_depth_zero(atoi(v));
    int qll = rp_head_get(stat, sn, "queue_origin_ll", v, sizeof v) ? atoi(v) : 1;
    uint32_t nsub = 0; const cap_pres *pres = NULL;
#define GROW(n, idx) do { if ((idx) >= (n)) n = (idx) + 1; } while (0)
    for (size_t i = 0; i < rp_nch; i++) {
        cap_chunk *c = &rp_ch[i].c; const uint8_t *d = rp_ch[i].d; uint32_t idx;
        if (c->tag == CAP_REGN) { cap_regn h; if (c->raw < sizeof h) RFAIL("short region chunk"); memcpy(&h, d, sizeof h); if (h.size != c->raw - sizeof h) RFAIL("region %u: size mismatch", h.index); rp_map_region(h.index, &h, d + sizeof h); }
        if (c->tag == CAP_SUBM) nsub = MAX(nsub, c->submit);
        if (c->tag == CAP_PRES && c->raw == sizeof *pres) pres = (const cap_pres *)d;
        if (c->tag == CAP_POBJ) for (size_t o = 0; o + sizeof(cap_pobj) <= c->raw; o += sizeof(cap_pobj)) { memcpy(&idx, d + o, 4); GROW(rp_npo, idx); }
        if (c->tag == CAP_TOBJ) for (size_t o = 0; o + 8 + sizeof(tex_t) <= c->raw; o += 8 + sizeof(tex_t)) { memcpy(&idx, d + o, 4); GROW(rp_nto, idx); }
        if (c->tag == CAP_PROG) for (size_t o = 0; o + sizeof(cap_prog) <= c->raw; o += sizeof(cap_prog)) { memcpy(&idx, d + o, 4); GROW(rp_npr, idx); }
        if (c->tag == CAP_DPOL) for (size_t o = 0; o + sizeof(cap_dpol) <= c->raw; o += sizeof(cap_dpol)) { memcpy(&idx, d + o, 4); GROW(rp_ndp, idx); }
        if (c->tag == CAP_SYNC && c->raw >= 4) { memcpy(&idx, d, 4); rp_nsy = MAX(rp_nsy, idx); }
    }
#undef GROW
    if (!pres) RFAIL("%s: no present (the capture did not finish)", path);
    rp_po = calloc(rp_npo + 1, sizeof *rp_po); rp_to = calloc(rp_nto + 1, sizeof *rp_to); rp_pr = calloc(rp_npr + 1, sizeof *rp_pr); rp_prc = calloc(rp_npr + 1, sizeof *rp_prc);
    rp_dp = calloc(rp_ndp + 1, sizeof *rp_dp); rp_sy = calloc(rp_nsy + 1, sizeof *rp_sy);
    for (size_t i = 0; i < rp_nch; i++) {
        cap_chunk *c = &rp_ch[i].c; const uint8_t *d = rp_ch[i].d; if (c->tag != CAP_POBJ) continue;
        for (size_t o = 0; o + sizeof(cap_pobj) <= c->raw; o += sizeof(cap_pobj)) { cap_pobj p; memcpy(&p, d + o, sizeof p);
            uint64_t m = p.mem; rp_dec_addr(NULL, &m, 0, 0); rp_po[p.index] = (pool_t){MAGIC('P', 'O', 'O', 'L'), p.flags, (uint8_t *)(uintptr_t)m, p.size, NULL, p.vmode}; }
    }
    for (size_t i = 0; i < rp_nch; i++) {
        cap_chunk *c = &rp_ch[i].c; const uint8_t *d = rp_ch[i].d;
        if (c->tag == CAP_TOBJ) for (size_t o = 0; o + 8 + sizeof(tex_t) <= c->raw; o += 8 + sizeof(tex_t)) { uint32_t idx; memcpy(&idx, d + o, 4); memcpy(&rp_to[idx], d + o + 8, sizeof(tex_t)); }
        if (c->tag == CAP_PROG) for (size_t o = 0; o + sizeof(cap_prog) <= c->raw; o += sizeof(cap_prog)) { cap_prog p; memcpy(&p, d + o, sizeof p);
            rp_prc[p.index] = p; prog_t *pr = &rp_pr[p.index]; pr->magic = MAGIC('P', 'R', 'O', 'G'); pr->nstage = p.nstage; pr->serial = p.serial; }
        if (c->tag == CAP_DPOL) for (size_t o = 0; o + sizeof(cap_dpol) <= c->raw; o += sizeof(cap_dpol)) { cap_dpol p; memcpy(&p, d + o, sizeof p);
            uint64_t po = p.pool; rp_dec_obj(NULL, CAP_K_POOL, &po, NULL, 0); rp_dp[p.index] = (dpool_t){(const pool_t *)(uintptr_t)po, p.off, p.n}; }
    }
    for (uint32_t i = 0; i < rp_nto; i++) cap_tex_fields(&rp_to[i], &rp_dec, NULL);
    for (uint32_t i = 0; i < rp_npr; i++) if (rp_pr[i].serial) { prog_keys_t *k = pk_get(rp_pr[i].serial); memcpy(k->key, rp_prc[i].key, sizeof k->key); k->ok = rp_prc[i].ok; }
    for (size_t i = 0; i < rp_nch; i++) if (rp_ch[i].c.tag == CAP_DPOL) for (size_t o = 0; o + sizeof(cap_dpol) <= rp_ch[i].c.raw; o += sizeof(cap_dpol)) { cap_dpol p; memcpy(&p, rp_ch[i].d + o, sizeof p);
        if (g_ntpb < NDPOOL) g_tpb[g_ntpb++] = (struct dpool_base){&rp_dp[p.index], p.tex_base}; if (g_nspb < NDPOOL) g_spb[g_nspb++] = (struct dpool_base){&rp_dp[p.index], p.samp_base}; }
    uint64_t ncpu = 0;
    for (size_t i = 0; i < rp_nch; i++) if (rp_ch[i].c.tag == CAP_TKND) for (size_t o = 0; o + sizeof(cap_tknd) <= rp_ch[i].c.raw; o += sizeof(cap_tknd)) {
        cap_tknd *k = (cap_tknd *)(rp_ch[i].d + o); cap_tex_fields(&k->t, &rp_dec, NULL); if (k->kind & CAP_TK_CPU) { mtl_tex_cpu_written(&k->t); ncpu++; } }
    for (size_t i = 0; i < rp_nch; i++) if (rp_ch[i].c.tag == CAP_TREG) for (size_t o = 0; o + sizeof(cap_treg) <= rp_ch[i].c.raw; o += sizeof(cap_treg)) {
        cap_treg *e = (cap_treg *)(rp_ch[i].d + o); if (e->ok) cap_tex_fields(&e->t, &rp_dec, NULL); }
    for (size_t i = 0; i < rp_nch; i++) if (rp_ch[i].c.tag == CAP_PADD && rp_ch[i].c.submit <= 1) rp_add_pools(rp_ch[i].d, rp_ch[i].c.raw);
    if (rp_bad) RFAIL("%llu captured pointers name no captured object or region", (unsigned long long)rp_bad);
    printf("loaded in %.0f ms: %u regions, %d pools, %u textures, %u programs, %u descriptor pools, %u submits; %llu CPU-written textures marked\n",
           (now_ns() - tl0) / 1e6, rp_nrg, rp_pools_added, rp_nto, rp_npr, rp_ndp, nsub, (unsigned long long)ncpu);

    const uint8_t *lpix = NULL; uint64_t lpixn = 0;
    for (size_t i = 0; i < rp_nch; i++) if (rp_ch[i].c.tag == CAP_LPIX) { lpix = rp_ch[i].d; lpixn = rp_ch[i].c.raw; }
    char live[65]; rp_hex(pres->sha256, live); int ok_all = 1;
    for (int rep = 0; rep < warm; rep++) {
        uint64_t exec_ns = 0, frame_no = rep == 0 && live_frame ? live_frame : 1000000 + (uint64_t)rep; int filled = 0;
        if (rep) for (uint32_t r = 0; r < rp_nrg; r++) if (rp_rg[r].mem) memcpy(rp_rg[r].mem, rp_rg[r].orig, (size_t)rp_rg[r].size);
        for (uint32_t s = 1; s <= nsub; s++) {
            for (size_t i = 0; i < rp_nch; i++) {
                cap_chunk *c = &rp_ch[i].c; uint8_t *d = rp_ch[i].d; if (c->submit != s) continue;
                if (c->tag == CAP_PADD && s > 1) rp_add_pools(d, c->raw);
                else if (c->tag == CAP_PDIF) for (size_t o = 0; o + 24 <= c->raw;) { uint32_t r; uint64_t off, len; memcpy(&r, d + o, 4); memcpy(&off, d + o + 8, 8); memcpy(&len, d + o + 16, 8);
                    if (r >= rp_nrg || off + len > rp_rg[r].size || o + 24 + len > c->raw) RFAIL("a pool change outside its region"); memcpy(rp_rg[r].mem + off, d + o + 24, (size_t)len); o += 24 + (size_t)len; }
                else if (c->tag == CAP_TREG) for (size_t o = 0; o + sizeof(cap_treg) <= c->raw; o += sizeof(cap_treg)) { cap_treg *e = (cap_treg *)(d + o); mtl_reg_tex(e->tid, e->ok && !e->gone ? &e->t : NULL, e->has_v ? &e->v : NULL); }
                else if (c->tag == CAP_SREG) for (size_t o = 0; o + sizeof(cap_sreg) <= c->raw; o += sizeof(cap_sreg)) { cap_sreg e; memcpy(&e, d + o, sizeof e); mtl_reg_samp(e.sid, &e.p); }
            }
            if (rep == 0) filled += rp_fill_textures(s, metal);   // later replays keep what the one before left in them (the warm state)
            uint32_t nh = 0; const uint32_t *top = NULL; rp_nseg = 0;
            for (size_t i = 0; i < rp_nch; i++) { cap_chunk *c = &rp_ch[i].c; if (c->submit != s) continue;
                if (c->tag == CAP_SEGM) { uint32_t idx, len; memcpy(&idx, rp_ch[i].d, 4); memcpy(&len, rp_ch[i].d + 4, 4); if (idx >= rp_nseg) { rp_seg = realloc(rp_seg, (idx + 1) * sizeof *rp_seg); rp_nseg = idx + 1; }
                    rp_seg[idx] = malloc(len + 8); memcpy(rp_seg[idx], rp_ch[i].d + 8, len); memset(rp_seg[idx] + len, 0, 8); }
                if (c->tag == CAP_SUBM) { memcpy(&nh, rp_ch[i].d, 4); top = (const uint32_t *)(rp_ch[i].d + 4); } }
            for (size_t i = 0; i < rp_nch; i++) { cap_chunk *c = &rp_ch[i].c; if (c->submit != s || c->tag != CAP_SEGM) continue;
                uint32_t idx, len; memcpy(&idx, rp_ch[i].d, 4); memcpy(&len, rp_ch[i].d + 4, 4);
                for (uint32_t o = 0; o < len;) { op_hdr h; memcpy(&h, rp_seg[idx] + o, 8); if (h.size < 8 || o + h.size > len) RFAIL("submit %u segment %u: corrupt record at %u", s, idx, o);
                    if (!cap_rec_fields(h.op, rp_seg[idx] + o + 8, h.size, &rp_dec, NULL) || rp_bad) RFAIL("submit %u segment %u: op %u at %u does not decode", s, idx, h.op, o); o += h.size; } }
            uint64_t *hs = malloc((size_t)MAX(nh, 1) * 8); for (uint32_t i = 0; i < nh; i++) { if (top[i] >= rp_nseg) RFAIL("submit %u: handle %u names no segment", s, i); hs[i] = (uint64_t)(uintptr_t)rp_seg[top[i]]; }
            uint64_t t1 = now_ns();
            mtl_set_frame(frame_no);
            mtl_set_queue_origin_lower_left(qll);
            mtl_sub_begin(); nvn_ft_poll(); mtl_sub_index(hs, (int)nh);
            for (uint32_t i = 0; i < nh; i++) { mtl_exec_rest(hs + i + 1, (int)(nh - i - 1)); mtl_exec((const uint8_t *)(uintptr_t)hs[i]); }
            mtl_exec_rest(NULL, 0); mtl_sub_commit();
            exec_ns += now_ns() - t1;
            mtl_finish();
            free(hs); for (uint32_t i = 0; i < rp_nseg; i++) free(rp_seg[i]);
        }
        uint64_t pv = pres->tex; if (!rp_dec_obj(NULL, CAP_K_TEX, &pv, NULL, 0)) RFAIL("the presented texture is not in the capture");
        cap_buf mine = {0}; uint32_t w = 0, h = 0, pf = 0;
        if (!vk_cap_read_tex((const tex_t *)(uintptr_t)pv, &mine, &w, &h, &pf)) RFAIL("the presented texture cannot be read back");
        uint8_t md[32]; vk_cap_sha256(mine.p, mine.n, md); char out[65]; rp_hex(md, out); int same = !memcmp(md, pres->sha256, 32);
        uint64_t npx = 0; uint32_t big = 0, x0 = UINT32_MAX, x1 = 0, y0 = UINT32_MAX, y1 = 0;
        int have_px = !same && lpix && lpixn == mine.n && w == pres->width && h == pres->height && pf == pres->pixfmt && w && h;
        if (have_px) { size_t bpp = (size_t)(mine.n / ((uint64_t)w * h));
            for (uint32_t y = 0; y < h; y++) for (uint32_t x = 0; x < w; x++) { const uint8_t *a = lpix + ((size_t)y * w + x) * bpp, *b = mine.p + ((size_t)y * w + x) * bpp; uint32_t dd = 0;
                for (size_t k = 0; k < bpp; k++) dd = MAX(dd, (uint32_t)abs((int)a[k] - (int)b[k]));
                if (dd) { npx++; big = MAX(big, dd); x0 = MIN(x0, x); x1 = MAX(x1, x); y0 = MIN(y0, y); y1 = MAX(y1, y); } } }
        int ok = same || (have_px && npx <= maxpx); ok_all &= ok;
        if (outp) { char p[1100]; snprintf(p, sizeof p, "%s%d.raw", outp, rep + 1); FILE *f = fopen(p, "wb"); if (f) { fwrite(mine.p, 1, mine.n, f); fclose(f); }
            snprintf(p, sizeof p, "%s%d.ppm", outp, rep + 1); rp_ppm(p, mine.p, w, h, pf);
            if (rep == 0 && lpix && lpixn == (uint64_t)w * h * 4) { snprintf(p, sizeof p, "%slive.ppm", outp); rp_ppm(p, lpix, pres->width, pres->height, pres->pixfmt); } }
        cap_buf_free(&mine);
        double gpu = mtl_gpu_ms(frame_no);
        if (rep == 0) printf("textures filled from the capture: %d\n", filled);
        if (have_px) printf("replay %d: %llu pixels differ from the live frame (largest byte difference %u, x %u-%u y %u-%u)\n", rep + 1, (unsigned long long)npx, big, x0, x1, y0, y1);
        printf("replay %d: digest %s %s live %s (%ux%u format %u); submit CPU %.2f ms, GPU %.2f ms\n", rep + 1, out, same ? "==" : "!=", live, w, h, pf, exec_ns / 1e6, gpu);
    }
    if (ok_all) printf("MATCH%s\n", maxpx ? " (within --max-pixels)" : "");
    else printf("[RED] MISMATCH: the replayed frame differs from the live frame\n");
    return ok_all ? 0 : 1;
}
