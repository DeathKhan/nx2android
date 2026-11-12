// Android port, Vulkan backend: the game's shaders (see gfx_vk.h).
// The bundle (written by dm/android/tools/shader_vk.py; its format in dm/android/tools/dmsv.py) is mapped read-only: entries sorted by key, each the SPIR-V of one Switch
// shader and its binary reflection. Version 2 stores each SPIR-V module as one zstd frame (dm/third_party/zstd), decompressed when the shader is first made (bundle_spv).
// The translator numbers every stage's bindings from 0 in set 0; at module creation each stage's bindings move to a set of its own (vertex 0, fragment 1,
// geometry 2, tessellation 3 / 4; compute 0) by rewriting the DescriptorSet decorations, so a program's stages never collide.
#include "gfx_vk.h"
#include "gfx_vk_ubo_extent.h"
#include "gfx_vk_gs_claim.h"
#include "dmsh_abi.h"
#include "nvn_hmap.h"
#include "zstd.h"
#include <CommonCrypto/CommonDigest.h>
#include <fcntl.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

// spv_size: 0 = the SPIR-V is stored as it is (spv_len bytes; every version 1 entry), else a zstd frame of spv_len bytes that decompresses to spv_size (version 2)
typedef struct __attribute__((packed)) { uint64_t key; uint32_t stage, spv_off, spv_len, refl_off, refl_len, spv_size; } bent;
static const uint8_t *g_bundle; static uint32_t g_nent; static const bent *g_ent;
// Set 0 is shared by every pipeline, including shaders loaded later. Its pool bindings therefore
// use the union of the entire bundle, never the first program or the currently active stages.
// Unknown reflection keeps the old full layout; an absent kind retains one compatible descriptor.
static unsigned g_pool_kinds = (1u << POOL_TEX_SLOTS) - 1;
static int g_pool_samp_seen;   // a shader of the bundle reads the sampler pool (bundle_pool_kinds)
static unsigned bundle_pool_kinds(const uint8_t *b, uint64_t size) {
    const unsigned all = (1u << POOL_TEX_SLOTS) - 1; uint32_t version, count;
    g_pool_samp_seen = 0;
    if (!b || size < 12 || memcmp(b, "DMSV", 4)) return all;
    memcpy(&version, b + 4, 4); memcpy(&count, b + 8, 4);
    if (version < 1 || version > 2 || !count || 12 + (uint64_t)count * sizeof(bent) > size) return all;
    unsigned kinds = 0;
    for (uint32_t i = 0; i < count; i++) {
        bent e; memcpy(&e, b + 12 + (uint64_t)i * sizeof(e), sizeof(e));
        if (e.stage > 7 || (uint64_t)e.spv_off + e.spv_len > size || e.spv_len < 20 ||
            (uint64_t)e.refl_off + e.refl_len > size || e.refl_len < 20) return all;
        uint32_t h[5]; memcpy(h, b + e.refl_off, sizeof(h));
        if (20 + (uint64_t)h[0] * sizeof(dmsv_bind) + h[1] > e.refl_len) return all;
        for (uint32_t j = 0; j < h[0]; j++) {
            dmsv_bind r; memcpy(&r, b + e.refl_off + 20 + (uint64_t)j * sizeof(r), sizeof(r));
            if (r.kind > RK_SAMPP || r.desc > RD_SAMPLER) return all;
            if (r.kind == RK_SAMPP) { if (r.desc != RD_SAMPLER || r.flags || !r.count) return all; g_pool_samp_seen = 1; continue; }
            if (r.kind != RK_TEXP) continue;
            if (r.desc != RD_IMAGE || (r.flags & ~1u) || !r.count) return all;
            int slot = r.dim == 1 ? (r.flags & 1) : r.dim == 3 ? 2 : r.dim == 2 ? 3 : -1;
            if (slot < 0) return all;
            kinds |= 1u << slot;
        }
    }
    return kinds;
}

// Every index record is checked once here (upstream nx2android #173: keys ascending, payloads after the index and inside the file, version 1 never compressed), so a lookup never
// reads past the mapping (a copy cut short, say a push stopped outside porter, would fault there); payloads and reflection counts are checked when their shader is requested
void vk_shaders_open(const char *path) {
    int fd = open(path, O_RDONLY | O_CLOEXEC); struct stat st;
    if (fd < 0 || fstat(fd, &st)) { if (getenv("DM_GL_ACTIVE")) VKLOG("no shader bundle at %s: an OpenGL ES game (gl_android.c) needs none", path);
                                    else VKLOG("[RED] no shader bundle at %s: nothing will be drawn", path);
        if (fd >= 0) close(fd);
        return; }
    const uint8_t *b = st.st_size >= 12 ? mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_SHARED, fd, 0) : MAP_FAILED; close(fd);
    if (b == MAP_FAILED || memcmp(b, "DMSV", 4)) { VKLOG("[RED] %s is not a shader bundle", path); if (b != MAP_FAILED) munmap((void *)b, (size_t)st.st_size); return; }
    uint32_t ver, n, nz = 0, bad = 0; memcpy(&ver, b + 4, 4); memcpy(&n, b + 8, 4); const bent *t = (const bent *)(b + 12); uint64_t sz = (uint64_t)st.st_size, raw = 0, stored = 0;
    if (ver < 1 || ver > 2) { VKLOG("[RED] %s: shader bundle version %u, this host reads 1 and 2: nothing will be drawn", path, ver); munmap((void *)b, (size_t)sz); return; }
    uint64_t table_end = 12 + 32ull * n;
    if (table_end > sz) bad = 1;
    else for (uint32_t i = 0; i < n; i++) { const bent *e = &t[i];
        if (e->stage > 7 || (i && e->key <= t[i - 1].key) || (ver == 1 && e->spv_size) ||
            e->spv_off < table_end || (uint64_t)e->spv_off + e->spv_len > sz ||
            e->refl_off < table_end || (uint64_t)e->refl_off + e->refl_len > sz || e->refl_len < 20) { bad++; continue; }
        raw += e->spv_size ? e->spv_size : e->spv_len; stored += e->spv_len; nz += e->spv_size != 0; }
    if (bad) { VKLOG("[RED] %s: invalid shader index (%u bad records, %u shaders, %llu bytes): nothing will be drawn", path, bad, n, (unsigned long long)sz); munmap((void *)b, (size_t)sz); return; }
    g_pool_kinds = bundle_pool_kinds(b, sz);
    g_bundle = b; g_nent = n; g_ent = t;
    VKLOG("shader bundle %s: version %u, %u shaders (%u zstd-compressed), %.1f MB; SPIR-V %.1f MB stored, %.1f MB decompressed", path, ver, n, nz, sz / 1048576.0, stored / 1048576.0, raw / 1048576.0);
}
static void bundle_open_once(void) {
    const char *p = getenv("DM_VK_SHADERS"); char buf[1024];
    if (!p) { const char *r = getenv("DM_ROOT"); snprintf(buf, sizeof buf, "%s/shaders.dmsv", r ? r : "."); p = buf; }
    vk_shaders_open(p);
}
// once, and every caller waits until it is open: the replay and the prewarm threads both look shaders up at start, and a lookup that ran before the bundle
// was open cached its shader as missing for good (every draw skipped as "no program" in some runs)
static void bundle_auto(void) { static pthread_once_t once = PTHREAD_ONCE_INIT; pthread_once(&once, bundle_open_once); }
int vk_shader_keys(uint32_t stage, uint64_t *out, int max) {   // the bundle's shaders of one stage (gfx_vk_exec.c compute_warm)
    bundle_auto(); int n = 0;
    for (uint32_t i = 0; i < g_nent && n < max; i++) { uint32_t st; memcpy(&st, &g_ent[i].stage, 4); if (st == stage) memcpy(&out[n++], &g_ent[i].key, 8); }
    return n;
}
static const bent *find(uint64_t key) {
    uint32_t lo = 0, hi = g_nent;
    while (lo < hi) { uint32_t m = (lo + hi) / 2; uint64_t k; memcpy(&k, &g_ent[m].key, 8); if (k < key) lo = m + 1; else hi = m; }
    if (lo < g_nent) { uint64_t k; memcpy(&k, &g_ent[lo].key, 8); if (k == key) return &g_ent[lo]; }
    return NULL;
}
int vk_shader_stage(uint64_t key) { bundle_auto(); const bent *e = g_bundle ? find(key) : NULL; return e ? (int)e->stage : -1; }   // the NVN stage the bundle has this shader as
// A shader's SPIR-V as the host's own copy (patched at module creation, then kept for the variants: vkshader_t spv), copied out of the mapping or decompressed from its zstd
// frame. Called outside g_sh_mu: the replay thread looks its shaders up under that lock while the prewarm threads make theirs. Decompression contexts (94 KB each) are kept
// for the next shader, one per thread decompressing at once. NULL, logged, when the module is not SPIR-V of its entry's length
static ZSTD_DCtx *g_dctx[8]; static int g_ndctx; static pthread_mutex_t g_dctx_mu = PTHREAD_MUTEX_INITIALIZER;
static uint64_t g_unz_n, g_unz_bytes, g_unz_ns;   // modules decompressed, their bytes, the time it took (the log line at 64, 128, 256 ...)
static uint32_t *bundle_spv(const bent *e, uint32_t *nw) {
    uint32_t n = e->spv_size ? e->spv_size : e->spv_len, *w = n >= 20 && !(n & 3) ? malloc(n) : NULL; size_t r = n; char why[96] = "";
    if (w && !e->spv_size) memcpy(w, g_bundle + e->spv_off, n);
    else if (w) {
        struct timespec a, b; clock_gettime(CLOCK_MONOTONIC, &a);
        ZSTD_DCtx *dc = NULL; pthread_mutex_lock(&g_dctx_mu); if (g_ndctx) dc = g_dctx[--g_ndctx]; pthread_mutex_unlock(&g_dctx_mu);
        if (!dc) dc = ZSTD_createDCtx();
        r = dc ? ZSTD_decompressDCtx(dc, w, n, g_bundle + e->spv_off, e->spv_len) : 0;   // checks the frame's checksum too
        pthread_mutex_lock(&g_dctx_mu); if (dc && g_ndctx < 8) { g_dctx[g_ndctx++] = dc; dc = NULL; } pthread_mutex_unlock(&g_dctx_mu); ZSTD_freeDCtx(dc);
        clock_gettime(CLOCK_MONOTONIC, &b);
        uint64_t k = __atomic_add_fetch(&g_unz_n, 1, __ATOMIC_RELAXED), by = __atomic_add_fetch(&g_unz_bytes, n, __ATOMIC_RELAXED),
                 ns = __atomic_add_fetch(&g_unz_ns, (uint64_t)((b.tv_sec - a.tv_sec) * 1000000000ll + (b.tv_nsec - a.tv_nsec)), __ATOMIC_RELAXED);
        if (k >= 64 && !(k & (k - 1))) VKLOG("shaders: %llu decompressed so far, %.1f MB of SPIR-V in %.1f ms", (unsigned long long)k, by / 1048576.0, ns / 1e6);
    }
    if (!w) snprintf(why, sizeof why, "%u bytes of SPIR-V in its entry, not a module", n);
    else if (ZSTD_isError(r)) snprintf(why, sizeof why, "its SPIR-V does not decompress (zstd error %d, zstd_errors.h)", (int)ZSTD_getErrorCode(r));
    else if (r != n) snprintf(why, sizeof why, "its SPIR-V decompresses to %zu bytes, its entry says %u", r, n);
    else if (w[0] != 0x07230203) snprintf(why, sizeof why, "not SPIR-V (first word %#x)", w[0]);
    else if (w[4]) snprintf(why, sizeof why, "invalid SPIR-V schema %u", w[4]);
    else if (w[1] < 0x10000 || w[1] > 0x10300 || (w[1] & 0xff)) snprintf(why, sizeof why, "unsupported SPIR-V version %#x", w[1]);
    // Our translator emits SPIR-V 1.3 for Vulkan 1.1; check framing before any
    // runtime patcher walks instructions. This is not semantic spirv-val.
    if (!*why) for (uint32_t at = 5; at < n / 4;) {
        uint32_t count = w[at] >> 16;
        if (!count || count > n / 4 - at) { snprintf(why, sizeof why, "invalid SPIR-V instruction at word %u", at); break; }
        at += count;
    }
    if (*why) { VKLOG("[RED] shader %016llx: %s: left out as if missing", (unsigned long long)e->key, why); free(w); return NULL; }
    *nw = n / 4; return w;
}

// Run-time capture: a shader a draw needs that the bundle does not have is saved as DM_ROOT/captured_shaders/<key>.bin ("DMSC", u32 control length, u32 code length, the control
// section as its key hashed it, the code area), and the porter translates the folder into the bundle at the next port (porter/pipeline.py, shader_vk.py --capture).
// Xenoblade Chronicles 3 sets hundreds of programs whose code is in no game file (GLSLC outputs of its Firehawk effects, and vertex / pixel shaders of models: a machine near
// the camera in Everlight Plain drew as a black silhouette, its depth pre-pass drawn and its G-buffer draw skipped); 51 of them in the opening and the first field.
// The game may free a control section once SetShaders returns, so SetShaders keeps a copy of each one the bundle lacks (shader_set); the file is written only when a draw misses
// the key (vk_shader_missing) and the code area then still hashes to it
typedef struct { uint64_t key, data; uint32_t nctl, ncode; uint8_t *ctl; } pend_t;
enum { NPEND = 4096 };
static pend_t g_pend[NPEND]; static unsigned g_npend; static pthread_mutex_t g_pend_mu = PTHREAD_MUTEX_INITIALIZER;
static void shader_set(const uint8_t k[8], const void *control, uint64_t data, uint32_t ncode) {
    uint64_t key = 0; for (int i = 0; i < 8; i++) key = key << 8 | k[i];
    if (!key || !control || !data || !ncode || ncode >= (1u << 24) || vk_shader_stage(key) >= 0) return;
    uint32_t nctl = 0x880, own; memcpy(&own, (const uint8_t *)control + 0x20, 4); if (own >= 0x760 && own < 0x880) nctl = own;   // as hle_nvn.c keys_hash
    uint8_t *c = malloc(nctl); if (!c) return; memcpy(c, control, nctl);
    pthread_mutex_lock(&g_pend_mu);
    pend_t *p = &g_pend[g_npend++ % NPEND]; free(p->ctl); *p = (pend_t){key, data, nctl, ncode, c};   // the oldest goes: a draw needs a program soon after it is set
    pthread_mutex_unlock(&g_pend_mu);
}
__attribute__((constructor)) static void shader_set_on(void) { extern void (*g_nvn_on_shader)(const uint8_t *, const void *, uint64_t, uint32_t); g_nvn_on_shader = shader_set; }
void vk_shader_missing(uint64_t key) {
    pend_t p = {0};
    pthread_mutex_lock(&g_pend_mu);
    for (unsigned i = 0; i < NPEND && i < g_npend; i++) if (g_pend[i].key == key && g_pend[i].ctl) { p = g_pend[i]; p.ctl = malloc(p.nctl); if (p.ctl) memcpy(p.ctl, g_pend[i].ctl, p.nctl); break; }
    pthread_mutex_unlock(&g_pend_mu);
    if (!p.ctl) { VKLOG("[RED] shader %016llx is not in the bundle (and was not set while the capture could see it)", (unsigned long long)key); return; }
    uint8_t md[CC_SHA256_DIGEST_LENGTH]; CC_SHA256_CTX hc; CC_SHA256_Init(&hc); CC_SHA256_Update(&hc, p.ctl, p.nctl); CC_SHA256_Update(&hc, (const void *)(uintptr_t)p.data, p.ncode); CC_SHA256_Final(md, &hc);
    uint64_t now = 0; for (int i = 0; i < 8; i++) now = now << 8 | md[i];
    if (now != key) { VKLOG("[RED] shader %016llx is not in the bundle; its code area changed since SetShaders, not saved", (unsigned long long)key); free(p.ctl); return; }
    const char *r = getenv("DM_ROOT"); char dir[1024], path[1100], tmp[1110]; snprintf(dir, sizeof dir, "%s/captured_shaders", r ? r : "."); mkdir(dir, 0755);
    snprintf(path, sizeof path, "%s/%016llx.bin", dir, (unsigned long long)key); snprintf(tmp, sizeof tmp, "%s.part", path);
    struct stat st; if (!stat(path, &st)) { VKLOG("[RED] shader %016llx is not in the bundle: captured in %s by an earlier run, the next port translates it", (unsigned long long)key, path); free(p.ctl); return; }
    FILE *f = fopen(tmp, "wb"); uint32_t hdr[3] = {0x43534d44, p.nctl, p.ncode};   // "DMSC"
    int ok = f && fwrite(hdr, 1, 12, f) == 12 && fwrite(p.ctl, 1, p.nctl, f) == p.nctl && fwrite((const void *)(uintptr_t)p.data, 1, p.ncode, f) == p.ncode;
    if (f && fclose(f)) ok = 0;
    if (!ok || rename(tmp, path)) { VKLOG("[RED] shader %016llx is not in the bundle, and %s could not be written", (unsigned long long)key, path); unlink(tmp); }
    else VKLOG("[RED] shader %016llx is not in the bundle: saved to %s, the next port translates it", (unsigned long long)key, path);
    free(p.ctl);
}
static VkShaderStageFlags stage_flag(int st) {
    switch (st) { case 0: return VK_SHADER_STAGE_VERTEX_BIT; case 1: return VK_SHADER_STAGE_FRAGMENT_BIT; case 2: return VK_SHADER_STAGE_GEOMETRY_BIT;
                  case 3: return VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT; case 4: return VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT; default: return VK_SHADER_STAGE_COMPUTE_BIT; }
}
static int set_of(int st) { return st >= 5 ? 0 : st + 1; }   // graphics stages after the texture table's set 0 (vk_tbl_dsl); 5 compute, 6 / 7: NVN2 task and mesh kernels, dispatched as compute (gfx_vk_exec.c mesh_draw)
static VkDescriptorType desc_type(int d) {
    switch (d) { case RD_UBO: return VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER; case RD_SSBO: return VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; case RD_SAMPLED: return VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                 case RD_STORAGE: return VK_DESCRIPTOR_TYPE_STORAGE_IMAGE; case RD_TBUF: return VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER; case RD_STORAGE_TBUF: return VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER;
                 case RD_IMAGE: return VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE; default: return VK_DESCRIPTOR_TYPE_SAMPLER; }
}
// Stage interfaces. Vulkan requires every input location of a stage to be declared as an output of the stage before it (VUID-RuntimeSpirv-OpEntryPoint-08743);
// the translator writes each stage apart, so a stage may read locations the one before never declares. Xenoblade Chronicles 3's passthrough geometry shaders
// (written out as ordinary ones, forwarding all 16 attributes) follow vertex shaders with two outputs, and Adreno refused those pipelines (VK_ERROR_UNKNOWN).
// The earlier stage then gets a variant that declares the missing vec4 outputs and leaves them unwritten; a vertex shader's variant then stores (0, 0, 0, 1) to them
// as to its own outputs (spv_vs_init), the values Maxwell gives unwritten outputs, where Vulkan leaves them undefined.
static int spv_is_vec4(const uint32_t *w, uint32_t fn, uint32_t t) {   // a vec4 of 32-bit floats, or an array of them (per-vertex inputs)
    for (int depth = 0; depth < 3; depth++) { uint32_t i = 5, found = 0;
        while (i < fn) { uint32_t n = w[i] >> 16; if (!n) return 0; if (w[i + 1] == t && ((w[i] & 0xffff) == 23 || (w[i] & 0xffff) == 28 || (w[i] & 0xffff) == 32)) { found = i; break; } i += n; }
        if (!found) return 0; uint32_t op = w[found] & 0xffff;
        if (op == 32) { t = w[found + 3]; continue; }   // OpTypePointer
        if (op == 28) { t = w[found + 2]; continue; }   // OpTypeArray
        if (w[found + 3] != 4) return 0; for (i = 5; i < fn;) { if ((w[i] & 0xffff) == 22 && w[i + 1] == w[found + 2]) return w[i + 2] == 32; i += w[i] >> 16; } return 0; }
    return 0;
}
static void spv_io_locs(const uint32_t *w, uint32_t nw, uint32_t *in, uint32_t *out) {
    *in = *out = 0; uint32_t fn = nw;
    for (uint32_t i = 5; i < nw;) { uint32_t n = w[i] >> 16; if (!n) return; if ((w[i] & 0xffff) == 54) { fn = i; break; } i += n; }
    for (uint32_t i = 5; i < fn;) { uint32_t n = w[i] >> 16;
        if ((w[i] & 0xffff) == 71 && n == 4 && w[i + 2] == 30 && w[i + 3] < 32) {   // OpDecorate Location
            uint32_t v = w[i + 1];
            for (uint32_t j = 5; j < fn;) { uint32_t m = w[j] >> 16; if ((w[j] & 0xffff) == 59 && w[j + 2] == v) {
                    if (spv_is_vec4(w, fn, w[j + 1])) { if (w[j + 3] == 1) *in |= 1u << w[i + 3]; else if (w[j + 3] == 3) *out |= 1u << w[i + 3]; }
                    else if (w[j + 3] == 3) *out |= 1u << w[i + 3];   // an output of another type still takes its location
                    break; }
                j += m; } }
        i += n; }
}
// A vertex shader's outputs start as (0, 0, 0, 1), as on Maxwell: the translator stores only the components the shader writes and Vulkan leaves the rest
// undefined (dmsh_build.py vs_init does the same for the Apple chain, found on Link's Awakening's lighting). Pokémon Legends: Arceus's deferred lighting draws
// with vertex shader 1327af7c2b78070e, which writes out0.xy only, and its pixel shaders divide by in0.w: every lit pixel was wrong (a washed-out picture,
// dark hair and cloth turned white). Each vec4 output with a Location gets that value stored at the entry point's start. NULL when the module is not as expected
static uint32_t *spv_vs_init(const uint32_t *w, uint32_t nw, uint32_t *onw) {
    uint32_t bound = w[3], fn = 0, ep = 0, f32 = 0, v4 = 0, c0 = 0, c1 = 0; uint8_t *ptr = calloc(bound, 1), *loc = calloc(bound, 1); uint32_t vars[32], nv = 0;
    for (uint32_t i = 5; i < nw;) { uint32_t n = w[i] >> 16, op = w[i] & 0xffff; if (!n || i + n > nw) goto fail;
        if (op == 54) { fn = i; break; }
        if (op == 15 && n >= 3 && w[i + 1] == 0) ep = w[i + 2];   // OpEntryPoint Vertex
        if (op == 22 && n == 3 && w[i + 2] == 32) f32 = w[i + 1];
        if (op == 23 && n == 4 && w[i + 2] == f32 && w[i + 3] == 4 && f32) v4 = w[i + 1];
        if (op == 32 && n == 4 && w[i + 2] == 3 && v4 && w[i + 3] == v4 && w[i + 1] < bound) ptr[w[i + 1]] = 1;   // OpTypePointer Output vec4
        if (op == 43 && n == 4 && w[i + 1] == f32 && f32) { if (w[i + 3] == 0) c0 = w[i + 2]; else if (w[i + 3] == 0x3f800000) c1 = w[i + 2]; }
        if (op == 71 && n == 4 && w[i + 2] == 30 && w[i + 1] < bound) loc[w[i + 1]] = 1;   // Location
        i += n; }
    if (!fn || !ep || !v4) goto fail;
    for (uint32_t i = 5; i < fn;) { uint32_t n = w[i] >> 16; if ((w[i] & 0xffff) == 59 && n >= 4 && w[i + 3] == 3 && w[i + 1] < bound && ptr[w[i + 1]] && w[i + 2] < bound && loc[w[i + 2]] && nv < 32) vars[nv++] = w[i + 2]; i += n; }
    if (!nv) goto fail;
    uint32_t at = 0;   // the entry point's first block, after its OpVariables
    for (uint32_t i = fn; i < nw;) { uint32_t n = w[i] >> 16, op = w[i] & 0xffff; if (!n) goto fail;
        if (op == 54 && w[i + 2] == ep) { uint32_t j = i + n; while (j < nw && (w[j] & 0xffff) != 248) j += w[j] >> 16; j += 2;
            while (j < nw && ((w[j] & 0xffff) == 59 || (w[j] & 0xffff) == 8 || (w[j] & 0xffff) == 317)) j += w[j] >> 16; at = j; break; }
        i += n; }
    if (!at || at >= nw) goto fail;
    uint32_t dec[16], nd = 0, id = bound;
    if (!c0) { c0 = id++; dec[nd++] = 4u << 16 | 43; dec[nd++] = f32; dec[nd++] = c0; dec[nd++] = 0; }
    if (!c1) { c1 = id++; dec[nd++] = 4u << 16 | 43; dec[nd++] = f32; dec[nd++] = c1; dec[nd++] = 0x3f800000; }
    uint32_t cv = id++; dec[nd++] = 7u << 16 | 44; dec[nd++] = v4; dec[nd++] = cv; dec[nd++] = c0; dec[nd++] = c0; dec[nd++] = c0; dec[nd++] = c1;   // OpConstantComposite
    uint32_t *o = malloc(((size_t)nw + nd + 3 * nv) * 4), no = 0;
    memcpy(o, w, (size_t)fn * 4); no = fn; o[3] = id;
    memcpy(&o[no], dec, nd * 4); no += nd;
    memcpy(&o[no], &w[fn], (size_t)(at - fn) * 4); no += at - fn;
    for (uint32_t k = 0; k < nv; k++) { o[no++] = 3u << 16 | 62; o[no++] = vars[k]; o[no++] = cv; }   // OpStore
    memcpy(&o[no], &w[at], (size_t)(nw - at) * 4); no += nw - at;
    free(ptr); free(loc); *onw = no; return o;
fail:
    free(ptr); free(loc); return NULL;
}
static uint32_t *spv_add_outputs(const uint32_t *w, uint32_t nw, uint32_t mask, uint32_t *onw) {
    uint32_t fn = nw, ep = 0, lastdec = 0, bound = w[3], f32 = 0, v4 = 0, pv4 = 0;
    for (uint32_t i = 5; i < nw;) { uint32_t n = w[i] >> 16, op = w[i] & 0xffff; if (!n) return NULL; if (op == 54) { fn = i; break; }
        if (op == 15 && !ep) ep = i; if (op == 71 || op == 72) lastdec = i + n;
        if (op == 22 && w[i + 2] == 32) f32 = w[i + 1];
        i += n; }
    if (!ep || !lastdec || !f32) return NULL;
    for (uint32_t i = 5; i < fn;) { if ((w[i] & 0xffff) == 23 && w[i + 2] == f32 && w[i + 3] == 4) v4 = w[i + 1]; i += w[i] >> 16; }
    if (v4) for (uint32_t i = 5; i < fn;) { if ((w[i] & 0xffff) == 32 && w[i + 2] == 3 && w[i + 3] == v4) pv4 = w[i + 1]; i += w[i] >> 16; }
    int k = __builtin_popcount(mask); uint32_t ids[32], nid = 0;
    uint32_t *o = malloc(((size_t)nw + 16 * k + 48) * 4), no = 5; memcpy(o, w, 20);
    for (int l = 0; l < 32; l++) if (mask >> l & 1) ids[nid++] = bound++;
    uint32_t tv4 = v4 ? v4 : bound++, tpv4 = pv4 ? pv4 : bound++, c0 = bound++, c1 = bound++, cv = bound++;   // (0, 0, 0, 1): what an input no stage wrote reads on the console
    for (uint32_t i = 5; i < fn;) { uint32_t n = w[i] >> 16;
        if (i == ep) { o[no++] = (n + nid) << 16 | 15; memcpy(&o[no], &w[i + 1], (n - 1) * 4); no += n - 1; memcpy(&o[no], ids, nid * 4); no += nid; }   // the entry point lists them
        else { memcpy(&o[no], &w[i], n * 4); no += n; }
        i += n;
        if (i == lastdec) { uint32_t q = 0; for (int l = 0; l < 32; l++) if (mask >> l & 1) { o[no++] = 4u << 16 | 71; o[no++] = ids[q++]; o[no++] = 30; o[no++] = (uint32_t)l; } }   // Location
    }
    if (!v4) { o[no++] = 4u << 16 | 23; o[no++] = tv4; o[no++] = f32; o[no++] = 4; }
    if (!pv4) { o[no++] = 4u << 16 | 32; o[no++] = tpv4; o[no++] = 3; o[no++] = tv4; }
    o[no++] = 4u << 16 | 43; o[no++] = f32; o[no++] = c0; o[no++] = 0; o[no++] = 4u << 16 | 43; o[no++] = f32; o[no++] = c1; o[no++] = 0x3f800000;   // OpConstant 0.0, 1.0
    o[no++] = 7u << 16 | 44; o[no++] = tv4; o[no++] = cv; o[no++] = c0; o[no++] = c0; o[no++] = c0; o[no++] = c1;   // OpConstantComposite (0, 0, 0, 1)
    // OpVariable Output with that initializer. Declared only, the outputs were undefined, and Adreno handed the fragment shader whatever its registers held: Ocarina of Time's
    // mesh-drawn room geometry (the pass-through vertex shaders of shader_meshwrap.py) reads locations its mesh shaders never write, and lit through them the probe cube and the
    // scene came out thousands of times too bright. Upstream pads such inputs to (0, 0, 0, 1) for the same reason (docs/reference/shaders.md, pixel shader inputs the mesh shader does not write)
    for (uint32_t q = 0; q < nid; q++) { o[no++] = 5u << 16 | 59; o[no++] = tpv4; o[no++] = ids[q]; o[no++] = 3; o[no++] = cv; }
    memcpy(&o[no], &w[fn], (nw - fn) * 4); no += nw - fn;
    o[3] = bound; *onw = no; return o;
}
static nvn_hmap g_shaders = {.klen = 16};   // {key, stage} → vkshader_t*
static pthread_mutex_t g_sh_mu = PTHREAD_MUTEX_INITIALIZER;
// Shared layouts keep pipeline layouts and descriptor bindings reusable across shaders; g_sh_mu protects this cache too.
static VkDescriptorSetLayout shader_layout(const VkDescriptorSetLayoutBinding *b, uint32_t n) {
    struct { uint32_t n, b[96][4]; } k = {.n = n};
    static nvn_hmap cache = {.klen = sizeof k};
    for (uint32_t i = 0; i < n; i++) {
        uint32_t j = i;
        while (j && k.b[j - 1][0] > b[i].binding) { memcpy(k.b[j], k.b[j - 1], sizeof k.b[j]); j--; }
        k.b[j][0] = b[i].binding; k.b[j][1] = b[i].descriptorType; k.b[j][2] = b[i].descriptorCount; k.b[j][3] = b[i].stageFlags;
    }
    VkDescriptorSetLayout layout = nvn_hm_get(&cache, &k);
    if (!layout) {
        VkDescriptorSetLayoutCreateInfo li = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, NULL, 0, n, b};
        VKCHK(vkCreateDescriptorSetLayout(V.dev, &li, NULL, &layout));
        void vk_dsl_register(VkDescriptorSetLayout l, const VkDescriptorSetLayoutBinding *b, uint32_t n); vk_dsl_register(layout, b, n);
        nvn_hm_put(&cache, &k, layout);
    }
    return layout;
}
int vk_tbl_n(void) {   // the texture tables' length (gfx_vk.h): DM_VK_TBL_N, 2048 unless set, 0 off. Fixed for the process: every graphics pipeline layout shares the one layout.
    // Welcome Tour binds its material texture list (cb12) with 15,872 bytes, 1,984 handles; tbl_set logs the longest bound, and an index past the length reads the last entry (the clamp)
    static int n = -1; if (n < 0) { const char *e = getenv("DM_VK_TBL_N"); n = e ? atoi(e) : 2048; if (n < 0) n = 0; if (n > 65536) n = 65536; }
    return n;
}
// The texture pool read marks (DM_VK_POOL_MARK=1, start-up, off unless set; gfx_vk_exec.c pool marks, gfx_vk.h POOL_MARK_B). Fixed for the process, as set 0 is: the layout has
// the marks' binding only with it, and only with it are the shaders that index the texture pool given their stores (spv_pool_mark). The device must let the shader stages store
// (fragmentStoresAndAtomics; vertexPipelineStoresAndAtomics for the vertex stages that index the pool, else those shaders leave the marks partial), the pool tables must be on, and
// one more dynamic storage descriptor must fit the pipeline layout's limit beside DM_VK_DYN_SSBO's thirds. Off, nothing below differs from before
static int g_mark_on, g_mark_vtx; static int g_mark_partial;
static void pool_mark_init(void) {
    const char *e = getenv("DM_VK_POOL_MARK"); if (!(e && atoi(e))) return;
    VkPhysicalDeviceFeatures2 f2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2}; vkGetPhysicalDeviceFeatures2(V.pd, &f2); const VkPhysicalDeviceFeatures f = f2.features;   // (all it has is enabled: gfx_vk_core.c)
    const VkPhysicalDeviceLimits *lim = &V.props.limits;
    uint32_t dyn = lim->maxDescriptorSetStorageBuffersDynamic, dyn_used = V.dyn_ssbo ? dyn / 3 * 3 : 0;
    const char *why = !vk_pool_n() ? "the texture pool tables are off (DM_VK_POOL_N=0, or no shader of the bundle reads a pool)" : !f.fragmentStoresAndAtomics ? "the device has no fragment stores (fragmentStoresAndAtomics)" :
                      dyn_used + 1 > dyn ? "no dynamic storage descriptor to spare beside DM_VK_DYN_SSBO's" : NULL;
    if (why) { VKLOG("[RED] pool read marks: off (DM_VK_POOL_MARK=1 asked): %s", why); return; }
    g_mark_on = 1; g_mark_vtx = f.vertexPipelineStoresAndAtomics;
    // (each graphics stage sees one storage buffer more: the translator's stage sets have at most Maxwell's 16 and the geometry claim's one; the limit is logged to check it)
    VKLOG("pool read marks: on (DM_VK_POOL_MARK): shaders that index the texture pool store the IDs they read; the pool tables give a texture its image only once it is read (the device binds %u storage buffers a stage, %u dynamic a layout)%s",
          lim->maxPerStageDescriptorStorageBuffers, dyn, g_mark_vtx ? "" : "; it has no vertex-stage stores: a vertex shader that indexes the pool leaves the marks partial");
}
int vk_pool_mark_on(void) { static pthread_once_t once = PTHREAD_ONCE_INIT; pthread_once(&once, pool_mark_init); return g_mark_on; }
int vk_pool_mark_partial(void) { return __atomic_load_n(&g_mark_partial, __ATOMIC_RELAXED); }
static void pool_mark_give_up(uint64_t key, int stage, const char *why) {   // one shader's reads cannot be seen: the marks decide nothing from now on (every ID counts as read, as without them)
    if (!__atomic_exchange_n(&g_mark_partial, 1, __ATOMIC_RELAXED)) VKLOG("[RED] pool read marks: shader %016llx (stage %d): %s; its reads cannot be marked, so from now on every texture ID counts as read (as without DM_VK_POOL_MARK)", (unsigned long long)key, stage, why);
}
// Pool lengths are process-wide, as is set 0. Only image kinds absent from every bundle entry
// have a one-element binding. Account for those placeholders in the device limit too; do not
// divide the available descriptor budget by four when only two full arrays can ever be used.
static int g_pool_nt, g_pool_ns;
static void pool_lengths_init(void) {
    bundle_auto();
    const char *e = getenv("DM_VK_POOL_N"); int n = e ? atoi(e) : 65536; if (n < 0) n = 0; if (n > 65536) n = 65536;
    int active = __builtin_popcount(g_pool_kinds), spare = POOL_TEX_SLOTS - active;
    int s = n ? 4096 : 0, tbl = (vk_tbl_n() ? vk_tbl_n() : 1) * TBL_SLOTS;
    long limi = (long)V.props.limits.maxPerStageDescriptorSampledImages - tbl - 256, lims = (long)V.props.limits.maxPerStageDescriptorSamplers - tbl - 256;
    if (limi > (long)V.props.limits.maxDescriptorSetSampledImages - tbl - 256) limi = (long)V.props.limits.maxDescriptorSetSampledImages - tbl - 256;
    if (lims > (long)V.props.limits.maxDescriptorSetSamplers - tbl - 256) lims = (long)V.props.limits.maxDescriptorSetSamplers - tbl - 256;
    if (n && active && (long)n * active + spare > limi) n = limi > spare ? (int)((limi - spare) / active) & ~255 : 0;
    if (n && !active && limi < spare) n = 0;
    if (s && (long)s > lims) s = lims > 0 ? (int)lims & ~255 : 0;
    if (!g_pool_kinds && !g_pool_samp_seen) n = s = 0;   // no shader of the bundle reads a pool: set 0 keeps one element per pool binding (4096 samplers made every set Welcome Tour binds larger for nothing)
    if (!n || !s) n = s = 0;
    VKLOG("texture pool tables: %d textures x %d kinds (bundle mask %#x, %d unused bindings of one), %d samplers (DM_VK_POOL_N %s; the device binds %u sampled images and %u samplers a stage, %u and %u a set)", n, active, g_pool_kinds, spare, s, e ? e : "unset",
          V.props.limits.maxPerStageDescriptorSampledImages, V.props.limits.maxPerStageDescriptorSamplers, V.props.limits.maxDescriptorSetSampledImages, V.props.limits.maxDescriptorSetSamplers);
    g_pool_nt = n; g_pool_ns = s;
}
static int pool_lengths(int *tex, int *samp) {
    static pthread_once_t once = PTHREAD_ONCE_INIT; pthread_once(&once, pool_lengths_init);
    if (tex) *tex = g_pool_nt; if (samp) *samp = g_pool_ns; return g_pool_nt;
}
int vk_pool_n(void) { int t; pool_lengths(&t, NULL); return t; }
int vk_pool_samp_n(void) { int s; pool_lengths(NULL, &s); return s; }
unsigned vk_pool_kinds(void) { return g_pool_kinds; }   // the pool kinds set 0 declares at full length (the others have one element)
VkDescriptorSetLayout vk_tbl_dsl(void) {   // one common set 0: large bindings only for kinds the whole bundle uses
    static VkDescriptorSetLayout l; VkDescriptorSetLayout r = __atomic_load_n(&l, __ATOMIC_ACQUIRE); if (r) return r;
    pthread_mutex_lock(&g_sh_mu);
    if (!l) { VkDescriptorSetLayoutBinding lb[SET0_BINDINGS + 1]; uint32_t n = (uint32_t)(vk_tbl_n() ? vk_tbl_n() : 1), np = (uint32_t)(vk_pool_n() ? vk_pool_n() : 1), ns = (uint32_t)(vk_pool_samp_n() ? vk_pool_samp_n() : 1);
        for (uint32_t i = 0; i < TBL_SLOTS; i++) lb[i] = (VkDescriptorSetLayoutBinding){i, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, n, VK_SHADER_STAGE_ALL_GRAPHICS, NULL};
        for (uint32_t i = 0; i < POOL_TEX_SLOTS; i++) lb[POOL_B0 + i] = (VkDescriptorSetLayoutBinding){POOL_B0 + i, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, (g_pool_kinds & (1u << i)) ? np : 1, VK_SHADER_STAGE_ALL_GRAPHICS, NULL};
        lb[POOL_SAMP_B] = (VkDescriptorSetLayoutBinding){POOL_SAMP_B, VK_DESCRIPTOR_TYPE_SAMPLER, ns, VK_SHADER_STAGE_ALL_GRAPHICS, NULL};
        // DM_VK_POOL_MARK=1: the pool read marks after them (binding SET0_BINDINGS, gfx_vk.h POOL_MARK_B), one storage buffer whose dynamic offset picks the submission slot's words
        uint32_t nb = SET0_BINDINGS; if (vk_pool_mark_on()) { lb[nb] = (VkDescriptorSetLayoutBinding){nb, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, 1, VK_SHADER_STAGE_ALL_GRAPHICS, NULL}; nb++; }
        VkDescriptorSetLayout m = shader_layout(lb, nb); __atomic_store_n(&l, m, __ATOMIC_RELEASE); }
    pthread_mutex_unlock(&g_sh_mu);
    return l;
}
// Is the id used only where a texture table's array length (which 0: the length operand of OpTypeArray) or its index clamp (which 1: an operand of GLSL.std.450 UMin) may be?
// Every other occurrence in an id position fails it; words known to be literals (strings, constants' values, decorations, image operand masks and so on) are passed over.
// A false failure only keeps that shader's arrays out of the table (its own set, as before)
static int spv_only_uses(const uint32_t *w, uint32_t nw, uint32_t id, int which) {
    for (uint32_t i = 5; i < nw;) { uint32_t n = w[i] >> 16, op = w[i] & 0xffff; if (!n || i + n > nw) return 0;
        if (op == 3 || op == 4 || op == 5 || op == 6 || op == 7 || op == 8 || op == 10 || op == 11 || op == 14 || op == 15 || op == 17 || op == 330) { i += n; continue; }   // strings and literals only
        for (uint32_t k = 1; k < n; k++) { if (w[i + k] != id) continue;
            if (which == 0 && op == 28 && k == 3) continue;                        // OpTypeArray's length
            if (which == 1 && op == 12 && k >= 5 && w[i + 4] == 38) continue;       // an operand of UMin
            if ((op == 43 || op == 50) && k == 2) continue;                         // its own definition
            if ((op == 43 || op == 50 || op == 45) && k >= 3) continue;             // a constant's value words
            if ((op == 71 && k >= 2) || (op == 72 && k >= 3) || (op == 16 && k >= 2) || (op == 331 && k == 2)) continue;   // decoration / execution mode literals
            if ((op == 12 && k == 4) || (op == 52 && k == 3)) continue;             // the extended instruction's number; OpSpecConstantOp's opcode
            if ((op == 21 || op == 22) && k >= 2) continue;                         // widths
            if ((op == 23 || op == 24) && k == 3) continue;                         // component / column counts
            if (op == 25 && k >= 3) continue;                                       // OpTypeImage's dim, depth, arrayed, ms, sampled, format
            if ((op == 32 && k == 2) || (op == 59 && k == 3) || (op == 54 && k == 3)) continue;   // storage classes, function control
            if ((op == 61 && k >= 4) || (op == 62 && k >= 3) || (op == 63 && k >= 3) || (op == 68 && k == 4)) continue;   // memory access, member index
            if ((op == 81 && k >= 4) || (op == 82 && k >= 5) || (op == 79 && k >= 5)) continue;   // composite indices, shuffle components
            if ((op == 246 && k == 3) || (op == 247 && k == 2) || (op == 250 && k >= 4) || (op == 251 && k >= 3)) continue;   // merge / branch literals, switch cases
            if (((op >= 87 && op <= 99) || (op >= 305 && op <= 315)) && w[i + k] < 0x200 && k >= 4) continue;   // an image operands mask (the ids this is asked about are never that small in practice; a wrong guess fails the shader, not the draw)
            return 0; }
        i += n; }
    return 1;
}
// cut the length of the texture-table arrays declared through the variables tv[] (ntv of them) from their old length to newn, and the UMin clamps of their indices with it
static int spv_tbl_shrink(uint32_t *w, uint32_t nw, const uint32_t *tv, int ntv, uint32_t newn) {
    uint32_t bound = w[3]; if (!bound || bound > (1u << 22)) return 0;
    uint32_t *ptr_of = calloc(bound, 4), *pointee = calloc(bound, 4), *arr_len = calloc(bound, 4), *cval = calloc(bound, 4), *ctype = calloc(bound, 4); int ok = 1;
    for (uint32_t i = 5; i < nw;) { uint32_t n = w[i] >> 16, op = w[i] & 0xffff; if (!n || i + n > nw) { ok = 0; break; }
        if (op == 59 && n >= 4 && w[i + 2] < bound) ptr_of[w[i + 2]] = w[i + 1];
        if (op == 32 && n == 4 && w[i + 1] < bound) pointee[w[i + 1]] = w[i + 3];
        if (op == 28 && n == 4 && w[i + 1] < bound) arr_len[w[i + 1]] = w[i + 3];
        if (op == 43 && n == 4 && w[i + 2] < bound) { cval[w[i + 2]] = w[i + 3]; ctype[w[i + 2]] = w[i + 1]; }
        i += n; }
    uint32_t lens[TBL_SLOTS] = {0}, clamps[TBL_SLOTS] = {0}; int nl = 0, nc = 0;
    for (int t = 0; t < ntv && ok; t++) {
        uint32_t v = tv[t], p = v < bound ? ptr_of[v] : 0, a = p && p < bound ? pointee[p] : 0, l = a && a < bound ? arr_len[a] : 0;
        if (!l || l >= bound || !ctype[l]) { ok = 0; break; }
        if (cval[l] <= newn) continue;   // already short enough (a pool array of the sampler range under a longer table, or a table array beside a longer one): as it is
        int had = 0; for (int j = 0; j < nl; j++) had |= lens[j] == l; if (!had) lens[nl++] = l;
        uint32_t c = 0; for (uint32_t id = 1; id < bound && !c; id++) if (ctype[id] == ctype[l] && cval[id] == cval[l] - 1 && id != l) c = id;   // the clamp constant: the same type, one less
        if (!c) { ok = 0; break; }
        had = 0; for (int j = 0; j < nc; j++) had |= clamps[j] == c; if (!had) clamps[nc++] = c;
    }
    for (int j = 0; j < nl && ok; j++) ok = spv_only_uses(w, nw, lens[j], 0);
    for (int j = 0; j < nc && ok; j++) ok = spv_only_uses(w, nw, clamps[j], 1);
    if (ok) {   // the constants' values, in place (OpConstant type id value)
        for (uint32_t i = 5; i < nw;) { uint32_t n = w[i] >> 16, op = w[i] & 0xffff;
            if (op == 43 && n == 4) { for (int j = 0; j < nl; j++) if (w[i + 2] == lens[j]) w[i + 3] = newn; for (int j = 0; j < nc; j++) if (w[i + 2] == clamps[j]) w[i + 3] = newn - 1; }
            i += n; } }
    free(ptr_of); free(pointee); free(arr_len); free(cval); free(ctype);
    return ok;
}
// Pool arrays are indexed by a handle each fragment reads from the material records (a varying), so the index can differ between the invocations of one draw. Vulkan requires such
// an index's resource to be decorated NonUniform (SPV_EXT_descriptor_indexing, shaderSampledImageArrayNonUniformIndexing); undecorated, the result is undefined and Adreno takes one
// invocation's descriptor for a whole wave. Metal's argument buffers have no such rule, which is why the Mac renders the same SPIR-V correctly. Decorates the access chains into
// the pool arrays, their indices, the loaded descriptors and the sampled images made from them; adds the capabilities and the extension. *pw may be reallocated. Returns the decorations added
static int spv_nonuniform(uint32_t **pw, uint32_t *pnw, const uint32_t *vars, int nvars) {
    uint32_t *w = *pw, nw = *pnw, bound = w[3]; if (!bound || bound > (1u << 22) || !nvars) return 0;
    uint8_t *mark = calloc(bound, 1); uint32_t ncap = 0, cap_end = 5, ann_end = 0, first_fn = nw; int has_ext = 0;
    for (int k = 0; k < nvars; k++) if (vars[k] < bound) mark[vars[k]] = 1;   // 1: a pool array variable
    for (uint32_t i = 5; i < nw;) { uint32_t n = w[i] >> 16, op = w[i] & 0xffff; if (!n || i + n > nw) { free(mark); return 0; }
        if (op == 17) { cap_end = i + n; ncap++; }
        if (op == 10 && n >= 2 && !memcmp(w + i + 1, "SPV_EXT_descriptor_indexing", 27)) has_ext = 1;
        if (op == 71 || op == 72 || op == 73 || op == 74 || op == 75 || op == 332 || op == 5632 || op == 5633) ann_end = i + n;
        if (op == 54 && first_fn == nw) first_fn = i;
        if ((op == 65 || op == 66) && n >= 5 && w[i + 3] < bound && mark[w[i + 3]] == 1) { if (w[i + 2] < bound) mark[w[i + 2]] = 2; if (w[i + 4] < bound && !mark[w[i + 4]]) mark[w[i + 4]] = 3; }
        else if (op == 61 && n >= 4 && w[i + 3] < bound && mark[w[i + 3]] == 2 && w[i + 2] < bound) mark[w[i + 2]] = 4;
        else if (op == 86 && n >= 5 && ((w[i + 3] < bound && mark[w[i + 3]] == 4) || (w[i + 4] < bound && mark[w[i + 4]] == 4)) && w[i + 2] < bound) mark[w[i + 2]] = 5;
        i += n; }
    uint32_t nd = 0; for (uint32_t id = 1; id < bound; id++) nd += mark[id] >= 2;
    if (!nd || !ann_end || ann_end > first_fn || cap_end > ann_end) { free(mark); return 0; }
    uint32_t add = 2 * 2 + (has_ext ? 0 : 8) + 3 * nd, *o = malloc((size_t)(nw + add) * 4), k = 0;
    memcpy(o, w, (size_t)cap_end * 4); k = cap_end;
    o[k++] = 2u << 16 | 17; o[k++] = 5301; o[k++] = 2u << 16 | 17; o[k++] = 5307;   // ShaderNonUniform, SampledImageArrayNonUniformIndexing
    if (!has_ext) { o[k++] = 8u << 16 | 10; memset(o + k, 0, 28); memcpy(o + k, "SPV_EXT_descriptor_indexing", 27); k += 7; }
    memcpy(o + k, w + cap_end, (size_t)(ann_end - cap_end) * 4); k += ann_end - cap_end;
    for (uint32_t id = 1; id < bound; id++) if (mark[id] >= 2) { o[k++] = 3u << 16 | 71; o[k++] = id; o[k++] = 5300; }
    memcpy(o + k, w + ann_end, (size_t)(nw - ann_end) * 4); k += nw - ann_end;
    free(mark); free(w); *pw = o; *pnw = k; return (int)nd;
}
// The pool read marks (DM_VK_POOL_MARK=1; gfx_vk_exec.c pool marks): after each access chain into a texture pool array (vars[], nvars of them) the shader stores 1 into a
// storage buffer of uints at the ID it indexes (marks[id] = 1), so the host learns which registered textures are drawn. The pool tables otherwise give every registered texture
// with contents an image and count it as drawn at every fill: Ocarina of Time's Kokiri Forest (Odin 3, 2026-11-11) held 5,957 images of 2.85 GB at the 3,072 MB image budget,
// which could drop none of them. A plain store, not an atomic: every writer writes the same 1, and the host only asks whether any did. The index is the translator's own clamp
// (GLSL.std.450 UMin of the handle's ID with the array's last index, which spv_tbl_shrink cut to the table's length; an OpBitcast to int may follow it) or a constant inside
// the table: robust buffer access is off, so an index that cannot be shown to be inside the np words fails the shader's marks (the caller then gives the marks up). The buffer
// is a new variable in set 0 (binding POOL_MARK_B; the caller's decoration pass sets both), with its own runtime array and BufferBlock struct, as the translator declares its
// storage buffers (SPIR-V 1.3); an 1.4 module would need it in its entry points' interfaces and fails instead. *pw may be reallocated. Returns the stores added (0: the module
// indexes no texture pool array and is unchanged), -1 with *why on failure (the module unchanged). *var_out: the buffer's variable
static int spv_pool_mark(uint32_t **pw, uint32_t *pnw, const uint32_t *vars, int nvars, uint32_t np, uint32_t *var_out, const char **why) {
    uint32_t *w = *pw, nw = *pnw, bound = nw >= 5 ? w[3] : 0; *var_out = 0; *why = NULL;
    if (!bound || bound > (1u << 22) || !np) { *why = "its header is not valid"; return -1; }
    if (w[1] >= 0x10400) { *why = "SPIR-V 1.4 or later (the buffer would have to join its entry points' interfaces)"; return -1; }
    uint8_t *pool = calloc(bound, 1); uint32_t *def = calloc(bound, 4); if (!pool || !def) { free(pool); free(def); *why = "out of memory"; return -1; }
    for (int k = 0; k < nvars; k++) if (vars[k] < bound) pool[vars[k]] = 1;   // the texture pool array variables
    uint32_t glsl = 0, u32 = 0, pu = 0, c0 = 0, c1 = 0, ann_end = 0, first_fn = 0, nsite = 0;
    for (uint32_t i = 5; i < nw;) { uint32_t n = w[i] >> 16, op = w[i] & 0xffff; if (!n || i + n > nw) { *why = "a malformed instruction"; break; }
        if (op == 11 && n >= 3 && !memcmp(&w[i + 2], "GLSL.std.450", 12)) glsl = w[i + 1];                     // OpExtInstImport
        if (op == 21 && n == 4 && w[i + 2] == 32 && w[i + 3] == 0 && !u32) u32 = w[i + 1];                     // OpTypeInt 32 0
        if (op == 32 && n == 4 && w[i + 2] == 2 && u32 && w[i + 3] == u32 && !pu) pu = w[i + 1];                // OpTypePointer Uniform uint
        if (op == 43 && n == 4 && u32 && w[i + 1] == u32 && w[i + 3] == 0 && !c0) c0 = w[i + 2];                // OpConstant uint 0
        if (op == 43 && n == 4 && u32 && w[i + 1] == u32 && w[i + 3] == 1 && !c1) c1 = w[i + 2];                // OpConstant uint 1
        if ((op == 71 || op == 72 || op == 73 || op == 74 || op == 75 || op == 332 || op == 5632 || op == 5633) && !first_fn) ann_end = i + n;   // the annotations
        if (op == 54 && !first_fn) first_fn = i;                                                                // the first OpFunction
        if ((op == 43 || op == 124 || op == 12) && n >= 4 && w[i + 2] < bound) def[w[i + 2]] = i;               // OpConstant, OpBitcast, OpExtInst: where their results are made
        if ((op == 65 || op == 66) && n >= 5 && w[i + 3] < bound && pool[w[i + 3]]) nsite++;                    // an access chain into a pool array
        i += n; }
    if (!*why && !nsite) { free(pool); free(def); return 0; }
    if (!*why && (!u32 || !glsl)) *why = "it has no 32-bit unsigned integer type or no GLSL.std.450 import (not the translator's clamp)";
    if (!*why && (!ann_end || !first_fn || ann_end > first_fn)) *why = "its annotations and functions are not where SPIR-V puts them";
    uint32_t next = bound, rta = next++, st = next++, pst = next++, var = next++; if (!pu) pu = next++; int add_c0 = !c0, add_c1 = !c1; if (add_c0) c0 = next++; if (add_c1) c1 = next++;
    uint32_t add = 20 + 3 + 3 + 4 + 4 + (pu >= bound ? 4 : 0) + (add_c0 ? 4 : 0) + (add_c1 ? 4 : 0) + 9 * nsite, *o = *why ? NULL : malloc((size_t)(nw + add) * 4), k = 0;
    if (!*why && !o) *why = "out of memory";
    if (*why) { free(pool); free(def); free(o); return -1; }
    memcpy(o, w, (size_t)ann_end * 4); k = ann_end;
    o[k++] = 4u << 16 | 71; o[k++] = rta; o[k++] = 6; o[k++] = 4;                     // OpDecorate rta ArrayStride 4
    o[k++] = 3u << 16 | 71; o[k++] = st; o[k++] = 3;                                  // OpDecorate st BufferBlock
    o[k++] = 5u << 16 | 72; o[k++] = st; o[k++] = 0; o[k++] = 35; o[k++] = 0;         // OpMemberDecorate st 0 Offset 0
    o[k++] = 4u << 16 | 71; o[k++] = var; o[k++] = 34; o[k++] = 0;                    // OpDecorate var DescriptorSet 0
    o[k++] = 4u << 16 | 71; o[k++] = var; o[k++] = 33; o[k++] = POOL_MARK_B;          // OpDecorate var Binding POOL_MARK_B
    memcpy(o + k, w + ann_end, (size_t)(first_fn - ann_end) * 4); k += first_fn - ann_end;
    o[k++] = 3u << 16 | 29; o[k++] = rta; o[k++] = u32;                               // OpTypeRuntimeArray rta uint
    o[k++] = 3u << 16 | 30; o[k++] = st; o[k++] = rta;                                // OpTypeStruct st rta
    o[k++] = 4u << 16 | 32; o[k++] = pst; o[k++] = 2; o[k++] = st;                    // OpTypePointer pst Uniform st
    if (pu >= bound) { o[k++] = 4u << 16 | 32; o[k++] = pu; o[k++] = 2; o[k++] = u32; }   // OpTypePointer pu Uniform uint
    if (add_c0) { o[k++] = 4u << 16 | 43; o[k++] = u32; o[k++] = c0; o[k++] = 0; }    // OpConstant uint 0
    if (add_c1) { o[k++] = 4u << 16 | 43; o[k++] = u32; o[k++] = c1; o[k++] = 1; }    // OpConstant uint 1
    o[k++] = 4u << 16 | 59; o[k++] = pst; o[k++] = var; o[k++] = 2;                   // OpVariable pst var Uniform
    int stores = 0;
    for (uint32_t i = first_fn; i < nw;) { uint32_t n = w[i] >> 16, op = w[i] & 0xffff;
        memcpy(o + k, w + i, (size_t)n * 4); k += n;
        if ((op == 65 || op == 66) && n >= 5 && w[i + 3] < bound && pool[w[i + 3]]) {
            uint32_t y = w[i + 4], ok = 0;   // the index; through an OpBitcast to the uint it was made as
            if (y < bound && def[y] && (w[def[y]] & 0xffff) == 124) y = w[def[y] + 3];
            if (y < bound && def[y] && w[def[y] + 1] == u32) { uint32_t d = def[y], dop = w[d] & 0xffff;
                if (dop == 43) ok = w[d + 3] < np;   // a constant index inside the table
                else if (dop == 12 && (w[d] >> 16) == 7 && w[d + 3] == glsl && w[d + 4] == 38)   // UMin(x, c) with c a constant inside the table: the translator's clamp
                    for (int a = 5; a <= 6 && !ok; a++) { uint32_t c = w[d + a]; ok = c < bound && def[c] && (w[def[c]] & 0xffff) == 43 && w[def[c] + 1] == u32 && w[def[c] + 3] < np; } }
            if (!ok) { *why = "a pool array index that is not the translator's clamp to the table"; break; }
            uint32_t p = next++;
            o[k++] = 6u << 16 | 65; o[k++] = pu; o[k++] = p; o[k++] = var; o[k++] = c0; o[k++] = y;   // OpAccessChain pu p var 0 index
            o[k++] = 3u << 16 | 62; o[k++] = p; o[k++] = c1;                                       // OpStore p 1
            stores++; }
        i += n; }
    free(pool); free(def);
    if (*why) { free(o); return -1; }
    o[3] = next; free(w); *pw = o; *pnw = k; *var_out = var; return stores;
}
// A pipeline currently combines vertex, fragment and geometry sets. Reserve at most one third of the total dynamic-storage limit per stage.
// Only directly bound single storage buffers participate; arrays, UBO-derived pointers, compute and unsupported alignment keep their static descriptor.
static uint32_t storage_dynamic_limit(int stage, int enabled, const VkPhysicalDeviceLimits *limits) {
    VkDeviceSize a = limits->minStorageBufferOffsetAlignment;
    if (!enabled || stage < 0 || stage >= 3 || !a || a > 256 || 256 % a) return 0;
    return limits->maxDescriptorSetStorageBuffersDynamic / 3;
}
static int storage_dynamic_binding(const dmsv_bind *b, uint32_t used, uint32_t limit) {
    return used < limit && b->desc == RD_SSBO && b->count == 1 && (b->kind == RK_GS_CLAIM || (b->kind == RK_SBUF && b->reg >= 0 && b->reg < 16));
}
static vkshader_t *vk_shader_q(uint64_t key, int stage, int quiet);
vkshader_t *vk_shader(uint64_t key, int stage) { return vk_shader_q(key, stage, 0); }
// A geometry shader's input primitive is fixed when it is translated, and Maxwell's shader header doesn't give it: shader_vk.py translates each geometry shader for
// triangles under its own key and for the other four kinds under derived keys. Read as triangles, the points Xenoblade Chronicles 3's particle updates draw were taken
// three at a time (Vulkan leaves a mismatch undefined), and the particles' records came out wrong: no husk particles, send-off particles or item orbs
vkshader_t *vk_shader_gs_topo(vkshader_t *gs, VkPrimitiveTopology topo) {
    int t;
    switch (topo) {
    case VK_PRIMITIVE_TOPOLOGY_POINT_LIST: t = 0; break;
    case VK_PRIMITIVE_TOPOLOGY_LINE_LIST: case VK_PRIMITIVE_TOPOLOGY_LINE_STRIP: t = 1; break;
    case VK_PRIMITIVE_TOPOLOGY_LINE_LIST_WITH_ADJACENCY: case VK_PRIMITIVE_TOPOLOGY_LINE_STRIP_WITH_ADJACENCY: t = 2; break;
    case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST_WITH_ADJACENCY: case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP_WITH_ADJACENCY: t = 4; break;
    default: return gs;   // triangles (lists, strips, fans): the shader's own translation
    }
    if (!gs) return gs;
    vkshader_t *v = vk_shader_q(gs->key ^ ((uint64_t)(t + 1) * 0x9E3779B97F4A7C15ull), 2, 1);
    { static int n; if (n < 8) { n++; VKLOG("geometry shader %016llx drawn with topology %d: %s", (unsigned long long)gs->key, (int)topo, v ? "its translation for that input" : "[RED] no translation for that input in the bundle, the triangles one"); } }
    if (!v) return gs;
    v->xfb = gs->xfb; memcpy(v->xfb_stride, gs->xfb_stride, sizeof v->xfb_stride);
    return v;
}
static void spv_runtime_dump(vkshader_t *s, const uint32_t *w, uint32_t nw);
static vkshader_t *vk_shader_q(uint64_t key, int stage, int quiet) {
    bundle_auto();
    struct { uint64_t key; uint64_t stage; } hk = {key, (uint64_t)stage};
    pthread_mutex_lock(&g_sh_mu);
    vkshader_t *s = nvn_hm_get(&g_shaders, &hk);
    if (s) { pthread_mutex_unlock(&g_sh_mu); return s == (void *)1 ? NULL : s; }
    const bent *e = g_bundle ? find(key) : NULL;
    if (!e) { nvn_hm_put(&g_shaders, &hk, (void *)1); pthread_mutex_unlock(&g_sh_mu);
              static int n; if (!quiet && n++ < 16) VKLOG("[RED] shader %016llx (stage %d) is not in the bundle", (unsigned long long)key, stage); return NULL; }
    pthread_mutex_unlock(&g_sh_mu);
    uint32_t nw = 0, *w = bundle_spv(e, &nw);   // copied or decompressed outside the lock
    pthread_mutex_lock(&g_sh_mu);
    if ((s = nvn_hm_get(&g_shaders, &hk))) { pthread_mutex_unlock(&g_sh_mu); free(w); return s == (void *)1 ? NULL : s; }   // another thread made it meanwhile
    if (!w) { nvn_hm_put(&g_shaders, &hk, (void *)1); pthread_mutex_unlock(&g_sh_mu); return NULL; }
    // reflection
    const uint8_t *r = g_bundle + e->refl_off; uint32_t hdr[5]; memcpy(hdr, r, 20);
    // Counts must fit this record, not merely the mapped file; widen before multiplying.
    if (20 + 20ull * hdr[0] + hdr[1] > e->refl_len) {
        VKLOG("[RED] shader %016llx: reflection exceeds its record: left out as if missing", (unsigned long long)key);
        free(w); nvn_hm_put(&g_shaders, &hk, (void *)1); pthread_mutex_unlock(&g_sh_mu); return NULL;
    }
    s = calloc(1, sizeof *s); s->key = key; s->stage = stage; s->set = set_of(stage);
    s->nbind = (int)hdr[0]; s->c1_len = hdr[1]; memcpy(s->local, hdr + 2, 12);
    if (stage == 6 || stage == 7) {   // a task / mesh kernel: the header words are shader_meshwrap.facts, not a local size (its LocalSize is in the SPIR-V)
        uint32_t w0 = s->local[0]; s->mesh_kind = (int)(w0 >> 24 & 3); s->mesh_threads = w0 & 0x3ff; s->mesh_pay = w0 >> 10 & 0x3ff; s->mesh_task = w0 >> 20 & 1; s->mesh_verts = w0 >> 21 & 7;
        s->mesh_max_v = s->local[1] & 0xffff; s->mesh_max_p = s->local[1] >> 16; s->mesh_ncomp = s->local[2]; s->local[0] = s->mesh_threads ? s->mesh_threads : 1; s->local[1] = s->local[2] = 1; }
    if (stage == 1 && s->local[0] >> 31) {   // a fragment shader's output map in its header words (shader_vk.py ps_omap_words); the output-location masks keep their low bits
        s->omap = (s->local[1] >> 16) | (s->local[2] >> 16) << 16; s->omap_known = 1; s->local[0] &= 0xffff; s->local[1] &= 0xffff; s->local[2] &= 0xffff; }
    s->b = (const dmsv_bind *)(r + 20); s->c1 = r + 20 + (size_t)20 * s->nbind;
    // texture tables (gfx_vk.h): the arrays of TBL_MIN+ combined image samplers read through one uniform buffer's handles, by slot; one source buffer per shader
    uint32_t tbind[TBL_SLOTS] = {0}, tvar[TBL_SLOTS] = {0}; int thave[TBL_SLOTS] = {0}, tbig = 0; s->tbl_reg = -1;
    if (vk_tbl_n() && stage <= 4) for (int i = 0; i < s->nbind && i < 96; i++) { const dmsv_bind *b = &s->b[i];
        if (b->kind != RK_TEX_UBO || b->count < TBL_MIN || b->samp != RS_TEX || (b->flags & 2) || (b->dim != 1 && b->dim != 3) || b->reg < 0 || b->reg >= 16) continue;
        int slot = b->dim == 3 ? 2 : (b->flags & 1) ? 1 : 0;
        if (thave[slot] || (s->ntbl && (s->tbl_reg != b->reg || s->tbl_off != b->off))) { static int nl; if (nl++ < 8) VKLOG("shader %016llx: texture array binding %u (%u entries, uniform buffer %d +%d) left in its stage's set: the table has %s", (unsigned long long)key, b->binding, b->count, b->reg, b->off, thave[slot] ? "that slot already" : "another source buffer"); continue; }
        thave[slot] = 1; tbind[slot] = b->binding; s->tbl[i] = (uint8_t)(1 + slot); s->ntbl++; s->tbl_reg = b->reg; s->tbl_off = b->off; if (b->count > (uint32_t)vk_tbl_n()) tbig = 1; }
    if (s->ntbl) {   // their variables (OpDecorate Binding), then the arrays cut to the table's length when they are longer
        for (uint32_t i = 5; i < nw;) { uint32_t n = w[i] >> 16, op = w[i] & 0xffff; if (!n) break;
            if (op == 71 && n >= 4 && w[i + 2] == 33) for (int t = 0; t < TBL_SLOTS; t++) if (thave[t] && w[i + 3] == tbind[t]) tvar[t] = w[i + 1];
            i += n; }
        uint32_t tv[TBL_SLOTS]; int ntv = 0; for (int t = 0; t < TBL_SLOTS; t++) if (thave[t]) { if (tvar[t]) tv[ntv++] = tvar[t]; else s->ntbl = 0; }
        if (s->ntbl && tbig && !spv_tbl_shrink(w, nw, tv, ntv, (uint32_t)vk_tbl_n())) s->ntbl = 0;
        if (!s->ntbl) { memset(s->tbl, 0, sizeof s->tbl); memset(thave, 0, sizeof thave); s->tbl_reg = -1; static int nl; if (nl++ < 8) VKLOG("[RED] shader %016llx: its texture arrays could not be moved to the table (their SPIR-V could not be cut to %d); left in its stage's set", (unsigned long long)key, vk_tbl_n()); }
        else { static int nl; if (nl++ < 4) VKLOG("shader %016llx: %d texture arrays moved to the table set (uniform buffer %d +%d, %d entries each)", (unsigned long long)key, s->ntbl, s->tbl_reg, s->tbl_off, vk_tbl_n()); }
    }
    // texture and sampler pools (gfx_vk.h RK_TEXP / RK_SAMPP; shader_vk.py classify pools): arrays indexed by the handle's texture or sampler ID, declared as long as the whole ID
    // ranges, into set 0's pool bindings by image kind (2D, 2D array, cube, 3D; the samplers under POOL_SAMP_B), cut to the tables' lengths with their index clamps. One a shader
    // cannot be given (the tables off, a compute kernel, a kind the tables have none of, an uncuttable module) leaves the shader undrawable: its own set would need 2^20 descriptors
    // Several arrays may share a kind (a 2D and a 2D shadow image type over the same texture IDs): they share the binding, as Vulkan lets variables of one descriptor type do
    struct { uint32_t var; int slot; } pv[96]; int npv = 0; const char *pfail = NULL;
    for (int i = 0; i < s->nbind && i < 96; i++) { const dmsv_bind *b = &s->b[i]; if (b->kind != RK_TEXP && b->kind != RK_SAMPP) continue;
        int slot = b->kind == RK_SAMPP ? POOL_TEX_SLOTS : (b->flags & 2) ? -1 : b->dim == 1 ? ((b->flags & 1) ? 1 : 0) : b->dim == 3 ? 2 : b->dim == 2 ? 3 : -1;
        if (!vk_pool_n()) pfail = "the pool tables are off (DM_VK_POOL_N=0)"; else if (stage > 4) pfail = "a compute kernel has no pool table set"; else if (slot < 0) pfail = "an image kind the pool tables have none of";
        if (pfail) { static int nl; if (nl++ < 8) VKLOG("[RED] shader %016llx (stage %d): texture pool array binding %u (dim %u, flags %#x, kind %d): %s", (unsigned long long)key, stage, b->binding, b->dim, b->flags, b->kind, pfail); continue; }
        s->tbl[i] = (uint8_t)(1 + TBL_SLOTS + slot); s->npool++; }
    if (s->npool && !pfail) {   // their variables (OpDecorate Binding, by the reflection's binding numbers), then the arrays cut: the textures' to vk_pool_n(), the samplers' to vk_pool_samp_n()
        for (uint32_t i = 5; i < nw;) { uint32_t n = w[i] >> 16, op = w[i] & 0xffff; if (!n) break;
            if (op == 71 && n >= 4 && w[i + 2] == 33) for (int k = 0; k < s->nbind && k < 96; k++) if (s->tbl[k] > TBL_SLOTS && s->b[k].binding == w[i + 3] && npv < 96) pv[npv++] = (typeof(pv[0])){w[i + 1], s->tbl[k] - 1 - TBL_SLOTS};
            i += n; }
        uint32_t tv[96], sv[96]; int ntv = 0, nsv = 0;
        for (int k = 0; k < npv; k++) { if (pv[k].slot == POOL_TEX_SLOTS) sv[nsv++] = pv[k].var; else tv[ntv++] = pv[k].var; }
        if (npv < s->npool) pfail = "an array variable was not found";
        if (!pfail && ntv && !spv_tbl_shrink(w, nw, tv, ntv, (uint32_t)vk_pool_n())) pfail = "its texture arrays could not be cut to the table's length";
        if (!pfail && nsv && !spv_tbl_shrink(w, nw, sv, nsv, (uint32_t)vk_pool_samp_n())) pfail = "its sampler array could not be cut to the table's length";
        { static int on = -1; if (on < 0) { const char *e = getenv("DM_VK_NONUNIFORM"); on = e && atoi(e); }   // DM_VK_NONUNIFORM=1 (start-up, opt-in): decorate them (measured on Ocarina of Time 2026-11-09: no visible change, so not by default)
          if (!pfail && on) { uint32_t all[192]; int na = 0; for (int k = 0; k < ntv; k++) all[na++] = tv[k]; for (int k = 0; k < nsv; k++) all[na++] = sv[k];
              int nd = spv_nonuniform(&w, &nw, all, na); static int nl; if (nl++ < 4) VKLOG("shader %016llx: %d pool array accesses decorated NonUniform", (unsigned long long)key, nd); } }
        // DM_VK_POOL_MARK=1: the texture IDs it reads stored for the host (spv_pool_mark); its buffer joins the pool variables so the decoration pass below puts it in set 0
        // at POOL_MARK_B. A shader that cannot be marked is still drawn as before, and the marks are given up for the run (vk_pool_mark_partial)
        if (!pfail && ntv && vk_pool_mark_on()) { uint32_t mv = 0; const char *why = NULL;
            int ns = stage != 1 && !g_mark_vtx ? (why = "the device has no stores in vertex stages (vertexPipelineStoresAndAtomics)", -1) : npv >= 96 ? (why = "too many pool arrays", -1) : spv_pool_mark(&w, &nw, tv, ntv, (uint32_t)vk_pool_n(), &mv, &why);
            if (ns < 0) pool_mark_give_up(key, stage, why);
            else if (ns) { pv[npv++] = (typeof(pv[0])){mv, POOL_MARK_B - POOL_B0}; s->mark_var = mv; static int nl; if (nl++ < 4) VKLOG("shader %016llx (stage %d): %d texture pool reads marked (DM_VK_POOL_MARK)", (unsigned long long)key, stage, ns); } }
    }
    if (pfail) { static int nl; if (nl++ < 16) VKLOG("[RED] shader %016llx (stage %d): %s; not drawn (its texture pool arrays can be bound only from the pool tables)", (unsigned long long)key, stage, pfail);
        nvn_hm_put(&g_shaders, &hk, (void *)1); pthread_mutex_unlock(&g_sh_mu); free(w); free(s); return NULL; }
    if (s->npool) { static int nl; if (nl++ < 4) VKLOG("shader %016llx (stage %d): %d pool arrays moved to the pool tables (%d textures a kind, %d samplers)", (unsigned long long)key, stage, s->npool, vk_pool_n(), vk_pool_samp_n()); }
    // SPIR-V with the stage's set (the table's variables: set 0, binding = slot; the pools': set 0, POOL_B0 + kind / POOL_SAMP_B)
    int f64_used = 0; uint32_t xb = 4;
    for (uint32_t i = 5; i < nw;) { uint32_t n = w[i] >> 16, op = w[i] & 0xffff; if (!n) break;
        if (op == 71 && n >= 4 && (w[i + 2] == 34 || w[i + 2] == 33)) { int slot = -1; for (int t = 0; t < TBL_SLOTS; t++) if (thave[t] && tvar[t] == w[i + 1]) slot = t;   // OpDecorate DescriptorSet / Binding
            for (int k = 0; k < npv; k++) if (pv[k].var == w[i + 1]) slot = POOL_B0 + pv[k].slot;
            if (w[i + 2] == 34) w[i + 3] = slot >= 0 ? 0 : (uint32_t)s->set; else if (slot >= 0) w[i + 3] = (uint32_t)slot; }
        if (op == 22 && n == 3 && w[i + 2] == 64) f64_used = 1;   // OpTypeFloat 64
        if (op == 16 && n >= 3 && w[i + 2] == 11) s->xfb = 1;     // OpExecutionMode Xfb: a vertex shader that captures transform feedback (shader_vk.py xfb_layout)
        if (op == 16 && n >= 3 && (w[i + 2] == 12 || w[i + 2] == 14 || w[i + 2] == 15)) s->depth_repl = 1;   // OpExecutionMode DepthReplacing / DepthGreater / DepthLess
        // each buffer's stride: the translator decorates an output XfbBuffer b, then XfbStride s (OpDecorate, or OpMemberDecorate for gl_Position's block)
        { uint32_t d = op == 71 && n >= 4 ? w[i + 2] : op == 72 && n >= 5 ? w[i + 3] : 0, v = op == 71 ? w[i + 3] : op == 72 && n >= 5 ? w[i + 4] : 0;
          if (d == 36) xb = v; else if (d == 37 && xb < 4) s->xfb_stride[xb] = v; }
        i += n; }
    // without VK_EXT_transform_feedback the capture declarations would make the module invalid: they are dropped (the capture's draws then write nothing, as before)
    if (s->xfb && !V.has_xfb) { uint32_t o = 5;
        for (uint32_t i = 5; i < nw;) { uint32_t n = w[i] >> 16, op = w[i] & 0xffff; if (!n || i + n > nw) { while (i < nw) w[o++] = w[i++]; break; }
            int drop = (op == 17 && n == 2 && w[i + 1] == 53) || (op == 16 && n >= 3 && w[i + 2] == 11) ||   // OpCapability TransformFeedback, OpExecutionMode Xfb
                       (op == 71 && n >= 3 && (w[i + 2] == 36 || w[i + 2] == 37)) || (op == 72 && n >= 4 && (w[i + 3] == 36 || w[i + 3] == 37));   // XfbBuffer / XfbStride
            if (drop) i += n; else for (uint32_t k = 0; k < n; k++) w[o++] = w[i++]; }
        nw = o; s->xfb = 0; VKRED_ONCE(90, "the device has no transform feedback: shaders that capture it run without"); }
    // the translator declares Float64 in every shader; the device has shaderFloat64 off (Adreno has none), which makes the declaration invalid: when the
    // shader has no 64-bit float type it becomes a second Shader declaration
    if (!f64_used) for (uint32_t i = 5; i < nw;) { uint32_t n = w[i] >> 16, op = w[i] & 0xffff; if (!n || op != 17) break; if (n == 2 && w[i + 1] == 10) w[i + 1] = 1; i += n; }
    // the translator declares fragment shader interlock in every pixel shader; the device has no VK_EXT_fragment_shader_interlock: when the shader uses no
    // interlock instruction, its OpCapability and OpExtension are dropped
    { int used = 0; for (uint32_t i = 5; i < nw;) { uint32_t n = w[i] >> 16, op = w[i] & 0xffff; if (!n) break; if (op == 5364 || op == 5365) used = 1; i += n; }   // OpBegin/EndInvocationInterlockEXT
      if (!used) { uint32_t o = 5;
        for (uint32_t i = 5; i < nw;) { uint32_t n = w[i] >> 16, op = w[i] & 0xffff; if (!n || i + n > nw) { while (i < nw) w[o++] = w[i++]; break; }
            int drop = (op == 17 && n == 2 && (w[i + 1] == 5378 || w[i + 1] == 5363 || w[i + 1] == 5372)) ||   // FragmentShader{Pixel,Sample,ShadingRate}InterlockEXT
                       (op == 10 && !memcmp(&w[i + 1], "SPV_EXT_fragment_shader_interlock", 33));
            if (drop) i += n; else for (uint32_t k = 0; k < n; k++) w[o++] = w[i++]; }
        nw = o; } }
    // subgroup operations of a class the stage lacks (not in supportedStages, VUID-RuntimeSpirv-None-06343; a class not in supportedOperations; quad operations outside
    // fragment and compute without quadOperationsInAllStages, -06342): the module runs as subgroups of one invocation (gfx_vk_subgroup.c). On a Retroid Pocket 5
    // (Adreno 650, Turnip: compute alone) Pokemon Pokopia's fragment shader 541aa7b27f1fae4d, the translator's SHFL helper, lost the device at the title screen.
    // Qualcomm's driver on the Y700 (Adreno 830) lists fragment and compute alone too: 12 of Tears of the Kingdom's vertex shaders (ballots) and 1 of Xenoblade
    // Chronicles 3's are rewritten there. DM_VK_SUBGROUP_ONE_LANE=0 (start-up): no module is rewritten, as before, for A/B tests
    static int one_lane = -1; if (one_lane < 0) { const char *e = getenv("DM_VK_SUBGROUP_ONE_LANE"); one_lane = !(e && !atoi(e)); if (!one_lane) VKLOG("subgroups: no one-invocation rewrite (DM_VK_SUBGROUP_ONE_LANE=0)"); }
    if (one_lane) { uint32_t ops = V.sg_stages & stage_flag(stage) ? V.sg_ops : 0; if (stage != 1 && stage != 5 && !V.sg_quad_all) ops &= ~(uint32_t)VK_SUBGROUP_FEATURE_QUAD_BIT;
      char why[64]; uint32_t n2 = 0, *w2 = spv_one_lane(w, nw, ops, &n2, why, sizeof why);
      if (w2) { free(w); w = w2; nw = n2; VKLOG("shader %016llx (stage %d): subgroups of one invocation (the device's subgroup operations in this stage: %#x)", (unsigned long long)key, stage, ops); }
      else if (why[0]) VKLOG("[RED] shader %016llx (stage %d) uses subgroup operations the device lacks in this stage (it has %#x), and %s is not covered by the one-invocation rewrite: left as it is", (unsigned long long)key, stage, ops, why); }
    if (stage == 0) { uint32_t n2 = 0, *w2 = spv_vs_init(w, nw, &n2); if (w2) { free(w); w = w2; nw = n2; } }
    // Maxwell's FMNMX returns the operand that is a number when the other is NaN, and its saturate turns NaN into 0 (IEEE minNum / maxNum, as Metal's fmin / fmax).
    // The translator emits GLSL.std.450 FMin / FMax / FClamp, whose result is undefined for a NaN operand, and Adreno passes the NaN on: their NaN-aware forms NMin /
    // NMax / NClamp give Maxwell's results. Xenoblade Chronicles 3's fur pixel shaders (4fd6e9e891e821cf, f8cc7339c930dc07) take max(0, sqrt(1 - x² - y²)) of
    // their fur direction texels; where x² + y² is just over 1 the NaN went through, the whole fur vector became NaN and its saturate 0, which the motion blur
    // reads as full speed down-left: the Nopons' fur smeared over their faces (the Mac and Eden: clean). DM_VK_NAN_MINMAX=0 (start-up, A/B): as translated
    { static int on = -1; if (on < 0) { const char *e = getenv("DM_VK_NAN_MINMAX"); on = !(e && !strcmp(e, "0")); }
      uint32_t glsl = 0;
      for (uint32_t i = 5; on && i < nw;) { uint32_t n = w[i] >> 16, op = w[i] & 0xffff; if (!n || i + n > nw) break;
          if (op == 11 && n >= 3 && !memcmp(&w[i + 2], "GLSL.std.450", 12)) glsl = w[i + 1];   // OpExtInstImport
          else if (op == 12 && n >= 5 && glsl && w[i + 3] == glsl) { uint32_t *x = &w[i + 4];   // OpExtInst: FMin 37 → NMin 79, FMax 40 → NMax 80, FClamp 43 → NClamp 81
              if (*x == 37) *x = 79; else if (*x == 40) *x = 80; else if (*x == 43) *x = 81; }
          i += n; } }
    // Qualcomm/Turnip repeat geometry execution for each output vertex. Claim each primitive in
    // fresh host-owned storage, only when analysis proves suppressed atomic results cannot affect raster output.
    // The descriptor is synthetic, never part of the guest ABI or on-disk reflection. Startup A/B: DM_VK_GS_CLAIM=0.
    if (stage == 2 && (V.props.vendorID == 0x5143 || strstr(V.props.deviceName, "Adreno"))) {
        static int on = -1; if (on < 0) { const char *e = getenv("DM_VK_GS_CLAIM"); on = !(e && !atoi(e)); }
        if (on && s->nbind < 95) { uint32_t n2 = 0, iv = 0; char why[128];
            uint32_t *w2 = spv_gs_claim(w, nw, (uint32_t)s->set, 95, &n2, &iv, why, sizeof why);
            if (w2) {
                dmsv_bind *b = malloc((size_t)(s->nbind + 1) * sizeof *b);
                if (b) { memcpy(b, s->b, (size_t)s->nbind * sizeof *b);
                    b[s->nbind++] = (dmsv_bind){.kind = RK_GS_CLAIM, .desc = RD_SSBO, .binding = 95, .count = 1, .reg = -1};
                    s->b = b; s->gs_claim_invocations = iv; free(w); w = w2; nw = n2;
                    VKLOG("shader %016llx: geometry side effects claimed once per primitive (%s)", (unsigned long long)key, why);
                } else free(w2);
            } else if (strcmp(why, "no memory side effects")) VKLOG("[RED] shader %016llx: geometry primitive claim declined: %s", (unsigned long long)key, why);
        }
    }
    // Bound graphics uniform shadow copies from the final base module, not the declared 64 KiB array size. Unknown uses stay full.
    // This is host analysis only: no shader code/bundle changes. External alternative modules bypass these bounds in cbuf_desc.
    if (stage >= 0 && stage < 3) {
        int any = 0; for (int i = 0; i < s->nbind; i++) any |= s->b[i].kind == RK_CBUF;
        if (any) { uint32_t ext[96]; uint8_t seen[16] = {0};
            vk_spv_ubo_extents(w, nw, (uint32_t)s->set, ext, 96);
            for (int i = 0; i < s->nbind; i++) { const dmsv_bind *b = &s->b[i];
                if (b->kind != RK_CBUF || b->reg < 0 || b->reg >= 16) continue;
                uint32_t n = b->desc == RD_UBO && b->count == 1 && b->binding < 96 ? ext[b->binding] : 0;
                if (!seen[b->reg]) { s->ubo_extent[b->reg] = n; seen[b->reg] = 1; }
                else if (!n || !s->ubo_extent[b->reg]) s->ubo_extent[b->reg] = 0;   // every alias of the guest slot must be bounded
                else if (n > s->ubo_extent[b->reg]) s->ubo_extent[b->reg] = n;
            }
        }
    }
    spv_io_locs(w, nw, &s->in_locs, &s->out_locs);
    VkShaderModuleCreateInfo ci = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, NULL, 0, (size_t)nw * 4, w};
    VkResult vr = vkCreateShaderModule(V.dev, &ci, NULL, &s->mod); s->spv = w; s->nspv = nw;
    if (vr) { VKLOG("[RED] shader %016llx: vkCreateShaderModule %d", (unsigned long long)key, vr); free(w); if (s->gs_claim_invocations) free((void *)s->b); free(s); nvn_hm_put(&g_shaders, &hk, (void *)1); pthread_mutex_unlock(&g_sh_mu); return NULL; }
    spv_runtime_dump(s, w, nw); // opt-in diagnostic, after every base-module rewrite
    // descriptor set layout: one entry per binding
    // The game's constant buffers and the driver segment are dynamic uniform buffers: their offset is given when the set is bound, so a set matches any draw's
    // with the same buffers (Unity updates a constant buffer before nearly every draw, and each update is a shadow at a new offset: NEO made a new set for
    // 94% of its draws, 5 ms a frame). Up to half the device's limit per stage, as the vertex and fragment sets share a pipeline layout. DM_VK_DYN_UBO=0: none.
    static int dyn_on = -1; if (dyn_on < 0) { const char *e = getenv("DM_VK_DYN_UBO"); dyn_on = !(e && !atoi(e)); }
    uint32_t dynmax = dyn_on ? V.props.limits.maxDescriptorSetUniformBuffersDynamic / 2 : 0;
    uint32_t sdmax = storage_dynamic_limit(stage, V.dyn_ssbo, &V.props.limits), nubo = 0, nsbuf = 0;
    if (V.dyn_ssbo) { static int noted; if (!noted++) VKLOG("dynamic storage descriptors enabled (DM_VK_DYN_SSBO, startup): max %u per graphics stage, alignment %llu; buffers and ranges unchanged",
        storage_dynamic_limit(0, 1, &V.props.limits), (unsigned long long)V.props.limits.minStorageBufferOffsetAlignment); }
    VkDescriptorSetLayoutBinding lb[96]; uint32_t nl = 0;
    for (int i = 0; i < s->nbind && nl < 96; i++) {
        const dmsv_bind *b = &s->b[i]; if (s->tbl[i]) continue;   // (in the table set)
        int du = b->desc == RD_UBO && (b->kind == RK_CBUF || b->kind == RK_DRV) && b->count == 1 && nubo < dynmax;
        int ds = storage_dynamic_binding(b, nsbuf, sdmax);
        s->dyn[i] = ds ? 2 : (uint8_t)du; s->ndyn += du || ds; nubo += du; nsbuf += ds;
        VkDescriptorType dt = ds ? VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC : du ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC : desc_type(b->desc);
        lb[nl++] = (VkDescriptorSetLayoutBinding){b->binding, dt, b->count ? b->count : 32, stage_flag(stage), NULL};
    }
    s->dsl = shader_layout(lb, nl);
    nvn_hm_put(&g_shaders, &hk, s);
    pthread_mutex_unlock(&g_sh_mu);
    return s;
}
// Integer texels. Maxwell's texture instructions return the texel's 32-bit words as they are, and the shader decides what they mean; the translator declares
// every sampled image as float, which matches only float views. A view of an integer format, or the stencil of a depth-stencil texture (DepthStencilMode
// STENCIL), must be read through a uint / int image in Vulkan, and Adreno converts the value otherwise. Xenoblade Chronicles 3 tags each pixel's material
// for its lighting from the stencil's bit 3 (shader 094f4afb9c876459 tests the raw word & 8): read as float, every tag was 0 and the lit scene was black.
// A variant of the module retypes those bindings' images to uint / int and bitcasts each texel read back to float, which gives the shader the raw words.
// Storage texel buffers too: a store (OpImageWrite) of a vec4 of floats into a retyped image is bitcast to the integer type first, so the words land as written.
// Final Fantasy Resonance (Unreal Engine 5.6) builds its particle index tables in R32UI buffers its compute shaders write through float images (812650f2ed1cba0d
// stores four floats of a uniform buffer); through the R32_UINT views its battle's haze and dust sprites (vertex shader fb570ef861debd17) drew nothing, with the
// variant they draw (a frame with that shader left out differs in those areas)
typedef struct { const uint32_t *w; uint32_t fn, bound, *at, *nt, nnt; struct { uint32_t old, kind, id; } map[64]; int nmap; } spvi;
static uint32_t spvi_find(const uint32_t *a, uint32_t lo, uint32_t hi, uint32_t op, const uint32_t *ops, uint32_t nops) {   // a declaration op with these operands (after the result id)
    for (uint32_t j = lo; j < hi;) { uint32_t m = a[j] >> 16; if (!m) break; if ((a[j] & 0xffff) == op && m == nops + 2 && !memcmp(&a[j + 2], ops, nops * 4)) return a[j + 1]; j += m; }
    return 0;
}
static uint32_t spvi_decl(spvi *c, uint32_t op, const uint32_t *ops, uint32_t nops, int reuse) {   // an existing declaration, or a new one
    uint32_t id = reuse ? spvi_find(c->w, 5, c->fn, op, ops, nops) : 0; if (!id && reuse) id = spvi_find(c->nt, 0, c->nnt, op, ops, nops);
    if (id || c->nnt + nops + 2 > 4096) return id;
    id = c->bound++; c->nt[c->nnt++] = (nops + 2) << 16 | op; c->nt[c->nnt++] = id; memcpy(&c->nt[c->nnt], ops, nops * 4); c->nnt += nops; return id;
}
static uint32_t spvi_scalar(spvi *c, uint32_t k) { uint32_t o[2] = {32, k - 1}; return spvi_decl(c, 21, o, 2, 1); }   // OpTypeInt 32, unsigned (kind 1) / signed (2)
static uint32_t spvi_vec4(spvi *c, uint32_t k) { uint32_t o[2] = {spvi_scalar(c, k), 4}; return spvi_decl(c, 23, o, 2, 1); }
// the type for old with its image's sampled type made uint / int: images, sampled images, arrays of them, pointers to those
static uint32_t spvi_type(spvi *c, uint32_t old, uint32_t k) {
    for (int j = 0; j < c->nmap; j++) if (c->map[j].old == old && c->map[j].kind == k) return c->map[j].id;
    uint32_t i = old < c->bound ? c->at[old] : 0; if (!i) return 0;
    uint32_t n = c->w[i] >> 16, op = c->w[i] & 0xffff, d[12]; if (n > 12 || n < 3) return 0; memcpy(d, &c->w[i + 2], (n - 2) * 4);
    if (op == 25) d[0] = spvi_scalar(c, k);                                                                          // OpTypeImage: the sampled type
    else if (op == 27 || op == 28 || op == 29) { if (!(d[0] = spvi_type(c, d[0], k))) return 0; }                  // OpTypeSampledImage, OpTypeArray, OpTypeRuntimeArray
    else if (op == 32 && n == 4) { if (!(d[1] = spvi_type(c, d[1], k))) return 0; }                                // OpTypePointer
    else return 0;
    uint32_t id = spvi_decl(c, op, d, n - 2, op != 28 && op != 29); if (!id) return 0;
    if (c->nmap < 64) c->map[c->nmap++] = (typeof(c->map[0])){old, k, id};
    return id;
}
static uint32_t *spv_int_images(const uint32_t *w, uint32_t nw, const uint16_t *bind, const uint8_t *kind, int nb, uint32_t set, uint32_t *onw) {
    spvi c = {.w = w, .fn = nw, .bound = w[3]};
    for (uint32_t i = 5; i < nw;) { uint32_t n = w[i] >> 16; if (!n) return NULL; if ((w[i] & 0xffff) == 54) { c.fn = i; break; } i += n; }   // the first OpFunction
    uint32_t base = c.bound; c.at = calloc(base, sizeof *c.at); c.nt = malloc(4096 * 4);
    uint8_t *taint = calloc(base, 1);   // ids that are (pointers to) the retyped images: their kind
    uint32_t *vset = malloc((size_t)base * 4); memset(vset, 0xff, (size_t)base * 4);   // each variable's DescriptorSet: the bindings named are the stage's set's (a texture table variable in set 0 can carry the same number)
    for (uint32_t i = 5; i < c.fn;) { uint32_t n = w[i] >> 16, op = w[i] & 0xffff; if (op == 71 && n >= 4 && w[i + 2] == 34 && w[i + 1] < base) vset[w[i + 1]] = w[i + 3]; i += n; }
    for (uint32_t i = 5; i < c.fn;) { uint32_t n = w[i] >> 16, op = w[i] & 0xffff;
        if (op >= 19 && op <= 39 && w[i + 1] < base) c.at[w[i + 1]] = i;   // OpType*
        if (op == 71 && n >= 4 && w[i + 2] == 33) for (int k = 0; k < nb; k++) if (w[i + 3] == bind[k] && w[i + 1] < base && vset[w[i + 1]] == set) taint[w[i + 1]] = kind[k];   // OpDecorate Binding
        i += n; }
    free(vset);
    // the functions, rewritten apart: every new type they need is known before the declarations are written
    uint32_t *fb = malloc(((size_t)(nw - c.fn) * 2 + 16) * 4), nf = 0, *mv = malloc(1024 * 4), nmv = 0; int ok = 1;
    for (uint32_t i = 5; i < c.fn && ok;) { uint32_t n = w[i] >> 16;   // the retyped variables, with their new pointer types
        if ((w[i] & 0xffff) == 59 && w[i + 2] < base && taint[w[i + 2]]) { uint32_t t = spvi_type(&c, w[i + 1], taint[w[i + 2]]); if (!t || nmv + n > 1024) ok = 0; else { memcpy(&mv[nmv], &w[i], n * 4); mv[nmv + 1] = t; nmv += n; } }
        i += n; }
    for (uint32_t i = c.fn; i < nw && ok;) { uint32_t n = w[i] >> 16, op = w[i] & 0xffff; if (!n) { ok = 0; break; }
        uint32_t *x = &fb[nf]; memcpy(x, &w[i], n * 4); nf += n;
        uint32_t src = n >= 4 ? w[i + 3] : 0, k = src && src < base ? taint[src] : 0;
        if (op == 99 && n >= 4 && w[i + 1] < base && taint[w[i + 1]]) {   // ImageWrite image coord texel: the texel (a vec4 of floats) bitcast to the integer vec4
            uint32_t kk = taint[w[i + 1]], tx = w[i + 3], tt = 0;
            for (uint32_t j = c.fn; j < nw;) { uint32_t m = w[j] >> 16; if (!m) break; if (m >= 3 && w[j + 2] == tx && (w[j] & 0xffff) != 71 && (w[j] & 0xffff) != 72) { tt = w[j + 1]; break; } j += m; }
            uint32_t ti = tt && tt < base ? c.at[tt] : 0, ci = ti && (w[ti] & 0xffff) == 23 && w[ti] >> 16 == 4 && w[ti + 3] == 4 && w[ti + 2] < base ? c.at[w[ti + 2]] : 0;
            if (ci && (w[ci] & 0xffff) == 21) { i += n; continue; }   // integer texels already (an image the translator typed uint): as it is
            if (!ci || (w[ci] & 0xffff) != 22 || w[ci + 2] != 32) { ok = 0; break; }   // not a vec4 of 32-bit floats: no variant
            uint32_t r2 = c.bound++; nf -= n; fb[nf++] = 4u << 16 | 124; fb[nf++] = spvi_vec4(&c, kk); fb[nf++] = r2; fb[nf++] = tx;   // OpBitcast
            x = &fb[nf]; memcpy(x, &w[i], n * 4); x[3] = r2; nf += n;
        } else if (k && (op == 61 || op == 65 || op == 66 || op == 83 || op == 100 || op == 86)) {   // Load, AccessChain, InBoundsAccessChain, CopyObject, Image, SampledImage
            uint32_t t = spvi_type(&c, x[1], k); if (!t) { ok = 0; break; } x[1] = t; if (x[2] < base) taint[x[2]] = (uint8_t)k;
        } else if (k && (op == 87 || op == 88 || op == 91 || op == 92 || op == 95 || op == 96 || op == 98)) {   // Sample{Implicit,Explicit}Lod, SampleProj*, Fetch, Gather, Read: a vec4 of the integer type, bitcast back
            uint32_t ft = x[1], r = x[2], r2 = c.bound++; x[1] = spvi_vec4(&c, k); x[2] = r2;
            fb[nf++] = 4u << 16 | 124; fb[nf++] = ft; fb[nf++] = r; fb[nf++] = r2;   // OpBitcast
        }
        i += n; }
    uint32_t *o = NULL, no = 5;
    if (ok && c.nnt < 4096) {
        o = malloc(((size_t)c.fn + c.nnt + nmv + nf + 8) * 4); memcpy(o, w, 20); o[3] = c.bound;
        for (uint32_t i = 5; i < c.fn;) { uint32_t n = w[i] >> 16; if (!((w[i] & 0xffff) == 59 && w[i + 2] < base && taint[w[i + 2]])) { memcpy(&o[no], &w[i], n * 4); no += n; } i += n; }
        memcpy(&o[no], c.nt, c.nnt * 4); no += c.nnt; memcpy(&o[no], mv, nmv * 4); no += nmv; memcpy(&o[no], fb, nf * 4); no += nf;
        *onw = no;
    }
    free(c.at); free(c.nt); free(taint); free(fb); free(mv);
    return o;
}
// A variant of a fragment shader that writes gl_FragDepth = gl_FragCoord.z before each return of its entry point (DepthReplacing): the same depth, tested after the
// shader instead of by Adreno's early culling (gfx_vk_exec.c fdz says when). NULL when the shader writes its depth already or the module is not as expected
static uint32_t *spv_frag_depth(const uint32_t *w, uint32_t nw, uint32_t *onw) {
    spvi c = {.w = w, .fn = nw, .bound = w[3]};
    uint32_t ep = 0, epfn = 0, firsttype = 0, coord = 0, c2 = 0;
    for (uint32_t i = 5; i < nw;) { uint32_t n = w[i] >> 16, op = w[i] & 0xffff; if (!n || i + n > nw) return NULL;
        if (op == 15) { if (ep || n < 3 || w[i + 1] != 4) return NULL; ep = i; epfn = w[i + 2]; }   // one entry point, a Fragment one
        if (op == 71 && n >= 4 && w[i + 2] == 11 && w[i + 3] == 22) return NULL;   // BuiltIn FragDepth: written already
        if (op == 71 && n >= 4 && w[i + 2] == 11 && w[i + 3] == 15) coord = w[i + 1];   // BuiltIn FragCoord
        if (!firsttype && ((op >= 19 && op <= 39) || (op >= 41 && op <= 52) || op == 59)) firsttype = i;   // the first type, constant or variable: the annotations end here
        if (op == 54) { c.fn = i; break; }   // the first OpFunction
        i += n; }
    if (!ep || !firsttype || c.fn == nw) return NULL;
    c.nt = malloc(4096 * 4);
    uint32_t o32[1] = {32}; uint32_t f32 = spvi_decl(&c, 22, o32, 1, 1), i32 = spvi_scalar(&c, 2);
    uint32_t ov4[2] = {f32, 4}; uint32_t v4 = spvi_decl(&c, 23, ov4, 2, 1);
    uint32_t op1[2] = {1, v4}, op2[2] = {1, f32}, op3[2] = {3, f32}; uint32_t pin4 = spvi_decl(&c, 32, op1, 2, 1), pinf = spvi_decl(&c, 32, op2, 2, 1), poutf = spvi_decl(&c, 32, op3, 2, 1);
    if (!f32 || !i32 || !v4 || !pin4 || !pinf || !poutf || c.nnt + 16 > 4096) { free(c.nt); return NULL; }
    for (uint32_t i = 5; i < c.fn;) { uint32_t n = w[i] >> 16; if ((w[i] & 0xffff) == 43 && n == 4 && w[i + 1] == i32 && w[i + 3] == 2) c2 = w[i + 2]; i += n; }   // OpConstant int 2
    if (!c2) { c2 = c.bound++; c.nt[c.nnt++] = 4u << 16 | 43; c.nt[c.nnt++] = i32; c.nt[c.nnt++] = c2; c.nt[c.nnt++] = 2; }
    int newcoord = !coord; if (newcoord) { coord = c.bound++; c.nt[c.nnt++] = 4u << 16 | 59; c.nt[c.nnt++] = pin4; c.nt[c.nnt++] = coord; c.nt[c.nnt++] = 1; }   // OpVariable Input
    uint32_t dv = c.bound++; c.nt[c.nnt++] = 4u << 16 | 59; c.nt[c.nnt++] = poutf; c.nt[c.nnt++] = dv; c.nt[c.nnt++] = 3;   // OpVariable Output
    uint32_t nret = 0; for (uint32_t i = c.fn; i < nw;) { uint32_t n = w[i] >> 16; if (!n) { free(c.nt); return NULL; } nret += (w[i] & 0xffff) == 253; i += n; }
    uint32_t *o = malloc(((size_t)nw + c.nnt + 16 + 12 * nret) * 4), no = 5; memcpy(o, w, 20);
    for (uint32_t i = 5; i < c.fn;) { uint32_t n = w[i] >> 16;
        if (i == firsttype) { uint32_t d[8] = {4u << 16 | 71, dv, 11, 22, 4u << 16 | 71, coord, 11, 15}; memcpy(&o[no], d, (newcoord ? 8 : 4) * 4); no += newcoord ? 8 : 4; }   // BuiltIn FragDepth (FragCoord)
        if (i == ep) { o[no++] = (n + 1 + newcoord) << 16 | 15; memcpy(&o[no], &w[i + 1], (n - 1) * 4); no += n - 1; o[no++] = dv; if (newcoord) o[no++] = coord;   // the entry point lists them
                       o[no++] = 3u << 16 | 16; o[no++] = epfn; o[no++] = 12; }   // OpExecutionMode DepthReplacing
        else { memcpy(&o[no], &w[i], n * 4); no += n; }
        i += n; }
    memcpy(&o[no], c.nt, c.nnt * 4); no += c.nnt;
    int in_ep = 0;
    for (uint32_t i = c.fn; i < nw;) { uint32_t n = w[i] >> 16, op = w[i] & 0xffff;
        if (op == 54) in_ep = w[i + 2] == epfn; else if (op == 56) in_ep = 0;   // OpFunction / OpFunctionEnd
        if (op == 253 && in_ep) { uint32_t a = c.bound++, b = c.bound++, x[12] = {5u << 16 | 65, pinf, a, coord, c2, 4u << 16 | 61, f32, b, a, 3u << 16 | 62, dv, b};   // AccessChain .z, Load, Store
                                  memcpy(&o[no], x, sizeof x); no += 12; }
        memcpy(&o[no], &w[i], n * 4); no += n; i += n; }
    o[3] = c.bound; *onw = no; free(c.nt); return o;
}
// Maxwell's fixed-function alpha test (nvnColorStateSetAlphaTest, nvnCommandBufferSetAlphaRef; rendering.md "The alpha test in the Vulkan replay"): a variant of a fragment
// shader that compares the alpha it wrote to colour 0 (the vec4 float output at location 0) with the reference before each return of its entry point, and kills the
// fragment when the comparison fails. The function (NVN_ALPHA_FUNC_*, 1 NEVER .. 8 ALWAYS) and the reference are specialization constants VK_ATEST_SPEC_FUNC and
// VK_ATEST_SPEC_REF (gfx_vk_exec.c build_pipeline gives them from the pipeline key), so one module serves every function and reference. Comparisons as dmsh_alpha.py's
// (Metal): ordered, but NOTEQUAL passes a NaN alpha. NULL when the shader has no such output (no colour 0, an integer one, the dual-source second output)
static uint32_t *spv_alpha_test(const uint32_t *w, uint32_t nw, uint32_t *onw) {
    spvi c = {.w = w, .fn = nw, .bound = w[3]};
    uint32_t ep = 0, epfn = 0, firsttype = 0, bound = w[3];
    for (uint32_t i = 5; i < nw;) { uint32_t n = w[i] >> 16, op = w[i] & 0xffff; if (!n || i + n > nw) return NULL;
        if (op == 15) { if (ep || n < 3 || w[i + 1] != 4) return NULL; ep = i; epfn = w[i + 2]; }   // one entry point, a Fragment one
        if (!firsttype && ((op >= 19 && op <= 39) || (op >= 41 && op <= 52) || op == 59)) firsttype = i;
        if (op == 54) { c.fn = i; break; }
        i += n; }
    if (!ep || !firsttype || c.fn == nw) return NULL;
    // colour 0: an Output variable decorated Location 0 (not Index 1) whose type is a pointer to a vec4 of 32-bit floats
    uint8_t *loc0 = calloc(bound, 1), *idx1 = calloc(bound, 1); uint32_t out = 0;
    for (uint32_t i = 5; i < firsttype;) { uint32_t n = w[i] >> 16, op = w[i] & 0xffff;
        if (op == 71 && n == 4 && w[i + 1] < bound && w[i + 2] == 30 && w[i + 3] == 0) loc0[w[i + 1]] = 1;
        if (op == 71 && n == 4 && w[i + 1] < bound && w[i + 2] == 32 && w[i + 3] == 1) idx1[w[i + 1]] = 1;
        i += n; }
    for (uint32_t i = firsttype; i < c.fn && !out;) { uint32_t n = w[i] >> 16, op = w[i] & 0xffff;
        if (op == 59 && n >= 4 && w[i + 3] == 3 && w[i + 2] < bound && loc0[w[i + 2]] && !idx1[w[i + 2]]) {   // OpVariable Output at location 0
            uint32_t pt = w[i + 1], vt = 0, ok = 0;
            for (uint32_t j = firsttype; j < c.fn;) { uint32_t m = w[j] >> 16; if ((w[j] & 0xffff) == 32 && m == 4 && w[j + 1] == pt && w[j + 2] == 3) vt = w[j + 3]; j += m; }
            for (uint32_t j = firsttype; vt && j < c.fn;) { uint32_t m = w[j] >> 16; if ((w[j] & 0xffff) == 23 && m == 4 && w[j + 1] == vt && w[j + 3] == 4) {
                    uint32_t ct = w[j + 2]; for (uint32_t q = firsttype; q < c.fn;) { uint32_t r = w[q] >> 16; if ((w[q] & 0xffff) == 22 && r == 3 && w[q + 1] == ct && w[q + 2] == 32) ok = 1; q += r; } }
                j += m; }
            if (ok) out = w[i + 2]; }
        i += n; }
    free(loc0); free(idx1);
    if (!out) return NULL;
    c.nt = malloc(4096 * 4);
    uint32_t o32[1] = {32}, ou[2] = {32, 0}; uint32_t f32 = spvi_decl(&c, 22, o32, 1, 1), u32 = spvi_decl(&c, 21, ou, 2, 1), bt = spvi_decl(&c, 20, NULL, 0, 1);
    uint32_t opf[2] = {3, f32}; uint32_t poutf = spvi_decl(&c, 32, opf, 2, 1);
    if (!f32 || !u32 || !bt || !poutf || c.nnt + 64 > 4096) { free(c.nt); return NULL; }
    #define K(op_, ty_, val_) ({ uint32_t id_ = c.bound++; c.nt[c.nnt++] = 4u << 16 | (op_); c.nt[c.nnt++] = (ty_); c.nt[c.nnt++] = id_; c.nt[c.nnt++] = (val_); id_; })
    uint32_t cu[8]; for (uint32_t v = 1; v < 8; v++) cu[v] = K(43, u32, v);   // OpConstant uint 1..7 (3 is also the alpha's index)
    uint32_t ct = c.bound++; c.nt[c.nnt++] = 3u << 16 | 41; c.nt[c.nnt++] = bt; c.nt[c.nnt++] = ct;   // OpConstantTrue
    uint32_t cf = c.bound++; c.nt[c.nnt++] = 3u << 16 | 42; c.nt[c.nnt++] = bt; c.nt[c.nnt++] = cf;   // OpConstantFalse
    uint32_t sfn = K(50, u32, 8), sref = K(50, f32, 0);   // OpSpecConstant: the function (ALWAYS unless given) and the reference
    #undef K
    uint32_t nret = 0; for (uint32_t i = c.fn; i < nw;) { uint32_t n = w[i] >> 16; if (!n) { free(c.nt); return NULL; } nret += (w[i] & 0xffff) == 253; i += n; }
    enum { PER_RET = 5 + 4 + 6 * 5 + 7 * 5 + 7 * 6 + 3 + 4 + 2 + 1 + 2 };
    uint32_t *o = malloc(((size_t)nw + c.nnt + 8 + (size_t)PER_RET * nret) * 4), no = 5; memcpy(o, w, 20);
    for (uint32_t i = 5; i < c.fn;) { uint32_t n = w[i] >> 16;
        if (i == firsttype) { uint32_t d[8] = {4u << 16 | 71, sfn, 1, VK_ATEST_SPEC_FUNC, 4u << 16 | 71, sref, 1, VK_ATEST_SPEC_REF}; memcpy(&o[no], d, sizeof d); no += 8; }   // SpecId
        memcpy(&o[no], &w[i], n * 4); no += n; i += n; }
    memcpy(&o[no], c.nt, c.nnt * 4); no += c.nnt;
    int in_ep = 0;
    for (uint32_t i = c.fn; i < nw;) { uint32_t n = w[i] >> 16, op = w[i] & 0xffff;
        if (op == 54) in_ep = w[i + 2] == epfn; else if (op == 56) in_ep = 0;
        if (op == 253 && in_ep) {
            uint32_t p = c.bound++, a = c.bound++, x[16], ne = 0;
            o[no++] = 5u << 16 | 65; o[no++] = poutf; o[no++] = p; o[no++] = out; o[no++] = cu[3];   // OpAccessChain colour 0 .w
            o[no++] = 4u << 16 | 61; o[no++] = f32; o[no++] = a; o[no++] = p;                         // OpLoad
            static const uint32_t cmp[8] = {0, 0, 184, 180, 188, 186, 183, 190};   // LESS FOrdLessThan, EQUAL FOrdEqual, LEQUAL FOrdLessThanEqual, GREATER FOrdGreaterThan, NOTEQUAL FUnordNotEqual, GEQUAL FOrdGreaterThanEqual
            for (uint32_t f = 2; f <= 7; f++) { x[f] = c.bound++; o[no++] = 5u << 16 | cmp[f]; o[no++] = bt; o[no++] = x[f]; o[no++] = a; o[no++] = sref; }
            for (uint32_t f = 1; f <= 7; f++) { x[8 + f] = c.bound++; o[no++] = 5u << 16 | 170; o[no++] = bt; o[no++] = x[8 + f]; o[no++] = sfn; o[no++] = cu[f]; }   // OpIEqual func, f
            uint32_t r = ct; for (uint32_t f = 7; f >= 1; f--) { uint32_t t = c.bound++; o[no++] = 6u << 16 | 169; o[no++] = bt; o[no++] = t; o[no++] = x[8 + f]; o[no++] = f == 1 ? cf : x[f]; o[no++] = r; r = t; }   // OpSelect
            uint32_t kl = c.bound++, ml = c.bound++; (void)ne;
            o[no++] = 3u << 16 | 247; o[no++] = ml; o[no++] = 0;              // OpSelectionMerge
            o[no++] = 4u << 16 | 250; o[no++] = r; o[no++] = ml; o[no++] = kl;   // OpBranchConditional pass → merge, else kill
            o[no++] = 2u << 16 | 248; o[no++] = kl; o[no++] = 1u << 16 | 252;    // OpLabel, OpKill
            o[no++] = 2u << 16 | 248; o[no++] = ml; }                            // OpLabel, then the OpReturn
        memcpy(&o[no], &w[i], n * 4); no += n; i += n; }
    o[3] = c.bound; *onw = no; free(c.nt); return o;
}
// Runtime-normalized ALT probes must retain the exact existing resource/type preamble.
// Only fresh declarations needed by the offline intermediate-output tool may be added.
// This is intentionally stricter than structural type equivalence: these diagnostics
// preserve original IDs. A bundle module with unshrunk/remapped pools is rejected.
// A configured list is all-or-nothing: a malformed suffix must not retain an
// earlier matching key. Empty preserves the explicit-directory, bounded dump mode.
static int spv_runtime_dump_match(const char *list, uint64_t wanted) {
    if (!list || !*list) return 1;
    const char *p = list; unsigned count = 0; int matched = 0;
    for (;;) {
        if (++count > 32) return 0;
        uint64_t key = 0;
        for (unsigned i = 0; i < 16; i++) {
            unsigned c = (unsigned char)*p, v;
            if (c >= '0' && c <= '9') v = c - '0';
            else if (c >= 'a' && c <= 'f') v = c - 'a' + 10;
            else if (c >= 'A' && c <= 'F') v = c - 'A' + 10;
            else return 0;
            key = key << 4 | v; p++;
        }
        matched |= key == wanted;
        if (!*p) return matched;
        if (*p++ != ',' || !*p) return 0;
    }
}
static int spv_runtime_extra(const uint32_t *w, uint32_t n, uint32_t oldbound, uint32_t bound) {
    uint32_t op = w[0] & 0xffff, id = 0;
    if ((op == 21 && n == 4) || (op == 22 && n == 3) || (op == 23 && n == 4)) id = w[1];
    else if (op == 32 && n == 4 && (w[2] == 1 || w[2] == 6)) id = w[1]; // Input/Private pointer type only
    else if ((op == 43 && n >= 4) || (op == 44 && n >= 4)) id = w[2];
    else if (op == 59 && (n == 4 || n == 5) && w[3] == 6) id = w[2]; // Private variable, no descriptor
    else return 0;
    return id >= oldbound && id < bound;
}
static int spv_runtime_private(const uint32_t *w, uint32_t nw, uint32_t id, uint32_t oldbound) {
    if (id < oldbound || id >= w[3]) return 0;
    for (uint32_t i = 5; i < nw;) {
        uint32_t n = w[i] >> 16, op = w[i] & 0xffff;
        if (!n || n > nw - i || op == 54) break;
        if (op == 59 && n >= 4 && w[i + 2] == id) return w[i + 3] == 6;
        i += n;
    }
    return 0;
}
// Trusted diagnostics may add exactly one float4 colour at location1, matching
// the existing location0 type. No other interface/resource change is permitted.
static uint32_t spv_runtime_mrt(const uint32_t *base, uint32_t bn, const uint32_t *alt, uint32_t an) {
    uint32_t out = 0, pointer = 0, vector = 0, scalar = 0, added = 0;
    unsigned outputs = 0, entries = 0, locations = 0, declarations = 0;
    for (uint32_t i = 5; i < bn;) {
        uint32_t n = base[i] >> 16, op = base[i] & 0xffff;
        if (!n || n > bn - i) return 0;
        if (op == 54) break;
        if (op == 15) { if (n < 4 || base[i + 1] != 4) return 0; entries++; }
        if (op == 59 && n >= 4 && base[i + 3] == 3) { outputs++; out = base[i + 2]; pointer = base[i + 1]; }
        i += n;
    }
    if (entries != 1 || outputs != 1) return 0;
    int location0 = 0;
    for (uint32_t i = 5; i < bn;) {
        uint32_t n = base[i] >> 16, op = base[i] & 0xffff;
        if (!n || n > bn - i) return 0;
        if (op == 54) break;
        if (op == 71 && n >= 4 && base[i + 1] == out) {
            if (base[i + 2] == 30 && base[i + 3] == 0) location0++;
            else return 0; // no Index/component/other output contract
        }
        if (op == 32 && n == 4 && base[i + 1] == pointer && base[i + 2] == 3) vector = base[i + 3];
        i += n;
    }
    if (location0 != 1 || !vector) return 0;
    for (uint32_t i = 5; i < bn;) {
        uint32_t n = base[i] >> 16, op = base[i] & 0xffff;
        if (op == 54) break;
        if (op == 23 && n == 4 && base[i + 1] == vector && base[i + 3] == 4) scalar = base[i + 2];
        i += n;
    }
    int float32 = 0;
    for (uint32_t i = 5; i < bn;) {
        uint32_t n = base[i] >> 16, op = base[i] & 0xffff;
        if (op == 54) break;
        if (op == 22 && n == 3 && base[i + 1] == scalar && base[i + 2] == 32) float32 = 1;
        i += n;
    }
    if (!float32) return 0;
    for (uint32_t i = 5; i < an;) {
        uint32_t n = alt[i] >> 16, op = alt[i] & 0xffff;
        if (!n || n > an - i) return 0;
        if (op == 54) break;
        if (op == 59 && n >= 4 && alt[i + 3] == 3 && alt[i + 2] >= base[3]) {
            if (++declarations != 1 || n != 4 || alt[i + 1] != pointer || alt[i + 2] >= alt[3]) return 0;
            added = alt[i + 2];
        }
        i += n;
    }
    if (!added) return 0;
    for (uint32_t i = 5; i < an;) {
        uint32_t n = alt[i] >> 16, op = alt[i] & 0xffff;
        if (op == 54) break;
        if (op == 71 && n >= 2 && alt[i + 1] == added) {
            if (n != 4 || alt[i + 2] != 30 || alt[i + 3] != 1) return 0;
            locations++;
        }
        i += n;
    }
    return locations == 1 ? added : 0;
}

static int spv_runtime_contract(const uint32_t *base, uint32_t bn, const uint32_t *alt, uint32_t an) {
    if (!base || !alt || bn < 5 || an < 5 || base[0] != 0x07230203u || alt[0] != base[0] ||
        alt[1] != base[1] || alt[2] != base[2] || alt[4] != base[4] || alt[3] < base[3]) return 0;
    // Validate every instruction boundary, including the function section.
    int in_function = 0, functions = 0;
    for (uint32_t i = 5; i < an;) { uint32_t n = alt[i] >> 16, op = alt[i] & 0xffff;
        if (!n || n > an - i) return 0;
        if (op == 54) { if (n != 5 || in_function) return 0; in_function = 1; functions++; }
        if (op == 56) { if (n != 1 || !in_function) return 0; in_function = 0; }
        i += n; }
    if (in_function || !functions) return 0;
    uint32_t mrt = spv_runtime_mrt(base, bn, alt, an), mrt_seen = 0;
    uint32_t b = 5, a = 5;
    while (a < an) {
        uint32_t na = alt[a] >> 16, oa = alt[a] & 0xffff;
        if (b >= bn) return 0;
        uint32_t nb = base[b] >> 16, ob = base[b] & 0xffff;
        if (!nb || nb > bn - b) return 0;
        if (oa == 54) return ob == 54 && (!mrt || mrt_seen == 7); // original declarations and complete optional MRT
        if (na == nb && !memcmp(alt + a, base + b, (size_t)nb * 4)) { a += na; b += nb; continue; }
        // A diagnostic MRT is an Output interface in every version; Private
        // globals are interface operands only in SPIR-V1.4 and later.
        if (oa == 15 && ob == 15 && na > nb &&
            !memcmp(alt + a + 1, base + b + 1, (size_t)(nb - 1) * 4)) {
            for (uint32_t j = nb; j < na; j++) {
                if (mrt && alt[a + j] == mrt) { if (mrt_seen & 1) return 0; mrt_seen |= 1; }
                else if (base[1] < 0x00010400u || !spv_runtime_private(alt, an, alt[a + j], base[3])) return 0;
            }
            a += na; b += nb; continue;
        }
        if (mrt && oa == 71 && na == 4 && alt[a + 1] == mrt && alt[a + 2] == 30 && alt[a + 3] == 1) {
            if (mrt_seen & 2) return 0;
            mrt_seen |= 2; a += na; continue;
        }
        if (mrt && oa == 59 && na == 4 && alt[a + 2] == mrt && alt[a + 3] == 3) {
            if (mrt_seen & 4) return 0;
            mrt_seen |= 4; a += na; continue;
        }
        if (!spv_runtime_extra(alt + a, na, base[3], alt[3])) return 0;
        a += na;
    }
    return 0;
}
// Separate, opt-in paired observer contract. Preserve every original declaration;
// only Flat uint location12 and (fragment only) uvec4 colour1 may extend the IO.
static const uint32_t *spv_observer_decl(const uint32_t *w, uint32_t nw, uint32_t id, unsigned opcode) {
    for (uint32_t i=5;i<nw;) { uint32_t n=w[i]>>16, op=w[i]&0xffff;
        if (!n || n>nw-i || op==54) return NULL;
        if (op==opcode && n>=3 && w[i+1]==id) return w+i;
        i+=n;
    }
    return NULL;
}
static int spv_observer_uint(const uint32_t *w,uint32_t nw,uint32_t id,int vector) {
    if(vector){const uint32_t *v=spv_observer_decl(w,nw,id,23);
        if(!v || v[0]>>16!=4 || v[3]!=4)return 0;id=v[2];}
    const uint32_t *t=spv_observer_decl(w,nw,id,21);
    return t && t[0]>>16==4 && t[2]==32 && t[3]==0;
}
static int spv_observer_contract(const uint32_t *base,uint32_t bn,const uint32_t *alt,uint32_t an,int fragment) {
    if(!base||!alt||bn<5||an<5||alt[3]<base[3])return 0;
    uint32_t ids[2]={0},ptr[2]={0},seen[2]={0};unsigned nids=fragment?2:1,entries=0;
    for(uint32_t i=5;i<bn;){uint32_t n=base[i]>>16,op=base[i]&0xffff;
        if(!n||n>bn-i)return 0;if(op==54)break;
        if(op==15){if(n<4||base[i+1]!=(fragment?4u:0u))return 0;entries++;}
        if(op==71&&n==4&&base[i+2]==30){
            if(base[i+3]==12)return 0;
            if(fragment&&base[i+3]==1)for(uint32_t j=5;j<bn;){uint32_t m=base[j]>>16;
                if(!m||m>bn-j)return 0;if((base[j]&0xffff)==54)break;
                if((base[j]&0xffff)==59&&m>=4&&base[j+2]==base[i+1]&&base[j+3]==3)return 0;
                j+=m;}
        }
        i+=n;
    }
    if(entries!=1)return 0;
    for(uint32_t i=5;i<an;){uint32_t n=alt[i]>>16,op=alt[i]&0xffff;
        if(!n||n>an-i)return 0;if(op==54)break;
        if(op==59&&n>=4&&alt[i+2]>=base[3]&&(alt[i+3]==1||alt[i+3]==3)){
            unsigned slot=fragment&&alt[i+3]==3?1:0;
            if(n!=4||slot>=nids||ids[slot]||alt[i+2]>=alt[3]||
                alt[i+3]!=(slot?3u:fragment?1u:3u))return 0;
            const uint32_t *t=spv_observer_decl(alt,an,alt[i+1],32);
            if(!t||t[0]>>16!=4||t[2]!=alt[i+3]||!spv_observer_uint(alt,an,t[3],slot==1))return 0;
            ids[slot]=alt[i+2];ptr[slot]=alt[i+1];
        }i+=n;
    }
    if(!ids[0]||(fragment&&!ids[1]))return 0;
    uint32_t *clean=malloc((size_t)an*4);if(!clean)return 0;
    memcpy(clean,alt,20);uint32_t out=5;int ok=1;
    for(uint32_t i=5;i<an&&ok;){uint32_t n=alt[i]>>16,op=alt[i]&0xffff;int strip=0;
        if(!n||n>an-i){ok=0;break;}
        if(op==54){memcpy(clean+out,alt+i,(size_t)(an-i)*4);out+=an-i;break;}
        for(unsigned k=0;k<nids;k++){
            if(op==71&&n>=3&&alt[i+1]==ids[k]){
                unsigned bit=0;
                if(n==4&&alt[i+2]==30&&alt[i+3]==(k?1u:12u))bit=1;
                if(!k&&n==3&&alt[i+2]==14)bit=2; // Flat
                if(!bit||(seen[k]&bit)){ok=0;break;}seen[k]|=bit;strip=1;
            }
            if(op==59&&n==4&&alt[i+2]==ids[k])strip=1;
            if(op==32&&n==4&&alt[i+1]==ptr[k]&&ptr[k]>=base[3])strip=1;
        }
        if(op==15){
            uint32_t start=out;clean[out++]=alt[i]; // retain original entry operands and only omit the added IO IDs
            uint32_t interface=3;while(interface<n){uint32_t word=alt[i+interface++];
                if(!(word&0xffu)||!(word&0xff00u)||!(word&0xff0000u)||!(word&0xff000000u))break;}
            for(uint32_t j=1;j<n;j++){int added=0;for(unsigned k=0;k<nids;k++)if(j>=interface&&alt[i+j]==ids[k]){
                if(seen[k]&4){ok=0;break;}seen[k]|=4;added=1;}
                if(!added)clean[out++]=alt[i+j];
            }
            clean[start]=(out-start)<<16|15;
        }else if(!strip){memcpy(clean+out,alt+i,(size_t)n*4);out+=n;}
        i+=n;
    }
    for(unsigned k=0;k<nids;k++)if(seen[k]!=(k?5u:7u))ok=0;
    if(ok)ok=spv_runtime_contract(base,bn,clean,out);
    free(clean);return ok;
}
// The pair has one immutable process-lifetime identity (pkey.alt=2). Neither half
// enters the generic per-shader ALT cache, and a partial failure publishes neither.
int vk_shader_observer_pair(vkshader_t *vs,vkshader_t *fs,VkShaderModule modules[2]) {
    static int tried,ready;static vkshader_t *bases[2];static VkShaderModule pair[2];
    if(!vs||!fs||vs->key!=0xc8ab0e758c17df4aull||fs->key!=0x96dbc0392691b437ull)return 0;
    const char *enable=getenv("DM_VK_FLAT_OBSERVER"),*dir=getenv("DM_VK_FLAT_OBSERVER_DIR");
    if(!enable||strcmp(enable,"1")||!dir||!*dir)return 0;
    pthread_mutex_lock(&g_sh_mu);
    if(!tried){
        tried=1;uint32_t *w[2]={0},nw[2]={0};vkshader_t *s[2]={vs,fs};VkShaderModule made[2]={0};int ok=1;
        for(unsigned i=0;i<2&&ok;i++){
            char path[768];if(snprintf(path,sizeof path,"%s/%016llx.spv",dir,(unsigned long long)s[i]->key)>=(int)sizeof path){ok=0;break;}
            FILE *f=fopen(path,"rb");if(!f){ok=0;break;}
            if(fseek(f,0,SEEK_END)){fclose(f);ok=0;break;}long n=ftell(f);rewind(f);
            if(n<=20||n>(16L<<20)||(n&3)){fclose(f);ok=0;break;}
            w[i]=malloc((size_t)n);nw[i]=(uint32_t)(n/4);
            ok=w[i]&&fread(w[i],1,(size_t)n,f)==(size_t)n;fclose(f);
            if(ok)ok=spv_observer_contract(s[i]->spv,s[i]->nspv,w[i],nw[i],i==1);
        }
        for(unsigned i=0;i<2&&ok;i++){VkShaderModuleCreateInfo ci={VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,NULL,0,(size_t)nw[i]*4,w[i]};
            ok=vkCreateShaderModule(V.dev,&ci,NULL,&made[i])==VK_SUCCESS;
        }
        if(ok){for(unsigned i=0;i<2;i++){pair[i]=made[i];bases[i]=s[i];unsigned char hash[CC_SHA256_DIGEST_LENGTH];char hex[65];
                CC_SHA256(w[i],(CC_LONG)nw[i]*4,hash);for(unsigned j=0;j<sizeof hash;j++)snprintf(hex+2*j,3,"%02x",hash[j]);
                VKLOG("Flat observer paired module: key=%016llx sha256=%s words=%u identity=2",(unsigned long long)s[i]->key,hex,nw[i]);}ready=1;
        }else{for(unsigned i=0;i<2;i++)if(made[i])vkDestroyShaderModule(V.dev,made[i],NULL);
            VKLOG("Flat observer: pair refused; both original modules retained");}
        free(w[0]);free(w[1]);
    }
    int ok=ready&&bases[0]==vs&&bases[1]==fs;
    if(ok&&modules){modules[0]=pair[0];modules[1]=pair[1];}
    pthread_mutex_unlock(&g_sh_mu);return ok;
}
static void spv_runtime_dump(vkshader_t *s, const uint32_t *w, uint32_t nw) {
    const char *dir = getenv("DM_VK_SPV_DUMP_DIR");
    static unsigned count;
    if (!dir || !*dir) return; // normal path: no key lookup, parsing or file work
    const char *key = getenv("DM_VK_SPV_DUMP_KEY");
    if (count >= (key && *key ? 32u : 16u) || !spv_runtime_dump_match(key, s->key)) return;
    count++;
    char path[768], part[800], meta[800];
    if (snprintf(path, sizeof path, "%s/%016llx.spv", dir, (unsigned long long)s->key) >= (int)sizeof path) return;
    snprintf(part, sizeof part, "%s.part", path);
    FILE *f = fopen(part, "wb");
    if (!f) { VKLOG("[RED] runtime SPIR-V dump could not open %s", part); return; }
    int ok = fwrite(w, 4, nw, f) == nw;
    if (fclose(f)) ok = 0;
    if (!ok || rename(part, path)) { VKLOG("[RED] runtime SPIR-V dump incomplete: %s", part); return; }
    snprintf(meta, sizeof meta, "%s.json", path);
    f = fopen(meta, "w");
    if (f) { fprintf(f, "{\n  \"schema\": \"vk-runtime-spv-v1\",\n  \"key\": \"%016llx\",\n  \"stage\": %d,\n  \"set\": %d,\n  \"word_count\": %u,\n  \"pool_arrays\": %d,\n  \"pool_textures\": %d,\n  \"pool_samplers\": %d,\n  \"requires_alt_runtime\": true\n}\n",
        (unsigned long long)s->key, s->stage, s->set, nw, s->npool, vk_pool_n(), vk_pool_samp_n()); fclose(f); }
    VKLOG("runtime SPIR-V dumped: %s (%u words, stage %d, set %d); ALT requires DM_VK_SPV_RUNTIME=1", path, nw, s->stage, s->set);
}

// DM_VK_SPV_DIR=<dir> with DM_VK_SPV_ALT=1 (live, debugging): pipelines made while it is on use <dir>/<key>.spv in place of the bundle's code where the file
// exists (same bindings), so two compilations of a shader can be compared in one run: the translator's module through spirv-opt, its GLSL through glslc...
static VkShaderModule alt_module(vkshader_t *s) {
    const char *observer=getenv("DM_VK_FLAT_OBSERVER");
    if(observer&&!strcmp(observer,"1")&&(s->key==0xc8ab0e758c17df4aull||s->key==0x96dbc0392691b437ull))
        return s->mod; // the observer pair can never enter the independent generic ALT path
    const char *dir = getenv("DM_VK_SPV_DIR"); char path[512]; if (!dir) return s->mod;
    snprintf(path, sizeof path, "%s/%016llx.spv", dir, (unsigned long long)s->key); FILE *f = fopen(path, "rb"); if (!f) return s->mod;
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET); uint32_t *w = n > 20 && n <= (64L << 20) && !(n & 3) ? malloc((size_t)n) : NULL; VkShaderModule m = s->mod;
    if (w && fread(w, 1, (size_t)n, f) == (size_t)n) { VkShaderModuleCreateInfo ci = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, NULL, 0, (size_t)n, w};
        const char *r = getenv("DM_VK_SPV_RUNTIME"); int runtime = r && atoi(r);
        if (runtime && !spv_runtime_contract(s->spv, s->nspv, w, (uint32_t)(n / 4))) {
            VKLOG("[RED] shader %016llx: runtime alternative rejected (preamble differs from current module): %s", (unsigned long long)s->key, path);
        } else {
            if (!runtime) for (uint32_t i = 5, nw = (uint32_t)(n / 4); i < nw;) { uint32_t k = w[i] >> 16; if (!k || k > nw - i) break; if ((w[i] & 0xffff) == 71 && k >= 4 && w[i + 2] == 34) w[i + 3] = (uint32_t)s->set; i += k; }
            if (vkCreateShaderModule(V.dev, &ci, NULL, &m)) m = s->mod; else VKLOG("shader %016llx: alternative code from %s%s", (unsigned long long)s->key, path, runtime ? " (verified runtime layout)" : "");
        } }
    free(w); fclose(f); return m;
}
// A fragment shader's dual-source variant (gfx_vk_exec.c pkey dual): its output at location 1 becomes location 0, index 1, the second source of target 0's blend.
// NULL when it has no output at location 1, or outputs past it (one dual-source attachment at most: maxFragmentDualSrcAttachments)
static uint32_t *spv_dual(const uint32_t *w, uint32_t nw, uint32_t *onw) {
    uint32_t bound = w[3], fn = nw, at = 0, v1 = 0; uint8_t *out = calloc(bound, 1); int past = 0;
    for (uint32_t i = 5; i < nw;) { uint32_t n = w[i] >> 16, op = w[i] & 0xffff; if (!n || i + n > nw) { free(out); return NULL; }
        if (op == 54) { fn = i; break; }
        if (op == 59 && n >= 4 && w[i + 3] == 3 && w[i + 2] < bound) out[w[i + 2]] = 1;   // OpVariable ... Output
        i += n; }
    for (uint32_t i = 5; i < fn;) { uint32_t n = w[i] >> 16, op = w[i] & 0xffff;
        if (op == 71 && n == 4 && w[i + 2] == 30 && w[i + 1] < bound && out[w[i + 1]]) { if (w[i + 3] == 1) { at = i; v1 = w[i + 1]; } else if (w[i + 3] > 1) past = 1; }   // Location
        if (op == 71 && n == 4 && w[i + 2] == 32) past = 1;   // an Index already: not a shader this is for
        i += n; }
    free(out);
    if (!at || past) return NULL;
    uint32_t *o = malloc(((size_t)nw + 4) * 4); memcpy(o, w, (size_t)(at + 4) * 4); o[at + 3] = 0;   // Location 1 → 0
    o[at + 4] = 4u << 16 | 71; o[at + 5] = v1; o[at + 6] = 32; o[at + 7] = 1;   // OpDecorate ... Index 1
    memcpy(&o[at + 8], &w[at + 4], (size_t)(nw - at - 4) * 4); *onw = nw + 4;
    return o;
}
static VkShaderModule vk_shader_module_(vkshader_t *s, uint64_t um, uint64_t im, uint32_t xo, int fd);
VkShaderModule vk_shader_module(vkshader_t *s, uint64_t um, uint64_t im, uint32_t xo, int fd) { extern int g_prof_on; extern uint64_t g_prof_ns[8]; struct timespec a, b; if (g_prof_on) clock_gettime(CLOCK_MONOTONIC, &a);
    VkShaderModule m = vk_shader_module_(s, um, im, xo, fd); if (g_prof_on) { clock_gettime(CLOCK_MONOTONIC, &b); g_prof_ns[3] += (uint64_t)((b.tv_sec - a.tv_sec) * 1000000000ll + (b.tv_nsec - a.tv_nsec)); } return m; }
static VkShaderModule vk_shader_module_(vkshader_t *s, uint64_t um, uint64_t im, uint32_t xo, int fd) {
    int dual = fd >> 1 & 1, at = fd >> 2 & 1; fd &= 1;   // fd bit 1: the dual-source variant; bit 2: the alpha test's (spv_alpha_test)
    if (fd && (s->stage != 1 || s->depth_repl)) fd = 0;
    if (dual && s->stage != 1) dual = 0;
    if (at && s->stage != 1) at = 0;
    if (!(um | im | xo | (uint32_t)fd | (uint32_t)dual | (uint32_t)at) || !s->spv) return s->mod;
    if (xo == 1u << 31) {   // DM_VK_SPV_ALT
        pthread_mutex_lock(&g_sh_mu); for (int i = 0; i < s->nvar; i++) if (s->var[i].um == 0 && s->var[i].im == 0 && s->var[i].xo == xo) { VkShaderModule m = s->var[i].mod; pthread_mutex_unlock(&g_sh_mu); return m; }
        VkShaderModule m = alt_module(s); if (s->nvar < 8) s->var[s->nvar++] = (typeof(s->var[0])){0, 0, xo, 0, m}; pthread_mutex_unlock(&g_sh_mu); return m; }
    xo &= ~(1u << 31);
    pthread_mutex_lock(&g_sh_mu);
    for (int i = 0; i < s->nvar; i++) if (s->var[i].um == um && s->var[i].im == im && s->var[i].xo == xo && s->var[i].fd == (fd | dual << 1 | at << 2)) { VkShaderModule m = s->var[i].mod; pthread_mutex_unlock(&g_sh_mu); return m; }
    uint16_t bind[64]; uint8_t kind[64]; int nb = 0;
    for (int i = 0; i < s->nbind && i < 64; i++) if ((um | im) >> i & 1) { bind[nb] = s->b[i].binding; kind[nb++] = um >> i & 1 ? 1 : 2; }
    uint32_t nw = s->nspv, *w = nb ? spv_int_images(s->spv, s->nspv, bind, kind, nb, (uint32_t)s->set, &nw) : NULL; VkShaderModule m = s->mod;
    if (xo && (w || !nb)) { uint32_t n2 = 0, *w2 = spv_add_outputs(w ? w : s->spv, nw, xo, &n2); free(w); w = w2; nw = n2;
        if (w && s->stage == 0) { uint32_t n3 = 0, *w3 = spv_vs_init(w, nw, &n3); if (w3) { free(w); w = w3; nw = n3; } } }   // the added outputs start as (0, 0, 0, 1) too
    if (fd && (w || !(nb || xo))) { uint32_t n2 = 0, *w2 = spv_frag_depth(w ? w : s->spv, nw, &n2); free(w); w = w2; nw = n2; }
    if (dual && (w || !(nb || xo || fd))) { uint32_t n2 = 0, *w2 = spv_dual(w ? w : s->spv, nw, &n2); free(w); w = w2; nw = n2; }
    if (at && (w || !(nb || xo || fd || dual))) { uint32_t n2 = 0, *w2 = spv_alpha_test(w ? w : s->spv, nw, &n2); free(w); w = w2; nw = n2; }
    if (w) { VkShaderModuleCreateInfo ci = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, NULL, 0, (size_t)nw * 4, w};
        if (vkCreateShaderModule(V.dev, &ci, NULL, &m)) { m = s->mod; VKLOG("[RED] shader %016llx: its variant was refused (bindings %#llx / %#llx, outputs %#x, fragment depth %d, dual source %d, alpha test %d)", (unsigned long long)s->key, (unsigned long long)um, (unsigned long long)im, xo, fd, dual, at); }
        else VKLOG("shader %016llx: variant for bindings %#llx (uint texels) %#llx (int texels), outputs added %#x, fragment depth written %d, dual source %d, alpha test %d", (unsigned long long)s->key, (unsigned long long)um, (unsigned long long)im, xo, fd, dual, at);
        free(w); }
    else VKLOG("[RED] shader %016llx: no variant could be made (bindings %#llx / %#llx, outputs %#x, fragment depth %d, dual source %d, alpha test %d)", (unsigned long long)s->key, (unsigned long long)um, (unsigned long long)im, xo, fd, dual, at);
    if (s->nvar < 8) s->var[s->nvar++] = (typeof(s->var[0])){um, im, xo, fd | dual << 1 | at << 2, m};
    pthread_mutex_unlock(&g_sh_mu);
    return m;
}
// Process-lifetime buffers: Ryujinx support values, 64 KB of zeros, and the immutable all-set raw-global write bitmap (Android does not consume the write log).
void vk_support_values(void *p) {
    memset(p, 0, VK_SUPPORT_BYTES);
    for (int i = 0; i < 73; i++) *(float *)((uint8_t *)p + 192 + 16 * i) = 1.0f; // render_scale[i].x (SupportBuffer.cs)
}
static VkBuffer static_buf(int which) {
    static VkBuffer b[3]; static pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;
    { VkBuffer r = __atomic_load_n(&b[which], __ATOMIC_ACQUIRE); if (r) return r; }   // made already: no lock (every draw binds them)
    pthread_mutex_lock(&mu);
    if (!b[which]) {
        VkDeviceSize size = which == 2 ? DMSH_GMEM_MAP_ENTRIES / 8 : 65536;
        VkBuffer nb; VkBufferCreateInfo bi = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, NULL, 0, size, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VK_SHARING_MODE_EXCLUSIVE};
        VKCHK(vkCreateBuffer(V.dev, &bi, NULL, &nb)); VkMemoryRequirements req; vkGetBufferMemoryRequirements(V.dev, nb, &req);
        VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL, req.size, vk_mem_type(req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 0)};
        VkDeviceMemory m; VKCHK(vkAllocateMemory(V.dev, &ai, NULL, &m)); VKCHK(vkBindBufferMemory(V.dev, nb, m, 0));
        void *p; VKCHK(vkMapMemory(V.dev, m, 0, VK_WHOLE_SIZE, 0, &p));
        if (which == 0) vk_support_values(p); else memset(p, which == 2 ? 0xff : 0, size);
        __atomic_store_n(&b[which], nb, __ATOMIC_RELEASE);   // published once filled
    }
    pthread_mutex_unlock(&mu);
    return b[which];
}
VkBuffer vk_support_buffer(void) { return static_buf(0); }
VkBuffer vk_zero_buffer(void) { return static_buf(1); }
VkBuffer vk_gmem_log_disabled(void) { return static_buf(2); }
// a shader's c1 constants: copied once into a buffer of their own (the bundle bytes do not change; the reflection is never compressed)
void vk_shader_c1(vkshader_t *s, VkBuffer *buf, VkDeviceSize *off, VkDeviceSize *len) {
    if (!s->c1_len) { *buf = vk_zero_buffer(); *off = 0; *len = 65536; return; }
    if (!s->c1_buf) {
        VkBufferCreateInfo bi = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, NULL, 0, (s->c1_len + 255) & ~255u, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_SHARING_MODE_EXCLUSIVE};
        VKCHK(vkCreateBuffer(V.dev, &bi, NULL, &s->c1_buf)); VkMemoryRequirements req; vkGetBufferMemoryRequirements(V.dev, s->c1_buf, &req);
        VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL, req.size, vk_mem_type(req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 0)};
        VkDeviceMemory m; VKCHK(vkAllocateMemory(V.dev, &ai, NULL, &m)); VKCHK(vkBindBufferMemory(V.dev, s->c1_buf, m, 0));
        void *p; VKCHK(vkMapMemory(V.dev, m, 0, VK_WHOLE_SIZE, 0, &p)); memset(p, 0, bi.size); memcpy(p, s->c1, s->c1_len);
    }
    *buf = s->c1_buf; *off = 0; *len = (s->c1_len + 255) & ~255u;
}
// pipeline layouts by the set layouts of their stages (an empty layout fills the sets of absent stages)
VkPipelineLayout vk_pipeline_layout(VkDescriptorSetLayout sets[VK_NSETS], int nsets) {
    static VkDescriptorSetLayout empty; static nvn_hmap cache = {.klen = VK_NSETS * sizeof(VkDescriptorSetLayout)}; static pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;
    pthread_mutex_lock(&mu);
    if (!empty) { VkDescriptorSetLayoutCreateInfo li = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO}; VKCHK(vkCreateDescriptorSetLayout(V.dev, &li, NULL, &empty)); }
    VkDescriptorSetLayout k[VK_NSETS]; for (int i = 0; i < VK_NSETS; i++) { k[i] = i < nsets && sets[i] ? sets[i] : (i < nsets ? empty : VK_NULL_HANDLE); if (i < nsets) sets[i] = k[i]; }   // absent stages get the empty layout (the caller allocates their sets from it)
    VkPipelineLayout pl = nvn_hm_get(&cache, k);
    if (!pl) {
        VkPipelineLayoutCreateInfo ci = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, NULL, 0, (uint32_t)nsets, k};
        VKCHK(vkCreatePipelineLayout(V.dev, &ci, NULL, &pl)); nvn_hm_put(&cache, k, pl);
    }
    pthread_mutex_unlock(&mu);
    return pl;
}
// The pass-through geometry shader shader_vk.py generates for a fragment shader that reads inputs per vertex (Ampere LDTRAM) on a device without VK_KHR_fragment_shader_barycentric:
// keyed by the fragment shader's key and shader_pvgs.PV_GS_SALT, with the usual topology variants beside it (vk_shader_gs_topo)
vkshader_t *vk_shader_pv_gs(vkshader_t *fs) { if (!fs) return NULL; vkshader_t *g = vk_shader_q(fs->key ^ 0x61C8864680B583EBull, 2, 1); if (g) g->pv_gs = 1; return g; }
// The pass-through vertex shader shader_vk.py generates for each mesh kernel (shader_meshwrap.vertex_shader): keyed by the mesh key and shader_meshwrap.MESH_VS_SALT, a vertex-stage entry
vkshader_t *vk_shader_mesh_vs(vkshader_t *ms) { return ms ? vk_shader_q(ms->key ^ 0x2545F4914F6CDD1Dull, 0, 1) : NULL; }
