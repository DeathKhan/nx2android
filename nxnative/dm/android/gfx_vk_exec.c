// Android port, Vulkan backend: the command replay (mtl_exec, see gfx_vk.h and nvn_obj.h's record format).
// State records update a shadow of the NVN state; a draw or dispatch turns it into a pipeline (cached by everything that shapes it), descriptor sets (one per
// stage, filled from the shader's reflection: uniform slots, the shader's c1, the driver's c0 / c2 segments, textures and samplers by handle, images, storage
// buffers) and the draw. Render passes open lazily at the first draw or clear after SetRenderTargets and close before anything that is not drawing.
// Version 1 orders everything with whole-pipeline barriers around passes, copies and dispatches.
#include "gfx_vk.h"
#include "upscale_square_vk.h"
#include "movie_fill_vk.h"   // the decoded movie picture in place of a skipped post-processing stage (passthru_replace)
#include "gfx_vk_slot.h"
#include "dmsh_abi.h"   // DMSH_GMEM_MAP_ENTRIES: the raw global memory write log's bits (stage_set RK_GMEM_LOG)
#include "nvn_hmap.h"
#include "nvn_queue.h"
#include "game_profile.h"
#include <math.h>
#include <string.h>
#include <sys/resource.h>
#include <unistd.h>

VkPipelineLayout vk_pipeline_layout(VkDescriptorSetLayout sets[VK_NSETS], int nsets);
void vk_shader_c1(vkshader_t *s, VkBuffer *buf, VkDeviceSize *off, VkDeviceSize *len);
VkBufferView vk_tbuf_view(const tex_t *t, int storage);
void vk_sub_update(uint64_t dst, const void *data, uint64_t n);
int prog_key(const prog_t *p, unsigned st, uint8_t key[8]);

// ---------------- NVN state shadow ----------------
static struct {
    const prog_t *prog; vkshader_t *sh[6]; int prog_ok;
    p_bind_buf ubo[6][16], ssbo[6][16];
    uint64_t tex[6][256], samp[6][128], img[6][32], tex_sseq[6][256], samp_sseq[6][128];   // *_sseq: the sampler registrations made before the bind (gfx_vk_tex.c vk_samp_at)
    blend_st blend[8]; color_st color; chmask_st chmask; ms_st ms; poly_st poly; ds_st ds;
    uint32_t vattr[16]; vs_st vstream[16]; NVNbufferRange vbuf[16];
    uint8_t push[2][0x100];   // NVN2 push constants of the graphics stages [0] and compute [1] (OP_PUSH_CONST; DMSH_PUSH_SIZE bytes each)
    float vp[16][4]; float dnear, dfar; p_scissor sc; int sc_set; uint8_t vp_sw[16][4];   // vp_sw: viewport swizzles XOR the identity (p_vp_swizzle)
    int sref[2], smask[2], swmask[2];
    p_poly_offset pofs; float blendc[4]; int depth_clamp, rast_discard, prim_restart; float alpha_ref;   // alpha_ref: SetAlphaRef (the alpha test, pkey atest)
    tex_t rt_col[8], rt_dep; view_t rt_cview[8], rt_dview; uint8_t rt_has_col[8], rt_has_cview[8], rt_has_dep, rt_has_dview; int rt_n; uint32_t rt_bad[8];   // rt_bad: the first word of a colour slot that is not a texture object (traces)
    int64_t row_stride, image_stride;
    // the open render pass
    int pass_open; uint32_t pass_w, pass_h; VkFormat cfmt[8], dfmt; VkImageAspectFlags daspect; uint32_t samples; int ncol;
    VkPipeline bound_pipe; int rp_pass;   // rp_pass: the open pass began with vkCmdBeginRenderPass
} S;
static p_tf_bind g_tf[4]; static uint64_t g_tf_off[4]; static int g_tf_on, g_tf_in_pass; static uint32_t g_tf_verts;   // transform feedback: buffers, bytes written into each, capture on, a capture in the open pass (xfb_begin), vertices captured since Begin
static int g_depth_zero = 1, g_lower_left;
void mtl_set_depth_zero(int on) { g_depth_zero = on; }
void mtl_set_origin_lower_left(int on) { g_lower_left = on; }
int vk_cap_origin_ll(void) { return g_lower_left; } int vk_cap_depth_zero(void) { return g_depth_zero; }   // DM_NVN_CAPTURE (gfx_vk_capture.inc): the state it records
// NVN's upper-left mode has clip space y up (as Metal and D3D): it is drawn with a flipped viewport over the same top-left rectangle, lower-left mode without
// one; the scissor is then never flipped and the winding follows nvn_mtl.m's. Drawn the other way, every blit flipped the picture (NEO docked: upside down)
// and the winding had to be inverted. DM_VK_FLIPY=0 brings the old way back
#include <dlfcn.h>
static int g_flipy;
// Lower-left mode then stores the window image bottom-up (GL's rows), so the present flips it as nvn_mtl.m does (Celeste's MonoGame keeps NVN's default lower-left origin)
int vk_window_bottom_up(void) { return g_lower_left && g_flipy; }
static void state_reset(void) {
    memset(&S, 0, sizeof S);
    S.dnear = 0; S.dfar = 1;
    S.chmask.mask = 0xffffffffu;
    S.color.logic_op = 3;
    S.poly.front = 1; S.poly.mode = 2;
    S.ds.dw = 1; S.ds.df = 2; S.ds.sf_f = S.ds.sf_b = 8; S.ds.fail_f = S.ds.zfail_f = S.ds.zpass_f = S.ds.fail_b = S.ds.zfail_b = S.ds.zpass_b = 1;
    S.smask[0] = S.smask[1] = S.swmask[0] = S.swmask[1] = 0xff;
    for (int i = 0; i < 8; i++) { S.blend[i].target = i; S.blend[i].src_rgb = S.blend[i].src_a = 2; S.blend[i].dst_rgb = S.blend[i].dst_a = 1; S.blend[i].eq_rgb = S.blend[i].eq_a = 1; }
}
static void prewarm_start(void); static void compute_warm_start(void); static void build_dump(void);
static uint64_t g_set_epoch = 1;   // the submission the descriptor set cache belongs to (stage_set)
static int32_t g_pass_bb[4];   // bounding box of the scissors the open pass's draws used (x0 y0 x1 y1)
static int g_pace; static uint64_t g_sub_d0, g_sub_hist[8]; static unsigned g_sub_hn;   // DM_VK_PART_PACE (part_pace)
static int g_cb_draws;   // draws in the command buffer being recorded (vk_cmd_split after DM_VK_CB_DRAWS draws, 0 = never)
static int g_psplit, g_psplit_depth, g_psplit_now, g_ord_parts, g_setsplit;   // DM_VK_PASS_SPLIT / _DEPTH (prepare_draw); g_psplit_now: the pass being begun goes on with the one just ended (begin_pass cuts before it);
                                                                    // g_ord_parts: ordinary DM_VK_CB_DRAWS cuts in this submission, kept to the old 15 (MAIN_CBS was 16): the extra parts are for pass splits
static int g_var_now;   // the bisection's change for the draw being recorded (see g_bis_var)
static int g_tracing;   // the draw being recorded is the one the bisection found (its descriptors are logged)
// replay counters, printed by gfx_vk_present.c every 600 presents (draws done and skipped by reason, passes, clears folded into load ops, dispatches)
struct { uint64_t draws, skip_prog, skip_rt, skip_pipe, passes, folded, dispatches, pipes_built, rt_kept, ns_exec, ns_desc, ns_pipe, max_pass_draws; } g_vkst; static uint64_t g_pass_draws;   // ns_*: time in the replay, its descriptor sets, its pipeline keys and lookups
static inline uint64_t tnow(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return (uint64_t)t.tv_sec * 1000000000ull + (uint64_t)t.tv_nsec; }
// The per-draw timers (about 10 a draw, ~3,000 draws a frame): the arm64 counter register read directly, a few ns against clock_gettime's ~20-40, in ns
static inline uint64_t tick(void) {
#if defined(__aarch64__)
    static uint64_t q; if (!q) { uint64_t f; __asm__ volatile("mrs %0, cntfrq_el0" : "=r"(f)); q = (1000000000ull << 32) / f; }
    uint64_t c; __asm__ volatile("mrs %0, cntvct_el0" : "=r"(c)); return (uint64_t)(((unsigned __int128)c * q) >> 32);
#else
    return tnow();
#endif
}
// Measurement switches, read again at every submission so DM_CTL "env" lines change them while the game runs (never for normal play: they draw wrong):
// DM_VK_X_NOBAR=1 no barrier before a pass, DM_VK_X_DEPTH=1 depth neither loaded nor stored, DM_VK_X_COLOR=1 colour not loaded unless cleared
// DM_VK_TRACE_CS=hex[,hex...] is consumed once, until all requested keys are seen (at most 600 submissions/32 dispatches). It never forces skipped/replaced shaders to execute.
enum { CS_TRACE_KEYS = 16, CS_TRACE_DISPATCHES = 32 };
static uint64_t g_cs_trace_keys[CS_TRACE_KEYS], g_cs_trace_bytes;
static unsigned g_cs_trace_n, g_cs_trace_seen, g_cs_trace_copies, g_cs_trace_wait, g_cs_trace_found;
static unsigned g_cs_buffer_copies; // raw buffer reservations in this request (independent of images)
static int g_cs_trace_buffers, g_cs_buffer_collect;
static unsigned g_cs_trace_frames; static uint64_t g_cs_trace_start_present;
extern uint64_t g_tex_frame;
static int g_cs_trace_dump, g_cs_trace_collect;
static void cs_trace_poll(void) {
    const char *request = getenv("DM_VK_TRACE_CS");
    int fresh = request && *request && strcmp(request, "0");
    g_cs_trace_collect = 0;
    if (g_cs_trace_n) {
        int frame_window_done = g_cs_trace_seen && g_tex_frame >= g_cs_trace_start_present &&
            g_tex_frame - g_cs_trace_start_present > g_cs_trace_frames;
        if ((!g_cs_trace_frames && g_cs_trace_found == ((1u << g_cs_trace_n) - 1)) ||
            (g_cs_trace_frames && frame_window_done) || g_cs_trace_seen >= CS_TRACE_DISPATCHES) {
            VKLOG("compute trace completed requested-key window: %u matching dispatches, %u bounded traces, %u image copies reserved",
                g_cs_trace_seen, g_cs_trace_seen < CS_TRACE_DISPATCHES ? g_cs_trace_seen : CS_TRACE_DISPATCHES, g_cs_trace_copies);
            g_cs_trace_n = 0;
        } else if (g_cs_trace_wait && !--g_cs_trace_wait) {
            VKLOG("compute trace expired: incomplete requested keys after 600 submissions (mask %#x)", g_cs_trace_found); g_cs_trace_n = 0;
        }
        // Empty and unrelated submissions do not consume the capture window.
        if (g_cs_trace_n && !fresh) return;
        if (g_cs_trace_n) VKLOG("compute trace: pending request replaced");
    }
    g_cs_trace_n = g_cs_trace_seen = g_cs_trace_copies = g_cs_trace_wait = g_cs_trace_found = g_cs_buffer_copies = 0; g_cs_trace_bytes = 0;
    if (!fresh) return;
    // Copy before setenv: its storage can move when the one-shot request is consumed.
    char keys[CS_TRACE_KEYS * 19 + 1]; snprintf(keys, sizeof keys, "%s", request); setenv("DM_VK_TRACE_CS", "0", 1);
    if (!strcmp(keys, "cancel")) { VKLOG("compute trace cancelled"); return; }
    for (const char *q = keys; *q && g_cs_trace_n < CS_TRACE_KEYS;) {
        while (*q == ' ' || *q == ',') q++;
        if (!*q) break;
        const char *start = q; if (q[0] == '0' && (q[1] == 'x' || q[1] == 'X')) q += 2;
        unsigned digits = 0; while ((*q >= '0' && *q <= '9') || (*q >= 'a' && *q <= 'f') || (*q >= 'A' && *q <= 'F')) { q++; digits++; }
        if (!digits || digits > 16 || (*q && *q != ',' && *q != ' ')) { VKLOG("compute trace: invalid key list; accepted prefix only"); break; }
        uint64_t key = strtoull(start, NULL, 16); int duplicate = 0;
        for (unsigned k = 0; k < g_cs_trace_n; k++) duplicate |= key == g_cs_trace_keys[k];
        if (key && !duplicate) g_cs_trace_keys[g_cs_trace_n++] = key;
    }
    const char *dump = getenv("DM_VK_TRACE_CS_DUMP"); g_cs_trace_dump = dump && atoi(dump);
    const char *buffers = getenv("DM_VK_TRACE_CS_BUFFERS"); g_cs_trace_buffers = buffers && atoi(buffers);
    const char *frames = getenv("DM_VK_TRACE_CS_FRAMES"); int nf = frames ? atoi(frames) : g_cs_trace_buffers ? 2 : 0;
    g_cs_trace_frames = nf < 0 ? 0 : nf > 4 ? 4 : (unsigned)nf; g_cs_trace_start_present = 0;
    if (g_cs_trace_n) {
        g_cs_trace_wait = 600;
        VKLOG("compute trace armed: %u keys, all requested keys within 600 submissions, at most %u dispatches; sampled-before/storage-before-after capture %s (requires active PASSDUMP)",
            g_cs_trace_n, CS_TRACE_DISPATCHES, g_cs_trace_dump ? "requested" : "off");
        if (g_cs_trace_frames) VKLOG("compute trace frame window: first partial present plus %u complete presents; raw GPU buffers %s",
            g_cs_trace_frames, g_cs_trace_buffers ? "on (64 x 64 KiB maximum)" : "off");
    }
}
static int cs_trace_match(uint64_t key) {
    for (unsigned k = 0; k < g_cs_trace_n; k++) if (g_cs_trace_keys[k] == key) {
        if (!g_cs_trace_seen) g_cs_trace_start_present = g_tex_frame;
        g_cs_trace_found |= 1u << k;
        if (++g_cs_trace_seen <= CS_TRACE_DISPATCHES) return 1;
        if (g_cs_trace_seen == CS_TRACE_DISPATCHES + 1) VKLOG("compute trace: dispatch limit reached");
        return 0;
    }
    return 0;
}
static int g_up_diag_left, g_up_diag_request; // DM_VK_UPSCALER_DIAG=N: bounded image/view resolution logs, default off
static int g_x_nobar, g_x_depth, g_x_color, g_x_cbdraws, g_x_tracepass = -1, g_sub_pass;   // DM_VK_TRACE_PASS=n: log every draw of the submission's pass n (once)
static int g_x_listpasses;
uint64_t vk_replay_ns(void) { return g_vkst.ns_exec; }   // the replay time so far (gfx_vk_present.c: a frame's cost for the adaptive rate)
// flicker experiments (DM_CTL env lines, read every submission): DM_VK_X_NOSETC no descriptor set reuse, DM_VK_X_NODRVC no driver segment reuse, DM_VK_X_LANDCUT deferred
// uniform updates land at each cut, DM_VK_X_NOKEEP no pass kept for the same targets, DM_VK_X_NOSHC no shadow reuse
static int g_fdz = 1;  // DM_VK_FDZ (live): fragment-depth variants for draws on depth written by fragment shaders (pipeline key fdz)
static uint64_t g_fdz_fs;   // DM_VK_FDZ_FS=<hex key> (live diagnostic): test late depth for this fragment shader, without a prior depth-writing pass
static uint64_t g_depth_func_fs; static int g_depth_func;   // DM_VK_DEPTH_FUNC_FS=<hex key>:<NVN func 1..8> (live diagnostic); preserves depth enable/write
enum { LRZ_READ = 2, LRZ_SWITCH = 4, LRZ_CUT = 8, LRZ_ALL = 14, LRZ_CUT_RD = 16, LRZ_CORNER = 32 };   /* LRZ_CUT_RD: only read at a cut; LRZ_CORNER: the last texel, not the first (tests) */   // DM_VK_LRZ_TOUCH bits: the read before a load pass, the write-back on a depth switch, the write-back at a cut
static int g_x_nosetc_bits;   // DM_VK_X_NOSETC bits (bisection): 2 no set reuse for the mesh kernels, 4 none for the draw stages, 8 no table ring
static int g_x_nosetc, g_x_nodrvc, g_x_landcut, g_x_nokeep, g_x_nobc, g_x_cutany, g_lrz_touch, g_xfb_on = 1, g_x_nogs, g_spv_alt, g_x_nodispbar; int g_x_noshc;   // g_xfb_on: DM_VK_XFB=0 captures no transform feedback (A/B)
static int g_drv_compact = 1, g_buffer_memo = 1; // DM_VK_DRV_COMPACT / DM_VK_BUFFER_MEMO=0: independent live A/B
static int g_x_nosetmemo, g_x_setverify;   // DM_VK_X_NOSETMEMO=1 (live, A/B): no per-stage set memo; DM_VK_X_SETVERIFY=1 (live): one memo hit in 64 checked (see stage_set)
static int g_vbound = 1;   // DM_VK_VBOUND (see vbuf_bound)
// DM_VK_PROF_SLOW=1 (live): a submission whose replay took over 40 ms logs where it went: pipelines built on the replay thread, textures (vk_tex: images made and
// packaged uploads, the latter also apart), the copied pools' sync before it, shader modules, and the rest
int g_prof_on; uint64_t g_prof_ns[8]; static uint64_t g_prof_t0, g_prof_pkg0, g_prof_npkg0, g_prof_pipes0;
static int g_async_pipe = 1;   // DM_VK_ASYNC_PIPE (see async_pipe)
static int g_lin_back = 1;   // DM_VK_LIN_BACK (see lin_back)
static uint64_t g_spv_key;   // DM_VK_SPV_KEY=<shader key> (live): limit alternative graphics modules to pipelines using this stage
static int g_ubo_bounds = 1;   // DM_VK_UBO_BOUNDS=0 (live, A/B): copy full uniform bindings into deferred-update shadows
static int g_gmem_log;   // DM_VK_GMEM_LOG=1 (live): restore cleared per-set write logs for A/B
static void original_index_read(void);
static int g_indirect_touch;   // DM_VK_INDIRECT_TOUCH (live): 1 transfer reads before indirect draws, 2 pass split alone, 0 off
static int g_samp_snap = 1;   // DM_NVN_SAMP_SNAP=0 (live): samplers as registered now, not as the bind saw them (gfx_vk_tex.c vk_samp_at)
static uint64_t g_x_flat[8]; static int g_x_flat_mode[8], g_x_nflat;   // DM_VK_X_FLAT_VS=<vertex shader key>[:mode][,...] (live debugging): mode bits 1 solid magenta pixel shader, 2 no depth test, 4 no blending (7 when left out)
// Additional diagnostic bits: 8 no alpha coverage, 16 no depth bias, 32 no depth test/write, 64 no stencil. Captured in the pipeline key (pkey.flat), so a live change takes effect.
static uint64_t g_x_logfs;   // DM_VK_LOG_FS=<fragment shader key> (live): one line per draw with it, with its samplers as the draw sees them
static int g_rd_touch = 1, g_x_rdkeep, g_x_nskipvs; static uint64_t g_x_skipvs[8];   // DM_VK_RD_TOUCH, DM_VK_X_RDKEEP, DM_VK_X_SKIPVS (see prepare_draw)
static char g_x_tracesize[40], g_x_traceframe[40]; static int g_x_traceframe_on;
// DM_VK_WATCH=<pool offset, hex>[#n] (live): log the next n (20) uses of images at that offset of their pool (attachment, sampled, copy), with the pass and present
// numbers, to find what fills a texture a pass reads
static long long g_watch_off = -1; static int g_watch_n; extern uint64_t g_tex_frame;
// DM_VK_UBO_LOG=<shader key, hex>[#n] (live): log the next n (40) uniform buffers that shader reads (present, stage, register, address, a hash and the first 16 words)
static uint64_t g_ubolog_key; static int g_ubolog_n;
static void ubo_log(const vkshader_t *s, int st, int reg, uint64_t a, uint64_t n) {
    if (!g_ubolog_key || s->key != g_ubolog_key || g_ubolog_n <= 0) return;
    g_ubolog_n--; static uint8_t buf[65536]; if (n > sizeof buf) n = sizeof buf; void vk_ubd_read(uint64_t a, uint64_t n, uint8_t *dst); vk_ubd_read(a, n, buf);
    { const char *dd = getenv("DM_VK_UBO_DUMP"); if (dd && atoi(dd)) {   // DM_VK_UBO_DUMP=1 (live, with UBO_LOG): the whole buffer as DM_ROOT/ubo_<present>_p<pass>_st<stage>_c<reg>.bin (compare with the Mac's DM_NVN_CBDUMP)
        char path[1024]; const char *root = getenv("DM_ROOT"); snprintf(path, sizeof path, "%s/ubo_%llu_p%d_st%d_c%d.bin", root ? root : ".", (unsigned long long)g_tex_frame, g_sub_pass - 1, st, reg);
        FILE *df = fopen(path, "wb"); if (df) { fwrite(buf, 1, (size_t)n, df); fclose(df); } } }
    uint64_t h = 1469598103934665603ull; for (uint64_t i = 0; i < n; i++) h = (h ^ buf[i]) * 1099511628211ull; float f[16] = {0}; memcpy(f, buf, n < 64 ? n : 64);
    // every word of 512 bytes, as floats: from byte 0, or from DM_VK_UBO_LOG_FROM=<byte offset> (live, a multiple of 16) for fields further in
    const char *fe = getenv("DM_VK_UBO_LOG_FROM"); uint32_t from = fe ? (uint32_t)atoi(fe) & ~15u : 0; if (from >= n) from = 0;
    char t[4096]; int k = from ? snprintf(t, sizeof t, "(from +%u) ", from) : 0; uint32_t nw = (uint32_t)(n - from < 512 ? n - from : 512) / 4; (void)f;
    for (uint32_t i = 0; i < nw && k < (int)sizeof t - 24; i++) { float v; memcpy(&v, buf + from + 4 * i, 4); k += snprintf(t + k, sizeof t - k, "%s%g", i ? (i % 4 ? " " : " | ") : "", v); }
    if (st == 0 && reg == 0 && S.vbuf[0].address) {   // the vertex stream 0 the draw reads, hashed the same way (CPU view at replay)
        uint64_t vn = S.vbuf[0].size > sizeof buf ? sizeof buf : S.vbuf[0].size, vh = 1469598103934665603ull; memcpy(buf, (const void *)(uintptr_t)S.vbuf[0].address, vn);
        for (uint64_t i = 0; i < vn; i++) vh = (vh ^ buf[i]) * 1099511628211ull;
        float v[6] = {0}; memcpy(v, buf, vn < 24 ? vn : 24);
        if (S.rt_has_dep) VKLOG("ubo: present %llu pass %d depth target pool %#llx+%#llx serial %llu; depth test %d func %d", (unsigned long long)g_tex_frame, g_sub_pass - 1,
                                (unsigned long long)(uintptr_t)S.rt_dep.pool, (unsigned long long)S.rt_dep.off, (unsigned long long)S.rt_dep.serial, S.ds.dt, S.ds.df);
        VKLOG("ubo: present %llu vertex stream 0 %#llx +%llu hash %016llx: %g %g %g %g %g %g", (unsigned long long)g_tex_frame, (unsigned long long)S.vbuf[0].address, (unsigned long long)S.vbuf[0].size,
              (unsigned long long)vh, v[0], v[1], v[2], v[3], v[4], v[5]);
        void vk_ubd_read(uint64_t a, uint64_t n, uint8_t *dst); vk_ubd_read(a, n, buf); }
    VKLOG("ubo: present %llu pass %d stage %d c%d %#llx +%llu hash %016llx: %s", (unsigned long long)g_tex_frame, g_sub_pass - 1, st, reg, (unsigned long long)a, (unsigned long long)n, (unsigned long long)h, t);
}
// DM_VK_TEXDUMP=<pool hex>+<offset hex> (read when a pass dump starts; the trace prints both): during the dump, the first sampled image there is blitted (level 0, up to 6 layers)
// into RGBA16F images and dumped with the passes, so a compressed texture (BC6H cube maps...) can be looked at decoded by the GPU
static long long g_td_off = -1; static unsigned long long g_td_pool; static vkimg_t *g_td_im; static int g_td_done;
void vk_texdump_note(vkimg_t *im) { if (g_td_off >= 0 && !g_td_done && !g_td_im && im && im->off == g_td_off && im->pool == g_td_pool) g_td_im = im; }
// the pass list (DM_VK_TRACE_PASS=-2, DM_VK_LIST_AT): the images each pass or dispatch reads, by pool offset, printed at its end
static uint64_t g_lst_rd[48]; static int g_lst_nrd;
// the images each stage's set was last collected with (prepare_draw: a memoized set keeps its last collection's), for the order barrier check below
typedef struct { uint64_t pool; int64_t off, size; } stimg_t;
static stimg_t g_stimg[3][32]; static int g_stn[3], g_coll_st = -1;
// Order-only barriers inside a pass (profile key order_keep; DM_VK_ORDER_KEEP=0/1, live, overrides it). Every barrier ends the pass (OP_BARRIER), and on a tiled GPU each
// pass begun again on the same attachments stores and loads all of them: Monster Hunter Rise orders its fragments (NVN barrier 0x2, sometimes 0x1) between almost every
// draw of its decal, transparent and effect passes on the 960x540 scene (B10G11R11 + R8 + D24S8), 98 of its 182 passes a frame begun again that way. With the key a barrier
// of only ORDER_PRIMITIVES / ORDER_FRAGMENTS / ORDER_FRAGMENTS_TILED leaves the pass open and is decided at the next draw (prepare_draw): the pass ends there, with the
// barrier, when that draw samples one of the pass's attachments, or a shader of it or of an earlier draw of the pass writes memory (shader_wmem). Attachment writes
// themselves keep their order inside a pass (primitive order). Off by default: Xenoblade Chronicles 3's characters tore with such barriers kept
static int g_order_keep, g_order_pending, g_order_check, g_pass_wmem; uint64_t g_order_kept, g_order_ended;   // (kept / ended: gfx_vk_present.c reports them)
static void list_reads(const char *what) {
    if (!g_lst_nrd) return;
    char t[1024]; int k = 0; for (int i = 0; i < g_lst_nrd && k < (int)sizeof t - 24; i++) k += snprintf(t + k, sizeof t - k, " %llx", (unsigned long long)g_lst_rd[i]);
    VKLOG("    %s reads%s", what, t); g_lst_nrd = 0;
}
void vk_watch(const vkimg_t *im, const char *what) {
    if (g_x_listpasses && im && what[0] == 's') { uint64_t o = (uint64_t)im->off; int i = 0; while (i < g_lst_nrd && g_lst_rd[i] != o) i++; if (i == g_lst_nrd && i < 48) g_lst_rd[g_lst_nrd++] = o; }
    if (g_coll_st >= 0 && im && what[0] == 's') { if (g_stn[g_coll_st] < 32) g_stimg[g_coll_st][g_stn[g_coll_st]++] = (stimg_t){im->pool, im->off, im->size}; else g_stn[g_coll_st] = 33; }   // 33: too many to keep
    if (g_watch_off < 0 || !im || im->off != g_watch_off || g_watch_n <= 0) return;
    g_watch_n--; VKLOG("watch: present %llu pass %d (%ux%u): %s %ux%u fmt %d pool %#llx+%#llx (image %p, nvn flags %#x, %d overlapping)", (unsigned long long)g_tex_frame, g_sub_pass - 1, S.pass_w, S.pass_h, what, im->w, im->h, im->fmt,
                       (unsigned long long)im->pool, (unsigned long long)im->off, (const void *)im, im->flags, im->novl);
}
static void set_profile_poll(void);
static int g_pdum_lo = -1, g_pdum_hi, g_pdum_nf; static unsigned g_pdum_kinds; static uint32_t g_pdum_f[16];   // + DM_VK_POOL_DUMMY_KINDS=mask, DM_VK_POOL_DUMMY_FMT=hex nvn formats
//   // DM_VK_POOL_DUMMY=lo-hi (live, diagnostic): those texture IDs read the dummy in the pool tables, to bisect which bindless read is wrong
static int g_pool_strict;
static uint64_t g_ring_full, g_ring_patch, g_ring_ids, g_ring_hits, g_ring_none, g_ring_ns;   // the pool tables ring (tbl_ring)
// DM_VK_TRACE_FS_BUFFERS=<fragment key>: first matching draw in each of at most three
// presents, default target 128x128. Consumed once; copies share the bounded compute completion ring.
// DM_VK_LIGHT_DIAG=<fragment key in hex>:<uint32 mask> (live, diagnostic only).
// The audited runtime shader may read support member 7 (int32 at byte 1376), unused by its original code.
// A nonzero mask selects only that fragment shader; zero restores the immutable normal support buffer.
static uint64_t g_light_diag_key, g_light_diag_bound_epoch; static uint32_t g_light_diag_mask; static int g_light_diag_pending;
enum { LIGHT_DIAG_OFFSET = 1376 };
_Static_assert(LIGHT_DIAG_OFFSET + sizeof(uint32_t) <= VK_SUPPORT_BYTES, "light diagnostic support lane");
static int light_diag_parse(const char *request, uint64_t *key, uint32_t *mask) {
    *key = 0; *mask = 0;
    if (!request || !*request || !strcmp(request, "0")) return 1;
    const char *p = request; if (p[0] == '0' && (p[1] == 'x' || p[1] == 'X')) p += 2;
    uint64_t k = 0; unsigned digits = 0;
    for (; *p && *p != ':'; p++) {
        unsigned c = (unsigned char)*p, v;
        if (c >= '0' && c <= '9') v = c - '0';
        else if (c >= 'a' && c <= 'f') v = c - 'a' + 10;
        else if (c >= 'A' && c <= 'F') v = c - 'A' + 10;
        else return 0;
        if (++digits > 16) return 0;
        k = (k << 4) | v;
    }
    if (!digits || !k || *p++ != ':') return 0;
    unsigned base = 10; if (p[0] == '0' && (p[1] == 'x' || p[1] == 'X')) { base = 16; p += 2; }
    uint64_t m = 0; digits = 0;
    for (; *p; p++) {
        unsigned c = (unsigned char)*p, v;
        if (c >= '0' && c <= '9') v = c - '0';
        else if (base == 16 && c >= 'a' && c <= 'f') v = c - 'a' + 10;
        else if (base == 16 && c >= 'A' && c <= 'F') v = c - 'A' + 10;
        else return 0;
        if (++digits > (base == 16 ? 8u : 10u)) return 0;
        m = m * base + v; if (m > UINT32_MAX) return 0;
    }
    if (!digits) return 0;
    if (m) { *key = k; *mask = (uint32_t)m; }
    return 1;
}
static void light_diag_poll(void) {
    uint64_t key; uint32_t mask; light_diag_parse(getenv("DM_VK_LIGHT_DIAG"), &key, &mask);
    if (key != g_light_diag_key || mask != g_light_diag_mask) {
        g_light_diag_pending = mask != 0; g_light_diag_bound_epoch = 0;
        VKLOG("lighting diagnostic: fragment %016llx mask %#x at support byte %u (0 restores normal inputs)",
            (unsigned long long)key, mask, LIGHT_DIAG_OFFSET);
    }
    g_light_diag_key = key; g_light_diag_mask = mask;
}
static void light_diag_support(int st, const vkshader_t *s, VkDescriptorBufferInfo *bi) {
    *bi = (VkDescriptorBufferInfo){vk_support_buffer(), 0, VK_SUPPORT_BYTES};
    if (!g_light_diag_mask || st != 1 || s->key != g_light_diag_key) return;
    static struct { uint64_t epoch, key; uint32_t mask; VkDescriptorBufferInfo bi; } cache;
    if (cache.epoch != g_set_epoch || cache.key != s->key || cache.mask != g_light_diag_mask) {
        VkBuffer buffer; VkDeviceSize offset;
        uint8_t *p = vk_scratch(VK_SUPPORT_BYTES, V.props.limits.minUniformBufferOffsetAlignment, &buffer, &offset);
        vk_support_values(p); memcpy(p + LIGHT_DIAG_OFFSET, &g_light_diag_mask, sizeof g_light_diag_mask);
        cache.epoch = g_set_epoch; cache.key = s->key; cache.mask = g_light_diag_mask;
        cache.bi = (VkDescriptorBufferInfo){buffer, offset, VK_SUPPORT_BYTES};
    }
    *bi = cache.bi; // immutable until this submission's completion; never edit a previous mask's buffer
    g_light_diag_bound_epoch = g_set_epoch;
}
static void light_diag_note_draw(uint64_t key, int draw) {
    if (!g_light_diag_pending || !g_light_diag_mask || key != g_light_diag_key || g_light_diag_bound_epoch != g_set_epoch) return;
    g_light_diag_pending = 0;
    VKLOG("lighting diagnostic applied: fragment %016llx mask %#x present %llu frame %llu pass %d draw %d submission %llu",
        (unsigned long long)key, g_light_diag_mask, (unsigned long long)g_tex_frame,
        (unsigned long long)vk_frame_current(), g_sub_pass - 1, draw, (unsigned long long)g_set_epoch);
}

static uint64_t g_fs_buffer_key, g_fs_buffer_start, g_fs_buffer_last;
static unsigned g_fs_buffer_frames, g_fs_buffer_seen, g_fs_buffer_copies;
static uint32_t g_fs_buffer_w, g_fs_buffer_h, g_fs_buffer_count;
static int g_fs_buffer_mesh;
enum { FS_BUFFER_BYTE_BUDGET = 48 << 20 };
static uint32_t g_fs_buffer_sbuf_max = 65536;
static uint64_t g_fs_buffer_bytes;
static int g_fs_buffer_collect;
// Image copies share the existing completion-owned pass readbacks; this selector
// is meaningful only inside an exact-count fragment-buffer capture.
static int g_fs_image_binding = -1, g_fs_image_collect;
static unsigned g_fs_image_copies;
static uint64_t g_fs_image_bytes;
static int g_draw_i; // submission draw identity, also recorded by bounded graphics captures
enum { FS_BUFFER_WINDOWS = 8 };
typedef struct { unsigned stage, reg; uint64_t offset, bytes; } fs_buffer_window;
static fs_buffer_window g_fs_buffer_windows[FS_BUFFER_WINDOWS];
static unsigned g_fs_buffer_windows_n;
// Extra storage-buffer windows are relative to the guest binding, not the aligned Vulkan base.
// DM_VK_TRACE_FS_WINDOWS=stage:reg:hexOffset:hexBytes[, ...], at most eight windows.
static int fs_buffer_number(const char *s, unsigned base, uint64_t *out) {
    size_t n = strlen(s); if (!n || n > (base == 16 ? 16u : 2u)) return 0;
    for (size_t i = 0; i < n; i++) {
        unsigned c = (unsigned char)s[i];
        if (!((c >= '0' && c <= '9') || (base == 16 && ((c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F'))))) return 0;
    }
    *out = strtoull(s, NULL, (int)base); return 1;
}
static int fs_buffer_windows_read(const char *request) {
    g_fs_buffer_windows_n = 0;
    if (!request || !*request) return 1;
    char text[512]; if (strlen(request) >= sizeof text) return 0; strcpy(text, request);
    char *item = text;
    while (*item) {
        char *next = strchr(item, ','); if (next) *next++ = 0;
        char *fields[4] = {item}; uint64_t v[4];
        for (unsigned i = 1; i < 4; i++) { char *c = strchr(fields[i - 1], ':'); if (!c) return 0; *c = 0; fields[i] = c + 1; }
        for (unsigned i = 0; i < 4; i++) if (!fs_buffer_number(fields[i], i < 2 ? 10 : 16, &v[i])) return 0;
        if (v[0] > 2 || v[1] > 15 || !v[3] || v[3] > 65536 || ((v[2] | v[3]) % 4) ||
            g_fs_buffer_windows_n == FS_BUFFER_WINDOWS) return 0;
        g_fs_buffer_windows[g_fs_buffer_windows_n++] = (fs_buffer_window){(unsigned)v[0], (unsigned)v[1], v[2], v[3]};
        if (!next) return 1;
        if (!*next) return 0;
        item = next;
    }
    return 0;
}
// Optional exact direct vertex/index count, captured with the other selection fields.
// Zero/unset preserves the previous selection; invalid input fails closed.
static int fs_buffer_count_read(const char *s) {
    g_fs_buffer_count = 0;
    if (!s || !*s) return 1;
    uint64_t v = 0;
    for (; *s; s++) {
        if (*s < '0' || *s > '9') return 0;
        v = v * 10 + (unsigned)(*s - '0');
        if (v > UINT32_MAX) return 0;
    }
    g_fs_buffer_count = (uint32_t)v; return 1;
}
static int fs_buffer_sbuf_limit_read(const char *s) {
    g_fs_buffer_sbuf_max = 65536;
    if (!s || !*s || !strcmp(s, "0")) return 1;
    uint64_t v = 0;
    for (; *s; s++) {
        if (*s < '0' || *s > '9') return 0;
        v = v * 10 + (unsigned)(*s - '0');
        if (v > (8u << 20)) return 0;
    }
    if (v < 65536 || v % 4) return 0;
    g_fs_buffer_sbuf_max = (uint32_t)v; return 1;
}
static int fs_image_binding_read(const char *s) {
    g_fs_image_binding = -1; g_fs_image_copies = 0; g_fs_image_bytes = 0;
    if (!s || !*s) return 1;
    uint64_t v = 0;
    for (; *s; s++) {
        if (*s < '0' || *s > '9') return 0;
        v = v * 10 + (unsigned)(*s - '0');
        if (v > UINT16_MAX) return 0;
    }
    if (!g_fs_buffer_count) return 0; // refuse an unbounded/ambiguous draw selection
    g_fs_image_binding = (int)v; return 1;
}
static void fs_buffer_poll(void) {
    const char *e = getenv("DM_VK_TRACE_FS_BUFFERS");
    // Loading/title submissions do not consume the capture window: only presentations after
    // the first matching draw count. Pending selection owns no readback allocations.
    if (g_fs_buffer_key && g_fs_buffer_seen && g_tex_frame >= g_fs_buffer_start &&
        g_tex_frame - g_fs_buffer_start >= g_fs_buffer_frames) {
        VKLOG("graphics buffer capture complete/expired: %u draws, %u readbacks reserved", g_fs_buffer_seen, g_fs_buffer_copies);
        g_fs_buffer_key = 0;
    }
    if (!e || !*e || !strcmp(e, "0")) return;
    char request[40]; snprintf(request, sizeof request, "%s", e); setenv("DM_VK_TRACE_FS_BUFFERS", "0", 1);
    g_fs_buffer_key = 0; g_fs_buffer_seen = g_fs_buffer_copies = 0; g_fs_buffer_bytes = 0; g_fs_buffer_last = UINT64_MAX;
    char *end; uint64_t key;
    const char *digits = !strncmp(request, "0x", 2) ? request + 2 : request;
    if (!fs_buffer_number(digits, 16, &key) || !key) { VKLOG("graphics buffer capture: invalid fragment key"); return; }
    g_fs_buffer_w = g_fs_buffer_h = 128;
    if ((e = getenv("DM_VK_TRACE_FS_SIZE")) && *e) {
        unsigned w, h; char tail;
        if (sscanf(e, "%ux%u%c", &w, &h, &tail) != 2 || !w || !h || w > 16384 || h > 16384) {
            VKLOG("graphics buffer capture: invalid target size"); return;
        }
        g_fs_buffer_w = w; g_fs_buffer_h = h;
    }
    if (!fs_buffer_sbuf_limit_read(getenv("DM_VK_TRACE_FS_SBUF_MAX"))) { VKLOG("graphics buffer capture: invalid storage-buffer byte cap"); return; }
    if (!fs_buffer_count_read(getenv("DM_VK_TRACE_FS_COUNT"))) { VKLOG("graphics buffer capture: invalid direct draw count"); return; }
    if (!fs_image_binding_read(getenv("DM_VK_TRACE_FS_IMAGE_BINDING"))) { VKLOG("graphics image capture: invalid binding or missing exact direct count"); return; }
    if (!fs_buffer_windows_read(getenv("DM_VK_TRACE_FS_WINDOWS"))) { g_fs_buffer_windows_n = 0; VKLOG("graphics buffer capture: invalid windows"); return; }
    e = getenv("DM_VK_TRACE_FS_FRAMES"); long frames = e ? strtol(e, &end, 10) : 3;
    if (e && (*end || end == e)) frames = 3;
    g_fs_buffer_frames = frames < 1 ? 1 : frames > 3 ? 3 : (unsigned)frames;
    g_fs_buffer_key = key;
    VKLOG("graphics buffer capture armed: FS %016llx target %ux%u, first matching draw per present, at most %u presents/64 copies/48 MiB, direct-count=%u (0=any), storage-max=%u (other buffers 64 KiB)",
        (unsigned long long)key, g_fs_buffer_w, g_fs_buffer_h, g_fs_buffer_frames, g_fs_buffer_count, g_fs_buffer_sbuf_max);
}
static int fs_buffer_match(uint64_t key, uint32_t width, uint32_t height, uint32_t count, int indirect_or_mesh) {
    if (g_fs_buffer_count && (indirect_or_mesh || count != g_fs_buffer_count)) return 0;
    if (!g_fs_buffer_key || key != g_fs_buffer_key || width != g_fs_buffer_w || height != g_fs_buffer_h ||
        g_fs_buffer_last == g_tex_frame || g_fs_buffer_seen >= g_fs_buffer_frames || g_fs_buffer_copies >= 64) return 0;
    if (g_fs_buffer_seen && (g_tex_frame < g_fs_buffer_start || g_tex_frame - g_fs_buffer_start >= g_fs_buffer_frames)) return 0;
    if (!g_fs_buffer_seen) g_fs_buffer_start = g_tex_frame;
    g_fs_buffer_last = g_tex_frame; g_fs_buffer_seen++;
    return 1;
}

static void x_read(void) { light_diag_poll(); original_index_read(); set_profile_poll(); cs_trace_poll(); fs_buffer_poll();
    { const char *e = getenv("DM_VK_POOL_STRICT"); g_pool_strict = e && atoi(e); }
    { const char *e = getenv("DM_VK_POOL_DUMMY"); int lo, hi; if (e && sscanf(e, "%d-%d", &lo, &hi) == 2) g_pdum_lo = lo, g_pdum_hi = hi; else g_pdum_lo = -1; }
    { const char *e = getenv("DM_VK_POOL_DUMMY_KINDS"); g_pdum_kinds = e ? (unsigned)strtoul(e, NULL, 0) : 0;   // bits 0-3: 2D, 2D array, cube, 3D tables
      g_pdum_nf = 0; for (const char *q = getenv("DM_VK_POOL_DUMMY_FMT"); q && *q && g_pdum_nf < 16;) { char *n; unsigned long v = strtoul(q, &n, 16); if (n == q) break; g_pdum_f[g_pdum_nf++] = (uint32_t)v; q = *n ? n + 1 : n; } }
    { const char *e = getenv("DM_VK_UPSCALER_DIAG"); int n = e ? atoi(e) : 0; if (n < 0) n = 0; if (n > 256) n = 256;
      if (!n) g_up_diag_left = 0; else if (n != g_up_diag_request) g_up_diag_left = n;
      g_up_diag_request = n; }
 const char *c = getenv("DM_VK_CB_DRAWS"); g_x_cbdraws = c ? atoi(c) : 128; g_sub_pass = 0;
    { const char *e = getenv("DM_VK_GMEM_LOG"); g_gmem_log = e && atoi(e); }
    { const char *e = getenv("DM_VK_INDIRECT_TOUCH"); int mode = e ? atoi(e) : 0; if (mode != 1 && mode != 2) mode = 0;
      if (mode != g_indirect_touch) VKLOG("indirect transfer touch mode %d (diagnostic: 0 off, 1 split and read, 2 split only)", mode); g_indirect_touch = mode; }
    const char *fy = getenv("DM_VK_FLIPY"); g_flipy = !(fy && !atoi(fy));   // on unless DM_VK_FLIPY=0
    if (g_x_listpasses) { g_x_listpasses = 0; setenv("DM_VK_TRACE_PASS", "-1", 1); }
    const char *tp = getenv("DM_VK_TRACE_PASS"); g_x_tracepass = tp ? atoi(tp) : -1;
    { const char *e = getenv("DM_VK_ORDER_KEEP"); g_order_keep = e ? atoi(e) != 0 : DM_GAME_ORDER_KEEP; g_order_pending = g_order_check = 0; }
    { static uint64_t until; static int done; const char *la = getenv("DM_VK_LIST_AT");   // DM_VK_LIST_AT=<present>: the pass list of every submission over two presents from there
      if (la && !done && g_tex_frame >= (uint64_t)atoll(la)) { done = 1; until = g_tex_frame + 2; }
      if (until && g_tex_frame < until) { g_x_tracepass = -2; g_x_listpasses = 1; g_lst_nrd = 0; VKLOG("--- submission at present %llu", (unsigned long long)g_tex_frame); } }
    { const char *w = getenv("DM_VK_WATCH"); static char last[40]; if (!w || !*w || !strcmp(w, "0")) g_watch_off = -1;
      else if (strcmp(w, last)) { g_watch_off = strtoll(w, NULL, 16); const char *h = strchr(w, '#'); g_watch_n = h ? atoi(h + 1) : 20; }
      snprintf(last, sizeof last, "%s", w ? w : ""); }
    { const char *w = getenv("DM_VK_UBO_LOG"); static char last[40]; if (!w || !*w || !strcmp(w, "0")) g_ubolog_key = 0;
      else if (strcmp(w, last)) { g_ubolog_key = strtoull(w, NULL, 16); const char *h = strchr(w, '#'); g_ubolog_n = h ? atoi(h + 1) : 40; }
      snprintf(last, sizeof last, "%s", w ? w : ""); }
    if (g_x_traceframe_on) { g_x_traceframe_on = 0; setenv("DM_VK_TRACE_FRAME", "", 1); }   // one submission traced
    { const char *tf = getenv("DM_VK_TRACE_FRAME"); snprintf(g_x_traceframe, sizeof g_x_traceframe, "%s", tf && strcmp(tf, "0") ? tf : ""); }
    { const char *tz = getenv("DM_VK_TRACE_SIZE"); snprintf(g_x_tracesize, sizeof g_x_tracesize, "%s", tz ? tz : ""); } const char *e; g_x_nobar = (e = getenv("DM_VK_X_NOBAR")) && atoi(e); g_x_depth = (e = getenv("DM_VK_X_DEPTH")) && atoi(e); g_x_color = (e = getenv("DM_VK_X_COLOR")) && atoi(e);
    g_drv_compact = !((e = getenv("DM_VK_DRV_COMPACT")) && !atoi(e)); g_buffer_memo = !((e = getenv("DM_VK_BUFFER_MEMO")) && !atoi(e));
    g_x_nosetc = (e = getenv("DM_VK_X_NOSETC")) && atoi(e) == 1; g_x_nosetc_bits = e ? atoi(e) & ~1 : 0; g_x_nodrvc = (e = getenv("DM_VK_X_NODRVC")) && atoi(e); g_x_landcut = (e = getenv("DM_VK_X_LANDCUT")) && atoi(e);
    g_x_nodispbar = (e = getenv("DM_VK_X_NODISPBAR")) && atoi(e);
    g_ubo_bounds = !((e = getenv("DM_VK_UBO_BOUNDS")) && !atoi(e));
    g_x_nokeep = (e = getenv("DM_VK_X_NOKEEP")) && atoi(e); g_x_nobc = (e = getenv("DM_VK_X_NOBC")) && atoi(e); g_x_noshc = (e = getenv("DM_VK_X_NOSHC")) && atoi(e); g_x_cutany = (e = getenv("DM_VK_X_CUTANY")) && atoi(e); g_x_nogs = (e = getenv("DM_VK_X_NOGS")) && atoi(e); g_spv_alt = (e = getenv("DM_VK_SPV_ALT")) && atoi(e); g_spv_key = (e = getenv("DM_VK_SPV_KEY")) ? strtoull(e, NULL, 16) : 0; g_rd_touch = !((e = getenv("DM_VK_RD_TOUCH")) && !atoi(e));
    { uint64_t key = (e = getenv("DM_VK_FDZ_FS")) ? strtoull(e, NULL, 16) : 0; if (key != g_fdz_fs) { g_fdz_fs = key; VKLOG("FDZ fragment diagnostic: %016llx (depth bias still excluded)", (unsigned long long)key); } }
    { uint64_t key = 0; int func = 0; char *end;
      if ((e = getenv("DM_VK_DEPTH_FUNC_FS")) && *e) { key = strtoull(e, &end, 16); if (*end == ':') { long f = strtol(end + 1, &end, 0); if (!*end && f >= 1 && f <= 8) func = (int)f; } if (!func) key = 0; }
      if (!key) func = 0;
      if (key != g_depth_func_fs || func != g_depth_func) { g_depth_func_fs = key; g_depth_func = func; VKLOG("depth function diagnostic: %016llx func %d (depth enable/write unchanged)", (unsigned long long)key, func); } }
    g_lrz_touch = (e = getenv("DM_VK_LRZ_TOUCH")) ? atoi(e) : 1; if (g_lrz_touch == 1) g_lrz_touch = LRZ_ALL;   /* 0 none, 1 all, else LRZ_* bits */ g_fdz = !((e = getenv("DM_VK_FDZ")) && !atoi(e)); g_xfb_on = !((e = getenv("DM_VK_XFB")) && !atoi(e));
    g_pace = !((e = getenv("DM_VK_PART_PACE")) && !atoi(e)); g_psplit = (e = getenv("DM_VK_PASS_SPLIT")) ? atoi(e) : 0; g_setsplit = (e = getenv("DM_VK_SETSPLIT")) && atoi(e); g_psplit_depth = (e = getenv("DM_VK_PASS_SPLIT_DEPTH")) && atoi(e);
    g_x_rdkeep = (e = getenv("DM_VK_X_RDKEEP")) && atoi(e); g_vbound = !((e = getenv("DM_VK_VBOUND")) && !atoi(e)); g_x_logfs = (e = getenv("DM_VK_LOG_FS")) ? strtoull(e, NULL, 16) : 0; g_x_nflat = 0; if ((e = getenv("DM_VK_X_FLAT_VS"))) for (char *q = (char *)e; *q && g_x_nflat < 8; ) { char *nx; uint64_t v = strtoull(q, &nx, 16); if (nx == q) break; int m = 7; if (*nx == ':') m = (int)strtol(nx + 1, &nx, 10); g_x_flat_mode[g_x_nflat] = m; g_x_flat[g_x_nflat++] = v; q = *nx ? nx + 1 : nx; } g_samp_snap = !((e = getenv("DM_NVN_SAMP_SNAP")) && !atoi(e)); g_lin_back = !((e = getenv("DM_VK_LIN_BACK")) && !atoi(e)); g_async_pipe = !((e = getenv("DM_VK_ASYNC_PIPE")) && !atoi(e)); g_prof_on = (e = getenv("DM_VK_PROF_SLOW")) && atoi(e); { extern volatile int g_vk_samp_log; if ((e = getenv("DM_VK_SAMP_LOG")) && atoi(e) > 0) { g_vk_samp_log = atoi(e); setenv("DM_VK_SAMP_LOG", "0", 1); } } g_x_nosetmemo = (e = getenv("DM_VK_X_NOSETMEMO")) && atoi(e); g_x_setverify = (e = getenv("DM_VK_X_SETVERIFY")) && atoi(e);
    g_x_nskipvs = 0; for (const char *q = getenv("DM_VK_X_SKIPVS"); q && *q && g_x_nskipvs < 8;) { char *n; uint64_t v = strtoull(q, &n, 16); if (n == q) break; if (v) g_x_skipvs[g_x_nskipvs++] = v; q = *n ? n + 1 : n; } }
// ---------------- conditional rendering (profile render_cond) ----------------
// As nvn_mtl.m (OP_RENDER_COND): SetRenderEnableConditional draws while the 64-bit counters of the reports at addr and addr + 16 are equal / differ, until SetRenderEnable(true).
// Evaluated on the CPU at replay from the records in memory: a report replayed earlier in this submission is not written yet (that happens when the GPU completes), so a timestamp
// one counts with its known counter 0 (vk_sub_complete writes 0 there under the key) and any other one is unknown, which keeps drawing, until the GPU has completed
// its submission and the counter is in memory (g_rc_occ). Inside a block evaluated false, draws and
// clears are dropped. Mario vs. Donkey Kong (gsys, as Tears of the Kingdom and Animal Crossing) clears its HDR scene target under such a condition after lighting it: drawn
// anyway, its levels were black under the HUD
// Upstream ea0c98df3 / 04419bc25 (port): a timestamp report's known counter outlives the submission that replayed it until the GPU has completed that submission (vk_fence_done):
// a later submission's condition, replayed while the report's bytes are not written yet, read the old ones (with DM_NVN_ASYNC or a GPU behind, the next submission is replayed
// before the last one completes). A later buffer write over a known report (copy, fill, inline update) makes it unknown again (keeps drawing). DM_VK_RC_KEEP=0 (start-up): per
// submission, as before. Render-enable state belongs to the NVN queue upstream and survives its submissions: DM_VK_RC_QUEUE=1 (start-up; off by default here) keeps g_rc_skip per
// queue across submissions instead of clearing it at each
static int g_rc_skip;   // a render-enable block evaluated false is active
#define RC_PEND 256
static struct { uint64_t addr, fence; int kind; } g_rc_pend[RC_PEND]; static int g_rc_npend;   // reports replayed and not yet landed: kind 1 = timestamp (counter 0), 2 = another counter or overwritten (unknown); fence: the submission's
static int rc_keep(void) { static int on = -1; if (on < 0) { const char *e = getenv("DM_VK_RC_KEEP"); on = !(e && *e == '0'); } return on; }
static int rc_queue_on(void) { static int on = -1; if (on < 0) { const char *e = getenv("DM_VK_RC_QUEUE"); on = e && *e == '1'; if (on) VKLOG("DM_VK_RC_QUEUE=1: render-enable state kept per queue across submissions"); } return on; }
static void rc_overwrite(uint64_t d, uint64_t n) {   // a buffer write replayed over [d, d + n): reports there are unknown until it lands
    for (int i = 0; i < g_rc_npend; i++) if (g_rc_pend[i].addr < d + n && d < g_rc_pend[i].addr + 16) { g_rc_pend[i].kind = 2; g_rc_pend[i].fence = mtl_last_fence() + 1; }
}
// Addresses written by a non-timestamp report, with the submission of the latest one: a condition on one keeps drawing until that submission's reports have landed, then reads
// the counter the GPU wrote. They were unknown for good, so a game that culls by occlusion queries drew everything: Ocarina of Time's main pass in Kokiri Forest drew up to
// 958 draws (8.6 ms of GPU, Odin 3) where it draws at most 178 (3.1 ms) with its own results, 112,569 of 155,344 conditions in 300 frames evaluating false. A set that grows: the fixed 4096-slot table this was filled up in The Legend of
// Zelda: Ocarina of Time (its reports move through their buffer as the camera turns), after which every report and every condition walked all 4096 slots (0.9 s of the
// replay thread's 4.4 s over 8 s of camera turns, Odin 3) and new addresses were no longer recorded. Replay thread only
static nvn_hmap g_rc_occ = {.klen = 8};
static void rc_occ_add(uint64_t a) {   // the value: the submission that writes it (as rc_note's fence)
    if (nvn_hm_put(&g_rc_occ, &a, (void *)(uintptr_t)(mtl_last_fence() + 1))) return;
    if (g_rc_occ.n >= 4096 && !(g_rc_occ.n & (g_rc_occ.n - 1))) VKLOG("render conditions: %u addresses written by occlusion reports so far", g_rc_occ.n);
}
static uint64_t rc_occ_fence(uint64_t a) { return (uint64_t)(uintptr_t)nvn_hm_get(&g_rc_occ, &a); }
static void rc_note(uint64_t a, int kind) {
    if (kind == 2) rc_occ_add(a);
    uint64_t f = mtl_last_fence() + 1;   // the submission being replayed (as OP_FENCE_SYNC)
    for (int i = 0; i < g_rc_npend; i++) if (g_rc_pend[i].addr == a) { g_rc_pend[i].kind = kind; g_rc_pend[i].fence = f; return; }
    if (g_rc_npend < RC_PEND) { g_rc_pend[g_rc_npend].addr = a; g_rc_pend[g_rc_npend].fence = f; g_rc_pend[g_rc_npend++].kind = kind; }
}
static int rc_value(uint64_t a, uint64_t *v) {   // 0 = unknown at replay time
    uint64_t done = vk_reports_fence_done(), of = rc_occ_fence(a);
    if (of > done) return 0;
    for (int i = g_rc_npend - 1; i >= 0; i--) if (g_rc_pend[i].addr == a && g_rc_pend[i].fence > done) { if (g_rc_pend[i].kind != 1) return 0; *v = 0; return 1; }
    *v = *(const volatile uint64_t *)(uintptr_t)a; return 1;
}
static void render_cond(const p_render_cond *rc) {
    if (rc->mode >= 2) { g_rc_skip = rc->mode == 3; return; }
    uint64_t x = 0, y = 0; int known = rc_value(rc->addr, &x) & rc_value(rc->addr + 16, &y);
    g_rc_skip = known && (rc->mode ? x == y : x != y);
}

void vk_exec_reset(void) {   // a new submission: NVN state does not carry over between command buffers, the replay starts from the defaults
    static int inited; if (!inited) { inited = 1; state_reset(); crash_add_note_hook(build_dump); prewarm_start(); compute_warm_start(); }
    if (rc_keep() && g_rc_npend) {   // reports whose submission has not completed keep what is known of them (above)
        uint64_t done = vk_reports_fence_done(); int n = 0;
        for (int i = 0; i < g_rc_npend; i++) if (g_rc_pend[i].fence > done) g_rc_pend[n++] = g_rc_pend[i];
        g_rc_npend = n;
    } else g_rc_npend = 0;
    if (nvn_logical_queues_on()) { /* persistent state restored by mtl_queue_activate */ }
    else if (rc_queue_on()) {   // render-enable state per NVN queue (mtl_queue_activate, gfx_common.c), saved at the next submission's start
        extern uint64_t g_nvn_active_queue; enum { RCQ = 16 }; static struct { uint64_t serial; int skip; } q[RCQ]; static uint64_t cur; static unsigned nq;
        if (cur) { unsigned i = 0; while (i < nq && q[i].serial != cur) i++; if (i == nq && nq < RCQ) q[nq++].serial = cur; if (i < nq) q[i].skip = g_rc_skip; }
        cur = g_nvn_active_queue; g_rc_skip = 0; for (unsigned i = 0; i < nq; i++) if (q[i].serial == cur) g_rc_skip = q[i].skip;
    } else g_rc_skip = 0;
    { void vk_tex_release_drain(void); vk_tex_release_drain(); }   // Finalized textures' CPU-written marks that no queued job needs any more (gfx_vk_tex.c)
    S.pass_open = 0; S.bound_pipe = VK_NULL_HANDLE; g_cb_draws = 0; g_ord_parts = 0; g_psplit_now = 0; g_set_epoch++; x_read();
    g_sub_hist[g_sub_hn++ % 8] = g_vkst.draws - g_sub_d0; g_sub_d0 = g_vkst.draws;   // (part_pace: the submission just ended)
    if (g_prof_on) { extern uint64_t g_pkg_ns, g_pkg_n; memset(g_prof_ns, 0, sizeof g_prof_ns); g_prof_t0 = tick(); g_prof_pkg0 = g_pkg_ns; g_prof_npkg0 = g_pkg_n; g_prof_pipes0 = g_vkst.pipes_built; }
    { extern volatile int g_vk_samp_log; if (g_vk_samp_log > 0) VKLOG("submission begins (present %llu, thread %d)", (unsigned long long)g_tex_frame, (int)gettid()); }
}

// ---------------- render passes ----------------
// A pass is begun lazily: begin_pass() only sets it up (pass_open 2), and the first command that must run inside it begins it (pass_commit, pass_open 1).
// Until then a clear of a whole attachment becomes that attachment's load op, so a tiler clears in tile memory instead of loading the
// attachment from memory and clearing it with a draw; a pass that is set up and never used is dropped.
static VkRenderingAttachmentInfo P_ca[8], P_da, P_sa; static VkRenderingInfo P_ri; static int P_cleared; static int P_split;   // the pass goes on with the one before it (a DM_VK_PASS_SPLIT cut)
// NVN may bind one subresource in multiple colour slots while masking all but
// one writer. Separate Vulkan attachments would load/store the same image twice
// (MK8's full-resolution normals in slots 3/4), which can lose the writer's contents.
// Keep writable aliases and route inactive duplicates to the first writer, or
// the first slot when none writes. Distinct views/layers are never combined.
static VkImageView P_alias_views[8]; static uint8_t P_alias[8];
static void color_alias_map(uint32_t n, const VkImageView views[8], uint32_t mask, uint8_t map[8]) {
    for (uint32_t i=0;i<8;i++) map[i]=(uint8_t)i;
    for (uint32_t i=0;i<n && i<8;i++) if (views[i] && map[i]==i) {
        uint32_t keep=i;
        for (uint32_t j=0;j<n && j<8;j++) if (views[j]==views[i] && ((mask>>(4*j))&15u)) {keep=j;break;}
        for (uint32_t j=0;j<n && j<8;j++) if (j!=keep && views[j]==views[i] && !((mask>>(4*j))&15u)) map[j]=(uint8_t)keep;
    }
}
static void bind_chmask(uint32_t mask) {
    if (S.pass_open && mask!=S.chmask.mask) {
        uint8_t map[8];color_alias_map((uint32_t)S.ncol,P_alias_views,mask,map);
        if (memcmp(map,P_alias,sizeof map)) vk_end_pass();
    }
    S.chmask.mask=mask;
}

// Render passes. Adreno's driver spent about 40 µs in each vkCmdBeginRendering (a quarter of the replay thread in Xenoblade Chronicles 3's battles, 128 passes a
// frame); a VkRenderPass is made once per set of formats and load operations, and an imageless framebuffer once per set of attachment descriptions, and passes
// then begin with vkCmdBeginRenderPass (the battle's replay went from about 20 to 17.5 ms a frame). Pipelines are made against a compatible render pass (the same
// formats, every load operation LOAD). DM_VK_RP=0 (start-up): dynamic rendering
typedef struct { uint32_t ncol, cfmt[8], cload[8], dfmt, dload, sload, dstore, samples, cstore[8]; } rpkey;   // cstore: 0 (memset) STORE; DONT_CARE for slot 0's scratch (DM_VK_SLOT0_FILL)
typedef struct { VkRenderPass rp; uint32_t w, h, layers, n; struct { uint32_t usage, flags, w, h, layers, fmt, fmt2, fmt3, fmt4; } a[9]; } fbkey;   // fmt2..fmt4: a format-listed image's other formats (an imageless framebuffer names the image's whole list)
static nvn_hmap g_rps = {.klen = sizeof(rpkey)}, g_fbs = {.klen = sizeof(fbkey)}; static pthread_mutex_t g_rp_mu = PTHREAD_MUTEX_INITIALIZER;
// Without dynamic rendering (a Vulkan 1.1 / 1.2 device, V.has_dynrender) passes must be render pass objects, whatever DM_VK_RP says (vk_init refuses a device with neither)
static int rp_on(void) { static int on = -1; if (on < 0) { const char *e = getenv("DM_VK_RP"); on = V.has_imageless && (!V.has_dynrender || !(e && !atoi(e))); VKLOG("passes: %s", on ? "render pass objects with imageless framebuffers" : "dynamic rendering"); } return on; }
// A pass with no attachments at all (begin_pass, DM_VK_NOATT) is begun with dynamic rendering even where passes are render pass objects (an imageless framebuffer
// with no attachments was the first suspect for the hang below; it hung either way). Its pipelines are made the same way
static int noatt_dyn(uint32_t ncol, uint32_t dfmt) { return !ncol && !dfmt && V.has_dynrender; }
// DM_VK_NOATT=1 (start-up) runs them; off by default: with them run, Welcome Tour hung the GPU the same way as soon as its lobby began (both pass kinds; kgsl fault
// ids 2, the game alive at "-- FPS"), so the fault is in the draws themselves: vertex shaders be816c4f2431b602 / afe80dc02de844a6 (subgroup operations emulated as
// one-invocation subgroups) writing memory under rasterizer discard, 96x54, 2048x2048 and 1312x1312 viewports, about 76 draws a frame. Skipped, the lobby draws
// without visible loss, so what they compute (culling or skinning records, most likely) is not needed by what the host draws, or is covered by a fallback
static int noatt_on(void) { static int on = -1; if (on < 0) { const char *e = getenv("DM_VK_NOATT"); on = e ? atoi(e) : 0; } return on; }
// What those draws are (shaders read from the bundle, 2026-11-08): be816c4f2431b602 (and its variant 7ff0cbe988494368) culls one 80-byte instance record per vertex
// (s0 at c0 0x110) against the view, picks a bucket (clamped to c5[8].y) and appends the record to that bucket's list: a counter per bucket in s1 (c0 0x150, an atomic
// add, warp-aggregated with MATCH and a ballot) gives the slot, written at s2 (c0 0x160) + (bucket * c5[7].z + slot) * 80. afe80dc02de844a6 then copies the N = c5[8].x
// counters into the instance counts (+4) of N 20-byte indexed indirect records per vertex (s1 at c0 0x150), which the following indirect draws of the pass use. Neither has
// a loop that does not end (be816 has none, afe80's runs N times) and the one-invocation subgroup rewrite is exact for them (spirv-val clean): the GPU's view of what they
// address is what can fault: a counter never reset or counted more than once (slots past a bucket's room, instance counts past the records), a range outside the pools
// (bound to the 64 KB zero buffer), or the counters' reset not yet landed. So with DM_VK_NOATT=1 each such draw gets a pass of its own, with the deferred uniform updates
// landed before it (a draw sees them only through uniform bindings, not through storage buffers or indirect arguments) and a full barrier after it, as NVN runs them in order
// (DM_VK_NOATT_ISO=0, start-up: as before, the draws of a run share a pass and the updates stay deferred). DM_VK_NOATT_LOG=<n> (start-up): the first n such draws, and the
// first n indirect draws after them, logged with their storage and constant buffers as the CPU sees them (noatt_log). DM_VK_ROBUST=1 (gfx_vk_core.c): out-of-range accesses
// do not fault, to tell an out-of-range address from the rest
static int noatt_iso(void) { static int on = -1; if (on < 0) { const char *e = getenv("DM_VK_NOATT_ISO"); on = e ? atoi(e) : 1; } return on; }
static int noatt_log_n(void) { static int n = -1; if (n < 0) { const char *e = getenv("DM_VK_NOATT_LOG"); n = e ? atoi(e) : 0; } return n; }
static int g_noatt_open, g_noatt_logged, g_noatt_ind_logged;   // the open pass has no attachments (and is to end after its draw); draws logged so far
static int fmt_has_depth(uint32_t f) { return f && f != VK_FORMAT_S8_UINT; }
static int fmt_has_stencil(uint32_t f) { return f == VK_FORMAT_D24_UNORM_S8_UINT || f == VK_FORMAT_D32_SFLOAT_S8_UINT || f == VK_FORMAT_S8_UINT || f == VK_FORMAT_D16_UNORM_S8_UINT; }
static VkRenderPass rp_get(const rpkey *k) {
    pthread_mutex_lock(&g_rp_mu); VkRenderPass rp = nvn_hm_get(&g_rps, k);
    if (!rp) {
        VkAttachmentDescription2 ad[9]; VkAttachmentReference2 cr[8], dr; uint32_t na = 0;
        for (uint32_t i = 0; i < k->ncol; i++) {
            if (!k->cfmt[i]) { cr[i] = (VkAttachmentReference2){VK_STRUCTURE_TYPE_ATTACHMENT_REFERENCE_2, NULL, VK_ATTACHMENT_UNUSED, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_ASPECT_COLOR_BIT}; continue; }
            ad[na] = (VkAttachmentDescription2){VK_STRUCTURE_TYPE_ATTACHMENT_DESCRIPTION_2, NULL, 0, (VkFormat)k->cfmt[i], (VkSampleCountFlagBits)k->samples, (VkAttachmentLoadOp)k->cload[i], (VkAttachmentStoreOp)k->cstore[i],
                VK_ATTACHMENT_LOAD_OP_DONT_CARE, VK_ATTACHMENT_STORE_OP_DONT_CARE, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL};
            cr[i] = (VkAttachmentReference2){VK_STRUCTURE_TYPE_ATTACHMENT_REFERENCE_2, NULL, na++, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_ASPECT_COLOR_BIT};
        }
        if (k->dfmt) {
            int hd = fmt_has_depth(k->dfmt), hs = fmt_has_stencil(k->dfmt);
            ad[na] = (VkAttachmentDescription2){VK_STRUCTURE_TYPE_ATTACHMENT_DESCRIPTION_2, NULL, 0, (VkFormat)k->dfmt, (VkSampleCountFlagBits)k->samples,
                hd ? (VkAttachmentLoadOp)k->dload : VK_ATTACHMENT_LOAD_OP_DONT_CARE, hd ? (VkAttachmentStoreOp)k->dstore : VK_ATTACHMENT_STORE_OP_DONT_CARE,
                hs ? (VkAttachmentLoadOp)k->sload : VK_ATTACHMENT_LOAD_OP_DONT_CARE, hs ? (VkAttachmentStoreOp)k->dstore : VK_ATTACHMENT_STORE_OP_DONT_CARE, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL};
            dr = (VkAttachmentReference2){VK_STRUCTURE_TYPE_ATTACHMENT_REFERENCE_2, NULL, na++, VK_IMAGE_LAYOUT_GENERAL, (hd ? VK_IMAGE_ASPECT_DEPTH_BIT : 0) | (hs ? VK_IMAGE_ASPECT_STENCIL_BIT : 0)};
        }
        VkSubpassDescription2 sd = {VK_STRUCTURE_TYPE_SUBPASS_DESCRIPTION_2, NULL, 0, VK_PIPELINE_BIND_POINT_GRAPHICS, 0, 0, NULL, k->ncol, cr, NULL, k->dfmt ? &dr : NULL, 0, NULL};
        VkRenderPassCreateInfo2 ci = {VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO_2, NULL, 0, na, ad, 1, &sd, 0, NULL, 0, NULL};
        if (vkCreateRenderPass2(V.dev, &ci, NULL, &rp)) { VKRED_ONCE(70, "vkCreateRenderPass2 failed"); rp = VK_NULL_HANDLE; } else nvn_hm_put(&g_rps, k, rp);
    }
    pthread_mutex_unlock(&g_rp_mu); return rp;
}
static VkRenderPass rp_compat(uint32_t ncol, const uint32_t *cfmt, uint32_t dfmt, uint32_t samples) {   // for pipelines: every load operation LOAD
    rpkey k; memset(&k, 0, sizeof k); k.ncol = ncol; for (uint32_t i = 0; i < ncol && i < 8; i++) { k.cfmt[i] = cfmt[i]; k.cload[i] = VK_ATTACHMENT_LOAD_OP_LOAD; }
    k.dfmt = dfmt; k.dload = k.sload = VK_ATTACHMENT_LOAD_OP_LOAD; k.dstore = VK_ATTACHMENT_STORE_OP_STORE; k.samples = samples ? samples : 1; return rp_get(&k);
}
static VkFramebuffer fb_get(const fbkey *k) {
    pthread_mutex_lock(&g_rp_mu); VkFramebuffer fb = nvn_hm_get(&g_fbs, k);
    if (!fb) {
        VkFramebufferAttachmentImageInfo ai[9]; VkFormat vf[9][4];
        for (uint32_t i = 0; i < k->n; i++) { uint32_t c[4] = {k->a[i].fmt, k->a[i].fmt2, k->a[i].fmt3, k->a[i].fmt4}, nf = 0;   // the image's list, without the unset entries (its own format first)
            for (int j = 0; j < 4; j++) if (c[j] || (j == 0 && !nf)) vf[i][nf++] = (VkFormat)c[j];
            ai[i] = (VkFramebufferAttachmentImageInfo){VK_STRUCTURE_TYPE_FRAMEBUFFER_ATTACHMENT_IMAGE_INFO, NULL, k->a[i].flags, k->a[i].usage, k->a[i].w, k->a[i].h, k->a[i].layers, nf, vf[i]}; }
        VkFramebufferAttachmentsCreateInfo aci = {VK_STRUCTURE_TYPE_FRAMEBUFFER_ATTACHMENTS_CREATE_INFO, NULL, k->n, ai};
        VkFramebufferCreateInfo ci = {VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO, &aci, VK_FRAMEBUFFER_CREATE_IMAGELESS_BIT, k->rp, k->n, NULL, k->w, k->h, k->layers};
        if (vkCreateFramebuffer(V.dev, &ci, NULL, &fb)) { VKRED_ONCE(71, "vkCreateFramebuffer failed"); fb = VK_NULL_HANDLE; } else nvn_hm_put(&g_fbs, k, fb);
    }
    pthread_mutex_unlock(&g_rp_mu); return fb;
}
// the attachments begin_pass found, for the render pass begin: each colour slot's (and the depth's) image, view, view format and layers, size at the view's level
static struct { vkimg_t *im; VkImageView v; uint32_t fmt, layers, w, h; } P_att[9];
// Exact-draw MRT measurement: no original shader input or output is modified.
static uint64_t g_mrt_key; static uint32_t g_mrt_mode; static int g_mrt_active, g_mrt_observer;
static void mrt_poll(uint64_t done_fence);
static void mrt_attach(void);


// Before a pass that goes on with a depth buffer (loads it), one texel of its depth is copied out: a transfer read makes Adreno's driver drop the buffer's
// low-resolution depth (LRZ), which it otherwise kept from earlier passes even within one command buffer. NEO's menu → field wipe drew whole 32x32 squares of
// its old frame brown for two frames (the overlay's tiles rejected); with the read, never. Copying the colour target out instead, or a barrier alone, did not help.
// Depth-only buffers too: 13 Sentinels: Aegis Rim draws its frame (206 sprites, depth test off) and then copies it to the window in one quad, both passes on one
// D32; the quad's second triangle came out as a checkerboard of stale bins in every frame (either diagonal; a command-buffer cut or this read cleared it, the
// depth attachment not loaded did not). write: the texel is also written back (unchanged), a transfer write. DM_VK_LRZ_TOUCH=0 (a DM_CTL "env" line) leaves both touches out
static void lrz_touch(vkimg_t *im, int write) {
    static VkBuffer b; static VkDeviceMemory m;
    if (!im || !(im->aspect & VK_IMAGE_ASPECT_DEPTH_BIT) || im->samples > 1) return;   // (multisampled images cannot be copied to a buffer)
    if (!b) { VkBufferCreateInfo bi = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, NULL, 0, 256, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_SHARING_MODE_EXCLUSIVE};
        VKCHK(vkCreateBuffer(V.dev, &bi, NULL, &b)); VkMemoryRequirements rq; vkGetBufferMemoryRequirements(V.dev, b, &rq);
        VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL, rq.size, vk_mem_type(rq.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0)};
        VKCHK(vkAllocateMemory(V.dev, &ai, NULL, &m)); VKCHK(vkBindBufferMemory(V.dev, b, m, 0)); }
    extern uint64_t g_vk_nlrz; g_vk_nlrz++;
    VkCommandBuffer cb = vk_cmd();
    VkBufferImageCopy c = {0, 0, 0, {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 0, 1}, {0, 0, 0}, {1, 1, 1}};
    if ((g_lrz_touch & LRZ_CORNER) && im->w && im->h) c.imageOffset = (VkOffset3D){(int32_t)im->w - 1, (int32_t)im->h - 1, 0};
    vkCmdCopyImageToBuffer(cb, im->img, VK_IMAGE_LAYOUT_GENERAL, b, 1, &c); vk_img_barrier_all(cb);   // (begin_pass's barrier already ordered it after the writes)
    if (write) { vkCmdCopyBufferToImage(cb, b, im->img, VK_IMAGE_LAYOUT_GENERAL, 1, &c); vk_img_barrier_all(cb); }
}
// The depth image of the last pass that had one. When a pass draws with another, that one gets the texel written back: Adreno otherwise applied its LRZ in later
// passes on other depth buffers and rejected whole triangles there. Xenoblade Chronicles 3's cloud layer (one 720-index dome drawn into a 200x112 target, its D32
// depth downsampled from the scene's 776x440 D24S8 by a shader writing gl_FragDepth) lost whole triangles while the camera turned and kept losing them with the
// camera still at some angles: a primitive-ID dump showed whole triangles missing, triangles reaching past the screen edge kept, the same with the dome's depth
// forced to a constant; the dome drawn with gl_FragDepth written (no LRZ for it) lost none in 47 frames. Reading the dome's depth buffer before its pass (the
// read above), or writing it back, did not help; writing back every depth buffer before every pass that loads one did (0 broken frames of 53 against 17 of 46,
// GPU +1 ms), as did writing back only the one a pass switches away from (0 genuine drops in about 140 frames, GPU +0.6-0.8 ms of 13-18)
static vkimg_t *g_lrz_prev;
void vk_lrz_forget(vkimg_t *im) { if (g_lrz_prev == im) g_lrz_prev = NULL; }   // an image dropped (gfx_vk_tex.c img_drop)
// 0: no pass could be begun (no render pass or framebuffer, and no dynamic rendering to fall back on): the caller records nothing that needs one
static void pass_journal_pass(const char *event);
static int pass_commit(void) {
    if (S.pass_open != 2) return S.pass_open == 1;
    vkimg_t *dim = S.daspect ? P_att[8].im : NULL;
    if ((g_lrz_touch & LRZ_SWITCH) && dim && g_lrz_prev && g_lrz_prev != dim) lrz_touch(g_lrz_prev, 1);
    if (dim) g_lrz_prev = dim;
    if ((g_lrz_touch & LRZ_READ) && dim && P_da.loadOp == VK_ATTACHMENT_LOAD_OP_LOAD) lrz_touch(dim, 0);
    void vk_pass_ts(int end, uint32_t w, uint32_t h, uint32_t fmt, uint32_t draws); vk_pass_ts(0, 0, 0, 0, 0);
    if (rp_on() && !noatt_dyn(P_ri.colorAttachmentCount, P_ri.pDepthAttachment || P_ri.pStencilAttachment ? 1u : 0u)) {
        rpkey k; memset(&k, 0, sizeof k); k.ncol = P_ri.colorAttachmentCount; k.samples = S.samples ? S.samples : 1;
        fbkey fk; memset(&fk, 0, sizeof fk); VkImageView views[9]; VkClearValue cv[9]; uint32_t na = 0;
        for (uint32_t i = 0; i < k.ncol; i++) if (P_ca[i].imageView) {
            k.cfmt[i] = P_att[i].fmt; k.cload[i] = P_ca[i].loadOp; k.cstore[i] = P_ca[i].storeOp; cv[na] = P_ca[i].clearValue; views[na] = P_ca[i].imageView;
            fk.a[na].usage = P_att[i].im->usage; fk.a[na].flags = P_att[i].im->cflags; fk.a[na].w = P_att[i].w; fk.a[na].h = P_att[i].h; fk.a[na].layers = P_att[i].layers;
            fk.a[na].fmt = P_att[i].im->fixed_fmt ? (uint32_t)P_att[i].im->fmt : P_att[i].fmt; fk.a[na].fmt2 = (uint32_t)P_att[i].im->fmt2;
            fk.a[na].fmt3 = (uint32_t)P_att[i].im->fmt_cls[0]; fk.a[na].fmt4 = (uint32_t)P_att[i].im->fmt_cls[1]; na++; }   // (a format-listed image (DM_VK_UBWC): its whole list, as imageless framebuffers require)
        if (P_ri.pDepthAttachment || P_ri.pStencilAttachment) {
            k.dfmt = P_att[8].fmt; k.dload = P_da.loadOp; k.sload = P_sa.imageView ? P_sa.loadOp : VK_ATTACHMENT_LOAD_OP_DONT_CARE; k.dstore = P_da.storeOp;
            cv[na] = P_da.loadOp == VK_ATTACHMENT_LOAD_OP_CLEAR ? P_da.clearValue : P_sa.clearValue; if (P_sa.loadOp == VK_ATTACHMENT_LOAD_OP_CLEAR) cv[na].depthStencil.stencil = P_sa.clearValue.depthStencil.stencil;
            views[na] = P_da.imageView;
            fk.a[na].usage = P_att[8].im->usage; fk.a[na].flags = P_att[8].im->cflags; fk.a[na].w = P_att[8].w; fk.a[na].h = P_att[8].h; fk.a[na].layers = P_att[8].layers; fk.a[na].fmt = P_att[8].fmt; na++; }
        VkRenderPass rp = rp_get(&k);
        fk.rp = rp_compat(k.ncol, k.cfmt, k.dfmt, k.samples); fk.w = P_ri.renderArea.extent.width; fk.h = P_ri.renderArea.extent.height; fk.layers = P_ri.layerCount; fk.n = na;
        VkFramebuffer fb = rp && fk.rp ? fb_get(&fk) : VK_NULL_HANDLE;
        if (fb) { VkRenderPassAttachmentBeginInfo abi = {VK_STRUCTURE_TYPE_RENDER_PASS_ATTACHMENT_BEGIN_INFO, NULL, na, views};
            VkRenderPassBeginInfo rbi = {VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO, &abi, rp, fb, P_ri.renderArea, na, cv};
            vkCmdBeginRenderPass(vk_cmd(), &rbi, VK_SUBPASS_CONTENTS_INLINE); S.rp_pass = 1; }
        else if (V.has_dynrender) { vkCmdBeginRendering(vk_cmd(), &P_ri); S.rp_pass = 0; }
        else { VKRED_ONCE(76, "render pass or framebuffer failed without dynamic rendering: pass skipped"); return 0; }   // (still pass_open 2: vk_end_pass drops it)
    } else { vkCmdBeginRendering(vk_cmd(), &P_ri); S.rp_pass = 0; }
    S.pass_open = 1; g_vkst.passes++; g_sub_pass++; g_pass_wmem = 0;
    pass_journal_pass("begin");
    for (int i = 0; i < 9; i++) { vkimg_t *a = P_att[i].im; if (a && a->rt_last != g_tex_frame + 1) { a->rt_frames = a->rt_last == g_tex_frame ? a->rt_frames + 1 : 1; a->rt_last = g_tex_frame + 1; } }   // (async_pipe)
    { extern uint64_t g_cmd_gen; g_cmd_gen++; }   // the bind cache starts empty in each pass: Adreno does not keep the state a pass's draws set for the next pass (XC3 drew its lighting black)
    if (g_x_tracepass == -2) { g_x_listpasses = 1; list_reads("before it, dispatches");
        char t[512]; int n = 0; static const char LO[] = "LCD";   // load ops: Load, Clear, Don't care; each attachment's image by pool offset
        for (uint32_t i = 0; i < P_ri.colorAttachmentCount; i++) if (P_ca[i].imageView && P_att[i].im) n += snprintf(t + n, sizeof t - n, " c%u=%llx:%c(%ux%u)", i, (unsigned long long)P_att[i].im->off, LO[P_ca[i].loadOp < 3 ? P_ca[i].loadOp : 2], P_att[i].im->w, P_att[i].im->h);
        if (P_ri.pDepthAttachment && P_att[8].im) n += snprintf(t + n, sizeof t - n, " d=%llx:%c(%ux%u)", (unsigned long long)P_att[8].im->off, LO[P_da.loadOp < 3 ? P_da.loadOp : 2], P_att[8].im->w, P_att[8].im->h);
        VKLOG("--- pass %d: %ux%u, colour formats %d %d, depth format %d;%s", g_sub_pass - 1, S.pass_w, S.pass_h, S.cfmt[0], S.cfmt[1], S.dfmt, t); }
    if (g_x_traceframe[0]) {   // DM_VK_TRACE_FRAME=WxH[:fmt[:fmt]] (live): trace every pass of the next submission whose targets match (a whole effect chain in one log)
        char b[48]; snprintf(b, sizeof b, "%ux%u:%d:%d", S.pass_w, S.pass_h, S.cfmt[0], S.cfmt[1]);
        g_x_tracepass = !strncmp(b, g_x_traceframe, strlen(g_x_traceframe)) ? g_sub_pass - 1 : -1; g_x_traceframe_on = 1; }
    if (g_x_tracesize[0] && g_x_tracepass < 0) {   // DM_VK_TRACE_SIZE=WxH[:format[:format]] (live): trace the next pass whose first colour target is that (passes are numbered anew each frame)
        char b[48]; snprintf(b, sizeof b, "%ux%u:%d:%d", S.pass_w, S.pass_h, S.cfmt[0], S.cfmt[1]);
        const char *h = strchr(g_x_tracesize, '#'); size_t l = h ? (size_t)(h - g_x_tracesize) : strlen(g_x_tracesize);   // #n: the (n+1)th such pass
        if (!strncmp(b, g_x_tracesize, l)) { if (h && atoi(h + 1) > 0) { snprintf(g_x_tracesize + l, sizeof g_x_tracesize - l, "#%d", atoi(h + 1) - 1); setenv("DM_VK_TRACE_SIZE", g_x_tracesize, 1); }
            else { g_x_tracepass = g_sub_pass - 1; g_x_tracesize[0] = 0; setenv("DM_VK_TRACE_SIZE", "", 1); } } }
    if (g_x_tracepass >= 0 && g_sub_pass - 1 == g_x_tracepass) VKLOG("=== traced pass %d: %ux%u, colour formats %d %d, depth format %d", g_x_tracepass, S.pass_w, S.pass_h, S.cfmt[0], S.cfmt[1], S.dfmt); g_pass_draws = 0; g_pass_bb[0] = g_pass_bb[1] = INT32_MAX; g_pass_bb[2] = g_pass_bb[3] = INT32_MIN;
    return 1;
}
void vk_ubd_land(VkCommandBuffer cb); int vk_ubd_shadow(uint64_t a, uint64_t n, VkBuffer *buf, VkDeviceSize *off); void vk_ubd_read(uint64_t a, uint64_t n, uint8_t *dst);
// DM_VK_PASSDUMP=1 (a DM_CTL "env" line; =N waits N - 1 presents first): the next submissions' passes each leave their first colour target in a buffer (up to 48 passes), written as
// DM_ROOT/pass_NN.ppm when the GPU is done (RGBA8 as is, RGBA16F clamped to 0..1, with its NaN / infinite pixel count logged)
// DM_VK_PASSDUMP_SKIP=n (read when a dump starts): leave out each submission's first n passes, to reach a frame's later passes (Xenoblade Chronicles 3's title frame has about 200);
// =WxH: start at the first pass of that size (pass numbers move by tens from frame to frame)
// DM_VK_PASSDUMP_SIZE=WxH[,WxH...] (read when a dump starts): only passes whose first colour target has one of these sizes, over every submission of the frame (no depth),
// to follow one effect's targets when the pass numbers move from frame to frame (Xenoblade Chronicles 3's 384x216 clouds)
// DM_VK_PASSDUMP_FRAMES=N (read when a dump starts): keep dumping for N presents (up to 96 dumps in all), to catch an effect that is wrong in single frames
typedef struct { VkBuffer b; VkDeviceMemory m; void *p; uint32_t w, h; VkFormat f; uint64_t fence; int pass; uint64_t present; uint64_t raw_bytes, key, frame; uint32_t depth, binding, dispatch, level, layer; VkFormat view_format; int storage, after; } pdump_t;
enum { PD_MAX = 96, PD_DRAW_MAX = 256 };
static pdump_t g_pd[PD_DRAW_MAX]; static int g_npd, g_pd_state, g_pd_skip, g_pd_frames; static char g_pd_size[128], g_pd_from[32];   // state 0 off, 1 dumping this submission's passes, 2 waiting to write
// Metadata only: no texture lookup, descriptor mutation, tracing flag or GPU command.
enum { PD_JOURNAL_MAX = 65536 };
static int g_pd_strict, g_pd_journal;
static uint32_t g_pd_journal_rows;
static uint64_t g_pd_journal_first, g_pd_journal_end, g_pd_journal_last_fs;
// Deliberately intrusive diagnostic: one watched cube face, the presentation after
// strict-FS arming, at most one initial image plus 255 issued-draw checkpoints.
static int g_pd_draw_mode, g_pd_draw_emit, g_pd_draw_before;
static uint64_t g_pd_draw_present, g_pd_draw_serial;
static VkImage g_pd_draw_image;
static unsigned g_pd_draw_ordinal;
static int pass_dump_draw_match(uint64_t present, VkImage image, uint64_t serial,
                                uint32_t level, uint32_t layer, uint32_t w, uint32_t h, uint32_t format) {
    return g_pd_draw_mode && g_pd_state == 1 && g_npd < PD_DRAW_MAX &&
           present == g_pd_draw_present && image == g_pd_draw_image && serial == g_pd_draw_serial &&
           !level && !layer && w == 128 && h == 128 && format == VK_FORMAT_B10G11R11_UFLOAT_PACK32;
}
static int pass_dump_draw_ready(void) {
    if (!g_pd_draw_mode || S.pass_open != 1 || !S.ncol || !S.rt_has_col[0] || !P_att[0].im || S.rast_discard) return 0;
    const view_t *v = S.rt_has_cview[0] ? &S.rt_cview[0] : NULL;
    return pass_dump_draw_match(g_tex_frame, P_att[0].im->img, S.rt_col[0].serial,
        v && v->set_levels ? v->base_level : 0, v && v->set_layers ? v->min_layer : 0,
        P_att[0].w, P_att[0].h, P_att[0].fmt);
}
static void pass_dump_draw_checkpoint(const char *kind, uint32_t count, uint32_t instances,
                                      uint64_t indirect, uint64_t vs, uint64_t task, uint64_t mesh) {
    if (!pass_dump_draw_ready()) return;
    const int before = !strcmp(kind, "before");
    unsigned ordinal = before ? 0 : ++g_pd_draw_ordinal;
    VKLOG("pd draw: %s ordinal=%u copy=%d present=%llu submission=%llu pass=%d draw=%d count=%u instances=%u indirect=%#llx vs=%016llx fs=%016llx task=%016llx mesh=%016llx image=%p serial=%llu mip=0 layer=0",
        kind, ordinal, g_npd, (unsigned long long)g_tex_frame, (unsigned long long)g_set_epoch,
        g_sub_pass - 1, before ? -1 : g_draw_i - 1, count, instances, (unsigned long long)indirect,
        (unsigned long long)(vs ? vs : S.sh[0] ? S.sh[0]->key : 0),
        (unsigned long long)(S.sh[1] ? S.sh[1]->key : 0), (unsigned long long)task, (unsigned long long)mesh,
        (void *)g_pd_draw_image, (unsigned long long)g_pd_draw_serial);
    g_pd_draw_emit = 1; vk_end_pass(); g_pd_draw_emit = 0;
}

static void pass_dump_draw_geometry(const char *kind, int prim, int64_t first, int64_t base_vertex,
                                   uint32_t base_instance, uint64_t ibuf, int itype) {
    if (!pass_dump_draw_ready()) return;
    VKLOG("pd draw geometry: kind=%s ordinal=%u present=%llu submission=%llu prepared-draw=%d prim=%d first=%lld basevertex=%lld baseinstance=%u ibuf=%#llx itype=%d ibytes=%u",
        kind, g_pd_draw_ordinal + 1, (unsigned long long)g_tex_frame, (unsigned long long)g_set_epoch, g_draw_i,
        prim, (long long)first, (long long)base_vertex, base_instance, (unsigned long long)ibuf, itype,
        itype < 0 ? 0u : itype == 0 ? 1u : itype == 1 ? 2u : 4u);
}
static int pass_dump_cube_match(int strict, uint32_t w, uint32_t h, uint32_t native_format, uint32_t view_format, uint32_t layers) {
    return !strict || (w == 128 && h == 128 && native_format == VK_FORMAT_B10G11R11_UFLOAT_PACK32 &&
                       view_format == VK_FORMAT_B10G11R11_UFLOAT_PACK32 && layers >= 6);
}
static int pass_journal_row(void) {
    if (!g_pd_journal || g_pd_state != 1 || g_tex_frame < g_pd_journal_first || g_tex_frame >= g_pd_journal_end) return 0;
    if (g_pd_journal_rows >= PD_JOURNAL_MAX) {
        if (g_pd_journal_rows == PD_JOURNAL_MAX) { g_pd_journal_rows++; VKLOG("pd journal: truncated after %u rows", PD_JOURNAL_MAX); }
        return 0;
    }
    g_pd_journal_rows++; return 1;
}
static void pass_journal_attachment(const char *event, int slot, const VkRenderingAttachmentInfo *a, const view_t *view) {
    vkimg_t *im = P_att[slot].im; if (!im || !a->imageView || !pass_journal_row()) return;
    uint32_t clear[4]; memcpy(clear, &a->clearValue, sizeof clear);
    uint32_t level = view && view->set_levels ? view->base_level : 0, layer = view && view->set_layers ? view->min_layer : 0;
    VKLOG("pd journal: attachment event=%s present=%llu submission=%llu pass=%d slot=%d serial=%llu image=%p view=%p pool=%#llx off=%#llx native=%u format=%u mip=%u layer=%u layers=%u extent=%ux%u load=%u store=%u clear=%08x/%08x/%08x/%08x",
        event, (unsigned long long)g_tex_frame, (unsigned long long)g_set_epoch, g_sub_pass - 1, slot,
        (unsigned long long)(slot == 8 ? S.rt_dep.serial : S.rt_col[slot].serial), (void *)im->img, (void *)a->imageView,
        (unsigned long long)im->pool, (unsigned long long)im->off, (unsigned)im->fmt, P_att[slot].fmt,
        level, layer, P_att[slot].layers, P_att[slot].w, P_att[slot].h, (unsigned)a->loadOp, (unsigned)a->storeOp,
        clear[0], clear[1], clear[2], clear[3]);
}
static void pass_journal_pass(const char *event) {
    if (!pass_journal_row()) return;
    if (!strcmp(event, "begin")) g_pd_journal_last_fs = 0;
    VKLOG("pd journal: pass event=%s present=%llu submission=%llu pass=%d native=%s extent=%ux%u layers=%u colours=%u draws=%llu split=%d",
        event, (unsigned long long)g_tex_frame, (unsigned long long)g_set_epoch, g_sub_pass - 1,
        S.rp_pass ? "renderpass" : "dynamic", S.pass_w, S.pass_h, P_ri.layerCount, P_ri.colorAttachmentCount,
        (unsigned long long)(!strcmp(event, "begin") ? 0 : g_pass_draws), P_split);
    for (int i = 0; i < S.ncol && i < 8; i++)
        pass_journal_attachment(event, i, &P_ca[i], S.rt_has_cview[i] ? &S.rt_cview[i] : NULL);
    if (P_ri.pDepthAttachment) pass_journal_attachment(event, 8, &P_da, S.rt_has_dview ? &S.rt_dview : NULL);
    if (P_ri.pStencilAttachment) pass_journal_attachment("stencil", 8, &P_sa, S.rt_has_dview ? &S.rt_dview : NULL);
}
static void pass_journal_arm(void) {
    const char *e = getenv("DM_VK_PASSDUMP_JOURNAL");
    g_pd_journal = e && !strcmp(e, "1"); g_pd_journal_rows = 0; g_pd_journal_last_fs = 0;
    g_pd_journal_first = g_tex_frame; g_pd_journal_end = g_tex_frame + (uint64_t)g_pd_frames;
    if (g_pd_journal_end < g_tex_frame) g_pd_journal = 0;
    if (g_pd_journal) { VKLOG("pd journal: armed present=%llu end-exclusive=%llu max-rows=%u",
        (unsigned long long)g_pd_journal_first, (unsigned long long)g_pd_journal_end, PD_JOURNAL_MAX);
        if (S.pass_open == 1) pass_journal_pass("active-at-arm"); }
}
static void pass_journal_draw(const char *kind, uint32_t count, uint32_t instances, uint64_t indirect, uint64_t vs, uint64_t task, uint64_t mesh) {
    if (!pass_journal_row()) return;
    g_pd_journal_last_fs = S.sh[1] ? S.sh[1]->key : 0;
    VKLOG("pd journal: prepared kind=%s present=%llu submission=%llu pass=%d draw=%d pipeline=%p vs=%016llx fs=%016llx gs=%016llx task=%016llx mesh=%016llx count=%u instances=%u indirect=%#llx discard=%d extent=%ux%u",
        kind, (unsigned long long)g_tex_frame, (unsigned long long)g_set_epoch, g_sub_pass - 1, g_draw_i, (void *)S.bound_pipe,
        (unsigned long long)vs, S.sh[1] ? (unsigned long long)S.sh[1]->key : 0ull, S.sh[2] ? (unsigned long long)S.sh[2]->key : 0ull,
        (unsigned long long)task, (unsigned long long)mesh, count, instances, (unsigned long long)indirect, S.rast_discard, S.pass_w, S.pass_h);
}
// Read-only identity of the queued colour copy. Last prepared FS is deliberately
// labelled: a later program bind can change S.sh before this native pass ends.
static void pass_journal_copy(int index, int slot, const vkimg_t *im, uint32_t view_format,
                              uint32_t level, uint32_t layer, uint32_t w, uint32_t h) {
    if (!pass_journal_row()) return;
    VKLOG("pd journal: copy index=%d present=%llu submission=%llu pass=%d slot=%d serial=%llu image=%p pool=%#llx off=%#llx native=%u view_format=%u mip=%u layer=%u extent=%ux%u last_prepared_fs=%016llx",
        index, (unsigned long long)g_tex_frame, (unsigned long long)g_set_epoch, g_sub_pass - 1, slot,
        (unsigned long long)S.rt_col[slot].serial, (void *)im->img, (unsigned long long)im->pool,
        (unsigned long long)im->off, (unsigned)im->fmt, view_format, level, layer, w, h,
        (unsigned long long)g_pd_journal_last_fs);
}
static void pass_journal_clear(const char *kind, const VkClearAttachment *a, const VkClearRect *r, uint32_t mask) {
    if (!pass_journal_row()) return;
    uint32_t clear[4]; memcpy(clear, &a->clearValue, sizeof clear);
    VKLOG("pd journal: clear kind=%s present=%llu submission=%llu pass=%d slot=%u aspect=%u mask=%#x rect=%d,%d,%u,%u layer=%u layers=%u value=%08x/%08x/%08x/%08x",
        kind, (unsigned long long)g_tex_frame, (unsigned long long)g_set_epoch, g_sub_pass - 1, a->colorAttachment, a->aspectMask, mask,
        r->rect.offset.x, r->rect.offset.y, r->rect.extent.width, r->rect.extent.height, r->baseArrayLayer, r->layerCount,
        clear[0], clear[1], clear[2], clear[3]);
}
static void pass_dump_depth(void);
static void texdump_flush(void);
static void cs_trace_buffers_poll(uint64_t done_fence);
static void pass_dump(void) {
    if (g_pd_draw_mode && (!g_pd_draw_emit || !pass_dump_draw_ready())) return;
    if (!g_pd_draw_mode) texdump_flush();
    if (!g_pd_draw_mode && g_pd_state == 1 && g_sub_pass - 1 < g_pd_skip) return;   // (a submission with fewer passes dumps nothing, and the next one is tried)
    { const char *pp = getenv("DM_VK_PASSDUMP_PASS"); if (!g_pd_draw_mode && g_pd_state == 1 && pp && *pp) {   // DM_VK_PASSDUMP_PASS=n[,n...] (live): only these pass numbers of each submission,
        int hit = 0; for (const char *q = pp; *q;) { char *e; long v = strtol(q, &e, 10); if (e == q) break; if (v == g_sub_pass - 1) hit = 1; q = *e ? e + 1 : e; }   // to follow a few passes
        if (!hit) return; } }                                                                                                                                       // over many frames
    { const char *pf = getenv("DM_VK_PASSDUMP_FMT"); int a, b;   // DM_VK_PASSDUMP_FMT=a/b (live): only passes whose first two colour formats are VkFormat a and b (pass numbers move)
      if (!g_pd_draw_mode && g_pd_state == 1 && pf && sscanf(pf, "%d/%d", &a, &b) == 2 && ((int)S.cfmt[0] != a || (int)S.cfmt[1] != b)) return; }
    if (!g_pd_draw_mode && g_pd_state == 1 && g_pd_from[0]) { char wh[32]; snprintf(wh, sizeof wh, "%ux%u", S.pass_w, S.pass_h); if (strcmp(wh, g_pd_from)) return; g_pd_from[0] = 0; }
    if (!g_pd_strict && !g_pd_size[0]) pass_dump_depth();
    if (!g_pd_strict && g_pd_state == 1 && g_pd_size[0] && S.rt_has_dep && (!S.ncol || !S.rt_has_col[0])) {   // a depth-only pass of a listed size (the shadow maps): its depth
        vkimg_t *dm = vk_tex(&S.rt_dep); char wh[32]; if (dm) { snprintf(wh, sizeof wh, "%ux%u", dm->w, dm->h); const char *f = strstr(g_pd_size, wh); size_t l = strlen(wh);
            if (f && (f == g_pd_size || f[-1] == ',') && (!f[l] || f[l] == ',')) pass_dump_depth(); } return; }
    int ci = 0; while (ci < S.ncol && ci < 8 && !S.rt_has_col[ci]) ci++;   // the first colour target bound (Pokémon Legends: Arceus's G-buffer leaves colour 0 unbound)
    const char *pt = getenv("DM_VK_PASSDUMP_TARGET"); if (!g_pd_draw_mode && pt && *pt && atoi(pt) >= 0 && atoi(pt) < 8 && S.rt_has_col[atoi(pt)]) ci = atoi(pt);   // =n: colour target n where the pass has one
    if (g_pd_state != 1 || g_npd >= (g_pd_draw_mode ? PD_DRAW_MAX : PD_MAX) || !S.ncol || ci >= S.ncol || ci >= 8) return;
    vkimg_t *im = vk_tex(&S.rt_col[ci]);

    if (!im || (im->fmt != VK_FORMAT_R8G8B8A8_UNORM && im->fmt != VK_FORMAT_R8G8B8A8_SRGB && im->fmt != VK_FORMAT_R16G16B16A16_SFLOAT && im->fmt != VK_FORMAT_B8G8R8A8_UNORM &&
                                                    im->fmt != VK_FORMAT_B10G11R11_UFLOAT_PACK32 && im->fmt != VK_FORMAT_E5B9G9R9_UFLOAT_PACK32 && im->fmt != VK_FORMAT_R16_SFLOAT && im->fmt != VK_FORMAT_R8_UNORM && im->fmt != VK_FORMAT_A2B10G10R10_UNORM_PACK32 && im->fmt != VK_FORMAT_R16G16_SFLOAT)) return;
    const view_t *cv = S.rt_has_cview[ci] ? &S.rt_cview[ci] : NULL;   // the view's level and layer (a cube face of an array, a mip)
    uint32_t lv = cv && cv->set_levels ? cv->base_level : 0, ly = cv && cv->set_layers ? cv->min_layer : 0, w = im->w >> lv ? im->w >> lv : 1, h = im->h >> lv ? im->h >> lv : 1;
    if (lv >= im->levels || ly >= im->layers) lv = ly = 0, w = im->w, h = im->h;
    if (!pass_dump_cube_match(g_pd_strict, w, h, (uint32_t)im->fmt, P_att[ci].fmt, im->layers)) return;
    if (g_pd_size[0]) { char wh[32]; snprintf(wh, sizeof wh, "%ux%u", w, h); const char *f = strstr(g_pd_size, wh); size_t l = strlen(wh);
        if (!f || (f != g_pd_size && f[-1] != ',') || (f[l] && f[l] != ',')) return;
        const char *dd = getenv("DM_VK_PASSDUMP_DEPTH"); if (!g_pd_strict && dd && atoi(dd)) pass_dump_depth(); }   // DM_VK_PASSDUMP_DEPTH=1: the size-filtered passes' depth too
    uint32_t bpp = im->fmt == VK_FORMAT_R16G16B16A16_SFLOAT ? 8 : im->fmt == VK_FORMAT_R16_SFLOAT ? 2 : im->fmt == VK_FORMAT_R8_UNORM ? 1 : 4; VkDeviceSize n = (VkDeviceSize)w * h * bpp;
    pdump_t *d = &g_pd[g_npd++]; d->w = w; d->h = h; d->f = im->fmt; d->pass = g_sub_pass - 1; d->present = g_tex_frame;
    if (g_pd_draw_mode) d->fence = mtl_last_fence() + 1;
    pass_journal_copy(g_npd - 1, ci, im, P_att[ci].fmt, lv, ly, w, h);
    VkBufferCreateInfo bi = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, NULL, 0, n, VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_SHARING_MODE_EXCLUSIVE};
    VKCHK(vkCreateBuffer(V.dev, &bi, NULL, &d->b)); VkMemoryRequirements rq; vkGetBufferMemoryRequirements(V.dev, d->b, &rq);
    VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL, rq.size, vk_mem_type(rq.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT, 0)};
    VKCHK(vkAllocateMemory(V.dev, &ai, NULL, &d->m)); VKCHK(vkBindBufferMemory(V.dev, d->b, d->m, 0)); VKCHK(vkMapMemory(V.dev, d->m, 0, VK_WHOLE_SIZE, 0, &d->p));
    VkCommandBuffer cb = vk_cmd(); vk_img_barrier_all(cb);
    VkBufferImageCopy c = {0, 0, 0, {VK_IMAGE_ASPECT_COLOR_BIT, lv, ly, 1}, {0, 0, 0}, {w, h, 1}};
    vkCmdCopyImageToBuffer(cb, im->img, VK_IMAGE_LAYOUT_GENERAL, d->b, 1, &c);
    if (g_pd_draw_mode) {
        VkBufferMemoryBarrier host = {VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER, NULL, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT,
            VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, d->b, 0, n};
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 0, NULL, 1, &host, 0, NULL);
    }
    vk_img_barrier_all(cb);
}
static void texdump_flush(void) {   // DM_VK_TEXDUMP: outside a render pass (pass_dump runs after its end)
    vkimg_t *im = g_td_im; if (!im || g_pd_state != 1) return; g_td_im = NULL; g_td_done = 1;
    { const char *rw = getenv("DM_VK_TEXDUMP_RAW"); if (rw && atoi(rw) > 0) {   // DM_VK_TEXDUMP_RAW=<bytes per 4x4 block> (8 or 16; compressed formats): level 0 of each layer as the image holds it, not decoded (pass_NN.raw)
        uint32_t bb = (uint32_t)atoi(rw), bw = (im->w + 3) / 4, bh = (im->h + 3) / 4; VkCommandBuffer cb = vk_cmd(); vk_img_barrier_all(cb);
        for (uint32_t ly = 0; ly < im->layers && ly < 6 && g_npd < PD_MAX; ly++) {
            pdump_t *d = &g_pd[g_npd++]; memset(d, 0, sizeof *d); d->w = im->w; d->h = im->h; d->depth = 1; d->f = im->fmt; d->view_format = im->fmt; d->layer = ly; d->pass = -100 - (int)ly; d->present = g_tex_frame; d->raw_bytes = (uint64_t)bw * bh * bb;
            VkBufferCreateInfo bi = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, NULL, 0, d->raw_bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_SHARING_MODE_EXCLUSIVE};
            VKCHK(vkCreateBuffer(V.dev, &bi, NULL, &d->b)); VkMemoryRequirements rq; vkGetBufferMemoryRequirements(V.dev, d->b, &rq);
            VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL, rq.size, vk_mem_type(rq.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT, 0)};
            VKCHK(vkAllocateMemory(V.dev, &ai, NULL, &d->m)); VKCHK(vkBindBufferMemory(V.dev, d->b, d->m, 0)); VKCHK(vkMapMemory(V.dev, d->m, 0, VK_WHOLE_SIZE, 0, &d->p));
            VkBufferImageCopy c = {0, 0, 0, {VK_IMAGE_ASPECT_COLOR_BIT, 0, ly, 1}, {0, 0, 0}, {im->w, im->h, 1}};
            vkCmdCopyImageToBuffer(cb, im->img, VK_IMAGE_LAYOUT_GENERAL, d->b, 1, &c); }
        vk_img_barrier_all(cb); VKLOG("texdump raw: %ux%u fmt %d layers %u at pool %#llx+%#llx", im->w, im->h, im->fmt, im->layers, (unsigned long long)im->pool, (unsigned long long)im->off); return; } }
    VkFormatProperties fp; vkGetPhysicalDeviceFormatProperties(V.pd, im->fmt, &fp);
    if (!(fp.optimalTilingFeatures & VK_FORMAT_FEATURE_BLIT_SRC_BIT)) { VKLOG("texdump: format %d cannot be blitted", im->fmt); return; }
    const char *tl = getenv("DM_VK_TEXDUMP_LEVEL"); uint32_t lv = tl && *tl ? (uint32_t)atoi(tl) : 0; if (lv >= im->levels) lv = im->levels - 1;   // =n: that mip level instead of level 0
    uint32_t w = im->w >> lv ? im->w >> lv : 1, h = im->h >> lv ? im->h >> lv : 1;
    VkCommandBuffer cb = vk_cmd(); vk_img_barrier_all(cb);
    for (uint32_t ly = 0; ly < im->layers && ly < 6 && g_npd < PD_MAX; ly++) {
        VkImageCreateInfo ic = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, NULL, 0, VK_IMAGE_TYPE_2D, VK_FORMAT_R16G16B16A16_SFLOAT, {w, h, 1}, 1, 1, VK_SAMPLE_COUNT_1_BIT,
            VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, VK_SHARING_MODE_EXCLUSIVE, 0, NULL, VK_IMAGE_LAYOUT_UNDEFINED};
        VkImage ti; VKCHK(vkCreateImage(V.dev, &ic, NULL, &ti)); VkMemoryRequirements rq; vkGetImageMemoryRequirements(V.dev, ti, &rq);
        VkMemoryAllocateInfo ma = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL, rq.size, vk_mem_type(rq.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0)};
        VkDeviceMemory tm; VKCHK(vkAllocateMemory(V.dev, &ma, NULL, &tm)); VKCHK(vkBindImageMemory(V.dev, ti, tm, 0));
        VkImageMemoryBarrier b = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, NULL, 0, VK_ACCESS_TRANSFER_WRITE_BIT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL, VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, ti, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, NULL, 0, NULL, 1, &b);
        VkImageBlit r = {{VK_IMAGE_ASPECT_COLOR_BIT, lv, ly, 1}, {{0, 0, 0}, {(int32_t)w, (int32_t)h, 1}}, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, {{0, 0, 0}, {(int32_t)w, (int32_t)h, 1}}};
        vkCmdBlitImage(cb, im->img, VK_IMAGE_LAYOUT_GENERAL, ti, VK_IMAGE_LAYOUT_GENERAL, 1, &r, VK_FILTER_NEAREST); vk_img_barrier_all(cb);
        pdump_t *d = &g_pd[g_npd++]; d->w = w; d->h = h; d->f = VK_FORMAT_R16G16B16A16_SFLOAT; d->pass = -100 - (int)ly; d->present = g_tex_frame;
        VkBufferCreateInfo bi = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, NULL, 0, (VkDeviceSize)w * h * 8, VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_SHARING_MODE_EXCLUSIVE};
        VKCHK(vkCreateBuffer(V.dev, &bi, NULL, &d->b)); vkGetBufferMemoryRequirements(V.dev, d->b, &rq);
        VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL, rq.size, vk_mem_type(rq.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT, 0)};
        VKCHK(vkAllocateMemory(V.dev, &ai, NULL, &d->m)); VKCHK(vkBindBufferMemory(V.dev, d->b, d->m, 0)); VKCHK(vkMapMemory(V.dev, d->m, 0, VK_WHOLE_SIZE, 0, &d->p));
        VkBufferImageCopy c = {0, 0, 0, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, {0, 0, 0}, {w, h, 1}};
        vkCmdCopyImageToBuffer(cb, ti, VK_IMAGE_LAYOUT_GENERAL, d->b, 1, &c); vk_img_barrier_all(cb);
        vk_defer_free_image(ti, tm);
    }
    VKLOG("texdump: %ux%u fmt %d layers %u levels %u (level %u dumped) at pool %#llx+%#llx: dumps with pass -100 - layer", im->w, im->h, im->fmt, im->layers, im->levels, lv, (unsigned long long)im->pool, (unsigned long long)im->off);
}
static void pass_dump_depth(void) {   // the pass's depth (D24S8 / D32 / D16) as a pdump_t with f = the depth format; written as a grayscale stretched between its min and max
    if (g_pd_state != 1 || g_npd >= PD_MAX || !S.rt_has_dep) return;
    vkimg_t *im = vk_tex(&S.rt_dep); if (!im || !(im->aspect & VK_IMAGE_ASPECT_DEPTH_BIT)) return;
    uint32_t bpp = im->fmt == VK_FORMAT_D16_UNORM ? 2 : 4; VkDeviceSize n = (VkDeviceSize)im->w * im->h * bpp;
    pdump_t *d = &g_pd[g_npd++]; d->w = im->w; d->h = im->h; d->f = im->fmt; d->pass = g_sub_pass - 1; d->present = g_tex_frame;
    VkBufferCreateInfo bi = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, NULL, 0, n, VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_SHARING_MODE_EXCLUSIVE};
    VKCHK(vkCreateBuffer(V.dev, &bi, NULL, &d->b)); VkMemoryRequirements rq; vkGetBufferMemoryRequirements(V.dev, d->b, &rq);
    VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL, rq.size, vk_mem_type(rq.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT, 0)};
    VKCHK(vkAllocateMemory(V.dev, &ai, NULL, &d->m)); VKCHK(vkBindBufferMemory(V.dev, d->b, d->m, 0)); VKCHK(vkMapMemory(V.dev, d->m, 0, VK_WHOLE_SIZE, 0, &d->p));
    VkCommandBuffer cb = vk_cmd(); vk_img_barrier_all(cb);
    uint32_t dly = S.rt_has_dview && S.rt_dview.set_layers && S.rt_dview.min_layer < im->layers ? S.rt_dview.min_layer : 0; d->layer = dly;   // the layer the pass draws (a cascade of a shadow array)
    VkBufferImageCopy c = {0, 0, 0, {VK_IMAGE_ASPECT_DEPTH_BIT, 0, dly, 1}, {0, 0, 0}, {im->w, im->h, 1}};
    vkCmdCopyImageToBuffer(cb, im->img, VK_IMAGE_LAYOUT_GENERAL, d->b, 1, &c); vk_img_barrier_all(cb);
}
static inline float half2f(uint16_t h) { uint32_t s = (uint32_t)(h >> 15) << 31, e = (h >> 10) & 31, m = h & 1023, f;
    if (!e) { if (!m) f = s; else { e = 1; while (!(m & 1024)) { m <<= 1; e--; } m &= 1023; f = s | ((e + 112) << 23) | (m << 13); } } else if (e == 31) f = s | 0x7f800000 | (m << 13); else f = s | ((e + 112) << 23) | (m << 13);
    float r; memcpy(&r, &f, 4); return r; }
// RGB9E5 has no sign/NaN/Inf encoding: each 9-bit mantissa shares the five-bit exponent.
static void rgb9e5_floats(uint32_t word, float rgb[3]) {
    float scale = ldexpf(1.0f, (int)(word >> 27) - 24);
    for (int c = 0; c < 3; c++) rgb[c] = (float)((word >> (9 * c)) & 511u) * scale;
}
// Native diagnostic texels, tightly packed x then y then z. No preview flattens a volume
// or discards channels. Called only after the copy fence and mapped-memory invalidation.
static void pass_dump_raw_native_named(const pdump_t *d, int index, const char *root, const char *prefix) {
    char path[1024], meta[1040]; snprintf(path, sizeof path, "%s/%s_%02d.raw", root ? root : ".", prefix, index);
    size_t written = 0; FILE *f = fopen(path, "wb");
    if (f) { written = fwrite(d->p, 1, (size_t)d->raw_bytes, f); fclose(f); }
    snprintf(meta, sizeof meta, "%s.json", path); f = fopen(meta, "wb");
    if (f) {
        fprintf(f, "{\"key\":\"%016llx\",\"dispatch\":%u,\"binding\":%u,\"phase\":\"%s\",\"storage\":%d,"
            "\"width\":%u,\"height\":%u,\"depth\":%u,\"mip\":%u,\"layer\":%u,\"native_format\":%d,\"view_format\":%d,"
            "\"bytes\":%llu,\"written\":%zu,\"present\":%llu,\"frame\":%llu,\"fence\":%llu,\"order\":\"x,y,z\",\"encoding\":\"native Vulkan texels\"}\n",
            (unsigned long long)d->key, d->dispatch, d->binding, d->after ? "after" : "before", d->storage,
            d->w, d->h, d->depth, d->level, d->layer, d->f, d->view_format, (unsigned long long)d->raw_bytes,
            written, (unsigned long long)d->present, (unsigned long long)d->frame, (unsigned long long)d->fence);
        fclose(f);
    }
    VKLOG("pass dump %02d: native %ux%ux%u format %d view %d mip %u layer %u key %016llx binding %u %s bytes %zu/%llu frame %llu fence %llu -> %s",
        index, d->w, d->h, d->depth, d->f, d->view_format, d->level, d->layer, (unsigned long long)d->key,
        d->binding, d->after ? "after" : "before", written, (unsigned long long)d->raw_bytes,
        (unsigned long long)d->frame, (unsigned long long)d->fence, path);
}
static void pass_dump_raw_native(const pdump_t *d, int index, const char *root) {
    pass_dump_raw_native_named(d, index, root, "pass");
}
// Startup-only key gate: a malformed nonempty key stays disabled rather than dumping unrelated menus.
// With this gate PASSDUMP=1 waits indefinitely without allocation, then starts at the first successful
// matching fragment draw into a 128x128 R11 cube face. The existing post-arm dump limits still apply.
static int g_pd_fs_mode = -2; static uint64_t g_pd_fs_key;
static int pass_dump_fs_parse(const char *p, uint64_t *key) {
    *key = 0;
    if (!p || !*p || !strcmp(p, "0")) return 0;
    if (p[0] == '0' && (p[1] == 'x' || p[1] == 'X')) p += 2;
    unsigned digits = 0; uint64_t value = 0;
    for (; *p; p++) {
        unsigned c = (unsigned char)*p, v;
        if (c >= '0' && c <= '9') v = c - '0';
        else if (c >= 'a' && c <= 'f') v = c - 'a' + 10;
        else if (c >= 'A' && c <= 'F') v = c - 'A' + 10;
        else return -1;
        if (++digits > 16) return -1;
        value = (value << 4) | v;
    }
    if (!digits || !value) return -1;
    *key = value; return 1;
}
static int pass_dump_fs_mode(void) {
    if (g_pd_fs_mode == -2) {
        g_pd_fs_mode = pass_dump_fs_parse(getenv("DM_VK_PASSDUMP_FS"), &g_pd_fs_key);
        if (g_pd_fs_mode) VKLOG("pass dump fragment gate: %016llx (%s; 128x128 R11 cube faces)",
            (unsigned long long)g_pd_fs_key, g_pd_fs_mode == 1 ? "waiting for matching draw" : "invalid key, disabled");
    }
    return g_pd_fs_mode;
}
static void pass_dump_arm(void) {
    const char *sk = getenv("DM_VK_PASSDUMP_SKIP"); g_pd_skip = sk && !strchr(sk, 'x') ? atoi(sk) : 0;
    snprintf(g_pd_from, sizeof g_pd_from, "%s", sk && strchr(sk, 'x') ? sk : "");
    const char *sz = getenv("DM_VK_PASSDUMP_SIZE"); snprintf(g_pd_size, sizeof g_pd_size, "%s", sz && strcmp(sz, "0") ? sz : "");
    { const char *td = getenv("DM_VK_TEXDUMP"); const char *pl = td ? strchr(td, '+') : NULL; g_td_pool = pl ? strtoull(td, NULL, 16) : 0; g_td_off = pl ? strtoll(pl + 1, NULL, 16) : -1; g_td_im = NULL; g_td_done = 0; }
    const char *fr = getenv("DM_VK_PASSDUMP_FRAMES"); g_pd_frames = fr && atoi(fr) > 1 ? atoi(fr) : 1;
    if (g_pd_fs_mode == 1) { if (g_pd_frames > 3) g_pd_frames = 3; if (!g_pd_size[0]) snprintf(g_pd_size, sizeof g_pd_size, "128x128"); }
    g_pd_state = 1; g_npd = 0; memset(g_pd, 0, sizeof g_pd); g_pd_strict = g_pd_fs_mode == 1; setenv("DM_VK_PASSDUMP", "0", 1); VKLOG("pass dump: the next submission's passes from pass %d", g_pd_skip);
    const char *draws = getenv("DM_VK_PASSDUMP_DRAWS");
    g_pd_draw_mode = g_pd_strict && draws && !strcmp(draws, "1") && P_att[0].im && S.rt_has_col[0];
    g_pd_draw_emit = g_pd_draw_before = 0; g_pd_draw_ordinal = 0;
    if (g_pd_draw_mode) {
        g_pd_draw_present = g_tex_frame + 1; g_pd_draw_image = P_att[0].im->img; g_pd_draw_serial = S.rt_col[0].serial;
        VKLOG("pd draw: armed first=%llu target=%llu image=%p serial=%llu max=%u (STORE/LOAD boundaries added)",
            (unsigned long long)g_tex_frame, (unsigned long long)g_pd_draw_present, (void *)g_pd_draw_image,
            (unsigned long long)g_pd_draw_serial, PD_DRAW_MAX);
    }
    pass_journal_arm();
}
static int pass_dump_start_requested(void) {
    if (g_pd_state) return 0;
    const char *e = getenv("DM_VK_PASSDUMP");
    if (!e || !atoi(e)) return 0;
    if (pass_dump_fs_mode()) return 1; // exact-key draw hook owns arming; menus never consume the request
    if (atoi(e) > 1) { char n[16]; snprintf(n, sizeof n, "%d", atoi(e) - 1); setenv("DM_VK_PASSDUMP", n, 1); return 1; }
    pass_dump_arm(); return 1;
}
static void pass_dump_arm_for_draw(uint64_t key, uint32_t w, uint32_t h, VkFormat format, uint32_t layers, int discard) {
    if (g_pd_state || pass_dump_fs_mode() != 1 || key != g_pd_fs_key || discard ||
        w != 128 || h != 128 || format != VK_FORMAT_B10G11R11_UFLOAT_PACK32 || layers < 6) return;
    const char *e = getenv("DM_VK_PASSDUMP"); if (!e || strcmp(e, "1")) return;
    const char *draws = getenv("DM_VK_PASSDUMP_DRAWS");
    if (draws && !strcmp(draws, "1")) {
        const view_t *v = S.rt_has_cview[0] ? &S.rt_cview[0] : NULL;
        if ((v && v->set_levels && v->base_level) || (v && v->set_layers && v->min_layer)) return;
    }
    pass_dump_arm();
}
void vk_pass_dump_poll(uint64_t done_fence) {   // gfx_vk_present.c, every present: start on the switch, write once the GPU finished
    mrt_poll(done_fence);
    cs_trace_buffers_poll(done_fence);
    if (pass_dump_start_requested()) return;
    if (g_pd_draw_mode && g_pd_state == 1) {
        if (g_tex_frame < g_pd_draw_present) return;
        VKLOG("pd draw: finished present=%llu issued=%u copies=%d%s", (unsigned long long)g_tex_frame,
            g_pd_draw_ordinal, g_npd, g_npd == PD_DRAW_MAX ? " (capacity reached)" : "");
        if (!g_npd) { g_pd_state = 0; g_pd_draw_mode = 0; return; }
        g_pd_state = 2; // every draw checkpoint already owns its exact upcoming submission fence
    }
    if (g_pd_state == 1 && g_npd && --g_pd_frames > 0 && g_npd < PD_MAX - 8) return;   // DM_VK_PASSDUMP_FRAMES=N: N presents in a row (with DM_VK_PASSDUMP_SIZE)
    if (g_pd_state == 1 && g_npd) { g_pd_state = 2; for (int i = 0; i < g_npd; i++) if (!g_pd[i].fence) g_pd[i].fence = mtl_last_fence() + 1; return; }
    if (g_pd_state != 2 || !g_npd) return;
    for (int i = 0; i < g_npd; i++) if (done_fence < g_pd[i].fence) return;
    const char *root = getenv("DM_ROOT");
    for (int i = 0; i < g_npd; i++) {
        pdump_t *d = &g_pd[i];
        VkMappedMemoryRange mr = {VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE, NULL, d->m, 0, VK_WHOLE_SIZE};
        VKCHK(vkInvalidateMappedMemoryRanges(V.dev, 1, &mr));   // the readback allocation may be HOST_CACHED without HOST_COHERENT
        if (d->raw_bytes) {
            pass_dump_raw_native(d, i, root);
            vkDestroyBuffer(V.dev, d->b, NULL); vkFreeMemory(V.dev, d->m, NULL); continue;
        }
        char path[1024]; snprintf(path, sizeof path, "%s/pass_%02d.ppm", root ? root : ".", i);
        VKLOG("pass dump %02d is the submission's pass %d (present %llu)", i, d->pass, (unsigned long long)d->present);
        FILE *f = fopen(path, "wb"); if (!f) continue; fprintf(f, "P6\n%u %u\n255\n", d->w, d->h);
        uint8_t *row = malloc((size_t)d->w * 3); uint64_t bad = 0; float mx = 0, mn = 0; double sum = 0;
        if (d->f == VK_FORMAT_D24_UNORM_S8_UINT || d->f == VK_FORMAT_D32_SFLOAT || d->f == VK_FORMAT_D32_SFLOAT_S8_UINT || d->f == VK_FORMAT_D16_UNORM || d->f == VK_FORMAT_X8_D24_UNORM_PACK32) {
            size_t np = (size_t)d->w * d->h; float lo = 1e30f, hi = -1e30f; float *v = malloc(np * sizeof *v);
            for (size_t k = 0; k < np; k++) { float x;
                if (d->f == VK_FORMAT_D16_UNORM) x = ((const uint16_t *)d->p)[k] / 65535.f;
                else if (d->f == VK_FORMAT_D32_SFLOAT || d->f == VK_FORMAT_D32_SFLOAT_S8_UINT) memcpy(&x, (const uint8_t *)d->p + 4 * k, 4);
                else x = (((const uint32_t *)d->p)[k] & 0xffffff) / 16777215.f;
                v[k] = x; if (x < lo) lo = x; if (x > hi) hi = x; }
            for (uint32_t y = 0; y < d->h; y++) { for (uint32_t x = 0; x < d->w; x++) { float t = hi > lo ? (v[(size_t)y * d->w + x] - lo) / (hi - lo) : 0; row[3 * x] = row[3 * x + 1] = row[3 * x + 2] = (uint8_t)(t * 255); } fwrite(row, 1, (size_t)d->w * 3, f); }
            { char rp[1100]; snprintf(rp, sizeof rp, "%s.f32", path); FILE *rf = fopen(rp, "wb"); if (rf) { fwrite(v, sizeof *v, np, rf); fclose(rf); } }   // the values themselves
            free(v); free(row); fclose(f); VKLOG("pass dump %02d: depth %ux%u format %d layer %u, values %.6f .. %.9g → %s", i, d->w, d->h, d->f, d->layer, lo, hi, path);
            vkDestroyBuffer(V.dev, d->b, NULL); vkFreeMemory(V.dev, d->m, NULL); continue;
        }
        if (d->f == VK_FORMAT_R16G16_SFLOAT) {   // signed two-channel fields (velocities): red and green around grey, 1/16 a step; the values themselves in .f32 (r, g a texel)
            size_t np = (size_t)d->w * d->h; float *v = malloc(np * 2 * sizeof *v), lo = 0, hi = 0; uint64_t nz = 0;
            for (size_t k = 0; k < np * 2; k++) { v[k] = half2f(((const uint16_t *)d->p)[k]); if (v[k] < lo) lo = v[k]; if (v[k] > hi) hi = v[k]; }
            for (uint32_t y = 0; y < d->h; y++) { for (uint32_t x = 0; x < d->w; x++) { const float *t = v + 2 * ((size_t)y * d->w + x); int on = t[0] != 0 || t[1] != 0; nz += on;
                for (int c = 0; c < 2; c++) { float q = 128 + t[c] * 16; row[3 * x + c] = (uint8_t)(q < 0 ? 0 : q > 255 ? 255 : q); } row[3 * x + 2] = on ? 255 : 128; }
                fwrite(row, 1, (size_t)d->w * 3, f); }
            { char rp[1100]; snprintf(rp, sizeof rp, "%s.f32", path); FILE *rf = fopen(rp, "wb"); if (rf) { fwrite(v, sizeof *v, np * 2, rf); fclose(rf); } }
            free(v); free(row); fclose(f); VKLOG("pass dump %02d: %ux%u RG16F, values %.3f .. %.3f, nonzero texels %llu → %s", i, d->w, d->h, lo, hi, (unsigned long long)nz, path);
            vkDestroyBuffer(V.dev, d->b, NULL); vkFreeMemory(V.dev, d->m, NULL); continue;
        }
        // Optional unclamped RGB float sidecar for shader probes and HDR diagnosis; the PPM remains a bounded preview.
        FILE *raw = NULL; float *rawrow = NULL; const char *pr = getenv("DM_VK_PASSDUMP_RAW");
        // RAW=2 also preserves every original RGBA16F word. Alpha participates in
        // lighting reductions, so the RGB preview/float sidecar alone cannot replay them.
        if (pr && atoi(pr) >= 2 && d->f == VK_FORMAT_R16G16B16A16_SFLOAT) {
            char rp[1100]; snprintf(rp, sizeof rp, "%s.rgba16f", path);
            FILE *rf = fopen(rp, "wb"); size_t bytes = (size_t)d->w * d->h * 8, written = 0;
            if (rf) { written = fwrite(d->p, 1, bytes, rf); fclose(rf); }
            VKLOG("pass dump %02d: RGBA16F bytes %zu/%zu -> %s", i, written, bytes, rp);
        }
        // Keep all four native RGBA8/BGRA8 channels too: the screen-space masks'
        // alpha is an input to later shading and the RGB preview cannot preserve it.
        if (pr && atoi(pr) >= 2 && (d->f == VK_FORMAT_R8G8B8A8_UNORM || d->f == VK_FORMAT_R8G8B8A8_SRGB ||
                                   d->f == VK_FORMAT_B8G8R8A8_UNORM || d->f == VK_FORMAT_B8G8R8A8_SRGB)) {
            char rp[1100]; snprintf(rp, sizeof rp, "%s.native", path);
            FILE *rf = fopen(rp, "wb"); size_t bytes = (size_t)d->w * d->h * 4, written = 0;
            if (rf) { written = fwrite(d->p, 1, bytes, rf); fclose(rf); }
            VKLOG("pass dump %02d: native format %u extent %ux%u bytes %zu/%zu -> %s",
                i, d->f, d->w, d->h, written, bytes, rp);
        }
        if (pr && atoi(pr) && (d->f == VK_FORMAT_R16G16B16A16_SFLOAT || d->f == VK_FORMAT_B10G11R11_UFLOAT_PACK32 || d->f == VK_FORMAT_E5B9G9R9_UFLOAT_PACK32 || d->f == VK_FORMAT_R16_SFLOAT)) {
            char rp[1100]; snprintf(rp, sizeof rp, "%s.f32", path); rawrow = malloc((size_t)d->w * 3 * sizeof *rawrow);
            if (rawrow) raw = fopen(rp, "wb");
        }
        const char *pa = getenv("DM_VK_PASSDUMP_ALPHA"); int alpha = pa && atoi(pa);
        for (uint32_t y = 0; y < d->h; y++) { for (uint32_t x = 0; x < d->w; x++) {
            if (d->f == VK_FORMAT_R16G16B16A16_SFLOAT || d->f == VK_FORMAT_B10G11R11_UFLOAT_PACK32 || d->f == VK_FORMAT_E5B9G9R9_UFLOAT_PACK32 || d->f == VK_FORMAT_R16_SFLOAT) { float v3[3];
                if (d->f == VK_FORMAT_R16G16B16A16_SFLOAT) { const uint16_t *p = (const uint16_t *)d->p + ((size_t)y * d->w + x) * 4; for (int c = 0; c < 3; c++) v3[c] = half2f(p[c]); }
                else if (d->f == VK_FORMAT_R16_SFLOAT) v3[0] = v3[1] = v3[2] = half2f(((const uint16_t *)d->p)[(size_t)y * d->w + x]);
                else if (d->f == VK_FORMAT_E5B9G9R9_UFLOAT_PACK32) rgb9e5_floats(((const uint32_t *)d->p)[(size_t)y * d->w + x], v3);
                else { uint32_t w = ((const uint32_t *)d->p)[(size_t)y * d->w + x];   // 11-bit and 10-bit floats: a half's exponent and top mantissa bits
                    v3[0] = half2f((uint16_t)((w & 0x7ff) << 4)); v3[1] = half2f((uint16_t)((w >> 11 & 0x7ff) << 4)); v3[2] = half2f((uint16_t)((w >> 22 & 0x3ff) << 5)); }
                if (raw) memcpy(rawrow + 3 * x, v3, sizeof v3);   // retain NaN/Inf as well as HDR values
                for (int c = 0; c < 3; c++) { float v = v3[c]; if (!isfinite(v)) { bad++; v = 1; } if (v > mx) mx = v; if (v < mn) mn = v; sum += v; row[3 * x + c] = (uint8_t)(v <= 0 ? 0 : v >= 1 ? 255 : v * 255 + .5f); } }
            else if (d->f == VK_FORMAT_R8_UNORM) row[3 * x] = row[3 * x + 1] = row[3 * x + 2] = ((const uint8_t *)d->p)[(size_t)y * d->w + x];
            else if (d->f == VK_FORMAT_A2B10G10R10_UNORM_PACK32) { uint32_t w = ((const uint32_t *)d->p)[(size_t)y * d->w + x]; for (int c = 0; c < 3; c++) row[3 * x + c] = (uint8_t)((w >> (10 * c) & 0x3ff) >> 2); }
            else { const uint8_t *p = (const uint8_t *)d->p + ((size_t)y * d->w + x) * 4; int bgr = d->f == VK_FORMAT_B8G8R8A8_UNORM; row[3 * x] = p[bgr ? 2 : 0]; row[3 * x + 1] = p[1]; row[3 * x + 2] = p[bgr ? 0 : 2];
                if (alpha) row[3 * x] = row[3 * x + 1] = row[3 * x + 2] = p[3]; } }   // DM_VK_PASSDUMP_ALPHA=1: the alpha channel as grey (8-bit RGBA / BGRA targets)
            fwrite(row, 1, (size_t)d->w * 3, f);
            if (raw) fwrite(rawrow, sizeof *rawrow, (size_t)d->w * 3, raw); }
        if (raw) fclose(raw); free(rawrow);
        free(row); fclose(f);
        VKLOG("pass dump %02d: %ux%u format %d → %s%s", i, d->w, d->h, d->f, path, d->f == VK_FORMAT_R16G16B16A16_SFLOAT ? "" : "");
        if (d->f == VK_FORMAT_R16G16B16A16_SFLOAT || d->f == VK_FORMAT_B10G11R11_UFLOAT_PACK32 || d->f == VK_FORMAT_E5B9G9R9_UFLOAT_PACK32 || d->f == VK_FORMAT_R16_SFLOAT) VKLOG("   NaN/inf channels %llu, finite values %.3f .. %.3f, mean %.5f", (unsigned long long)bad, mn, mx, sum / (3.0 * d->w * d->h));
        if (d->f == VK_FORMAT_B10G11R11_UFLOAT_PACK32 && d->w * d->h <= 4) { char t[200]; int k = 0;   // a tiny target (an exposure value): every texel
            for (uint32_t q = 0; q < d->w * d->h; q++) { uint32_t w = ((const uint32_t *)d->p)[q];
                k += snprintf(t + k, sizeof t - k, " (%g %g %g)", half2f((uint16_t)((w & 0x7ff) << 4)), half2f((uint16_t)((w >> 11 & 0x7ff) << 4)), half2f((uint16_t)((w >> 22 & 0x3ff) << 5))); }
            VKLOG("   texels:%s", t); }
        vkDestroyBuffer(V.dev, d->b, NULL); vkFreeMemory(V.dev, d->m, NULL);
    }
    g_npd = 0; g_pd_state = 0;
}
// Crumbs, for a GPU hang or a lost device: what each pass recorded, kept by the host always (a few stores a draw): its submission, its draws and dispatches, and the
// shader keys of the last CRUMB_KEYS distinct pipelines it used (vertex / fragment / geometry keys of a draw, a dispatch's compute key). vk_crumbs_log lists one
// submission's passes: on a lost device the first unfinished one (vk_device_lost, gfx_vk_core.c), and a submission the GPU took over DM_VK_HANG_MS on, or whose end it
// never wrote (the driver's hang recovery cuts a hung submission short and signals its fence; hang_check, gfx_vk_core.c). DM_VK_CRUMBS=1 (start-up) adds GPU marks:
// every pass end and dispatch writes {submission, pass} into host-visible memory after a full barrier, so a lost device also says the last work the GPU finished
// (vk_crumbs_report). The per-submission listing and the larger ring are droidplay-core's idea (f77a684b8, 2b3a134aa), on our crumbs
enum { CRUMB_RING = 2048, CRUMB_KEYS = 8 };
static int g_crumbs = -1; static VkBuffer g_crumb_buf; static volatile uint32_t *g_crumb;
typedef struct { uint32_t sub, draws, dispatches; int pass, n, next; uint64_t k[CRUMB_KEYS][3]; } crumb_t; static crumb_t g_cring[CRUMB_RING]; static unsigned g_cn;
static int crumbs_on(void) {   // the GPU marks (DM_VK_CRUMBS=1)
    if (g_crumbs < 0) { const char *e = getenv("DM_VK_CRUMBS"); g_crumbs = e && atoi(e);
        if (g_crumbs) { VkBufferCreateInfo bi = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, NULL, 0, 256, VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_SHARING_MODE_EXCLUSIVE};
            VKCHK(vkCreateBuffer(V.dev, &bi, NULL, &g_crumb_buf)); VkMemoryRequirements rq; vkGetBufferMemoryRequirements(V.dev, g_crumb_buf, &rq);
            VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL, rq.size, vk_mem_type(rq.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 0)};
            VkDeviceMemory m; VKCHK(vkAllocateMemory(V.dev, &ai, NULL, &m)); VKCHK(vkBindBufferMemory(V.dev, g_crumb_buf, m, 0)); void *p; VKCHK(vkMapMemory(V.dev, m, 0, VK_WHOLE_SIZE, 0, &p)); g_crumb = p; g_crumb[0] = 0; } }
    return g_crumbs;
}
static crumb_t *crumb_cur(void) { crumb_t *c = &g_cring[(g_cn ? g_cn - 1 : 0) % CRUMB_RING]; uint32_t sub = (uint32_t)mtl_last_fence() + 1;
    if (!g_cn || c->sub != sub || c->pass != g_sub_pass - 1) { c = &g_cring[g_cn++ % CRUMB_RING]; memset(c, 0, sizeof *c); c->sub = sub; c->pass = g_sub_pass - 1; } return c; }
static void crumb_keys(crumb_t *c, uint64_t a, uint64_t b, uint64_t c3) {   // the pipeline's keys, unless already among the pass's last CRUMB_KEYS (the oldest goes)
    for (int i = 0; i < c->n; i++) if (c->k[i][0] == a && c->k[i][1] == b && c->k[i][2] == c3) return;
    uint64_t *k = c->k[c->next]; k[0] = a; k[1] = b; k[2] = c3; c->next = (c->next + 1) % CRUMB_KEYS; if (c->n < CRUMB_KEYS) c->n++;
}
static void crumb_draw(uint64_t a, uint64_t b, uint64_t c3) { crumb_t *c = crumb_cur(); c->draws++; crumb_keys(c, a, b, c3); }
static void crumb_mark(int dispatch) { if (!crumbs_on()) return; VkCommandBuffer cb = vk_cmd(); vk_img_barrier_all(cb);
    vkCmdFillBuffer(cb, g_crumb_buf, 0, 4, ((uint32_t)(mtl_last_fence() + 1) & 0xffff) << 16 | (dispatch ? 0x8000u : 0) | ((uint32_t)(g_sub_pass - 1) & 0x7fff)); }
static void crumb_dispatch(uint64_t key) { crumb_t *c = crumb_cur(); c->dispatches++; crumb_keys(c, 0, 0, key); crumb_mark(1); }
// The passes submission fence recorded, in order, at most 24 lines: a run of passes with the same shaders as the one before is one line. Any thread (the completion
// thread's hang check, a lost device's): the replay may be recording a later submission meanwhile, so a pass being written can read torn
void vk_crumbs_log(uint64_t fence, const char *why) {
    uint32_t sub = (uint32_t)fence; unsigned end = g_cn, first = end > CRUMB_RING ? end - CRUMB_RING : 0, found = 0, lines = 0, same = 0, sdraws = 0; const crumb_t *prev = NULL;
    VKLOG("[hang] submission %llu (%s), what it recorded:", (unsigned long long)fence, why);
    for (unsigned i = first; i < end; i++) {
        const crumb_t *c = &g_cring[i % CRUMB_RING]; if (c->sub != sub) continue;
        found++;
        if (prev && c->n == prev->n && !memcmp(c->k, prev->k, sizeof c->k)) { same++; sdraws += c->draws + c->dispatches; prev = c; continue; }
        if (same) { VKLOG("[hang]   and %u more passes with the same shaders (%u draws and dispatches)", same, sdraws); same = sdraws = 0; lines++; }
        if (lines >= 24) continue;
        char ks[CRUMB_KEYS * 56 + 1]; int o = 0; ks[0] = 0;
        for (int k = 0; k < c->n; k++) { const uint64_t *x = c->k[(c->next - c->n + k + CRUMB_KEYS) % CRUMB_KEYS];   // oldest first
            o += x[0] | x[1] ? snprintf(ks + o, sizeof ks - (size_t)o, " %016llx/%016llx/%016llx", (unsigned long long)x[0], (unsigned long long)x[1], (unsigned long long)x[2])
                             : snprintf(ks + o, sizeof ks - (size_t)o, " cs %016llx", (unsigned long long)x[2]); if (o >= (int)sizeof ks) break; }
        VKLOG("[hang]   pass %d: %u draws, %u dispatches; last shaders (vertex/fragment/geometry or cs):%s", c->pass, c->draws, c->dispatches, c->n ? ks : " none");
        lines++; prev = c;
    }
    if (same) VKLOG("[hang]   and %u more passes with the same shaders (%u draws and dispatches)", same, sdraws);
    if (!found) VKLOG("[hang]   nothing kept (older than the last %d passes recorded, or no pass)", CRUMB_RING);
    else if (lines >= 24) VKLOG("[hang]   (%u passes in all; the first 24 lines shown)", found);
}
void vk_crumbs_report(void) {   // DM_VK_CRUMBS=1: where the GPU marks say it got to, and the passes after it
    if (g_crumbs <= 0 || !g_crumb) return;
    uint32_t v = g_crumb[0], sub = v >> 16, pass = v & 0x7fff;
    VKLOG("[crumbs] the GPU last finished submission %u (low 16 bits) %s %u", sub, v & 0x8000 ? "a dispatch after pass" : "pass", pass);
    int shown = 0;
    for (unsigned i = g_cn > CRUMB_RING ? g_cn - CRUMB_RING : 0; i < g_cn && shown < 12; i++) { crumb_t *c = &g_cring[i % CRUMB_RING];
        if ((c->sub & 0xffff) < sub || ((c->sub & 0xffff) == sub && c->pass <= (int)pass)) continue;
        shown++; VKLOG("[crumbs] then submission %u pass %d (%u draws, %u dispatches):", c->sub, c->pass, c->draws, c->dispatches);
        for (int k = 0; k < c->n; k++) VKLOG("[crumbs]   shaders %016llx %016llx %016llx", (unsigned long long)c->k[k][0], (unsigned long long)c->k[k][1], (unsigned long long)c->k[k][2]); }
}
void vk_prof_end(void) {
    if (!g_prof_on || !g_prof_t0) return;
    extern uint64_t g_pkg_ns, g_pkg_n; uint64_t tot = tick() - g_prof_t0; g_prof_t0 = 0;
    if (tot < 40000000ull) return;
    double ms = 1e-6; uint64_t known = g_prof_ns[0] + g_prof_ns[1] + g_prof_ns[2] + g_prof_ns[3] + g_prof_ns[4] + g_prof_ns[5] + g_prof_ns[6] + g_prof_ns[7];
    VKLOG("slow replay (present %llu): %.1f ms; pipelines built here %llu in %.1f ms, textures %.1f ms (%llu packaged uploads, %.1f ms of it), pools sync %.1f ms, shader modules %.1f ms, pipeline binds %.1f ms, descriptor sets %.1f ms, cuts %.1f ms, compute pipelines %.1f ms, other %.1f ms",
          (unsigned long long)g_tex_frame, tot * ms, (unsigned long long)(g_vkst.pipes_built - g_prof_pipes0), g_prof_ns[0] * ms, g_prof_ns[1] * ms, (unsigned long long)(g_pkg_n - g_prof_npkg0),
          (g_pkg_ns - g_prof_pkg0) * ms, g_prof_ns[2] * ms, g_prof_ns[3] * ms, g_prof_ns[4] * ms, g_prof_ns[5] * ms, g_prof_ns[6] * ms, g_prof_ns[7] * ms, (tot > known ? tot - known : 0) * ms);
}
// Linear render targets (NVN LINEAR_RENDER_TARGET) are their storage on the Switch: the CPU can read what the GPU drew straight from the pool. Here a Vulkan image holds
// the drawing and the pool memory kept what it held before. Super Mario Odyssey's fluid surface (al::FluidSimulateWave: the Cap Kingdom ground fog's wave solver ends in a
// 128x128 RGBA16F linear target that the fog's vertex shader samples, and a 256x128 R32F one) is such a target. So a pass that drew into one copies each such colour
// target back to its storage, at its stride, marked as written by the GPU (the pools' copy-back carries it to the CPU). DM_VK_LIN_BACK=0 (live, A/B): not
static void lin_back(void) {
    for (int i = 0; i < S.ncol && i < 8; i++) {
        vkimg_t *im = P_att[i].im; const tex_t *t = S.rt_has_col[i] ? &S.rt_col[i] : NULL; const vkfmt_t *f = t ? vk_fmt(t->d.format) : NULL;
        if (!im || im->promoted || !t || !t->mem || !(t->d.flags & NVN_TEXTURE_FLAGS_LINEAR_RENDER_TARGET) || !f || !f->bpb || f->bw > 1 || im->levels != 1 || im->layers != 1 || im->d != 1) continue;
        size_t stride = t->d.stride ? t->d.stride : (size_t)im->w * f->bpb, len = stride * (im->h - 1) + (size_t)im->w * f->bpb; VkBuffer b; VkDeviceSize o;
        if (stride % f->bpb || stride < (size_t)im->w * f->bpb || !vk_buf_at((uint64_t)(uintptr_t)t->mem, len, &b, &o) || o % f->bpb) continue;
        VkCommandBuffer cb = vk_cmd(); vk_img_barrier_all(cb);
        VkBufferImageCopy r = {o, (uint32_t)(stride / f->bpb), 0, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, {0, 0, 0}, {im->w, im->h, 1}};
        vkCmdCopyImageToBuffer(cb, im->img, VK_IMAGE_LAYOUT_GENERAL, b, 1, &r); vk_img_barrier_all(cb);
        void vk_gpu_wrote(uint64_t a, uint64_t n); vk_gpu_wrote((uint64_t)(uintptr_t)t->mem, len); vk_readback_unknown();
        { static int n; if (n++ < 4) VKLOG("linear render target %ux%u (nvn format %#x, stride %zu) copied back to its storage %p", im->w, im->h, t->d.format, stride, (void *)t->mem); }
    }
}
void vk_end_pass(void) {
    if (g_x_listpasses && S.pass_open == 1) { Dl_info di; void *ra = __builtin_return_address(0); uintptr_t base = dladdr((void *)vk_end_pass, &di) ? (uintptr_t)di.dli_fbase : 0; VKLOG("    ended from %#llx (%llu draws)", (unsigned long long)((uintptr_t)ra - base), (unsigned long long)g_pass_draws); list_reads("it"); }   // (DM_VK_TRACE_PASS=-2: who ends each pass)
    if (S.pass_open == 2 && P_cleared) (void)pass_commit();   // only clears: the pass still runs, for its load ops
    if (S.pass_open == 1) { extern uint32_t g_vk_pass_flags; g_vk_pass_flags = (uint32_t)(S.ncol & 15) | (S.daspect ? 16u : 0) | (S.daspect && P_da.loadOp == VK_ATTACHMENT_LOAD_OP_LOAD ? 32u : 0) | (S.ncol && P_ca[0].loadOp == VK_ATTACHMENT_LOAD_OP_LOAD ? 64u : 0) | (P_split ? 128u : 0); }
    if (S.pass_open == 1) { pass_journal_pass("end"); vk_occ_end(); if (S.rp_pass) vkCmdEndRenderPass(vk_cmd()); else vkCmdEndRendering(vk_cmd()); void vk_pass_ts(int end, uint32_t w, uint32_t h, uint32_t fmt, uint32_t draws); vk_pass_ts(1, S.pass_w, S.pass_h, (uint32_t)S.cfmt[0], (uint32_t)g_pass_draws | (g_pass_draws ? (uint32_t)((g_pass_bb[2] - g_pass_bb[0]) & 0xffff) << 16 : 0));  pass_dump(); if (g_x_tracepass >= 0 && g_sub_pass - 1 == g_x_tracepass) { setenv("DM_VK_TRACE_PASS", "-1", 1); g_x_tracepass = -1; } }
    if (S.pass_open == 1 && g_lin_back && g_pass_draws) lin_back();
    if (S.pass_open == 1) crumb_mark(0);
    S.pass_open = 0; S.bound_pipe = VK_NULL_HANDLE; g_tf_in_pass = 0;
}
// a pass end before something that reads or writes pool memory other than through a uniform binding (copies, fills, dispatches, buffer ↔ texture
// copies, the submission's end): the deferred uniform updates land first (draws read them through their shadows, so ordinary pass ends keep them)
void vk_end_pass_land(void) { vk_end_pass(); if (vk_sub_open()) vk_ubd_land(vk_cmd()); }
static int pass_whole(VkRect2D r) { return !r.offset.x && !r.offset.y && r.extent.width >= S.pass_w && r.extent.height >= S.pass_h; }
// Only a native command-buffer boundary advances the depth/LRZ cut generation.
// A refused ordinary budget or exhausted slot must retain the current draw count.
// DM_VK_PART_PACE (live, default 1; 0 = the fixed thresholds alone): the parts of a long submission are spread over its draws. A submission has MAIN_CBS parts; cut
// at every DM_VK_CB_DRAWS / DM_VK_PASS_SPLIT draws, Donkey Kong Bananza's outdoor frames (about 12,000 draws) spent all of them in the first 3.6 ms of the replay on
// 128-draw slices of one cheap pass, and the other ~80 passes (about 14 ms of GPU work) went to the GPU together at the commit: the GPU idled 11 ms of the
// replay and then ran 15 ms after it (the DM_VK_PASS_TIMES "chain" line), which the game's frame-rate controller (its frame start to the GPU's end) read as a 30 ms frame
// and dropped to 30 / 20 fps. With pacing a cut also waits until the submission's remaining draws (estimated by the largest of the last eight submissions) shared
// over the parts left reach the threshold, so the parts last to the end of the frame and the GPU works while the replay records
static uint64_t part_pace(uint64_t floor) {
    if (!g_pace) return floor;
    uint64_t est = 0; for (int i = 0; i < 8; i++) if (g_sub_hist[i] > est) est = g_sub_hist[i];
    uint64_t done = g_vkst.draws - g_sub_d0; int left = vk_cmd_parts_left();
    if (left < 1 || est <= done) return floor;
    uint64_t per = (est - done) / (uint64_t)(left + 1);   // (the commit sends the last part)
    return per > floor ? per : floor;
}
static int pass_cut(int interior, uint64_t *cuts) {
    if (!interior && g_ord_parts >= 15) { vk_cmd_split_refused(0); return 0; }
    if (vk_cmd_parts_left() <= 0) { vk_cmd_split_refused(1); return 0; }
    if (g_x_landcut) vk_ubd_land(vk_cmd());
    if (!vk_cmd_split()) return 0;
    if (!interior) g_ord_parts++;
    g_cb_draws = 0; (*cuts)++; return 1;
}
// Adreno 8xx and an empty colour slot 0 (DM_VK_SLOT0_FILL). A pass whose colour attachments start past slot 0 (slot 0 a null view, VK_ATTACHMENT_UNUSED in a render
// pass) hung the GPU of an Adreno 830 and lost the device every frame: a OnePlus Pad 3 (driver 0x8032004a) whose pass bound a target in slot 5 with slots 0-4 empty, which a
// later pass drew into as slot 0; a scratch target in slot 0 alone fixed it (droidplay-core 87bd41d62, the idea). Preventive here: no such hang is known on the Odin 3's
// driver (512.800). Slot 0 of such a pass gets a scratch RGBA8 image at least the pass's size (one, grown when a pass is larger; the old ones stay: submissions in flight
// name their views), loaded and stored DONT_CARE (on a tiler it never leaves tile memory), and the pipelines write nothing to it (prepare_draw: write mask 0 where
// P_att[0] is the scratch). Single-sample, single-layer passes only. DM_VK_SLOT0_FILL=1 / 0 (start-up): on / off whatever the GPU; by default on for the Adreno 8xx
// (named "Adreno (TM) 8nn" by Qualcomm's driver and by Turnip), off elsewhere
static vkimg_t g_s0_image; static VkImageView g_s0_view; static uint64_t g_s0_passes; static int g_s0_failed;
static int adreno_model(const VkPhysicalDeviceProperties *p) {   // 830 for "Adreno (TM) 830" (Qualcomm's vendor ID), 0 for any other GPU
    const char *n = p->vendorID == 0x5143 ? strstr(p->deviceName, "Adreno (TM) ") : NULL; return n ? atoi(n + 12) : 0;
}
static int slot0_fill_on(void) {
    static int on = -1;
    if (on < 0) { const char *e = getenv("DM_VK_SLOT0_FILL"); int m = adreno_model(&V.props);
        on = e && *e ? atoi(e) != 0 : m >= 800 && m < 900;
        VKLOG("empty colour slot 0: %s (DM_VK_SLOT0_FILL%s; %s)", on ? "a scratch target there in multi-target passes" : "left empty", e && *e ? "" : " default", V.props.deviceName); }
    return on;
}
// whether a pass of ncol colour slots, slot 0 without a view, gets the scratch there
static int slot0_fill_wanted(int ncol, int slot0_view, uint32_t samples, uint32_t layers) { return ncol > 1 && !slot0_view && samples == 1 && layers == 1 && slot0_fill_on(); }
static int slot0_scratch(uint32_t w, uint32_t h) {   // the scratch image, at least w x h; 0: none could be made (the slot stays empty)
    if (g_s0_view && g_s0_image.w >= w && g_s0_image.h >= h) return 1;
    if (g_s0_failed) return 0;
    vkimg_t im = {0}; VkImageView v = VK_NULL_HANDLE;
    im.w = w > g_s0_image.w ? w : g_s0_image.w; im.h = h > g_s0_image.h ? h : g_s0_image.h;
    VkImageCreateInfo ci = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, NULL, 0, VK_IMAGE_TYPE_2D, VK_FORMAT_R8G8B8A8_UNORM, {im.w, im.h, 1}, 1, 1, VK_SAMPLE_COUNT_1_BIT,
                            VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, VK_SHARING_MODE_EXCLUSIVE, 0, NULL, VK_IMAGE_LAYOUT_UNDEFINED};
    if (vkCreateImage(V.dev, &ci, NULL, &im.img)) goto fail;
    VkMemoryRequirements mr; vkGetImageMemoryRequirements(V.dev, im.img, &mr);
    VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL, mr.size, vk_mem_type(mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0)};
    vk_mem_tag = "slot0-scratch"; VkResult r = vkAllocateMemory(V.dev, &ai, NULL, &im.mem); vk_mem_tag = NULL;
    if (r || vkBindImageMemory(V.dev, im.img, im.mem, 0)) goto fail;
    VkImageViewCreateInfo vi = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, NULL, 0, im.img, VK_IMAGE_VIEW_TYPE_2D, ci.format, {0, 0, 0, 0}, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
    if (vkCreateImageView(V.dev, &vi, NULL, &v)) goto fail;
    VkImageMemoryBarrier b = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, NULL, 0, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                              VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, im.img, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
    vkCmdPipelineBarrier(vk_cmd(), VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 0, 0, NULL, 0, NULL, 1, &b);
    im.fmt = ci.format; im.usage = ci.usage; im.aspect = VK_IMAGE_ASPECT_COLOR_BIT; im.d = im.layers = im.levels = im.samples = 1; im.vtype = VK_IMAGE_VIEW_TYPE_2D; im.layout = VK_IMAGE_LAYOUT_GENERAL;
    VKLOG("empty colour slot 0: scratch target %ux%u (DM_VK_SLOT0_FILL)", im.w, im.h);
    g_s0_image = im; g_s0_view = v; return 1;   // (the one before stays: a submission in flight may still name its view)
fail:
    VKLOG("[RED] empty colour slot 0: no scratch target %ux%u could be made: such passes keep it empty from here", im.w, im.h); g_s0_failed = 1;
    if (v) vkDestroyImageView(V.dev, v, NULL);
    if (im.img) vkDestroyImage(V.dev, im.img, NULL);
    if (im.mem) vkFreeMemory(V.dev, im.mem, NULL);
    return 0;
}
static int begin_pass(void) {
    if (S.pass_open) return 1;
    int psplit = g_psplit_now; g_psplit_now = 0;   // a pass split (prepare_draw): taken before any early return, so the flag never forces a later pass's cut
    VkCommandBuffer cb = vk_cmd();
    VkRenderingAttachmentInfo *ca = P_ca; memset(ca, 0, sizeof P_ca);
    for (int i = 0; i < 9; i++) P_att[i].im = NULL;
    VkRenderingAttachmentInfo da = {0}, sa = {0};
    uint32_t w = UINT32_MAX, h = UINT32_MAX, layers = UINT32_MAX, samples = 1; int ncol = 0, have = 0;
    for (int i = 0; i < S.rt_n; i++) {
        S.cfmt[i] = VK_FORMAT_UNDEFINED;
        ca[i].sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO; ca[i].imageLayout = VK_IMAGE_LAYOUT_GENERAL; ca[i].loadOp = VK_ATTACHMENT_LOAD_OP_LOAD; ca[i].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        if (!S.rt_has_col[i]) continue;
        vkimg_t *im; VkImageView v = vk_tex_view(&S.rt_col[i], S.rt_has_cview[i] ? &S.rt_cview[i] : NULL, 0xff, 0, 1, &im);
        if (!v) continue;
        vk_tex_prepare(cb, im);
        uint32_t lv = S.rt_has_cview[i] && S.rt_cview[i].set_levels ? S.rt_cview[i].base_level : 0;
        uint32_t iw = im->w >> lv ? im->w >> lv : 1, ih = im->h >> lv ? im->h >> lv : 1;
        // a block-compressed texture drawn into through an uncompressed view (RGBA32UI over BC3: one texel per 4x4 block, Mario Kart 8 Deluxe compressing its
        // reflection cubemaps on the GPU) is that view's size in blocks: with the texel size the render area was 16 times the attachment, which Vulkan does not allow (the render area must lie within every attachment view)
        { const vkfmt_t *tf = vk_fmt(S.rt_col[i].d.format), *vf = S.rt_has_cview[i] && S.rt_cview[i].set_format ? vk_fmt(S.rt_cview[i].format) : tf;
          if (tf && vf && tf->compressed && !tf->dec && !vf->compressed && tf->bw && tf->bh) { iw = (iw + tf->bw - 1) / tf->bw; ih = (ih + tf->bh - 1) / tf->bh;
              static int said; if (!said++) VKLOG("colour target: block-compressed format %#x drawn through view format %#x: pass in blocks, %ux%u", S.rt_col[i].d.format, S.rt_cview[i].format, iw, ih); } }
        if (iw < w) w = iw; if (ih < h) h = ih; samples = im->samples; have = 1;
        ca[i].imageView = v; S.cfmt[i] = vk_view_fmt; ncol = i + 1; P_att[i].im = im;   // (the view's format: a render pass's attachment and its pipelines take it)
        P_att[i].fmt = vk_view_fmt; P_att[i].layers = vk_view_layers ? vk_view_layers : 1; P_att[i].w = iw; P_att[i].h = ih;
        if (P_att[i].layers < layers) layers = P_att[i].layers;
    }
    for (int i=0;i<8;i++) P_alias_views[i]=ca[i].imageView;
    color_alias_map((uint32_t)ncol,P_alias_views,S.chmask.mask,P_alias);
    for (int i=0;i<ncol;i++) if (P_alias[i]!=i) {
        ca[i].imageView=VK_NULL_HANDLE;ca[i].loadOp=VK_ATTACHMENT_LOAD_OP_DONT_CARE;ca[i].storeOp=VK_ATTACHMENT_STORE_OP_DONT_CARE;
        S.cfmt[i]=VK_FORMAT_UNDEFINED;P_att[i].im=NULL;
        static int said;if (!said++) VKLOG("inactive colour attachment alias: slot %d uses slot %u (one image load/store)",i,(unsigned)P_alias[i]);
    }
    S.dfmt = VK_FORMAT_UNDEFINED; S.daspect = 0;
    if (S.rt_has_dep) {
        vkimg_t *im; VkImageView v = vk_tex_view(&S.rt_dep, S.rt_has_dview ? &S.rt_dview : NULL, 0xff, 0, 1, &im);
        if (v) {
            vk_tex_prepare(cb, im);
            uint32_t lv = S.rt_has_dview && S.rt_dview.set_levels ? S.rt_dview.base_level : 0;
            uint32_t iw = im->w >> lv ? im->w >> lv : 1, ih = im->h >> lv ? im->h >> lv : 1;
            if (iw < w) w = iw; if (ih < h) h = ih; samples = im->samples; have = 1;
            da = (VkRenderingAttachmentInfo){VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, NULL, v, VK_IMAGE_LAYOUT_GENERAL, 0, NULL, VK_IMAGE_LAYOUT_UNDEFINED, VK_ATTACHMENT_LOAD_OP_LOAD, VK_ATTACHMENT_STORE_OP_STORE};
            S.dfmt = im->fmt; S.daspect = im->aspect; P_att[8].im = im; P_att[8].fmt = vk_view_fmt; P_att[8].layers = vk_view_layers ? vk_view_layers : 1; P_att[8].w = iw; P_att[8].h = ih;
            if (P_att[8].layers < layers) layers = P_att[8].layers;
            if (im->aspect & VK_IMAGE_ASPECT_STENCIL_BIT) sa = da;
        }
    }
    if (!have) {
        // No target at all was set (NVN allows it: vertex shaders that write memory under rasterizer discard, fragment shaders with image stores). Nintendo Switch 2 Welcome
        // Tour draws about 76 a frame that way in its expo lobby (skipped before, 4 % of its draws); Vulkan has passes with no attachments too, sized here by the viewport. A
        // target that was set but has no usable image (a view that failed) is still skipped: such a draw would put nothing anywhere
        int bound = S.rt_has_dep; for (int i = 0; i < S.rt_n && i < 8; i++) bound |= S.rt_has_col[i] || S.rt_bad[i];   // (a slot set to something not a texture: failed, skipped as upstream nx2android #194 does)
        if (bound) { VKRED_ONCE(30, "draw with no usable render target: skipped"); return 0; }
        if (!noatt_on()) { static int n; if (n++ < 4) VKLOG("[RED] draw with no render target set (rasterizer discard %d, vertex %016llx, fragment %016llx): skipped (DM_VK_NOATT=1 runs such draws; they hung the Adreno 830 in Welcome Tour)",
                                                           S.rast_discard, S.sh[0] ? (unsigned long long)S.sh[0]->key : 0ull, S.sh[1] ? (unsigned long long)S.sh[1]->key : 0ull); return 0; }
        float vx = S.vp[0][0] > 0 ? S.vp[0][0] : 0, vy = S.vp[0][1] > 0 ? S.vp[0][1] : 0, vw = S.vp[0][2], vh = S.vp[0][3];
        w = vw > 0 ? (uint32_t)(vx + vw + 0.5f) : 1; h = vh > 0 ? (uint32_t)(vy + vh + 0.5f) : 1;
        if (noatt_iso()) { vk_end_pass_land(); cb = vk_cmd(); }   // the updates recorded before the draw land before it (no pass is open here; see noatt_on)
        if (w > 16384) w = 16384; if (h > 16384) h = 16384; if (!w) w = 1; if (!h) h = 1; layers = 1; samples = 1; ncol = 0;
        static int n; if (n++ < 6) VKLOG("pass with no attachments (%d target slots set, none a texture): rasterizer discard %d, vertex %016llx, fragment %016llx, %ux%u from the viewport", S.rt_n, S.rast_discard,
                                         S.sh[0] ? (unsigned long long)S.sh[0]->key : 0ull, S.sh[1] ? (unsigned long long)S.sh[1]->key : 0ull, w, h);
    }
    if (have && slot0_fill_wanted(ncol, ca[0].imageView != VK_NULL_HANDLE, samples, layers) && slot0_scratch(w, h)) {   // an Adreno 8xx's empty slot 0 (DM_VK_SLOT0_FILL, above)
        ca[0] = (VkRenderingAttachmentInfo){VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, NULL, g_s0_view, VK_IMAGE_LAYOUT_GENERAL, 0, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED,
                                            VK_ATTACHMENT_LOAD_OP_DONT_CARE, VK_ATTACHMENT_STORE_OP_DONT_CARE};
        S.cfmt[0] = VK_FORMAT_R8G8B8A8_UNORM; P_att[0].im = &g_s0_image; P_att[0].fmt = VK_FORMAT_R8G8B8A8_UNORM; P_att[0].layers = 1; P_att[0].w = g_s0_image.w; P_att[0].h = g_s0_image.h;
        if (g_s0_passes++ < 4) VKLOG("empty colour slot 0: scratch target for a %ux%u pass with %d colour slots (DM_VK_SLOT0_FILL; %llu so far)", w, h, ncol, (unsigned long long)g_s0_passes);
    }
    // The cut (DM_VK_CB_DRAWS): only before a pass whose depth attachment no earlier pass of this submission drew with. Adreno keeps a low-resolution copy of a depth
    // attachment (LRZ) that it trusts within a command buffer; a pass that goes on with a depth buffer begun in an earlier command buffer had whole 8x8 blocks of its
    // nearest objects rejected in some frames (NEO: bushes, trees and the phone booth near the camera flickered light blue). DM_VK_X_CUTANY=1 cuts before any pass.
    // The rule looks only at the pass at the cut: a later pass may still go on with a depth buffer drawn before it. Hades II cuts before its depth-less passes and then
    // goes on with the 1280x720 D24S8 of the passes before the cut: its lights went out for single frames while Melinoe walked near them (one-frame dark spikes in
    // 6-23 of 600 recorded frames; 0 with DM_VK_CB_DRAWS=0, also 2-15 with the parts submitted together, 135 without the LRZ touches). Such a pass first gets a texel of its
    // depth written back (lrz_touch), the transfer write that made Adreno drop a stale LRZ for Xenoblade Chronicles 3 (DM_VK_LRZ_TOUCH=0 leaves it out)
    // The write has its own cost: Animal Crossing: New Horizons' island lost its lit ground in a full-width band over the top third (the first bin row of a later pass)
    // in 6-22 of 60 frames with it, 0 of 60 with only a read at the cut (DM_VK_LRZ_TOUCH bit 16 without bit 8) or with no cuts; the texel at the far corner instead
    // of the first kept the band at the top. That profile runs uncut (DM_VK_CB_DRAWS=0, also 2.4 ms less GPU); the default here stays the write Hades II was checked with
    static uint64_t g_cuts; int cut_touch = 0;
    // The attachment lookups above already resolved and marked these images written for this pass.
    vkimg_t *dim = P_att[8].im;
    if (psplit || (g_x_cbdraws > 0 && g_cb_draws >= part_pace((uint64_t)g_x_cbdraws) && (!dim || dim->depth_sub != g_set_epoch || g_x_cutany))) {   // (a pass split: prepare_draw checked the depth rule and the part budget)
        if (pass_cut(psplit, &g_cuts)) cb = vk_cmd();   // deferred uniform updates stay deferred unless explicitly requested above
    }
    if (dim) { cut_touch = (g_lrz_touch & (LRZ_CUT | LRZ_CUT_RD)) && dim->depth_sub == g_set_epoch && dim->depth_cut != g_cuts; dim->depth_sub = g_set_epoch; dim->depth_cut = g_cuts; }
    { static int n; uint32_t cs[9] = {0}; int any = 0, mis = 0; uint32_t s0 = 0;   // multisampled passes: every attachment's sample count (they must agree)
      for (int i = 0; i < ncol; i++) if (ca[i].imageView) { cs[i] = P_att[i].im->samples; if (cs[i] > 1) any = 1; if (!s0) s0 = cs[i]; else if (cs[i] != s0) mis = 1; }
      if (da.imageView) { cs[8] = dim->samples; if (cs[8] > 1) any = 1; if (s0 && cs[8] != s0) mis = 1; }
      if ((any || mis) && n < 10) { n++; VKLOG("pass %ux%u samples: colour %u %u %u %u, depth %u%s", w, h, cs[0], cs[1], cs[2], cs[3], cs[8], mis ? " MISMATCH" : ""); } }
    if (!g_x_nobar) vk_img_barrier_all(cb);
    if (cut_touch) lrz_touch(dim, (g_lrz_touch & LRZ_CUT) != 0);   // (after the barrier: the earlier command buffer's depth writes come first)
    if (g_x_depth) { da.loadOp = sa.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE; da.storeOp = sa.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE; }
    if (g_x_color) for (int i = 0; i < ncol; i++) ca[i].loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    P_da = da; P_sa = sa; P_cleared = 0; P_split = psplit;
    P_ri = (VkRenderingInfo){VK_STRUCTURE_TYPE_RENDERING_INFO, NULL, 0, {{0, 0}, {w, h}}, layers, 0, (uint32_t)ncol, P_ca,
                             (S.daspect & VK_IMAGE_ASPECT_DEPTH_BIT) ? &P_da : NULL, (S.daspect & VK_IMAGE_ASPECT_STENCIL_BIT) ? &P_sa : NULL};
    S.pass_open = 2; S.pass_w = w; S.pass_h = h; S.samples = samples; S.ncol = ncol; S.bound_pipe = VK_NULL_HANDLE;
    g_noatt_open = !have && noatt_iso();
    if (g_mrt_active) mrt_attach();
    return 1;
}
static void mrt_support(int st, const vkshader_t *s, VkDescriptorBufferInfo *bi) {
    if (!g_mrt_active || g_mrt_observer || st != 1 || s->key != g_mrt_key) return;
    VkBuffer b; VkDeviceSize o;
    uint8_t *p = vk_scratch(VK_SUPPORT_BYTES, V.props.limits.minUniformBufferOffsetAlignment, &b, &o);
    vk_support_values(p); memcpy(p + LIGHT_DIAG_OFFSET, &g_mrt_mode, sizeof g_mrt_mode);
    *bi = (VkDescriptorBufferInfo){b, o, VK_SUPPORT_BYTES}; // immutable until this submission completes
}
// Diagnostic only: one extra native attachment for the audited CCA396 draw.
// Skip the first matching presentation, then modes1..12, once per presentation.
// Readbacks have their own fences/filenames and do not consume the ordinary dumper.
enum { MRT_MODES = 12, MRT_DUMPS = 3 * MRT_MODES };
static vkimg_t g_mrt_image, *g_mrt_color; static VkImageView g_mrt_view;
static pdump_t g_mrt_dumps[MRT_DUMPS]; static unsigned g_mrt_ndump, g_mrt_done, g_mrt_fixed_mode;
static uint64_t g_mrt_first = UINT64_MAX, g_mrt_last = UINT64_MAX;
static int g_mrt_config = -1, g_mrt_clear, g_mrt_failed;
static unsigned g_mrt_skip, g_mrt_qualifying;
static uint32_t g_mrt_width=128,g_mrt_height=128;
static VkFormat g_mrt_format=VK_FORMAT_R32G32B32A32_SFLOAT;
static uint32_t mrt_active_slots(int n, const uint8_t has[8]) {
    uint32_t mask = 0;
    for (int i = 0; i < n && i < 8; i++) if (has[i]) mask |= 1u << i;
    return mask;
}
static int mrt_match(uint64_t key, uint32_t count, uint32_t instances, int prim, int indirect, int indexed,
                     uint32_t ncol, uint32_t w, uint32_t h, uint32_t format, uint32_t samples,
                     uint32_t layers, uint32_t level, uint32_t layer) {
    return g_mrt_key && !g_mrt_failed && g_mrt_done < (g_mrt_fixed_mode ? 1u : MRT_MODES) && key == g_mrt_key &&
        count == 396 && instances <= 1 && prim == 4 && !indirect && indexed && ncol == 1 &&
        w == 128 && h == 128 && format == VK_FORMAT_B10G11R11_UFLOAT_PACK32 &&
        samples == 1 && layers >= 6 && !level && !layer;
}
static int mrt_choose(uint64_t present) {
    if (g_mrt_first == UINT64_MAX) {
        g_mrt_first = g_mrt_last = present;
        VKLOG("MRT diagnostic: skipped cold first matching present %llu", (unsigned long long)present); return 0;
    }
    if (present <= g_mrt_first || present == g_mrt_last || g_mrt_done >= (g_mrt_fixed_mode ? 1u : MRT_MODES)) return 0;
    g_mrt_last = present; g_mrt_mode = g_mrt_fixed_mode ? g_mrt_fixed_mode : g_mrt_done + 1; return 1;
}
static int mrt_image_create(void) {
    if (g_mrt_image.img) return 1;
    VkFormatProperties fp; vkGetPhysicalDeviceFormatProperties(V.pd, g_mrt_format, &fp);
    if (!(fp.optimalTilingFeatures & VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT) ||
        !(fp.optimalTilingFeatures & VK_FORMAT_FEATURE_TRANSFER_SRC_BIT)) return 0;
    VkImageCreateInfo ci = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, NULL, 0, VK_IMAGE_TYPE_2D,
        g_mrt_format, {g_mrt_width,g_mrt_height,1}, 1, 1, VK_SAMPLE_COUNT_1_BIT,
        VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
        VK_SHARING_MODE_EXCLUSIVE, 0, NULL, VK_IMAGE_LAYOUT_UNDEFINED};
    VkResult r = vkCreateImage(V.dev, &ci, NULL, &g_mrt_image.img); if (r) goto fail;
    VkMemoryRequirements mr; vkGetImageMemoryRequirements(V.dev, g_mrt_image.img, &mr);
    uint32_t type = vk_mem_type(mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0);
    if (type == UINT32_MAX) goto fail;
    VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL, mr.size, type};
    if (vkAllocateMemory(V.dev, &ai, NULL, &g_mrt_image.mem) ||
        vkBindImageMemory(V.dev, g_mrt_image.img, g_mrt_image.mem, 0)) goto fail;
    VkImageViewCreateInfo vi = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, NULL, 0, g_mrt_image.img,
        VK_IMAGE_VIEW_TYPE_2D, ci.format, {0,0,0,0}, {VK_IMAGE_ASPECT_COLOR_BIT,0,1,0,1}};
    if (vkCreateImageView(V.dev, &vi, NULL, &g_mrt_view)) goto fail;
    g_mrt_image.fmt=ci.format;g_mrt_image.usage=ci.usage;g_mrt_image.aspect=VK_IMAGE_ASPECT_COLOR_BIT;
    g_mrt_image.w=g_mrt_width;g_mrt_image.h=g_mrt_height;g_mrt_image.d=g_mrt_image.layers=g_mrt_image.levels=g_mrt_image.samples=1;
    g_mrt_image.vtype=VK_IMAGE_VIEW_TYPE_2D;
    VkImageMemoryBarrier barrier={VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,NULL,0,VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
        VK_IMAGE_LAYOUT_UNDEFINED,VK_IMAGE_LAYOUT_GENERAL,VK_QUEUE_FAMILY_IGNORED,VK_QUEUE_FAMILY_IGNORED,
        g_mrt_image.img,{VK_IMAGE_ASPECT_COLOR_BIT,0,1,0,1}};
    vkCmdPipelineBarrier(vk_cmd(),VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        0,0,NULL,0,NULL,1,&barrier);
    g_mrt_image.layout=VK_IMAGE_LAYOUT_GENERAL;return 1;
fail:
    if(g_mrt_view)vkDestroyImageView(V.dev,g_mrt_view,NULL);
    if(g_mrt_image.img)vkDestroyImage(V.dev,g_mrt_image.img,NULL);
    if(g_mrt_image.mem)vkFreeMemory(V.dev,g_mrt_image.mem,NULL);
    memset(&g_mrt_image,0,sizeof g_mrt_image);g_mrt_view=VK_NULL_HANDLE;return 0;
}
static int mrt_copy(vkimg_t *im, unsigned slot, int after) {
    if (!im || g_mrt_ndump >= MRT_DUMPS) return 0;
    pdump_t d={0}; d.w=g_mrt_width;d.h=g_mrt_height;d.depth=1;d.f=d.view_format=im->fmt;d.key=g_mrt_key;
    d.raw_bytes=(uint64_t)g_mrt_width*g_mrt_height*(slot?16u:4u);d.binding=slot;d.dispatch=g_mrt_mode;d.after=after;
    d.present=g_tex_frame;d.frame=vk_frame_current();d.fence=mtl_last_fence()+1;d.pass=g_sub_pass-1;
    VkBufferCreateInfo bi={VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,NULL,0,d.raw_bytes,VK_BUFFER_USAGE_TRANSFER_DST_BIT,VK_SHARING_MODE_EXCLUSIVE};
    if(vkCreateBuffer(V.dev,&bi,NULL,&d.b))goto fail;
    VkMemoryRequirements mr;vkGetBufferMemoryRequirements(V.dev,d.b,&mr);
    uint32_t type=vk_mem_type(mr.memoryTypeBits,VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_CACHED_BIT,0);
    if(type==UINT32_MAX)type=vk_mem_type(mr.memoryTypeBits,VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,0);
    if(type==UINT32_MAX)goto fail;
    VkMemoryAllocateInfo ai={VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,NULL,mr.size,type};
    if(vkAllocateMemory(V.dev,&ai,NULL,&d.m)||vkBindBufferMemory(V.dev,d.b,d.m,0)||vkMapMemory(V.dev,d.m,0,VK_WHOLE_SIZE,0,&d.p))goto fail;
    VkCommandBuffer cb=vk_cmd();vk_img_barrier_all(cb);
    VkBufferImageCopy copy={0,0,0,{VK_IMAGE_ASPECT_COLOR_BIT,0,0,1},{0,0,0},{g_mrt_width,g_mrt_height,1}};
    vkCmdCopyImageToBuffer(cb,im->img,VK_IMAGE_LAYOUT_GENERAL,d.b,1,&copy);
    VkBufferMemoryBarrier host={VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,NULL,VK_ACCESS_TRANSFER_WRITE_BIT,VK_ACCESS_HOST_READ_BIT,
        VK_QUEUE_FAMILY_IGNORED,VK_QUEUE_FAMILY_IGNORED,d.b,0,d.raw_bytes};
    vkCmdPipelineBarrier(cb,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_HOST_BIT,0,0,NULL,1,&host,0,NULL);
    vk_img_barrier_all(cb);g_mrt_dumps[g_mrt_ndump]=d;
    VKLOG("MRT copy: index=%u mode=%u slot=%u after=%d present=%llu frame=%llu draw=%d pass=%d image=%p bytes=%llu fence=%llu",
        g_mrt_ndump,g_mrt_mode,slot,after,(unsigned long long)d.present,(unsigned long long)d.frame,g_draw_i,
        d.pass,(void*)im->img,(unsigned long long)d.raw_bytes,(unsigned long long)d.fence);
    g_mrt_ndump++; return 1;
fail:
    if(d.b)vkDestroyBuffer(V.dev,d.b,NULL);
    if(d.m)vkFreeMemory(V.dev,d.m,NULL);
    VKLOG("MRT diagnostic: readback allocation refused; capture stopped");g_mrt_failed=1;return 0;
}
static void mrt_attach(void) {
    P_att[1].im=&g_mrt_image;P_att[1].fmt=g_mrt_format;
    P_att[1].layers=1;P_att[1].w=g_mrt_width;P_att[1].h=g_mrt_height;
    P_ca[1]=(VkRenderingAttachmentInfo){VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,NULL,g_mrt_view,
        VK_IMAGE_LAYOUT_GENERAL,0,VK_NULL_HANDLE,VK_IMAGE_LAYOUT_UNDEFINED,
        g_mrt_clear?VK_ATTACHMENT_LOAD_OP_CLEAR:VK_ATTACHMENT_LOAD_OP_LOAD,VK_ATTACHMENT_STORE_OP_STORE};
    for(unsigned i=0;i<4;i++)P_ca[1].clearValue.color.uint32[i]=g_mrt_observer?0u:0x7fc00000u;
    g_mrt_clear=0;P_ri.colorAttachmentCount=2;S.ncol=2;S.cfmt[1]=g_mrt_format;
}
static void mrt_detach(void) {
    g_mrt_active=0;S.ncol=1;S.cfmt[1]=VK_FORMAT_UNDEFINED;P_att[1].im=NULL;
    memset(&P_ca[1],0,sizeof P_ca[1]);P_ri.colorAttachmentCount=1;
}
static void mrt_abort(void) {
    if(!g_mrt_active)return;
    vk_end_pass();mrt_detach();g_mrt_failed=1;
    VKLOG("MRT diagnostic: selected draw did not issue; mode %u has no after image",g_mrt_mode);
}
static int mrt_observer_match(uint64_t vs,uint64_t fs,uint64_t mesh,int is_mesh,int prim,int indexed,
        uint32_t ncol,uint32_t w,uint32_t h,uint32_t fmt,uint32_t samples,uint32_t layers,uint32_t level,uint32_t layer) {
    return vs==0xc8ab0e758c17df4aull && fs==0x96dbc0392691b437ull && mesh==0xedeefae4c37b0257ull &&
        is_mesh && prim==4 && indexed && ncol==1 && w==1440 && h==810 &&
        fmt==VK_FORMAT_B10G11R11_UFLOAT_PACK32 && samples==1 && layers==1 && !level && !layer;
}
static int mrt_prepare(uint32_t count,uint32_t instances,int prim,int indirect,int indexed) {
    if(g_mrt_config<0){g_mrt_config=0;uint64_t k=0;
        const char *e=getenv("DM_VK_MRT_FS");
        if(e&&pass_dump_fs_parse(e,&k)==1&&k==0xcca3264814bd7187ull){g_mrt_key=k;g_mrt_config=1;}
        const char *mode=getenv("DM_VK_MRT_MODE");uint64_t m=0;
        if(mode&&*mode){if(!fs_buffer_number(mode,10,&m)||m<1||m>MRT_MODES){g_mrt_config=0;g_mrt_key=0;}else g_mrt_fixed_mode=(unsigned)m;}
        if(e&&*e)VKLOG("MRT diagnostic: %s; selected CCA396, cube face0, skip first update; fixed mode %u (0 cycles12)",g_mrt_config?"armed":"invalid key/off",g_mrt_fixed_mode);
        const char *observer=getenv("DM_VK_FLAT_OBSERVER");
        if(observer&&*observer){
            // Mutually exclusive with generic ALT/CCA diagnostics: pair activation is all-or-none.
            uint64_t skip=0;const char *se=getenv("DM_VK_FLAT_OBSERVER_SKIP");
            int valid=!strcmp(observer,"1")&&(!e||!*e)&&!g_spv_alt&&(!se||!*se||fs_buffer_number(se,10,&skip))&&skip<=32;
            g_mrt_config=valid;g_mrt_key=valid?0x96dbc0392691b437ull:0;
            if(valid){g_mrt_observer=1;g_mrt_skip=(unsigned)skip;g_mrt_fixed_mode=1;
                g_mrt_width=1440;g_mrt_height=810;g_mrt_format=VK_FORMAT_R32G32B32A32_UINT;}
            VKLOG("Flat observer: %s; exact mesh/VS/FS, one 1440x810 draw after %llu qualifying draws",valid?"armed":"invalid/off",(unsigned long long)skip);
        }
    }
    if(!g_mrt_config)return 1;
    // set_rt retains unused array elements; only slots below rt_n are currently bound.
    // A failed attachment inside that active range is still not a free probe slot.
    uint32_t active_slots=mrt_active_slots(S.rt_n,S.rt_has_col);
    const view_t *v=S.rt_has_cview[0]?&S.rt_cview[0]:NULL;
    vkimg_t *im=P_att[0].im;
    uint32_t level=v&&v->set_levels?v->base_level:0,layer=v&&v->set_layers?v->min_layer:0;
    int eligible=!(active_slots&~1u)&&im&&S.sh[1]&&!S.rast_discard&&!S.sh[2]&&!g_tf_on&&!g_fs_buffer_mesh&&
       g_spv_alt&&g_spv_key==g_mrt_key&&!g_light_diag_mask&&
       !(S.color.blend_en&1)&&!((S.color.alpha_func>=1)&&(S.color.alpha_func<=7))&&
       mrt_match(S.sh[1]->key,count,instances,prim,indirect,indexed,(uint32_t)S.ncol,
          S.pass_w,S.pass_h,P_att[0].fmt,S.samples,im->layers,level,layer)&&
       P_ri.layerCount==1&&im->fmt==VK_FORMAT_B10G11R11_UFLOAT_PACK32;
    if(g_mrt_observer){
        eligible=!(active_slots&~1u)&&im&&S.sh[0]&&S.sh[1]&&S.sh[4]&&!S.rast_discard&&!S.sh[2]&&!g_tf_on&&
            !g_spv_alt&&!g_light_diag_mask&&!g_mrt_failed&&!g_mrt_done&&
            !(S.color.blend_en&1)&&(S.color.logic_op==0||S.color.logic_op==3)&&!((S.color.alpha_func>=1)&&(S.color.alpha_func<=7))&&
            mrt_observer_match(S.sh[0]->key,S.sh[1]->key,S.sh[4]->key,g_fs_buffer_mesh,prim,indexed,
                S.ncol,S.pass_w,S.pass_h,P_att[0].fmt,S.samples,im->layers,level,layer)&&
            P_ri.layerCount==1&&im->fmt==VK_FORMAT_B10G11R11_UFLOAT_PACK32&&im->samples==1&&
            im->w==1440&&im->h==810&&(!P_att[8].im||P_att[8].im->samples==1);
        if(eligible&&g_mrt_qualifying++<g_mrt_skip)return 1;
        if(eligible&&!vk_shader_observer_pair(S.sh[0],S.sh[1],NULL)){g_mrt_failed=1;return 1;}
    }
    static unsigned candidate_logs;
    if(candidate_logs<3&&S.sh[1]&&S.sh[1]->key==g_mrt_key&&count==396&&S.pass_w==128&&S.pass_h==128){
        candidate_logs++;
        VKLOG("MRT eligibility: eligible=%d present=%llu submission=%llu pass=%d draw=%d key=%016llx count=%u instances=%u prim=%d indirect=%d indexed=%d rt_n=%d active_slots=%#x stored_slots=%#x ncol=%d pass_open=%d extent=%ux%u view_fmt=%u image=%p native_fmt=%u samples=%u native_samples=%u depth_samples=%u image_layers=%u view_layers=%u level=%u layer=%u discard=%d gs=%016llx xfb=%d mesh=%d alt=%d altkey=%016llx light=%u blend=%#x alpha=%u failed=%d done=%u fixed=%u first=%llu last=%llu",
            eligible,(unsigned long long)g_tex_frame,(unsigned long long)g_set_epoch,g_sub_pass-1,g_draw_i,
            (unsigned long long)S.sh[1]->key,count,instances,prim,indirect,indexed,S.rt_n,active_slots,
            mrt_active_slots(8,S.rt_has_col),S.ncol,S.pass_open,S.pass_w,S.pass_h,P_att[0].fmt,
            im?(void*)im->img:NULL,im?(unsigned)im->fmt:0,S.samples,im?im->samples:0,
            P_att[8].im?P_att[8].im->samples:0,im?im->layers:0,P_ri.layerCount,level,layer,
            S.rast_discard,S.sh[2]?(unsigned long long)S.sh[2]->key:0ull,g_tf_on,g_fs_buffer_mesh,g_spv_alt,
            (unsigned long long)g_spv_key,g_light_diag_mask,S.color.blend_en,S.color.alpha_func,
            g_mrt_failed,g_mrt_done,g_mrt_fixed_mode,(unsigned long long)g_mrt_first,(unsigned long long)g_mrt_last);
    }
    if(!eligible||(!g_mrt_observer&&!mrt_choose(g_tex_frame)))return 1;
    if(g_mrt_observer)g_mrt_mode=1;
    // Finish an earlier pending clear normally, then measure the original attachment.
    vk_end_pass();g_mrt_color=im;
    if(!mrt_image_create()||!mrt_copy(im,0,0)){g_mrt_failed=1;return begin_pass();}
    g_mrt_active=1;g_mrt_clear=1;
    if(!begin_pass()){mrt_detach();g_mrt_failed=1;return 0;}
    VKLOG("MRT selected: mode=%u present=%llu key=%016llx count=%u original=%p serial=%llu diagnostic=%p depth=%p",
        g_mrt_mode,(unsigned long long)g_tex_frame,(unsigned long long)g_mrt_key,count,(void*)im->img,
        (unsigned long long)S.rt_col[0].serial,(void*)g_mrt_image.img,P_att[8].im?(void*)P_att[8].im->img:NULL);
    return 1;
}
static void mrt_after_draw(int64_t base_vertex,uint32_t base_instance,uint64_t ibuf) {
    if(!g_mrt_active)return;
    VKLOG("MRT issued: mode=%u present=%llu frame=%llu submission=%llu pass=%d draw=%d mesh-observer=%d basevertex=%lld baseinstance=%u ibuf=%#llx",
        g_mrt_mode,(unsigned long long)g_tex_frame,(unsigned long long)vk_frame_current(),
        (unsigned long long)g_set_epoch,g_sub_pass-1,g_draw_i-1,g_mrt_observer,(long long)base_vertex,base_instance,(unsigned long long)ibuf);
    vk_end_pass();mrt_copy(g_mrt_color,0,1);mrt_copy(&g_mrt_image,1,1);
    g_mrt_done++;mrt_detach();
}
// Flat observer: an observed draw whose debug output holds no covered fragment (Ocarina of Time: most of the pair's mesh draws are culled to nothing on the GPU,
// the first ones of a frame among them) is dropped unwritten and the observer re-arms for a later present, up to 240 attempts
static unsigned g_mrt_attempts;
static void mrt_poll(uint64_t done_fence) {
    if(!g_mrt_ndump)return;
    unsigned pending=0;
    for(unsigned i=0;i<g_mrt_ndump;i++){pdump_t *d=&g_mrt_dumps[i];if(d->b&&d->fence>done_fence)pending++;}
    if(pending)return;
    int hit=!g_mrt_observer;
    for(unsigned i=0;i<g_mrt_ndump;i++){pdump_t *d=&g_mrt_dumps[i];if(!d->b)continue;
        VkMappedMemoryRange r={VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,NULL,d->m,0,VK_WHOLE_SIZE};vkInvalidateMappedMemoryRanges(V.dev,1,&r);
        if(g_mrt_observer&&d->binding==1){const uint32_t *w=d->p;for(uint64_t k=3;k<d->raw_bytes/4;k+=4)if(w[k]==0x43415252u){hit=1;break;}}}
    int retry=g_mrt_observer&&!hit&&g_mrt_done&&!g_mrt_failed&&g_mrt_attempts<240;
    if(g_mrt_observer)VKLOG("Flat observer: attempt %u at present %llu %s",g_mrt_attempts+1,(unsigned long long)g_mrt_dumps[0].present,hit?"covered fragments: written":retry?"covered nothing: re-armed":"covered nothing: giving up");
    for(unsigned i=0;i<g_mrt_ndump;i++){pdump_t *d=&g_mrt_dumps[i];if(!d->b)continue;
        if(!retry)pass_dump_raw_native_named(d,(int)i,getenv("DM_ROOT"),g_mrt_observer?"flat-observer":"mrt");
        vkDestroyBuffer(V.dev,d->b,NULL);vkFreeMemory(V.dev,d->m,NULL);d->b=VK_NULL_HANDLE;
    }
    if(retry){g_mrt_attempts++;g_mrt_ndump=0;g_mrt_done=0;g_mrt_qualifying=0;g_mrt_skip=0;return;}
    if(!g_mrt_active&&(g_mrt_done==(g_mrt_fixed_mode?1u:MRT_MODES)||g_mrt_failed)&&g_mrt_image.img){
        vkDestroyImageView(V.dev,g_mrt_view,NULL);vkDestroyImage(V.dev,g_mrt_image.img,NULL);vkFreeMemory(V.dev,g_mrt_image.mem,NULL);
        memset(&g_mrt_image,0,sizeof g_mrt_image);g_mrt_view=VK_NULL_HANDLE;
    }
}
static VkRect2D scissor_rect(void) {
    int32_t x = 0, y = 0; int64_t x1 = S.pass_w, y1 = S.pass_h;
    if (S.sc_set) {
        int64_t sx = S.sc.x, sy = S.sc.y, sx1 = (int64_t)S.sc.x + S.sc.w, sy1 = (int64_t)S.sc.y + S.sc.h;
        if (g_lower_left && !g_flipy) { int64_t t = (int64_t)S.pass_h - sy1; sy1 = (int64_t)S.pass_h - sy; sy = t; }
        if (sx > x) x = (int32_t)sx; if (sy > y) y = (int32_t)sy; if (sx1 < x1) x1 = sx1; if (sy1 < y1) y1 = sy1;
    }
    if (x1 < x) x1 = x; if (y1 < y) y1 = y;
    return (VkRect2D){{x, y}, {(uint32_t)(x1 - x), (uint32_t)(y1 - y)}};
}
// A clear of some channels only (NVN's ClearColor mask): vkCmdClearAttachments writes every channel, so it is a draw of dm/android/shaders/maskclear.* with the mask as
// the colour write mask, cut to the clear's rectangle by the scissor. Paper Mario: The Thousand-Year Door clears alpha alone (mask 0x8) over a rectangle of its scene
// target when Mario comes out of the water: cleared in full, the floor and water there went black for a frame. One pipeline per pass layout and mask, for attachment 0
#include "maskclear_vert.spv.h"
#include "maskclear_frag.spv.h"
#include "flatdbg_frag.spv.h"
typedef struct { uint32_t ncol, cfmt[8], dfmt, samples, mask; } mckey;
static struct { VkPipelineLayout pl; VkShaderModule vs, fs; int bad; mckey k[32]; VkPipeline p[32]; int n; } g_mc;
static VkPipeline mc_pipeline(uint32_t mask) {
    mckey k; memset(&k, 0, sizeof k); k.ncol = (uint32_t)S.ncol; for (int i = 0; i < S.ncol && i < 8; i++) k.cfmt[i] = (uint32_t)S.cfmt[i];
    k.dfmt = (uint32_t)S.dfmt; k.samples = S.samples ? S.samples : 1; k.mask = mask;
    for (int i = 0; i < g_mc.n; i++) if (!memcmp(&g_mc.k[i], &k, sizeof k)) return g_mc.p[i];
    if (g_mc.bad || g_mc.n == 32) return VK_NULL_HANDLE;
    if (!g_mc.pl) {
        VkPushConstantRange pr = {VK_SHADER_STAGE_FRAGMENT_BIT, 0, 16};
        VkPipelineLayoutCreateInfo li = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, NULL, 0, 0, NULL, 1, &pr};
        VkShaderModuleCreateInfo vi = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, NULL, 0, sizeof maskclear_vert_spv, maskclear_vert_spv}, fi = vi;
        fi.codeSize = sizeof maskclear_frag_spv; fi.pCode = maskclear_frag_spv;
        if (vkCreatePipelineLayout(V.dev, &li, NULL, &g_mc.pl) || vkCreateShaderModule(V.dev, &vi, NULL, &g_mc.vs) || vkCreateShaderModule(V.dev, &fi, NULL, &g_mc.fs)) { g_mc.bad = 1; return VK_NULL_HANDLE; }
    }
    VkPipelineShaderStageCreateInfo ss[2] = {{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, NULL, 0, VK_SHADER_STAGE_VERTEX_BIT, g_mc.vs, "main", NULL},
                                             {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, NULL, 0, VK_SHADER_STAGE_FRAGMENT_BIT, g_mc.fs, "main", NULL}};
    VkPipelineVertexInputStateCreateInfo vin = {VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    VkPipelineInputAssemblyStateCreateInfo ia = {VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO, NULL, 0, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, 0};
    VkPipelineViewportStateCreateInfo vs = {VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO, NULL, 0, 1, NULL, 1, NULL};
    VkPipelineRasterizationStateCreateInfo rs = {VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO, NULL, 0, 0, 0, VK_POLYGON_MODE_FILL, VK_CULL_MODE_NONE, VK_FRONT_FACE_COUNTER_CLOCKWISE, 0, 0, 0, 0, 1.0f};
    VkPipelineMultisampleStateCreateInfo ms = {VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO, NULL, 0, (VkSampleCountFlagBits)k.samples, 0, 0, NULL, 0, 0};
    VkPipelineDepthStencilStateCreateInfo ds = {VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    VkPipelineColorBlendAttachmentState cba[8]; memset(cba, 0, sizeof cba);
    cba[0].colorWriteMask = mask & 0xf;   // NVN's mask bits R G B A are Vulkan's
    VkPipelineColorBlendStateCreateInfo cb = {VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO, NULL, 0, 0, 0, k.ncol, cba, {0, 0, 0, 0}};
    VkDynamicState dyn[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dy = {VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO, NULL, 0, 2, dyn};
    VkFormat cf[8]; for (uint32_t i = 0; i < k.ncol; i++) cf[i] = (VkFormat)k.cfmt[i];
    VkFormat df = (VkFormat)k.dfmt; int has_stencil = df == VK_FORMAT_D24_UNORM_S8_UINT || df == VK_FORMAT_D32_SFLOAT_S8_UINT || df == VK_FORMAT_S8_UINT;
    VkPipelineRenderingCreateInfo rci = {VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO, NULL, 0, k.ncol, cf, df == VK_FORMAT_S8_UINT ? VK_FORMAT_UNDEFINED : df, has_stencil ? df : VK_FORMAT_UNDEFINED};
    VkRenderPass prp = rp_on() ? rp_compat(k.ncol, k.cfmt, k.dfmt, k.samples) : VK_NULL_HANDLE;
    VkGraphicsPipelineCreateInfo ci = {VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, prp ? NULL : &rci, 0, 2, ss, &vin, &ia, NULL, &vs, &rs, &ms, &ds, &cb, &dy, g_mc.pl, prp, 0, VK_NULL_HANDLE, -1};
    VkPipeline p = VK_NULL_HANDLE;
    if ((!prp && !V.has_dynrender) || vkCreateGraphicsPipelines(V.dev, V.pcache, 1, &ci, NULL, &p)) { VKRED_ONCE(32, "masked clear pipeline failed: cleared in full"); p = VK_NULL_HANDLE; }
    vk_pipeline_cache_changed();
    g_mc.k[g_mc.n] = k; g_mc.p[g_mc.n++] = p; return p;
}
static int g_bc_stale;
static int clear_masked(const p_clear_color *c, VkRect2D r) {   // 1 = done
    const vkfmt_t *f = vk_fmt(S.rt_col[c->index].d.format);
    if (c->index != 0 || (f && f->is_int)) return 0;
    VkPipeline p = mc_pipeline(c->mask); if (!p) return 0;
    if (!pass_commit()) return 1;
    VkCommandBuffer cb = vk_cmd();
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, p); S.bound_pipe = VK_NULL_HANDLE;
    VkViewport vp = {0, 0, (float)S.pass_w, (float)S.pass_h, 0, 1}; vkCmdSetViewport(cb, 0, 1, &vp); vkCmdSetScissor(cb, 0, 1, &r);
    vkCmdPushConstants(cb, g_mc.pl, VK_SHADER_STAGE_FRAGMENT_BIT, 0, 16, c->rgba);
    VkClearAttachment ja = {VK_IMAGE_ASPECT_COLOR_BIT, (uint32_t)c->index, {{{c->rgba[0], c->rgba[1], c->rgba[2], c->rgba[3]}}}};
    VkClearRect jr = {r, 0, P_ri.layerCount ? P_ri.layerCount : 1}; pass_journal_clear("masked-draw", &ja, &jr, c->mask);
    vk_occ_end(); vkCmdDraw(cb, 3, 1, 0, 0);
    g_bc_stale = 1;   // the next draw sets its viewport, scissor and bindings again (bc_fresh)
    return 1;
}
static void clear_color(const p_clear_color *c) {
    if (!begin_pass()) return;
    if (c->index < 0 || c->index >= S.ncol) return;
    p_clear_color mapped=*c;mapped.index=P_alias[c->index];c=&mapped;
    if (S.cfmt[c->index] == VK_FORMAT_UNDEFINED || !c->mask) return;
    if ((c->mask & 0xf) != 0xf) { VkRect2D r = scissor_rect(); if (!r.extent.width || !r.extent.height) return;
        if (clear_masked(c, r)) return;
        VKRED_ONCE(31, "partial channel clear mask %#x cleared in full", c->mask); }
    VkClearAttachment a = {VK_IMAGE_ASPECT_COLOR_BIT, (uint32_t)c->index, {{{c->rgba[0], c->rgba[1], c->rgba[2], c->rgba[3]}}}};
    const vkfmt_t *f = vk_fmt(S.rt_col[c->index].d.format);
    if (f && f->is_int) for (int i = 0; i < 4; i++) { if (f->is_sint) a.clearValue.color.int32[i] = (int32_t)c->rgba[i]; else a.clearValue.color.uint32[i] = c->rgba[i] > 0 ? (uint32_t)c->rgba[i] : 0; }
    // every layer of the pass: NVN clears the whole bound target, and a layered pass (a 3D texture drawn slice by slice, the layer from a geometry shader) cleared
    // only its layer 0 here. FINAL FANTASY RESONANCE adds its lights into 32x32x32 volumes (vertex shader f701ff79cec258e6, 32 instances, additive blend): slices 1-31
    // kept what earlier frames added, and the pass of fog boxes that reads them (vertex shader 1300ca45ddca3566) came out different every frame on its first road
    // (pass dumps over 4 frames, centre of the road: 29.6, 40.7, 31.7, 39.7; with every layer cleared 30.4 in each, the passes before it the same either way)
    VkClearRect r = {scissor_rect(), 0, P_ri.layerCount ? P_ri.layerCount : 1};
    if (S.pass_open == 2 && (c->mask & 0xf) == 0xf && pass_whole(r.rect)) {   // the whole attachment before any draw: its load op
        P_ca[c->index].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR; P_ca[c->index].clearValue.color = a.clearValue.color; P_cleared = 1; g_vkst.folded++; return;
    }
    if (!pass_commit()) return;
    if (r.rect.extent.width && r.rect.extent.height) { pass_journal_clear("colour", &a, &r, c->mask); vkCmdClearAttachments(vk_cmd(), 1, &a, 1, &r); }
}
static void clear_ds(const p_clear_ds *c) {
    if (!begin_pass() || !S.daspect) return;
    VkImageAspectFlags as = (c->depth_mask ? VK_IMAGE_ASPECT_DEPTH_BIT : 0) | (c->stencil_mask && (S.daspect & VK_IMAGE_ASPECT_STENCIL_BIT) ? VK_IMAGE_ASPECT_STENCIL_BIT : 0);
    as &= S.daspect; if (!as) return;
    VkClearAttachment a = {as, 0, {.depthStencil = {c->depth, (uint32_t)c->stencil}}};
    VkClearRect r = {scissor_rect(), 0, P_ri.layerCount ? P_ri.layerCount : 1};   // every layer (clear_color)
    if (g_x_tracepass == -2) VKLOG("--- depth-stencil clear: depth %g (mask %d), stencil %d (mask %#x), %ux%u at %d,%d, in pass %d%s", c->depth, (int)c->depth_mask, (int)c->stencil, (unsigned)c->stencil_mask,
        r.rect.extent.width, r.rect.extent.height, r.rect.offset.x, r.rect.offset.y, g_sub_pass - 1 + (S.pass_open == 2), S.pass_open == 2 ? " (before its first draw)" : "");
    if (S.pass_open == 2 && pass_whole(r.rect) && (!(as & VK_IMAGE_ASPECT_STENCIL_BIT) || (c->stencil_mask & 0xff) == 0xff)) {   // into the load ops
        if (as & VK_IMAGE_ASPECT_DEPTH_BIT) { P_da.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR; P_da.clearValue.depthStencil = a.clearValue.depthStencil; }
        if (as & VK_IMAGE_ASPECT_STENCIL_BIT) { P_sa.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR; P_sa.clearValue.depthStencil = a.clearValue.depthStencil; }
        P_cleared = 1; g_vkst.folded++; return;
    }
    if (!pass_commit()) return;
    if (r.rect.extent.width && r.rect.extent.height) { pass_journal_clear("depth-stencil", &a, &r, (uint32_t)c->stencil_mask); vkCmdClearAttachments(vk_cmd(), 1, &a, 1, &r); }
}

// ---------------- formats and enums ----------------
static VkFormat vertex_format(int f, int want_int, int want_sint) {
    static const uint16_t rgb[] = {VK_FORMAT_R8G8B8_UNORM, VK_FORMAT_R8G8B8_SNORM, VK_FORMAT_R8G8B8_UINT, VK_FORMAT_R8G8B8_SINT, VK_FORMAT_R16G16B16_SFLOAT, VK_FORMAT_R16G16B16_UNORM,
        VK_FORMAT_R16G16B16_SNORM, VK_FORMAT_R16G16B16_UINT, VK_FORMAT_R16G16B16_SINT, VK_FORMAT_R32G32B32_SFLOAT, VK_FORMAT_R32G32B32_UINT, VK_FORMAT_R32G32B32_SINT};
    static const uint16_t rgb_scaled[] = {VK_FORMAT_R8G8B8_USCALED, VK_FORMAT_R8G8B8_SSCALED, VK_FORMAT_R16G16B16_USCALED, VK_FORMAT_R16G16B16_SSCALED, VK_FORMAT_R32G32B32_UINT, VK_FORMAT_R32G32B32_SINT};
    VkFormat v;
    if (f >= 0x19 && f <= 0x24) v = (VkFormat)rgb[f - 0x19];
    else if (f >= 0x5D && f <= 0x62) v = (VkFormat)rgb_scaled[f - 0x5D];
    else { const vkfmt_t *e = vk_fmt(f); v = e ? e->vk : VK_FORMAT_UNDEFINED; }
    static uint8_t checked[0x100];   // a format the device cannot read as a vertex attribute gives undefined values: named once
    if (v != VK_FORMAT_UNDEFINED && f > 0 && f < 0x100 && !checked[f]) { checked[f] = 1; VkFormatProperties fp; vkGetPhysicalDeviceFormatProperties(V.pd, v, &fp);
        if (!(fp.bufferFeatures & VK_FORMAT_FEATURE_VERTEX_BUFFER_BIT)) VKLOG("[RED] vertex attribute format %#x (vk %d): the device cannot read it as a vertex attribute", f, v); }
    if (want_int) {   // the shader declared an integer input: the attribute must be an integer format of the same layout
        switch (v) {
        case VK_FORMAT_R32_SFLOAT: return want_sint ? VK_FORMAT_R32_SINT : VK_FORMAT_R32_UINT;
        case VK_FORMAT_R32G32_SFLOAT: return want_sint ? VK_FORMAT_R32G32_SINT : VK_FORMAT_R32G32_UINT;
        case VK_FORMAT_R32G32B32_SFLOAT: return want_sint ? VK_FORMAT_R32G32B32_SINT : VK_FORMAT_R32G32B32_UINT;
        case VK_FORMAT_R32G32B32A32_SFLOAT: return want_sint ? VK_FORMAT_R32G32B32A32_SINT : VK_FORMAT_R32G32B32A32_UINT;
        case VK_FORMAT_R8G8B8A8_UNORM: case VK_FORMAT_R8G8B8A8_USCALED: return want_sint ? VK_FORMAT_R8G8B8A8_SINT : VK_FORMAT_R8G8B8A8_UINT;
        case VK_FORMAT_R16G16_UNORM: case VK_FORMAT_R16G16_USCALED: return want_sint ? VK_FORMAT_R16G16_SINT : VK_FORMAT_R16G16_UINT;
        case VK_FORMAT_R16G16B16A16_UNORM: case VK_FORMAT_R16G16B16A16_USCALED: return want_sint ? VK_FORMAT_R16G16B16A16_SINT : VK_FORMAT_R16G16B16A16_UINT;
        default: return v;
        }
    }
    return v;
}
static const VkBlendFactor BF[0x65] = {
    [1] = VK_BLEND_FACTOR_ZERO, [2] = VK_BLEND_FACTOR_ONE, [3] = VK_BLEND_FACTOR_SRC_COLOR, [4] = VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR, [5] = VK_BLEND_FACTOR_SRC_ALPHA,
    [6] = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA, [7] = VK_BLEND_FACTOR_DST_ALPHA, [8] = VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA, [9] = VK_BLEND_FACTOR_DST_COLOR,
    [10] = VK_BLEND_FACTOR_ONE_MINUS_DST_COLOR, [11] = VK_BLEND_FACTOR_SRC_ALPHA_SATURATE, [0x10] = VK_BLEND_FACTOR_SRC1_COLOR, [0x11] = VK_BLEND_FACTOR_ONE_MINUS_SRC1_COLOR,
    [0x12] = VK_BLEND_FACTOR_SRC1_ALPHA, [0x13] = VK_BLEND_FACTOR_ONE_MINUS_SRC1_ALPHA, [0x61] = VK_BLEND_FACTOR_CONSTANT_COLOR, [0x62] = VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_COLOR,
    [0x63] = VK_BLEND_FACTOR_CONSTANT_ALPHA, [0x64] = VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_ALPHA};
static VkBlendFactor bfac(unsigned f) { return f < 0x65 && (f == 1 || BF[f]) ? BF[f] : VK_BLEND_FACTOR_ONE; }
static VkBlendOp bop(unsigned e) { return e >= 1 && e <= 5 ? (VkBlendOp)(e - 1) : VK_BLEND_OP_ADD; }
static VkCompareOp cmpop(unsigned f) { return f >= 1 && f <= 8 ? (VkCompareOp)(f - 1) : VK_COMPARE_OP_ALWAYS; }
static VkStencilOp sop(unsigned o) { return o >= 1 && o <= 8 ? (VkStencilOp)(o - 1) : VK_STENCIL_OP_KEEP; }
static VkPrimitiveTopology topo(int p) {
    switch (p) { case 0: return VK_PRIMITIVE_TOPOLOGY_POINT_LIST; case 1: return VK_PRIMITIVE_TOPOLOGY_LINE_LIST; case 2: case 3: return VK_PRIMITIVE_TOPOLOGY_LINE_STRIP;
                 case 5: case 8: return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP; case 6: case 9: return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN;
                 case 0xA: return VK_PRIMITIVE_TOPOLOGY_LINE_LIST_WITH_ADJACENCY; case 0xB: return VK_PRIMITIVE_TOPOLOGY_LINE_STRIP_WITH_ADJACENCY;
                 case 0xC: return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST_WITH_ADJACENCY; case 0xD: return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP_WITH_ADJACENCY; case 0xE: return VK_PRIMITIVE_TOPOLOGY_PATCH_LIST;
                 default: return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST; }
}

// ---------------- programs and pipelines ----------------
static uint64_t key64(const uint8_t k[8]) { uint64_t v = 0; for (int i = 0; i < 8; i++) v = v << 8 | k[i]; return v; }
// DM_NVN_SHDUMP (debugging; hle_nvn.c then keeps each program's control sections): a shader missing from the bundle is written to <files>/shader_<key>.bin as its 0x880-byte
// control section followed by its code area (control +0x704 bytes), so it can be looked for in the game files (Xenoblade Chronicles 3's title had one such program)
static void shader_dump(const prog_t *p, unsigned i, uint64_t kk) {
    static int on = -1; if (on < 0) on = getenv("DM_NVN_SHDUMP") != NULL;
    if (!on || i >= p->nstage || !p->sd[i].control || !p->sd[i].data) return;
    uint32_t ncode; memcpy(&ncode, (const uint8_t *)p->sd[i].control + 0x704, 4);
    const char *dir = getenv("DM_ROOT"); char path[1024]; snprintf(path, sizeof path, "%s/shader_%016llx.bin", dir ? dir : ".", (unsigned long long)kk);
    FILE *f = fopen(path, "wb"); if (!f) return;
    fwrite(p->sd[i].control, 1, 0x880, f); if (ncode < (1u << 24)) fwrite((const void *)(uintptr_t)p->sd[i].data, 1, ncode, f); fclose(f);
    VKLOG("shader %016llx written to %s", (unsigned long long)kk, path);
}
static void bind_program(const p_bind_program *bp) {
    // NVN binds by stage mask (vertex 1, fragment 2, geometry 4, tessellation 8 / 0x10, compute 0x20): only the masked stages change, so a game that keeps
    // its vertex and fragment shaders in programs of their own (NEO, AI: THE SOMNIUM FILES) binds them one after the other and draws with both.
    // prog_key's index is the position in the program's shader array (as the game passed them to SetShaders), not the stage: the stage is the bundle's.
    int vk_shader_stage(uint64_t key);
    // NVN2's task (0x40) and mesh (0x80) stages: the bundle has their kernels under stages 6 / 7, and the replay keeps them at 3 / 4 as the host keeps their bindings (hle_nvn.c st_host; mesh_draw)
    uint32_t mask = bp->stages ? (uint32_t)bp->stages & 0xff : 0xff;
    for (int st = 0; st < 6; st++) if (mask >> st & 1) S.sh[st] = NULL;
    if (mask & 0x40) S.sh[3] = NULL; if (mask & 0x80) S.sh[4] = NULL;
    S.prog = bp->prog;
    if (S.prog) for (unsigned i = 0; i < 6; i++) {
        uint8_t k[8]; if (!prog_key(S.prog, i, k)) break;
        uint64_t kk = key64(k); int st = kk ? vk_shader_stage(kk) : -1;
        if (kk && st < 0) {   // the draw or dispatch is skipped. Logged once per shader (the first 16 binds were 4 shaders of Xenoblade Chronicles 3 over and over, which hid others)
            static uint64_t seen[256]; static int nseen; int known = 0; for (int j = 0; j < nseen; j++) if (seen[j] == kk) { known = 1; break; }
            if (!known && nseen < 256) { seen[nseen++] = kk; void vk_shader_missing(uint64_t key); vk_shader_missing(kk); shader_dump(S.prog, i, kk); } }
        if (st < 0 || st > 7 || !(mask >> st & 1)) continue;
        S.sh[st >= 6 ? st - 3 : st] = vk_shader(kk, st);
    }
    // a fragment shader whose per-vertex inputs a generated geometry shader feeds (shader_vk.py per_vertex gs, shader_pvgs.py): it takes the geometry stage of a program that has none
    if (S.sh[2] && S.sh[2]->pv_gs && (mask & 2)) S.sh[2] = NULL;   // the fragment stage was rebound: the generated one belonged to the old fragment shader
    if (!S.sh[2] && S.sh[1]) S.sh[2] = vk_shader_pv_gs(S.sh[1]);   // NULL when the bundle has none (no per-vertex reads, or translated for barycentrics)
    else if (S.sh[2] && !S.sh[2]->pv_gs && S.sh[1] && vk_shader_pv_gs(S.sh[1])) VKRED_ONCE(115, "program with a geometry shader of its own and a fragment shader that reads inputs per vertex: those inputs go unwritten");
    S.prog_ok = S.sh[0] || S.sh[5] || S.sh[4];   // a vertex-only program draws depth only (prepare_draw); a mesh program draws through mesh_draw
}
typedef struct {
    uint64_t key[3]; uint32_t topo, restart, raster_discard, cull, front, mode, depth_clamp, bias_en;
    uint32_t dt, dw, df, st, sf[2], sfail[2], szfail[2], szpass[2];
    uint32_t nattr; struct { uint32_t loc, binding, fmt, off; } attr[16];
    uint32_t nbind; struct { uint32_t binding, stride, rate, divisor; } bind[17];
    uint32_t ncol, cfmt[8], dfmt, samples, a2c, blend_en, logic_en, logic_op, wmask[8];
    struct { uint32_t src, dst, srca, dsta, eq, eqa; } bl[8];
    uint32_t clip_neg1, alt, fdz, dual;   // fdz: the fragment shader's variant that writes gl_FragDepth = gl_FragCoord.z (fdz below); dual: its dual-source variant (below)
    uint32_t flat;   // live flat-shader diagnostic mode, captured for asynchronous builds and pipeline lookup
    uint64_t um[3], im[3];   // each stage's sampled bindings given integer views (vk_shader_module)
    uint32_t atest, aref;   // the alpha test: NVN_ALPHA_FUNC_* (0 none) and the reference's bits (the fragment shader's alpha test variant, its specialization constants). Last: older records read with zeros
} pkey;
static nvn_hmap g_pipes = {.klen = sizeof(pkey)}; static pthread_mutex_t g_pipe_mu = PTHREAD_MUTEX_INITIALIZER; static pthread_cond_t g_pipe_ready = PTHREAD_COND_INITIALIZER;   // the replay thread and the prewarm threads
static VkPipeline build_pipeline(const pkey *k, VkPipelineLayout layout, vkshader_t *const stg[3]);
static void pipe_record(const pkey *k);
// The last key and its pipeline (the replay thread's only; pipelines are never dropped): runs of draws with one pipeline skip the pipeline-key hash and the lock
static pkey g_last_k; static VkPipeline g_last_p;
// Pipelines built in the background. A pipeline met for the first time costs the driver's compile on the replay thread, 20-35 ms each on Adreno 830 (no pipeline
// libraries there): in Super Mario Odyssey the first cap throw at a breakable object, the first swim, made 2-7 new pipelines a frame (175 ms frames, seen as 40, 30
// fps or lower for a moment), then never again in that run; the prewarm (below) covers what an earlier run met. A draw whose pipeline does not exist yet is queued for
// a background thread and left out until it is built, a few frames, when every target of its pass was drawn in each of the two presents before: a target drawn every
// frame shows it a few frames late. Not two presents at any time: an image kept from the title screen and drawn once at a scene's load lost SMO's fog for the run. A target drawn for the first time (an environment map made once) builds at once, as a draw left out there would stay missing, and so
// do draws with rasterizer discard, transform feedback, or shader memory writes (later draws may read them). Age goes by the image: counted by size and format instead, SMO's first scene
// after Resume lost its coin counter and tutorial movie for good and showed the Cap Kingdom fog unparted (targets drawn once, in classes the title screen had aged).
// So a capture's new full-screen targets still build at once (SMO's first frog capture: 33 pipelines, two 300 ms frames, once). So do passes into a linear render target:
// the CPU reads it (lin_back), and SMO's fog wave solver feeds what it reads into the next step, so one step left out lost the Cap Kingdom fog for the run.
// DM_VK_ASYNC_PIPE=0 (live): every pipeline is built at once
#define PIPE_PENDING ((VkPipeline)(uintptr_t)2)
enum { APQ = 256 };
static pkey g_apq[APQ]; static int g_apq_h, g_apq_t, g_apq_on;   // queued keys (a draw left out counts as skipped for no pipeline)
static pthread_mutex_t g_apq_mu = PTHREAD_MUTEX_INITIALIZER; static pthread_cond_t g_apq_cv = PTHREAD_COND_INITIALIZER;
static void async_pipe_complete(const pkey *k, VkPipeline p) {
    pthread_mutex_lock(&g_pipe_mu);
    VkPipeline had = nvn_hm_get(&g_pipes, k);   // a prewarm thread made it meanwhile (as graphics_pipeline does: the one in the table may be bound already, the new one goes)
    if (had && had != PIPE_PENDING) { if (p != (void *)1) vkDestroyPipeline(V.dev, p, NULL); }
    else { nvn_hm_put(&g_pipes, k, p); if (p != (void *)1) pipe_record(k); }
    pthread_cond_broadcast(&g_pipe_ready); pthread_mutex_unlock(&g_pipe_mu);   // wake counted draws waiting on either success or failure
}
static void *async_pipe_thread(void *arg) {
    (void)arg; pthread_setname_np(pthread_self(), "vk async pipe");
    for (;;) {
        pthread_mutex_lock(&g_apq_mu); while (g_apq_h == g_apq_t) pthread_cond_wait(&g_apq_cv, &g_apq_mu);
        pkey k = g_apq[g_apq_t % APQ]; g_apq_t++; pthread_mutex_unlock(&g_apq_mu);
        vkshader_t *stg[3] = {0}; VkPipeline p = (VkPipeline)1;
        for (int s = 0; s < 3; s++) if (k.key[s] && !(s == 1 && k.raster_discard)) stg[s] = vk_shader(k.key[s], s);
        if (stg[0]) { VkDescriptorSetLayout dsl[VK_NSETS] = {vk_tbl_dsl()}; int nsets = 1;
            for (int s = 0; s < 3; s++) if (stg[s]) { dsl[stg[s]->set] = stg[s]->dsl; if (stg[s]->set + 1 > nsets) nsets = stg[s]->set + 1; }
            uint64_t t0 = tick(); p = build_pipeline(&k, vk_pipeline_layout(dsl, nsets), stg);
            { static int n; if (n++ < 6) VKLOG("pipeline built in the background in %.1f ms (shaders %016llx %016llx)", (tick() - t0) / 1e6, (unsigned long long)k.key[0], (unsigned long long)k.key[1]); } }
        async_pipe_complete(&k, p);
    }
    return NULL;
}
static int shader_wmem(vkshader_t *s);
// The packed reflection does not reliably mark every write (images have no rw bit, and Bananza's shipped culling GS has an unmarked storage buffer), and a storage
// binding says nothing either way: Ocarina of Time's large vertex shaders (27d946c5a63213a1, 321663d0cac5fd32) only read their three storage buffers, and counted as
// writers each of their pipelines was compiled on the replay thread (1.1 s a pipeline, 6 s frozen at the first world entry; Odin 3, 2026-11-10). The SPIR-V itself
// says whether a stage stores to memory (spv_wmem; without its SPIR-V a shader is assumed to write).
static int async_shader_writes(vkshader_t *s) { return s && (s->xfb || shader_wmem(s)); }
// Samples-passed counting: a draw left out changes a count only if the count is reported. The ops that follow the draw being replayed (the rest of its command buffer,
// of the buffers that called it, and of the submission's later ones) are read up to the first samples-passed counter op: 1 when that is a reset (the samples of every
// draw before it are discarded unreported), 0 when it is a report, a conditional-render op, or not found. One walk per stretch between counter ops.
static const uint8_t *g_la_at, *g_la_ret[16]; static int g_la_depth; static uint32_t g_la_epoch;   // replay thread: the op after the current one, the callers' next ops
static int occ_reset_next(void) {
    static uint32_t epoch = ~0u; static int ans; extern const uint64_t *vk_rest_h; extern int vk_rest_n;
    if (epoch == g_la_epoch) return ans;
    epoch = g_la_epoch; ans = 0;
    if (!g_la_at || g_la_depth > 16) return 0;
    const uint8_t *p = g_la_at, *loc[32]; int nl = 0, up = g_la_depth, rest = 0;
    for (int left = 400000; left > 0; left--) {
        op_hdr h; memcpy(&h, p, sizeof h); const uint8_t *a = p + 8;
        switch (h.op) {
        case OP_END: if (nl) { p = loc[--nl]; continue; } if (up) { p = g_la_ret[--up]; continue; }
            if (rest < vk_rest_n && vk_rest_h[rest]) { p = (const uint8_t *)(uintptr_t)vk_rest_h[rest++]; continue; }
            return 0;
        case OP_JUMP: { const uint8_t *t; memcpy(&t, a, 8); p = t; continue; }
        case OP_CALL: { uint64_t t; memcpy(&t, a, 8); if (nl == 32 || !t) return 0; loc[nl++] = p + h.size; p = (const uint8_t *)(uintptr_t)t; continue; }
        case OP_REPORT_COUNTER: { uint64_t t; memcpy(&t, a, 8); if (t == NVN_COUNTER_TYPE_SAMPLES_PASSED) return 0; break; }
        case OP_RESET_COUNTER: { uint64_t t; memcpy(&t, a, 8); if (t == NVN_COUNTER_TYPE_SAMPLES_PASSED) return ans = 1; break; }
        case OP_RENDER_COND: return 0;
        default: break;
        }
        if (h.size < sizeof h) return 0;
        p += h.size;
    }
    return 0;
}
static int pipe_must_draw(const pkey *k) {   // 1: the draw cannot be left out while its pipeline is built
    if (!g_async_pipe || k->alt == 2 || k->raster_discard) return 1;
    // Samples-passed is an observable draw side effect even without shader writes. Counting remains enabled after its first reset;
    // a pipeline miss must preserve the draw unless its samples are proven to be reset before any report (occ_reset_next).
    if (vk_occ_counting() && !occ_reset_next()) return 1;
    // A regular rasterizing draw can still produce another draw's inputs (Bananza's culling GS writes indirect triangle counts).
    // Attachment age says nothing about these side effects: compile at once for writes in any graphics stage, as for XFB.
    for (int st = 0; st < 5; st++) if (async_shader_writes(S.sh[st])) return 1;
    for (int i = 0; i < S.ncol && i < 8; i++) if (S.rt_has_col[i] && (S.rt_col[i].d.flags & NVN_TEXTURE_FLAGS_LINEAR_RENDER_TARGET)) return 1;   // (the CPU reads it: lin_back)
    int natt = 0; for (int i = 0; i < 9; i++) { const vkimg_t *a = P_att[i].im; if (!a) continue; natt++; if (a->rt_frames < 3) {
            if (g_prof_on) { static int n; if (n++ < 300) VKLOG("pipeline built here: attachment %d %ux%u nvn format %#x drawn in %llu presents in a row (shaders %016llx %016llx)", i, a->w, a->h, a->nvn_format, (unsigned long long)a->rt_frames, (unsigned long long)k->key[0], (unsigned long long)k->key[1]); }
            return 1; } }   // (counted in this present too)
    if (!natt) return 1;
    return 0;
}
static int apq_push(const pkey *k) {   // 1: queued for the background threads, pending in the table; under nothing (takes g_pipe_mu)
    pthread_mutex_lock(&g_apq_mu);
    if (!g_apq_on) { for (int i = 0; i < 2; i++) { pthread_t t; pthread_attr_t a; pthread_attr_init(&a); pthread_attr_setdetachstate(&a, PTHREAD_CREATE_DETACHED);
            if (!pthread_create(&t, &a, async_pipe_thread, NULL)) { g_apq_on++; }
            pthread_attr_destroy(&a); } }
    int ok = g_apq_on && g_apq_h - g_apq_t < APQ;   // no worker: build synchronously, never leave an unfulfillable pending entry
    if (ok) { g_apq[g_apq_h % APQ] = *k; g_apq_h++; pthread_cond_signal(&g_apq_cv); }
    pthread_mutex_unlock(&g_apq_mu);
    if (ok) { pthread_mutex_lock(&g_pipe_mu); if (!nvn_hm_get(&g_pipes, k)) nvn_hm_put(&g_pipes, k, PIPE_PENDING); pthread_mutex_unlock(&g_pipe_mu); }
    return ok;
}
static int async_pipe(const pkey *k) { return !pipe_must_draw(k) && apq_push(k); }   // 1: queued (the draw is left out)
// A pipeline built here has a sibling that differs only in the fragment shader's depth-write variant (fdz, prepare_draw): which of the two a draw takes changes with
// its depth buffer's last writers, and the other is asked for within the same present or soon after (Ocarina of Time: 20 of 31 shader sets met in two sessions were
// built in both, one after the other, 0.6 s each for the large vertex shaders). The sibling is built on a background thread at the same time; a draw that needs it
// and cannot be left out waits for that build (graphics_pipeline) instead of starting it.
static void pipe_sibling(const pkey *k) {
    const vkshader_t *fs = S.sh[1];
    if (!g_async_pipe || !g_fdz || k->alt == 2 || k->raster_discard || !fs || fs->depth_repl || !k->dt || k->bias_en || !k->dfmt) return;
    pkey sib = *k; sib.fdz = !k->fdz;
    pthread_mutex_lock(&g_pipe_mu); int have = nvn_hm_get(&g_pipes, &sib) != NULL; pthread_mutex_unlock(&g_pipe_mu);
    if (!have) apq_push(&sib);
}
// The long-frame line's "new pipelines" (hle_nvn.c): the graphics pipelines built on the replay thread and the time they took. It read 0 through frames frozen for seconds
static uint64_t g_sync_pipes, g_sync_pipe_ns;
void mtl_create_stats(uint64_t *lib, uint64_t *pso) { *lib = 0; *pso = __atomic_load_n(&g_sync_pipes, __ATOMIC_RELAXED); }
void mtl_create_ns(uint64_t *lib, uint64_t *pso) { *lib = 0; *pso = __atomic_load_n(&g_sync_pipe_ns, __ATOMIC_RELAXED); }
static VkPipeline graphics_pipeline(const pkey *k, VkPipelineLayout layout) {
    if (g_last_p && !memcmp(k, &g_last_k, sizeof *k)) return g_last_p == (void *)1 ? VK_NULL_HANDLE : g_last_p;
    pthread_mutex_lock(&g_pipe_mu); VkPipeline p = nvn_hm_get(&g_pipes, k);
    // This key may have been queued by a draw that could be left out, or as a sibling (pipe_sibling). Wait for that build rather than dropping a draw that cannot be.
    while (p == PIPE_PENDING && pipe_must_draw(k)) { pthread_cond_wait(&g_pipe_ready, &g_pipe_mu); p = nvn_hm_get(&g_pipes, k); }
    pthread_mutex_unlock(&g_pipe_mu);
    if (p == PIPE_PENDING) return VK_NULL_HANDLE;
    if (p) { g_last_k = *k; g_last_p = p; return p == (void *)1 ? VK_NULL_HANDLE : p; }
    if (async_pipe(k)) return VK_NULL_HANDLE;
    vkshader_t *stg[3] = {S.sh[0], S.sh[1], S.sh[2]};
    pipe_sibling(k);
    uint64_t tb = tick(); p = build_pipeline(k, layout, stg); tb = tick() - tb; if (g_prof_on) g_prof_ns[0] += tb; g_vkst.pipes_built++;
    __atomic_add_fetch(&g_sync_pipes, 1, __ATOMIC_RELAXED); __atomic_add_fetch(&g_sync_pipe_ns, tb, __ATOMIC_RELAXED);
    if (tb > 50000000ull) VKLOG("pipeline built on the replay thread in %.0f ms (shaders %016llx %016llx %016llx; present %llu)", tb / 1e6, (unsigned long long)k->key[0], (unsigned long long)k->key[1], (unsigned long long)k->key[2], (unsigned long long)g_tex_frame);
    pthread_mutex_lock(&g_pipe_mu);
    VkPipeline had = nvn_hm_get(&g_pipes, k);
    if (had) { if (p != (void *)1) vkDestroyPipeline(V.dev, p, NULL); p = had; }   // a prewarm thread made it meanwhile
    else { nvn_hm_put(&g_pipes, k, p); if (p != (void *)1) pipe_record(k); }
    pthread_mutex_unlock(&g_pipe_mu);
    return p == (void *)1 ? VK_NULL_HANDLE : p;
}
// ---- pipeline prewarm: every pipeline the game needed is recorded (DM_ROOT/vk_pipelines.bin) and made again at the next boot on background
// threads, before the game reaches it, so a scene seen once never stalls a frame on a pipeline compile again (the driver's compile, not the
// pipeline cache's lookup, is the cost: tens of milliseconds per pipeline on Adreno). DM_VK_NOPREWARM=1 turns it off.
#define PIPE_MAGIC 0x31455050u   // "PPE1"; the header also holds sizeof(pkey), so a changed key layout starts a new file
static FILE *g_prec;
// The records are written by a thread of their own: a flush to the app's files folder (FUSE) took tens of milliseconds, and made under g_pipe_mu it held up the replay's
// pipeline lookups (a 66 ms frame with no pipeline built in it)
enum { PREC_Q = 64 };
static pkey g_precq[PREC_Q]; static int g_precq_n, g_precw_on; static pthread_mutex_t g_precq_mu = PTHREAD_MUTEX_INITIALIZER; static pthread_cond_t g_precq_cv = PTHREAD_COND_INITIALIZER;
static void *pipe_record_thread(void *arg) {
    (void)arg; pthread_setname_np(pthread_self(), "vk pipe record"); static pkey b[PREC_Q];
    for (;;) {
        pthread_mutex_lock(&g_precq_mu); while (!g_precq_n) pthread_cond_wait(&g_precq_cv, &g_precq_mu);
        int n = g_precq_n; memcpy(b, g_precq, sizeof *b * (size_t)n); g_precq_n = 0; pthread_mutex_unlock(&g_precq_mu);
        fwrite(b, sizeof *b, (size_t)n, g_prec); fflush(g_prec);
    }
    return NULL;
}
static void pipe_record(const pkey *k) {   // (under g_pipe_mu: only queued here)
    if (!g_prec || k->alt || k->flat) return;   // alternate-code and flat-shader pipelines are a debugging run's own
    pthread_mutex_lock(&g_precq_mu);
    if (!g_precw_on) { g_precw_on = 1; pthread_t t; pthread_attr_t a; pthread_attr_init(&a); pthread_attr_setdetachstate(&a, PTHREAD_CREATE_DETACHED); pthread_create(&t, &a, pipe_record_thread, NULL); pthread_attr_destroy(&a); }
    if (g_precq_n < PREC_Q) g_precq[g_precq_n++] = *k;   // (a full queue drops the record: that pipeline is only not prewarmed next time)
    pthread_cond_signal(&g_precq_cv); pthread_mutex_unlock(&g_precq_mu);
}
static pkey *g_pw; static size_t g_pw_n; static volatile size_t g_pw_next, g_pw_done;
static void *prewarm_thread(void *arg) {
    (void)arg; pthread_setname_np(pthread_self(), "vk prewarm");
    setpriority(PRIO_PROCESS, 0, 10);   // below the game's threads: prewarm uses idle cores
    for (;;) {
        size_t i = __atomic_fetch_add(&g_pw_next, 1, __ATOMIC_RELAXED); if (i >= g_pw_n) break;
        const pkey *k = &g_pw[i];
        pthread_mutex_lock(&g_pipe_mu); int have = nvn_hm_get(&g_pipes, k) != NULL; pthread_mutex_unlock(&g_pipe_mu);
        if (have) continue;
        vkshader_t *stg[3] = {0}; int ok = 1;
        for (int s = 0; s < 3; s++) if (k->key[s] && !(s == 1 && k->raster_discard)) { stg[s] = vk_shader(k->key[s], s); if (!stg[s]) ok = 0; }
        if (!ok || !stg[0]) continue;
        VkDescriptorSetLayout dsl[VK_NSETS] = {vk_tbl_dsl()}; int nsets = 1;
        for (int s = 0; s < 3; s++) if (stg[s]) { dsl[stg[s]->set] = stg[s]->dsl; if (stg[s]->set + 1 > nsets) nsets = stg[s]->set + 1; }
        VkPipeline p = build_pipeline(k, vk_pipeline_layout(dsl, nsets), stg);
        pthread_mutex_lock(&g_pipe_mu);
        if (!nvn_hm_get(&g_pipes, k)) nvn_hm_put(&g_pipes, k, p); else if (p != (void *)1) vkDestroyPipeline(V.dev, p, NULL);
        pthread_mutex_unlock(&g_pipe_mu);
        size_t d = __atomic_add_fetch(&g_pw_done, 1, __ATOMIC_RELAXED);
        if (d == g_pw_n || d % 500 == 0) VKLOG("prewarm: %zu of %zu pipelines", d, g_pw_n);
    }
    return NULL;
}
static void prewarm_start(void) {
    const char *root = getenv("DM_ROOT"); if (!root) return;
    char path[1024]; snprintf(path, sizeof path, "%s/vk_pipelines.bin", root);
    uint32_t hdr[2] = {PIPE_MAGIC, (uint32_t)sizeof(pkey)}, fh[2] = {0};
    FILE *f = fopen(path, "rb");
    if (f && fread(fh, sizeof fh, 1, f) == 1 && fh[0] == hdr[0] && fh[1] == hdr[1]) {
        fseek(f, 0, SEEK_END); long n = (ftell(f) - (long)sizeof fh) / (long)sizeof(pkey); fseek(f, sizeof fh, SEEK_SET);
        if (n > 0) { g_pw = malloc((size_t)n * sizeof(pkey)); g_pw_n = fread(g_pw, sizeof(pkey), (size_t)n, f); }
    } else if (f && fh[0] == hdr[0] && fh[1] == (uint32_t)offsetof(pkey, atest)) {   // records from before the alpha test fields (appended last): read with them zero
        size_t os = offsetof(pkey, atest); fseek(f, 0, SEEK_END); long n = (ftell(f) - (long)sizeof fh) / (long)os; fseek(f, sizeof fh, SEEK_SET);
        if (n > 0) { g_pw = calloc((size_t)n, sizeof(pkey)); uint8_t *r = malloc(os); for (long i = 0; i < n && fread(r, os, 1, f) == 1; i++) { memcpy(&g_pw[i], r, os); g_pw_n++; } free(r); }
        VKLOG("prewarm: %zu pipelines recorded before the alpha test key fields, read with the test off", g_pw_n);
    }
    if (f) fclose(f);
    // rewrite the file with the records read (dropping a torn last record), then append to it
    g_prec = fopen(path, "wb"); if (g_prec) { fwrite(hdr, sizeof hdr, 1, g_prec); if (g_pw_n) fwrite(g_pw, sizeof(pkey), g_pw_n, g_prec); fflush(g_prec); }
    if (!g_pw_n || getenv("DM_VK_NOPREWARM")) { VKLOG("prewarm: %zu pipelines recorded%s", g_pw_n, g_pw_n ? ", off (DM_VK_NOPREWARM)" : ""); return; }
    long nc = sysconf(_SC_NPROCESSORS_ONLN); int nt = nc >= 8 ? 3 : nc >= 4 ? 2 : 1;
    VKLOG("prewarm: %zu recorded pipelines on %d threads", g_pw_n, nt);
    for (int i = 0; i < nt; i++) { pthread_t t; pthread_attr_t a; pthread_attr_init(&a); pthread_attr_setdetachstate(&a, PTHREAD_CREATE_DETACHED); pthread_create(&t, &a, prewarm_thread, NULL); pthread_attr_destroy(&a); }
}
// ---- what each thread is building, for the crash dump. Issue #64: Super Mario Bros. Wonder crashed in the Adreno 750's shader compiler (libllvm-qgl.so, driver 512.762)
// at its title scene, with no log line to say which pipeline or which thread; and a pipeline that crashes is never recorded for the prewarm, so every run met it again.
// A slot per thread in a build: the crash dump prints each (the crashing thread's marked), with the shaders and the state that pick the driver's compile
enum { NBUILD = 16 };
static struct { _Atomic int tid; volatile int phase; char what[400]; } g_build[NBUILD];   // phase 0: claimed, 1: making shader modules, 2: in the driver's pipeline compile
static int build_begin(void) {   // the slot, or -1 when all are taken (nothing noted)
    int me = gettid(); for (int i = 0; i < NBUILD; i++) { int z = 0; if (atomic_compare_exchange_strong(&g_build[i].tid, &z, me)) { g_build[i].phase = 0; return i; } }
    return -1;
}
static void build_end(int i) { if (i >= 0) { g_build[i].phase = 0; atomic_store(&g_build[i].tid, 0); } }
static void build_dump(void) {   // (the crash handler: no locks, no allocation)
    int me = gettid(), n = 0;
    for (int i = 0; i < NBUILD; i++) { int t = atomic_load(&g_build[i].tid); if (!t || !g_build[i].phase) continue; n++;
        char nm[64]; crash_thread_name((unsigned)t, nm, sizeof nm);
        fprintf(stderr, "[dm] [vk] %s thread %#x '%s' was %s: %s\n", t == me ? "this (the crashing)" : "another", (unsigned)t, nm,
                g_build[i].phase == 1 ? "making its shader modules" : "in the driver's pipeline compile", g_build[i].what); }
    if (!n) fprintf(stderr, "[dm] [vk] no pipeline was being built\n");
}
static VkPipeline build_pipeline(const pkey *k, VkPipelineLayout layout, vkshader_t *const stg_in[3]) {
    VkPipeline p; vkshader_t *stg[3] = {stg_in[0], stg_in[1], stg_in[2] ? vk_shader_gs_topo(stg_in[2], (VkPrimitiveTopology)k->topo) : NULL};
    int bs = build_begin();
    if (bs >= 0) { snprintf(g_build[bs].what, sizeof g_build[bs].what, "graphics pipeline, shaders %016llx %016llx %016llx; topology %u, polygon mode %u, samples %u, "
            "%u colour targets (formats %u %u %u %u), depth format %u; raster discard %u, fragment depth variant %u, dual source %u, transform feedback %d, logic op %u, "
            "vertex attributes %u in %u bindings, integer-texel views %#llx/%#llx %#llx/%#llx, alternative code %u",
            (unsigned long long)k->key[0], (unsigned long long)k->key[1], (unsigned long long)k->key[2], k->topo, k->mode, k->samples, k->ncol, k->cfmt[0], k->cfmt[1], k->cfmt[2], k->cfmt[3],
            k->dfmt, k->raster_discard, k->fdz, k->dual, stg[0] ? stg[0]->xfb : 0, k->logic_en ? k->logic_op : 0, k->nattr, k->nbind,
            (unsigned long long)k->um[0], (unsigned long long)k->im[0], (unsigned long long)k->um[1], (unsigned long long)k->im[1], k->alt); g_build[bs].phase = 1; }
    VkPipelineShaderStageCreateInfo ss[3]; uint32_t ns = 0;
    VkShaderStageFlagBits fl[3] = {VK_SHADER_STAGE_VERTEX_BIT, VK_SHADER_STAGE_FRAGMENT_BIT, VK_SHADER_STAGE_GEOMETRY_BIT};
    // the outputs each stage must declare for the next (vk_shader_module): vertex → geometry or fragment, geometry → fragment
    const vkshader_t *fs = stg[1] && !k->raster_discard ? stg[1] : NULL; uint32_t xo[3] = {0};
    if (stg[0]) xo[0] = (stg[2] ? stg[2]->in_locs : fs ? fs->in_locs : 0) & ~stg[0]->out_locs;
    if (stg[2]) xo[2] = (fs ? fs->in_locs : 0) & ~stg[2]->out_locs;
    VkShaderModule observer[2]={0};
    if(k->alt==2&&!vk_shader_observer_pair(stg[0],stg[1],observer)){build_end(bs);return VK_NULL_HANDLE;}
    for (int i = 0; i < 3; i++) if (stg[i] && !(i == 1 && k->raster_discard)) ss[ns++] = (VkPipelineShaderStageCreateInfo){VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, vk_sg_pin(fl[i], NULL), 0, fl[i], (k->alt==2&&i<2)?observer[i]:vk_shader_module(stg[i], k->um[i], k->im[i], k->alt==1 && !k->um[i] && !k->im[i] && !xo[i] ? 1u << 31 : xo[i], i == 1 ? (k->fdz ? 1 : 0) | (k->dual ? 2 : 0) | (k->atest ? 4 : 0) : 0), "main", NULL};   // (pNext: DM_VK_SUBGROUP_PIN)
    // the alpha test (pkey atest): the function and reference as the variant's specialization constants
    static const VkSpecializationMapEntry at_me[2] = {{VK_ATEST_SPEC_FUNC, 0, 4}, {VK_ATEST_SPEC_REF, 4, 4}}; uint32_t at_v[2] = {k->atest, k->aref};
    VkSpecializationInfo at_si = {2, at_me, sizeof at_v, at_v};
    if (k->atest) for (uint32_t i = 0; i < ns; i++) if (ss[i].stage == VK_SHADER_STAGE_FRAGMENT_BIT) ss[i].pSpecializationInfo = &at_si;
    int flat = fs ? (int)k->flat : 0;
    if (flat & 1) {   // DM_VK_X_FLAT_VS: the fragment stage becomes flatdbg.frag
        static VkShaderModule fm; static pthread_mutex_t fm_mu = PTHREAD_MUTEX_INITIALIZER; pthread_mutex_lock(&fm_mu); if (!fm) { VkShaderModuleCreateInfo mi = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, NULL, 0, sizeof flatdbg_frag_spv, flatdbg_frag_spv}; VKCHK(vkCreateShaderModule(V.dev, &mi, NULL, &fm)); } pthread_mutex_unlock(&fm_mu);
        for (uint32_t i = 0; i < ns; i++) if (ss[i].stage == VK_SHADER_STAGE_FRAGMENT_BIT) ss[i].module = fm;
    }
    if (flat) { static int n; if (n++ < 12) VKLOG("debugging: vertex shader %016llx with fragment %016llx: mode %d (1 magenta, 2 no depth test, 4 no blending); %u colour targets, blending %#x (target 0: src %u dst %u op %u, alpha src %u dst %u op %u, write mask %#x), depth test %u write %u func %u",
        (unsigned long long)stg[0]->key, (unsigned long long)fs->key, flat, k->ncol, k->blend_en, k->bl[0].src, k->bl[0].dst, k->bl[0].eq, k->bl[0].srca, k->bl[0].dsta, k->bl[0].eqa, k->wmask[0], k->dt, k->dw, k->df); }
    VkVertexInputAttributeDescription ad[16]; VkVertexInputBindingDescription bd[17]; VkVertexInputBindingDivisorDescriptionEXT dd[17]; uint32_t ndd = 0;
    for (uint32_t i = 0; i < k->nattr; i++) ad[i] = (VkVertexInputAttributeDescription){k->attr[i].loc, k->attr[i].binding, (VkFormat)k->attr[i].fmt, k->attr[i].off};
    for (uint32_t i = 0; i < k->nbind; i++) { bd[i] = (VkVertexInputBindingDescription){k->bind[i].binding, k->bind[i].stride, (VkVertexInputRate)k->bind[i].rate};
        if (k->bind[i].rate && k->bind[i].divisor > 1) dd[ndd++] = (VkVertexInputBindingDivisorDescriptionEXT){k->bind[i].binding, k->bind[i].divisor}; }
    VkPipelineVertexInputDivisorStateCreateInfoEXT dvs = {VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_DIVISOR_STATE_CREATE_INFO_EXT, NULL, ndd, dd};
    VkPipelineVertexInputStateCreateInfo vi = {VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO, ndd ? &dvs : NULL, 0, k->nbind, bd, k->nattr, ad};
    VkPipelineInputAssemblyStateCreateInfo ia = {VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO, NULL, 0, (VkPrimitiveTopology)k->topo, k->restart};
    VkPipelineViewportDepthClipControlCreateInfoEXT dcc = {VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_DEPTH_CLIP_CONTROL_CREATE_INFO_EXT, NULL, k->clip_neg1};
    VkPipelineViewportStateCreateInfo vs = {VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO, k->clip_neg1 && V.has_depth_clip_ctl ? &dcc : NULL, 0, 1, NULL, 1, NULL};
    VkPipelineRasterizationStateCreateInfo rs = {VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO, NULL, 0, k->depth_clamp, k->raster_discard,
        k->mode == 0 ? VK_POLYGON_MODE_POINT : k->mode == 1 ? VK_POLYGON_MODE_LINE : VK_POLYGON_MODE_FILL, k->cull, k->front ? VK_FRONT_FACE_COUNTER_CLOCKWISE : VK_FRONT_FACE_CLOCKWISE,
        k->bias_en, 0, 0, 0, 1.0f};
    VkPipelineMultisampleStateCreateInfo ms = {VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO, NULL, 0, (VkSampleCountFlagBits)k->samples, 0, 0, NULL, k->a2c, 0};
    if (flat & 8) ms.alphaToCoverageEnable = 0;
    if (flat & 16) rs.depthBiasEnable = 0;
    VkStencilOpState sf = {sop(k->sfail[0]), sop(k->szpass[0]), sop(k->szfail[0]), cmpop(k->sf[0]), 0xff, 0xff, 0}, sb = {sop(k->sfail[1]), sop(k->szpass[1]), sop(k->szfail[1]), cmpop(k->sf[1]), 0xff, 0xff, 0};
    VkPipelineDepthStencilStateCreateInfo ds = {VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO, NULL, 0, k->dt, k->dt && k->dw, cmpop(k->df), 0, k->st, sf, sb, 0, 1};
    if (flat & 2) { ds.depthTestEnable = ds.depthWriteEnable = ds.stencilTestEnable = 0; }
    if (flat & 32) ds.depthTestEnable = ds.depthWriteEnable = 0;
    if (flat & 64) ds.stencilTestEnable = 0;
    VkPipelineColorBlendAttachmentState cba[8];
    for (uint32_t i = 0; i < k->ncol; i++) cba[i] = (VkPipelineColorBlendAttachmentState){(k->blend_en >> i) & 1, bfac(k->bl[i].src), bfac(k->bl[i].dst), bop(k->bl[i].eq),
        bfac(k->bl[i].srca), bfac(k->bl[i].dsta), bop(k->bl[i].eqa), k->wmask[i]};
    if (flat & 4) for (uint32_t i = 0; i < k->ncol; i++) { cba[i].blendEnable = 0; cba[i].colorWriteMask = 0xf; }
    VkPipelineColorBlendStateCreateInfo cb = {VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO, NULL, 0, k->logic_en, (VkLogicOp)k->logic_op, k->ncol, cba, {0, 0, 0, 0}};
    VkDynamicState dyn[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR, VK_DYNAMIC_STATE_BLEND_CONSTANTS, VK_DYNAMIC_STATE_STENCIL_COMPARE_MASK,
                            VK_DYNAMIC_STATE_STENCIL_WRITE_MASK, VK_DYNAMIC_STATE_STENCIL_REFERENCE, VK_DYNAMIC_STATE_DEPTH_BIAS, VK_DYNAMIC_STATE_LINE_WIDTH};
    VkPipelineDynamicStateCreateInfo dy = {VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO, NULL, 0, sizeof dyn / sizeof *dyn, dyn};
    VkFormat cf[8]; for (uint32_t i = 0; i < k->ncol; i++) cf[i] = (VkFormat)k->cfmt[i];
    VkFormat df = (VkFormat)k->dfmt;
    int has_stencil = df == VK_FORMAT_D24_UNORM_S8_UINT || df == VK_FORMAT_D32_SFLOAT_S8_UINT || df == VK_FORMAT_S8_UINT;
    VkPipelineRenderingCreateInfo rci = {VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO, NULL, 0, k->ncol, cf, df == VK_FORMAT_S8_UINT ? VK_FORMAT_UNDEFINED : df, has_stencil ? df : VK_FORMAT_UNDEFINED};
    VkRenderPass prp = rp_on() && !noatt_dyn(k->ncol, k->dfmt) ? rp_compat(k->ncol, k->cfmt, k->dfmt, k->samples) : VK_NULL_HANDLE;   // with render passes, pipelines are made against a compatible one (noatt_dyn: none)
    if (!prp && !V.has_dynrender) { build_end(bs); return (VkPipeline)1; }   // no compatible render pass and no dynamic rendering: no valid pipeline (pass_commit skips such passes)
    VkGraphicsPipelineCreateInfo ci = {VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, prp ? NULL : &rci, 0, ns, ss, &vi, &ia, NULL, &vs, &rs, &ms, &ds, &cb, &dy, layout, prp, 0, VK_NULL_HANDLE, -1};
    if (bs >= 0) g_build[bs].phase = 2;
    VkResult r = vkCreateGraphicsPipelines(V.dev, V.pcache, 1, &ci, NULL, &p);
    vk_pipeline_cache_changed();
    build_end(bs);
    static uint64_t n; uint64_t c = __atomic_add_fetch(&n, 1, __ATOMIC_RELAXED);
    if (r) { LOG("[RED] [vk] graphics pipeline failed (%d): shaders %016llx %016llx %016llx", r, (unsigned long long)k->key[0], (unsigned long long)k->key[1], (unsigned long long)k->key[2]); p = (VkPipeline)1; }
    else if (c % 200 == 0) VKLOG("%llu graphics pipelines", (unsigned long long)c);
    __atomic_add_fetch(&g_vkst.pipes_built, 1, __ATOMIC_RELAXED);
    return p;   // (VkPipeline)1 = failed
}

// ---------------- descriptors ----------------
static uint64_t c2_handle(int st, int off) {   // the driver's handle table c2, as dmsh_build.py lays it out
    if (off >= 0x20 && off < 0x120) return S.tex[st][(off - 0x20) / 8];
    if (off >= 0x120 && off < 0x160) return S.img[st][(off - 0x120) / 8];
    if (off >= 0x168 && off < 0x568) return S.tex[st][128 + (off - 0x168) / 8];
    if (off >= 0x568 && off < 0x968) return S.samp[st][(off - 0x568) / 8];
    return 0;
}
static uint32_t ubo_word(int st, int reg, int off) {
    const p_bind_buf *u = reg >= 0 && reg < 16 ? &S.ubo[st][reg] : NULL; uint32_t v = 0;
    if (u && u->addr && off >= 0 && (uint64_t)off + 4 <= (u->size ? u->size : 4) && vk_pool_peek(u->addr + off, 4)) vk_ubd_read(u->addr + off, 4, (uint8_t *)&v);   // with the deferred updates
    return v;
}
// The texture and sampler pools the replayed command buffer last bound, as bases in the host's tables (hle_nvn.c dpool_base; 0 for the first pool of each kind). Handles that bind
// calls pass were rebased when recorded (hle_nvn.c bind_handles); handles the replay reads from the game's own memory (bindless: a uniform buffer word) are rebased here. DARK SOULS:
// REMASTERED binds one of 12 sampler pools and reads its scene's texture handles from its uniform buffers: every texture took the sampler of the same ID in the first pool, and its
// tiled walls and floors (texture coordinates of -3.5 and so on) were clamped into streaks
static int g_pool_tex_base, g_pool_samp_base;
int nvn_dpool_base(const void *pool, int tex);
static uint64_t mem_handle(uint64_t h) {
    uint64_t tid = h & 0xfffff, sid = h >> 20 & 0xfff;
    if (tid) tid += (uint64_t)g_pool_tex_base;
    if (sid) sid += (uint64_t)g_pool_samp_base;
    return tid | sid << 20;
}
static uint32_t g_dispatch[3], g_draw_binst;
typedef struct { VkBuffer b; VkDeviceSize off, len; } mesh_buf;   // a buffer range of the mesh emulation (mesh_draw)
static struct { mesh_buf ctl, taskout, launch, vtx, idx; } g_mesh;   // what stage_set binds to the RK_MESH_* bindings of the kernel being set up (mesh_draw)
#include "gfx_vk_gs_claim_runtime.h"
static struct { uint32_t count, instances; int prim, indirect; } g_claim_draw;
static VkDescriptorBufferInfo g_draw_claim;
// Fresh host-coherent ranges are disjoint for every draw and live until the owning submission slot completes.
// They are never cleared/reused while an earlier command-buffer part can execute; no pass cut or GPU clear is needed.
static void gs_claim_prepare(const vkshader_t *s) {
    g_draw_claim = (VkDescriptorBufferInfo){0};
    if (!s || !s->gs_claim_invocations) return;
    g_draw_claim = (VkDescriptorBufferInfo){vk_zero_buffer(), 0, 4}; // enabled=0: unsupported draws preserve their original shader behavior
    static uint64_t epoch, used; static int bucketed;
    if (epoch != g_set_epoch) { epoch = g_set_epoch; used = 0;
        const char *e = getenv("DM_VK_GS_CLAIM_BUCKET"); bucketed = !(e && !atoi(e)); } // live per submission;0 restores exact ranges for A/B

    uint64_t alignment = V.props.limits.minStorageBufferOffsetAlignment; if (alignment < 4) alignment = 4;
    int reason; uint64_t n = gs_claim_size(g_claim_draw.count, g_claim_draw.instances, s->gs_claim_invocations,
        g_claim_draw.prim, g_claim_draw.indirect, alignment, GS_CLAIM_BUDGET - used, V.props.limits.maxStorageBufferRange, &reason);
    if (!n) {
        static unsigned seen;
        if (reason > GS_CLAIM_EMPTY && !(seen & (1u << reason))) { seen |= 1u << reason;
            static const char *const why[] = {"", "empty draw", "GPU/indirect draw count", "multiple instances", "unproven topology", "claim slot bound", "storage range limit", "submission scratch budget"};
            VKLOG("[RED] geometry side-effect claims disabled: %s (shader %016llx, count %u, instances %u, primitive %d); original shader behavior retained",
                why[reason], (unsigned long long)s->key, g_claim_draw.count, g_claim_draw.instances, g_claim_draw.prim); }
        return;
    }
    // Padding only helps when the claim offset is dynamic: static descriptors still
    // differ by offset. Layouts are immutable, so cache this per shader transition.
    static const vkshader_t *layout_shader; static int dynamic_claim;
    if (layout_shader != s) { layout_shader = s; dynamic_claim = 0;
        for (int i = 0; i < s->nbind; i++) if (s->b[i].kind == RK_GS_CLAIM && s->dyn[i] == 2) { dynamic_claim = 1; break; } }
    n = gs_claim_range(n, alignment, GS_CLAIM_BUDGET - used, V.props.limits.maxStorageBufferRange, bucketed && dynamic_claim);
    uint32_t *p = vk_scratch(n, alignment, &g_draw_claim.buffer, &g_draw_claim.offset);
    memset(p, 0, (size_t)n); p[0] = 1; g_draw_claim.range = n; used += n + alignment - 1;
}

#define SB_ALIGN 256   // storage buffers are bound from this boundary below their address (dm/android/tools/shader_vk.py SB_ALIGN, which the shaders were translated for)
int dm_sdk_ounce(void);   // loader.c: the game's sdk is built for the Switch 2 (NVN2)
static void drv_segment(int st, int bank, uint8_t *d, int dyn) {
    memset(d, 0, 0x1000);
    if (bank == 0) {
        int base = st == 0 ? DMSH_C0_SSBO_VS : st == 5 ? DMSH_C0_SSBO_CS : st == 1 ? DMSH_C0_SSBO_PS : st == 2 ? DMSH_C0_SSBO_GS : st == 3 ? DMSH_C0_SSBO_TS : st == 4 ? DMSH_C0_SSBO_MS : -1;   // 3 / 4: NVN2's task and mesh stages (mesh_draw)   // (the geometry stage's table was left out before: a geometry shader reading its storage buffer addresses from c0 read 0)
        if (base >= 0) for (int r = 0; r < 16; r++) { uint64_t a = S.ssbo[st][r].addr; uint32_t n = (uint32_t)S.ssbo[st][r].size; memcpy(d + base + 16 * r, &a, 8); memcpy(d + base + 16 * r + 8, &n, 4); }
        if (st != 5) { const float mp = 2047.0f; memcpy(d + 0x610, &mp, 4); memcpy(d + 0x644, &g_draw_binst, 4); }
        else { memcpy(d + 0x200, g_dispatch, 12); if (dm_sdk_ounce()) memcpy(d + 4, g_dispatch, 12);   // NVN2 compute reads the workgroup counts at +4 (DMSH_C0_DISPATCH_NVN2), below its per-shader layout
               for (int r = 0; r < 16; r++) { uint64_t a = S.ubo[5][r].addr; uint32_t n = (uint32_t)S.ubo[5][r].size; memcpy(d + 0x210 + 16 * r, &a, 8); memcpy(d + 0x210 + 16 * r + 8, &n, 4); } }
    } else if (bank == 24) {   // NVN2 push constants (DMSH_PUSH_BANK): the graphics stages' block, or compute's (OP_PUSH_CONST)
        memcpy(d, S.push[st == 5], sizeof S.push[0]);
    } else if (!dyn) {
        for (int off = 0x20; off < 0x968; off += 8) { uint64_t h = c2_handle(st, off); memcpy(d + off, &h, 8); }
    } else {
        // Slot-index handles only for profiles declaring tex_dyn_slots.
        for (int r = 0; r < 8; r++) { uint32_t h = (uint32_t)S.img[st][r]; memcpy(d + 0x120 + 8*r, &h, 4); }
        for (int r = 0; r < 32; r++) { uint64_t h = S.tex[st][r]; uint32_t v = h ? (uint32_t)r | ((h >> 20) ? (uint32_t)r << 20 : 0) : 0; memcpy(d + 0x20 + 8*r, &v, 4); }
        for (int r = 0; r < 128; r++) {
            uint32_t v = S.tex[st][128+r] ? (uint32_t)r : 0; memcpy(d + 0x168 + 8*r, &v, 4);
            v = S.samp[st][r] ? (uint32_t)r << 20 : 0; memcpy(d + 0x568 + 8*r, &v, 4);
        }
    }
}
// vk_samp_by_id without the registry lock while no sampler ID was registered with other parameters (g_samp_gen): a 32-slot texture array took the lock
// per slot, a lock the game thread takes for every texture it registers (DM_NVN_ASYNC). VkSamplers are never destroyed, so a kept handle stays valid
extern uint64_t g_samp_gen;
static VkSampler samp_cached(int sid) {
    enum { SC = 1024 }; static struct { uint64_t gen, job; int sid; VkSampler s; } c[SC]; unsigned k = (unsigned)sid % SC;
    uint64_t nvn_replaying_job(void);
    uint64_t job = nvn_logical_queues_on() ? nvn_replaying_job() : 0;
    uint64_t g = __atomic_load_n(&g_samp_gen, __ATOMIC_ACQUIRE);   // read before the lookup: a registration racing it leaves the entry outdated, never wrong and current
    if (c[k].gen == g && c[k].job == job && c[k].sid == sid) return c[k].s;
    VkSampler s = vk_samp_by_id(sid); c[k].gen = g; c[k].job = job; c[k].sid = sid; c[k].s = s; return s;
}
// Separate dynamic sampler arrays are explicit slot bindings too: their recorded
// sequence takes precedence over the job's implicit/bindless sampler cutoff.
static VkSampler samp_dynamic_slot(int st, int slot, int dynslots) {
    if (!dynslots) return VK_NULL_HANDLE;
    int sid = (int)(S.samp[st][slot] >> 20);
    uint64_t seq = S.samp_sseq[st][slot];
    if (!sid) { sid = (int)(S.tex[st][slot] >> 20); seq = S.tex_sseq[st][slot]; }
    if (!sid) return VK_NULL_HANDLE;
    if (nvn_logical_queues_on() && g_samp_snap) {
        VkSampler vk_samp_at(int sid, uint64_t seq, int nearest);
        return seq ? vk_samp_at(sid, seq, 0) : VK_NULL_HANDLE;
    }
    return vk_samp_by_id(sid);
}
static uint64_t g_tex_sseq;   // stage_set: the bind count of the texture slot being collected (0 for handles read from memory)
static VkSampler samp_for(int st, const dmsv_bind *b, uint64_t texh, int nearest) {   // nearest: its unfiltered twin (integer views)
    int sid = 0; uint64_t seq = 0;
    switch (b->samp) {
    case RS_TEX: sid = (int)(texh >> 20); seq = g_tex_sseq; break;
    case RS_SLOT: sid = b->samp_arg >= 0 && b->samp_arg < 128 ? (int)(S.samp[st][b->samp_arg] >> 20) : 0; seq = sid ? S.samp_sseq[st][b->samp_arg] : 0; break;
    case RS_ID: sid = b->samp_arg; break;
    case RS_UBO: sid = (int)(mem_handle(ubo_word(st, b->samp_arg & 0xff, b->samp_arg >> 8) & ~0xfffffull) >> 20); break;
    default: break;
    }
    VkSampler vk_samp_by_id_nearest(int sid), vk_samp_at(int sid, uint64_t seq, int nearest); uint64_t mtl_samp_seq(void);
    if (!g_samp_snap || seq >= mtl_samp_seq()) seq = 0;   // no registration since the bind: the newest (the cached lookups)
    VkSampler s = !sid ? VK_NULL_HANDLE : seq ? vk_samp_at(sid, seq, nearest) : nearest ? vk_samp_by_id_nearest(sid) : samp_cached(sid);
    static int nsl; if (g_tracing || g_cs_buffer_collect || (g_fs_buffer_collect && st == 1) || (g_x_logfs && S.sh[1] && S.sh[1]->key == g_x_logfs && nsl++ < 60)) { char d[256]; void vk_samp_describe_at(int sid, uint64_t seq, char *out, size_t n); vk_samp_describe_at(sid, seq, d, sizeof d); VKLOG("  %sbinding %d: %s (texture handle %#llx, present %llu, pass %d, thread %d)", (g_tracing || g_cs_buffer_collect) ? "" : g_fs_buffer_collect && st == 1 ? "graphics " : "logfs ", b->binding, d, (unsigned long long)texh, (unsigned long long)g_tex_frame, g_sub_pass - 1, (int)gettid()); }
    return s ? s : vk_dummy_sampler();
}
// texture views by ID within a submission: a draw's texture arrays (32 slots) looked every slot up again, each through the registry's and the view table's
// locks (NEO: about 2 ms a frame). An entry holds while no texture ID is registered again and no view is dropped (g_tex_gen)
enum { TVC = 4096 };
static struct { uint64_t epoch, gen; int tid, dim, arr, num, storage, sc; VkImageView v; vkimg_t *im; } g_tvc[TVC];   // sc: the integer class of a storage view (img_fmt_cls)
// what a view's texels are, as gfx_vk_shader.c's integer-texel variants need it: 0 float, 1 uint (unsigned integer formats, a stencil), 2 int
static int view_num(const tex_t *t, const view_t *v) {
    const vkfmt_t *f = vk_fmt(v && v->set_format ? v->format : t->d.format); if (!f) return 0;
    if (f->depth || f->stencil) return f->stencil && (!f->depth || t->stencil || (v && v->set_dsmode && v->dsmode == 1));   // vk_tex_view's aspect
    return f->is_int ? (f->is_sint ? 2 : 1) : 0;
}
// the integer class a storage image binding's shader declares, from the SPIR-V image format the translator recorded (dmsv_bind.fmt, shader_vk.py): 1 uint, 2 sint,
// 0 float or none declared (the format 0, Unknown, is left alone)
static int img_fmt_cls(uint8_t f) {
    switch (f) {
    case 30: case 31: case 32: case 33: case 34: case 35: case 36: case 37: case 38: case 39: case 40: return 1;   // Rgba32ui .. R64ui
    case 21: case 22: case 23: case 24: case 25: case 26: case 27: case 28: case 29: case 41: return 2;           // Rgba32i .. R8i, R64i
    }
    return 0;
}
static uint64_t g_num_um, g_num_im;   // stage_set: bindings of the stage being built that got uint / int views (bit = reflection index)
extern uint64_t g_tex_gen;
// The exact sampled/storage views resolved by stage_set, for one traced native dispatch. Readback copies retain native bytes:
// the log records requested view format too, so a reinterpretation is never silently described as a conversion.
enum { CS_TRACE_IMAGES = 8, CS_TRACE_COPIES = 16 };
typedef struct { vkimg_t *im; uint32_t binding, level, layer, w, h, depth, bpp; VkFormat view_format; int storage, raw, graphics; uint64_t bytes; } cs_trace_image;
static cs_trace_image g_cs_images[CS_TRACE_IMAGES]; static unsigned g_cs_images_n, g_cs_images_copies;
static uint32_t cs_trace_bpp(VkFormat f) {
    switch (f) {
    case VK_FORMAT_R8_UNORM: return 1;
    case VK_FORMAT_R8G8_UNORM: case VK_FORMAT_R16_SFLOAT: return 2;
    case VK_FORMAT_R16G16B16A16_SFLOAT: return 8;
    case VK_FORMAT_R8G8B8A8_UNORM: case VK_FORMAT_R8G8B8A8_SRGB: case VK_FORMAT_B8G8R8A8_UNORM:
    case VK_FORMAT_B10G11R11_UFLOAT_PACK32: case VK_FORMAT_E5B9G9R9_UFLOAT_PACK32: case VK_FORMAT_A2B10G10R10_UNORM_PACK32: case VK_FORMAT_R16G16_SFLOAT: return 4;
    default: return 0;
    }
}
// Bound each native copy before multiplication; even UINT32_MAX extents cannot overflow.
static uint64_t cs_trace_image_bytes(uint32_t w, uint32_t h, uint32_t depth, uint32_t bpp) {
    uint64_t n = w, cap = 128u << 20;
    if (!w || !h || !depth || !bpp || n > cap / h) return 0;
    n *= h; if (n > cap / depth) return 0;
    n *= depth; if (n > cap / bpp) return 0;
    return n * bpp;
}
static void cs_trace_image_note(const dmsv_bind *b, const VkDescriptorImageInfo *ii, vkimg_t *im, const tex_t *t, const view_t *v, int storage) {
    if ((!g_cs_trace_collect && !g_fs_image_collect) || !ii->imageView || !im) return;
    int graphics = g_fs_image_collect;
    unsigned *used_copies = graphics ? &g_fs_image_copies : &g_cs_trace_copies;
    uint64_t *used_bytes = graphics ? &g_fs_image_bytes : &g_cs_trace_bytes;
    unsigned copy_limit = graphics ? 2 : CS_TRACE_COPIES;
    uint64_t byte_limit = graphics ? (32u << 20) : (256u << 20);
    unsigned pd_limit = g_pd_draw_mode ? PD_DRAW_MAX : PD_MAX;
    const char *why = NULL; unsigned copies = storage ? 2 : 1; uint32_t level = v && v->set_levels ? v->base_level : 0, layer = v && v->set_layers ? v->min_layer : 0;
    uint32_t bpp = cs_trace_bpp(im->fmt), w = 0, h = 0, depth = 1; int volume = im->vtype == VK_IMAGE_VIEW_TYPE_3D;
    if (g_pd_state != 1) why = "PASSDUMP is not capturing";
    else if (im->aspect != VK_IMAGE_ASPECT_COLOR_BIT || im->samples != 1 ||
        (!volume && (im->d != 1 || (im->vtype != VK_IMAGE_VIEW_TYPE_2D && im->vtype != VK_IMAGE_VIEW_TYPE_2D_ARRAY))))
        why = "requires single-sample 2D/3D colour image";
    else if (!(im->usage & VK_IMAGE_USAGE_TRANSFER_SRC_BIT)) why = "no transfer-source usage";
    else if (level >= im->levels || level >= 32 || layer >= im->layers || !im->w || !im->h || !im->d) why = "invalid selected mip/layer/extent";
    else if (volume && (layer || im->layers != 1 || (v && v->set_layers && v->nlayers != 1) ||
        (v && v->set_target && v->target != NVN_TEXTURE_TARGET_3D))) why = "3D capture requires the complete selected mip volume";
    else if (!volume && (b->flags & 1) && (!v || !v->set_layers || v->nlayers != 1)) why = "multi-layer image view";
    else if (!bpp) why = "native format is not supported by pass dump";
    if (!why) { w = im->w >> level; h = im->h >> level; if (!w) w = 1; if (!h) h = 1;
        if (volume) { depth = im->d >> level; if (!depth) depth = 1; }
        for (unsigned i = 0; i < g_cs_images_n; i++) if (g_cs_images[i].im == im && g_cs_images[i].level == level && g_cs_images[i].layer == layer && g_cs_images[i].storage == storage && g_cs_images[i].graphics == graphics) return;
        uint64_t bytes = cs_trace_image_bytes(w, h, depth, bpp);
        if (g_cs_images_n >= CS_TRACE_IMAGES || *used_copies + copies > copy_limit || g_npd + g_cs_images_copies + copies > pd_limit ||
            !bytes || (graphics && bytes > (16u << 20)) || copies * bytes > byte_limit || *used_bytes > byte_limit - copies * bytes)
            why = graphics ? "two-copy/32 MiB graphics image budget" : "capture count/256 MiB budget";
        else { const vkfmt_t *f = vk_fmt(v && v->set_format ? v->format : t->d.format);
            g_cs_images[g_cs_images_n++] = (cs_trace_image){.im=im, .binding=b->binding, .level=level, .layer=layer,
                .w=w, .h=h, .depth=depth, .bpp=bpp, .view_format=f ? f->vk : im->fmt, .storage=storage,
                .raw=graphics || volume || im->fmt == VK_FORMAT_R8G8_UNORM, .graphics=graphics, .bytes=bytes};
            *used_copies += copies; g_cs_images_copies += copies; *used_bytes += copies * bytes; }
    }
    if (why) VKLOG("%s trace image binding %u not captured: %s", graphics ? "graphics" : "compute", b->binding, why);
}
static void cs_trace_images_copy(VkCommandBuffer cb, uint64_t key, int after) {
    for (unsigned i = 0; i < g_cs_images_n && g_pd_state == 1 && g_npd < (g_pd_draw_mode ? PD_DRAW_MAX : PD_MAX); i++) {
        const cs_trace_image *im = &g_cs_images[i]; if (after && !im->storage) continue; pdump_t d = {0}; VkResult r;
        d.w = im->w; d.h = im->h; d.depth = im->depth; d.f = im->im->fmt; d.pass = im->graphics ? -3000 - g_draw_i : -2000 - (int)(2 * g_cs_trace_seen + (unsigned)after); d.present = g_tex_frame;
        d.raw_bytes = im->raw ? im->bytes : 0; d.key = key; d.binding = im->binding; d.dispatch = im->graphics ? (uint32_t)g_draw_i : g_cs_trace_seen;
        d.level = im->level; d.layer = im->layer; d.view_format = im->view_format; d.storage = im->storage; d.after = after;
        d.frame = vk_frame_current(); d.fence = mtl_last_fence() + 1;
        VkBufferCreateInfo bi = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, NULL, 0, im->bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_SHARING_MODE_EXCLUSIVE, 0, NULL};
        if ((r = vkCreateBuffer(V.dev, &bi, NULL, &d.b))) goto fail;
        VkMemoryRequirements req; vkGetBufferMemoryRequirements(V.dev, d.b, &req);
        uint32_t type = vk_mem_type(req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT, 0);
        if (type == UINT32_MAX) type = vk_mem_type(req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, 0);
        if (type == UINT32_MAX) { r = VK_ERROR_FEATURE_NOT_PRESENT; goto fail; }
        VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL, req.size, type};
        if ((r = vkAllocateMemory(V.dev, &ai, NULL, &d.m)) || (r = vkBindBufferMemory(V.dev, d.b, d.m, 0)) ||
            (r = vkMapMemory(V.dev, d.m, 0, VK_WHOLE_SIZE, 0, &d.p))) goto fail;
        vk_img_barrier_all(cb);
        VkBufferImageCopy copy = {0, 0, 0, {VK_IMAGE_ASPECT_COLOR_BIT, im->level, im->layer, 1}, {0, 0, 0}, {d.w, d.h, d.depth}};
        vkCmdCopyImageToBuffer(cb, im->im->img, VK_IMAGE_LAYOUT_GENERAL, d.b, 1, &copy);
        VkMemoryBarrier host = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, NULL, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT};
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &host, 0, NULL, 0, NULL);
        vk_img_barrier_all(cb);
        g_pd[g_npd] = d;
        VKLOG("%s trace dump %02d: key %016llx dispatch/draw %u %s %s binding %u image %p mip %u layer %u %ux%ux%u native/view format %d/%d bytes %llu present %llu frame %llu fence %llu after graphics pass %d%s",
            im->graphics ? "graphics" : "compute", g_npd++, (unsigned long long)key, d.dispatch, after ? "after" : "before", im->storage ? "storage" : "sampled", im->binding, (void *)im->im->img,
            im->level, im->layer, d.w, d.h, d.depth, d.f, im->view_format, (unsigned long long)im->bytes,
            (unsigned long long)d.present, (unsigned long long)d.frame, (unsigned long long)d.fence, g_sub_pass - 1, im->raw ? " native-raw" : "");
        continue;
fail:
        if (d.b) vkDestroyBuffer(V.dev, d.b, NULL);
        if (d.m) vkFreeMemory(V.dev, d.m, NULL);
        VKLOG("compute trace: image readback allocation failed (%d)", r);
    }
}
// Exact GPU buffer snapshots for selected native compute dispatches. Pool/scratch buffers
// have TRANSFER_SRC usage; immutable C1/support allocations do not and are never copied here.
// DM_VK_TRACE_CS_BUFFERS=1: storage before+after, uniform/driver before. At most 64 x 64 KiB.
enum { CS_BUFFER_BINDINGS = 64, CS_BUFFER_DUMPS = 64, CS_BUFFER_BYTES = 65536 };
typedef struct { VkDescriptorBufferInfo source; uint32_t binding; int kind, reg, storage, stage, graphics; uint64_t guest, shader; VkDeviceSize descriptor_offset, descriptor_range, window_offset; uint32_t bytes; } cs_trace_buffer;
typedef struct { VkBuffer b; VkDeviceMemory m; void *p; uint64_t fence, key, present, frame, id; uint32_t dispatch, binding, bytes; int after, kind, reg, stage, graphics, pass, draw; uint64_t guest, shader; VkDeviceSize source_offset, source_range, descriptor_offset, window_offset; } cs_buffer_dump;
static cs_trace_buffer g_cs_buffers[CS_BUFFER_BINDINGS]; static unsigned g_cs_buffers_n, g_cs_buffers_reserved;
static cs_buffer_dump g_cs_buffer_dump[CS_BUFFER_DUMPS]; static uint64_t g_cs_buffer_id;
static uint64_t g_trace_sbuf_guest; // resolved by sbuf_desc itself; diagnostics never peek guest memory
static void cs_trace_buffer_note_one(int st, uint64_t shader, const dmsv_bind *b,
        const VkDescriptorBufferInfo *source, int graphics, int storage, uint64_t guest,
        VkDeviceSize descriptor_offset, VkDeviceSize descriptor_range, VkDeviceSize window_offset) {
    unsigned *budget = graphics ? &g_fs_buffer_copies : &g_cs_buffer_copies;
    const char *why = NULL; unsigned copies = storage && !graphics ? 2 : 1, active = 0;
    uint64_t active_bytes = 0, pending_bytes = 0;
    uint32_t cap = graphics && storage ? g_fs_buffer_sbuf_max : CS_BUFFER_BYTES;
    uint32_t bytes = (uint32_t)(source->range < cap ? source->range : cap);
    for (unsigned i = 0; i < CS_BUFFER_DUMPS; i++) if (g_cs_buffer_dump[i].b) {
        active++; if (graphics && g_cs_buffer_dump[i].graphics) active_bytes += g_cs_buffer_dump[i].bytes;
    }
    if (graphics) for (unsigned i = 0; i < g_cs_buffers_n; i++) if (g_cs_buffers[i].graphics) pending_bytes += g_cs_buffers[i].bytes;
    if (!source->buffer || source->buffer == vk_zero_buffer()) why = "unbound/static zero buffer has no transfer-source usage";
    else if (!source->range || source->range == VK_WHOLE_SIZE || source->offset % 4 || source->range % 4 ||
             source->offset > UINT64_MAX - source->range) why = "requires finite, 4-byte aligned descriptor range";
    else if (g_cs_buffers_n >= CS_BUFFER_BINDINGS || *budget + copies > CS_BUFFER_DUMPS ||
             active + g_cs_buffers_reserved + copies > CS_BUFFER_DUMPS) why = "64-readback budget";
    else if (graphics && (g_fs_buffer_bytes > FS_BUFFER_BYTE_BUDGET || bytes > FS_BUFFER_BYTE_BUDGET - g_fs_buffer_bytes ||
             active_bytes + pending_bytes > FS_BUFFER_BYTE_BUDGET ||
             bytes > FS_BUFFER_BYTE_BUDGET - active_bytes - pending_bytes)) why = "48 MiB graphics readback budget";
    if (why) { VKLOG("%s buffer stage %d binding %u not captured: %s", graphics ? "graphics" : "compute", st, b->binding, why); return; }
    g_cs_buffers[g_cs_buffers_n++] = (cs_trace_buffer){*source, b->binding, b->kind, b->reg, storage, st, graphics, guest, shader,
        descriptor_offset, descriptor_range, window_offset, bytes};
    if (graphics) g_fs_buffer_bytes += bytes;
    *budget += copies; g_cs_buffers_reserved += copies;
}
static void cs_trace_buffer_note(int st, uint64_t shader, const dmsv_bind *b, const VkDescriptorBufferInfo *source) {
    int graphics = st >= 0 && st < 3 && g_fs_buffer_collect;
    if (!graphics && !(st == 5 && g_cs_buffer_collect)) return;
    int storage = b->kind == RK_SBUF || b->kind == RK_SBUF_UBO;
    if (!storage && b->kind != RK_CBUF && b->kind != RK_DRV) return;
    uint64_t guest = 0;
    if (b->kind == RK_SBUF && b->reg >= 0 && b->reg < 16) guest = S.ssbo[st][b->reg].addr;
    if (b->kind == RK_SBUF_UBO) guest = g_trace_sbuf_guest;
    if (b->kind == RK_CBUF && b->reg >= 0 && b->reg < 16) guest = S.ubo[st][b->reg].addr;
    VkDeviceSize prefix = storage ? guest % SB_ALIGN : 0;
    // Full default capture begins at the actual descriptor base (including its alignment prefix).
    cs_trace_buffer_note_one(st, shader, b, source, graphics, storage, guest - prefix,
        source->offset, source->range, 0);
    if (!graphics || !storage || source->range == VK_WHOLE_SIZE) return;
    for (unsigned i = 0; i < g_fs_buffer_windows_n; i++) {
        const fs_buffer_window *w = &g_fs_buffer_windows[i];
        if (w->stage != (unsigned)st || w->reg != (unsigned)b->reg) continue;
        if (prefix > source->range || w->offset > source->range - prefix ||
            w->bytes > source->range - prefix - w->offset || source->offset > UINT64_MAX - prefix - w->offset ||
            guest > UINT64_MAX - w->offset) {
            VKLOG("graphics buffer stage %d reg %d window %#llx+%#llx outside descriptor range %#llx", st, b->reg,
                (unsigned long long)w->offset, (unsigned long long)w->bytes, (unsigned long long)source->range); continue;
        }
        VkDescriptorBufferInfo sub = {source->buffer, source->offset + prefix + w->offset, w->bytes};
        cs_trace_buffer_note_one(st, shader, b, &sub, 1, storage, guest + w->offset,
            source->offset, source->range, prefix + w->offset);
    }
}
static void cs_trace_buffers_copy(VkCommandBuffer cb, uint64_t key, int after) {
    for (unsigned i = 0; i < g_cs_buffers_n; i++) {
        const cs_trace_buffer *s = &g_cs_buffers[i]; if (after && (!s->storage || s->graphics)) continue;
        unsigned slot = 0; while (slot < CS_BUFFER_DUMPS && g_cs_buffer_dump[slot].b) slot++;
        if (slot == CS_BUFFER_DUMPS) { VKLOG("compute buffer capture has no free completion slot"); return; }
        cs_buffer_dump d = {0}; VkResult r;
        d.key = key; d.present = g_tex_frame; d.frame = vk_frame_current(); d.dispatch = g_cs_trace_seen; d.binding = s->binding;
        d.kind = s->kind; d.reg = s->reg; d.guest = s->guest; d.after = after;
        d.stage = s->stage; d.graphics = s->graphics; d.shader = s->shader;
        d.draw = s->graphics ? g_draw_i : -1; d.pass = s->graphics ? g_sub_pass : -1;
        d.source_offset = s->source.offset; d.source_range = s->descriptor_range;
        d.descriptor_offset = s->descriptor_offset; d.window_offset = s->window_offset;
        d.bytes = s->bytes; // immutable validated size reserved at descriptor collection
        VkBufferCreateInfo bi = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, NULL, 0, d.bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_SHARING_MODE_EXCLUSIVE, 0, NULL};
        if ((r = vkCreateBuffer(V.dev, &bi, NULL, &d.b))) goto fail;
        VkMemoryRequirements req; vkGetBufferMemoryRequirements(V.dev, d.b, &req);
        uint32_t type = vk_mem_type(req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT, 0);
        if (type == UINT32_MAX) type = vk_mem_type(req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, 0);
        if (type == UINT32_MAX) { r = VK_ERROR_FEATURE_NOT_PRESENT; goto fail; }
        VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL, req.size, type};
        if ((r = vkAllocateMemory(V.dev, &ai, NULL, &d.m)) || (r = vkBindBufferMemory(V.dev, d.b, d.m, 0)) ||
            (r = vkMapMemory(V.dev, d.m, 0, VK_WHOLE_SIZE, 0, &d.p))) goto fail;
        vk_img_barrier_all(cb); // producer writes -> this transfer read, also across earlier queue submissions
        VkBufferCopy copy = {s->source.offset, 0, d.bytes};
        vkCmdCopyBuffer(cb, s->source.buffer, d.b, 1, &copy);
        VkBufferMemoryBarrier host = {VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER, NULL,
            VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT, VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, d.b, 0, d.bytes};
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 0, NULL, 1, &host, 0, NULL);
        vk_img_barrier_all(cb); // finish source reads before the dispatch or a later writer
        d.fence = mtl_last_fence() + 1; d.id = ++g_cs_buffer_id; g_cs_buffer_dump[slot] = d;
        VKLOG("%s buffer queued %llu: key %016llx dispatch %u %s binding %u kind %d reg %d guest %#llx source %p+%llu range %llu captured %u bytes fence %llu present %llu frame %llu",
            d.graphics ? "graphics" : "compute", (unsigned long long)d.id, (unsigned long long)key, d.dispatch, after ? "after" : "before", d.binding, d.kind, d.reg,
            (unsigned long long)d.guest, (void *)s->source.buffer, (unsigned long long)d.source_offset, (unsigned long long)d.source_range,
            d.bytes, (unsigned long long)d.fence, (unsigned long long)d.present, (unsigned long long)d.frame);
        continue;
fail:
        if (d.b) vkDestroyBuffer(V.dev, d.b, NULL);
        if (d.m) vkFreeMemory(V.dev, d.m, NULL);
        VKLOG("compute buffer capture allocation failed (%d), key %016llx binding %u %s", r, (unsigned long long)key, s->binding, after ? "after" : "before");
    }
}
static void cs_trace_buffers_poll(uint64_t done_fence) {
    for (unsigned i = 0; i < CS_BUFFER_DUMPS; i++) {
        cs_buffer_dump *d = &g_cs_buffer_dump[i]; if (!d->b || d->fence > done_fence) continue;
        VkMappedMemoryRange mr = {VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE, NULL, d->m, 0, VK_WHOLE_SIZE};
        VkResult r = vkInvalidateMappedMemoryRanges(V.dev, 1, &mr);
        const char *root = getenv("DM_ROOT"); char path[1024];
        if (d->graphics) snprintf(path, sizeof path, "%s/fsbuf_%04llu_%016llx_s%d_b%u_draw%d_before.bin", root ? root : ".",
            (unsigned long long)d->id, (unsigned long long)d->key, d->stage, d->binding, d->draw);
        else snprintf(path, sizeof path, "%s/csbuf_%04llu_%016llx_b%u_%s.bin", root ? root : ".",
            (unsigned long long)d->id, (unsigned long long)d->key, d->binding, d->after ? "after" : "before");
        size_t written = 0; FILE *f = !r ? fopen(path, "wb") : NULL;
        if (f) { written = fwrite(d->p, 1, d->bytes, f); fclose(f); }
        if (d->graphics) {
            char meta[1040]; snprintf(meta, sizeof meta, "%s.json", path); FILE *mf = fopen(meta, "wb");
            if (mf) { fprintf(mf, "{\"fragment_key\":\"%016llx\",\"shader_key\":\"%016llx\",\"stage\":%d,\"binding\":%u,\"kind\":%d,\"reg\":%d,"
                "\"guest\":\"%#llx\",\"native_offset\":%llu,\"descriptor_range\":%llu,\"descriptor_offset\":%llu,\"window_offset\":%llu,\"bytes\":%u,\"written\":%zu,\"present\":%llu,\"frame\":%llu,"
                "\"pass\":%d,\"draw\":%d,\"fence\":%llu,\"phase\":\"before\",\"result\":%d}\n",
                (unsigned long long)d->key, (unsigned long long)d->shader, d->stage, d->binding, d->kind, d->reg,
                (unsigned long long)d->guest, (unsigned long long)d->source_offset, (unsigned long long)d->source_range, (unsigned long long)d->descriptor_offset, (unsigned long long)d->window_offset, d->bytes, written,
                (unsigned long long)d->present, (unsigned long long)d->frame, d->pass, d->draw, (unsigned long long)d->fence, r); fclose(mf); }
        }
        VKLOG("%s buffer landed %llu: key %016llx dispatch %u %s binding %u kind %d reg %d guest %#llx offset %llu range %llu bytes %u written %zu fence %llu present %llu frame %llu result %d -> %s",
            d->graphics ? "graphics" : "compute", (unsigned long long)d->id, (unsigned long long)d->key, d->dispatch, d->after ? "after" : "before", d->binding, d->kind, d->reg,
            (unsigned long long)d->guest, (unsigned long long)d->source_offset, (unsigned long long)d->source_range, d->bytes, written,
            (unsigned long long)d->fence, (unsigned long long)d->present, (unsigned long long)d->frame, r, path);
        vkDestroyBuffer(V.dev, d->b, NULL); vkFreeMemory(V.dev, d->m, NULL); memset(d, 0, sizeof *d);
    }
}
// Bounded native-upscaler diagnostics: inspect the descriptors actually selected, without disabling their caches.
static int g_up_native_trace;
static void up_native_texture_trace(const dmsv_bind *b, uint64_t h, const VkDescriptorImageInfo *ii, vkimg_t *im, int storage, int cached) {
    if (!g_up_native_trace) return;
    const tex_t *t = NULL; const view_t *v = NULL; int tid = (int)(h & 0xfffff);
    int found = tid && vk_tex_by_id(tid, &v, &t) && t;
    if (found) cs_trace_image_note(b, ii, im, t, v, storage);
    VKLOG("compute trace binding %u %s: handle %#llx id %d cached %d view %p image %p, descriptor layout %d image layout %d, usage %#x flags %#x, %ux%u fmt %d gpu_w %d wseq %llu",
        b->binding, storage ? "storage" : "sampled", (unsigned long long)h, tid, cached, (void *)ii->imageView, im ? (void *)im->img : NULL,
        ii->imageLayout, im ? im->layout : 0, im ? im->usage : 0, found ? t->d.flags : 0, im ? im->w : 0, im ? im->h : 0, im ? im->fmt : 0,
        im ? (int)im->gpu_w : -1, im ? (unsigned long long)im->wseq : 0);
    if (found) VKLOG("compute trace binding %u registration: serial %llu pool %p+%#llx bytes %#llx nvn fmt %#x target %d; texture swizzle %d/%#x; view fmt %d/%#x target %d/%d levels %d/%u+%u layers %d/%u+%u swizzle %d/%d,%d,%d,%d",
        b->binding, (unsigned long long)t->serial, (void *)t->pool, (unsigned long long)t->off, (unsigned long long)t->L.size, t->d.format, t->d.target, t->set_swz, t->swz,
        v ? v->set_format : 0, v ? v->format : 0, v ? v->set_target : 0, v ? v->target : 0, v ? v->set_levels : 0, v ? v->base_level : 0, v ? v->nlevels : 0,
        v ? v->set_layers : 0, v ? v->min_layer : 0, v ? v->nlayers : 0, v ? v->set_swizzle : 0, v ? v->swz[0] : 0, v ? v->swz[1] : 0, v ? v->swz[2] : 0, v ? v->swz[3] : 0);
}
// Observe the descriptor just resolved by the normal path, without changing cache
// eligibility or creating another view. Only the requested binding queues a native copy.
static void fs_trace_texture_note(int st, const dmsv_bind *b, uint64_t h,
                                  const VkDescriptorImageInfo *ii, vkimg_t *im) {
    if (!g_fs_buffer_collect || st != 1) return;
    const tex_t *t = NULL; const view_t *v = NULL; int tid = (int)(h & 0xfffff);
    int found = tid && vk_tex_by_id(tid, &v, &t) && t;
    const vkfmt_t *f = found ? vk_fmt(v && v->set_format ? v->format : t->d.format) : NULL;
    VKLOG("graphics image descriptor: present=%llu draw=%d binding=%u reg=%d handle=%#llx sampler=%p view=%p image=%p layout=%u serial=%llu native/view=%u/%u extent=%ux%ux%u levels=%u layers=%u usage=%#x aspect=%#x samples=%u",
        (unsigned long long)g_tex_frame, g_draw_i, b->binding, b->reg, (unsigned long long)h,
        (void *)ii->sampler, (void *)ii->imageView, im ? (void *)im->img : NULL, (unsigned)ii->imageLayout,
        found ? (unsigned long long)t->serial : 0ull, im ? (unsigned)im->fmt : 0u, f ? (unsigned)f->vk : 0u,
        im ? im->w : 0u, im ? im->h : 0u, im ? im->d : 0u, im ? im->levels : 0u, im ? im->layers : 0u,
        im ? im->usage : 0u, im ? im->aspect : 0u, im ? im->samples : 0u);
    if (found) VKLOG("graphics image view: binding=%u pool=%p+%#llx nvn-format=%#x target=%d texture-swizzle=%d/%#x view-format=%d/%#x target=%d/%d mip=%u+%u layer=%u+%u swizzle=%d/%d,%d,%d,%d",
        b->binding, (void *)t->pool, (unsigned long long)t->off, t->d.format, t->d.target, t->set_swz, t->swz,
        v ? v->set_format : 0, v ? v->format : 0, v ? v->set_target : 0, v ? v->target : 0,
        v && v->set_levels ? v->base_level : 0, v && v->set_levels ? v->nlevels : im ? im->levels : 0,
        v && v->set_layers ? v->min_layer : 0, v && v->set_layers ? v->nlayers : im ? im->layers : 0,
        v ? v->set_swizzle : 0, v ? v->swz[0] : 0, v ? v->swz[1] : 0, v ? v->swz[2] : 0, v ? v->swz[3] : 0);
    if ((int)b->binding != g_fs_image_binding) return;
    if (!found || !im || !ii->imageView) { VKLOG("graphics image binding %u not captured: unresolved or dummy descriptor", b->binding); return; }
    g_fs_image_collect = 1; cs_trace_image_note(b, ii, im, t, v, 0); g_fs_image_collect = 0;
}
static vkimg_t *g_desc_im;   // the image tex_desc_for's last view is of (NULL: the dummy), for the ring of pool tables
static int tex_desc_for(int st, const dmsv_bind *b, uint64_t h, VkDescriptorImageInfo *ii, int storage) {   // the view's view_num
    g_desc_im = NULL;
    int tid = (int)(h & 0xfffff); const view_t *v = NULL; const tex_t *t = NULL;
    ii->imageLayout = VK_IMAGE_LAYOUT_GENERAL; ii->sampler = storage ? VK_NULL_HANDLE : samp_for(st, b, h, 0); ii->imageView = VK_NULL_HANDLE;
    int sc = storage ? img_fmt_cls(b->fmt) : 0;   // FINAL FANTASY TACTICS' UI composite declares its window image r32ui over an RGBA8 texture: the view is in the shader's class
    unsigned ck = ((unsigned)tid * 2654435761u ^ (unsigned)b->dim * 97u ^ (unsigned)(b->flags & 1) ^ (unsigned)storage * 17u ^ (unsigned)sc * 0x9E37u) % TVC;
    int logfs = g_x_logfs && S.sh[1] && S.sh[1]->key == g_x_logfs && st == 1; static int nlogfs; if (logfs && nlogfs >= 60) logfs = 0;
    if (tid && g_var_now != 3 && !g_tracing && !logfs && g_tvc[ck].epoch == g_set_epoch && g_tvc[ck].gen == g_tex_gen && g_tvc[ck].tid == tid && g_tvc[ck].dim == b->dim && g_tvc[ck].arr == (b->flags & 1) && g_tvc[ck].storage == storage && g_tvc[ck].sc == sc) {
        ii->imageView = g_tvc[ck].v; g_desc_im = g_tvc[ck].im; if (g_tvc[ck].im) { g_tvc[ck].im->last_use = mtl_last_fence() + 1; if (storage) g_tvc[ck].im->gpu_w = 1; } vk_texdump_note(g_tvc[ck].im);
        if (g_tvc[ck].num && !storage) ii->sampler = samp_for(st, b, h, 1);
        if (g_up_native_trace) up_native_texture_trace(b, h, ii, g_tvc[ck].im, storage, 1);
        return g_tvc[ck].num;
    }
    int num = 0;
    if (tid && g_var_now != 3 && vk_tex_by_id(tid, &v, &t) && t) {
        // a storage image the shader declares as integers over a float or unorm texture of the same texel size: a view of the same image in that class, so the shader
        // reads the texels' bits (FINAL FANTASY TACTICS' UI composite loads its RGBA8 window as r32ui; through a float view its blend came out yellow and white). Integer
        // textures keep their own views and the shader variants of gfx_vk_shader.c
        extern __thread VkFormat vk_view_fmt;
        VkFormat vf = 0; const vkfmt_t *tf = sc && !view_num(t, v) ? vk_fmt(v && v->set_format ? v->format : t->d.format) : NULL;
        if (tf && !tf->compressed && !tf->dec && !tf->depth && !tf->stencil) vf = vk_int_fmt(tf->bpb, sc == 2);
        vkimg_t *im = NULL; ii->imageView = vf ? vk_tex_view_as(t, v, b->dim, b->flags & 1, storage ? 2 : 0, &im, vf) : vk_tex_view(t, v, b->dim, b->flags & 1, storage ? 2 : 0, &im); if (storage && im) im->gpu_w = 1; g_desc_im = im;
        if (vf && ii->imageView && vk_view_fmt == vf) { static int once; if (!once++) VKLOG("storage image: nvn format %#x viewed as vk %d for a %s image binding", t->d.format, vf, sc == 1 ? "uint" : "int"); }
        if (logfs) nlogfs++;
        if (g_tracing || logfs) VKLOG("  %stexture binding %d handle %#llx id %d: %ux%u nvn format %#x target %d serial %llu pool %p+%#llx size %#llx → image %p (%ux%u fmt %d layers %u levels %u, gpu-written %d) view %p", logfs ? "logfs " : "", b->binding, (unsigned long long)h, tid, t->d.width, t->d.height, t->d.format, t->d.target,
                             (unsigned long long)t->serial, (const void *)t->pool, (unsigned long long)t->off, (unsigned long long)t->L.size, (void *)im, im ? im->w : 0, im ? im->h : 0, im ? im->fmt : 0, im ? im->layers : 0, im ? im->levels : 0, im ? (int)im->gpu_w : -1, (void *)ii->imageView);
        if (im) im->last_use = mtl_last_fence() + 1;
        if (ii->imageView) num = view_num(t, v);
        if (g_up_native_trace) up_native_texture_trace(b, h, ii, im, storage, 0);
        if (ii->imageView) { g_tvc[ck].epoch = g_set_epoch; g_tvc[ck].gen = g_tex_gen; g_tvc[ck].tid = tid; g_tvc[ck].dim = b->dim; g_tvc[ck].arr = b->flags & 1; g_tvc[ck].storage = storage; g_tvc[ck].sc = sc; g_tvc[ck].v = ii->imageView; g_tvc[ck].im = im; g_tvc[ck].num = num; }
    }
    if (num && !storage) ii->sampler = samp_for(st, b, h, 1);
    if (!ii->imageView) {
        const vkfmt_t *f = t ? vk_fmt(t->d.format) : NULL;
        ii->imageView = vk_dummy_view(b->dim == 0xff ? 1 : b->dim, b->flags & 1, (b->flags & 2) != 0, storage, f && f->is_int); num = f && f->is_int ? (f->is_sint ? 2 : 1) : 0;
    }
    return num;
}
// Resolve every stage's storage images before building any descriptors or starting graphics rendering.
// An image may need one content-preserving allocation upgrade; doing that during stage_set would invalidate
// an already-built sampled descriptor or end the render pass beneath a prepared draw.
static void stage_storage_prepare(int st, vkshader_t *s) {
    if (!s) return;
    if (!s->storage_images) {
        s->storage_images = 2;
        for (int i = 0; i < s->nbind; i++) if (s->b[i].kind == RK_IMG || s->b[i].kind == RK_IMG_UBO) { s->storage_images = 1; break; }
    }
    if (s->storage_images != 1) return;
    for (int i = 0; i < s->nbind; i++) {
        const dmsv_bind *b = &s->b[i]; uint64_t h;
        if (b->kind == RK_IMG) h = b->reg >= 0 && b->reg < 32 ? S.img[st][b->reg] : 0;
        else if (b->kind == RK_IMG_UBO) h = mem_handle(ubo_word(st, b->reg, b->off));
        else continue;
        const tex_t *t = NULL; const view_t *view = NULL; int tid = (int)(h & 0xfffff);
        if (tid && vk_tex_by_id(tid, &view, &t) && t) { tex_t image = *t; int ok = vk_tex_storage_prepare(&image, view); t = &image;
            image.d.flags |= NVN_TEXTURE_FLAGS_IMAGE; if (ok) mtl_tex_gpu_written(&image);   // only successful storage bindings write its storage (as nvn_mtl.m)
            static int nl; if ((!ok || t->d.format == 0x3D + 3 || t->d.format == 0x3D + 2) && nl++ < 8) VKLOG("storage image preflight: stage %d shader %016llx binding %u texture ID %d serial %llu %ux%u nvn format %#x: %s", st, (unsigned long long)s->key, b->binding, tid, (unsigned long long)t->serial, t->d.width, t->d.height, t->d.format, ok ? "storage usage ready" : "[RED] no image stores"); }
    }
}
// Reuse within a submission (the cache is emptied when it begins, vk_exec_reset): descriptor sets by their whole content (layout, and every buffer range,
// view, sampler and texel buffer view written), and driver segments by their bytes. Unity draws many times with the same bindings, and every new set
// is an allocation and a vkUpdateDescriptorSets; the driver segment's copy is shared so that a set using it can match.
enum { SETC = 8192, DRVC = 1024 };
static struct { uint64_t h, epoch; VkDescriptorSetLayout l; VkDescriptorSet set; } g_setc[SETC];
// Equality bytes are kept in ordinary CPU memory: mapped scratch may be uncached (upstream nx2android f8d0afaba). n: the live prefix drv_segment writes
// (at most c2's 0x968; bytes past it are zero); a compact entry (ours: graphics c0 by its key fields, g_drvk) keeps none. GPU-written segments stay outside this cache
static struct { uint64_t h, epoch; VkBuffer b; VkDeviceSize o; size_t n; int compact; uint8_t data[0x968]; } g_drvc[DRVC];
uint64_t g_set_hits, g_set_new, g_ns_set_build, g_ns_set_hash, g_ns_set_new;   // descriptor sets: reused, new; time collecting their descriptors, hashing them, allocating and writing new ones
// DM_VK_SET_PROFILE=1: collect 64 complete presentation windows, then print the 32 costliest stage/shader pairs and turn off.
// Existing collection/hash/new-set timers are reused. No per-binding clocks, guest-data reads or descriptor behavior changes.
enum { SET_PROFILE_ROWS = 512, SET_PROFILE_FRAMES = 64 };
typedef struct {
    uint64_t key, collected, memo, eligible, hits, fresh, collect_ns, hash_ns, new_ns;
    int stage, bindings, dynamic, used;
} set_profile_t;
static set_profile_t g_set_profile[SET_PROFILE_ROWS], g_set_profile_overflow;
static uint64_t g_set_profile_first, g_set_profile_end;
static set_profile_t *set_profile_row(int stage, const vkshader_t *s) {
    if (!g_set_profile_end || g_tex_frame < g_set_profile_first || g_tex_frame >= g_set_profile_end) return NULL;
    unsigned pos = (unsigned)(s->key ^ (s->key >> 32) ^ (uint64_t)(unsigned)stage * 0x9e3779b9u) % SET_PROFILE_ROWS;
    for (unsigned i = 0; i < SET_PROFILE_ROWS; i++, pos = (pos + 1) % SET_PROFILE_ROWS) {
        set_profile_t *r = &g_set_profile[pos];
        if (!r->used) { r->used = 1; r->key = s->key; r->stage = stage; r->bindings = s->nbind; r->dynamic = s->ndyn; return r; }
        if (r->key == s->key && r->stage == stage) return r;
    }
    return &g_set_profile_overflow;
}
static uint64_t set_profile_cost(const set_profile_t *r) { return r->collect_ns + r->hash_ns + r->new_ns; }
static void set_profile_report(uint64_t frames, int partial) {
    if (!frames) { VKLOG("descriptor profile cancelled before its first frame"); return; }
    set_profile_t total = g_set_profile_overflow; unsigned rows = 0;
    for (unsigned i = 0; i < SET_PROFILE_ROWS; i++) {
        const set_profile_t *r = &g_set_profile[i]; if (!r->used) continue; rows++;
        total.collected += r->collected; total.memo += r->memo; total.eligible += r->eligible; total.hits += r->hits; total.fresh += r->fresh;
        total.collect_ns += r->collect_ns; total.hash_ns += r->hash_ns; total.new_ns += r->new_ns;
    }
    double per_ms = (double)frames * 1e6, per = (double)frames;
    VKLOG("descriptor profile: %llu %sframes from present %llu, %u stage/key pairs; per frame collected %.1f memo %.1f cache-hit %.1f new %.1f; collect %.3f hash %.3f allocate+update %.3f ms",
        (unsigned long long)frames, partial ? "partial-window " : "", (unsigned long long)g_set_profile_first, rows,
        total.collected / per, total.memo / per, total.hits / per, total.fresh / per,
        total.collect_ns / per_ms, total.hash_ns / per_ms, total.new_ns / per_ms);
    uint8_t shown[SET_PROFILE_ROWS] = {0};
    for (unsigned rank = 0; rank < 32 && rank < rows; rank++) {
        int best = -1;
        for (unsigned i = 0; i < SET_PROFILE_ROWS; i++) if (g_set_profile[i].used && !shown[i] &&
            (best < 0 || set_profile_cost(&g_set_profile[i]) > set_profile_cost(&g_set_profile[best]))) best = (int)i;
        if (best < 0) break;
        shown[best] = 1; const set_profile_t *r = &g_set_profile[best];
        VKLOG("descriptor profile stage %d key %016llx bindings %d dynamic %d: per frame collected %.1f memo %.1f eligible %.1f cache-hit %.1f new %.1f; collect %.3f hash %.3f allocate+update %.3f ms",
            r->stage, (unsigned long long)r->key, r->bindings, r->dynamic,
            r->collected / per, r->memo / per, r->eligible / per, r->hits / per, r->fresh / per,
            r->collect_ns / per_ms, r->hash_ns / per_ms, r->new_ns / per_ms);
    }
    if (g_set_profile_overflow.collected || g_set_profile_overflow.memo)
        VKLOG("descriptor profile overflow: %llu collected, %llu memo calls beyond %u keys; %.3f ms/frame included in total",
            (unsigned long long)g_set_profile_overflow.collected, (unsigned long long)g_set_profile_overflow.memo,
            SET_PROFILE_ROWS, set_profile_cost(&g_set_profile_overflow) / per_ms);
}
static void set_profile_poll(void) {
    const char *e = getenv("DM_VK_SET_PROFILE"); int requested = e && !strcmp(e, "1");
    if (g_set_profile_end) {
        int done = g_tex_frame >= g_set_profile_end;
        if (done || !requested) {
            uint64_t frames = done ? SET_PROFILE_FRAMES : g_tex_frame < g_set_profile_first ? 0 : g_tex_frame - g_set_profile_first + 1;
            set_profile_report(frames, !done); g_set_profile_end = 0; setenv("DM_VK_SET_PROFILE", "0", 1);
        }
        return;
    }
    if (!requested) return;
    if (g_tex_frame > UINT64_MAX - SET_PROFILE_FRAMES - 1) { setenv("DM_VK_SET_PROFILE", "0", 1); return; }
    memset(g_set_profile, 0, sizeof g_set_profile); memset(&g_set_profile_overflow, 0, sizeof g_set_profile_overflow);
    g_set_profile_first = g_tex_frame + 1; g_set_profile_end = g_set_profile_first + SET_PROFILE_FRAMES;
    VKLOG("descriptor profile armed: presents %llu..%llu (64 frames; auto-off)",
        (unsigned long long)g_set_profile_first, (unsigned long long)(g_set_profile_end - 1));
}
// content hash of descriptor sets and driver segments: 8 bytes a step (a set with 32-slot texture arrays is several KB, which byte-wise FNV took milliseconds a frame over)
static inline uint64_t fnv(uint64_t h, const void *p, size_t n) {
    const uint8_t *b = p; size_t i = 0;
    for (; i + 8 <= n; i += 8) { uint64_t w; memcpy(&w, b + i, 8); h = (h ^ w) * 0x9E3779B97F4A7C15ull; h ^= h >> 29; }
    for (; i < n; i++) { h ^= b[i]; h *= 0x100000001b3ull; }
    return h;
}
// Graphics c0 contains only one stage's 16 address/uint32-size entries, fixed point-size data and base instance.
// Identify those exact bytes before constructing the 4 KB segment. Compute c0 and all other banks keep their full-byte key.
_Static_assert(DMSH_C0_SSBO_MS + 16 * DMSH_C0_SSBO_SLOTS <= 0x968, "g_drvc data holds bank 0's live prefix");
typedef struct { uint64_t address[16]; uint32_t size[16], base_instance, table; } drv_gkey;
static drv_gkey g_drvk[DRVC];   // the compact entries' key fields, compared on a hit as the full path compares the bytes (upstream nx2android 88f75952c)
static uint64_t drv_graphics_key(int st, drv_gkey *kp) {
    drv_gkey key; memset(&key, 0, sizeof key);
    key.table = st == 0 ? DMSH_C0_SSBO_VS : st == 1 ? DMSH_C0_SSBO_PS : DMSH_C0_SSBO_GS;
    key.base_instance = g_draw_binst;
    for (int r = 0; r < 16; r++) { key.address[r] = S.ssbo[st][r].addr; key.size[r] = (uint32_t)S.ssbo[st][r].size; }
    *kp = key; return fnv(0x6772617068696330ull, &key, sizeof key); // separate domain from the full-byte bank hashes
}
static void drv_shared(int st, int bank, VkBuffer *buf, VkDeviceSize *off, int dyn) {   // one copy per distinct content in the submission
    uint8_t d[0x1000]; int compact = g_drv_compact && bank == 0 && st >= 0 && st < 3;
    uint64_t h; drv_gkey gk; size_t n = 0;
    if (compact) h = drv_graphics_key(st, &gk);
    else { drv_segment(st, bank, d, dyn); n = bank == 24 ? 0x100 : bank ? 0x968 : DMSH_C0_SSBO_MS + 16 * DMSH_C0_SSBO_SLOTS;
        h = fnv(0xcbf29ce484222325ull, d, n) ^ (uint64_t)bank << 56; }   // (bank 0 through the mesh stage table, the last thing drv_segment writes: at 0x648 the task and mesh kernels, whose tables are at 0x670 / 0x770, all hashed alike and shared the first one built)
    unsigned k = (unsigned)(h % DRVC);
    // a hash match is checked against the content: the key fields (compact) or the live bytes' CPU copy, over the longer of the two prefixes (a longer tail is not the shorter one's zeros)
    if (g_drvc[k].epoch == g_set_epoch && g_drvc[k].h == h && !g_x_nodrvc && g_drvc[k].compact == compact &&
        (compact ? !memcmp(&g_drvk[k], &gk, sizeof gk) : !memcmp(g_drvc[k].data, d, n > g_drvc[k].n ? n : g_drvc[k].n))) { *buf = g_drvc[k].b; *off = g_drvc[k].o; return; }
    if (compact) drv_segment(st, bank, d, dyn);
    uint8_t *p = vk_scratch(0x1000, V.props.limits.minUniformBufferOffsetAlignment, buf, off); memcpy(p, d, sizeof d);
    g_drvc[k].h = h; g_drvc[k].epoch = g_set_epoch; g_drvc[k].b = *buf; g_drvc[k].o = *off; g_drvc[k].n = n; g_drvc[k].compact = compact;
    if (compact) g_drvk[k] = gk; else { memcpy(g_drvc[k].data, d, n); memset(g_drvc[k].data + n, 0, sizeof g_drvc[k].data - n); }
}
// A binding that runs past its chunk's end into the pool's next chunk (pools over ZC_CHUNK come in chunks, each its own buffer, and the replay looks a binding up by its first byte):
// the GPU would read past the buffer, and with robustBufferAccess off it saw zeros. Found with smaller chunks in a fork: Pokemon Pokopia's per-frame constants, a 64 MB ring, crossed
// a 16 MB chunk's end about 2.5 times a second on an AYN Thor, one frame of black objects with a white bloom halo or a model's bones collapsed each time a draw read past it (7 in 3
// minutes at one spot, 0 with this). A constant buffer is bound from a copy of the whole range out of CPU memory (vk_ubd_read, as a deferred update's shadow); a storage or index
// range is memory the GPU may write, so it is counted and named, not copied
extern uint64_t g_ubd_gen;
static uint64_t g_cross[6];   // ranges across a chunk's end: constant buffers copied, storage buffers, vertex buffers (their bound range), index buffers (the draw's indices), storage and index ranges bound from a copy
static void vk_cross(int k, uint64_t a, uint64_t n, const vkpool_t *p) {
    // vertex buffers: the stream's bound range, which a draw may read only part of (Pokopia, 16 MB chunks: 32,768 such bindings in 3 minutes of ~1 MB ranges, no wrong frame)
    static const char *const what[6] = {"constant buffer ranges (bound from a copy)", "storage buffer ranges the shader writes, or over 16 MB (clipped to the chunk)", "vertex buffer bound ranges (a draw may read less)", "index ranges over 16 MB or of indirect draws (clipped)",
                                        "read-only storage buffer ranges (bound from a CPU copy: GPU writes to them since the last copy-back are not seen)", "index ranges (bound from a CPU copy)"};
    uint64_t c = ++g_cross[k]; if (!(c & (c - 1))) VKLOG("%s%llu %s across a memory pool chunk's end (this one %#llx +%#llx, the chunk ends at %#llx)", k == 1 || k == 3 ? "[RED] " : "", (unsigned long long)c, what[k],
                                                           (unsigned long long)a, (unsigned long long)n, (unsigned long long)p->hi);
}
// a constant buffer binding's descriptor (stage_set, set_memo_get): its pool's buffer from the alignment boundary below it, or the shadow of the deferred updates over it
static void cbuf_desc(int st, const vkshader_t *s, const dmsv_bind *b, VkDescriptorBufferInfo *bi) {
    const p_bind_buf *u = b->reg >= 0 && b->reg < 16 ? &S.ubo[st][b->reg] : NULL;
    uint64_t n = u && u->size ? (u->size > 65536 ? 65536 : u->size) : 65536; VkBuffer buf; VkDeviceSize off;
    vk_pool_tag = "uniform"; vkpool_t *p = g_var_now != 9 && u && u->addr ? vk_pool_of(u->addr, 1) : NULL;
    if (!p) { *bi = (VkDescriptorBufferInfo){vk_zero_buffer(), 0, 65536}; return; }
    buf = p->buf; off = u->addr - p->lo;
    if (off + n > p->hi - p->lo) {
        if (p->hi < p->end && u->addr + n <= p->end) {   // into the pool's next chunk (vk_cross): the whole range from CPU memory, one copy per submission
            static struct { uint64_t a, n, ep, gen; VkBuffer b; VkDeviceSize o; } cc[16]; unsigned k = (unsigned)(u->addr >> 8) % 16;
            if (cc[k].a != u->addr || cc[k].n != n || cc[k].ep != g_set_epoch || cc[k].gen != g_ubd_gen) {
                uint8_t *d = vk_scratch(n, V.props.limits.minUniformBufferOffsetAlignment, &buf, &off); vk_ubd_read(u->addr, n, d); vk_cross(0, u->addr, n, p);
                cc[k].a = u->addr; cc[k].n = n; cc[k].ep = g_set_epoch; cc[k].gen = g_ubd_gen; cc[k].b = buf; cc[k].o = off;
            } else { buf = cc[k].b; off = cc[k].o; }
            ubo_log(s, st, b->reg, u->addr, n); *bi = (VkDescriptorBufferInfo){buf, off, n}; return;
        }
        n = p->hi - p->lo - off;
    }
    ubo_log(s, st, b->reg, u->addr, n);
    // Only a copied shadow is shortened, after the guest/pool range checks above; direct descriptors and cross-pool copies keep their range.
    // A deferred update entirely outside the proven reads needs no shadow. The smaller scratch buffer gets the matching smaller descriptor range.
    // Compute's alternative module is latched at startup, so this first scope is graphics, whose alternative flag is part of its pipeline key.
    uint64_t shadow_n = n;
    if (g_ubo_bounds && st >= 0 && st < 3 && !g_spv_alt && u && s->ubo_extent[b->reg] && s->ubo_extent[b->reg] < shadow_n) shadow_n = s->ubo_extent[b->reg];
    if (vk_ubd_shadow(u->addr, shadow_n, &buf, &off)) { *bi = (VkDescriptorBufferInfo){buf, off, shadow_n}; return; }
    VkDeviceSize al = V.props.limits.minUniformBufferOffsetAlignment; if (off % al) VKRED_ONCE(40, "uniform buffer at an offset not aligned to %llu", (unsigned long long)al);
    *bi = (VkDescriptorBufferInfo){buf, p->boff + off - off % al, n + off % al};   // (off is from the pool's start, its buffer starts boff before it)
}
// Shared by collection and memo refresh, including range clipping and compute dirty tracking.
static void sbuf_desc(int st, const dmsv_bind *b, VkDescriptorBufferInfo *bi) {
    VkBuffer buf; VkDeviceSize off;
    uint64_t a = 0, n = 0;
    if (b->kind == RK_SBUF && b->reg >= 0 && b->reg < 16) { a = S.ssbo[st][b->reg].addr; n = S.ssbo[st][b->reg].size; }
    else if (b->kind == RK_UBO_PTR && b->reg >= 0 && b->reg < 16) { a = S.ubo[st][b->reg].addr; n = S.ubo[st][b->reg].size; }
    else if (b->kind == RK_SBUF_UBO) { a = (uint64_t)ubo_word(st, b->reg, b->off) | (uint64_t)ubo_word(st, b->reg, b->off + 4) << 32; }
    if (g_fs_buffer_collect || g_cs_buffer_collect) g_trace_sbuf_guest = a;
    vk_pool_tag = "storage"; vkpool_t *p = a ? vk_pool_of(a, 1) : NULL;
    if (p) { buf = p->buf; off = a - p->lo;
        if (n && off + n > p->hi - p->lo && p->hi < p->end && a + n <= p->end) {   // into the pool's next chunk: a range this shader only reads, up to 16 MB, is bound from a copy of CPU memory (as cbuf_desc; one a submission), as the translator's 256-byte boundary below it
            if (!(b->flags & 4) && n <= (16u << 20)) {
                static struct { uint64_t a, n, ep, gen; VkBuffer b; VkDeviceSize o; } sc[16]; unsigned k = (unsigned)(a >> 8) % 16; uint64_t a0 = a - a % SB_ALIGN, n0 = n + a % SB_ALIGN;
                if (sc[k].a != a || sc[k].n != n || sc[k].ep != g_set_epoch || sc[k].gen != g_ubd_gen) { uint8_t *d = vk_scratch(n0, SB_ALIGN, &buf, &off); vk_ubd_read(a0, n0, d); vk_cross(4, a, n, p);
                    sc[k].a = a; sc[k].n = n; sc[k].ep = g_set_epoch; sc[k].gen = g_ubd_gen; sc[k].b = buf; sc[k].o = off; } else { buf = sc[k].b; off = sc[k].o; }
                *bi = (VkDescriptorBufferInfo){buf, off, n0}; return;
            }
            vk_cross(1, a, n, p);
        }
        if (!n || off + n > p->hi - p->lo) n = p->hi - p->lo - off;
        // bound from the 256-byte boundary below it, the boundary the translator offsets every access from (shader_vk.py SB_ALIGN; a multiple of any device's minimum alignment)
        const VkDeviceSize al = SB_ALIGN; if (V.props.limits.minStorageBufferOffsetAlignment > al) VKRED_ONCE(41, "storage buffer alignment %llu above %llu", (unsigned long long)V.props.limits.minStorageBufferOffsetAlignment, (unsigned long long)al);
        *bi = (VkDescriptorBufferInfo){buf, p->boff + off - off % al, n + off % al};   // (off is from the pool's start, its buffer starts boff before it)
        if (st == 5 && b->kind != RK_UBO_PTR) { void vk_gpu_wrote(uint64_t a, uint64_t n); vk_gpu_wrote(a, n); }   // a dispatch's storage buffers: taken as written
    } else *bi = (VkDescriptorBufferInfo){vk_zero_buffer(), 0, 65536};
}
static uint32_t storage_dynamic_offset(VkDescriptorBufferInfo *bi, VkDeviceSize alignment) {
    if (!alignment || bi->offset > UINT32_MAX || bi->offset % alignment || bi->range == VK_WHOLE_SIZE) return 0;
    uint32_t offset = (uint32_t)bi->offset; bi->offset = 0; return offset;
}
// The set a stage's last draw got, and what it was built from. Between two draws most stages change no texture, sampler, image or storage buffer, only
// the offsets of their constant buffers, which go to the bind (dynamic uniform buffers). Then nothing is collected or hashed: the constant buffers alone are
// looked at again, and their buffer and range must be the same (the offset too for one that is not dynamic). It holds within one submission (the sets and
// the driver segments' copies are its own) while the stage's shader and bindings (g_bind_gen, g_ubo_gen; only real changes count), the texture IDs (g_tex_gen),
// sampler IDs (g_samp_gen), memory pools (vk_pool_gen) and the base instance (the driver segment's c0 + 0x644) are unchanged. Images keep the last_use the
// first build gave them: the fence it names is the same for the whole submission. Not for compute (its driver segment holds the dispatch size and its storage
// buffers are marked written), nor for a shader that reads handles or addresses from uniform buffer memory (that memory changes under the same bindings).
// DM_VK_X_NOSETMEMO=1 (live): always collected. DM_VK_X_SETVERIFY=1 (live): one hit in 64 is collected anyway and compared, [RED] on a difference
enum { MEMO_CB = 48 }; // bounded CBUF/direct-SSBO/c0/claim dependencies
typedef struct { vkshader_t *s; uint64_t epoch, bind, ssbo, ubo, ubd, tex, samp, h, um, im; uint32_t pool, binst, dofs[96]; int ncb; uint8_t ci[MEMO_CB], cpos[MEMO_CB]; VkDescriptorBufferInfo cd[MEMO_CB]; VkDescriptorSet set; } set_memo_t;
static set_memo_t g_smemo[3]; static uint64_t g_bind_gen[6], g_ssbo_gen[6], g_ubo_gen[6];   // separate handle, direct-storage and uniform binding generations
uint64_t g_set_memo, g_set_memo_bad;   // memo hits, and differences the verify mode found
extern uint64_t g_ubd_gen; uint32_t vk_pool_gen(void);
// A cached shader may be rebound after switching away. Invalidate its dependency, not only the shader currently bound.
// Compute has no set memo; unchanged bytes and stages without the graphics push bank keep their descriptors.
static void push_constants(const p_push_const *r) {
    if (r->compute >= 2 || r->offset > sizeof S.push[0] || r->size > sizeof S.push[0] - r->offset) return;
    uint8_t *d = S.push[r->compute] + r->offset;
    if (!r->size || !memcmp(d, r->data, r->size)) return;
    memcpy(d, r->data, r->size);
    if (r->compute) return;
    for (int st = 0; st < 3; st++) {
        const vkshader_t *s = g_smemo[st].s;
        if (s) for (int i = 0; i < s->nbind; i++)
            if (s->b[i].kind == RK_DRV && s->b[i].reg == DMSH_PUSH_BANK) { g_smemo[st].s = NULL; break; }
    }
}
static int memo_buffer(const dmsv_bind *b) {
    return b->kind == RK_CBUF || b->kind == RK_SBUF || b->kind == RK_GS_CLAIM || (b->kind == RK_DRV && b->reg == 0);
}
static int set_memo_ok(vkshader_t *s) {
    if (!s->setmemo) { int ok = 1, ncb = 0, claim = 0;
        for (int i = 0; i < s->nbind; i++) { const dmsv_bind *b = &s->b[i]; ncb += memo_buffer(b); claim |= b->kind == RK_GS_CLAIM;
            ok &= b->kind != RK_TEXD && b->kind != RK_SAMPD && b->kind != RK_SBUF_UBO && b->kind != RK_UBO_PTR && (b->kind != RK_TEX_UBO || s->tbl[i]) && b->kind != RK_IMG_UBO && b->kind != RK_TBUF_UBO && b->samp != RS_UBO && i < 256;   // (a texture table binding is in the table set, tbl_set, not this one)
            ok &= (b->kind != RK_TEXP && b->kind != RK_SAMPP) || s->tbl[i];   // (a pool array is in the table set too: vk_shader draws no shader whose pool arrays are not)
            ok &= !(b->kind >= RK_MESH_CTL && b->kind <= RK_MESH_IDX); }   // (the mesh emulation's buffers change every draw: mesh_draw)
        s->setmemo = ok && ncb <= MEMO_CB ? (claim ? 3 : 1) : 2; }
    return (s->setmemo & 1) && (s->setmemo != 3 || g_buffer_memo);
}
static int set_memo_get(int st, vkshader_t *s, uint32_t *dofs) {
    set_memo_t *m = &g_smemo[st];
    if (m->s != s || m->epoch != g_set_epoch || m->bind != g_bind_gen[st] || m->tex != g_tex_gen || m->samp != __atomic_load_n(&g_samp_gen, __ATOMIC_ACQUIRE) || m->pool != vk_pool_gen() || m->binst != g_draw_binst) return 0;
    int cb_changed = m->ubo != g_ubo_gen[st] || m->ubd != g_ubd_gen, sb_changed = m->ssbo != g_ssbo_gen[st];
    if (sb_changed && !g_buffer_memo) return 0; // original conservative invalidation for controlled A/B
    if (cb_changed || sb_changed || s->setmemo == 3) {
        for (int c = 0; c < m->ncb; c++) {
            const dmsv_bind *b = &s->b[m->ci[c]]; VkDescriptorBufferInfo d;
            if (b->kind == RK_CBUF && cb_changed) cbuf_desc(st, s, b, &d);
            else if (b->kind == RK_SBUF && sb_changed) sbuf_desc(st, b, &d);
            else if (b->kind == RK_DRV && b->reg == 0 && sb_changed) {
                VkBuffer buf; VkDeviceSize off; drv_shared(st, 0, &buf, &off, 0); d = (VkDescriptorBufferInfo){buf, off, 0x1000};
            } else if (b->kind == RK_GS_CLAIM) d = g_draw_claim.buffer ? g_draw_claim : (VkDescriptorBufferInfo){vk_zero_buffer(), 0, 4};
            else continue;
            uint32_t offset = 0;
            if (s->dyn[m->ci[c]] == 2) offset = storage_dynamic_offset(&d, V.props.limits.minStorageBufferOffsetAlignment);
            else if (s->dyn[m->ci[c]]) { offset = (uint32_t)d.offset; d.offset = 0; } // same dynamic-uniform conversion as collection
            if (d.buffer != m->cd[c].buffer || d.range != m->cd[c].range || d.offset != m->cd[c].offset) return 0;
            if (m->cpos[c] != 0xff) m->dofs[m->cpos[c]] = offset;
        }
        m->ubo = g_ubo_gen[st]; m->ubd = g_ubd_gen; m->ssbo = g_ssbo_gen[st];
    }
    memcpy(dofs, m->dofs, sizeof *dofs * (size_t)s->ndyn); g_num_um = m->um; g_num_im = m->im;
    return 1;
}
// Keep the exact effective address and range. A large offset or unexpected alignment remains in the descriptor with dynamic offset zero,
// so an opt-in dynamic-storage layout remains valid even when this individual binding cannot benefit from offset-independent reuse.
// dofs (at least s->ndyn entries): the dynamic offsets of the set's dynamic uniform and storage buffers, in binding order as vkCmdBindDescriptorSets takes them
// indirect_c0 (compute only): a fresh segment for this dispatch's GPU-written counts; never put mutable bytes in drv_shared's content cache (upstream nx2android #206)
static VkDescriptorSet stage_set(int st, vkshader_t *s, uint32_t *dofs, VkDescriptorBufferInfo *indirect_c0) {
    set_profile_t *sp = set_profile_row(st, s);
    // Check before descriptor memo hits: storage/unknown descriptors can write beyond tracked transfer ranges.
    if (vk_readback_ranges()) for (int i = 0; i < s->nbind; i++) {
        int d = s->b[i].desc;
        int k = s->b[i].kind;
        if (k >= RK_MESH_CTL && k <= RK_MESH_IDX) continue;   // the mesh emulation's own buffers (mesh_draw), not pool memory
        if (k == RK_GS_CLAIM) continue;   // host scratch only, never guest pool memory
        if (k == RK_SBUF || k == RK_SBUF_UBO || k == RK_IMG || k == RK_IMG_UBO || k == RK_IMG_TBUF || (k == RK_GMEM_MAP && (s->b[i].flags & 4))) { vk_readback_unknown(); break; }   // (a raw global memory store lands in any pool)
        if (d != RD_UBO && d != RD_SAMPLED && d != RD_TBUF && d != RD_IMAGE && d != RD_SAMPLER) { vk_readback_unknown(); break; }
    }
    int memo = st < 3 && !(g_mrt_active && st == 1) && !g_fs_buffer_collect && !g_x_nosetmemo && !g_tracing && !g_var_now && !g_x_nosetc && !g_x_nodrvc && !g_ubolog_key && g_td_off < 0 && set_memo_ok(s), verify = 0;
    if (sp) sp->eligible += memo != 0;
    uint32_t vd[96]; uint64_t vum = 0, vim = 0;   // verify: what the memo gave
    if (memo && set_memo_get(st, s, dofs)) { static unsigned nv; if (!g_x_setverify || ++nv % 64) { g_set_memo++; if (sp) sp->memo++; return g_smemo[st].set; }
        verify = 1; memcpy(vd, dofs, sizeof *dofs * (size_t)s->ndyn); vum = g_num_um; vim = g_num_im; }
    uint64_t gtex = g_tex_gen, gsamp = __atomic_load_n(&g_samp_gen, __ATOMIC_ACQUIRE); uint32_t gpool = vk_pool_gen();   // before collecting: a change racing it (game thread) leaves the memo outdated
    VkDescriptorSet set = VK_NULL_HANDLE; struct { uint32_t b, o; int i; } dp[96]; int nd = 0; uint64_t tb0 = tick();
    VkWriteDescriptorSet w[96]; VkDescriptorBufferInfo bi[96]; VkDescriptorImageInfo ii[96 * 4]; VkBufferView bv[96]; uint32_t nw = 0, nii = 0;
    int ncb = 0; uint8_t mci[MEMO_CB]; VkDescriptorBufferInfo mcd[MEMO_CB];   // normalized buffer dependencies, for the memo
    g_num_um = g_num_im = 0;
    int dynslots = 0;
    if (DM_GAME_TEX_DYN_SLOTS) for (int j = 0; j < s->nbind; j++)
        if (s->b[j].kind == RK_TEXD || s->b[j].kind == RK_SAMPD) dynslots = 1;
    for (int i = 0; i < s->nbind && nw < 96; i++) {
        const dmsv_bind *b = &s->b[i]; if (s->tbl[i]) continue;   // in the table set (tbl_set)
        VkWriteDescriptorSet *wr = &w[nw]; memset(wr, 0, sizeof *wr);
        wr->sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; wr->dstSet = set; wr->dstBinding = b->binding; wr->descriptorCount = 1;
        VkBuffer buf; VkDeviceSize off, len;
        switch (b->kind) {
        case RK_GS_CLAIM: bi[nw] = g_draw_claim.buffer ? g_draw_claim : (VkDescriptorBufferInfo){vk_zero_buffer(), 0, 4}; break;
        case RK_SUPPORT: light_diag_support(st, s, &bi[nw]); mrt_support(st, s, &bi[nw]); break;
        case RK_C1: vk_shader_c1(s, &buf, &off, &len); bi[nw] = (VkDescriptorBufferInfo){buf, off, len}; break;
        case RK_CBUF: cbuf_desc(st, s, b, &bi[nw]); break;
        case RK_DRV: {
            if (indirect_c0 && st == 5 && b->reg == 0) {
                if (!indirect_c0->buffer) {   // several bindings of c0 in one shader share this dispatch's segment
                    uint8_t *d = vk_scratch(0x1000, V.props.limits.minUniformBufferOffsetAlignment, &indirect_c0->buffer, &indirect_c0->offset);
                    drv_segment(st, 0, d, dynslots); indirect_c0->range = 0x1000;
                }
                bi[nw] = *indirect_c0;   // retain the real offset before dynamic descriptors move it into dofs
            } else { VkBuffer sb; VkDeviceSize so; drv_shared(st, b->reg, &sb, &so, dynslots); bi[nw] = (VkDescriptorBufferInfo){sb, so, 0x1000}; }
            break; }
        // Raw stores use the pool table and, on Metal, log the pool they wrote. Android does not consume that log and keeps conservative readback tracking above.
        // GlobalToStorage skips AtomicOr when the pool's bit is set; an immutable all-set bitmap avoids that unused atomic and per-set allocation/clear.
        // Real stores and atomics run after the log guard, unchanged. DM_VK_GMEM_LOG=1 restores the cleared logs for comparison.
        case RK_GMEM_MAP: { VkDeviceSize n; VkBuffer gb = vk_gmem_map(&n); bi[nw] = (VkDescriptorBufferInfo){gb, 0, n}; break; }
        case RK_GMEM_LOG: {
            VkDeviceSize n = DMSH_GMEM_MAP_ENTRIES / 8;
            if (g_gmem_log) { VkBuffer lb; VkDeviceSize lo; uint8_t *z = vk_scratch(n, SB_ALIGN, &lb, &lo); memset(z, 0, n); bi[nw] = (VkDescriptorBufferInfo){lb, lo, n}; }
            else bi[nw] = (VkDescriptorBufferInfo){vk_gmem_log_disabled(), 0, n};
            break;
        }
        case RK_MESH_CTL: case RK_MESH_TASKOUT: case RK_MESH_LAUNCH: case RK_MESH_VTX: case RK_MESH_IDX: {   // the mesh emulation's buffers of the draw being replayed (mesh_draw sets g_mesh before each kernel)
            const mesh_buf *mb = b->kind == RK_MESH_CTL ? &g_mesh.ctl : b->kind == RK_MESH_TASKOUT ? &g_mesh.taskout : b->kind == RK_MESH_LAUNCH ? &g_mesh.launch : b->kind == RK_MESH_VTX ? &g_mesh.vtx : &g_mesh.idx;
            bi[nw] = mb->b ? (VkDescriptorBufferInfo){mb->b, mb->off, mb->len} : (VkDescriptorBufferInfo){vk_zero_buffer(), 0, 65536}; break; }
        case RK_SBUF: case RK_UBO_PTR: case RK_SBUF_UBO: sbuf_desc(st, b, &bi[nw]); break;
        case RK_TEX: case RK_TEX_UBO: case RK_TEX_C2A: {
            int cnt = b->count ? b->count : 32; if (nii + (uint32_t)cnt > sizeof ii / sizeof *ii) cnt = 1;
            for (int k = 0; k < cnt; k++) {
                uint64_t h = b->kind == RK_TEX ? vk_tex_slot_handle(S.tex[st], S.img[st], (unsigned)b->reg) :
                             b->kind == RK_TEX_UBO ? mem_handle(ubo_word(st, b->reg, b->off + 8 * k)) : (b->off >= 0 ? c2_handle(st, b->off + 8 * k) : 0);
                g_tex_sseq = b->kind == RK_TEX && b->reg >= 0 && b->reg < 256 ? S.tex_sseq[st][b->reg] : 0;
                int num = tex_desc_for(st, b, h, &ii[nii + k], 0);
                fs_trace_texture_note(st, b, h, &ii[nii + k], g_desc_im);
                if (num && i < 64) { if (num == 1) g_num_um |= 1ull << i; else g_num_im |= 1ull << i; }
            }
            wr->pImageInfo = &ii[nii]; nii += (uint32_t)cnt; wr->descriptorCount = (uint32_t)cnt; wr->descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; nw++;
            continue; }
        case RK_TEXD: case RK_SAMPD: {
            int cnt = b->count ? b->count : 32;
            if (cnt > 32 || nii + (uint32_t)cnt > sizeof ii / sizeof *ii) die("dynamic texture array exceeds capacity");
            for (int k = 0; k < cnt; k++) {
                VkDescriptorImageInfo *di = &ii[nii + k]; memset(di, 0, sizeof *di);
                if (b->kind == RK_SAMPD) {
                    di->sampler = samp_dynamic_slot(st, k, dynslots);
                    if (!di->sampler) di->sampler = vk_dummy_sampler();
                } else {
                    uint64_t h = dynslots ? (S.tex[st][k] ? S.tex[st][k] : S.tex[st][128+k]) : 0;
                    const view_t *v = NULL; const tex_t *t = NULL;
                    int tid = (int)(h & 0xfffff), fits = 0;
                    if (tid && vk_tex_by_id(tid, &v, &t) && t) {
                        int target = v && v->set_target ? v->target : t->d.target;
                        fits = b->dim == 3 ? (t->d.width == t->d.height && (target == 8 || target == 9 || (target == 4 && t->d.depth >= 6))) :
                               b->dim == 2 ? target == 2 : b->dim == 1 ? (target == 1 || target == 4 || target == 8 || target == 9) : (target == 0 || target == 3);
                        if (view_num(t, v)) fits = 0; // these arrays declare floating-point texels
                    }
                    dmsv_bind vb = *b; vb.samp = RS_TEX;
                    tex_desc_for(st, &vb, fits ? h : 0, di, 0);
                    di->sampler = VK_NULL_HANDLE; // separate sampled image, not combined
                }
            }
            if (dynslots) { static int noted; if (!noted++) VKLOG("dynamic texture slot arrays enabled (32 entries)"); }
            wr->pImageInfo = &ii[nii]; nii += (uint32_t)cnt; wr->descriptorCount = (uint32_t)cnt;
            wr->descriptorType = b->kind == RK_TEXD ? VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE : VK_DESCRIPTOR_TYPE_SAMPLER;
            nw++; continue;
        }
        case RK_IMG: case RK_IMG_UBO: {
            uint64_t h = b->kind == RK_IMG ? (b->reg >= 0 && b->reg < 32 ? S.img[st][b->reg] : 0) : mem_handle(ubo_word(st, b->reg, b->off));
            int num = tex_desc_for(st, b, h, &ii[nii], 1);
            if (i < 64) { if (num == 1) g_num_um |= 1ull << i; else if (num == 2) g_num_im |= 1ull << i; }   // image reads/stores preserve the registered view's integer words
            wr->pImageInfo = &ii[nii++]; wr->descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE; nw++;
            continue; }
        case RK_TBUF: case RK_TBUF_UBO: case RK_IMG_TBUF: {
            uint64_t h = b->kind == RK_TBUF ? vk_tex_slot_handle(S.tex[st], S.img[st], (unsigned)b->reg) : b->kind == RK_IMG_TBUF ? (b->reg >= 0 && b->reg < 32 ? S.img[st][b->reg] : 0) : mem_handle(ubo_word(st, b->reg, b->off));
            const view_t *v; const tex_t *t = NULL; int tid = (int)(h & 0xfffff);
            bv[nw] = tid && vk_tex_by_id(tid, &v, &t) && t && t->d.target == NVN_TEXTURE_TARGET_BUFFER ? vk_tbuf_view(t, b->kind == RK_IMG_TBUF) : VK_NULL_HANDLE;
            if ((g_tracing && g_x_tracepass >= 0) || (st == 5 && g_x_tracepass == -2)) VKLOG("  texel buffer binding %d (kind %d, cbuf %d +%d): handle %#llx → %s format %#x, %u elements at %#llx%s", b->binding, b->kind, b->reg, b->off,
                (unsigned long long)h, t ? "texture" : "no texture", t ? t->d.format : 0, t ? t->d.width : 0, t ? (unsigned long long)(uintptr_t)t->mem : 0ull, bv[nw] ? "" : " (the dummy: zeros)");
            if (!bv[nw]) bv[nw] = vk_dummy_tbuf(b->kind == RK_IMG_TBUF);
            else if (i < 64) { int num = view_num(t, NULL); if (num == 1) g_num_um |= 1ull << i; else if (num == 2) g_num_im |= 1ull << i; }   // texel buffers read and write their words alike (storage ones too: gfx_vk_shader.c spv_int_images)
            wr->pTexelBufferView = &bv[nw]; wr->descriptorType = b->kind == RK_IMG_TBUF ? VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER : VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER; nw++;
            continue; }
        default: {   // unknown: the descriptor type it was declared with, filled with what reads zero
            if (b->desc == RD_UBO || b->desc == RD_SSBO) bi[nw] = (VkDescriptorBufferInfo){vk_zero_buffer(), 0, 65536};
            else if (b->desc == RD_SAMPLED || b->desc == RD_STORAGE) {
                ii[nii] = (VkDescriptorImageInfo){b->desc == RD_SAMPLED ? vk_dummy_sampler() : VK_NULL_HANDLE, vk_dummy_view(b->dim == 0xff ? 1 : b->dim, b->flags & 1, (b->flags & 2) != 0, b->desc == RD_STORAGE, 0), VK_IMAGE_LAYOUT_GENERAL};
                int cnt = b->count ? b->count : 32; for (int k = 1; k < cnt && nii + k < sizeof ii / sizeof *ii; k++) ii[nii + k] = ii[nii];
                wr->pImageInfo = &ii[nii]; nii += (uint32_t)cnt; wr->descriptorCount = (uint32_t)cnt;
                wr->descriptorType = b->desc == RD_SAMPLED ? VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER : VK_DESCRIPTOR_TYPE_STORAGE_IMAGE; nw++; continue;
            } else if (b->desc == RD_TBUF || b->desc == RD_STORAGE_TBUF) { bv[nw] = vk_dummy_tbuf(b->desc == RD_STORAGE_TBUF); wr->pTexelBufferView = &bv[nw];
                wr->descriptorType = b->desc == RD_TBUF ? VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER : VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER; nw++; continue; }
            else continue;
        }
        }
        if ((st == 5 && g_cs_buffer_collect) || (st < 3 && g_fs_buffer_collect))
            cs_trace_buffer_note(st, s->key, b, &bi[nw]); // actual offset before dynamic normalization
        wr->pBufferInfo = &bi[nw]; wr->descriptorType = b->desc == RD_SSBO ? VK_DESCRIPTOR_TYPE_STORAGE_BUFFER : VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        if (s->dyn[i]) {   // the offset goes to the bind, so the set's content (and its reuse) does not depend on it
            uint32_t offset;
            if (s->dyn[i] == 2) { offset = storage_dynamic_offset(&bi[nw], V.props.limits.minStorageBufferOffsetAlignment); wr->descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC; }
            else { if (bi[nw].offset > UINT32_MAX) VKRED_ONCE(42, "dynamic uniform buffer offset past 4 GB");
                offset = (uint32_t)bi[nw].offset; bi[nw].offset = 0; wr->descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC; }
            int j = nd++; while (j > 0 && dp[j - 1].b > b->binding) { dp[j] = dp[j - 1]; j--; }
            dp[j].b = b->binding; dp[j].o = offset; dp[j].i = i;
        }
        if (memo_buffer(b) && ncb < MEMO_CB) { mci[ncb] = (uint8_t)i; mcd[ncb++] = bi[nw]; }
        nw++;
    }
    for (int j = 0; j < nd; j++) dofs[j] = dp[j].o;
    if (g_tracing || g_up_native_trace) for (int j = 0; j < nd; j++) VKLOG("  stage %d binding %u: dynamic offset %u", st, dp[j].b, dp[j].o);
    if (g_tracing || g_up_native_trace) for (uint32_t i = 0; i < nw; i++) {   // the refused draw (bisection): every descriptor it is given
        const VkWriteDescriptorSet *x = &w[i];
        if (x->pBufferInfo) VKLOG("  stage %d binding %u type %d: buffer %p +%llu size %llu", st, x->dstBinding, x->descriptorType, (void *)x->pBufferInfo->buffer, (unsigned long long)x->pBufferInfo->offset, (unsigned long long)x->pBufferInfo->range);
        else if (x->pImageInfo) for (uint32_t k = 0; k < x->descriptorCount && k < 4; k++) VKLOG("  stage %d binding %u[%u] type %d: view %p sampler %p", st, x->dstBinding, k, x->descriptorType, (void *)x->pImageInfo[k].imageView, (void *)x->pImageInfo[k].sampler);
        else if (x->pTexelBufferView) VKLOG("  stage %d binding %u type %d: texel buffer view %p", st, x->dstBinding, x->descriptorType, (void *)x->pTexelBufferView[0]);
    }
    // the set's content, then an earlier identical set of this submission or a new one
    uint64_t tb1 = tick(); g_ns_set_build += tb1 - tb0; if (sp) { sp->collected++; sp->collect_ns += tb1 - tb0; }
    uint64_t h = fnv(0xcbf29ce484222325ull, &s->dsl, sizeof s->dsl);
    for (uint32_t i = 0; i < nw; i++) {
        const VkWriteDescriptorSet *x = &w[i]; h = fnv(h, &x->dstBinding, 4); h = fnv(h, &x->descriptorType, 4); h = fnv(h, &x->descriptorCount, 4);
        if (x->pBufferInfo) h = fnv(h, x->pBufferInfo, sizeof *x->pBufferInfo);
        else if (x->pImageInfo) for (uint32_t j = 0; j < x->descriptorCount; j++) {
            const VkDescriptorImageInfo *im = &x->pImageInfo[j];
            // Vulkan fields only: imageLayout can be followed by uninitialized C padding.
            // Three whole words retain the fast hash path without depending on those bytes.
            uint64_t fields[] = {(uint64_t)im->sampler, (uint64_t)im->imageView, (uint64_t)im->imageLayout};
            h = fnv(h, fields, sizeof fields);
        }
        else if (x->pTexelBufferView) h = fnv(h, x->pTexelBufferView, sizeof *x->pTexelBufferView);
    }
    unsigned k = (unsigned)(h % SETC); uint64_t tb2 = tick(); g_ns_set_hash += tb2 - tb1; if (sp) sp->hash_ns += tb2 - tb1;
    if (g_setc[k].epoch == g_set_epoch && g_setc[k].h == h && g_setc[k].l == s->dsl && !g_tracing && !g_x_nosetc && !(g_x_nosetc_bits & (st >= 3 ? 2 : 4))) { g_set_hits++; if (sp) sp->hits++; set = g_setc[k].set; }
    else { set = vk_alloc_set(s->dsl); g_set_new++; if (sp) sp->fresh++;
        for (uint32_t i = 0; i < nw; i++) w[i].dstSet = set;
        vkUpdateDescriptorSets(V.dev, nw, w, 0, NULL);
        g_setc[k].h = h; g_setc[k].epoch = g_set_epoch; g_setc[k].l = s->dsl; g_setc[k].set = set; uint64_t elapsed = tick() - tb2; g_ns_set_new += elapsed; if (sp) sp->new_ns += elapsed; }
    // verify: the same content (hash), integer-texel bindings and dynamic offsets as the memo gave. A lone difference in dofs alone can be a driver segment
    // copied again after its g_drvc entry was taken by another (the memo's copy is as good); anything repeated is a dependency the memo misses
    if (verify && (h != g_smemo[st].h || g_num_um != vum || g_num_im != vim || memcmp(vd, dofs, sizeof *dofs * (size_t)s->ndyn))) { g_set_memo_bad++; static int n;
        if (n++ < 20) VKLOG("[RED] descriptor set memo: stage %d shader %016llx: a fresh set differs from the memo's (hash %016llx / %016llx, um %#llx / %#llx, im %#llx / %#llx, dynamic offsets %s)", st, (unsigned long long)s->key,
                            (unsigned long long)h, (unsigned long long)g_smemo[st].h, (unsigned long long)g_num_um, (unsigned long long)vum, (unsigned long long)g_num_im, (unsigned long long)vim, memcmp(vd, dofs, sizeof *dofs * (size_t)s->ndyn) ? "differ" : "same"); }
    if (memo) { set_memo_t *m = &g_smemo[st];   // (set_memo_ok: at most MEMO_CB constant buffers)
        m->s = s; m->epoch = g_set_epoch; m->bind = g_bind_gen[st]; m->ssbo = g_ssbo_gen[st]; m->ubo = g_ubo_gen[st]; m->ubd = g_ubd_gen; m->tex = gtex; m->samp = gsamp; m->pool = gpool; m->binst = g_draw_binst;
        m->h = h; m->um = g_num_um; m->im = g_num_im; m->set = set; memcpy(m->dofs, dofs, sizeof *dofs * (size_t)s->ndyn); m->ncb = ncb; memcpy(m->ci, mci, (size_t)ncb); memcpy(m->cd, mcd, sizeof *mcd * (size_t)ncb);
        for (int c = 0; c < ncb; c++) { m->cpos[c] = 0xff; for (int j = 0; j < nd; j++) if (dp[j].i == mci[c]) m->cpos[c] = (uint8_t)j; } }
    return set;
}

// ---------------- texture tables (gfx_vk.h: set 0 of every graphics pipeline layout) ----------------
// The set for the handles a stage's table reads: element k of each slot's array is the texture of the handle word at off + 8k of the stage's uniform buffer reg, through a view of
// the slot's kind (2D, 2D array, cube; what does not fit the kind, or has integer texels, gets the dummy), up to the buffer's bound size and the table's length; the rest dummies.
// Sets are kept per submission by the handles' hash (and the texture, sampler and pool generations), as stage_set's are; a draw's set is bound only when it differs from the last
// bound (draw). DM_VK_TBL_N=0 leaves the arrays in the stages' sets as before
// The texture and sampler pool tables (gfx_vk.h RK_TEXP / RK_SAMPP) share the set: element k of each kind's sampled-image array is the texture registered under ID k (the bound
// pool's base added, mem_handle) through a view of the kind, the dummy where none fits or the texels are integers; element k of the sampler array is sampler ID k. Written when a
// stage of the draw reads a pool (pool), from the registries as they are, and kept by the same cache: a set lives as long as no texture or sampler registration changes
// tbl[] is the actual remapping selected by shader.c, including every pool kind used by a stage.
// Mesh/task kernels have no table set (shader.c rejects pool bindings there); their raster pass uses prepare_draw.
static unsigned tbl_kinds(const vkshader_t *s) {
    unsigned kinds = 0;
    if (s) for (int i = 0; i < s->nbind && i < 96; i++)
        if (s->tbl[i] > 0 && s->tbl[i] <= SET0_BINDINGS) kinds |= 1u << (s->tbl[i] - 1);
    return kinds;
}
static struct { uint64_t h, epoch, tex, samp; uint32_t n, pool, kinds; int tex_base, samp_base; VkDescriptorSet set; } g_tblc[64];
static uint64_t g_tbl_builds, g_tbl_hits, g_tbl_binds, g_ns_tbl, g_pool_builds; static uint32_t g_tbl_max_n, g_tbl_max_bytes;
// Pool-only contents may outlive a submission only through the allocator's completed-slot lease.
// No cached native pointers are retained: invalidation drops CPU metadata, never in-flight sets.
// Every hit still refreshes each distinct texture source, including aliases with different serials.
enum { TBL_PERSIST_ENTRIES = 32, TBL_PERSIST_ENTRY_BYTES = 8 << 20, TBL_PERSIST_BYTES = 32 << 20 };
typedef struct { uint64_t tex, samp, pool, kinds, tex_base, samp_base, tex_seq, samp_seq; } tbl_persist_key;
typedef struct { tex_t *v; size_t n, cap; nvn_hmap seen; int valid; uint64_t hash; } tbl_sources;
typedef struct { size_t refs, n; uint64_t hash; tex_t v[]; } tbl_payload;
typedef struct { tbl_persist_key key; VkDescriptorSet set; uint64_t lease, age; tbl_payload *data; } tbl_persist_entry;
static tbl_persist_entry g_tblp[TBL_PERSIST_ENTRIES];
static size_t g_tblp_bytes;
typedef struct { uint64_t attempts, key_miss, shape_miss, lease_miss, stores, invalid, changed, no_lease, budget_evict, replaced, sources, mismatch[5]; } tbl_persist_stats;
static tbl_persist_stats g_tblpd;
static uint64_t g_tblp_age, g_tblp_hits, g_tblp_refresh_ns, g_tblp_refresh_n, g_tblp_fallbacks, g_tbl_full_ns;
static int tbl_persist_enabled(void) { const char *e = getenv("DM_VK_POOL_PERSIST"); return e && atoi(e) != 0; }
static int tbl_persist_identity(tbl_persist_key *key, unsigned kinds) {
    uint64_t tex_seq = 0, samp_seq = 0;
    if (!vk_registry_snapshot(&tex_seq, &samp_seq)) return 0;
    *key = (tbl_persist_key){g_tex_gen, __atomic_load_n(&g_samp_gen, __ATOMIC_ACQUIRE), vk_pool_gen(), kinds,
        (uint32_t)g_pool_tex_base, (uint32_t)g_pool_samp_base, tex_seq, samp_seq};
    return 1;
}
static void tbl_sources_add(tbl_sources *b, const tex_t *t) {
    if (!b->valid || nvn_hm_get(&b->seen, t)) return;
    if (b->n == b->cap) {
        size_t cap = b->cap ? b->cap * 2 : 256;
        if (cap > TBL_PERSIST_ENTRY_BYTES / sizeof *b->v) { b->valid = 0; return; }
        void *v = realloc(b->v, cap * sizeof *b->v);
        if (!v) { b->valid = 0; return; } b->v = v; b->cap = cap;
    }
    b->hash = (b->hash ^ nvn_hm_hash(t, sizeof *t)) * 0x100000001b3ull;
    b->v[b->n++] = *t; nvn_hm_put(&b->seen, t, (void *)1);
}
static void tbl_sources_free(tbl_sources *b) {
    free(b->v); free(b->seen.keys); free(b->seen.vals); memset(b, 0, sizeof *b);
}
static void tbl_payload_release(tbl_payload *data) {
    if (data && !--data->refs) { g_tblp_bytes -= sizeof *data + data->n * sizeof *data->v; free(data); }
}
static void tbl_persist_drop(tbl_persist_entry *e) {
    tbl_payload_release(e->data); memset(e, 0, sizeof *e);
}
// 1: hit, 0: no lease; -1: refresh invalidated descriptors, caller must rewrite the claimed set.
static int tbl_persist_take(const tbl_persist_key *key, VkDescriptorSet *set) {
    const tbl_persist_entry *latest = NULL; int exact = 0; g_tblpd.attempts++;
    for (unsigned i = 0; i < TBL_PERSIST_ENTRIES; i++) {
        tbl_persist_entry *e = &g_tblp[i]; if (!e->lease) continue;
        if (e->key.kinds == key->kinds && e->key.tex_base == key->tex_base && e->key.samp_base == key->samp_base &&
            (!latest || e->age > latest->age)) latest = e;
        if (memcmp(&e->key, key, sizeof *key)) continue;
        exact = 1; if (!vk_descriptor_claim(e->set, e->lease)) continue;
        *set = e->set; uint64_t start = tick(); int ok = 1;
        for (size_t j = 0; j < e->data->n; j++) {
            vkimg_t *im = vk_tex(&e->data->v[j]); g_tblp_refresh_n++;
            if (!im) { ok = 0; break; }
            im->last_use = mtl_last_fence() + 1; vk_texdump_note(im);
        }
        g_tblp_refresh_ns += tick() - start;
        tbl_persist_key now;
        if (!ok || !tbl_persist_identity(&now, (unsigned)key->kinds) || memcmp(key, &now, sizeof now)) {
            vk_descriptor_lease(*set, 1); tbl_persist_drop(e); g_tblp_fallbacks++; return -1;
        }
        e->age = ++g_tblp_age; g_tblp_hits++; return 1;
    }
    if (exact) g_tblpd.lease_miss++;
    else {
        g_tblpd.key_miss++;
        if (!latest) g_tblpd.shape_miss++;
        else {
            g_tblpd.mismatch[0] += latest->key.tex != key->tex;
            g_tblpd.mismatch[1] += latest->key.samp != key->samp;
            g_tblpd.mismatch[2] += latest->key.pool != key->pool;
            g_tblpd.mismatch[3] += latest->key.tex_seq != key->tex_seq;
            g_tblpd.mismatch[4] += latest->key.samp_seq != key->samp_seq;
        }
    }
    return 0;
}
static void tbl_persist_save(const tbl_persist_key *key, VkDescriptorSet set, tbl_sources *b) {
    tbl_persist_key now;
    if (!b->valid) { g_tblpd.invalid++; return; }
    if (!tbl_persist_identity(&now, (unsigned)key->kinds) || memcmp(key, &now, sizeof now)) { g_tblpd.changed++; return; }
    size_t bytes = sizeof(tbl_payload) + b->n * sizeof *b->v; unsigned chosen = 0; tbl_payload *data = NULL;
    // Slots/masks often reference the same source list. Sharing its immutable bytes avoids
    // six slots evicting one another's metadata without increasing the fixed memory budget.
    // Hash only narrows candidates; exact ordered byte equality remains the proof.
    for (unsigned i = 0; i < TBL_PERSIST_ENTRIES; i++) {
        tbl_payload *candidate = g_tblp[i].data;
        if (candidate && candidate->hash == b->hash && candidate->n == b->n &&
            (!b->n || !memcmp(candidate->v, b->v, b->n * sizeof *b->v))) { data = candidate; data->refs++; break; }
    }
    for (unsigned i = 0; i < TBL_PERSIST_ENTRIES; i++) {
        if (g_tblp[i].set == set || !g_tblp[i].lease) { chosen = i; break; }
        if (g_tblp[i].age < g_tblp[chosen].age) chosen = i;
    }
    if (g_tblp[chosen].lease) g_tblpd.replaced++;
    tbl_persist_drop(&g_tblp[chosen]);
    if (!data) {
        while (bytes > TBL_PERSIST_BYTES - g_tblp_bytes) {
            unsigned oldest = 0;
            for (unsigned i = 1; i < TBL_PERSIST_ENTRIES; i++)
                if (g_tblp[i].lease && (!g_tblp[oldest].lease || g_tblp[i].age < g_tblp[oldest].age)) oldest = i;
            if (!g_tblp[oldest].lease) return;
            tbl_persist_drop(&g_tblp[oldest]); g_tblpd.budget_evict++;
        }
        data = malloc(bytes); if (!data) return;
        data->refs = 1; data->n = b->n; data->hash = b->hash;
        if (b->n) memcpy(data->v, b->v, b->n * sizeof *b->v);
        g_tblp_bytes += bytes;
    }
    uint64_t lease = vk_descriptor_lease(set, 0);
    if (!lease) { tbl_payload_release(data); g_tblpd.no_lease++; return; }
    g_tblp[chosen] = (tbl_persist_entry){*key, set, lease, ++g_tblp_age, data};
    g_tblpd.stores++; g_tblpd.sources += b->n;
}
void vk_table_report(void) {
    static tbl_persist_stats prev;
    if (g_tblpd.attempts != prev.attempts) {
        VKLOG("persistent table decisions (last 600): %llu attempts, %llu key misses (%llu without matching kinds/bases; latest matching-kind differences: texture %llu, sampler %llu, pool %llu, texture snapshot %llu, sampler snapshot %llu), %llu exact-key lease misses; %llu stores averaging %.0f sources, rejected source %llu / changed dependency %llu / lease %llu, metadata evictions budget %llu / replacement %llu",
            (unsigned long long)(g_tblpd.attempts - prev.attempts), (unsigned long long)(g_tblpd.key_miss - prev.key_miss), (unsigned long long)(g_tblpd.shape_miss - prev.shape_miss),
            (unsigned long long)(g_tblpd.mismatch[0] - prev.mismatch[0]), (unsigned long long)(g_tblpd.mismatch[1] - prev.mismatch[1]), (unsigned long long)(g_tblpd.mismatch[2] - prev.mismatch[2]),
            (unsigned long long)(g_tblpd.mismatch[3] - prev.mismatch[3]), (unsigned long long)(g_tblpd.mismatch[4] - prev.mismatch[4]), (unsigned long long)(g_tblpd.lease_miss - prev.lease_miss),
            (unsigned long long)(g_tblpd.stores - prev.stores), g_tblpd.stores > prev.stores ? (double)(g_tblpd.sources - prev.sources) / (double)(g_tblpd.stores - prev.stores) : 0,
            (unsigned long long)(g_tblpd.invalid - prev.invalid), (unsigned long long)(g_tblpd.changed - prev.changed), (unsigned long long)(g_tblpd.no_lease - prev.no_lease),
            (unsigned long long)(g_tblpd.budget_evict - prev.budget_evict), (unsigned long long)(g_tblpd.replaced - prev.replaced));
    }
    prev = g_tblpd;
    static uint64_t builds, hits, persistent, all_ns, full_ns, refresh_ns, refresh_n, fallback;
    { extern uint64_t g_tex_gen_why[4]; VKLOG("texture generation bumps (cumulative): orphan %llu, first use %llu, view drop %llu, registration %llu", (unsigned long long)g_tex_gen_why[0], (unsigned long long)g_tex_gen_why[1], (unsigned long long)g_tex_gen_why[2], (unsigned long long)g_tex_gen_why[3]); }
    VKLOG("pool table ring (cumulative): %llu whole writes, %llu patches (%llu IDs), %llu reuses, %llu times no free set; %.1f ms in all", (unsigned long long)g_ring_full, (unsigned long long)g_ring_patch, (unsigned long long)g_ring_ids, (unsigned long long)g_ring_hits, (unsigned long long)g_ring_none, g_ring_ns / 1e6);
    VKLOG("texture tables per frame (last 600): %.2f full builds, %.1f submission hits, %.2f persistent hits; total %.3f ms, build %.3f ms, refresh %.3f ms (%.1f sources), %.2f refresh fallbacks; metadata %.2f MB",
        (g_tbl_builds - builds) / 600.0, (g_tbl_hits - hits) / 600.0, (g_tblp_hits - persistent) / 600.0,
        (g_ns_tbl - all_ns) / 6e8, (g_tbl_full_ns - full_ns) / 6e8, (g_tblp_refresh_ns - refresh_ns) / 6e8,
        (g_tblp_refresh_n - refresh_n) / 600.0, (g_tblp_fallbacks - fallback) / 600.0, g_tblp_bytes / 1048576.0);
    builds = g_tbl_builds; hits = g_tbl_hits; persistent = g_tblp_hits; all_ns = g_ns_tbl; full_ns = g_tbl_full_ns;
    refresh_ns = g_tblp_refresh_ns; refresh_n = g_tblp_refresh_n; fallback = g_tblp_fallbacks;
}
// Trace the immutable set actually used instead of forcing a fresh large allocation per draw.
// Content dependencies still invalidate normally; DM_VK_X_NOSETC remains the explicit cache bypass.
static void tbl_trace(unsigned c, int st, const vkshader_t *s, int reused) {
    if (!g_tracing) return;
    const typeof(g_tblc[0]) *t = &g_tblc[c];
    VKLOG("  table set %p %s: stage %d shader %016llx kinds %#x, %u handles hash %016llx, pool bases %d/%d, epoch %llu generations %llu/%llu/%u",
          (void *)(uintptr_t)t->set, reused ? "reused" : "built", st, (unsigned long long)(s ? s->key : 0), t->kinds, t->n,
          (unsigned long long)t->h, t->tex_base, t->samp_base, (unsigned long long)t->epoch,
          (unsigned long long)t->tex, (unsigned long long)t->samp, t->pool);
}

// ---------------- the pool tables' ring (DM_VK_POOL_RING=0: off) ----------------
// The Mac patches its pool tables (nvn_mtl.m pool_gpu): a change takes only the IDs registered since the table was built. Here a table is one descriptor set of 4 x 65,536 images
// and 4,096 samplers, and tbl_set wrote it whole on every texture registration: Ocarina of Time's Kokiri Forest registers about 130 changed textures a frame, so 2-3 whole tables
// (50-60 ms) a frame, and a fresh 68 MB descriptor pool for each. Instead RING_N sets live for the whole run, each with every pool kind and the samplers: a set the GPU is done
// with (its last submission's fence completed, and not bound in the one being recorded) is brought up to date by rewriting only the IDs the registration log names after its
// position (gfx_vk_tex.c vk_reglog_take, the entries visible to the replaying job), and written whole only when an image went or an ID got contents (the view drop, orphan and
// first-use generations), the pool bases changed, the log wrapped or the replay went back to an earlier job. An image whose views went
// (eviction; gfx_vk_tex.c's drop log) has only the IDs that held it written again: Ocarina of Time places each frame's render targets at new offsets of one pool, so images
// go nearly every frame, and every one had the set after it written whole (21 ms on the Odin 3, 4 x 65,536 entries). IDs that read the dummy for lack of
// contents or of first-use budget are retried by later patches, RING_RETRY at a time. Each take marks the images the set holds as used by the submission (eviction waits for it)
enum { RING_N = 6, RING_TIDS = 2048, RING_RETRY = 256, RING_DROPS = 1024 };
typedef struct { VkDescriptorSet set; uint64_t fence, logpos, fullgen, sampgen, sampseq, jobpos, dropseq, firstgen; uint32_t poolgen; int tex_base, samp_base, valid; vkimg_t **img; int *itid; size_t nimg, capimg; int *pend; size_t npend, cappend, pcur; } ring_slot;
static ring_slot g_ring[RING_N]; static int g_ring_on = -1, g_ring_cur = -1;
uint64_t vk_reglog_visible_end(uint64_t before); uint64_t vk_reglog_head(void); int vk_reglog_pending(uint64_t p, uint64_t before); uint64_t vk_reglog_take(uint64_t from, uint64_t before, int *tids, int max, int *n);
int vk_alloc_sets_owned(VkDescriptorSetLayout l, int n, VkDescriptorSet *out); unsigned vk_pool_kinds(void); uint64_t nvn_replaying_job(void);
extern uint64_t g_tex_gen_why[4], g_samp_gen;
uint64_t vk_droplog_seq(void); int vk_droplog_take(uint64_t from, const void **out, int max, uint64_t *to);
static uint64_t ring_fullgen(void) { return g_tex_gen_why[0]; }   // (view drops: the drop log; first uses: the waiting IDs)
static void ring_img(ring_slot *r, vkimg_t *im, int tid) {
    if (!im) return;
    if (r->nimg && r->img[r->nimg - 1] == im && r->itid[r->nimg - 1] == tid) return;   // (the same texture through each kind)
    if (r->nimg == r->capimg) { r->capimg = r->capimg ? 2 * r->capimg : 4096; r->img = realloc(r->img, r->capimg * sizeof *r->img); r->itid = realloc(r->itid, r->capimg * sizeof *r->itid); }
    r->img[r->nimg] = im; r->itid[r->nimg++] = tid;
}
static void ring_pend(ring_slot *r, int tid) {
    if (r->npend == r->cappend) { r->cappend = r->cappend ? 2 * r->cappend : 1024; r->pend = realloc(r->pend, r->cappend * sizeof *r->pend); }
    r->pend[r->npend++] = tid;
}
// One entry of kind ps for pool index k (the ID minus the bound pool's base), as tbl_set's fill decides it; 1 when it should be tried again later (no contents yet, or no budget)
static int ring_entry(int st, ring_slot *r, int ps, int k, VkDescriptorImageInfo *d, int *lazy_n, uint64_t *lazy_b) {
    dmsv_bind vb = {.kind = RK_TEXP, .desc = RD_IMAGE, .dim = (uint8_t)(ps == 2 ? 3 : ps == 3 ? 2 : 1), .flags = (uint8_t)(ps == 1 ? 1 : 0), .samp = RS_NONE, .reg = -1, .off = -1};
    VkDescriptorImageInfo dummy = {VK_NULL_HANDLE, vk_dummy_view(vb.dim, vb.flags & 1, 0, 0, 0), VK_IMAGE_LAYOUT_GENERAL}; *d = dummy;
    uint64_t hh = mem_handle((uint64_t)k); const view_t *v = NULL; const tex_t *t = NULL;
    if (!vk_tex_by_id((int)(hh & 0xfffff), &v, &t) || !t) return 0;
    int target = v && v->set_target ? v->target : t->d.target;
    int fits = ps == 2 ? (t->d.width == t->d.height && (target == 8 || target == 9 || (target == 4 && t->d.depth >= 6))) : ps == 3 ? target == 2 : (target == 1 || target == 4 || target == 8 || target == 9);
    if (g_pool_strict && fits) fits = ps == 0 ? target == 1 : ps == 1 ? target == 4 : ps == 2 ? target == 8 : 1;
    if (fits && view_num(t, v)) fits = 0;
    if (fits) { int id_ = (int)(hh & 0xfffff), any = g_pdum_lo >= 0 || g_pdum_kinds || g_pdum_nf, hit = any;   // the bisection switches, as tbl_set's fill
        if (g_pdum_lo >= 0 && (id_ < g_pdum_lo || id_ > g_pdum_hi)) hit = 0;
        if (g_pdum_kinds && !(g_pdum_kinds & (1u << ps))) hit = 0;
        if (g_pdum_nf) { int f = 0; for (int q = 0; q < g_pdum_nf; q++) if ((uint32_t)t->d.format == g_pdum_f[q]) f = 1; if (!f) hit = 0; }
        if (hit) fits = 0; }
    if (!fits) return 0;
    int ok = vk_tex_pool_ok(t);
    if (ok == 1) { if (*lazy_n < 64 && *lazy_b < (64u << 20)) { (*lazy_n)++; *lazy_b += t->L.size; } else return 1; }
    if (!ok) return 1;
    tex_desc_for(st, &vb, hh, d, 0); d->sampler = VK_NULL_HANDLE;
    if (!d->imageView) { *d = dummy; return 1; }   // (no image yet: waits like one without contents)
    ring_img(r, g_desc_im, k + g_pool_tex_base); return 0;
}
// The pool-texture prefetch (profile env DM_VK_POOL_PREFETCH=1). The tables' fills give a registered texture its first use lazily, 64 a fill, and nothing does before a
// shader reads the tables: Ocarina of Time registers about 3,010 textures (1,015 MB) 3 s after its start and draws its title and file select without the tables for
// 30 s more, so the first frame of the world took them all at once: 36 table builds at one fence, 2,196 first uploads, 742 MB of image memory from the kernel, one frame
// of 2.7 to 3.0 s (Odin 3, 2026-11-10). The prefetch makes those first uses ahead, a few a frame: once a present, from any draw, it walks PF_IDS registered IDs on from a
// cursor and gives the ones the tables would take (ring_entry's kinds, vk_tex_pool_ok 1) their first use once they have waited PF_AGE presents, within PF_N textures,
// PF_B bytes and PF_NS a frame. Their image memory is asked softly (vk_image_memory_soft): the kernel's allocation runs on a helper thread and the texture waits for it.
// Not past 80% of the texture budget: what the budget drops is ready again, and the prefetch would bring it straight back.
// A cycle that finds nothing to do slows the walk to PF_IDLE_IDS a present
static void pool_prefetch(void) {
    static int on = -1; if (on < 0) { const char *e = getenv("DM_VK_POOL_PREFETCH"); on = e && atoi(e); if (on) VKLOG("pool prefetch: on (DM_VK_POOL_PREFETCH)"); }
    static uint64_t last = ~0ull; if (!on || g_tex_frame == last) return; last = g_tex_frame;
    if (vk_image_memory_busy()) return;
    enum { PF_IDS = 256, PF_IDLE_IDS = 16, PF_AGE = 30, PF_N = 16 }; const uint64_t PF_B = 4u << 20, PF_NS = 1500000;
    static uint32_t *seen; static int cap, cur = 1, idle, work; static uint64_t done_n, done_b, logged_n;
    int hi = vk_tex_id_max(); if (hi < 1) return;
    if (hi >= cap) { int nc = hi + 4096; uint32_t *p = realloc(seen, (size_t)nc * sizeof *p); if (!p) return; memset(p + cap, 0, (size_t)(nc - cap) * sizeof *p); seen = p; cap = nc; }
    uint64_t bu, ev, evb, sw; vk_img_budget_stats(&bu, &ev, &evb, &sw); if (bu && vk_img_vkbytes() > bu / 10 * 8) return;
    uint64_t t0 = tick(), bytes = 0; int n = 0; uint32_t now = (uint32_t)g_tex_frame + 1;
    vk_tex_pool_fill(1); vk_image_memory_soft(1);
    for (int i = idle ? PF_IDLE_IDS : PF_IDS; i > 0; i--) {
        if (cur > hi) { cur = 1;   // a cycle is over
            if (!work && !idle && done_n != logged_n) { logged_n = done_n; VKLOG("pool prefetch: %llu first uses made ahead so far (%.0f MB of storage); nothing waits at present %llu", (unsigned long long)done_n, done_b / 1048576.0, (unsigned long long)g_tex_frame); }
            idle = !work; work = 0; }
        int id = cur; const view_t *v = NULL; const tex_t *t = NULL;
        if (seen[id] == ~0u && vk_tex_by_id(id, &v, &t) && t) { cur++; continue; }   // (had no image: left to a real bind)
        if (!vk_tex_by_id(id, &v, &t) || !t) { seen[id] = 0; cur++; continue; }
        int target = v && v->set_target ? v->target : t->d.target;
        if (!(target == 1 || target == 2 || target == 4 || target == 8 || target == 9) || view_num(t, v) || vk_tex_pool_ok(t) != 1) { seen[id] = 0; cur++; continue; }
        work = 1; idle = 0;
        if (!seen[id]) seen[id] = now;
        if (now - seen[id] < PF_AGE) { cur++; continue; }
        uint64_t size = t->L.size; vkimg_t *im = vk_tex(t);
        if (!im) { int sf = vk_image_memory_soft_failed(); if (sf == 2) on = 0; if (sf) break;   // its memory is on its way: this one again when the helper is done
            seen[id] = ~0u; cur++; continue; }
        seen[id] = 0; cur++; n++; bytes += size; done_n++; done_b += size;
        if (n >= PF_N || bytes >= PF_B || tick() - t0 >= PF_NS) break;
    }
    vk_image_memory_soft(0); vk_tex_pool_fill(0);
}
static VkDescriptorSet tbl_ring(int st, unsigned poolmask) {
    // Global sampler generations describe the newest registration, while queued jobs
    // resolve IDs at their own enqueue snapshot. Never reuse historical contents
    // without that visible identity (legacy unpinned jobs use the ordinary table path).
    uint64_t texseq, sampseq;
    if (!vk_registry_snapshot(&texseq, &sampseq)) return VK_NULL_HANDLE;
    if (g_ring_on < 0) { const char *e = getenv("DM_VK_POOL_RING"); g_ring_on = !e || atoi(e);
        if (g_ring_on) { VkDescriptorSet sets[RING_N]; if (!vk_alloc_sets_owned(vk_tbl_dsl(), RING_N, sets)) { g_ring_on = 0; VKLOG("[RED] pool table ring: its descriptor pool could not be made; tables are built as before"); }
            else { for (int i = 0; i < RING_N; i++) g_ring[i].set = sets[i]; VKLOG("pool table ring: %d persistent sets (DM_VK_POOL_RING=0 builds tables as before)", RING_N); } } }
    if (!g_ring_on || !(poolmask & ((1u << POOL_TEX_SLOTS) - 1))) return VK_NULL_HANDLE;
    uint64_t t0 = tick(), f = mtl_last_fence() + 1, job = nvn_replaying_job(), jp = job ? job : UINT64_MAX, fg = ring_fullgen(), sg = __atomic_load_n(&g_samp_gen, __ATOMIC_ACQUIRE); uint32_t pg = vk_pool_gen();
    if (g_ring_cur >= 0) { ring_slot *r = &g_ring[g_ring_cur];
        if (r->valid && r->fence == f && r->fullgen == fg && r->sampgen == sg && r->sampseq == sampseq && r->poolgen == pg && r->tex_base == g_pool_tex_base && r->samp_base == g_pool_samp_base && r->jobpos <= jp && r->dropseq == vk_droplog_seq() && r->firstgen == g_tex_gen_why[1] && !vk_reglog_pending(r->logpos, jp)) {
            g_ring_hits++; g_ring_ns += tick() - t0; return r->set; } }
    // a set the GPU is done with and the recording command buffer does not hold: the most recently written one (fewest IDs to take again)
    uint64_t done = vk_fence_done(); int pick = -1;
    for (int i = 0; i < RING_N; i++) { ring_slot *r = &g_ring[i]; if (r->fence > done || r->fence == f) continue;
        if (pick < 0 || (r->valid && (!g_ring[pick].valid || r->logpos > g_ring[pick].logpos))) pick = i; }
    if (pick < 0) { g_ring_none++; return VK_NULL_HANDLE; }
    ring_slot *r = &g_ring[pick]; const int NP = vk_pool_n(), NS = vk_pool_samp_n(); const unsigned kinds = vk_pool_kinds();
    static VkDescriptorImageInfo *dd; static VkWriteDescriptorSet *ww; static int *tids;
    if (!dd) { dd = calloc((size_t)POOL_TEX_SLOTS * NP + NS + (size_t)(RING_TIDS + RING_RETRY) * POOL_TEX_SLOTS, sizeof *dd); ww = calloc((size_t)(RING_TIDS + RING_RETRY + RING_DROPS) * POOL_TEX_SLOTS + 8, sizeof *ww); tids = calloc(RING_TIDS + RING_DROPS, sizeof *tids); }
    int full = !r->valid || r->fullgen != fg || r->tex_base != g_pool_tex_base || r->samp_base != g_pool_samp_base || r->jobpos > jp || r->nimg > 8u * (size_t)NP;
    int lazy_n = 0; uint64_t lazy_b = 0; uint32_t nw = 0; uint64_t head = vk_reglog_head(), pos = r->logpos; int ntid = 0;
    if (!full) { pos = vk_reglog_take(r->logpos, jp, tids, RING_TIDS, &ntid); if (pos == UINT64_MAX || (ntid == RING_TIDS && vk_reglog_pending(pos, jp))) full = 1; }
    // the images that went since this set was written: their IDs again, and out of the list before anything reads it (taken before the entries are written: one that
    // goes while they are is found by the next set's take)
    static const void *gone[RING_DROPS]; uint64_t dseq; int ngone = vk_droplog_take(r->dropseq, gone, RING_DROPS, &dseq); if (ngone < 0) full = 1;
    if (!full && ngone) for (size_t i = 0; i < r->nimg; ) { int hit = 0; for (int q = 0; q < ngone && !hit; q++) hit = r->img[i] == gone[q];
        if (!hit) { i++; continue; }
        if (ntid == RING_TIDS + RING_DROPS) { full = 1; break; }
        tids[ntid++] = r->itid[i]; r->nimg--; r->img[i] = r->img[r->nimg]; r->itid[i] = r->itid[r->nimg]; }
    // an image got its contents at a real bind since (a first use): every ID that waits on the dummy is tried again now, not RING_RETRY of them.
    // Pool memory that came or went (vk_pool_gen) takes a patch and no more: an entry already written holds a live image, which only the logs above take back, and what a pool
    // changes is vk_tex_pool_ok's answer for an ID without contents (its memory inside a live pool), which waits in the retry list. Writing the set whole for it was 23 ms,
    // 237 times in 900 frames of Ocarina of Time's Kokiri Forest (the game makes and frees pools as it streams)
    uint64_t fu = g_tex_gen_why[1]; int all = !full && r->firstgen != fu; if (all && r->npend > (size_t)(RING_TIDS + RING_DROPS - ntid)) full = 1;
    vk_tex_pool_fill(1);
    if (full) {   // every entry, as tbl_set's fill
        r->nimg = 0; r->npend = 0; r->pcur = 0; pos = vk_reglog_visible_end(jp); (void)head;   // later jobs' registrations stay in the log for the next patch
        static uint8_t *again; if (!again) again = calloc((size_t)NP, 1); else memset(again, 0, (size_t)NP);
        VkDescriptorImageInfo *d = dd; int hw = vk_tex_id_max() - g_pool_tex_base + 1; if (hw > NP) hw = NP;
        for (int ps = 0; ps < POOL_TEX_SLOTS; ps++) {
            int len = (kinds & (1u << ps)) ? NP : 1;
            for (int k = 0; k < len; k++) { if (k == 0 || k >= hw) { dmsv_bind vb = {.dim = (uint8_t)(ps == 2 ? 3 : ps == 3 ? 2 : 1), .flags = (uint8_t)(ps == 1 ? 1 : 0)};
                    d[k] = (VkDescriptorImageInfo){VK_NULL_HANDLE, vk_dummy_view(vb.dim, vb.flags & 1, 0, 0, 0), VK_IMAGE_LAYOUT_GENERAL}; continue; }
                if (ring_entry(st, r, ps, k, &d[k], &lazy_n, &lazy_b)) again[k] = 1; }
            ww[nw++] = (VkWriteDescriptorSet){VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, NULL, r->set, (uint32_t)(POOL_B0 + ps), 0, (uint32_t)len, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, d, NULL, NULL}; d += len; }
        for (int k = 1; k < NP; k++) if (again[k]) ring_pend(r, k + g_pool_tex_base);
        g_ring_full++;
    } else {   // the IDs logged since, and some of those left the dummy
        VkDescriptorImageInfo *d = dd; int nret = 0;
        const int cap = all ? RING_TIDS + RING_DROPS : RING_TIDS;
        if (r->npend && ntid < cap) { size_t room = (size_t)(cap - ntid), take = all || r->npend < RING_RETRY ? r->npend : RING_RETRY; if (take > room) take = room;   // retried: removed from the list, added again by ring_entry's answer below
            for (size_t i = 0; i < take; i++) { size_t j = (r->pcur + i) % r->npend; if (ntid < cap) tids[ntid++] = r->pend[j]; nret++; }
            if (take == r->npend) r->npend = 0; else { size_t keep = 0; int *np_ = malloc(r->npend * sizeof *np_);   // the ones not taken this time, in order after the cursor
                for (size_t i = take; i < r->npend; i++) np_[keep++] = r->pend[(r->pcur + i) % r->npend]; memcpy(r->pend, np_, keep * sizeof *np_); free(np_); r->npend = keep; r->pcur = 0; } }
        for (int i = 0; i < ntid; i++) { int k = tids[i] - g_pool_tex_base; if (k < 1 || k >= NP) continue; int again = 0;
            for (int ps = 0; ps < POOL_TEX_SLOTS; ps++) { if (!(kinds & (1u << ps))) continue;
                again |= ring_entry(st, r, ps, k, d, &lazy_n, &lazy_b);
                ww[nw++] = (VkWriteDescriptorSet){VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, NULL, r->set, (uint32_t)(POOL_B0 + ps), (uint32_t)k, 1, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, d, NULL, NULL}; d++; }
            if (again) ring_pend(r, tids[i]); }
        g_ring_patch++; g_ring_ids += (uint64_t)ntid; (void)nret;
    }
    vk_tex_pool_fill(0);
    if (full || r->sampgen != sg || r->sampseq != sampseq || r->samp_base != g_pool_samp_base) {   // the samplers, whole (4,096)
        VkDescriptorImageInfo *d = dd + (size_t)POOL_TEX_SLOTS * NP; d[0] = (VkDescriptorImageInfo){vk_dummy_sampler(), VK_NULL_HANDLE, VK_IMAGE_LAYOUT_GENERAL};
        int hw = vk_samp_id_max() - g_pool_samp_base + 1; if (hw > NS) hw = NS;
        for (int k = 1; k < NS; k++) { VkSampler sm = k < hw && !(g_pdum_kinds & 16) ? vk_samp_by_id(g_pool_samp_base + k) : VK_NULL_HANDLE; d[k] = (VkDescriptorImageInfo){sm ? sm : vk_dummy_sampler(), VK_NULL_HANDLE, VK_IMAGE_LAYOUT_GENERAL}; }
        ww[nw++] = (VkWriteDescriptorSet){VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, NULL, r->set, POOL_SAMP_B, 0, (uint32_t)NS, VK_DESCRIPTOR_TYPE_SAMPLER, d, NULL, NULL}; }
    if (nw) vkUpdateDescriptorSets(V.dev, nw, ww, 0, NULL);
    r->valid = 1; r->fullgen = fg; r->sampgen = sg; r->sampseq = sampseq; r->poolgen = pg; r->tex_base = g_pool_tex_base; r->samp_base = g_pool_samp_base; r->jobpos = jp; r->logpos = pos; r->fence = f; r->dropseq = dseq; r->firstgen = fu;
    for (size_t i = 0; i < r->nimg; i++) r->img[i]->last_use = f;   // (full rewrites drop the list; a view drop or orphan forces one before a freed image could be reached)
    g_ring_cur = pick; g_ring_ns += tick() - t0;
    return r->set;
}
static VkDescriptorSet tbl_set(int st, vkshader_t *s, unsigned poolmask) {   // poolmask: kinds used by ALL active raster stages, bit 0..3 images and bit 4 samplers
    uint64_t t0 = tick(); const int N = vk_tbl_n(), NP = vk_pool_n(), NS = vk_pool_samp_n(); const int ntbl = s && s->ntbl;
    static uint32_t words[2 * 65536 / 8]; static VkDescriptorImageInfo *ii;   // the handles read (8 bytes each), and TBL_SLOTS x N + POOL_TEX_SLOTS x NP + NS descriptors
    const p_bind_buf *u = ntbl && s->tbl_reg >= 0 && s->tbl_reg < 16 ? &S.ubo[st][s->tbl_reg] : NULL; uint32_t n = 0;
    if (!ntbl && poolmask && !g_x_nosetc && !(g_x_nosetc_bits & 8) && !g_var_now) { VkDescriptorSet rs = tbl_ring(st, poolmask); if (rs) { g_ns_tbl += tick() - t0; g_tbl_binds++; return rs; } }
    if (u && u->addr && s->tbl_off >= 0) { uint64_t a = u->addr + (uint64_t)s->tbl_off, size = u->size ? u->size : 65536; if (size > 65536) size = 65536;
        vkpool_t *p = vk_pool_peek(a, 8);
        if (p && size > (uint64_t)s->tbl_off) { uint64_t bytes = size - (uint64_t)s->tbl_off; if (a + bytes > p->end) bytes = p->end > a ? p->end - a : 0;
            n = (uint32_t)(bytes / 8); if (n > (uint32_t)N) n = (uint32_t)N; if (n) vk_ubd_read(a, (uint64_t)n * 8, (uint8_t *)words);
            if (bytes > g_tbl_max_bytes || n > g_tbl_max_n) { g_tbl_max_bytes = bytes > g_tbl_max_bytes ? (uint32_t)bytes : g_tbl_max_bytes; g_tbl_max_n = n > g_tbl_max_n ? n : g_tbl_max_n;
                static int nl; if (nl++ < 12) VKLOG("texture table: stage %d uniform buffer %d +%d bound with %llu bytes: %u handles read (table length %d)", st, s->tbl_reg, s->tbl_off, (unsigned long long)size, n, N); } } }
    // Pool bindings do not change registry generations. The bases and the handle-table kinds must be part of the identity:
    // an absent handle kind is filled with dummies, so a later shader using that kind needs a different set even with identical handles.
    unsigned kinds = (tbl_kinds(s) & ((1u << TBL_SLOTS) - 1)) | (poolmask << TBL_SLOTS);
    uint64_t h = 0xcbf29ce484222325ull ^ (uint64_t)n ^ (uint64_t)kinds << 32;
    h = (h ^ (uint32_t)g_pool_tex_base) * 0x100000001b3ull;
    h = (h ^ (uint32_t)g_pool_samp_base) * 0x100000001b3ull;
    for (uint32_t k = 0; k < n; k++) h = (h ^ words[2 * k]) * 0x100000001b3ull;   // the high word of each handle is not read
    // Pool-only tables have no handle words; their kind bits live high in h. Fold them into the bucket so alternating kinds coexist.
    uint64_t gtex = g_tex_gen, gsamp = __atomic_load_n(&g_samp_gen, __ATOMIC_ACQUIRE); uint32_t gpool = vk_pool_gen(); unsigned c = (unsigned)((h ^ (h >> 32)) % (sizeof g_tblc / sizeof *g_tblc));
    if (g_tblc[c].epoch == g_set_epoch && g_tblc[c].h == h && g_tblc[c].n == n && g_tblc[c].kinds == kinds && g_tblc[c].tex_base == g_pool_tex_base && g_tblc[c].samp_base == g_pool_samp_base && g_tblc[c].tex == gtex && g_tblc[c].samp == gsamp && g_tblc[c].pool == gpool && !g_x_nosetc) { tbl_trace(c, st, s, 1); g_tbl_hits++; g_ns_tbl += tick() - t0; return g_tblc[c].set; }
    tbl_persist_key persistent_key = {0}; VkDescriptorSet set = VK_NULL_HANDLE;
    int persistent = !ntbl && !n && poolmask && !g_x_nosetc && !g_var_now && tbl_persist_enabled() && tbl_persist_identity(&persistent_key, kinds);
    tbl_sources sources = {.seen = {.klen = sizeof(tex_t)}, .valid = persistent};
    if (persistent) {
        int hit = tbl_persist_take(&persistent_key, &set);
        if (hit == 1) {
            g_tblc[c] = (typeof(g_tblc[0])){h, g_set_epoch, gtex, gsamp, n, gpool, kinds, g_pool_tex_base, g_pool_samp_base, set};
            tbl_trace(c, st, s, 1); g_ns_tbl += tick() - t0; return set;
        }
        // Refresh may have changed image views or registration state. Rebuild against the new
        // identity, retaining ownership of the claimed set rather than handing it out twice.
        if (hit < 0) {
            persistent = tbl_persist_identity(&persistent_key, kinds); sources.valid = persistent;
            gtex = g_tex_gen; gsamp = __atomic_load_n(&g_samp_gen, __ATOMIC_ACQUIRE); gpool = vk_pool_gen();
        }
    }
    uint64_t build_start = tick();
    if (!ii) ii = calloc((size_t)TBL_SLOTS * (size_t)(N ? N : 1) + (size_t)POOL_TEX_SLOTS * (size_t)(NP ? NP : 1) + (size_t)(NS ? NS : 1), sizeof *ii);
    if (!set) set = vk_alloc_set(vk_tbl_dsl());
    VkWriteDescriptorSet w[SET0_BINDINGS]; uint32_t nwr = 0;
    if (poolmask && NP && NS) {   // Only statically used bindings need descriptors; never update a set after recording it.
        VkDescriptorImageInfo *d = ii + (size_t)TBL_SLOTS * (size_t)(N ? N : 1); uint32_t filled[POOL_TEX_SLOTS + 1] = {0}, skipped = 0, deferred = 0;
        // textures bound only from the registry (vk_tex_pool_ok): those with contents, and at most POOL_LAZY_N first uses of POOL_LAZY_B bytes a fill (the rest the next submission's);
        // the others get the dummy and the tables are built again when a real bind gives them contents
        enum { POOL_LAZY_N = 64 }; const uint64_t POOL_LAZY_B = 64u << 20; int lazy_n = 0; uint64_t lazy_b = 0;
        vk_tex_pool_fill(1);
        for (int ps = 0; ps < POOL_TEX_SLOTS; ps++, d += NP) {
            if (!(poolmask & (1u << ps))) continue;
            dmsv_bind vb = {.kind = RK_TEXP, .desc = RD_IMAGE, .dim = (uint8_t)(ps == 2 ? 3 : ps == 3 ? 2 : 1), .flags = (uint8_t)(ps == 1 ? 1 : 0), .samp = RS_NONE, .reg = -1, .off = -1};
            VkDescriptorImageInfo dummy = {VK_NULL_HANDLE, vk_dummy_view(vb.dim, vb.flags & 1, 0, 0, 0), VK_IMAGE_LAYOUT_GENERAL}; d[0] = dummy;
            int hw = vk_tex_id_max() - g_pool_tex_base + 1; if (hw > NP) hw = NP;   // IDs past the highest registered: the dummy without a lookup (the table's length is the device's, 65536; a scene registers tens of thousands)
            for (int k = hw > 1 ? hw : 1; k < NP; k++) d[k] = dummy;
            for (int k = 1; k < hw; k++) {
                uint64_t hh = mem_handle((uint64_t)k); const view_t *v = NULL; const tex_t *t = NULL; int fits = 0;
                if (vk_tex_by_id((int)(hh & 0xfffff), &v, &t) && t) { int target = v && v->set_target ? v->target : t->d.target;
                    fits = ps == 2 ? (t->d.width == t->d.height && (target == 8 || target == 9 || (target == 4 && t->d.depth >= 6))) : ps == 3 ? target == 2 : (target == 1 || target == 4 || target == 8 || target == 9);
                    if (g_pool_strict && fits) fits = ps == 0 ? target == 1 : ps == 1 ? target == 4 : ps == 2 ? target == 8 : 1;   // DM_VK_POOL_STRICT (x_read, test): as the Mac's pool_entry, each table holds only textures of its own type
                    if (fits && view_num(t, v)) fits = 0;   // (the arrays declare floating-point texels)
                    {   // DM_VK_POOL_DUMMY / _KINDS / _FMT (x_read): the conditions given must all hold
                        int id_ = (int)(hh & 0xfffff), any = g_pdum_lo >= 0 || g_pdum_kinds || g_pdum_nf, hit = any;
                        if (g_pdum_lo >= 0 && (id_ < g_pdum_lo || id_ > g_pdum_hi)) hit = 0;
                        if (g_pdum_kinds && !(g_pdum_kinds & (1u << ps))) hit = 0;
                        if (g_pdum_nf) { int f = 0; for (int q = 0; q < g_pdum_nf; q++) if ((uint32_t)t->d.format == g_pdum_f[q]) f = 1; if (!f) hit = 0; }
                        if (hit) fits = 0; }
                    if (fits) { int ok = vk_tex_pool_ok(t);
                        if (ok == 1) { if (lazy_n < POOL_LAZY_N && lazy_b < POOL_LAZY_B) { lazy_n++; lazy_b += t->L.size; } else { ok = 0; deferred++; } }
                        if (!ok) { fits = 0; skipped++; } } }
                if (fits) { tex_t source; if (sources.valid) source = *t;
                    tex_desc_for(st, &vb, hh, &d[k], 0); d[k].sampler = VK_NULL_HANDLE;
                    if (!d[k].imageView) d[k] = dummy; else filled[ps]++;
                    if (sources.valid) {
                        if (d[k].imageView == dummy.imageView) sources.valid = 0;
                        else tbl_sources_add(&sources, &source);
                    }
                } else d[k] = dummy;
            }
            w[nwr++] = (VkWriteDescriptorSet){VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, NULL, set, (uint32_t)(POOL_B0 + ps), 0, (uint32_t)NP, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, d, NULL, NULL};
        }
        vk_tex_pool_fill(0);
        // Deferred/no-content entries can become usable from CPU writes without registry changes.
        // Keep rebuilding those sets until every eligible source has a concrete descriptor.
        if (deferred || skipped) sources.valid = 0;
        if (poolmask & (1u << POOL_TEX_SLOTS)) {
            d[0] = (VkDescriptorImageInfo){vk_dummy_sampler(), VK_NULL_HANDLE, VK_IMAGE_LAYOUT_GENERAL};
            int hw = vk_samp_id_max() - g_pool_samp_base + 1; if (hw > NS) hw = NS;
            // The registry ID includes the pool base; only the guest-local ID is 12 bits.
            for (int k = 1; k < NS; k++) { VkSampler sm = k < hw && !(g_pdum_kinds & 16) ? vk_samp_by_id(g_pool_samp_base + k) : VK_NULL_HANDLE;   // DM_VK_POOL_DUMMY_KINDS bit 4: the dummy sampler throughout (diagnostic)
                if (sm) filled[POOL_TEX_SLOTS]++; d[k] = (VkDescriptorImageInfo){sm ? sm : vk_dummy_sampler(), VK_NULL_HANDLE, VK_IMAGE_LAYOUT_GENERAL}; }
            w[nwr++] = (VkWriteDescriptorSet){VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, NULL, set, POOL_SAMP_B, 0, (uint32_t)NS, VK_DESCRIPTOR_TYPE_SAMPLER, d, NULL, NULL};
        }
        g_pool_builds++;
        if (g_pool_builds <= 4 || !(g_pool_builds & (g_pool_builds - 1))) VKLOG("texture pool tables: build %llu (stage %d shader %016llx): %u 2D, %u 2D-array, %u cube, %u 3D textures of %d entries, %u samplers of %d; %d first uses made (%u deferred to a later fill, %u left the dummy: no contents and too large, or storage gone); the highest IDs registered %d / %d (pool bases %d / %d)",
            (unsigned long long)g_pool_builds, st, (unsigned long long)(s ? s->key : 0), filled[0], filled[1], filled[2], filled[3], NP, filled[4], NS, lazy_n, deferred, skipped, vk_tex_id_max(), vk_samp_id_max(), g_pool_tex_base, g_pool_samp_base);
        if ((poolmask & ((1u << POOL_TEX_SLOTS) - 1)) && vk_tex_id_max() - g_pool_tex_base >= NP) VKRED_ONCE(101, "texture pool table: a texture ID (%d) is registered past the table's %d entries: reads of it take the last entry (DM_VK_POOL_N raises the length)", vk_tex_id_max() - g_pool_tex_base, NP);
        if ((poolmask & (1u << POOL_TEX_SLOTS)) && vk_samp_id_max() - g_pool_samp_base >= NS) VKRED_ONCE(102, "sampler pool table: a sampler ID (%d) is registered past the table's %d entries", vk_samp_id_max() - g_pool_samp_base, NS);
    }
    for (int slot = 0; slot < TBL_SLOTS && ntbl; slot++) {
        const dmsv_bind *b = NULL; for (int i = 0; i < s->nbind && i < 96; i++) if (s->tbl[i] == 1 + slot) b = &s->b[i];
        dmsv_bind vb = b ? *b : (dmsv_bind){.kind = RK_TEX_UBO, .desc = RD_SAMPLED, .dim = (uint8_t)(slot == 2 ? 3 : 1), .flags = (uint8_t)(slot == 1 ? 1 : 0), .samp = RS_TEX, .reg = (int16_t)s->tbl_reg, .off = s->tbl_off};
        VkDescriptorImageInfo dummy = {vk_dummy_sampler(), vk_dummy_view(vb.dim, vb.flags & 1, 0, 0, 0), VK_IMAGE_LAYOUT_GENERAL}, *d = ii + (size_t)slot * (size_t)N;
        for (uint32_t k = 0; k < n && b; k++) {
            uint64_t hh = mem_handle(words[2 * k]); int tid = (int)(hh & 0xfffff); const view_t *v = NULL; const tex_t *t = NULL; int fits = 0;
            if (tid && vk_tex_by_id(tid, &v, &t) && t) { int target = v && v->set_target ? v->target : t->d.target;
                fits = vb.dim == 3 ? (t->d.width == t->d.height && (target == 8 || target == 9 || (target == 4 && t->d.depth >= 6))) : (target == 1 || target == 4 || target == 8 || target == 9);
                if (fits && view_num(t, v)) fits = 0; }   // (the arrays declare floating-point texels)
            if (fits) { tex_desc_for(st, &vb, hh, &d[k], 0); if (!d[k].imageView) d[k] = dummy; } else d[k] = dummy;
        }
        for (uint32_t k = b ? n : 0; k < (uint32_t)N; k++) d[k] = dummy;
        w[nwr++] = (VkWriteDescriptorSet){VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, NULL, set, (uint32_t)slot, 0, (uint32_t)N, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, d, NULL, NULL};
    }
    if (nwr) vkUpdateDescriptorSets(V.dev, nwr, w, 0, NULL);
    if (persistent) tbl_persist_save(&persistent_key, set, &sources);
    tbl_sources_free(&sources); g_tbl_full_ns += tick() - build_start;
    g_tblc[c] = (typeof(g_tblc[0])){h, g_set_epoch, gtex, gsamp, n, gpool, kinds, g_pool_tex_base, g_pool_samp_base, set};
    tbl_trace(c, st, s, 0);
    g_tbl_builds++; g_ns_tbl += tick() - t0;
    if (g_tbl_builds <= 4 || !(g_tbl_builds & (g_tbl_builds - 1))) VKLOG("texture tables: %llu sets built (%llu reused, %llu bound, %llu with the pool tables; %.1f ms in all), the longest %u handles of %u bytes bound; stage %d shader %016llx uniform buffer %d +%d: %u handles", (unsigned long long)g_tbl_builds, (unsigned long long)g_tbl_hits, (unsigned long long)g_tbl_binds, (unsigned long long)g_pool_builds, g_ns_tbl / 1e6, g_tbl_max_n, g_tbl_max_bytes, st, (unsigned long long)(s ? s->key : 0), s ? s->tbl_reg : -1, s ? s->tbl_off : -1, n);
    return set;
}

// ---------------- draws ----------------
// What the draws of the command buffer being recorded set last: a draw that would set the same again records nothing. Vulkan keeps dynamic state and
// bindings for the whole command buffer, across render passes and across pipelines that declare the same dynamic state (every one of ours, build_pipeline);
// a new command buffer (g_cmd_gen, gfx_vk_core.c) starts empty. Xenoblade Chronicles 3 docked: ~1,900 draws a frame, each with 10 dynamic-state calls and
// its binds, most the same as the draw before; the driver's time grows with what is recorded (vkEndCommandBuffer too). DM_VK_X_NOBC=1: everything every draw
extern uint64_t g_cmd_gen;
static struct { uint64_t gen; VkViewport vp; VkRect2D sc; float bc[4]; int sm[2], sw[2], sr[2], line; p_poly_offset po; VkPipelineLayout lay; uint32_t nsets, ndofs;
                VkDescriptorSet sets[VK_NSETS], tbl; uint32_t dofs[3 * 96]; VkBuffer vb[17]; VkDeviceSize vo[17]; VkBuffer ib; VkDeviceSize io; VkIndexType it; } g_bc;

static int bc_fresh(void) { if (g_bc.gen != g_cmd_gen || g_x_nobc || g_bc_stale) { g_bc_stale = 0; memset(&g_bc, 0, sizeof g_bc); g_bc.gen = g_cmd_gen; g_bc.it = (VkIndexType)-1; return 1; } return 0; }
#define BC_SAME(field, val) (!fresh && !memcmp(&g_bc.field, &(val), sizeof g_bc.field))
static void bc_index(VkCommandBuffer cb, VkBuffer b, VkDeviceSize o, VkIndexType t) {
    int fresh = bc_fresh(); if (!fresh && g_bc.ib == b && g_bc.io == o && g_bc.it == t) return;
    vkCmdBindIndexBuffer(cb, b, o, t); g_bc.ib = b; g_bc.io = o; g_bc.it = t;
}
static void set_dynamic(VkCommandBuffer cb) {
    int fresh = bc_fresh();
    float x = S.vp[0][0], y = S.vp[0][1], w = S.vp[0][2], h = S.vp[0][3];
    if (w == 0 && h == 0) { w = (float)S.pass_w; h = (float)S.pass_h; }
    VkViewport vp = {x, y, w, h, S.dnear, S.dfar};
    if (g_lower_left && !g_flipy) { vp.y = (float)S.pass_h - y; vp.height = -h; }   // lower-left viewports count y from the bottom
    else if (!g_lower_left && g_flipy) { vp.y = y + h; vp.height = -h; }        // clip space y up over the same top-left rectangle
    if (!g_depth_zero && !V.has_depth_clip_ctl) { /* GL depth convention without clip control: left as is (rare) */ }
    // Viewport swizzle (Absolum's FNA3D sets NEGATIVE_Y every frame; Advance Wars 1+2's Unity sets it too): NEGATIVE_Y negates clip y, the same as the viewport flipped about
    // its centre; facing is decided after the viewport transform in Vulkan as on NVN, so the winding stays. Other swizzles are drawn unswizzled (nvn_mtl.m does the same)
    { uint32_t sw; memcpy(&sw, S.vp_sw[0], 4);
      if (sw == 0x100) { vp.y += vp.height; vp.height = -vp.height; }
      else if (sw) VKRED_ONCE(81, "viewport swizzle %d,%d,%d,%d not implemented (drawn unswizzled)", S.vp_sw[0][0], S.vp_sw[0][1] ^ 2, S.vp_sw[0][2] ^ 4, S.vp_sw[0][3] ^ 6); }
    if (vp.minDepth < 0) vp.minDepth = 0; if (vp.maxDepth > 1) vp.maxDepth = 1;
    if (!BC_SAME(vp, vp)) { vkCmdSetViewport(cb, 0, 1, &vp); g_bc.vp = vp; }
    VkRect2D sc = scissor_rect(); if (!BC_SAME(sc, sc)) { vkCmdSetScissor(cb, 0, 1, &sc); g_bc.sc = sc; }
    { int32_t vx0 = (int32_t)fmaxf(vp.x, 0), vy0 = (int32_t)fmaxf(fminf(vp.y, vp.y + vp.height), 0), vx1 = (int32_t)(vp.x + vp.width), vy1 = (int32_t)fmaxf(vp.y, vp.y + vp.height);   // what this draw can touch
      int32_t x0 = sc.offset.x > vx0 ? sc.offset.x : vx0, y0 = sc.offset.y > vy0 ? sc.offset.y : vy0, x1 = sc.offset.x + (int32_t)sc.extent.width < vx1 ? sc.offset.x + (int32_t)sc.extent.width : vx1, y1 = sc.offset.y + (int32_t)sc.extent.height < vy1 ? sc.offset.y + (int32_t)sc.extent.height : vy1;
      if (x0 < g_pass_bb[0]) g_pass_bb[0] = x0; if (y0 < g_pass_bb[1]) g_pass_bb[1] = y0; if (x1 > g_pass_bb[2]) g_pass_bb[2] = x1; if (y1 > g_pass_bb[3]) g_pass_bb[3] = y1; }
    if (!BC_SAME(bc, S.blendc)) { vkCmdSetBlendConstants(cb, S.blendc); memcpy(g_bc.bc, S.blendc, sizeof g_bc.bc); }
    if (!BC_SAME(sm, S.smask)) { vkCmdSetStencilCompareMask(cb, VK_STENCIL_FACE_FRONT_BIT, (uint32_t)S.smask[0]); vkCmdSetStencilCompareMask(cb, VK_STENCIL_FACE_BACK_BIT, (uint32_t)S.smask[1]); memcpy(g_bc.sm, S.smask, sizeof g_bc.sm); }
    if (!BC_SAME(sw, S.swmask)) { vkCmdSetStencilWriteMask(cb, VK_STENCIL_FACE_FRONT_BIT, (uint32_t)S.swmask[0]); vkCmdSetStencilWriteMask(cb, VK_STENCIL_FACE_BACK_BIT, (uint32_t)S.swmask[1]); memcpy(g_bc.sw, S.swmask, sizeof g_bc.sw); }
    if (!BC_SAME(sr, S.sref)) { vkCmdSetStencilReference(cb, VK_STENCIL_FACE_FRONT_BIT, (uint32_t)S.sref[0]); vkCmdSetStencilReference(cb, VK_STENCIL_FACE_BACK_BIT, (uint32_t)S.sref[1]); memcpy(g_bc.sr, S.sref, sizeof g_bc.sr); }
    if (!BC_SAME(po, S.pofs)) { vkCmdSetDepthBias(cb, S.pofs.units, S.pofs.clamp, S.pofs.factor); g_bc.po = S.pofs; }
    if (fresh || !g_bc.line) { vkCmdSetLineWidth(cb, 1.0f); g_bc.line = 1; }
}
static int g_no_cull = -1;
// When the driver refuses a submission (vkEndCommandBuffer), the next ones find the draw it refuses: only draws below a limit are recorded, the limit
// bisected between a submission that passed and one that failed; the draw found is logged with its whole state (vk_bisect_result, gfx_vk_core.c)
int g_bis_on, g_bis_lo, g_bis_hi, g_bis_trace = -1, g_bis_var;   // g_bis_var: the change tried on the found draw (1 depth bias values 0, 2 no depth bias, 3 dummy textures, 4 vertex offset 0, 5 zero vertex buffer, 6 a new pass)
static int draw_allowed(void) { int i = g_draw_i++; if (i == g_bis_trace) return g_bis_var == 7 ? 0 : 2;
    return !g_bis_on || i < (g_bis_lo + g_bis_hi) / 2; }
void vk_bisect_reset_count(void) { g_draw_i = 0; }
static int dbg_draw(void) { static int v = -1; if (v < 0) { const char *e = getenv("DM_VK_DBG_DRAW"); v = e ? atoi(e) : 0; } return v; }   // debugging: 1 no draw calls, 2 no vertex buffers, 4 no descriptor sets
// a transfer read of 4 bytes of each buffer the addresses are in (outside a render pass, between barriers): see prepare_draw's runs of rasterizer discard
static void rd_read(const uint64_t *a, int n) {
    VkCommandBuffer cb = vk_cmd(); VkBuffer seen[16]; int ns = 0;
    for (int i = 0; i < n; i++) { VkBuffer b, db; VkDeviceSize o, dof; if (!vk_buf_at(a[i], 4, &b, &o)) continue;
        int dup = 0; for (int j = 0; j < ns; j++) dup |= seen[j] == b; if (dup) continue; seen[ns++] = b;
        vk_scratch(4, 4, &db, &dof); VkBufferCopy r = {o, dof, 4}; vk_img_barrier_all(cb); vkCmdCopyBuffer(cb, b, db, 1, &r); vk_img_barrier_all(cb); }
}
// everything a draw needs bound; 0 when the draw cannot be done (logged once per reason)
static int fs_buffers_before_draw(uint64_t key, VkCommandBuffer *cb) {
    if (!g_cs_buffers_n && !g_cs_images_n) return 1;
    // STORE preceding draws, copy bound native buffers outside rendering, then LOAD the attachments.
    // Do not land deferred uniforms: the selected descriptors may refer to their shadows.
    vk_end_pass(); *cb = vk_cmd();
    cs_trace_buffers_copy(*cb, key, 0);
    if (g_cs_images_n) cs_trace_images_copy(*cb, key, 0);
    g_cs_buffers_n = g_cs_buffers_reserved = 0;
    g_cs_images_n = g_cs_images_copies = 0;
    if (!begin_pass() || !pass_commit()) return 0;
    *cb = vk_cmd(); return 1;
}
static int pass_dump_draw_initial(VkCommandBuffer *cb) {
    if (g_pd_draw_before || !pass_dump_draw_ready()) return 1;
    g_pd_draw_before = 1;
    pass_dump_draw_checkpoint("before", 0, 0, 0, 0, 0, 0);
    if (!begin_pass() || !pass_commit()) return 0;
    *cb = vk_cmd(); return 1;
}
// A shader that writes memory: a store through a pointer into a storage buffer, a uniform block (only BufferBlock storage can be written) or an image, an image write, an
// atomic other than a load, a memory copy into such a pointer (SPIR-V scanned once). Without its SPIR-V: assumed to write
static int spv_wmem(const uint32_t *w, uint32_t n) {
    if (!w || n < 5 || w[0] != 0x07230203) return 1;
    uint32_t bound = w[3]; uint8_t *mp = calloc(bound ? bound : 1, 1); if (!mp) return 1;   // 1: a pointer type into memory, 2: a pointer into memory
    int wr = 0;
    for (uint32_t i = 5; i < n && !wr;) {
        uint32_t op = w[i] & 0xffff, wc = w[i] >> 16; if (!wc || i + wc > n) break;
        const uint32_t *o = w + i;
        if (op == 32 && wc >= 4 && o[1] < bound && (o[2] == 2 || o[2] == 11 || o[2] == 12 || o[2] == 5349)) mp[o[1]] = 1;   // OpTypePointer: Uniform, Image, StorageBuffer, PhysicalStorageBuffer
        else if ((op == 59 || op == 65 || op == 66 || op == 67 || op == 60 || op == 55 || op == 57 || op == 61 || op == 83 || op == 169 || op == 245) && wc >= 3 && o[1] < bound && o[2] < bound && mp[o[1]] == 1) mp[o[2]] = 2;   // OpVariable, access chains, OpImageTexelPointer; a pointer as a function's parameter or result, loaded, copied, selected
        else if (op == 62 && wc >= 3 && o[1] < bound && mp[o[1]] == 2) wr = 1;   // OpStore
        else if (op == 63 && wc >= 3 && o[1] < bound && mp[o[1]] == 2) wr = 1;   // OpCopyMemory
        else if (op == 99) wr = 1;   // OpImageWrite
        else if ((op >= 228 && op <= 242) || op == 5614 || op == 5615 || op == 6035) wr = 1;   // atomics but OpAtomicLoad (227)
        i += wc;
    }
    free(mp); return wr;
}
static int shader_wmem(vkshader_t *s) { if (!s) return 0; if (!s->wmem) s->wmem = spv_wmem(s->spv, s->nspv) ? 1 : 2; return s->wmem == 1; }
static int order_feedback(int nst) {   // a stage of the draw samples one of the open pass's attachments (its images by their storage)
    for (int st = 0; st < nst; st++) {
        if (!S.sh[st]) continue;
        if (g_stn[st] > 32) return 1;
        for (int j = 0; j < g_stn[st]; j++) for (int a = 0; a < 9; a++) { const vkimg_t *im = P_att[a].im;
            if (im && im->pool == g_stimg[st][j].pool && g_stimg[st][j].off < im->off + im->size && im->off < g_stimg[st][j].off + g_stimg[st][j].size) return 1; }
    }
    return 0;
}
static int prepare_draw(int prim, int indexed) {
    if (g_mrt_active) mrt_abort(); // a previous selected draw returned before its GPU command
    // DM_VK_CB_DRAWS (read every submission, default 128, 0 = never): the replay is cut every that many draws and what is recorded goes to the GPU at once
    // (vk_cmd_split), so the GPU works on the frame while the rest is replayed instead of after all of it: NEO's Scramble Crossing 36.7 → 15.7 ms a frame
    g_cb_draws++;   // the cut itself happens between passes (begin_pass); inside one only with DM_VK_PASS_SPLIT (below): a pass cut in two stores and reloads its attachments
    g_tracing = g_draw_i == g_bis_trace && g_bis_trace >= 0;
    // the change being tried applies to the found draw and every later one (the later draws are refused too: changing one draw alone proves nothing)
    int var = g_bis_trace >= 0 && g_draw_i >= g_bis_trace ? g_bis_var : 0; if (g_bis_var) g_tracing = 0;   // full logs only the first time
    g_var_now = 0;
    if (var == 6) vk_end_pass();   // a new pass for this draw
    if (g_noatt_open && S.pass_open) vk_end_pass();   // a pass with no attachments holds one draw (noatt_on): the next pass begins with a barrier after it
    // A program may have no fragment stage (a depth-only draw: NVN binds by stage mask, and Metroid Dread draws about nine a frame so in its intro; nvn_mtl.m makes them vertex-only pipelines):
    // drawn with no fragment shader and no colour written (Vulkan leaves colour outputs undefined without one), as before only rasterizer-discard draws were
    if (!S.sh[0]) { g_vkst.skip_prog++; VKRED_ONCE(50, "draw with a program whose shaders are not in the bundle: skipped"); return 0; }
    if (g_x_nogs && S.sh[2]) return 0;   // DM_VK_X_NOGS=1 (live, A/B): draws with a geometry shader are left out
    for (int i = 0; i < g_x_nskipvs; i++) if (S.sh[0]->key == g_x_skipvs[i]) return 0;   // DM_VK_X_SKIPVS=<key>[,<key>...] (live, up to 8): draws with these vertex shaders are left out
    // A run of draws with rasterizer discard gets a pass of its own: the pass ends where such a run begins and where it ends (and begin_pass puts a barrier
    // before the next). They are vertex shaders that write memory the later draws read: Xenoblade Chronicles 3 transforms its meshes with point draws
    // (724ddce6 then 37d04447, cf66cef4), which share passes with its G-buffer draws. NVN's GPU runs draws in order; Adreno, a tiled GPU, runs the vertex
    // shaders of a pass's draws in its binning pass, and draws read vertices not yet written: thin black slivers over the cliffs and grass in 17-20 of
    // 361 frames, 0 with this (0 also with the draws of 37d04447 and cf66cef4 left out). A pass for each such draw instead, splitting the runs, lost
    // Noah's hair (724ddce6's draws come in pairs in one pass); the run kept together draws it. About +1 ms of replay, +0.1 ms of GPU a frame, 33.2 ms
    // frames held. DM_VK_X_RDKEEP=1 (live, A/B): they stay in the pass
    // Where a run ends, the memory its shaders wrote is read by a transfer (rd_read): with barriers alone, the draws of the next passes still met vertices the
    // run had not written yet. Xenoblade Chronicles 3 sways its grass with point draws (5f6491f9 for its depth pre-pass, c0882609 for its G-buffer) whose records
    // the grass cards draw from, the G-buffer's with depth EQUAL against the pre-pass: whole blades failed the test and stayed black, a different few each frame
    // (the grass flicker, stable on the Switch). Grass blades the G-buffer never wrote: 900-2,300 a frame, 0 with this (8 dumps); flicker on screen 4.2-7.7
    // spiking blocks a frame, 0.0-0.5. Everything the two passes read matched (records over 60 frames, instances, index data, shader code: the G-buffer's
    // vertex shader drawing the pre-pass too changed nothing), and a full GPU wait after each submission, explicit barrier bits or a barrier to vertex input
    // did not help; drawing from a copy of the records did, as did a command buffer cut or a transfer read of the written buffer at either end of the run.
    // About +0.45 ms of GPU a frame, 33.2 ms frames held. DM_VK_RD_TOUCH=0 (live, A/B): no read
    { static int prev; static uint64_t wa[16]; static int nwa; int rd = S.rast_discard; if (!g_x_rdkeep && rd != prev && S.pass_open) vk_end_pass();
      if (rd) { if (!prev) nwa = 0; for (int r = 0; r < 16 && nwa < 16; r++) { uint64_t x = S.ssbo[0][r].addr; int have = 0; for (int i = 0; i < nwa; i++) have |= wa[i] == x; if (x && !have) wa[nwa++] = x; } }
      else if (prev && g_rd_touch && !S.pass_open) rd_read(wa, nwa);
      prev = rd; }
    // DM_VK_PASS_SPLIT=n (live, 0 = off): a pass is ended after every n of its draws and begun again on the same attachments (every load op LOAD: clears folded into
    // the first part's load ops are not run again), with a cut before it (begin_pass, g_psplit_now), so its first parts reach the GPU while the replay records the rest:
    // DM_VK_CB_DRAWS cuts only between passes, and Unicorn Overlord's battle map draws ~2,030 of its ~3,546 draws in one 1280x720 pass (13.3 ms of GPU, all after its
    // replay: the 15.5 ms GPU tail). Not inside a run of rasterizer discard (it stays in one pass, see above), and only a pass without depth: a pass that goes on with a
    // depth buffer begun in an earlier command buffer meets Adreno's stale LRZ (begin_pass). DM_VK_PASS_SPLIT_DEPTH=1 (live, A/B) splits those too, with begin_pass's
    // write-back of a depth texel (lrz_touch) before the second part. The pipeline, dynamic state, sets and buffers are bound again (vk_end_pass, g_cmd_gen); deferred
    // uniform updates stay deferred (the draws read their shadows), as at any cut. Only while vk_cmd_split can still begin a part: otherwise the pass would be cut for nothing
    if (g_psplit > 0 && g_x_cbdraws > 0 && S.pass_open == 1 && g_pass_draws >= (uint64_t)g_psplit && (uint64_t)g_cb_draws >= part_pace((uint64_t)g_psplit) && !S.rast_discard && (!S.daspect || g_psplit_depth)) {
        if (vk_cmd_parts_left() > 0) { { static int n; if (n++ < 3) VKLOG("pass split: pass %d (%ux%u) after %llu draws, depth %d", g_sub_pass - 1, S.pass_w, S.pass_h, (unsigned long long)g_pass_draws, S.dfmt); }
            vk_end_pass(); g_psplit_now = 1; } else vk_cmd_split_refused(1); }
    else if (g_psplit > 0 && S.pass_open == 1 && g_pass_draws == (uint64_t)g_psplit && S.daspect && !g_psplit_depth) { static int n; if (n++ < 3) VKLOG("pass split: pass %d (%ux%u) has depth (format %d): not split", g_sub_pass - 1, S.pass_w, S.pass_h, S.dfmt); }
    for (int st = 0; st < 3; st++) if (!(st == 1 && S.rast_discard)) stage_storage_prepare(st, S.sh[st]);
    if (g_order_pending) { g_order_pending = 0; g_order_check = S.pass_open == 1; }   // an order-only barrier kept the pass open: decided below, once the draw's images are known
    if (!begin_pass() || !mrt_prepare(g_claim_draw.count, g_claim_draw.instances, prim, g_claim_draw.indirect, indexed) || !pass_commit()) { g_vkst.skip_rt++; return 0; }
    if (g_x_tracepass >= 0 && g_sub_pass - 1 == g_x_tracepass) g_tracing = 1;
    uint64_t t0 = tick();
    VkCommandBuffer cb = vk_cmd();
    if (g_no_cull < 0) g_no_cull = getenv("DM_VK_NOCULL") != NULL;
    pkey k; memset(&k, 0, sizeof k);
    for (int i = 0; i < 3; i++) k.key[i] = S.sh[i] ? S.sh[i]->key : 0;
    if (S.sh[0] && S.sh[1] && !S.rast_discard) for (int i = 0; i < g_x_nflat; i++) if (k.key[0] == g_x_flat[i]) k.flat = (uint32_t)g_x_flat_mode[i] & 127;
    k.topo = topo(prim); k.restart = S.prim_restart && (k.topo == VK_PRIMITIVE_TOPOLOGY_LINE_STRIP || k.topo == VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP || k.topo == VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN) && indexed;
    k.raster_discard = S.rast_discard; k.alt = (uint32_t)(g_spv_alt && (!g_spv_key || k.key[0] == g_spv_key || k.key[1] == g_spv_key || k.key[2] == g_spv_key)); k.cull = g_no_cull ? 0 : S.poly.cull;
    // counter-clockwise front faces in Vulkan's framebuffer: Vulkan's clip space has y down, Metal's up (nvn_mtl.m: CCW when (front == CCW) != lower left), and the
    // lower-left origin mode flips the viewport again, so CCW when (front == CCW) == lower left. Taken as given, upper-left games (NEO) culled their ground and walls
    k.front = ((S.poly.front == NVN_FRONT_FACE_CCW) == (g_lower_left != 0)) != (g_flipy != 0); k.mode = S.poly.mode; k.depth_clamp = S.depth_clamp;
    k.bias_en = (S.poly.offset_en & (S.poly.mode == 0 ? 1 : S.poly.mode == 1 ? 2 : 4)) != 0;
    if (var == 2) k.bias_en = 0;
    if (var == 8) k.raster_discard = 1;   // no fragment shader in the pipeline
    int has_depth = (S.daspect & VK_IMAGE_ASPECT_DEPTH_BIT) != 0, has_stencil = (S.daspect & VK_IMAGE_ASPECT_STENCIL_BIT) != 0;
    k.dt = has_depth && S.ds.dt; k.dw = S.ds.dw; k.df = S.ds.df; k.st = has_stencil && S.ds.st;
    if (g_depth_func_fs && S.sh[1] && S.sh[1]->key == g_depth_func_fs) k.df = (uint32_t)g_depth_func;
    static int npl; if (g_x_logfs && S.sh[1] && S.sh[1]->key == g_x_logfs && npl++ < 60) VKLOG("logfs: present %llu pass %d (%ux%u, %llu draws before): depth attachment %d test %d (state %d) func %u write %u; depth range %.9g %.9g; viewport %.1f %.1f %.1f %.1f",
        (unsigned long long)g_tex_frame, g_sub_pass - 1, S.pass_w, S.pass_h, (unsigned long long)g_pass_draws, has_depth, k.dt, S.ds.dt, k.df, k.dw, S.dnear, S.dfar, S.vp[0][0], S.vp[0][1], S.vp[0][2], S.vp[0][3]);
    // A depth buffer that a fragment shader wrote through gl_FragDepth: Adreno's early culling (LRZ) then rejects whole triangles of later draws that test it, in some frames
    // and not others, with the same draws, state and depth contents. Hades II downsamples its scene depth into a 640x360 D24S8 with such a shader (Blit2DDepth), then
    // draws its lights into it with the depth test on and no depth write: the lights went out in turn every 24-48 frames (whole halves of light quads missing in pass dumps),
    // and Xenoblade Chronicles 3's cloud dome (its depth downsampled the same way) lost whole triangles. The light shader rewritten to write gl_FragDepth = gl_FragCoord.z
    // (the same depth, tested after the shader) brought the pulses to 0 in 5 recordings, against 8-21; neither writing the depth image back by transfer (one texel or the
    // whole image) nor waiting for the GPU after each submission changed them. So draws that test such a buffer (written in the last two presents) use that variant.
    // Not with a depth bias (gl_FragCoord.z need not carry it). DM_VK_FDZ=0 (live): never
    { vkimg_t *dim = has_depth ? P_att[8].im : NULL; vkshader_t *fsh = k.raster_discard ? NULL : S.sh[1];
      if (dim && fsh && fsh->depth_repl && k.dt && k.dw) dim->fdz_frame = g_tex_frame + 1;
      k.fdz = g_fdz && dim && fsh && !fsh->depth_repl && k.dt && !k.bias_en &&
          ((dim->fdz_frame && g_tex_frame + 1 - dim->fdz_frame <= 2) || (g_fdz_fs && fsh->key == g_fdz_fs)); }
    k.sf[0] = S.ds.sf_f; k.sf[1] = S.ds.sf_b; k.sfail[0] = S.ds.fail_f; k.szfail[0] = S.ds.zfail_f; k.szpass[0] = S.ds.zpass_f; k.sfail[1] = S.ds.fail_b; k.szfail[1] = S.ds.zfail_b; k.szpass[1] = S.ds.zpass_b;
    // vertex input: every location the vertex shader reads
    vkshader_t *vsh = S.sh[0]; uint32_t in_mask = vsh ? vsh->local[0] : 0, int_mask = vsh ? vsh->local[1] | vsh->local[2] : 0, sint_mask = vsh ? vsh->local[1] : 0;
    uint32_t used_streams = 0; int need_zero = 0;
    for (int loc = 0; loc < 16; loc++) {
        if (!(in_mask >> loc & 1)) continue;
        va_st a; memcpy(&a, &S.vattr[loc], 4);
        VkFormat f = a.fmt ? vertex_format(a.fmt, int_mask >> loc & 1, sint_mask >> loc & 1) : VK_FORMAT_UNDEFINED;
        if (f == VK_FORMAT_UNDEFINED || !S.vbuf[a.stream].address) { k.attr[k.nattr++] = (typeof(k.attr[0])){(uint32_t)loc, 16, int_mask >> loc & 1 ? VK_FORMAT_R32G32B32A32_UINT : VK_FORMAT_R32G32B32A32_SFLOAT, 0}; need_zero = 1; continue; }
        k.attr[k.nattr++] = (typeof(k.attr[0])){(uint32_t)loc, a.stream, f, a.offset};
        used_streams |= 1u << a.stream;
    }
    for (int s = 0; s < 16; s++) if (used_streams >> s & 1) k.bind[k.nbind++] = (typeof(k.bind[0])){(uint32_t)s, S.vstream[s].stride, S.vstream[s].divisor ? 1u : 0u, S.vstream[s].divisor};
    if (need_zero) k.bind[k.nbind++] = (typeof(k.bind[0])){16, 0, 0, 0};
    // attachments and blending
    k.ncol = (uint32_t)S.ncol; k.dfmt = S.dfmt; k.samples = S.samples ? S.samples : 1; k.a2c = S.ms.a2c;
    k.logic_en = S.color.logic_op != 3 && S.color.logic_op != 0 && 0;   // logic ops: not on mobile render targets in version 1
    k.logic_op = S.color.logic_op;
    for (int i = 0; i < S.ncol; i++) {
        k.cfmt[i] = S.cfmt[i];
        k.wmask[i] = S.cfmt[i] && S.sh[1] ? (S.chmask.mask >> (4 * i)) & 0xf : 0;   // no fragment shader: no colour written
        // Components outside the fragment shader's output map (SPH word 18; port of upstream 51ecd693b): the GPU leaves them as they were, the translator does not store them,
        // and Vulkan would write them undefined. DM_VK_OMAP=0 (start-up): no such mask
        { static int om = -1; if (om < 0) { const char *e = getenv("DM_VK_OMAP"); om = !(e && *e == '0'); }
          if (om && k.wmask[i] && S.sh[1]->omap_known) k.wmask[i] &= S.sh[1]->omap >> (4 * i) & 0xf; }
        const vkfmt_t *f = S.rt_has_col[i] ? vk_fmt(S.rt_col[i].d.format) : NULL;
        if (((S.color.blend_en >> i) & 1) && f && !f->is_int) {
            k.blend_en |= 1u << i;
            k.bl[i].src = S.blend[i].src_rgb; k.bl[i].dst = S.blend[i].dst_rgb; k.bl[i].srca = S.blend[i].src_a; k.bl[i].dsta = S.blend[i].dst_a; k.bl[i].eq = S.blend[i].eq_rgb; k.bl[i].eqa = S.blend[i].eq_a;
        }
    }
    if (S.ncol > 1 && P_att[0].im == &g_s0_image) { k.wmask[0] = 0; k.blend_en &= ~1u; }   // slot 0's scratch (DM_VK_SLOT0_FILL): nothing written there
    if (g_mrt_active) { k.wmask[1] = 0xf; k.blend_en &= ~2u; } // probe output is independent of original output map/blend state
    // Dual-source blending: NVN blends colour target 0 with the shader's output 1 as its second source (factors SRC1_*, 0x10-0x13), the translator declares that output at
    // location 1, and Vulkan takes the second source only from location 0, index 1 (with location 1 the blend reads undefined values: Pokémon Legends: Arceus's post effects,
    // shader ha_post_effects variation 4, washed the whole picture out; the validation layer named it). Such a pipeline takes the fragment shader's dual-source variant
    #define SRC1F(x) ((x) >= 0x10 && (x) <= 0x13)
    k.dual = (k.blend_en & 1) && S.sh[1] && !k.raster_discard && (SRC1F(k.bl[0].src) || SRC1F(k.bl[0].dst) || SRC1F(k.bl[0].srca) || SRC1F(k.bl[0].dsta));
    #undef SRC1F
    // Maxwell's alpha test (nvnColorStateSetAlphaTest / SetAlphaRef): colour 0's alpha against the reference, the fragment dropped when the comparison fails. Not when colour 0
    // is an integer target or absent (as OpenGL and nvn_mtl.m atest_key). Donkey Kong Bananza's Skills grid: ui2d clears each icon capture's rectangle of the UI target
    // with a quad of alpha 0 and blending off under GREATER 0 before copying the rectangle; drawn without the test, every cell was an opaque black square
    { uint32_t af = S.color.alpha_func; const vkfmt_t *f0 = S.ncol && S.rt_has_col[0] ? vk_fmt(S.rt_col[0].d.format) : NULL;
      if (af >= 1 && af <= 7 && S.sh[1] && !k.raster_discard && S.cfmt[0] && f0 && !f0->is_int) { k.atest = af; if (af != 1) memcpy(&k.aref, &S.alpha_ref, 4); if (k.aref == 0x80000000u) k.aref = 0; } }
    k.clip_neg1 = !g_depth_zero;
    // layout and pipeline
    VkDescriptorSetLayout dsl[VK_NSETS] = {vk_tbl_dsl()}; int nsets = 1;   // set 0: the texture table (the same layout in every graphics pipeline layout, so a bound table set outlives pipeline changes)
    for (int st = 0; st < 3; st++) if (S.sh[st] && !(st == 1 && k.raster_discard)) { dsl[S.sh[st]->set] = S.sh[st]->dsl; if (S.sh[st]->set + 1 > nsets) nsets = S.sh[st]->set + 1; }
    // the last draw's set layouts and what vk_pipeline_layout made of them (it fills absent stages with the empty layout): its lock and hash took ~10% of the
    // replay thread on Unicorn Overlord's battle map. Pipeline layouts are never destroyed
    static VkDescriptorSetLayout l_in[VK_NSETS], l_out[VK_NSETS]; static int l_n = -1; static VkPipelineLayout l_lay; VkPipelineLayout layout;
    if (nsets == l_n && !memcmp(dsl, l_in, sizeof l_in)) { memcpy(dsl, l_out, sizeof l_out); layout = l_lay; }
    else { memcpy(l_in, dsl, sizeof l_in); l_n = nsets; layout = l_lay = vk_pipeline_layout(dsl, nsets); memcpy(l_out, dsl, sizeof l_out); }
    gs_claim_prepare(S.sh[2]);
    int capture_buffers = S.sh[1] && !S.rast_discard && fs_buffer_match(S.sh[1]->key, S.pass_w, S.pass_h, g_claim_draw.count, g_claim_draw.indirect || g_fs_buffer_mesh);
    if (capture_buffers) { g_cs_buffers_n = g_cs_buffers_reserved = 0; g_cs_images_n = g_cs_images_copies = 0; g_fs_buffer_collect = 1; }
    // descriptor sets first: the views they are given decide which shader variants the pipeline takes (integer texels, vk_shader_module)
    g_var_now = var;
    VkDescriptorSet sets[VK_NSETS] = {0}, tbl = VK_NULL_HANDLE;
    uint32_t dofs[VK_NSETS][96], ndofs = 0, dall[3 * 96];
    uint64_t t1d = tick();
    vkshader_t *ts = NULL; int tst = 0; unsigned poolmask = 0;
    for (int st = 0; st < 3; st++) if (S.sh[st] && !(st == 1 && k.raster_discard)) {
        uint64_t m0 = g_set_memo; int n0 = g_stn[st]; g_stn[st] = 0; g_coll_st = st;   // the images it is collected with (vk_watch); a memoized set keeps the last ones
        sets[S.sh[st]->set] = stage_set(st, S.sh[st], dofs[S.sh[st]->set], NULL); k.um[st] = g_num_um; k.im[st] = g_num_im;
        g_coll_st = -1; if (g_set_memo != m0) g_stn[st] = n0;
        if (S.sh[st]->ntbl && !ts) { ts = S.sh[st]; tst = st; } poolmask |= tbl_kinds(S.sh[st]) >> TBL_SLOTS; }   // (two stages with tables of different sources: the first stage's; not seen)
    g_fs_buffer_collect = 0;
    pool_prefetch();
    if (ts || poolmask) tbl = tbl_set(ts ? tst : 0, ts, poolmask);   // set 0: the handle source and the union of every active stage's pool kinds
    if (g_order_check) {   // the order-only barrier before this draw (see g_order_keep): kept, or the pass ends here with it
        g_order_check = 0; vkshader_t *fs = k.raster_discard ? NULL : S.sh[1];
        if (g_pass_wmem || shader_wmem(S.sh[0]) || shader_wmem(fs) || shader_wmem(S.sh[2]) || order_feedback(3)) {
            g_order_ended++; vk_end_pass(); vk_img_barrier_all(vk_cmd());
            if (!begin_pass() || !mrt_prepare(g_claim_draw.count, g_claim_draw.instances, prim, g_claim_draw.indirect, indexed) || !pass_commit()) { g_vkst.skip_rt++; return 0; }
            cb = vk_cmd();
        } else g_order_kept++;
    }
    g_pass_wmem |= shader_wmem(S.sh[0]) | shader_wmem(k.raster_discard ? NULL : S.sh[1]) | shader_wmem(S.sh[2]);
    uint64_t tdesc = tick() - t1d; g_vkst.ns_desc += tdesc; if (g_prof_on) g_prof_ns[5] += tdesc;
    if (g_tracing) {
        VKLOG("  pipeline: topology %u restart %u discard %u cull %u front %u mode %u clamp %u bias %u; depth test %u write %u func %u stencil %u; attributes %u bindings %u; colour %u (format %u) samples %u a2c %u; blend enable %#x logic %u/%u; fragment depth variant %u (depth written by a shader at present %llu, now %llu)",
              k.topo, k.restart, k.raster_discard, k.cull, k.front, k.mode, k.depth_clamp, k.bias_en, k.dt, k.dw, k.df, k.st, k.nattr, k.nbind, k.ncol, k.cfmt[0], k.samples, k.a2c, k.blend_en, k.logic_en, k.logic_op,
              k.fdz, has_depth && P_att[8].im ? (unsigned long long)P_att[8].im->fdz_frame : 0ull, (unsigned long long)g_tex_frame + 1);
        if (k.st) VKLOG("  stencil: func %u/%u fail %u/%u zfail %u/%u pass %u/%u; compare mask %#x/%#x write mask %#x/%#x reference %#x/%#x", k.sf[0], k.sf[1], k.sfail[0], k.sfail[1], k.szfail[0], k.szfail[1], k.szpass[0], k.szpass[1],
                        (unsigned)S.smask[0], (unsigned)S.smask[1], (unsigned)S.swmask[0], (unsigned)S.swmask[1], (unsigned)S.sref[0], (unsigned)S.sref[1]);
        for (uint32_t i = 0; i < k.ncol; i++) VKLOG("  blend %u: src %#x dst %#x eq %u, alpha src %#x dst %#x eq %u, write mask %#x", i, k.bl[i].src, k.bl[i].dst, k.bl[i].eq, k.bl[i].srca, k.bl[i].dsta, k.bl[i].eqa, k.wmask[i]);
        for (int st_ = 0; st_ < 2; st_++) for (int r_ = 0; r_ < 16; r_++) if (S.ssbo[st_][r_].addr) VKLOG("  guest storage buffer: stage %d register %d address %#llx size %#llx", st_, r_, (unsigned long long)S.ssbo[st_][r_].addr, (unsigned long long)S.ssbo[st_][r_].size);
        { const vkfmt_t *f0 = S.ncol && S.rt_has_col[0] ? vk_fmt(S.rt_col[0].d.format) : NULL; VKLOG("  colour state: nvn blend enables %#x, target 0 bound %d nvn format %#x (vk %d, integer %d); nvn blend 0: src %#x dst %#x eq %u / %#x %#x %u; constants %g %g %g %g", (unsigned)S.color.blend_en, S.ncol ? S.rt_has_col[0] : 0, S.ncol ? (unsigned)S.rt_col[0].d.format : 0, f0 ? (int)f0->vk : -1, f0 ? f0->is_int : -1, (unsigned)S.blend[0].src_rgb, (unsigned)S.blend[0].dst_rgb, (unsigned)S.blend[0].eq_rgb, (unsigned)S.blend[0].src_a, (unsigned)S.blend[0].dst_a, (unsigned)S.blend[0].eq_a, S.blendc[0], S.blendc[1], S.blendc[2], S.blendc[3]); }
        for (uint32_t i = 0; i < k.nattr; i++) VKLOG("  attribute location %u binding %u format %u offset %u", k.attr[i].loc, k.attr[i].binding, k.attr[i].fmt, k.attr[i].off);
        for (uint32_t i = 0; i < k.nbind; i++) VKLOG("  binding %u stride %u rate %u divisor %u", k.bind[i].binding, k.bind[i].stride, k.bind[i].rate, k.bind[i].divisor);
    }
    if(g_mrt_active&&g_mrt_observer){
        uint32_t extra=(S.sh[1]->in_locs&~S.sh[0]->out_locs);
        if(k.um[0]||k.im[0]||k.um[1]||k.im[1]||extra||k.fdz||k.dual||k.atest||k.flat||k.logic_en){
            VKLOG("Flat observer: native shader variants needed; both originals retained");
            mrt_abort();return prepare_draw(prim,indexed);
        }
        k.alt=2; // immutable validated pair, distinct from generic ALT=1; never persisted
    }
    crumb_draw(k.key[0], k.key[1], k.key[2]);
    VkPipeline p = graphics_pipeline(&k, layout);
    uint64_t t1 = tick(); g_vkst.ns_pipe += t1 - t0 - tdesc;
    if (!p) { g_vkst.skip_pipe++; if (capture_buffers) { g_cs_buffers_n = g_cs_buffers_reserved = 0; g_cs_images_n = g_cs_images_copies = 0; } return 0; }
    if (capture_buffers && !fs_buffers_before_draw(S.sh[1]->key, &cb)) { g_vkst.skip_rt++; return 0; }
    if (!pass_dump_draw_initial(&cb)) { g_vkst.skip_rt++; return 0; }
    if (pass_dump_draw_ready()) {
        uint32_t bc[4]; memcpy(bc, S.blendc, sizeof bc);
        VKLOG("pd draw state: present=%llu submission=%llu prepared-draw=%d pipeline=%p keyhash=%016llx keybytes=%zu dt=%u dw=%u df=%u st=%u cull=%u front=%u discard=%u samples=%u blend=%#x logic=%u/%u constants=%08x/%08x/%08x/%08x",
            (unsigned long long)g_tex_frame, (unsigned long long)g_set_epoch, g_draw_i, (void *)p,
            (unsigned long long)fnv(1469598103934665603ull, &k, sizeof k), sizeof k, k.dt, k.dw, k.df, k.st,
            k.cull, k.front, k.raster_discard, k.samples, k.blend_en, k.logic_en, k.logic_op, bc[0], bc[1], bc[2], bc[3]);
        for (uint32_t i = 0; i < k.ncol; i++) VKLOG("pd draw blend: slot=%u format=%u mask=%#x src=%#x dst=%#x eq=%u srca=%#x dsta=%#x eqa=%u",
            i, k.cfmt[i], k.wmask[i], k.bl[i].src, k.bl[i].dst, k.bl[i].eq, k.bl[i].srca, k.bl[i].dsta, k.bl[i].eqa);
        for (int st = 0; st < 3; st++) if (S.sh[st]) VKLOG("pd draw stage: stage=%d key=%016llx in=%#x out=%#x uint=%#llx int=%#llx",
            st, (unsigned long long)k.key[st], S.sh[st]->in_locs, S.sh[st]->out_locs,
            (unsigned long long)k.um[st], (unsigned long long)k.im[st]);
        // Exact initialized key includes every module rewrite/layout/output flag,
        // rather than guessing which flags are material to this diagnostic.
        const unsigned char *raw = (const unsigned char *)&k;
        for (size_t off = 0; off < sizeof k; off += 64) { char hex[129]; size_t n = sizeof k - off; if (n > 64) n = 64;
            for (size_t i = 0; i < n; i++) snprintf(hex + 2 * i, 3, "%02x", raw[off + i]);
            VKLOG("pd draw pkey: offset=%zu hex=%s", off, hex);
        }
    }
    if (g_ubolog_key && g_ubolog_n > 0 && S.sh[0] && S.sh[0]->key == g_ubolog_key) {   // DM_VK_UBO_LOG: the pipeline the draw got, and its key's hash
        uint64_t kh = fnv(1469598103934665603ull, &k, sizeof k); VKLOG("ubo: present %llu pipeline %p key %016llx um %llx/%llx im %llx/%llx", (unsigned long long)g_tex_frame, (void *)p, (unsigned long long)kh,
              (unsigned long long)k.um[0], (unsigned long long)k.um[1], (unsigned long long)k.im[0], (unsigned long long)k.im[1]); }
    g_vkst.draws++; if (++g_pass_draws > g_vkst.max_pass_draws) g_vkst.max_pass_draws = g_pass_draws;
    if (p != S.bound_pipe) { uint64_t tb = g_prof_on ? tick() : 0; vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, p); S.bound_pipe = p; if (g_prof_on) g_prof_ns[4] += tick() - tb; }
    set_dynamic(cb);
    if (var == 1) { vkCmdSetDepthBias(cb, 0, 0, 0); memset(&g_bc.po, 0xff, sizeof g_bc.po); }
    if (pass_dump_draw_ready()) VKLOG("pd draw dynamic: prepared-draw=%d viewport=%a/%a/%a/%a/%a/%a scissor=%d/%d/%u/%u bias=%a/%a/%a stencil-mask=%#x/%#x stencil-write=%#x/%#x stencil-ref=%#x/%#x",
        g_draw_i, g_bc.vp.x, g_bc.vp.y, g_bc.vp.width, g_bc.vp.height, g_bc.vp.minDepth, g_bc.vp.maxDepth,
        g_bc.sc.offset.x, g_bc.sc.offset.y, g_bc.sc.extent.width, g_bc.sc.extent.height,
        var == 1 ? 0. : (double)g_bc.po.units, var == 1 ? 0. : (double)g_bc.po.clamp, var == 1 ? 0. : (double)g_bc.po.factor,
        g_bc.sm[0], g_bc.sm[1], g_bc.sw[0], g_bc.sw[1], g_bc.sr[0], g_bc.sr[1]);

    uint32_t at[VK_NSETS + 1] = {0};   // where each set's dynamic offsets start in dall
    for (int i = 0; i < nsets; i++) { at[i] = ndofs; for (int st = 0; st < 3; st++)   // the dynamic offsets: by set, then binding
        if (S.sh[st] && !(st == 1 && k.raster_discard) && S.sh[st]->set == i) { memcpy(dall + ndofs, dofs[i], sizeof(uint32_t) * (size_t)S.sh[st]->ndyn); ndofs += (uint32_t)S.sh[st]->ndyn; } }
    at[nsets] = ndofs;
    for (int i = 1; i < nsets; i++) if (!sets[i]) sets[i] = vk_alloc_set(dsl[i]);   // an absent stage's set: from the empty layout vk_pipeline_layout filled in (set 0, the table, is bound apart below, only when a stage reads one)
    // DM_VK_SETSPLIT=1 (live): under the same layout, only the run of sets whose handle or dynamic offsets changed is bound again. Adreno's vkCmdBindDescriptorSets
    // took ~22% of the replay thread on Unicorn Overlord's battle map (one load in it), and a draw that moves only its vertex stage's offsets sent the fragment set too
    if (!(dbg_draw() & 4) && g_setsplit && !bc_fresh() && g_bc.lay == layout && g_bc.nsets == (uint32_t)nsets && g_bc.ndofs == ndofs) {
        int lo = -1, hi = -1;
        for (int i = 1; i < nsets; i++) if (g_bc.sets[i] != sets[i] || memcmp(g_bc.dofs + at[i], dall + at[i], sizeof *dall * (at[i + 1] - at[i]))) { if (lo < 0) lo = i; hi = i; }
        if (lo >= 0) { vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, layout, (uint32_t)lo, (uint32_t)(hi - lo + 1), sets + lo, at[hi + 1] - at[lo], dall + at[lo]);
            memcpy(g_bc.sets + lo, sets + lo, sizeof *sets * (size_t)(hi - lo + 1)); memcpy(g_bc.dofs + at[lo], dall + at[lo], sizeof *dall * (at[hi + 1] - at[lo])); } }
    else if (!(dbg_draw() & 4)) { int fresh = bc_fresh();
        if (fresh || g_bc.lay != layout || g_bc.nsets != (uint32_t)nsets || g_bc.ndofs != ndofs || memcmp(g_bc.sets + 1, sets + 1, sizeof *sets * (size_t)(nsets - 1)) || memcmp(g_bc.dofs, dall, sizeof *dall * ndofs)) {
            vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, layout, 1, (uint32_t)(nsets - 1), sets + 1, ndofs, dall);
            g_bc.lay = layout; g_bc.nsets = (uint32_t)nsets; g_bc.ndofs = ndofs; memcpy(g_bc.sets, sets, sizeof *sets * (size_t)nsets); memcpy(g_bc.dofs, dall, sizeof *dall * ndofs); } }
    if (tbl && !(dbg_draw() & 4) && (g_bc.tbl != tbl || g_bc.gen != g_cmd_gen)) { vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, layout, 0, 1, &tbl, 0, NULL); g_bc.tbl = tbl; g_tbl_binds++; }   // the texture table: set 0, kept across pipelines (every graphics layout's set 0 is vk_tbl_dsl)
    // vertex buffers
    for (uint32_t i = 0; i < k.nbind; i++) {
        uint32_t b = k.bind[i].binding; VkBuffer buf; VkDeviceSize off;
        if (b == 16) { buf = vk_zero_buffer(); off = 0; }
        else if ((vk_pool_tag = "vertex", !vk_buf_at(S.vbuf[b].address, 1, &buf, &off))) { buf = vk_zero_buffer(); off = 0; VKRED_ONCE(51, "vertex buffer outside the memory pools"); }
        else if (S.vbuf[b].size) { vkpool_t *vp = vk_pool_peek(S.vbuf[b].address, 1); if (vp && S.vbuf[b].address + S.vbuf[b].size > vp->hi && vp->hi < vp->end) vk_cross(2, S.vbuf[b].address, S.vbuf[b].size, vp); }
        if (g_var_now == 5) { buf = vk_zero_buffer(); off = 0; }
        if (!(dbg_draw() & 2) && b < 17) { int fresh = bc_fresh(); if (fresh || g_bc.vb[b] != buf || g_bc.vo[b] != off) { vkCmdBindVertexBuffers(cb, b, 1, &buf, &off); g_bc.vb[b] = buf; g_bc.vo[b] = off; } }
    }
    if (S.sh[1] && S.ncol && P_att[0].im)
        pass_dump_arm_for_draw(S.sh[1]->key, S.pass_w, S.pass_h, S.cfmt[0], P_att[0].im->layers, k.raster_discard);
    if (S.sh[1] && !k.raster_discard) light_diag_note_draw(S.sh[1]->key, g_draw_i);
    (void)indexed;
    return 1;
}
// quads without an index buffer: one buffer of 0 1 2 0 2 3, 4 5 6 4 6 7, ... for up to QUAD_MAX quads (more are drawn as the first QUAD_MAX)
#define QUAD_MAX 65536u
static VkBuffer quad_index_buffer(uint32_t *nq) {
    static VkBuffer b; static pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;
    if (*nq > QUAD_MAX) { VKRED_ONCE(58, "quad draw of %u quads: the first %u drawn", *nq, QUAD_MAX); *nq = QUAD_MAX; }
    pthread_mutex_lock(&mu);
    if (!b) {
        VkBufferCreateInfo bi = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, NULL, 0, QUAD_MAX * 6 * 4, VK_BUFFER_USAGE_INDEX_BUFFER_BIT, VK_SHARING_MODE_EXCLUSIVE};
        VKCHK(vkCreateBuffer(V.dev, &bi, NULL, &b)); VkMemoryRequirements req; vkGetBufferMemoryRequirements(V.dev, b, &req);
        VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL, req.size, vk_mem_type(req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 0)};
        VkDeviceMemory m; VKCHK(vkAllocateMemory(V.dev, &ai, NULL, &m)); VKCHK(vkBindBufferMemory(V.dev, b, m, 0));
        uint32_t *p; VKCHK(vkMapMemory(V.dev, m, 0, VK_WHOLE_SIZE, 0, (void **)&p));
        static const uint32_t pat[6] = {0, 1, 2, 0, 2, 3};
        for (uint32_t q = 0; q < QUAD_MAX; q++) for (int k = 0; k < 6; k++) p[q * 6 + k] = q * 4 + pat[k];
    }
    pthread_mutex_unlock(&mu);
    return b;
}
// quads: a triangle list of indices made on the CPU (from the game's index buffer for indexed draws)
static int quad_indices(uint32_t count, const uint8_t *src, int itype, int32_t first, VkBuffer *buf, VkDeviceSize *off) {
    uint32_t nq = count / 4; if (!nq) return 0;
    uint32_t *d = vk_scratch((VkDeviceSize)nq * 6 * 4, 4, buf, off);
    for (uint32_t q = 0; q < nq; q++) for (int k = 0; k < 6; k++) {
        static const int pat[6] = {0, 1, 2, 0, 2, 3}; uint32_t i = q * 4 + (uint32_t)pat[k];
        d[q * 6 + k] = !src ? (uint32_t)first + i : itype == 0 ? src[i] : itype == 1 ? ((const uint16_t *)src)[i] : ((const uint32_t *)src)[i];
    }
    return (int)(nq * 6);
}
// ---------------- transform feedback ----------------
// NVN's capture state (OP_TF_BIND / OP_TF_STATE). Each capturing draw (feedback on, a vertex shader translated with Xfb: shader_vk.py xfb_layout) is a Vulkan capture of its own:
// the buffers bound at the bytes written so far, Begin, the draw, End. So no pipeline is bound and no pass ends while a capture is active, and no counter buffer is read back:
// the host counts the bytes, as nvn_mtl.m tf_draw does (vertices the primitives emit × the buffer's stride). Xenoblade Chronicles 3 steps its GPU particles this way (8 shaders,
// up to 9 outputs interleaved in one buffer); before, those draws wrote nothing and the particles kept stale or zero state
static uint32_t xfb_verts(int prim, uint32_t n) {   // vertices a capture writes for n vertices of this primitive (strips and fans are written as lists)
    switch (topo(prim)) {
    case VK_PRIMITIVE_TOPOLOGY_LINE_LIST: return n & ~1u;
    case VK_PRIMITIVE_TOPOLOGY_LINE_STRIP: return n >= 2 ? (n - 1) * 2 : 0;
    case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST: return n / 3 * 3;
    case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP: case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN: return n >= 3 ? (n - 2) * 3 : 0;
    default: return n;
    }
}
// A geometry shader's capture (Xenoblade Chronicles 3's particle update, 293ecebc0136519b: what it emits, 172 bytes a particle) writes a count no host can know: the bytes go to a
// counter buffer, 16 bytes for each control buffer's slot (one u32 a feedback buffer), which the next capture of the sequence starts from and DrawTransformFeedback draws by
// (vkCmdDrawIndirectByteCountEXT). Without it the particle update wrote nothing: no red particles at husks, none rising at a send-off, no glow on far items
enum { TFC = 4096 }; static struct { uint64_t ctl; uint32_t verts; uint8_t gs, ctr_ok, dirty; } g_tf_ctl[TFC];
static VkBuffer g_tfc_buf; static int g_tfc_cur = -1, g_bc_slot = -1;   // the counter buffer; the slot of the open sequence; a DrawTransformFeedback drawn by its counter
// Logical queues can park between any two command records. Persistent guest state
// belongs to the queue; native encoders and descriptor caches belong to a submission.
typedef struct vk_queue_state {
    struct vk_queue_state *next;
    uint64_t serial;
    typeof(S) state;
    p_tf_bind tf[4]; uint64_t tf_off[4];
    int tf_on, tf_cur, rc_skip, tex_base, samp_base;
    uint32_t tf_verts;
    uint64_t rt_pins[9];
} vk_queue_state;
static vk_queue_state *g_vk_queues, *g_vk_active;
static void queue_targets_pin(vk_queue_state *q) {
    for (int i = 0; i < 9; i++) {
        uint64_t serial = i == 8 ? (S.rt_has_dep ? S.rt_dep.serial : 0) :
            (i < S.rt_n && S.rt_has_col[i] ? S.rt_col[i].serial : 0);
        if (serial != q->rt_pins[i]) {
            if (serial) {
                tex_t *dst = i == 8 ? &q->state.rt_dep : &q->state.rt_col[i];
                const tex_t *src = i == 8 ? &S.rt_dep : &S.rt_col[i];
                mtl_tex_snapshot(dst, src);
            }
            mtl_tex_unpin(q->rt_pins[i]); q->rt_pins[i] = serial;
        }
    }
}
void vk_queue_preserve(void) {
    if (!nvn_logical_queues_on() || !g_vk_active) return;
    vk_queue_state *q = g_vk_active;
    queue_targets_pin(q);
    q->state = S;
    // bind_program already resolved all stages into process-lifetime vkshader_t
    // cache entries. Never retain the job-owned program descriptor after that bind.
    q->state.prog = NULL;
    memcpy(q->tf, g_tf, sizeof g_tf); memcpy(q->tf_off, g_tf_off, sizeof g_tf_off);
    q->tf_on = g_tf_on; q->tf_verts = g_tf_verts; q->tf_cur = g_tfc_cur;
    q->rc_skip = g_rc_skip; q->tex_base = g_pool_tex_base; q->samp_base = g_pool_samp_base;
}
void mtl_queue_activate(uint64_t serial) {
    extern uint64_t g_nvn_active_queue;
    g_nvn_active_queue = serial;
    if (!nvn_logical_queues_on() || (g_vk_active && g_vk_active->serial == serial)) return;
    vk_queue_preserve();
    vk_queue_state *q = g_vk_queues;
    while (q && q->serial != serial) q = q->next;
    if (!q) {
        q = calloc(1, sizeof *q); if (!q) die("[vk] queue state allocation failed");
        q->serial = serial; q->tf_cur = -1; q->next = g_vk_queues; g_vk_queues = q;
        state_reset(); q->state = S;
    }
    g_vk_active = q; S = q->state;
    S.prog = NULL; S.pass_open = 0; S.bound_pipe = VK_NULL_HANDLE;
    memcpy(g_tf, q->tf, sizeof g_tf); memcpy(g_tf_off, q->tf_off, sizeof g_tf_off);
    g_tf_on = q->tf_on; g_tf_verts = q->tf_verts; g_tfc_cur = q->tf_cur; g_tf_in_pass = 0;
    g_rc_skip = q->rc_skip; g_pool_tex_base = q->tex_base; g_pool_samp_base = q->samp_base;
    // mtl_sub_begin advances g_set_epoch; this also protects callers which activate
    // a queue before a present instead of immediately beginning its next submission.
    g_bc_stale = 1; g_set_epoch++;
}
void mtl_queue_forget(uint64_t serial) {
    vk_queue_state **at = &g_vk_queues;
    while (*at && (*at)->serial != serial) at = &(*at)->next;
    if (!*at) return;
    vk_queue_state *q = *at; *at = q->next;
    if (g_vk_active == q) {
        extern uint64_t g_nvn_active_queue; g_nvn_active_queue = 0;
        g_vk_active = NULL; state_reset(); memset(g_tf, 0, sizeof g_tf); memset(g_tf_off, 0, sizeof g_tf_off);
        g_tf_on = g_tf_in_pass = 0; g_tf_verts = 0; g_tfc_cur = -1; g_rc_skip = 0;
        g_pool_tex_base = g_pool_samp_base = 0; g_bc_stale = 1; g_set_epoch++;
    }
    for (int i = 0; i < 9; i++) mtl_tex_unpin(q->rt_pins[i]);
    free(q);
}
static int tfc_ready(void) {
    static int failed; if (g_tfc_buf || failed) return g_tfc_buf != VK_NULL_HANDLE;
    if (!vkCmdDrawIndirectByteCountEXT) { failed = 1; VKRED_ONCE(91, "transform feedback from a geometry shader: no vkCmdDrawIndirectByteCountEXT, its captures are dropped"); return 0; }
    VkBufferCreateInfo bi = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, NULL, 0, (VkDeviceSize)TFC * 16, VK_BUFFER_USAGE_TRANSFORM_FEEDBACK_COUNTER_BUFFER_BIT_EXT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_SHARING_MODE_EXCLUSIVE};
    VkDeviceMemory m; VKCHK(vkCreateBuffer(V.dev, &bi, NULL, &g_tfc_buf)); VkMemoryRequirements rq; vkGetBufferMemoryRequirements(V.dev, g_tfc_buf, &rq);
    VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL, rq.size, vk_mem_type(rq.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0)};
    VKCHK(vkAllocateMemory(V.dev, &ai, NULL, &m)); VKCHK(vkBindBufferMemory(V.dev, g_tfc_buf, m, 0));
    VKLOG("transform feedback from a geometry shader: counter buffer of %d slots", TFC); return 1;
}
static vkshader_t *xfb_shader(void) { return S.sh[2] && S.sh[2]->xfb ? S.sh[2] : S.sh[0]; }   // the last stage before rasterization captures
static int xfb_begin(VkCommandBuffer cb) {   // inside the pass, after the pipeline is bound; 1 when a capture began, 2 when it counts on the GPU (a geometry shader)
    vkshader_t *vs = xfb_shader();
    if (!g_tf_on || !V.has_xfb || !vs || !vs->xfb || !g_xfb_on) return 0;
    int gs = vs == S.sh[2];
    if (gs && (g_tfc_cur < 0 || !tfc_ready())) return 0;
    vk_readback_unknown();   // capture extents need not match the input vertex count
    VkBuffer b[4]; VkDeviceSize o[4], n[4];
    for (int k = 0; k < 4; k++) {
        uint64_t from = gs ? 0 : g_tf_off[k];   // a counted capture is bound at the buffer's start: the counter says where it goes on
        vkpool_t *p = g_tf[k].addr ? vk_pool_of(g_tf[k].addr + from, 4) : NULL;
        if (vs->xfb_stride[k] && p && from < g_tf[k].size && vk_buf_at(g_tf[k].addr + from, 4, &b[k], &o[k])) {
            n[k] = g_tf[k].size - from; if (o[k] - p->boff + n[k] > p->hi - p->lo) n[k] = p->hi - p->lo - (o[k] - p->boff);   // (o is the buffer's offset: a slab slot's starts at boff)
        } else { vk_scratch(64, 64, &b[k], &o[k]); n[k] = 64; if (vs->xfb_stride[k]) VKRED_ONCE(83, "transform feedback buffer %d unbound, full or outside the memory pools: its capture is dropped", k); }
    }
    vkCmdBindTransformFeedbackBuffersEXT(cb, 0, 4, b, o, n);
    if (gs) { VkBuffer cbs[4] = {g_tfc_buf, g_tfc_buf, g_tfc_buf, g_tfc_buf}; VkDeviceSize co[4]; for (int k = 0; k < 4; k++) co[k] = (VkDeviceSize)g_tfc_cur * 16 + 4 * k;
        int resume = g_tf_ctl[g_tfc_cur].ctr_ok; vkCmdBeginTransformFeedbackEXT(cb, 0, 4, resume ? cbs : NULL, resume ? co : NULL); return 2; }
    vkCmdBeginTransformFeedbackEXT(cb, 0, 0, NULL, NULL);
    return 1;
}
static void xfb_end(VkCommandBuffer cb, int prim, uint32_t count, uint32_t inst, int xf) {   // after the draw: the capture ends and the bytes it wrote are counted
    void vk_gpu_wrote(uint64_t a, uint64_t n);
    if (xf == 2) {
        VkBuffer cbs[4] = {g_tfc_buf, g_tfc_buf, g_tfc_buf, g_tfc_buf}; VkDeviceSize co[4]; for (int k = 0; k < 4; k++) co[k] = (VkDeviceSize)g_tfc_cur * 16 + 4 * k;
        vkCmdEndTransformFeedbackEXT(cb, 0, 4, cbs, co); g_tf_in_pass = 1; g_tf_ctl[g_tfc_cur].dirty = 1;
        g_tf_ctl[g_tfc_cur].ctr_ok = 1; g_tf_ctl[g_tfc_cur].gs = 1;
        for (int k = 0; k < 4; k++) if (S.sh[2]->xfb_stride[k] && g_tf[k].addr) vk_gpu_wrote(g_tf[k].addr, g_tf[k].size);   // how much is only known on the GPU
        { static int n; if (n++ < 4) VKLOG("transform feedback: geometry shader %016llx captured, counted on the GPU (control buffer %#llx)", (unsigned long long)S.sh[2]->key, (unsigned long long)g_tf_ctl[g_tfc_cur].ctl); }
        return;
    }
    vkCmdEndTransformFeedbackEXT(cb, 0, 0, NULL, NULL);
    uint32_t v = xfb_verts(prim, count) * (inst ? inst : 1); g_tf_in_pass = 1; g_tf_verts += v;
    for (int k = 0; k < 4; k++) {
        uint64_t m = (uint64_t)v * S.sh[0]->xfb_stride[k]; if (!m || !g_tf[k].addr) continue;
        if (g_tf_off[k] + m > g_tf[k].size) m = g_tf[k].size > g_tf_off[k] ? g_tf[k].size - g_tf_off[k] : 0;
        if (m) vk_gpu_wrote(g_tf[k].addr + g_tf_off[k], m);
        g_tf_off[k] += m;
    }
}
// DrawTransformFeedback draws the vertices the captures ended into one control buffer wrote: counted since Begin (g_tf_verts) and kept by control buffer address at End / Pause
// (a table by address: Xenoblade Chronicles 3 gives every particle emitter a control buffer of its own, hundreds of them, 32 bytes apart)
static int tf_ctl_slot(uint64_t ctl, int add) {   // the entry of a control buffer; with add, a new one (the oldest in its probe range is replaced when they are all taken)
    unsigned h = (unsigned)((ctl >> 5) * 2654435761u) % TFC;
    for (int k = 0; k < 16; k++) { unsigned i = (h + (unsigned)k) % TFC; if (g_tf_ctl[i].ctl == ctl) return (int)i; if (!g_tf_ctl[i].ctl) return add ? (int)i : -1; }
    return add ? (int)h : -1;
}
static void tf_state(const p_tf_state *t) {
    switch (t->what & 3) {
    case 0: memset(g_tf_off, 0, sizeof g_tf_off); g_tf_verts = 0; g_tf_on = 1;   // Begin: written from the start of each buffer
        g_tfc_cur = tf_ctl_slot(t->ctl, 1); g_tf_ctl[g_tfc_cur].ctl = t->ctl; g_tf_ctl[g_tfc_cur].ctr_ok = 0; g_tf_ctl[g_tfc_cur].gs = 0; break;
    case 3: g_tf_on = 1; g_tfc_cur = tf_ctl_slot(t->ctl, 1); if (g_tf_ctl[g_tfc_cur].ctl != t->ctl) { g_tf_ctl[g_tfc_cur].ctl = t->ctl; g_tf_ctl[g_tfc_cur].ctr_ok = 0; g_tf_ctl[g_tfc_cur].gs = 0; } break;   // Resume: on from the bytes written
    default: g_tf_on = 0;                                                // End / Pause: what was captured is read by later draws (as vertices), after the barrier the next pass begins with
        if (g_tf_in_pass) { vk_end_pass(); g_tf_in_pass = 0; }
        int i = tf_ctl_slot(t->ctl, 1); g_tf_ctl[i].ctl = t->ctl; g_tf_ctl[i].verts = g_tf_verts;
        if (g_tf_ctl[i].gs && g_tf_ctl[i].dirty) {   // a counted capture: its bytes into the game's control buffer too (+0, one u32 a feedback buffer), for anything that reads it there
            VkBuffer b; VkDeviceSize o; g_tf_ctl[i].dirty = 0;
            if (vk_buf_at(t->ctl, 16, &b, &o)) { VkCommandBuffer cb = vk_cmd(); VkBufferCopy c = {(VkDeviceSize)i * 16, o, 16};
                VkMemoryBarrier mb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, NULL, VK_ACCESS_MEMORY_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT};
                vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &mb, 0, NULL, 0, NULL); vkCmdCopyBuffer(cb, g_tfc_buf, b, 1, &c);
                void vk_gpu_wrote(uint64_t a, uint64_t n); vk_gpu_wrote(t->ctl, 16); } }
    }
}
static void draw(const p_draw *d);
static void draw_tf(const p_draw_tf *t) {
    int i = tf_ctl_slot(t->ctl, 0);
    if (i < 0) { VKRED_ONCE(86, "DrawTransformFeedback with a control buffer no capture ended into (%#llx): nothing drawn", (unsigned long long)t->ctl); return; }
    if (g_tf_ctl[i].gs) {   // drawn by the counter its geometry shader's captures wrote (see tfc_ready)
        if (!g_tf_ctl[i].ctr_ok) return;
        p_draw d = {t->prim, 0, 1, 1, 0, 0, 0, 0, 0}; g_bc_slot = i; draw(&d); g_bc_slot = -1; return; }
    p_draw d = {t->prim, 0, (int32_t)g_tf_ctl[i].verts, 1, 0, 0, 0, 0, 0};
    if (d.count) draw(&d);
}
static void trace_draw(const char *kind, int prim, uint32_t count, uint32_t inst, int itype) {
    VKLOG("draw: %s prim %d count %u instances %u itype %d; shaders %016llx %016llx %016llx; targets %d (formats %d %d %d %d) depth format %d samples %u, pass %ux%u; viewport %.0f %.0f %.0f %.0f; streams:",
          kind, prim, count, inst, itype, (unsigned long long)(S.sh[0] ? S.sh[0]->key : 0), (unsigned long long)(S.sh[1] ? S.sh[1]->key : 0), (unsigned long long)(S.sh[2] ? S.sh[2]->key : 0),
          S.ncol, S.cfmt[0], S.cfmt[1], S.cfmt[2], S.cfmt[3], S.dfmt, S.samples, S.pass_w, S.pass_h, S.vp[0][0], S.vp[0][1], S.vp[0][2], S.vp[0][3]);
    for (int i = 0; i < S.rt_n; i++) if (S.rt_bad[i]) VKLOG("  colour target %d: not a texture object (first word %#x): left out", i, S.rt_bad[i]);
    for (int i = 0; i < S.rt_n; i++) if (S.rt_has_col[i]) { vkimg_t *im = vk_tex(&S.rt_col[i]); const tex_t *t = &S.rt_col[i];
        VKLOG("  colour target %d: serial %llu → image %p: %ux%ux%u layers %u levels %u samples %u fmt %d vtype %d; nvn target %d format %#x flags %#x stride %u size %llu pool %p+%#llx",
              i, (unsigned long long)t->serial, (void *)im, im ? im->w : 0, im ? im->h : 0, im ? im->d : 0, im ? im->layers : 0, im ? im->levels : 0, im ? im->samples : 0, im ? im->fmt : 0, im ? im->vtype : 0,
              t->d.target, t->d.format, t->d.flags, t->d.stride, (unsigned long long)t->L.size, (const void *)t->pool, (unsigned long long)t->off);
        if (S.rt_has_cview[i]) { const view_t *v = &S.rt_cview[i]; VKLOG("  its view: levels %d (%u+%u) layers %d (%u+%u) target %d (%u) format %d (%#x)",
              v->set_levels, v->base_level, v->nlevels, v->set_layers, v->min_layer, v->nlayers, v->set_target, v->target, v->set_format, v->format); } }
    if (S.rt_has_dep) { vkimg_t *im = vk_tex(&S.rt_dep); const tex_t *t = &S.rt_dep;
        VKLOG("  depth target: serial %llu → image %p: %ux%u fmt %d; nvn format %#x flags %#x size %llu pool %p+%#llx", (unsigned long long)t->serial, (void *)im, im ? im->w : 0, im ? im->h : 0,
              im ? im->fmt : 0, t->d.format, t->d.flags, (unsigned long long)t->L.size, (const void *)t->pool, (unsigned long long)t->off); }
    VKLOG("  scissor set %d: %d %d %d %d; depth range %.3f %.3f; viewport count first %.1f %.1f %.1f %.1f; alpha test %u ref %g", S.sc_set, S.sc.x, S.sc.y, S.sc.w, S.sc.h, S.dnear, S.dfar, S.vp[0][0], S.vp[0][1], S.vp[0][2], S.vp[0][3], S.color.alpha_func, S.alpha_ref);
    vkshader_t *vsh = S.sh[0]; uint32_t in_mask = vsh ? vsh->local[0] : 0;
    for (int l = 0; l < 16; l++) if (in_mask >> l & 1) { uint32_t a = S.vattr[l]; VKLOG("  location %d: attrib state %#x (stream %u) stride %d divisor %d buffer %#llx +%llu", l, a, (a >> 16) & 0x1f,
        S.vstream[(a >> 16) & 0xf].stride, S.vstream[(a >> 16) & 0xf].divisor, (unsigned long long)S.vbuf[(a >> 16) & 0xf].address, (unsigned long long)S.vbuf[(a >> 16) & 0xf].size); }
}
// Vertex fetches past a stream's bound size. NVN binds each stream with a size (BindVertexBuffers), and Maxwell's fetch reads zeros past it; Vulkan's binding carries no
// size here (robustBufferAccess is off), so Adreno read on into whatever followed the stream in the pool. agl draws full-screen passes past their streams: Super Mario
// Odyssey's alNoiseShader 9 vertices from a 36-byte stream (one triangle) and an alProgramShader pass 18 vertices from 72 bytes, every frame in Cap Kingdom; the extra
// vertices made triangles of the pool's next bytes (on Metal the hard-edged mist slabs of the Cascade Kingdom, seedds/nxnative issue #558; issue #42 here, seams on
// volumetrics). For a direct draw, a stream whose last
// element read (the last vertex, the last instance of a per-instance stream, element 0 of a stride-0 stream) reaches past its bound size is bound as a copy of its bound
// bytes followed by zeros. Indexed and indirect draws are not bounded (their last element is not known without reading the indices or the arguments).
// DM_VK_VBOUND=0 (live): bound as before (A/B)
static uint32_t vfmt_bytes(VkFormat f) {   // a vertex attribute's bytes (the formats vertex_format returns)
    if (f >= VK_FORMAT_R8_UNORM && f <= VK_FORMAT_R8_SRGB) return 1;
    if (f >= VK_FORMAT_R8G8_UNORM && f <= VK_FORMAT_R8G8_SRGB) return 2;
    if (f >= VK_FORMAT_R8G8B8_UNORM && f <= VK_FORMAT_B8G8R8_SRGB) return 3;
    if (f >= VK_FORMAT_R8G8B8A8_UNORM && f <= VK_FORMAT_A2B10G10R10_SINT_PACK32) return 4;
    if (f >= VK_FORMAT_R16_UNORM && f <= VK_FORMAT_R16_SFLOAT) return 2;
    if (f >= VK_FORMAT_R16G16_UNORM && f <= VK_FORMAT_R16G16_SFLOAT) return 4;
    if (f >= VK_FORMAT_R16G16B16_UNORM && f <= VK_FORMAT_R16G16B16_SFLOAT) return 6;
    if (f >= VK_FORMAT_R16G16B16A16_UNORM && f <= VK_FORMAT_R16G16B16A16_SFLOAT) return 8;
    if (f >= VK_FORMAT_R32_UINT && f <= VK_FORMAT_R32_SFLOAT) return 4;
    if (f >= VK_FORMAT_R32G32_UINT && f <= VK_FORMAT_R32G32_SFLOAT) return 8;
    if (f >= VK_FORMAT_R32G32B32_UINT && f <= VK_FORMAT_R32G32B32_SFLOAT) return 12;
    if (f == VK_FORMAT_B10G11R11_UFLOAT_PACK32 || f == VK_FORMAT_E5B9G9R9_UFLOAT_PACK32) return 4;
    return 16;
}
static uint64_t g_vbound_n;
static void vbuf_bound(VkCommandBuffer cb, uint32_t first, uint32_t nvert, uint32_t binst, uint32_t inst) {
    vkshader_t *vsh = S.sh[0]; if (!g_vbound || !vsh || !nvert || (dbg_draw() & 2) || g_var_now == 5) return;
    uint32_t in_mask = vsh->local[0], int_mask = vsh->local[1] | vsh->local[2], sint_mask = vsh->local[1]; uint64_t need[16] = {0};
    for (int loc = 0; loc < 16; loc++) {
        if (!(in_mask >> loc & 1)) continue;
        va_st a; memcpy(&a, &S.vattr[loc], 4); if (!a.fmt || !S.vbuf[a.stream].address) continue;
        VkFormat f = vertex_format(a.fmt, int_mask >> loc & 1, sint_mask >> loc & 1); if (f == VK_FORMAT_UNDEFINED) continue;
        uint32_t s = a.stream, st = S.vstream[s].stride, dv = S.vstream[s].divisor;
        uint64_t last = !st ? 0 : dv ? binst + (inst - 1) / dv : (uint64_t)first + nvert - 1, e = last * st + a.offset + vfmt_bytes(f);
        if (e > need[s]) need[s] = e;
    }
    for (uint32_t s = 0; s < 16; s++) {
        uint64_t n = need[s], sz = S.vbuf[s].size; if (n <= sz || !sz) continue;
        if (n > (16u << 20)) { VKRED_ONCE(87, "vertex stream %u: a draw reads %llu bytes of a %llu-byte binding: not bounded", s, (unsigned long long)n, (unsigned long long)sz); continue; }
        VkBuffer b; VkDeviceSize o; uint8_t *p = vk_scratch(n, 16, &b, &o);
        vk_ubd_read(S.vbuf[s].address, sz, p); memset(p + sz, 0, (size_t)(n - sz));
        vkCmdBindVertexBuffers(cb, s, 1, &b, &o); g_bc.vb[s] = b; g_bc.vo[s] = o;
        if (g_vbound_n++ < 8) VKLOG("vertex stream %u: draw of %u vertices from %u (shader %016llx) reads %llu bytes past its bound %llu: bound as a copy padded with zeros",
                                    s, nvert, first, (unsigned long long)vsh->key, (unsigned long long)(n - sz), (unsigned long long)sz);
    }
}
static void mesh_draw(const p_draw *d);   // NVN2 task / mesh draws (below, after dispatch)
// DM_VK_NOATT_LOG (see noatt_on): a draw in a pass with no attachments, with each storage buffer of its vertex shader (address, size, the pool chunk it lies in, a deferred
// update over it, its first 8 words) and words 0x70-0x8f of each constant buffer (c[7] and c[8], the bucket room and counts of be816c4f2431b602 / afe80dc02de844a6), as the
// CPU sees them now: memory the GPU wrote in submissions that completed, with the deferred updates applied. Across frames a counter that only grows is one never reset
static void noatt_log(const char *kind, uint32_t count, uint32_t inst, int64_t first, uint64_t ind) {
    void vk_ubd_read(uint64_t a, uint64_t n, uint8_t *dst); int vk_ubd_overlaps(uint64_t a, uint64_t n);
    if (g_noatt_logged >= noatt_log_n()) return;
    g_noatt_logged++; g_noatt_ind_logged = 0;
    const vkshader_t *s = S.sh[0];
    VKLOG("noatt %d: %s present %llu vertex %016llx count %u instances %u first %lld indirect %#llx", g_noatt_logged, kind, (unsigned long long)g_tex_frame, s ? (unsigned long long)s->key : 0ull,
          count, inst, (long long)first, (unsigned long long)ind);
    for (int i = 0; s && i < s->nbind; i++) {
        const dmsv_bind *b = &s->b[i]; uint32_t w[8] = {0}; char t[160]; int k = 0;
        if (b->kind == RK_SBUF && b->reg >= 0 && b->reg < 16) {
            const p_bind_buf *u = &S.ssbo[0][b->reg]; vkpool_t *p = u->addr ? vk_pool_peek(u->addr, 1) : NULL;
            uint64_t n = u->size && u->size < 32 ? u->size : 32; if (p && u->addr + n > p->hi) n = p->hi - u->addr;
            if (p) vk_ubd_read(u->addr, n & ~3ull, (uint8_t *)w);
            for (uint64_t j = 0; j < n / 4; j++) k += snprintf(t + k, sizeof t - k, " %#x", w[j]);
            if (p) VKLOG("  storage %d (binding %u): %#llx +%#llx, pool chunk %#llx..%#llx (%s, %s), %s; words%s", b->reg, b->binding, (unsigned long long)u->addr, (unsigned long long)u->size,
                         (unsigned long long)p->lo, (unsigned long long)p->hi, p->backed ? "backed" : "not backed yet", p->zero_copy ? "zero-copy" : "copied",
                         vk_ubd_overlaps(u->addr, u->size ? u->size : 4) ? "a deferred update over it" : "no deferred update over it", t);
            else VKLOG("  [RED] storage %d (binding %u): %#llx +%#llx in no memory pool: bound to the 64 KB zero buffer, which the shader then writes", b->reg, b->binding, (unsigned long long)u->addr, (unsigned long long)u->size);
        } else if (b->kind == RK_CBUF && b->reg >= 0 && b->reg < 16) {
            const p_bind_buf *u = &S.ubo[0][b->reg];
            if (u->addr && u->size >= 0x90 && vk_pool_peek(u->addr, 0x90)) { vk_ubd_read(u->addr + 0x70, 32, (uint8_t *)w); for (int j = 0; j < 8; j++) k += snprintf(t + k, sizeof t - k, " %#x", w[j]); }
            VKLOG("  uniform %d (binding %u): %#llx +%#llx; words at 0x70%s", b->reg, b->binding, (unsigned long long)u->addr, (unsigned long long)u->size, k ? t : ": (none)");
        }
    }
}
// and an indirect draw after one, with its arguments as the CPU sees them (instance counts there are those afe80dc02de844a6 wrote in completed submissions)
static void noatt_log_ind(const char *kind, uint64_t ind, int words) {
    void vk_ubd_read(uint64_t a, uint64_t n, uint8_t *dst);
    if (!ind || !g_noatt_logged || g_noatt_ind_logged >= noatt_log_n() || !vk_pool_peek(ind, (uint64_t)words * 4)) return;
    g_noatt_ind_logged++; uint32_t w[5] = {0}; vk_ubd_read(ind, (uint64_t)words * 4, (uint8_t *)w);
    VKLOG("  after noatt %d: %s, vertex %016llx, arguments at %#llx: %u %u %u %d %u", g_noatt_logged, kind, S.sh[0] ? (unsigned long long)S.sh[0]->key : 0ull, (unsigned long long)ind, w[0], w[1], w[2], (int)w[3], words > 4 ? w[4] : 0);
}
// An indirect-count multi-draw (hle_nvn.c nvnCommandBufferMultiDrawElementsIndirectCount) arrives as one draw per record: only those below the u32 count the game's GPU culling wrote may be
// made. Drawn regardless, the records past it (stale, written for other meshes) put wrong geometry on the screen, at the camera angles where the count was low (MH Rise, big flat polygons and
// streaks). The whole group is one draw with a count at its first record, the rest are skipped: 1 skip this record, 0 draw it as the group (*cb, *co: the count) or, without the feature, alone
static int mdi_log_now(void) { static int64_t f = -2; if (f == -2) { const char *e = getenv("DM_VK_MDI_LOG"); f = e ? atoll(e) : -1; } return f >= 0 && (int64_t)g_tex_frame >= f && (int64_t)g_tex_frame < f + 2; }
static int mdi_group(uint64_t cnt, uint64_t ind, int i, int max, int stride, int rec, VkBuffer *cb, VkDeviceSize *co) {
    if (!cnt) return 0;
    static int on = -1; if (on < 0) { const char *e = getenv("DM_VK_MDI"); on = e ? atoi(e) : 1; }   // 0: every record drawn, whatever the count
    static int64_t logf = -2; if (logf == -2) { const char *e = getenv("DM_VK_MDI_LOG"); logf = e ? atoll(e) : -1; }
    if (logf >= 0 && (int64_t)g_tex_frame >= logf && (int64_t)g_tex_frame < logf + 2 && !i && ind) {   // the count as the GPU copy holds it now (written by the GPU earlier) and as the CPU's memory has it; records with instances below and past it
        vkpool_t *cp = vk_pool_of(cnt, 4), *rp = vk_pool_of(ind, (uint64_t)stride * (uint64_t)max); uint32_t gc = ~0u, cc = 0; vk_ubd_read(cnt, 4, (uint8_t *)&cc);
        if (cp && cp->shadow) memcpy(&gc, cp->shadow + (cnt - cp->lo), 4);
        int below = 0, past = 0, zero = 0; if (rp && rp->shadow) for (int k = 0; k < max; k++) { uint32_t w[2]; memcpy(w, rp->shadow + (ind - rp->lo) + (uint64_t)k * (uint64_t)stride, 8); if (!w[0] || !w[1]) zero++; else if ((uint32_t)k < gc) below++; else past++; }
        VKLOG("[mdi] frame %llu count at %#llx (pool %#llx): GPU copy %u, CPU %u; max %d; records at %#llx (pool %#llx): %d live below the count, %d live past it, %d empty", (unsigned long long)g_tex_frame, (unsigned long long)cnt, (unsigned long long)(cp ? cp->lo : 0), gc, cc, max, (unsigned long long)ind, (unsigned long long)(rp ? rp->lo : 0), below, past, zero);
        static int nrec; if (max > 1 && rp && rp->shadow && nrec < 40) { nrec++; for (int k = 0; k < max && k < 8; k++) { uint32_t w[5]; memcpy(w, rp->shadow + (ind - rp->lo) + (uint64_t)k * (uint64_t)stride, sizeof w); VKLOG("[mdir] %#llx #%d: %u %u %u %d %u", (unsigned long long)ind, k, w[0], w[1], w[2], (int)w[3], w[4]); } } }
    if (!on) return 0;
    if (on == 2 || on == 3) { vkpool_t *cp = vk_pool_of(cnt, 4); int copied = cp && cp->shadow; if (copied != (on == 2)) return 0; }   // A/B: the count honoured only in copied pools (2) or only outside them (3)
    if (!V.has_draw_count || stride < rec || stride & 3 || (uint32_t)max > V.props.limits.maxDrawIndirectCount) { VKRED_ONCE(52, "multi-draw indirect with a count, not usable here (max %d, stride %d): every record drawn", max, stride); return 0; }
    if (i) return 1;
    if (!vk_buf_at(cnt, 4, cb, co)) { VKRED_ONCE(55, "multi-draw indirect: count outside the memory pools, nothing drawn"); return 1; }
    { static int n; if (!n++) VKLOG("multi-draw indirect with a count: drawn by the GPU's count (first: max %d, stride %d)", max, stride); }
    return 0;
}
static void draw(const p_draw *d) {
    if (d->prim == DM_PRIM_MESH_TASKS) { mesh_draw(d); return; }
    VkBuffer mcb = VK_NULL_HANDLE; VkDeviceSize mco = 0; if (mdi_group(d->mdi_cnt, d->ind, d->mdi_i, d->mdi_max, d->mdi_stride, 16, &mcb, &mco)) return;
    int prim = d->prim;
    // before prepare_draw, whose driver segment carries it (c0 + 0x644): set after it, each draw's segment had the draw before's. An indirect draw's is in GPU memory: 0
    g_draw_binst = d->ind ? 0 : (uint32_t)d->base_instance;
    g_claim_draw = (typeof(g_claim_draw)){(uint32_t)d->count, (uint32_t)d->instances, d->prim, d->ind || d->mdi_cnt || g_bc_slot >= 0};
    // Read only a word from the arguments into scratch, outside the pass. This diagnoses stale GPU buffer visibility without modifying guest bytes.
    if (g_indirect_touch && d->ind) { uint64_t a = d->ind & ~3ull; vk_end_pass(); if (g_indirect_touch == 1) rd_read(&a, 1); }
    if (!prepare_draw(prim == 7 ? 4 : prim, prim == 7)) return;
    if (g_pd_draw_mode) pass_dump_draw_geometry("vertices", d->prim, d->first, 0, (uint32_t)d->base_instance, 0, -1);
    pass_journal_draw("vertices", (uint32_t)d->count, (uint32_t)d->instances, d->ind, S.sh[0] ? S.sh[0]->key : 0, 0, 0);
    if (g_noatt_open) noatt_log("draw", (uint32_t)d->count, (uint32_t)d->instances, d->first, d->ind); else if (d->ind) noatt_log_ind("draw", d->ind, 4);
    VkCommandBuffer cb = vk_cmd();
    int g_ok = draw_allowed(); if (g_ok == 2 || (g_tracing && g_x_tracepass >= 0)) trace_draw("draw", d->prim, (uint32_t)d->count, (uint32_t)d->instances, -1);
    if (g_tracing && g_x_tracepass >= 0 && !d->ind && S.vbuf[0].address) {   // the first vertices stream 0 gives the draw (with the deferred updates), every word as a float
        uint32_t st = S.vstream[0].stride ? S.vstream[0].stride : 16, nw = st / 4 < 16 ? st / 4 : 16;
        for (uint32_t v = 0; v < (uint32_t)d->count && v < 4; v++) { uint64_t o = (uint64_t)((uint32_t)d->first + v) * st; if (o + nw * 4 > S.vbuf[0].size) break;
            float w[16]; vk_ubd_read(S.vbuf[0].address + o, nw * 4, (uint8_t *)w); char t[400]; int k = 0; for (uint32_t i = 0; i < nw; i++) k += snprintf(t + k, sizeof t - k, " %g", w[i]);
            VKLOG("  vertex %u of stream 0 (+%llu):%s", (uint32_t)d->first + v, (unsigned long long)o, t); } }
    if (g_var_now == 10 && prim == 7) g_ok = 0; if (g_var_now == 11 && prim != 7) g_ok = 0;   // bisection: quads from the found draw on, or the rest
    { static int nq, nn; if (g_var_now) { if (prim == 7) nq++; else nn++; if ((nq + nn) % 50 == 1) VKLOG("draws from the found one on: %d quads, %d others", nq, nn); } }
    uint32_t inst = d->instances ? (uint32_t)d->instances : 1;
    { static int n; if (n < 12 && (d->ind || (uint32_t)d->count > (1u << 20) || inst > (1u << 16))) { n++;
        VKLOG("unusual draw: count %u instances %u indirect %#llx prim %d first %d", (unsigned)d->count, inst, (unsigned long long)d->ind, d->prim, d->first); } }
    if (d->ind) {
        VkBuffer b; VkDeviceSize o;
        if (g_tf_on && S.sh[0] && S.sh[0]->xfb) VKRED_ONCE(84, "indirect draw under transform feedback: not captured");
        if (!vk_buf_at(d->ind, mcb ? (uint64_t)d->mdi_stride * (uint64_t)d->mdi_max : 16, &b, &o)) VKRED_ONCE(53, "indirect draw arguments outside the memory pools");   // (an else after the nested if once went to the inner one)
        else if (!(dbg_draw() & 1) && g_ok) { vk_occ_draw(); if (mcb) vkCmdDrawIndirectCount(cb, b, o, mcb, mco, (uint32_t)d->mdi_max, (uint32_t)d->mdi_stride); else vkCmdDrawIndirect(cb, b, o, 1, 16); pass_dump_draw_checkpoint("vertices-indirect", 0, 0, d->ind, 0, 0, 0); }
        return;
    }
    if (prim == 7) {   // quads: the triangle list indices of one static buffer (built once), the first vertex as the vertex offset
        uint32_t nq = (uint32_t)d->count / 4; if (!nq) return;
        vbuf_bound(cb, (uint32_t)d->first, nq * 4, (uint32_t)d->base_instance, inst);
        if (g_tf_on && S.sh[0] && S.sh[0]->xfb) VKRED_ONCE(85, "quads under transform feedback: not captured");
        VkBuffer qb = quad_index_buffer(&nq);
        bc_index(cb, qb, 0, VK_INDEX_TYPE_UINT32); if (!(dbg_draw() & 1) && g_ok) { vk_occ_draw(); vkCmdDrawIndexed(cb, nq * 6, inst, 0, g_var_now == 4 ? 0 : d->first, (uint32_t)d->base_instance); pass_dump_draw_checkpoint("quads", nq * 6, inst, 0, 0, 0, 0); } return; }
    if (prim == 2) VKRED_ONCE(54, "line loops drawn as line strips");
    vbuf_bound(cb, (uint32_t)d->first, (uint32_t)d->count, (uint32_t)d->base_instance, inst);
    if (g_bc_slot >= 0) {   // DrawTransformFeedback of a geometry shader's capture: its vertex count is the counter's bytes over the stream's stride
        vkshader_t *vsh = S.sh[0]; uint32_t st = 0;
        for (int loc = 0; vsh && loc < 16 && !st; loc++) if (vsh->local[0] >> loc & 1) { va_st a; memcpy(&a, &S.vattr[loc], 4); if (a.fmt) st = S.vstream[a.stream].stride; }
        if (!st) { VKRED_ONCE(92, "DrawTransformFeedback of a geometry shader's capture with no vertex stride: nothing drawn"); return; }
        // The particles' update draws last frame's capture this way while capturing this frame's (a sequence of its own): the capture goes around the draw as at any other
        if (!(dbg_draw() & 1) && g_ok) { int xf = xfb_begin(cb); { vk_occ_draw(); vkCmdDrawIndirectByteCountEXT(cb, inst, (uint32_t)d->base_instance, g_tfc_buf, (VkDeviceSize)g_bc_slot * 16, 0, st); }
            if (xf == 2) xfb_end(cb, prim, 0, inst, xf); else if (xf) { vkCmdEndTransformFeedbackEXT(cb, 0, 0, NULL, NULL); VKRED_ONCE(93, "a vertex shader's capture around a DrawTransformFeedback of a counted capture: its bytes are not counted"); }
            pass_dump_draw_checkpoint("transform-feedback", 0, inst, 0, 0, 0, 0);
            { static int n; if (n++ < 6) VKLOG("transform feedback: DrawTransformFeedback by the counter of control buffer %#llx, stride %u%s", (unsigned long long)g_tf_ctl[g_bc_slot].ctl, st, xf == 2 ? ", captured again" : ""); } }
        return;
    }
    if (!(dbg_draw() & 1) && g_ok) { int xf = xfb_begin(cb); { vk_occ_draw(); vkCmdDraw(cb, (uint32_t)d->count, inst, (uint32_t)d->first, (uint32_t)d->base_instance); } if (xf) xfb_end(cb, prim, (uint32_t)d->count, inst, xf); pass_dump_draw_checkpoint("direct", (uint32_t)d->count, inst, 0, 0, 0, 0); }
}
// DM_VK_ORIGINAL_INDEX=<producer GS key>:<consumer VS key> (live diagnostic, off by default).
// Capture direct triangle draws whose GS writes argument slot 0 and index slot 1, then use their original indices for matching indirect draws.
// This isolates index compaction: colour shaders, depth/stencil and raster state stay intact. Never a rendering fix or a performance mode.
enum { OI_MAX = 1024, OI_TEX = 8 };
typedef struct {
    uint32_t inputs[3], attr[16], streams, cbufs, ntex, sc_set, restart, origin, flip, depth_clamp;
    NVNbufferRange vb[16]; vs_st vs[16]; uint64_t vb_hash[16];
    struct { uint64_t addr, size, hash; } cb[16];
    struct { uint64_t handle, seq, serial, view_hash; uint32_t reg, dim, flags, samp; } tex[OI_TEX];
    uint64_t depth_pool, depth_off, depth_serial, depth_view, c1_hash; uint32_t c1_len;
    float vp[16][4], near, far; p_scissor sc; poly_st poly; uint8_t push[0x100];
} original_index_sig;
typedef struct { uint64_t args, indices, index_hash; p_draw_idx draw; original_index_sig sig; } original_index_entry;
static original_index_entry g_oi[OI_MAX]; static uint64_t g_oi_gs, g_oi_vs;
static unsigned g_oi_n, g_oi_hits, g_oi_miss, g_oi_mismatch, g_oi_refused, g_oi_logs;
static int g_oi_dump;
static uint64_t original_index_hash(const void *data, size_t n, uint64_t hash) {
    const uint8_t *p = data; for (size_t i = 0; i < n; i++) hash = (hash ^ p[i]) * 1099511628211ull; return hash;
}
static int original_index_range(uint64_t addr, uint64_t size, uint64_t *hash) {
    if (!addr || !size || size > 1024 * 1024 || !vk_pool_peek(addr, size)) return 0;
    uint8_t data[4096]; *hash = 1469598103934665603ull;
    for (uint64_t off = 0; off < size;) { uint64_t n = size - off; if (n > sizeof data) n = sizeof data;
        vk_ubd_read(addr + off, n, data); *hash = original_index_hash(data, (size_t)n, *hash); off += n; }
    return 1;
}
static int original_index_signature(original_index_sig *sig) {
    const vkshader_t *vs = S.sh[0]; if (!vs || vs->xfb || !S.rt_has_dep || S.rast_discard || g_tf_on) return 0;
    memset(sig, 0, sizeof *sig); memcpy(sig->inputs, vs->local, sizeof sig->inputs);
    for (int loc = 0; loc < 16; loc++) if (vs->local[0] >> loc & 1) {
        va_st a; memcpy(&a, &S.vattr[loc], sizeof a); if (a.stream >= 16) return 0; sig->attr[loc] = S.vattr[loc]; sig->streams |= 1u << a.stream; }
    for (int i = 0; i < 16; i++) if (sig->streams >> i & 1) {
        sig->vb[i] = S.vbuf[i]; sig->vs[i] = S.vstream[i];
        if (!original_index_range(sig->vb[i].address, sig->vb[i].size, &sig->vb_hash[i])) return 0; }
    for (int i = 0; i < vs->nbind; i++) { const dmsv_bind *b = &vs->b[i];
        if (b->kind == RK_SUPPORT) continue;
        if (b->kind == RK_C1) { if (vs->c1_len && !vs->c1) return 0; sig->c1_len = vs->c1_len; sig->c1_hash = original_index_hash(vs->c1, vs->c1_len, 1469598103934665603ull); continue; }
        if (b->kind == RK_CBUF && b->reg >= 0 && b->reg < 16 && b->count == 1) {
            int reg = b->reg; const p_bind_buf *u = &S.ubo[0][reg]; sig->cbufs |= 1u << reg;
            sig->cb[reg].addr = u->addr; sig->cb[reg].size = u->size;
            if (u->size > 65536 || !original_index_range(u->addr, u->size, &sig->cb[reg].hash)) return 0;
        } else if (b->kind == RK_TEX && b->desc == RD_SAMPLED && b->reg >= 0 && b->reg < 256 && b->count == 1 && b->samp == RS_TEX && sig->ntex < OI_TEX) {
            const tex_t *t; const view_t *view; uint64_t h = S.tex[0][b->reg];
            if (!vk_tex_by_id((int)(h & 0xfffff), &view, &t) || !t) return 0;
            unsigned j = sig->ntex++; sig->tex[j].handle = h; sig->tex[j].seq = S.tex_sseq[0][b->reg]; sig->tex[j].serial = t->serial;
            if (view) sig->tex[j].view_hash = original_index_hash(view, sizeof *view, 1469598103934665603ull);
            sig->tex[j].reg = (uint32_t)b->reg; sig->tex[j].dim = b->dim; sig->tex[j].flags = b->flags; sig->tex[j].samp = b->samp;
        } else return 0; // storage, bindless, pointer-derived and driver inputs cannot be proved equal by this diagnostic
    }
    for (unsigned i = 1; i < sig->ntex; i++) for (unsigned j = i; j && sig->tex[j].reg < sig->tex[j - 1].reg; j--) {
        typeof(sig->tex[0]) t = sig->tex[j]; sig->tex[j] = sig->tex[j - 1]; sig->tex[j - 1] = t; }
    sig->depth_pool = (uint64_t)(uintptr_t)S.rt_dep.pool; sig->depth_off = S.rt_dep.off; sig->depth_serial = S.rt_dep.serial;
    if (S.rt_has_dview) sig->depth_view = original_index_hash(&S.rt_dview, sizeof S.rt_dview, 1469598103934665603ull);
    memcpy(sig->vp, S.vp, sizeof sig->vp); sig->near = S.dnear; sig->far = S.dfar; sig->sc = S.sc; sig->sc_set = S.sc_set;
    sig->poly = S.poly; sig->restart = S.prim_restart; sig->origin = g_lower_left; sig->flip = g_flipy; sig->depth_clamp = S.depth_clamp;
    memcpy(sig->push, S.push[0], sizeof sig->push); return 1;
}
// BEGIN original-index GPU snapshot diagnostic
// DM_VK_ORIGINAL_INDEX_DUMP=<GS>:<VS>: one matched submission, unchanged rendering. Copies occur immediately before each consumer.
// One bounded allocation is retained until its submission completes; no GPU wait is introduced. The TSV indexes exact bytes in the binary.
enum { OID_BYTES = 32 * 1024 * 1024 };
typedef struct { uint64_t args, generated, original, offset[3], size, present; uint32_t count, draw, pass; } original_index_dump_record;
static struct { VkBuffer b; VkDeviceMemory m; uint8_t *p; uint64_t used, fence; unsigned n; original_index_dump_record records[OI_MAX]; } g_oid;
static void original_index_dump_poll(uint64_t done) {
    if (!g_oid.b || !g_oid.n || done < g_oid.fence) return;
    VkMappedMemoryRange mr = {VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE, NULL, g_oid.m, 0, VK_WHOLE_SIZE};
    VKCHK(vkInvalidateMappedMemoryRanges(V.dev, 1, &mr));
    const char *root = getenv("DM_ROOT"); char path[1024];
    snprintf(path, sizeof path, "%s/index_dump_%llu.bin", root ? root : ".", (unsigned long long)g_oid.records[0].present);
    FILE *f = fopen(path, "wb"); int ok = f && fwrite(g_oid.p, 1, (size_t)g_oid.used, f) == g_oid.used; if (f) fclose(f);
    snprintf(path, sizeof path, "%s/index_dump_%llu.tsv", root ? root : ".", (unsigned long long)g_oid.records[0].present);
    f = fopen(path, "w"); if (f) {
        fprintf(f, "# GPU snapshots immediately before colour draw; binary offsets exclude alignment padding; uint16 little-endian indices; args are five uint32 words\n");
        fprintf(f, "record\tpresent\tpass\tdraw\tsource_count\targs_address\tgenerated_address\toriginal_address\targs_offset\tgenerated_offset\toriginal_offset\tindex_bytes\n");
        for (unsigned i = 0; i < g_oid.n; i++) { const original_index_dump_record *r = &g_oid.records[i];
            fprintf(f, "%u\t%llu\t%u\t%u\t%u\t0x%llx\t0x%llx\t0x%llx\t%llu\t%llu\t%llu\t%llu\n", i,
                (unsigned long long)r->present, r->pass, r->draw, r->count, (unsigned long long)r->args, (unsigned long long)r->generated, (unsigned long long)r->original,
                (unsigned long long)r->offset[0], (unsigned long long)r->offset[1], (unsigned long long)r->offset[2], (unsigned long long)r->size);
        } fclose(f);
    }
    VKLOG("original-index GPU dump: %u records, %llu bytes, fence %llu complete, binary %s; metadata %s", g_oid.n,
        (unsigned long long)g_oid.used, (unsigned long long)g_oid.fence, ok ? "written" : "FAILED", path);
    vkDestroyBuffer(V.dev, g_oid.b, NULL); vkFreeMemory(V.dev, g_oid.m, NULL); memset(&g_oid, 0, sizeof g_oid);
}
static void original_index_snapshot(const original_index_entry *r, const p_draw_idx *d) {
    if (g_oid.n == OI_MAX || (g_oid.b && g_oid.fence != mtl_last_fence() + 1)) return;
    uint64_t bytes = (uint64_t)r->draw.count * 2, addr[3] = {d->ind, d->ibuf, r->draw.ibuf}, size[3] = {20, bytes, bytes};
    VkBuffer src[3]; VkDeviceSize off[3]; VkBufferCopy copies[3]; uint64_t at = g_oid.used; original_index_dump_record rec = {0};
    for (unsigned i = 0; i < 3; i++) { uint64_t prefix = addr[i] & 3, n = (size[i] + prefix + 3) & ~3ull;
        if (!n || n > OID_BYTES - at || !vk_buf_at(addr[i] - prefix, n, &src[i], &off[i])) { g_oi_refused++; return; }
        copies[i] = (VkBufferCopy){off[i], at, n}; rec.offset[i] = at + prefix; at += n;
    }
    if (!g_oid.b) {
        VkBufferCreateInfo bi = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, NULL, 0, OID_BYTES, VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_SHARING_MODE_EXCLUSIVE};
        VKCHK(vkCreateBuffer(V.dev, &bi, NULL, &g_oid.b)); VkMemoryRequirements rq; vkGetBufferMemoryRequirements(V.dev, g_oid.b, &rq);
        VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL, rq.size, vk_mem_type(rq.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT, 0)};
        VKCHK(vkAllocateMemory(V.dev, &ai, NULL, &g_oid.m)); VKCHK(vkBindBufferMemory(V.dev, g_oid.b, g_oid.m, 0));
        VKCHK(vkMapMemory(V.dev, g_oid.m, 0, VK_WHOLE_SIZE, 0, (void **)&g_oid.p)); g_oid.fence = mtl_last_fence() + 1;
        setenv("DM_VK_ORIGINAL_INDEX_DUMP", "0", 1); // Keep this submission's captures, then stop until explicitly requested again.
    }
    vk_end_pass(); VkCommandBuffer cb = vk_cmd(); vk_img_barrier_all(cb);
    for (unsigned i = 0; i < 3; i++) vkCmdCopyBuffer(cb, src[i], g_oid.b, 1, &copies[i]);
    VkMemoryBarrier host = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, NULL, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT};
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &host, 0, NULL, 0, NULL); vk_img_barrier_all(cb);
    rec.args = d->ind; rec.generated = d->ibuf; rec.original = r->draw.ibuf; rec.size = bytes;
    rec.present = g_tex_frame; rec.count = (uint32_t)r->draw.count; rec.draw = (uint32_t)g_draw_i; rec.pass = (uint32_t)g_sub_pass;
    g_oid.records[g_oid.n++] = rec; g_oid.used = at;
}
// END original-index GPU snapshot diagnostic
static void original_index_read(void) {
    original_index_dump_poll(vk_fence_done());
    if (g_oi_gs && (g_oi_n || g_oi_hits || g_oi_miss || g_oi_refused) && (g_oi_logs++ < 8 || g_tex_frame % 600 == 0))
        VKLOG("original-index diagnostic: captured %u, replayed %u, no source %u, signature mismatch %u, refused %u", g_oi_n, g_oi_hits, g_oi_miss, g_oi_mismatch, g_oi_refused);
    uint64_t gs = 0, vs = 0; const char *e = getenv("DM_VK_ORIGINAL_INDEX"); char *end;
    const char *dump = getenv("DM_VK_ORIGINAL_INDEX_DUMP"); g_oi_dump = dump && *dump && strcmp(dump, "0"); if (g_oi_dump) e = dump;
    if (e && *e) { gs = strtoull(e, &end, 16); if (end != e && *end == ':') { const char *p = end + 1; vs = strtoull(p, &end, 16); if (end == p || *end) vs = 0; } }
    if (!gs || !vs) gs = vs = 0;
    if (gs != g_oi_gs || vs != g_oi_vs) { g_oi_logs = 0; VKLOG("original-index diagnostic GS %016llx -> VS %016llx (SSBO 0 arguments / 1 indices; exact input signatures)", (unsigned long long)gs, (unsigned long long)vs); }
    g_oi_gs = gs; g_oi_vs = vs; g_oi_n = g_oi_hits = g_oi_miss = g_oi_mismatch = g_oi_refused = 0;
}
static void original_index_capture(const p_draw_idx *d) {
    if (!g_oi_gs || !S.sh[2] || S.sh[2]->key != g_oi_gs) return;
    const p_bind_buf *args = &S.ssbo[2][0], *indices = &S.ssbo[2][1]; int slots = 0;
    unsigned i; for (i = 0; i < g_oi_n; i++) if (g_oi[i].args == args->addr && g_oi[i].indices == indices->addr) break;
    if (i == OI_MAX) { g_oi_refused++; return; }
    original_index_entry *r = &g_oi[i]; r->args = 0; // a later unsupported writer cannot leave an older source usable
    if (d->ind || d->mdi_cnt || d->prim != 4 || d->itype != 1 || d->count <= 0 || d->instances != 1 || d->base_vertex || d->base_instance || !S.ds.dt || !S.ds.dw) { g_oi_refused++; return; }
    for (int i = 0; i < S.sh[2]->nbind; i++) { const dmsv_bind *b = &S.sh[2]->b[i]; if (b->kind == RK_SBUF && b->desc == RD_SSBO && b->count == 1 && b->reg >= 0 && b->reg < 2) slots |= 1 << b->reg; }
    if (slots != 3 || !args->addr || args->size < 20 || !indices->addr || indices->size < (uint64_t)d->count * 2) { g_oi_refused++; return; }
    if (!original_index_signature(&r->sig) || !original_index_range(d->ibuf, (uint64_t)d->count * 2, &r->index_hash)) { g_oi_refused++; return; }
    r->args = args->addr; r->indices = indices->addr; r->draw = *d; if (i == g_oi_n) g_oi_n++;
}
static int original_index_replay(const p_draw_idx *d, p_draw_idx *out) {
    if (!g_oi_vs || !S.sh[0] || S.sh[0]->key != g_oi_vs || S.sh[2] || !d->ind) return 0;
    if (d->mdi_cnt || d->prim != 4 || d->itype != 1 || !S.ds.dt || S.ds.dw || S.ds.df != 3) { g_oi_refused++; return 0; }
    original_index_entry *r = NULL; for (unsigned i = 0; i < g_oi_n; i++) if (g_oi[i].args == d->ind && g_oi[i].indices == d->ibuf) { r = &g_oi[i]; break; }
    if (!r) { g_oi_miss++; return 0; }
    original_index_sig sig; uint64_t hash;
    if (!original_index_signature(&sig) || !original_index_range(r->draw.ibuf, (uint64_t)r->draw.count * 2, &hash)) { g_oi_refused++; return 0; }
    if (memcmp(&sig, &r->sig, sizeof sig) || hash != r->index_hash) { g_oi_mismatch++; return 0; }
    if (g_oi_dump) { original_index_snapshot(r, d); return 0; }
    *out = r->draw; g_oi_hits++;
    if (g_oi_logs < 2 && g_oi_hits <= 2) VKLOG("original-index replay: arguments %#llx generated indices %#llx -> original %#llx count %d", (unsigned long long)d->ind, (unsigned long long)d->ibuf, (unsigned long long)out->ibuf, out->count);
    return 1;
}

static void draw_idx(const p_draw_idx *d) {
    p_draw_idx original; if (g_oi_vs && original_index_replay(d, &original)) d = &original;
    VkBuffer mcb = VK_NULL_HANDLE; VkDeviceSize mco = 0; if (mdi_group(d->mdi_cnt, d->ind, d->mdi_i, d->mdi_max, d->mdi_stride, 20, &mcb, &mco)) return;
    int prim = d->prim;
    g_draw_binst = d->ind ? 0 : (uint32_t)d->base_instance;   // before prepare_draw (see draw)
    g_claim_draw = (typeof(g_claim_draw)){(uint32_t)d->count, (uint32_t)d->instances, d->prim, d->ind || d->mdi_cnt};
    // Both the indirect count and the generated indices may have been written by a prior shader. Four bytes touch each backing buffer; no unknown index extent is copied.
    if (g_indirect_touch && d->ind) { uint64_t a[2] = {d->ind & ~3ull, d->ibuf & ~3ull}; vk_end_pass(); if (g_indirect_touch == 1) rd_read(a, 2); }
    if (!prepare_draw(prim == 7 ? 4 : prim, 1)) return;
    if (g_pd_draw_mode) pass_dump_draw_geometry("indexed", d->prim, 0, d->base_vertex, (uint32_t)d->base_instance, d->ibuf, d->itype);
    pass_journal_draw("indexed", (uint32_t)d->count, (uint32_t)d->instances, d->ind, S.sh[0] ? S.sh[0]->key : 0, 0, 0);
    if (g_noatt_open) noatt_log("indexed draw", (uint32_t)d->count, (uint32_t)d->instances, d->base_vertex, d->ind); else if (d->ind) noatt_log_ind("indexed draw", d->ind, 5);
    VkCommandBuffer cb = vk_cmd();
    int g_ok = draw_allowed(); if (g_ok == 2 || (g_tracing && g_x_tracepass >= 0)) trace_draw("indexed draw", d->prim, (uint32_t)d->count, (uint32_t)d->instances, d->itype);
    if (d->ind && g_tracing && g_x_tracepass >= 0) { uint32_t w[5]; vk_ubd_read(d->ind, 20, (uint8_t *)w);   // the arguments as the CPU sees them now (a GPU writer's last merged copy)
        VKLOG("  indirect arguments at %#llx: count %u instances %u first index %u base vertex %d base instance %u", (unsigned long long)d->ind, w[0], w[1], w[2], (int)w[3], w[4]); }
    if (g_var_now == 11) g_ok = 0;
    VkIndexType it = d->itype == 0 ? VK_INDEX_TYPE_UINT8_EXT : d->itype == 1 ? VK_INDEX_TYPE_UINT16 : VK_INDEX_TYPE_UINT32;
    uint32_t isz = d->itype == 0 ? 1 : d->itype == 1 ? 2 : 4;
    VkBuffer ib; VkDeviceSize io;
    vk_pool_tag = "index"; if (!vk_buf_at(d->ibuf, 1, &ib, &io)) { VKRED_ONCE(56, "index buffer outside the memory pools"); return; }
    if (!d->ind) { vkpool_t *ip = vk_pool_peek(d->ibuf, 1); uint64_t in = (uint64_t)d->count * isz;
        if (ip && d->ibuf + in > ip->hi && ip->hi < ip->end) {   // the draw's indices run into the pool's next chunk: up to 16 MB of them bound from a copy of CPU memory (the game writes its index buffers on the CPU)
            if (in <= (16u << 20) && d->ibuf + in <= ip->end) { uint8_t *dst = vk_scratch(in, 4, &ib, &io); memcpy(dst, (const void *)(uintptr_t)d->ibuf, in); vk_cross(5, d->ibuf, in, ip); }
            else vk_cross(3, d->ibuf, in, ip); } }
    if (g_ubolog_key && S.sh[0] && S.sh[0]->key == g_ubolog_key && g_ubolog_n > 0) {   // DM_VK_UBO_LOG: the index data the draw reads (CPU view at replay)
        const uint8_t *ix = (const uint8_t *)(uintptr_t)d->ibuf; uint64_t h = 1469598103934665603ull; uint32_t mn = ~0u, mx = 0;
        for (uint32_t i = 0; i < (uint32_t)d->count; i++) { uint32_t v = isz == 1 ? ix[i] : isz == 2 ? ((const uint16_t *)ix)[i] : ((const uint32_t *)ix)[i]; h = (h ^ v) * 1099511628211ull; if (v < mn) mn = v; if (v > mx) mx = v; }
        VKLOG("ubo: present %llu index buffer %#llx count %d base vertex %d indirect %#llx hash %016llx range %u..%u", (unsigned long long)g_tex_frame, (unsigned long long)d->ibuf, d->count, d->base_vertex,
              (unsigned long long)d->ind, (unsigned long long)h, mn, mx);
        if (S.vbuf[0].address && mn != ~0u) { uint32_t st = S.vstream[0].stride ? S.vstream[0].stride : 12; uint64_t v0 = (uint64_t)(int64_t)d->base_vertex + mn;   // the first vertex this draw reads, every word
            if ((v0 + 1) * st <= S.vbuf[0].size) { const float *vf = (const float *)(uintptr_t)(S.vbuf[0].address + v0 * st); char t[512]; int k = 0;
                for (uint32_t i = 0; i < st / 4 && i < 24; i++) k += snprintf(t + k, sizeof t - k, " %g", vf[i]);
                VKLOG("ubo: present %llu first vertex read (%llu):%s", (unsigned long long)g_tex_frame, (unsigned long long)v0, t); } }
        static uint64_t dumped; if (dumped != h && S.vbuf[0].address && d->count <= 4096) { dumped = h; char t[1024]; int k = 0; uint32_t i0 = 0;   // once per index data: every index,
            for (uint32_t i = 0; i < (uint32_t)d->count; i++) { uint32_t v = isz == 1 ? ix[i] : isz == 2 ? ((const uint16_t *)ix)[i] : ((const uint32_t *)ix)[i];   // then every word of stream 0's vertices
                k += snprintf(t + k, sizeof t - k, " %u", v); if (k > 900 || i + 1 == (uint32_t)d->count) { VKLOG("ubo: indices from %u:%s", i0, t); k = 0; i0 = i + 1; } }
            uint32_t st = S.vstream[0].stride ? S.vstream[0].stride : 12, nv = (uint32_t)(S.vbuf[0].size / st);
            for (uint32_t i = 0; i < nv && i <= mx; i++) { const float *vf = (const float *)(uintptr_t)(S.vbuf[0].address + (uint64_t)i * st);   // every word of stream 0's vertex (up to 8)
                for (uint32_t w = 0; w < st / 4 && w < 8; w++) k += snprintf(t + k, sizeof t - k, "%s%g", w ? "," : " ", vf[w]);
                if (k > 800 || i + 1 == nv || i == mx) { VKLOG("ubo: vertices to %u:%s", i, t); k = 0; } } } }
    uint32_t inst = d->instances ? (uint32_t)d->instances : 1;
    { static int n; if (n < 12 && (d->itype == 0 || d->ind || (uint32_t)d->count > (1u << 20) || inst > (1u << 16))) { n++;
        VKLOG("unusual indexed draw: itype %d count %u instances %u indirect %#llx prim %d base vertex %d", d->itype, (unsigned)d->count, inst, (unsigned long long)d->ind, d->prim, d->base_vertex); } }
    if (d->ind) { VkBuffer b; VkDeviceSize o; bc_index(cb, ib, io, it); if (!vk_buf_at(d->ind, mcb ? (uint64_t)d->mdi_stride * (uint64_t)d->mdi_max : 20, &b, &o)) VKRED_ONCE(97, "indexed indirect draw arguments outside the memory pools");
        else if (!(dbg_draw() & 1) && g_ok) { vk_occ_draw(); if (mcb) vkCmdDrawIndexedIndirectCount(cb, b, o, mcb, mco, (uint32_t)d->mdi_max, (uint32_t)d->mdi_stride); else vkCmdDrawIndexedIndirect(cb, b, o, 1, 20); if (g_oi_gs) original_index_capture(d); pass_dump_draw_checkpoint("indexed-indirect", 0, 0, d->ind, 0, 0, 0); } return; }
    if (prim == 7) { VkBuffer b; VkDeviceSize o; int n = quad_indices((uint32_t)d->count, (const uint8_t *)(uintptr_t)d->ibuf, d->itype, 0, &b, &o); if (!n) return;
        bc_index(cb, b, o, VK_INDEX_TYPE_UINT32); if (!(dbg_draw() & 1) && g_ok) { vk_occ_draw(); vkCmdDrawIndexed(cb, (uint32_t)n, inst, 0, d->base_vertex, (uint32_t)d->base_instance); if (g_oi_gs) original_index_capture(d); pass_dump_draw_checkpoint("indexed-quads", (uint32_t)n, inst, 0, 0, 0, 0); } return; }
    if (io % isz) { VKRED_ONCE(57, "index buffer offset not aligned to its index size"); return; }
    bc_index(cb, ib, io, it);
    if (!(dbg_draw() & 1) && g_ok) { int xf = xfb_begin(cb); { vk_occ_draw(); vkCmdDrawIndexed(cb, (uint32_t)d->count, inst, 0, d->base_vertex, (uint32_t)d->base_instance); if (g_oi_gs) original_index_capture(d); } if (xf) xfb_end(cb, prim, (uint32_t)d->count, inst, xf); pass_dump_draw_checkpoint("direct", (uint32_t)d->count, inst, 0, 0, 0, 0); mrt_after_draw(d->base_vertex,(uint32_t)d->base_instance,d->ibuf); }
}
static nvn_hmap g_cpipes = {.klen = 24};   // {shader, its integer-texel bindings}
static pthread_mutex_t g_cpipe_mu = PTHREAD_MUTEX_INITIALIZER;   // the replay thread and compute_warm
static VkPipeline compute_build(vkshader_t *cs, uint64_t um, uint64_t im) {   // (VkPipeline)1 when it fails
    VkDescriptorSetLayout dsl[VK_NSETS] = {cs->dsl}; VkPipeline p;
    int bs = build_begin();   // (the crash dump's note: build_pipeline)
    if (bs >= 0) { snprintf(g_build[bs].what, sizeof g_build[bs].what, "compute pipeline, shader %016llx, integer-texel views %#llx/%#llx", (unsigned long long)cs->key, (unsigned long long)um, (unsigned long long)im); g_build[bs].phase = 1; }
    // DM_VK_SPV_ALT=1 in env.txt (with DM_VK_SPV_DIR): compute shaders take the alternative code too, from the start (compute_warm builds them all then, and they are kept)
    static int alt = -1; if (alt < 0) { const char *e = getenv("DM_VK_SPV_ALT"); alt = e && atoi(e); }
    VkComputePipelineCreateInfo ci = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO, NULL, 0, {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, vk_sg_pin(VK_SHADER_STAGE_COMPUTE_BIT, cs->local), 0, VK_SHADER_STAGE_COMPUTE_BIT, vk_shader_module(cs, um, im, alt && !um && !im ? 1u << 31 : 0, 0), "main", NULL}, vk_pipeline_layout(dsl, 1), VK_NULL_HANDLE, -1};   // (pNext: DM_VK_SUBGROUP_PIN)
    if (bs >= 0) g_build[bs].phase = 2;
    VkResult r = vkCreateComputePipelines(V.dev, V.pcache, 1, &ci, NULL, &p); build_end(bs);
    vk_pipeline_cache_changed();
    if (r) { LOG("[RED] [vk] compute pipeline failed (%d): %016llx, local %ux%ux%u, %u bytes of SPIR-V", r, (unsigned long long)cs->key, cs->local[0], cs->local[1], cs->local[2], cs->nspv * 4); p = (VkPipeline)1; }
    return p;
}
// Compute pipelines made ahead. One depends only on its shader (and integer-texel variants, rare), and building one costs Adreno 45-105 ms: Super Mario Odyssey's
// particle emitters run on compute shaders (EffectPtcl), and the first cap throw at a breakable object built one on the replay thread (a 65 ms frame), the first scenes
// several. A game has few (Odyssey 41), so at the first submission a background thread builds every compute shader in the bundle with no integer-texel variant (up to a cap, below)
static void *compute_warm(void *arg) {
    (void)arg; pthread_setname_np(pthread_self(), "vk compute warm"); setpriority(PRIO_PROCESS, 0, 10);
    int vk_shader_keys(uint32_t stage, uint64_t *out, int max); static uint64_t keys[4096]; int n = vk_shader_keys(5, keys, 4096), made = 0; uint64_t t0 = tick();
    // A bundle with many compute shaders is left to the replay: Monster Hunter Rise's (4096 or more) held ~550 MB more from the title on, and a Retroid Pocket 5
    // (Adreno 650, 8 GB) ran out of GPU memory loading the first area (vkAllocateMemory -2 at a 4 MB pool). DM_VK_COMPUTE_WARM_MAX (start-up, default 256): the most made ahead
    { const char *e = getenv("DM_VK_COMPUTE_WARM_MAX"); int cap = e ? atoi(e) : 256;
      if (n > cap) { VKLOG("compute pipelines made ahead: none, the bundle has %d%s compute shaders (more than %d, DM_VK_COMPUTE_WARM_MAX)", n, n == 4096 ? " or more" : "", cap); return NULL; } }
    for (int i = 0; i < n; i++) {
        uint64_t ck[3] = {keys[i], 0, 0};
        pthread_mutex_lock(&g_cpipe_mu); int have = nvn_hm_get(&g_cpipes, ck) != NULL; pthread_mutex_unlock(&g_cpipe_mu);
        vkshader_t *cs = have ? NULL : vk_shader(keys[i], 5); if (!cs) continue;
        VkPipeline p = compute_build(cs, 0, 0); made++;
        pthread_mutex_lock(&g_cpipe_mu); if (!nvn_hm_get(&g_cpipes, ck)) nvn_hm_put(&g_cpipes, ck, p); else if (p != (void *)1) vkDestroyPipeline(V.dev, p, NULL); pthread_mutex_unlock(&g_cpipe_mu);
    }
    VKLOG("compute pipelines made ahead: %d of %d compute shaders in %.0f ms", made, n, (tick() - t0) / 1e6);
    return NULL;
}
static void compute_warm_start(void) {
    if (getenv("DM_VK_NOPREWARM")) return;
    pthread_t t; pthread_attr_t a; pthread_attr_init(&a); pthread_attr_setdetachstate(&a, PTHREAD_CREATE_DETACHED); pthread_create(&t, &a, compute_warm, NULL); pthread_attr_destroy(&a);
}
// ---- a game's spatial upscaler replaced by a bilinear scale (DM_VK_UPSCALER_CS, profile-selected shader keys).
// Bananza's agl NDSR has 24 alternative compute shaders for one dispatch, each sampling texture slot 0 and writing image slot 1,
// with read-only weights in storage slot 2. It has no history texture. Keep any dispatch outside that binding contract.
// The existing colour approximation stays: variants that write encoded colour still need an explicit conversion (open).
static uint64_t g_ups_keys[64]; static uint8_t g_ups_noted[64]; static int g_ups_n = -1;
static int upscaler_kernel(uint64_t key) {
    if (g_ups_n < 0) { g_ups_n = 0; const char *e = getenv("DM_VK_UPSCALER_CS");
        for (const char *q = e; q && *q && g_ups_n < 64; ) { char *end; uint64_t k = strtoull(q, &end, 16); if (end == q) break; g_ups_keys[g_ups_n++] = k; q = *end == ',' ? end + 1 : end; }
        if (g_ups_n) VKLOG("upscaler network: %d shader variants named (DM_VK_UPSCALER_CS); compatible dispatches use bilinear scaling", g_ups_n); }
    for (int i = 0; i < g_ups_n; i++) if (g_ups_keys[i] == key) return 1;
    return 0;
}
static int upscaler_note(uint64_t key, uint8_t bit) {   // one successful binding trace and one refusal per named shader, not per frame
    for (int i = 0; i < g_ups_n; i++) if (g_ups_keys[i] == key) {
        int first = !(g_ups_noted[i] & bit); g_ups_noted[i] |= bit; return first;
    }
    return 0;
}
static int upscaler_contract(const vkshader_t *cs) {
    int sampled = 0, output = 0;
    for (int i = 0; i < cs->nbind; i++) { const dmsv_bind *b = &cs->b[i];
        switch (b->kind) {
        case RK_TEX:
            if (sampled++ || b->reg != 0 || b->desc != RD_SAMPLED || b->dim != 1 || b->flags || b->count != 1) return 0;
            break;
        case RK_IMG:
            if (output++ || b->reg != 1 || b->desc != RD_STORAGE || b->dim != 1 || b->flags || b->count != 1) return 0;
            break;
        case RK_SUPPORT: case RK_CBUF: case RK_C1: case RK_DRV: break;
        case RK_SBUF: if (b->reg != 2 || (b->flags & 4)) return 0; break;   // weights, no shader side effect to skip
        default: return 0;   // another image, bindless resource or global memory might carry an intermediate or history
        }
    }
    return sampled == 1 && output == 1;
}
typedef struct { vkimg_t *im; tex_t tex; view_t view; int tid, has_view, movie_output, allow_promotion, needs_opaque_alpha; VkFormat format; uint32_t level, layer, w, h; } upscaler_image;
// Log only metadata already resolved by this lookup. Matching slot bits identify all aliases of the selected handle;
// no second registry lookup, image creation, or content read is performed by diagnostics.
static void upscaler_image_diag(uint64_t handle, int sampled, const upscaler_image *u, int registered, const vkfmt_t *f, const char *why) {
    if (g_up_diag_left <= 0) return;
    g_up_diag_left--;
    uint32_t slots = 0;
    for (unsigned i = 0; i < 32; i++) {
        uint64_t h = sampled ? vk_tex_slot_handle(S.tex[5], S.img[5], i) : S.img[5][i];
        if (h == handle) slots |= 1u << i;
    }
    const tex_t *t = registered ? &u->tex : NULL; const view_t *v = u->has_view ? &u->view : NULL; const vkimg_t *im = u->im;
    VKLOG("upscaler image diag key %016llx present %llu %s slots %#x handle %#llx id %d: %s; texture serial %llu pool %p+%#llx fmt %#x target %d %ux%u; view format %d/%#x target %d/%d levels %d/%u+%u layers %d/%u+%u; requested VkFormat %d native VkFormat %d image %p usage %#x aspect %#x type %d %ux%ux%u levels %u layers %u samples %u; swizzle texture %d/%#x view %d/%u,%u,%u,%u",
        (unsigned long long)(S.sh[5] ? S.sh[5]->key : 0), (unsigned long long)g_tex_frame, sampled ? "sampled" : "output", slots,
        (unsigned long long)handle, u->tid, why ? why : "accepted", (unsigned long long)(t ? t->serial : 0), t ? (void *)t->pool : NULL,
        (unsigned long long)(t ? t->off : 0), t ? t->d.format : 0, t ? t->d.target : 0, t ? t->d.width : 0, t ? t->d.height : 0,
        v ? v->set_format : 0, v ? v->format : 0, v ? v->set_target : 0, v ? v->target : 0,
        v ? v->set_levels : 0, v ? v->base_level : 0, v ? v->nlevels : 0, v ? v->set_layers : 0, v ? v->min_layer : 0, v ? v->nlayers : 0,
        f ? f->vk : 0, im ? im->fmt : 0, im ? (void *)im->img : NULL, im ? im->usage : 0, im ? im->aspect : 0, im ? im->vtype : 0,
        im ? im->w : 0, im ? im->h : 0, im ? im->d : 0, im ? im->levels : 0, im ? im->layers : 0, im ? im->samples : 0,
        t ? t->set_swz : 0, t ? t->swz : 0, v ? v->set_swizzle : 0, v ? v->swz[0] : 0, v ? v->swz[1] : 0, v ? v->swz[2] : 0, v ? v->swz[3] : 0);
}
static int upscaler_image_at(uint64_t handle, int sampled, upscaler_image *u) {
    const tex_t *t; const view_t *v; const vkfmt_t *f = NULL; const char *why = NULL; int registered = 0, ok = 0;
    u->tid = (int)(handle & 0xfffff);
    if (!u->tid) { why = "zero texture ID"; goto done; }
    if (!vk_tex_by_id(u->tid, &v, &t) || !t) { why = "texture ID not registered"; goto done; }
    // vk_tex_by_id returns thread-local registration storage, reused by the output lookup.
    u->tex = *t; u->has_view = v != NULL; if (v) u->view = *v; t = &u->tex; v = u->has_view ? &u->view : NULL; registered = 1;
    vkimg_t *im = u->im = vk_tex(t);
    if (!im) { why = "native image unavailable"; goto done; }
    if (im->aspect != VK_IMAGE_ASPECT_COLOR_BIT) { why = "non-colour aspect"; goto done; }
    if (im->samples > 1) { why = "multisampled image"; goto done; }
    if (im->d != 1 || (im->vtype != VK_IMAGE_VIEW_TYPE_2D && im->vtype != VK_IMAGE_VIEW_TYPE_2D_ARRAY)) { why = "native image is not 2D"; goto done; }
    if (v && v->set_target && v->target != NVN_TEXTURE_TARGET_2D && v->target != NVN_TEXTURE_TARGET_2D_ARRAY) { why = "view target is not 2D"; goto done; }
    f = vk_fmt(v && v->set_format ? v->format : t->d.format);
    if (!f) { why = "unknown view format"; goto done; }
    if (f->is_int) { why = "integer view format"; goto done; }
    if (f->compressed || f->dec) { why = "compressed or decoded view format"; goto done; }
    u->format = f->vk;
    int rgb = f->vk == VK_FORMAT_E5B9G9R9_UFLOAT_PACK32 || f->vk == VK_FORMAT_B10G11R11_UFLOAT_PACK32;
    // storage_promote preserves RGB values in RGBA16F; this is a backing change, not a guest format reinterpretation.
    // Only the upscaler opts in, and its pair guard below prevents copying the backing's arbitrary alpha into an RGBA target.
    int promotion = u->allow_promotion && im->promoted && im->fmt == VK_FORMAT_R16G16B16A16_SFLOAT && rgb && t->d.format == im->nvn_format &&
                    (!v || !v->set_format || v->format == t->d.format);
    if ((im->promoted && !promotion) || (f->vk != im->fmt && !promotion &&
        !(u->movie_output && !sampled && f->vk == VK_FORMAT_R8G8B8A8_UNORM && im->fmt == VK_FORMAT_R8G8B8A8_SRGB))) {
        why = "view/native format reinterpretation"; goto done; // only proven storage promotion or the movie compatible-copy path
    }
    u->needs_opaque_alpha = promotion;
    if (sampled) { // blits do not apply a sampled view's channel selectors
        const VkComponentSwizzle sw[4] = {f->swz.r, f->swz.g, f->swz.b, f->swz.a};
        for (int c = 0; c < 4; c++) {
            if (sw[c] != VK_COMPONENT_SWIZZLE_IDENTITY && sw[c] != (VkComponentSwizzle)(VK_COMPONENT_SWIZZLE_R + c)) { why = "format channel swizzle"; goto done; }
            int sel = v && v->set_swizzle ? v->swz[c] : t->set_swz ? (t->swz >> (3 * c)) & 7 : c + 2;
            if (sel != c + 2 && !(c == 3 && sel == 1 && rgb)) { why = "texture/view channel swizzle"; goto done; }
        }
    }
    u->level = v && v->set_levels ? v->base_level : 0; u->layer = v && v->set_layers ? v->min_layer : 0;
    if (u->level >= im->levels || u->level >= 32 || u->layer >= im->layers) { why = "view mip/layer outside native image"; goto done; }
    u->w = im->w >> u->level; if (!u->w) u->w = 1; u->h = im->h >> u->level; if (!u->h) u->h = 1;
    ok = 1;
done:
    upscaler_image_diag(handle, sampled, u, registered, f, why);
    return ok;
}
static int upscaler_alpha_compatible(const upscaler_image *in, const upscaler_image *out) {
    return !in->needs_opaque_alpha || (out->im && (out->im->fmt == VK_FORMAT_B10G11R11_UFLOAT_PACK32 || out->im->fmt == VK_FORMAT_E5B9G9R9_UFLOAT_PACK32));
}
// NDSR c[24] (S.push[1]): origin, then inclusive last texel relative to that origin; input rectangle followed by output.
// This is the same layout decoded by up_mfx_rect in nvn_mtl.m. Check in 64 bits before forming Vulkan's signed offsets.
static int upscaler_rect(const uint32_t v[4], uint32_t w, uint32_t h, VkOffset3D offsets[2]) {
    uint64_t x1 = (uint64_t)v[0] + v[2] + 1, y1 = (uint64_t)v[1] + v[3] + 1;
    if (x1 > w || y1 > h || x1 > INT32_MAX || y1 > INT32_MAX) return 0;
    offsets[0] = (VkOffset3D){(int32_t)v[0], (int32_t)v[1], 0}; offsets[1] = (VkOffset3D){(int32_t)x1, (int32_t)y1, 1};
    return 1;
}
// Profile metadata names shaders whose output encoding has been audited as
// linear. This does not claim that a spatial filter reproduces the neural luma.
// Parse once: missing or malformed metadata declines cubic for the whole list.
static int upscaler_linear_contract(uint64_t key) {
    static uint64_t keys[64]; static int count = -1;
    if (count < 0) {
        count = 0; const char *p = getenv("DM_VK_UPSCALER_LINEAR_CS");
        while (p && *p) {
            uint64_t value = 0;
            if (count == 64) { count = 0; break; }
            int i;
            for (i = 0; i < 16; i++) {
                unsigned char c = (unsigned char)p[i]; unsigned digit;
                if (c >= '0' && c <= '9') digit = c - '0';
                else if (c >= 'a' && c <= 'f') digit = c - 'a' + 10;
                else if (c >= 'A' && c <= 'F') digit = c - 'A' + 10;
                else break;
                value = (value << 4) | digit;
            }
            if (i != 16 || (p[16] && p[16] != ',') || (p[16] == ',' && !p[17])) { count = 0; break; }
            keys[count++] = value; p += 16; if (*p == ',') p++;
        }
        if (count) VKLOG("upscaler network: %d audited linear-output variants (DM_VK_UPSCALER_LINEAR_CS)", count);
    }
    for (int i = 0; i < count; i++) if (keys[i] == key) return 1;
    return 0;
}
// Opt-in Catmull-Rom A/B. Preserve HDR; the audited families clamp their
// reconstructed neural luma, not final RGB. Encoded-output siblings stay linear-blit.
static VkFilter upscaler_filter(uint64_t key, int requested, VkFormatFeatureFlags source_features,
    const upscaler_image *in, const upscaler_image *out, const uint32_t rects[8], const char **why) {
    *why = "disabled";
    if (!requested) return VK_FILTER_LINEAR;
    *why = "EXT_filter_cubic unavailable";
    if (!V.has_filter_cubic) return VK_FILTER_LINEAR;
    *why = "unaudited shader colour contract";
    if (!upscaler_linear_contract(key)) return VK_FILTER_LINEAR;
    *why = "source format lacks cubic filtering";
    if (!(source_features & VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_CUBIC_BIT_EXT)) return VK_FILTER_LINEAR;
    *why = "only single-level 2D R11G11B10F images are audited";
    const upscaler_image *images[2] = {in, out};
    for (int i = 0; i < 2; i++) {
        const upscaler_image *u = images[i]; const vkimg_t *im = u->im;
        if (!im || im->fmt != VK_FORMAT_B10G11R11_UFLOAT_PACK32 || im->vtype != VK_IMAGE_VIEW_TYPE_2D ||
            im->levels != 1 || im->layers != 1 || im->d != 1 || im->samples != 1 ||
            u->level || u->layer || u->w != im->w || u->h != im->h) return VK_FILTER_LINEAR;
        if (u->has_view && ((u->view.set_levels && u->view.nlevels != 1) ||
                           (u->view.set_layers && u->view.nlayers != 1) ||
                           (u->view.set_target && u->view.target != NVN_TEXTURE_TARGET_2D))) return VK_FILTER_LINEAR;
        // Vulkan must clamp blits to the image edge, but may also clamp to the
        // source region. Full-image rectangles make the two interpretations equal.
        *why = "cubic requires full input and output rectangles";
        const uint32_t *r = rects + 4 * i;
        if (r[0] || r[1] || (uint64_t)r[2] + 1 != u->w || (uint64_t)r[3] + 1 != u->h) return VK_FILTER_LINEAR;
        *why = "only single-level 2D R11G11B10F images are audited";
    }
    *why = "cubic is only audited for enlargement in both axes";
    if (out->w <= in->w || out->h <= in->h) return VK_FILTER_LINEAR;
    *why = NULL; return VK_FILTER_CUBIC_EXT;
}
// The audited signed-square variant's extra coordinate transform is identity.
// A spatial replacement has not implemented arbitrary transforms; preserve the native dispatch when these change.
static int upscaler_square_coordinates(const void *push, uint32_t iw, uint32_t ih, uint32_t ow, uint32_t oh) {
    const uint8_t *p = push; uint32_t affine[4], output[2]; float source_recip[2], output_recip[2];
    memcpy(affine, p + 0x50, sizeof affine); memcpy(output, p + 0x60, sizeof output);
    memcpy(source_recip, p + 0x20, sizeof source_recip); memcpy(output_recip, p + 0x68, sizeof output_recip);
    if (!iw || !ih || !ow || !oh || affine[0] != 0x3f800000u || affine[1] != 0x3f800000u ||
        (affine[2] & 0x7fffffffu) || (affine[3] & 0x7fffffffu) || output[0] != ow || output[1] != oh) return 0;
    const uint32_t sizes[4] = {iw, ih, ow, oh};
    const float actual[4] = {source_recip[0], source_recip[1], output_recip[0], output_recip[1]};
    for (int i = 0; i < 4; i++) {
        float expected = 1.0f / (float)sizes[i];
        // Four float epsilons tolerate reciprocal calculation rounding, not a different coordinate mapping.
        if (!isfinite(actual[i]) || fabsf(actual[i] - expected) > expected * 0x1p-21f) return 0;
    }
    return 1;
}
static int upscaler_replace(vkshader_t *cs, const p_dispatch *d) {
    const char *why = "binding contract"; upscaler_image in = {.allow_promotion = 1}, out = {.allow_promotion = 1}; uint32_t rects[8]; memcpy(rects, S.push[1], sizeof rects);
    if (!upscaler_contract(cs)) goto keep;
    why = "indirect or empty dispatch"; if (d->ind || d->x <= 0 || d->y <= 0 || d->z != 1) goto keep;
    why = "input/output image or view";
    if (!upscaler_image_at(vk_tex_slot_handle(S.tex[5], S.img[5], 0), 1, &in) || !upscaler_image_at(S.img[5][1], 0, &out) || in.im == out.im) goto keep;
    why = "promoted RGB source requires an RGB destination (backing alpha is not defined)";
    if (!upscaler_alpha_compatible(&in, &out)) goto keep;
    VkImageBlit bl = {.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, in.level, in.layer, 1}, .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, out.level, out.layer, 1}};
    why = "input/output rectangle";
    if (!upscaler_rect(rects, in.w, in.h, bl.srcOffsets) || !upscaler_rect(rects + 4, out.w, out.h, bl.dstOffsets) || rects[6] < rects[2] || rects[7] < rects[3]) goto keep;
    VkFilter filter = VK_FILTER_LINEAR;
    const char *replacement = "bilinear";
    if (vk_upscale_square_key(cs->key)) {
        why = "signed-square variant has an unsupported coordinate transform";
        if (!upscaler_square_coordinates(S.push[1], in.w, in.h, out.w, out.h)) goto keep;
        // The small shader samples exactly the selected subresource; inherited mip/layer ranges cannot change its coordinates.
        view_t selected = in.has_view ? in.view : (view_t){0};
        selected.set_levels = 1; selected.base_level = in.level; selected.nlevels = 1;
        selected.set_layers = 1; selected.min_layer = in.layer; selected.nlayers = 1;
        selected.set_target = 1; selected.target = NVN_TEXTURE_TARGET_2D;
        vkimg_t *resolved = NULL;
        VkImageView source_view = vk_tex_view(&in.tex, &selected, 1, 0, 0, &resolved);
        why = "signed-square spatial contract or resources";
        if (!source_view || resolved != in.im || !vk_upscale_square(in.im, source_view, out.im, &bl)) goto keep;
        replacement = "bilinear encoded RGB, then signed-square (no neural residual/sharpening)";
        goto upscale_done;
    }
    // Format capability queries only when the pair changes, not once per frame.
    static VkFormat last_in, last_out; static int formats_ok; static VkFormatFeatureFlags source_features;
    if (last_in != in.im->fmt || last_out != out.im->fmt) { VkFormatProperties a, b;
        last_in = in.im->fmt; last_out = out.im->fmt; vkGetPhysicalDeviceFormatProperties(V.pd, last_in, &a); vkGetPhysicalDeviceFormatProperties(V.pd, last_out, &b);
        VkFormatFeatureFlags need = VK_FORMAT_FEATURE_BLIT_SRC_BIT | VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT;
        formats_ok = (a.optimalTilingFeatures & need) == need && (b.optimalTilingFeatures & VK_FORMAT_FEATURE_BLIT_DST_BIT);
        source_features = a.optimalTilingFeatures;
    }
    why = "unsupported image blit";
    if (!formats_ok || !(in.im->usage & VK_IMAGE_USAGE_TRANSFER_SRC_BIT) || !(out.im->usage & VK_IMAGE_USAGE_TRANSFER_DST_BIT)) goto keep;
    const char *ce = getenv("DM_VK_UPSCALER_CUBIC"), *cubic_reason;
    int cubic_requested = ce && atoi(ce);
    filter = upscaler_filter(cs->key, cubic_requested, source_features, &in, &out, rects, &cubic_reason);
    replacement = filter == VK_FILTER_CUBIC_EXT ? "hardware Catmull-Rom cubic" : "bilinear";
    if (cubic_requested && filter == VK_FILTER_LINEAR && upscaler_note(cs->key, 8))
        VKLOG("upscaler network %016llx: cubic declined (%s); retaining bilinear", (unsigned long long)cs->key, cubic_reason);
    vk_end_pass_land(); VkCommandBuffer cb = vk_cmd(); vk_tex_prepare(cb, in.im); vk_tex_prepare(cb, out.im); vk_img_barrier_all(cb);
    vkCmdBlitImage(cb, in.im->img, VK_IMAGE_LAYOUT_GENERAL, out.im->img, VK_IMAGE_LAYOUT_GENERAL, 1, &bl, filter);
upscale_done:
    {   // the game's render size (its dynamic resolution) going into the upscaler, over 600 upscales: the pause menu's Resolution level is judged by it
        static unsigned n, wmin, wmax, hmin, hmax; static double ws, hs; unsigned w = rects[2] + 1, h = rects[3] + 1;
        if (!n || w < wmin) { wmin = w; }
        if (!n || h < hmin) { hmin = h; }
        if (w > wmax) { wmax = w; }
        if (h > hmax) { hmax = h; }
        ws += w; hs += h;
        if (++n == 600) { VKLOG("upscaler input (the game's render size) over 600 frames: %ux%u to %ux%u, mean %.0fx%.0f; output %ux%u", wmin, hmin, wmax, hmax, ws / n, hs / n, rects[6] + 1, rects[7] + 1); n = 0; wmax = hmax = 0; ws = hs = 0; }
    }
    out.im->gpu_w = 1; vk_img_barrier_all(vk_cmd());
    if (upscaler_note(cs->key, filter == VK_FILTER_CUBIC_EXT ? 4 : 1)) {
        VKLOG("upscaler network %016llx: %s; input slot 0 id %d fmt %d view mip %u layer %u (%ux%u), rect %u,%u + %ux%u; output slot 1 id %d fmt %d view mip %u layer %u (%ux%u), rect %u,%u + %ux%u",
              (unsigned long long)cs->key, replacement, in.tid, in.im->fmt, in.level, in.layer, in.w, in.h, rects[0], rects[1], rects[2] + 1, rects[3] + 1,
              out.tid, out.im->fmt, out.level, out.layer, out.w, out.h, rects[4], rects[5], rects[6] + 1, rects[7] + 1);
    }
    return 1;
keep:
    if (upscaler_note(cs->key, 2)) VKLOG("upscaler network %016llx: keeping shader (%s); ids %d/%d; raw input rect %u,%u,%u,%u output %u,%u,%u,%u",
        (unsigned long long)cs->key, why, in.tid, out.tid, rects[0], rects[1], rects[2], rects[3], rects[4], rects[5], rects[6], rects[7]);
    return 0;
}
// Native-upscaler A/B: -1 runs continuously without per-dispatch diagnostics;
// positive N retains the traced countdown, and 0 uses the profile replacement.
static int upscaler_native_dispatch(uint64_t key, int *trace) {
    const char *native = getenv("DM_VK_UPSCALER_NATIVE"); int left = native ? atoi(native) : 0;
    *trace = 0;
    if (left == -1) return 1;
    if (left <= 0) return 0;
    *trace = 1; char next[24]; snprintf(next, sizeof next, "%d", left - 1); setenv("DM_VK_UPSCALER_NATIVE", next, 1);
    VKLOG("upscaler network %016llx: original compute dispatch (%d diagnostic dispatches remaining)", (unsigned long long)key, left - 1);
    return 1;
}
// ---- a compute stage replaced by a copy of its sampled texture into its storage image (DM_VK_PASSTHRU_CS=<key>,…): the output stages of a movie enhancement
// network whose layers the driver cannot build (Ocarina of Time's NDVE: ssbo activations + the decoded picture in a texture → the enhanced picture in an image).
// The picture goes through unenhanced: the blit scales the whole texture over the whole image. Any binding set is accepted as long as the shader has exactly one
// sampled 2D texture and one storage 2D image; a dispatch the blit cannot serve falls through (to the skip list, or the shader)
static uint64_t g_pt_keys[64]; static uint8_t g_pt_noted[64]; static int g_pt_n = -1;
static int passthru_kernel(uint64_t key) {
    if (g_pt_n < 0) { g_pt_n = 0; const char *e = getenv("DM_VK_PASSTHRU_CS");
        for (const char *q = e; q && *q && g_pt_n < 64; ) { char *end; uint64_t k = strtoull(q, &end, 16); if (end == q) break; g_pt_keys[g_pt_n++] = k; q = *end == ',' ? end + 1 : end; }
        if (g_pt_n) VKLOG("[RED] %d compute shaders replaced by a texture → image copy (DM_VK_PASSTHRU_CS): their pictures go through unprocessed", g_pt_n); }
    for (int i = 0; i < g_pt_n; i++) if (g_pt_keys[i] == key) return 1;
    return 0;
}
static int passthru_note(uint64_t key, uint8_t bit) { for (int i = 0; i < g_pt_n; i++) if (g_pt_keys[i] == key) { int first = !(g_pt_noted[i] & bit); g_pt_noted[i] |= bit; return first; } return 0; }
static int passthru_replace(vkshader_t *cs, const p_dispatch *d) {
    const char *why = "binding contract"; int tregs[2] = {-1, -1}, nt = 0, ireg = -1, treg = -1;
    for (int i = 0; i < cs->nbind; i++) { const dmsv_bind *b = &cs->b[i];
        if (b->kind == RK_TEX && b->desc == RD_SAMPLED) { if (nt >= 2 || b->dim != 1 || b->count != 1 || b->reg > 31) goto keep; tregs[nt++] = b->reg; }
        else if (b->kind == RK_IMG && b->desc == RD_STORAGE) { if (ireg >= 0 || b->dim != 1 || b->count != 1) goto keep; ireg = b->reg; } }
    if (!nt || ireg < 0 || ireg > 15) goto keep;
    why = "indirect dispatch"; if (d->ind) goto keep;
    int movie = vk_movie_fill_kernel(cs->key);
    upscaler_image in = {0}, out = {.movie_output = movie}; why = "input/output image or view";
    if (!upscaler_image_at(S.img[5][ireg], 0, &out)) goto keep;
    // DM_VK_MOVIE_FILL_CS names it: the decoded movie picture itself goes into its output image (movie_fill_vk.c), in place of the copy of a skipped layer's black output
    if (movie && vk_movie_fill(out.im, (uint32_t)out.format, out.level, out.layer)) {
        if (vk_movie_fill_note(cs->key)) VKLOG("movie fill %016llx: the decoded picture → image slot %d id %d fmt %d %ux%u", (unsigned long long)cs->key, ireg, out.tid, out.im->fmt, out.w, out.h);
        return 1; }
    // A failed movie bridge must not relax the ordinary passthrough's format contract.
    if (out.format != out.im->fmt) { why = "movie format bridge unavailable"; goto keep; }
    // two textures (a stage blending the network's output with the original picture): the one the size of the output, else the lower slot
    for (int k = 0; k < nt; k++) { upscaler_image c = {0};
        if (upscaler_image_at(vk_tex_slot_handle(S.tex[5], S.img[5], tregs[k]), 1, &c) && c.im != out.im && (treg < 0 || (c.w == out.w && c.h == out.h && !(in.w == out.w && in.h == out.h)))) { in = c; treg = tregs[k]; } }
    if (treg < 0) goto keep;
    static VkFormat last_in, last_out; static int formats_ok;
    if (last_in != in.im->fmt || last_out != out.im->fmt) { VkFormatProperties a, b;
        last_in = in.im->fmt; last_out = out.im->fmt; vkGetPhysicalDeviceFormatProperties(V.pd, last_in, &a); vkGetPhysicalDeviceFormatProperties(V.pd, last_out, &b);
        VkFormatFeatureFlags need = VK_FORMAT_FEATURE_BLIT_SRC_BIT | VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT;
        formats_ok = (a.optimalTilingFeatures & need) == need && (b.optimalTilingFeatures & VK_FORMAT_FEATURE_BLIT_DST_BIT); }
    why = "unsupported image blit";
    if (!formats_ok || !(in.im->usage & VK_IMAGE_USAGE_TRANSFER_SRC_BIT) || !(out.im->usage & VK_IMAGE_USAGE_TRANSFER_DST_BIT)) goto keep;
    VkImageBlit bl = {.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, in.level, in.layer, 1}, .srcOffsets = {{0, 0, 0}, {(int32_t)in.w, (int32_t)in.h, 1}},
                      .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, out.level, out.layer, 1}, .dstOffsets = {{0, 0, 0}, {(int32_t)out.w, (int32_t)out.h, 1}}};
    vk_end_pass_land(); VkCommandBuffer cb = vk_cmd(); vk_tex_prepare(cb, in.im); vk_tex_prepare(cb, out.im); vk_img_barrier_all(cb);
    vkCmdBlitImage(cb, in.im->img, VK_IMAGE_LAYOUT_GENERAL, out.im->img, VK_IMAGE_LAYOUT_GENERAL, 1, &bl, VK_FILTER_LINEAR);
    out.im->gpu_w = 1; vk_img_barrier_all(cb);
    if (passthru_note(cs->key, 1)) VKLOG("passthrough %016llx: texture slot %d id %d fmt %d %ux%u → image slot %d id %d fmt %d %ux%u (%dx%dx%d groups)",
        (unsigned long long)cs->key, treg, in.tid, in.im->fmt, in.w, in.h, ireg, out.tid, out.im->fmt, out.w, out.h, d->x, d->y, d->z);
    return 1;
keep:
    if (passthru_note(cs->key, 2)) VKLOG("passthrough %016llx: not served (%s): the dispatch goes on to the skip list or the shader", (unsigned long long)cs->key, why);
    return 0;
}
// ---- dispatches left out (DM_VK_SKIP_CS=<key>,<key>…, a bring-up switch): a compute shader the device's driver cannot build (Ocarina of Time's NDVE movie
// networks, 1.3-1.8 MB of SPIR-V with 500-1,000 subgroup shuffles: Adreno 830's compiler refuses the first and throws std::bad_alloc on the second, which ends
// the app) is neither built nor run; the first dispatch of each named shader logs its binding contract (textures, images, storage buffers, group counts), the
// facts a replacement (upscaler_replace's kind) needs
static uint64_t g_skip_keys[64]; static uint8_t g_skip_noted[64]; static int g_skip_n = -1;
static int skip_kernel(uint64_t key, const p_dispatch *d) {
    if (g_skip_n < 0) { g_skip_n = 0; const char *e = getenv("DM_VK_SKIP_CS");
        for (const char *q = e; q && *q && g_skip_n < 64; ) { char *end; uint64_t k = strtoull(q, &end, 16); if (end == q) break; g_skip_keys[g_skip_n++] = k; q = *end == ',' ? end + 1 : end; }
        if (g_skip_n) VKLOG("[RED] %d compute shaders are skipped, not run (DM_VK_SKIP_CS)", g_skip_n); }
    for (int i = 0; i < g_skip_n; i++) if (g_skip_keys[i] == key) {
        if (!g_skip_noted[i]) { g_skip_noted[i] = 1; vkshader_t *cs = S.sh[5];
            char line[1024]; int o = snprintf(line, sizeof line, "skipped dispatch %016llx %dx%dx%d%s local %ux%ux%u:", (unsigned long long)key, d->x, d->y, d->z, d->ind ? " (indirect)" : "", cs->local[0], cs->local[1], cs->local[2]);
            for (int j = 0; j < cs->nbind && o < (int)sizeof line - 48; j++) { const dmsv_bind *b = &cs->b[j];
                o += snprintf(line + o, sizeof line - (size_t)o, " [kind %d desc %d reg %d dim %d count %d flags %#x]", b->kind, b->desc, b->reg, b->dim, b->count, b->flags); }
            for (int r = 0; r < 16 && o < (int)sizeof line - 40; r++) if (S.ssbo[5][r].addr) o += snprintf(line + o, sizeof line - (size_t)o, " s%d %#llx+%#llx", r, (unsigned long long)S.ssbo[5][r].addr, (unsigned long long)S.ssbo[5][r].size);
            VKLOG("%s", line); }
        return 1;
    }
    return 0;
}
// Validate all three indirect argument words before recording any copy. A pool can have several VkBuffers (chunks), and a slab's
// buffer begins at boff. CPU memory is not a source: a preceding GPU command may generate these counts during this submission.
static int dispatch_indirect_spans(uint64_t addr, VkBuffer src[3], VkBufferCopy spans[3]) {
    if ((addr & 3) || addr > UINT64_MAX - 12) return 0;
    uint64_t base = 0, end = 0; int n = 0;
    for (unsigned i = 0; i < 3; i++) {
        uint64_t a = addr + 4 * i; vkpool_t *p = vk_pool_of(a, 4);
        if (!p || !p->buf || a < p->lo || a >= p->hi || p->hi - a < 4 || a < p->base || a >= p->end || p->end - a < 4) return 0;
        if (!i) { base = p->base; end = p->end; if (end - addr < 12) return 0; }
        else if (p->base != base || p->end != end) return 0;
        uint64_t off = a - p->lo;
        if (p->boff > UINT64_MAX - off) return 0;
        off += p->boff;
        if ((off & 3) || off > UINT64_MAX - 4) return 0;
        if (n && src[n - 1] == p->buf && spans[n - 1].srcOffset + spans[n - 1].size == off) spans[n - 1].size += 4;
        else { src[n] = p->buf; spans[n++] = (VkBufferCopy){off, 4 * i, 4}; }
    }
    return n;
}
static void dispatch(const p_dispatch *d) {
    vkshader_t *cs = S.sh[5]; int native_trace = 0;
    if (!cs) { VKRED_ONCE(60, "dispatch with no compute shader in the bundle: skipped"); return; }
    int selected_trace = cs_trace_match(cs->key);
    if (selected_trace) {
        extern uint64_t g_nvn_active_queue; uint64_t nvn_replaying_job(void);
        VKLOG("compute trace request key %016llx dispatch %u present %llu groups %dx%dx%d indirect %#llx after graphics pass %d queue %llu job %llu frame %llu fence %llu",
            (unsigned long long)cs->key, g_cs_trace_seen, (unsigned long long)g_tex_frame, d->x, d->y, d->z, (unsigned long long)d->ind, g_sub_pass - 1,
            (unsigned long long)g_nvn_active_queue, (unsigned long long)nvn_replaying_job(), (unsigned long long)vk_frame_current(), (unsigned long long)(mtl_last_fence() + 1));
    }
    if (g_ups_n && upscaler_kernel(cs->key)) {
        if (!upscaler_native_dispatch(cs->key, &native_trace) && upscaler_replace(cs, d)) {
            if (selected_trace) VKLOG("compute trace key %016llx returned: upscaler replacement", (unsigned long long)cs->key); return; }
    }
    if (g_pt_n && passthru_kernel(cs->key) && passthru_replace(cs, d)) {
        if (selected_trace) VKLOG("compute trace key %016llx returned: passthrough/movie replacement", (unsigned long long)cs->key); return; }
    if (skip_kernel(cs->key, d)) { if (selected_trace) VKLOG("compute trace key %016llx returned: DM_VK_SKIP_CS", (unsigned long long)cs->key); return; }
    native_trace |= selected_trace;
    VkBuffer src[3]; VkBufferCopy spans[3]; int nspans = 0;   // the indirect counts' source ranges (upstream #206); none: ours' direct indirect dispatch below
    if (d->ind && !(nspans = dispatch_indirect_spans(d->ind, src, spans))) VKRED_ONCE(61, "indirect dispatch arguments are not one aligned memory pool range: %#llx (dispatched from it directly; the shader reads 0 group counts)", (unsigned long long)d->ind);
    g_vkst.dispatches++;
    vk_end_pass_land();
    stage_storage_prepare(5, cs);
    VkCommandBuffer cb = vk_cmd();
    VkDescriptorSetLayout dsl[VK_NSETS] = {cs->dsl}; VkPipelineLayout layout = vk_pipeline_layout(dsl, 1);
    // the group counts before the descriptors: the driver segment stage_set builds carries them at c0 + 0x200 (drv_segment), as nvn_mtl.m's does. Set after it, every
    // dispatch read the previous one's: Link's Awakening's light culling (tile = workgroup y × c0[0x200] + workgroup x) ran after a 2x1x1 dispatch, so of its 30x17
    // tiles it wrote only 0-61. Indirect counts are copied on the GPU into a fresh c0 segment below
    g_dispatch[0] = (uint32_t)d->x; g_dispatch[1] = (uint32_t)d->y; g_dispatch[2] = (uint32_t)d->z;
    if (native_trace) {
        uint8_t c0[0x1000]; drv_segment(5, 0, c0, 0); uint64_t weight_address; uint32_t weight_size;
        memcpy(&weight_address, c0 + DMSH_C0_SSBO_CS + 2 * 16, 8); memcpy(&weight_size, c0 + DMSH_C0_SSBO_CS + 2 * 16 + 8, 4);
        VKLOG("compute trace native contract %016llx: groups %dx%dx%d indirect %#llx local %ux%ux%u; c0 weights slot 2 %#llx+%u, c24 bytes %zu",
            (unsigned long long)cs->key, d->x, d->y, d->z, (unsigned long long)d->ind, cs->local[0], cs->local[1], cs->local[2],
            (unsigned long long)weight_address, weight_size, sizeof S.push[1]);
        for (int i = 0; i < cs->nbind; i++) {
            const dmsv_bind *b = &cs->b[i];
            if (b->kind == RK_SBUF && b->reg >= 0 && b->reg < 16)
                VKLOG("compute trace storage key %016llx binding %u guest slot %d address %#llx size %#llx flags %#x",
                      (unsigned long long)cs->key, b->binding, b->reg, (unsigned long long)S.ssbo[5][b->reg].addr,
                      (unsigned long long)S.ssbo[5][b->reg].size, b->flags);
        }
        for (unsigned row = 0; row < sizeof S.push[1] / 64; row++) {
            uint32_t words[16]; char line[16 * 9 + 1]; int at = 0; memcpy(words, (const uint8_t *)S.push[1] + row * 64, sizeof words);
            for (unsigned j = 0; j < 16; j++) at += snprintf(line + at, sizeof line - (size_t)at, "%s%08x", j ? " " : "", words[j]);
            VKLOG("compute trace c24 key %016llx +%#x: %s", (unsigned long long)cs->key, row * 64, line);
        }
    }
    g_up_native_trace = native_trace; g_cs_images_n = g_cs_images_copies = 0; g_cs_trace_collect = selected_trace && g_cs_trace_dump;
    g_cs_buffers_n = g_cs_buffers_reserved = 0; g_cs_buffer_collect = selected_trace && g_cs_trace_buffers;
    VkDescriptorBufferInfo indirect_c0 = {0};
    uint32_t dofs[96]; VkDescriptorSet set = stage_set(5, cs, dofs, nspans ? &indirect_c0 : NULL);
    g_up_native_trace = 0; g_cs_trace_collect = 0; g_cs_buffer_collect = 0;
    uint64_t ck[3] = {cs->key, g_num_um, g_num_im};
    pthread_mutex_lock(&g_cpipe_mu); VkPipeline p = nvn_hm_get(&g_cpipes, ck); pthread_mutex_unlock(&g_cpipe_mu);
    if (!p) {
        uint64_t tb = g_prof_on ? tick() : 0; p = compute_build(cs, ck[1], ck[2]); if (g_prof_on) g_prof_ns[7] += tick() - tb;
        pthread_mutex_lock(&g_cpipe_mu); VkPipeline had = nvn_hm_get(&g_cpipes, ck);
        if (had) { if (p != (void *)1) vkDestroyPipeline(V.dev, p, NULL); p = had; } else nvn_hm_put(&g_cpipes, ck, p);   // (compute_warm made it meanwhile)
        pthread_mutex_unlock(&g_cpipe_mu);
    }
    if (native_trace) VKLOG("compute trace native pipeline %p set %p layout %p integer views %#llx/%#llx dynamic offsets %d",
        (void *)p, (void *)set, (void *)layout, (unsigned long long)ck[1], (unsigned long long)ck[2], cs->ndyn);
    if (p == (void *)1) { if (selected_trace) VKLOG("compute trace key %016llx returned: pipeline unavailable", (unsigned long long)cs->key); return; }
    if (g_cs_images_n) cs_trace_images_copy(cb, cs->key, 0);
    if (g_cs_buffers_n) cs_trace_buffers_copy(cb, cs->key, 0);
    vk_img_barrier_all(cb);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, p);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &set, (uint32_t)cs->ndyn, dofs);
    void vk_pass_ts(int end, uint32_t w, uint32_t h, uint32_t fmt, uint32_t draws); vk_pass_ts(0, 0, 0, 0, 0);   // DM_VK_PASS_TIMES: dispatches are timed as passes
    int dispatched = 1;
    if (d->ind && nspans) {   // the counts copied on the GPU into this dispatch's c0 (+0x200, where drv_segment puts direct counts), one snapshot for the shader and the dispatch
        VkBuffer b = indirect_c0.buffer; VkDeviceSize o = indirect_c0.offset + 0x200;
        if (!b) vk_scratch(12, 4, &b, &o);   // the shader does not read c0; the dispatch still needs one contiguous GPU record
        for (int i = 0; i < nspans; i++) { spans[i].dstOffset += o; vkCmdCopyBuffer(cb, src[i], b, 1, &spans[i]); }
        VkMemoryBarrier mb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, NULL, VK_ACCESS_TRANSFER_WRITE_BIT,
                              VK_ACCESS_UNIFORM_READ_BIT | VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_INDIRECT_COMMAND_READ_BIT};
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
        vkCmdDispatchIndirect(cb, b, o);
    }
    else if (d->ind) { VkBuffer b; VkDeviceSize o; if (!(d->ind & 3) && vk_buf_at(d->ind, 12, &b, &o) && !(o & 3)) vkCmdDispatchIndirect(cb, b, o); else dispatched = 0; }   // (Vulkan: a 4-byte-aligned offset)
    else vkCmdDispatch(cb, (uint32_t)d->x, (uint32_t)d->y, (uint32_t)d->z);
    vk_pass_ts(1, (uint32_t)d->x, (uint32_t)d->y, PT_DISPATCH, (uint32_t)d->z);
    if (g_cs_images_n) cs_trace_images_copy(cb, cs->key, 1);
    if (g_cs_buffers_n && dispatched) cs_trace_buffers_copy(cb, cs->key, 1);
    if (selected_trace) VKLOG("compute trace key %016llx native dispatch %s%s", (unsigned long long)cs->key,
        dispatched ? "recorded" : "not recorded: indirect buffer unavailable", dispatched && d->ind ? " (indirect counts remain GPU-owned)" : "");
    if (!g_x_nodispbar) vk_img_barrier_all(cb);   // DM_VK_X_NODISPBAR=1 (live, a cost measurement that may draw wrong): no barrier after the dispatch
    crumb_dispatch(cs->key);
    if (g_x_tracepass == -2 || mdi_log_now()) {   // (DM_VK_TRACE_PASS=-2 lists the dispatches too, with their storage buffers: the addresses for a DM_CTL memdump)
        char sb[512]; int o = 0; for (int r = 0; r < 16 && o < (int)sizeof sb - 40; r++) if (S.ssbo[5][r].addr) o += snprintf(sb + o, sizeof sb - (size_t)o, " s%d %#llx+%#llx", r, (unsigned long long)S.ssbo[5][r].addr, (unsigned long long)S.ssbo[5][r].size);
        VKLOG("--- dispatch %016llx %dx%dx%d%s after pass %d;%s", (unsigned long long)cs->key, d->x, d->y, d->z, d->ind ? " (indirect)" : "", g_sub_pass - 1, o ? sb : " no storage buffers");
    }
}

// ---------------- NVN2 task / mesh draws without VK_EXT_mesh_shader (mesh_draw) ----------------
// The Odin 3's stock Adreno 830 driver (512.800.72) has no mesh shading, so a DrawMeshTasks (p_draw.prim DM_PRIM_MESH_TASKS; hle_nvn.c) is replayed the classic way: compute
// dispatches, then an indexed indirect draw. shader_vk.py wraps each task / mesh kernel the translator made (its stage memory in shared memory, dmsh_meshstage.py) so that it
// fills its inputs from and copies its outputs to storage buffers it declares (shader_meshwrap.py has the layouts; the backend sees them as RK_MESH_* bindings):
//   1. the task kernel, one dispatch of the draw's count workgroups (an indirect draw's {count, first} record copied in by meshprep.comp): each workgroup's payload → taskout
//   2. meshlaunch.comp, one workgroup: the payloads' word 1 (mesh workgroups to launch) summed into a launch list of {task, number among its launches} a mesh workgroup, and the
//      header the next steps read: the mesh dispatch's counts and the draw's arguments
//   3. the mesh kernel, an indirect dispatch of the total: each workgroup's vertices → vtx (max_v of them, ncomp words each), its primitives' indices → idx (max_p * verts of them,
//      32-bit, the unemitted ones degenerate), so the draw is a fixed max_p primitives a workgroup in workgroup order, as the hardware rasterizes them
//   4. the program's fragment shader (and a generated per-vertex geometry shader, bind_program) under the mesh shader's pass-through vertex shader (shader_meshwrap.vertex_shader:
//      vertex gl_VertexIndex's words from vtx → gl_Position and the mesh shader's output locations), vkCmdDrawIndexedIndirect of the header's arguments
// A program with no task shader skips 1 and 2: count mesh workgroups from first, and the header is written here. The kernels' own bindings are the host's stage 3 / 4 state
// (hle_nvn.c st_host keeps NVN2's stages 6 / 7 there; stage_set(3 / 4) and drv_segment's 0x670 / 0x770 tables). The outputs live in a GPU-only arena the host never writes,
// bumped per draw and reset each submission (the queue runs submissions in order, and every step is fenced by vk_img_barrier_all), with this submission's scratch as the overflow.
// DM_VK_MESH_EMU=0 (start-up): the draws are skipped with a note, as before. DM_VK_MESH_MAX (default 2048): the most mesh workgroups a draw with a task shader may launch, which sizes
// its buffers (meshlaunch.comp drops the rest and counts them in header word 9); DM_VK_MESH_MAX_TASKS (default 4096): the most task workgroups of an indirect draw; DM_VK_MESH_ARENA_MB (default 48)
#include "meshlaunch_comp.spv.h"
#include "meshprep_comp.spv.h"
enum { MESH_HDR = 16, MESH_HDR_BYTES = 64 };   // launch list header words (shader_meshwrap.py LAUNCH_HDR): 0-2 mesh dispatch, 4-8 draw arguments, 9 overflow, 12-14 task dispatch
static struct { VkBuffer b; VkDeviceMemory m; VkDeviceSize size, used; uint64_t epoch; int bad, noted; } g_mesh_arena;
static struct { VkPipeline launch, prep; VkPipelineLayout pl; VkDescriptorSetLayout dl; int bad; } g_mesh_k;
static uint64_t g_mesh_draws, g_mesh_skipped;
static int mesh_emu_on(void) { static int on = -1; if (on < 0) { const char *e = getenv("DM_VK_MESH_EMU"); on = !(e && !atoi(e)); if (!on) VKLOG("DM_VK_MESH_EMU=0: task / mesh draws are skipped"); } return on; }
static void *g_mesh_alloc_p; static int g_mesh_dump_on;   // DM_VK_MESH_DUMP: the last allocation's host pointer (scratch only), the arena bypassed
// A draw's outputs are read only by the draw recorded right after its kernels, in a pass that ends before the next mesh draw's kernels are recorded (vk_end_pass_land at
// mesh_draw_one's start; commands run in order). So when a draw's buffers do not fit in what is left of the arena they start over at its beginning, where earlier draws'
// outputs are finished with. Before this the rest of the submission's mesh draws took scratch at full capacity each (Ocarina of Time: 500 MB a submission slot, 2.5 GB in all)
static uint64_t g_mesh_wraps;
static void mesh_reserve(VkDeviceSize total) {
    if (g_mesh_arena.epoch != g_set_epoch || !g_mesh_arena.b || total > g_mesh_arena.size || g_mesh_arena.used + total <= g_mesh_arena.size) return;
    g_mesh_arena.used = 0; if (!g_mesh_wraps++) VKLOG("mesh emulation: the arena (%llu MB) is reused within a submission from here on", (unsigned long long)(g_mesh_arena.size >> 20));
}
static mesh_buf mesh_alloc(VkDeviceSize n) {   // GPU-only memory for this draw: the arena, else this submission's scratch
    mesh_buf r; g_mesh_alloc_p = NULL; n = (n + 255) & ~255ull; if (!n) n = 256;
    if (g_mesh_arena.epoch != g_set_epoch) { g_mesh_arena.epoch = g_set_epoch; g_mesh_arena.used = 0; }
    if (!g_mesh_arena.b && !g_mesh_arena.bad) {
        const char *e = getenv("DM_VK_MESH_ARENA_MB"); VkDeviceSize mb = e && atoi(e) > 0 ? (VkDeviceSize)atoi(e) : 48;
        VkBufferCreateInfo bi = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, NULL, 0, mb << 20, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_SHARING_MODE_EXCLUSIVE};
        VkMemoryRequirements rq; VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        if (vkCreateBuffer(V.dev, &bi, NULL, &g_mesh_arena.b)) { g_mesh_arena.b = VK_NULL_HANDLE; g_mesh_arena.bad = 1; }
        else { vkGetBufferMemoryRequirements(V.dev, g_mesh_arena.b, &rq); ai.allocationSize = rq.size; ai.memoryTypeIndex = vk_mem_type(rq.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0);
            if (vkAllocateMemory(V.dev, &ai, NULL, &g_mesh_arena.m) || vkBindBufferMemory(V.dev, g_mesh_arena.b, g_mesh_arena.m, 0)) { vkDestroyBuffer(V.dev, g_mesh_arena.b, NULL); g_mesh_arena.b = VK_NULL_HANDLE; g_mesh_arena.bad = 1; }
            else { g_mesh_arena.size = mb << 20; VKLOG("mesh emulation: %llu MB arena for the kernels' outputs", (unsigned long long)mb); } }
        if (g_mesh_arena.bad) VKLOG("[RED] mesh emulation: no arena (vkCreateBuffer / vkAllocateMemory failed); the outputs go to the submission's scratch memory");
    }
    if (!g_mesh_dump_on && g_mesh_arena.b && g_mesh_arena.used + n <= g_mesh_arena.size) { r.b = g_mesh_arena.b; r.off = g_mesh_arena.used; r.len = n; g_mesh_arena.used += n; return r; }
    if (g_mesh_arena.b && !g_mesh_arena.noted++) VKLOG("mesh emulation: the arena (%llu MB) is full this submission; the rest of its mesh draws use scratch memory (DM_VK_MESH_ARENA_MB)", (unsigned long long)(g_mesh_arena.size >> 20));
    g_mesh_alloc_p = vk_scratch(n, 256, &r.b, &r.off); r.len = n; return r;
}
static int mesh_kernels(void) {   // meshlaunch.comp and meshprep.comp: three storage buffers, 32 bytes of push constants
    if (g_mesh_k.launch || g_mesh_k.bad) return !g_mesh_k.bad;
    VkDescriptorSetLayoutBinding lb[3] = {{0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL}, {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL}, {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL}};
    VkDescriptorSetLayoutCreateInfo li = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, NULL, 0, 3, lb};
    VkPushConstantRange pr = {VK_SHADER_STAGE_COMPUTE_BIT, 0, 32};
    g_mesh_k.bad = 1;
    if (vkCreateDescriptorSetLayout(V.dev, &li, NULL, &g_mesh_k.dl)) return 0;
    void vk_dsl_register(VkDescriptorSetLayout l, const VkDescriptorSetLayoutBinding *b, uint32_t n); vk_dsl_register(g_mesh_k.dl, lb, 3);   // (vk_alloc_set counts its pool needs)
    VkPipelineLayoutCreateInfo pli = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, NULL, 0, 1, &g_mesh_k.dl, 1, &pr};
    if (vkCreatePipelineLayout(V.dev, &pli, NULL, &g_mesh_k.pl)) return 0;
    const uint32_t *code[2] = {meshlaunch_comp_spv, meshprep_comp_spv}; size_t len[2] = {sizeof meshlaunch_comp_spv, sizeof meshprep_comp_spv}; VkPipeline *out[2] = {&g_mesh_k.launch, &g_mesh_k.prep};
    for (int i = 0; i < 2; i++) {
        VkShaderModuleCreateInfo mi = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, NULL, 0, len[i], code[i]}; VkShaderModule sm;
        if (vkCreateShaderModule(V.dev, &mi, NULL, &sm)) return 0;
        VkComputePipelineCreateInfo ci = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO, NULL, 0, {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, NULL, 0, VK_SHADER_STAGE_COMPUTE_BIT, sm, "main", NULL}, g_mesh_k.pl, VK_NULL_HANDLE, -1};
        VkResult r = vkCreateComputePipelines(V.dev, V.pcache, 1, &ci, NULL, out[i]); vkDestroyShaderModule(V.dev, sm, NULL); vk_pipeline_cache_changed();   // (gfx_vk_pipeline_cache.inc: a new entry to save; upstream nx2android #213)
        if (r) { *out[i] = VK_NULL_HANDLE; return 0; }
    }
    g_mesh_k.bad = 0; return 1;
}
static void mesh_helper(VkCommandBuffer cb, VkPipeline p, const mesh_buf b[3], const uint32_t pc[8]) {   // one dispatch of a helper kernel over three buffers
    VkDescriptorSet set = vk_alloc_set(g_mesh_k.dl); VkDescriptorBufferInfo bi[3]; VkWriteDescriptorSet w[3];
    for (int i = 0; i < 3; i++) { bi[i] = (VkDescriptorBufferInfo){b[i].b, b[i].off, b[i].len}; w[i] = (VkWriteDescriptorSet){VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, NULL, set, (uint32_t)i, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, NULL, &bi[i], NULL}; }
    vkUpdateDescriptorSets(V.dev, 3, w, 0, NULL);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, p); vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, g_mesh_k.pl, 0, 1, &set, 0, NULL);
    vkCmdPushConstants(cb, g_mesh_k.pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, 32, pc); vkCmdDispatch(cb, 1, 1, 1); vk_img_barrier_all(cb);
}
static VkPipeline mesh_cpipe(vkshader_t *k) {   // the kernel's compute pipeline, from the dispatch cache (compute_build; (VkPipeline)1 when it failed)
    uint64_t ck[3] = {k->key, g_num_um, g_num_im};
    pthread_mutex_lock(&g_cpipe_mu); VkPipeline p = nvn_hm_get(&g_cpipes, ck); pthread_mutex_unlock(&g_cpipe_mu);
    if (!p) { p = compute_build(k, ck[1], ck[2]);
        pthread_mutex_lock(&g_cpipe_mu); VkPipeline had = nvn_hm_get(&g_cpipes, ck); if (had) { if (p != (void *)1) vkDestroyPipeline(V.dev, p, NULL); p = had; } else nvn_hm_put(&g_cpipes, ck, p); pthread_mutex_unlock(&g_cpipe_mu); }
    return p;
}
static int mesh_kernel(VkCommandBuffer cb, int st, vkshader_t *k, const mesh_buf *ind, VkDeviceSize ind_at, uint32_t groups) {   // the task (st 3) or mesh (st 4) kernel: its set from the host's stage state, a dispatch
    uint32_t dofs[96]; VkDescriptorSet set = stage_set(st, k, dofs, NULL);
    VkPipeline p = mesh_cpipe(k); if (p == (VkPipeline)1) return 0;
    VkDescriptorSetLayout dsl[VK_NSETS] = {k->dsl}; VkPipelineLayout layout = vk_pipeline_layout(dsl, 1);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, p); vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &set, (uint32_t)k->ndyn, dofs);
    if (ind) vkCmdDispatchIndirect(cb, ind->b, ind->off + ind_at); else vkCmdDispatch(cb, groups, 1, 1);
    vk_img_barrier_all(cb); return 1;
}
// A task-less indirect mesh draw's count is written by the GPU and has no bound here: its workgroups were capped at DM_VK_MESH_MAX (2048) and the rest dropped
// without a word. It is drawn as slices of up to DM_VK_MESH_MAX workgroups each, meshprep.comp giving each slice its part of the count. Every slice has buffers
// of the full capacity (4 slices of every such draw took Ocarina of Time's world load over the Odin's memory), so the number of slices follows the count the
// record holds now, which is the one the GPU wrote for an earlier frame (the record is pool memory, mapped here), with half as much again for growth, up to
// DM_VK_MESH_BATCHES (start-up, default 8). The log has each new highest count
static void mesh_draw_one(const p_draw *d, uint32_t batch);
static void mesh_draw(const p_draw *d) {
    static uint32_t batches, cap_m, top; if (!batches) { const char *e = getenv("DM_VK_MESH_BATCHES"); batches = e && atoi(e) > 0 ? (uint32_t)atoi(e) : 8; if (batches > 64) batches = 64;
        e = getenv("DM_VK_MESH_MAX"); cap_m = e && atoi(e) > 0 ? (uint32_t)atoi(e) : 2048; }
    vkshader_t *ts = S.sh[3]; int taskless = !(ts && ts->mesh_kind == 1 && S.sh[4] && S.sh[4]->mesh_task);
    uint32_t n = 1;
    if (d->ind && taskless) { VkBuffer rb; VkDeviceSize ro; uint32_t seen = vk_buf_at(d->ind, 8, &rb, &ro) ? *(const volatile uint32_t *)(uintptr_t)d->ind : 0;
        uint64_t want = ((uint64_t)seen + seen / 2 + cap_m - 1) / cap_m; n = want < 1 ? 1 : want > batches ? batches : (uint32_t)want;
        if (seen > top) { top = seen; VKLOG("mesh draw without a task shader: indirect count %u (capacity %u a slice, %u slices%s); mesh %016llx fragment %016llx", seen, cap_m, n,
            want > batches ? ", NOT ENOUGH: DM_VK_MESH_BATCHES" : "", S.sh[4] ? (unsigned long long)S.sh[4]->key : 0ull, S.sh[1] ? (unsigned long long)S.sh[1]->key : 0ull); } }
    for (uint32_t b = 0; b < n; b++) mesh_draw_one(d, b);
}
static void mesh_draw_one(const p_draw *d, uint32_t batch) {
    vkshader_t *ts = S.sh[3], *ms = S.sh[4];
    if (!mesh_emu_on()) { g_mesh_skipped++; VKRED_ONCE(160, "task / mesh draw skipped (DM_VK_MESH_EMU=0)"); return; }
    if (ts && ts->mesh_kind != 1) ts = NULL;
    if (!ms || ms->mesh_kind != 2) { g_mesh_skipped++; VKRED_ONCE(161, "task / mesh draw without its mesh kernel in the bundle (task %016llx mesh %016llx): skipped", ts ? (unsigned long long)ts->key : 0ull, ms ? (unsigned long long)ms->key : 0ull); return; }
    if (ms->mesh_task && !ts) { g_mesh_skipped++; VKRED_ONCE(162, "mesh shader %016llx expects a task shader the program has not: skipped", (unsigned long long)ms->key); return; }
    if (!ms->mesh_task) ts = NULL;
    vkshader_t *vs = vk_shader_mesh_vs(ms);
    if (!vs) { g_mesh_skipped++; VKRED_ONCE(163, "mesh shader %016llx has no pass-through vertex shader in the bundle: skipped", (unsigned long long)ms->key); return; }
    if (!mesh_kernels()) { g_mesh_skipped++; VKRED_ONCE(164, "mesh emulation: the helper kernels failed to build: task / mesh draws skipped"); return; }
    if (!d->ind && d->count <= 0) return;
    static uint32_t cap_m, cap_t; if (!cap_m) { const char *e = getenv("DM_VK_MESH_MAX"); cap_m = e && atoi(e) > 0 ? (uint32_t)atoi(e) : 2048; e = getenv("DM_VK_MESH_MAX_TASKS"); cap_t = e && atoi(e) > 0 ? (uint32_t)atoi(e) : 4096; }
    uint32_t count = d->ind ? 0 : (uint32_t)d->count, first = d->ind ? 0 : (uint32_t)d->first, verts = ms->mesh_verts ? ms->mesh_verts : 3, ipw = ms->mesh_max_p * verts;
    uint32_t ntask = ts ? (d->ind ? cap_t : count) : 0, nmesh = ts ? cap_m : (d->ind ? cap_m : count), pay = ts ? ts->mesh_pay : ms->mesh_pay;
    if (!ntask && !nmesh) return;
    vk_end_pass_land();   // the dispatches run outside the pass (prepare_draw begins it again for the draw)
    VkCommandBuffer cb = vk_cmd();
    // the buffers: control words (host-written; an indirect draw's record copied in by meshprep.comp), the task payloads, the launch list, the vertices and indices
    mesh_buf ctl; uint32_t *cw = vk_scratch(64, 256, &ctl.b, &ctl.off); ctl.len = 64; memset(cw, 0, 64); cw[0] = first; cw[1] = ts ? ntask : nmesh;
    // DM_VK_MESH_DUMP=n (live): the next n draws with a task shader keep their buffers in host-visible scratch; three submissions later the log has what the kernels wrote
    static struct { uint64_t epoch, mk, tk, frame; int pass; uint32_t ntask, nmesh, pay, ncomp, maxv, ipw; const uint32_t *t, *l, *v, *i; } md[16]; static int nmd, md_left;
    { const char *e = getenv("DM_VK_MESH_DUMP"); if (e && atoi(e) > 0 && !md_left && !nmd) { md_left = atoi(e) > 16 ? 16 : atoi(e); setenv("DM_VK_MESH_DUMP", "0", 1); } }
    for (int q = 0; q < nmd; q++) if (md[q].epoch && vk_fence_done() >= md[q].epoch) {   // (epoch: the fence of the submission that ran the kernels)
        const uint32_t *L = md[q].l, *T = md[q].t; const float *Vf = (const float *)md[q].v; uint32_t tot = L[0] < md[q].nmesh ? L[0] : md[q].nmesh, nz = 0, sum = 0, mx = 0;
        for (uint32_t t = 0; t < md[q].ntask; t++) { uint32_t c = T[t * md[q].pay + 1]; if (c) nz++; sum += c; if (c > mx) mx = c; }
        VKLOG("mesh dump: task %016llx mesh %016llx pass %d: header %u %u %u %u | %u %u %u %u %u | dropped %u; payload %u words, %u of %u tasks launch (sum %u, most %u); payload[0] %08x %08x %08x %08x %08x %08x %08x %08x",
              (unsigned long long)md[q].tk, (unsigned long long)md[q].mk, md[q].pass, L[0], L[1], L[2], L[3], L[4], L[5], L[6], L[7], L[8], L[9], md[q].pay, nz, md[q].ntask, sum, mx, T[0], T[1], T[2], T[3], T[4], T[5], T[6], T[7]);
        uint32_t nv = tot * md[q].maxv, w0 = 0, nan = 0, in = 0, zero = 0; float lo[3] = {1e30f, 1e30f, 1e30f}, hi[3] = {-1e30f, -1e30f, -1e30f};
        for (uint32_t k = 0; k < nv; k++) { const float *P = Vf + (size_t)k * md[q].ncomp; if (P[0] != P[0] || P[1] != P[1] || P[3] != P[3]) { nan++; continue; } if (P[0] == 0 && P[1] == 0 && P[2] == 0 && P[3] == 0) { zero++; continue; } if (P[3] <= 0) { w0++; continue; }
            float x = P[0] / P[3], y = P[1] / P[3], z = P[2] / P[3]; if (x >= -1 && x <= 1 && y >= -1 && y <= 1) in++; if (x < lo[0]) lo[0] = x; if (x > hi[0]) hi[0] = x; if (y < lo[1]) lo[1] = y; if (y > hi[1]) hi[1] = y; if (z < lo[2]) lo[2] = z; if (z > hi[2]) hi[2] = z; }
        VKLOG("mesh dump:   %u workgroups, %u vertices: %u all-zero, %u NaN, %u w<=0, %u on screen; ndc x %g..%g y %g..%g z %g..%g", tot, nv, zero, nan, w0, in, lo[0], hi[0], lo[1], hi[1], lo[2], hi[2]);
        for (uint32_t g = 0; g < 3 && g < tot; g++) { uint32_t wg = g == 0 ? 0 : g == 1 ? tot / 2 : tot - 1; const float *P = Vf + (size_t)wg * md[q].maxv * md[q].ncomp; const uint32_t *I = md[q].i + (size_t)wg * md[q].ipw;
            VKLOG("mesh dump:   workgroup %u (task %u, launch %u): v0 %g %g %g %g  v1 %g %g %g %g  v2 %g %g %g %g; indices %u %u %u %u %u %u .. last %u %u %u", wg, L[16 + 2 * wg], L[17 + 2 * wg], P[0], P[1], P[2], P[3],
                  P[md[q].ncomp], P[md[q].ncomp + 1], P[md[q].ncomp + 2], P[md[q].ncomp + 3], P[2 * md[q].ncomp], P[2 * md[q].ncomp + 1], P[2 * md[q].ncomp + 2], P[2 * md[q].ncomp + 3], I[0], I[1], I[2], I[3], I[4], I[5], I[md[q].ipw - 3], I[md[q].ipw - 2], I[md[q].ipw - 1]);
            for (uint32_t v = 0; v < 4 && v < md[q].maxv; v++) { char b[1024]; int n = 0; const uint32_t *U = (const uint32_t *)P + (size_t)v * md[q].ncomp; for (uint32_t c = 0; c < md[q].ncomp && n < 980; c++) n += snprintf(b + n, sizeof b - n, " %08x", U[c]); VKLOG("mesh dump:     wg %u v%u:%s", wg, v, b); } }
        md[q].epoch = 0; int any = 0; for (int z = 0; z < nmd; z++) any |= md[z].epoch != 0; if (!any && !md_left) { nmd = 0; g_mesh_dump_on = 0; } }
    int md_this = ts && md_left > 0;
    if (md_this) { const char *f = getenv("DM_VK_MESH_DUMP_MESH"); char h[20]; snprintf(h, sizeof h, "%016llx", (unsigned long long)ms->key); if (f && *f && !strstr(f, h)) md_this = 0; }   // DM_VK_MESH_DUMP_MESH=key,key: only those mesh shaders
    if (md_this) g_mesh_dump_on = 1;
    { VkDeviceSize z[4] = {ts ? (VkDeviceSize)ntask * pay * 4 : 256, MESH_HDR_BYTES + (VkDeviceSize)nmesh * 8, (VkDeviceSize)nmesh * ms->mesh_max_v * (ms->mesh_ncomp ? ms->mesh_ncomp : 1) * 4, (VkDeviceSize)nmesh * ipw * 4}, t = 0;
      for (int i = 0; i < 4; i++) t += z[i] ? (z[i] + 255) & ~255ull : 256; mesh_reserve(t); }
    mesh_buf taskout = mesh_alloc(ts ? (VkDeviceSize)ntask * pay * 4 : 256); const uint32_t *md_t = g_mesh_alloc_p; mesh_buf launch = mesh_alloc(MESH_HDR_BYTES + (VkDeviceSize)nmesh * 8); const uint32_t *md_l = g_mesh_alloc_p;
    mesh_buf vtx = mesh_alloc((VkDeviceSize)nmesh * ms->mesh_max_v * (ms->mesh_ncomp ? ms->mesh_ncomp : 1) * 4); const uint32_t *md_v = g_mesh_alloc_p; mesh_buf idx = mesh_alloc((VkDeviceSize)nmesh * ipw * 4); const uint32_t *md_i = g_mesh_alloc_p;
    if (md_this && md_t && md_l && md_v && md_i && nmd < 16) { md_left--; md[nmd].frame = g_tex_frame; md[nmd].epoch = mtl_last_fence() + 1; md[nmd].mk = ms->key; md[nmd].tk = ts->key; md[nmd].pass = g_sub_pass; md[nmd].ntask = ntask; md[nmd].nmesh = nmesh; md[nmd].pay = pay;
        md[nmd].ncomp = ms->mesh_ncomp ? ms->mesh_ncomp : 1; md[nmd].maxv = ms->mesh_max_v; md[nmd].ipw = ipw; md[nmd].t = md_t; md[nmd].l = md_l; md[nmd].v = md_v; md[nmd].i = md_i; nmd++; }
    if (d->ind) {   // {count, first} at a game address: meshprep.comp copies it into ctl and the header's dispatch arguments
        VkBuffer rb; VkDeviceSize ro; if (!vk_buf_at(d->ind, 8, &rb, &ro)) { g_mesh_skipped++; VKRED_ONCE(165, "DrawMeshTasksIndirect record outside the memory pools: skipped"); return; }
        mesh_buf rec = {rb, ro & ~255ull, (ro & 255) + 8}; uint32_t pc[8] = {(uint32_t)(ro & 255) / 4, ts ? cap_t : cap_m, ts != NULL, ipw, ts ? 0 : batch};
        mesh_helper(cb, g_mesh_k.prep, (mesh_buf[3]){rec, ctl, launch}, pc);
    } else if (!ts) {   // no task: the header is known here
        uint32_t hdr[MESH_HDR] = {nmesh, 1, 1, 0, nmesh * ipw, 1, 0, 0, 0, 0, 0, 0, nmesh, 1, 1, 0}; vkCmdUpdateBuffer(cb, launch.b, launch.off, sizeof hdr, hdr); vk_img_barrier_all(cb);
    }
    g_mesh.ctl = ctl; g_mesh.taskout = taskout; g_mesh.launch = launch; g_mesh.vtx = vtx; g_mesh.idx = idx;
    if (ts) {
        if (!mesh_kernel(cb, 3, ts, d->ind ? &launch : NULL, 12 * 4, ntask)) { g_mesh_skipped++; return; }
        uint32_t pc[8] = {pay, nmesh, ipw, 0}; mesh_helper(cb, g_mesh_k.launch, (mesh_buf[3]){ctl, taskout, launch}, pc);
    }
    if (!mesh_kernel(cb, 4, ms, ts || d->ind ? &launch : NULL, 0, nmesh)) { g_mesh_skipped++; return; }
    S.bound_pipe = VK_NULL_HANDLE;
    // the draw: the pass-through vertex shader in the vertex stage, the program's fragment stage as it is, an indexed indirect draw of the header's arguments
    vkshader_t *vs0 = S.sh[0]; S.sh[0] = vs; g_draw_binst = 0;
    g_fs_buffer_mesh = 1;
    int prim = verts == 3 ? 4 : verts == 2 ? 1 : 0, ok = prepare_draw(prim, 1);
    g_fs_buffer_mesh = 0;
    S.sh[0] = vs0;
    if (ok && g_pd_draw_mode) pass_dump_draw_geometry("mesh", prim, d->first, 0, 0, 0, -1);
    if (ok) pass_journal_draw("mesh", count, 1, d->ind, vs->key, ts ? ts->key : 0, ms->key);
    if (ok) { cb = vk_cmd(); int g_ok = draw_allowed(); bc_index(cb, idx.b, idx.off, VK_INDEX_TYPE_UINT32);
        if (!(dbg_draw() & 1) && g_ok) {
            if (pass_journal_row()) VKLOG("pd journal: mesh-emitted present=%llu submission=%llu pass=%d draw=%d task-groups=%u mesh-capacity=%u indirect-buffer=%p offset=%llu",
                (unsigned long long)g_tex_frame, (unsigned long long)g_set_epoch, g_sub_pass - 1, g_draw_i - 1, ntask, nmesh, (void *)launch.b, (unsigned long long)(launch.off + 16));
            vk_occ_draw(); vkCmdDrawIndexedIndirect(cb, launch.b, launch.off + 16, 1, 20); pass_dump_draw_checkpoint("mesh", count, 1, d->ind, vs->key, ts ? ts->key : 0, ms->key);
            if(g_mrt_active){VKLOG("Flat observer mesh issued: vs=%016llx fs=%016llx mesh=%016llx task=%016llx groups=%u capacity=%u index-buffer=%p offset=%llu indirect-buffer=%p offset=%llu",
                (unsigned long long)vs->key,(unsigned long long)S.sh[1]->key,(unsigned long long)ms->key,ts?(unsigned long long)ts->key:0ull,
                ntask,nmesh,(void*)idx.b,(unsigned long long)idx.off,(void*)launch.b,(unsigned long long)(launch.off+16));
                mrt_after_draw(0,0,(uint64_t)(uintptr_t)idx.b);}
        } }
    if(g_mrt_active)mrt_abort();
    { static int left = -1; const char *ml = getenv("DM_VK_MESH_LOG"); if (ml && atoi(ml) > 0 && left < 0) left = atoi(ml); if (left > 0) { left--;   // DM_VK_MESH_LOG=n (live): the next n mesh draws, each with its keys and pass
        VKLOG("mesh log: present %llu pass %d %ux%u: task %016llx mesh %016llx vs %016llx (outs %#x) fragment %016llx (ins %#x) workgroups %u%s", (unsigned long long)g_tex_frame, g_sub_pass - 1, S.pass_w, S.pass_h, ts ? (unsigned long long)ts->key : 0ull, (unsigned long long)ms->key,
              (unsigned long long)vs->key, vs->out_locs, S.sh[1] ? (unsigned long long)S.sh[1]->key : 0ull, S.sh[1] ? S.sh[1]->in_locs : 0, nmesh, ok ? "" : " [not prepared]");
        { char b[900]; int n = 0; n += snprintf(b + n, sizeof b - n, "arena %llu scr %d%d%d%d launch %llu ind %llx count %d first %d", (unsigned long long)g_mesh_arena.used, md_t != NULL, md_l != NULL, md_v != NULL, md_i != NULL, (unsigned long long)launch.off, (unsigned long long)d->ind, d->count, d->first);
          if (d->ind && vk_pool_of(d->ind, 1)) n += snprintf(b + n, sizeof b - n, " rec %u %u", ((const volatile uint32_t *)(uintptr_t)d->ind)[0], ((const volatile uint32_t *)(uintptr_t)d->ind)[1]);
          for (int st = 3; st <= 4; st++) { for (int r = 0; r < 4; r++) { uint64_t a = S.ssbo[st][r].addr; if (!a) continue; n += snprintf(b + n, sizeof b - n, " | s%d.%d %llx+%llx", st, r, (unsigned long long)a, (unsigned long long)S.ssbo[st][r].size);
                if (vk_pool_of(a, 1)) { const volatile uint32_t *w = (const volatile uint32_t *)(uintptr_t)a; n += snprintf(b + n, sizeof b - n, " %08x %08x %08x %08x", w[0], w[1], w[2], w[3]); } }
              for (int r = 0; r < 6; r++) { uint64_t a = S.ubo[st][r].addr; if (!a) continue; n += snprintf(b + n, sizeof b - n, " | u%d.%d %llx+%llx", st, r, (unsigned long long)a, (unsigned long long)S.ubo[st][r].size); } }
          VKLOG("mesh log:   %s", b);
          n = 0; for (int st = 1; st <= 4; st += 3) for (int r = 0; r < 8; r++) { uint64_t a = S.ubo[st][r].addr, z = S.ubo[st][r].size; if (!a) continue; uint32_t h = 2166136261u; if (z <= 65536 && vk_pool_of(a, 1) && vk_pool_of(a + z - 1, 1)) for (uint64_t q = 0; q < z; q++) h = (h ^ ((const volatile uint8_t *)(uintptr_t)a)[q]) * 16777619u;
              n += snprintf(b + n, sizeof b - n, " U%d.%d %llx+%llx #%08x", st, r, (unsigned long long)a, (unsigned long long)z, h); }
          for (int r = 0; r < 32; r++) if (S.tex[1][r]) n += snprintf(b + n, sizeof b - n, " t%d=%llx", r, (unsigned long long)S.tex[1][r]);
          VKLOG("mesh log:   %s", b); }
        if (!left) setenv("DM_VK_MESH_LOG", "0", 1); } }
    if (g_mesh_draws++ < 8 || g_mesh_draws % 2000 == 0 || g_tracing || g_x_tracepass >= 0)
        VKLOG("mesh draw %llu: task %016llx (%u workgroups%s) → mesh %016llx (up to %u workgroups of %u vertices / %u primitives, %u components) → up to %u primitives; fragment %016llx%s%s", (unsigned long long)g_mesh_draws, ts ? (unsigned long long)ts->key : 0ull,
              ts ? ntask : 0, d->ind ? ", indirect" : "", (unsigned long long)ms->key, nmesh, ms->mesh_max_v, ms->mesh_max_p, ms->mesh_ncomp, nmesh * ms->mesh_max_p, S.sh[1] ? (unsigned long long)S.sh[1]->key : 0ull,
              S.sh[2] ? " through a per-vertex geometry shader" : "", ok ? "" : " [draw not prepared]");
}

// ---------------- copies and clears ----------------
static VkImageSubresourceLayers sub_of(vkimg_t *im, const view_t *v, int has_v, int z) {
    uint32_t lv = has_v && v->set_levels ? v->base_level : 0, ly = has_v && v->set_layers ? v->min_layer : 0;
    int layered = im->layers > 1;
    return (VkImageSubresourceLayers){im->aspect & VK_IMAGE_ASPECT_COLOR_BIT ? VK_IMAGE_ASPECT_COLOR_BIT : (im->aspect & VK_IMAGE_ASPECT_DEPTH_BIT ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_STENCIL_BIT), lv, ly + (layered ? (uint32_t)z : 0), 1};
}
static void copy_b2t(const p_copy_b2t *c, int to_texture) {
    vk_end_pass_land();
    vkimg_t *im = vk_tex(c->tex); const vkfmt_t *f = vk_fmt(c->tex->d.format);
    if (!im || !f) return;
    int zc_tex_wants(const void *, uint64_t); if (to_texture && zc_tex_wants((const void *)(uintptr_t)c->src, 1)) { void dm_codec_pool_gpu_read(uint64_t, uint64_t); dm_codec_pool_gpu_read(c->src, 1); }   // a picture slot read by the GPU: the codec's CPU path fills it
    { static int lin_diag = -1; if (lin_diag < 0) { const char *e = getenv("DM_VK_LIN_DIAG"); lin_diag = e && atoi(e); }   // (gfx_vk_tex.c: the same switch)
      static uint64_t seen[16]; static int nseen;
      if (lin_diag && c->tex->d.width >= 960 && nseen < 16) { int k = 0; while (k < nseen && seen[k] != c->tex->serial) k++;
        if (k == nseen) { seen[nseen++] = c->tex->serial;
            VKLOG("buffer %s texture copy: texture ser=%llu %ux%u nvn format %#x flags %#x, buffer %#llx, region %d,%d %ux%u, row stride %lld image stride %lld",
                  to_texture ? "→" : "←", (unsigned long long)c->tex->serial, c->tex->d.width, c->tex->d.height, c->tex->d.format, c->tex->d.flags, (unsigned long long)c->src,
                  c->r.xoffset, c->r.yoffset, (unsigned)c->r.width, (unsigned)c->r.height, (long long)S.row_stride, (long long)S.image_stride); } } }
    VkBuffer b; VkDeviceSize o;
    if (im->promoted) { VKRED_ONCE(172, "buffer ↔ texture copy of an image promoted for image stores (nvn format %#x, %ux%u): skipped, its texels are not the guest's layout", c->tex->d.format, c->tex->d.width, c->tex->d.height); return; }
    vk_pool_tag = "buffer-texture copy"; if (!vk_buf_at(c->src, 1, &b, &o)) { VKRED_ONCE(70, "buffer of a buffer ↔ texture copy outside the memory pools"); return; }
    VkCommandBuffer cb = vk_cmd(); vk_tex_prepare(cb, im); vk_img_barrier_all(cb);
    unsigned bw = f->bw ? f->bw : 1, bh = f->bh ? f->bh : 1;
    uint32_t rowlen = S.row_stride > 0 && f->bpb ? (uint32_t)(S.row_stride / f->bpb) * bw : 0;
    uint32_t imgh = S.image_stride > 0 && S.row_stride > 0 ? (uint32_t)(S.image_stride / S.row_stride) * bh : 0;
    int layered = im->layers > 1, depth = c->r.depth ? c->r.depth : 1;
    VkBufferImageCopy r = {o, rowlen, imgh, sub_of(im, &c->view, c->has_view, c->r.zoffset), {c->r.xoffset, c->r.yoffset, layered ? 0 : c->r.zoffset},
                           {(uint32_t)c->r.width, (uint32_t)(c->r.height ? c->r.height : 1), layered ? 1 : (uint32_t)depth}};
    if (layered) r.imageSubresource.layerCount = (uint32_t)depth;
    if (f->dec) {   // a BC format the device lacks (gfx_vk_tex.c bc_decode): the blocks decoded on the CPU from the buffer's memory as the replay reaches the copy
        if (!to_texture) { VKRED_ONCE(71, "copy of a BC texture decoded on the CPU into a buffer: skipped"); return; }
        uint32_t w = (uint32_t)c->r.width, h = (uint32_t)(c->r.height ? c->r.height : 1), nz = (uint32_t)depth;
        size_t pitch = S.row_stride > 0 ? (size_t)S.row_stride : (size_t)((w + 3) / 4) * f->bpb, slice = S.image_stride > 0 ? (size_t)S.image_stride : pitch * ((h + 3) / 4), out = (size_t)w * h * f->dbpp;
        VkBuffer sb; VkDeviceSize so; uint8_t *p = vk_scratch(out * nz, 16, &sb, &so);
        for (uint32_t z = 0; z < nz; z++) vk_bc_decode(f, (const uint8_t *)(uintptr_t)c->src + z * slice, pitch, w, h, 1, p + z * out);
        b = sb; r.bufferOffset = so; r.bufferRowLength = 0; r.bufferImageHeight = 0;
    }
    if (g_watch_off >= 0) { char w[96]; snprintf(w, sizeof w, "buffer copy %s (buffer %#llx, %dx%d)", to_texture ? "destination" : "source", (unsigned long long)c->src, c->r.width, c->r.height); vk_watch(im, w); }
    if (to_texture) { vkCmdCopyBufferToImage(cb, b, im->img, VK_IMAGE_LAYOUT_GENERAL, 1, &r); im->gpu_w = 1; }
    else { vkCmdCopyImageToBuffer(cb, im->img, VK_IMAGE_LAYOUT_GENERAL, b, 1, &r);
        void vk_gpu_wrote(uint64_t a, uint64_t n); vkpool_t *p = vk_pool_peek(c->src, 1); vk_gpu_wrote(c->src, p ? p->hi - c->src : 1); vk_readback_unknown(); }   // (to the pool's end: the copy's extent in memory is not worked out here)
    vk_img_barrier_all(cb);
}
// Scaled copies of depth and stencil: Adreno blits no depth-stencil format (Pokémon Legends: Arceus copies its 1920x1080 D24S8 depth to a 480x270 one, which stayed unwritten), so each
// aspect goes image → buffer, through dm/android/shaders/depthscale.comp (the nearest texel), buffer → image. Two device buffers kept from copy to copy, grown when a region needs more
#include "depthscale_comp.spv.h"
static struct { VkPipeline p; VkPipelineLayout pl; VkDescriptorSetLayout dl; VkDescriptorPool dp; VkBuffer b[2]; VkDeviceMemory m[2]; VkDeviceSize n[2]; VkDescriptorSet set; int bad; } g_dsc;
static int dsc_init(void) {
    if (g_dsc.p || g_dsc.bad) return !g_dsc.bad;
    VkDescriptorSetLayoutBinding lb[2] = {{0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL}, {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL}};
    VkDescriptorSetLayoutCreateInfo li = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, NULL, 0, 2, lb};
    VkPushConstantRange pr = {VK_SHADER_STAGE_COMPUTE_BIT, 0, 20};
    VkDescriptorPoolSize ps = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 128};
    VkDescriptorPoolCreateInfo pi = {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, NULL, 0, 64, 1, &ps};
    VkShaderModuleCreateInfo mi = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, NULL, 0, sizeof depthscale_comp_spv, depthscale_comp_spv}; VkShaderModule sm = VK_NULL_HANDLE;
    if (vkCreateDescriptorSetLayout(V.dev, &li, NULL, &g_dsc.dl) || vkCreateDescriptorPool(V.dev, &pi, NULL, &g_dsc.dp) || vkCreateShaderModule(V.dev, &mi, NULL, &sm)) { g_dsc.bad = 1; return 0; }
    VkPipelineLayoutCreateInfo pli = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, NULL, 0, 1, &g_dsc.dl, 1, &pr};
    VkComputePipelineCreateInfo ci = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO, NULL, 0, {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, NULL, 0, VK_SHADER_STAGE_COMPUTE_BIT, sm, "main", NULL}};
    if (vkCreatePipelineLayout(V.dev, &pli, NULL, &g_dsc.pl) || (ci.layout = g_dsc.pl, vkCreateComputePipelines(V.dev, VK_NULL_HANDLE, 1, &ci, NULL, &g_dsc.p))) g_dsc.bad = 1;
    vkDestroyShaderModule(V.dev, sm, NULL);
    if (g_dsc.bad) VKLOG("[RED] the scaled depth copy's compute pipeline could not be made");
    return !g_dsc.bad;
}
static int dsc_buf(int i, VkDeviceSize n) {   // buffer i holds at least n bytes
    if (g_dsc.n[i] >= n) return 1;
    if (g_dsc.b[i]) vk_defer_free_buffer(g_dsc.b[i], g_dsc.m[i]);
    g_dsc.b[i] = VK_NULL_HANDLE; g_dsc.n[i] = 0; g_dsc.set = VK_NULL_HANDLE; n = n < (1u << 20) ? 1u << 20 : n + n / 2;
    VkBufferCreateInfo bi = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, NULL, 0, n, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_SHARING_MODE_EXCLUSIVE};
    if (vkCreateBuffer(V.dev, &bi, NULL, &g_dsc.b[i])) return 0;
    VkMemoryRequirements rq; vkGetBufferMemoryRequirements(V.dev, g_dsc.b[i], &rq);
    VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL, rq.size, vk_mem_type(rq.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0)};
    if (vkAllocateMemory(V.dev, &ai, NULL, &g_dsc.m[i])) { vkDestroyBuffer(V.dev, g_dsc.b[i], NULL); g_dsc.b[i] = VK_NULL_HANDLE; return 0; }
    vkBindBufferMemory(V.dev, g_dsc.b[i], g_dsc.m[i], 0); g_dsc.n[i] = n;
    return 1;
}
static int depth_scale(VkCommandBuffer cb, vkimg_t *s, vkimg_t *d, VkImageSubresourceLayers ss, VkImageSubresourceLayers ds, const p_copy_t2t *k) {   // 1: copied
    if (s->fmt != d->fmt || !(s->aspect & (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT)) || s->aspect != d->aspect || !dsc_init()) return 0;
    uint32_t sw = (uint32_t)k->sr.width, sh = (uint32_t)k->sr.height, dw = (uint32_t)k->dr.width, dh = (uint32_t)k->dr.height;
    for (int a = 0; a < 2; a++) {
        VkImageAspectFlags as = a ? VK_IMAGE_ASPECT_STENCIL_BIT : VK_IMAGE_ASPECT_DEPTH_BIT; if (!(s->aspect & as)) continue;
        uint32_t bytes = a ? 1 : s->fmt == VK_FORMAT_D16_UNORM ? 0 : 4; if (!bytes) return 0;   // (16-bit depth: two texels a word, not handled)
        if (!dsc_buf(0, ((VkDeviceSize)sw * sh * bytes + 3) & ~3ull) || !dsc_buf(1, ((VkDeviceSize)dw * dh * bytes + 3) & ~3ull)) return 0;
        if (!g_dsc.set) {
            VkDescriptorSetAllocateInfo sa = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, NULL, g_dsc.dp, 1, &g_dsc.dl};
            if (vkAllocateDescriptorSets(V.dev, &sa, &g_dsc.set)) { g_dsc.set = VK_NULL_HANDLE; VKRED_ONCE(76, "scaled depth copy: no descriptor set left"); return 0; }
            VkDescriptorBufferInfo bi[2] = {{g_dsc.b[0], 0, VK_WHOLE_SIZE}, {g_dsc.b[1], 0, VK_WHOLE_SIZE}};
            VkWriteDescriptorSet w[2] = {{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, NULL, g_dsc.set, 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, NULL, &bi[0], NULL},
                                         {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, NULL, g_dsc.set, 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, NULL, &bi[1], NULL}};
            vkUpdateDescriptorSets(V.dev, 2, w, 0, NULL);
        }
        VkImageSubresourceLayers sl = ss, dl = ds; sl.aspectMask = as; dl.aspectMask = as;
        VkBufferImageCopy r = {0, 0, 0, sl, {k->sr.xoffset, k->sr.yoffset, 0}, {sw, sh, 1}};
        vkCmdCopyImageToBuffer(cb, s->img, VK_IMAGE_LAYOUT_GENERAL, g_dsc.b[0], 1, &r); vk_img_barrier_all(cb);
        uint32_t pc[5] = {sw, sh, dw, dh, bytes}, n = bytes == 4 ? dw * dh : (dw * dh + 3) / 4, groups = (n + 63) / 64, gx = groups < 65535 ? groups : 65535, gy = (groups + gx - 1) / gx;
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, g_dsc.p);
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, g_dsc.pl, 0, 1, &g_dsc.set, 0, NULL);
        vkCmdPushConstants(cb, g_dsc.pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof pc, pc);
        vkCmdDispatch(cb, gx, gy, 1); vk_img_barrier_all(cb);
        VkBufferImageCopy w = {0, 0, 0, dl, {k->dr.xoffset, k->dr.yoffset, 0}, {dw, dh, 1}};
        vkCmdCopyBufferToImage(cb, g_dsc.b[1], d->img, VK_IMAGE_LAYOUT_GENERAL, 1, &w); vk_img_barrier_all(cb);
    }
    S.bound_pipe = VK_NULL_HANDLE;   // the compute pipeline took the bind point the replay's dispatches use
    return 1;
}
static void copy_t2t(const p_copy_t2t *c) {
    vk_end_pass();
    vkimg_t *s = vk_tex(c->src), *d = vk_tex(c->dst);
    if (!s || !d) return;
    vk_watch(s, "copy source"); vk_watch(d, "copy destination");
    VkCommandBuffer cb = vk_cmd(); vk_tex_prepare(cb, s); vk_tex_prepare(cb, d); vk_img_barrier_all(cb);
    VkImageSubresourceLayers ss = sub_of(s, &c->sv, c->has_sv, c->sr.zoffset), ds = sub_of(d, &c->dv, c->has_dv, c->dr.zoffset);
    // Vulkan refuses (vkEndCommandBuffer fails for the whole submission) a region outside either image or a blit the format cannot do: regions are clipped
    // to both images' levels, a copy with nothing left is skipped
    int32_t sw = (int32_t)(s->w >> ss.mipLevel ? s->w >> ss.mipLevel : 1), sh = (int32_t)(s->h >> ss.mipLevel ? s->h >> ss.mipLevel : 1);
    int32_t dw = (int32_t)(d->w >> ds.mipLevel ? d->w >> ds.mipLevel : 1), dh = (int32_t)(d->h >> ds.mipLevel ? d->h >> ds.mipLevel : 1);
    p_copy_t2t k = *c; c = &k; d->gpu_w = 1;
    if (k.sr.height <= 0) k.sr.height = 1; if (k.dr.height <= 0) k.dr.height = 1;
    if (ss.mipLevel >= s->levels || ds.mipLevel >= d->levels || k.sr.xoffset < 0 || k.sr.yoffset < 0 || k.dr.xoffset < 0 || k.dr.yoffset < 0 || k.sr.xoffset >= sw || k.sr.yoffset >= sh || k.dr.xoffset >= dw || k.dr.yoffset >= dh) {
        VKRED_ONCE(72, "texture copy outside its images (src %dx%d @%d,%d of %dx%d level %u, dst %dx%d @%d,%d of %dx%d level %u): skipped", c->sr.width, c->sr.height, c->sr.xoffset, c->sr.yoffset, sw, sh, ss.mipLevel,
                   c->dr.width, c->dr.height, c->dr.xoffset, c->dr.yoffset, dw, dh, ds.mipLevel); vk_img_barrier_all(cb); return; }
    int same = k.sr.width == k.dr.width && k.sr.height == k.dr.height;
    if (k.sr.xoffset + k.sr.width > sw) { VKRED_ONCE(73, "texture copy source region clipped to its image"); k.sr.width = sw - k.sr.xoffset; if (same) k.dr.width = k.sr.width; }
    if (k.sr.yoffset + k.sr.height > sh) { VKRED_ONCE(73, "texture copy source region clipped to its image"); k.sr.height = sh - k.sr.yoffset; if (same) k.dr.height = k.sr.height; }
    if (k.dr.xoffset + k.dr.width > dw) { VKRED_ONCE(74, "texture copy destination region clipped to its image"); k.dr.width = dw - k.dr.xoffset; if (same) k.sr.width = k.dr.width; }
    if (k.dr.yoffset + k.dr.height > dh) { VKRED_ONCE(74, "texture copy destination region clipped to its image"); k.dr.height = dh - k.dr.yoffset; if (same) k.sr.height = k.dr.height; }
    if (k.sr.width <= 0 || k.sr.height <= 0 || k.dr.width <= 0 || k.dr.height <= 0) { vk_img_barrier_all(cb); return; }
    if (!(c->flags & DM_COPY_RESOLVE) && !(k.sr.width == k.dr.width && k.sr.height == k.dr.height && s->fmt == d->fmt)) {   // a blit: both formats must allow it (and linear filtering)
        VkFormatProperties sp, dp; vkGetPhysicalDeviceFormatProperties(V.pd, s->fmt, &sp); vkGetPhysicalDeviceFormatProperties(V.pd, d->fmt, &dp);
        if (!(sp.optimalTilingFeatures & VK_FORMAT_FEATURE_BLIT_SRC_BIT) || !(dp.optimalTilingFeatures & VK_FORMAT_FEATURE_BLIT_DST_BIT) || (s->aspect != d->aspect)) {
            if (depth_scale(cb, s, d, ss, ds, &k)) return;
            VKRED_ONCE(75, "texture blit between formats %d → %d that cannot be blitted (%dx%d → %dx%d, present %llu): skipped", s->fmt, d->fmt, k.sr.width, k.sr.height, k.dr.width, k.dr.height, (unsigned long long)g_tex_frame); vk_img_barrier_all(cb); return; }
        if (!(sp.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT) || (s->aspect & ~VK_IMAGE_ASPECT_COLOR_BIT)) k.flags &= ~1u;   // nearest where linear is not allowed
    }
    if (s == d && !memcmp(&ss, &ds, sizeof ss) && k.sr.xoffset == k.dr.xoffset && k.sr.yoffset == k.dr.yoffset && k.sr.width == k.dr.width && k.sr.height == k.dr.height && k.sr.zoffset == k.dr.zoffset) {
        vk_img_barrier_all(cb); return; }   // a copy of a region onto itself (two textures over the same storage): nothing to do, and Vulkan forbids overlapping copies
    if (c->flags & DM_COPY_RESOLVE) {
        VkImageResolve r = {ss, {c->sr.xoffset, c->sr.yoffset, 0}, ds, {c->dr.xoffset, c->dr.yoffset, 0}, {(uint32_t)c->sr.width, (uint32_t)c->sr.height, 1}};
        vkCmdResolveImage(cb, s->img, VK_IMAGE_LAYOUT_GENERAL, d->img, VK_IMAGE_LAYOUT_GENERAL, 1, &r);
    } else if (c->sr.width == c->dr.width && c->sr.height == c->dr.height && s->fmt == d->fmt) {
        VkImageCopy r = {ss, {c->sr.xoffset, c->sr.yoffset, s->layers > 1 ? 0 : c->sr.zoffset}, ds, {c->dr.xoffset, c->dr.yoffset, d->layers > 1 ? 0 : c->dr.zoffset},
                         {(uint32_t)c->sr.width, (uint32_t)(c->sr.height ? c->sr.height : 1), s->layers > 1 ? 1 : (uint32_t)(c->sr.depth ? c->sr.depth : 1)}};
        vkCmdCopyImage(cb, s->img, VK_IMAGE_LAYOUT_GENERAL, d->img, VK_IMAGE_LAYOUT_GENERAL, 1, &r);
    } else {
        VkImageBlit r = {ss, {{c->sr.xoffset, c->sr.yoffset, 0}, {c->sr.xoffset + c->sr.width, c->sr.yoffset + (c->sr.height ? c->sr.height : 1), 1}},
                         ds, {{c->dr.xoffset, c->dr.yoffset, 0}, {c->dr.xoffset + c->dr.width, c->dr.yoffset + (c->dr.height ? c->dr.height : 1), 1}}};
        vkCmdBlitImage(cb, s->img, VK_IMAGE_LAYOUT_GENERAL, d->img, VK_IMAGE_LAYOUT_GENERAL, 1, &r, (c->flags & 1) ? VK_FILTER_LINEAR : VK_FILTER_NEAREST);
    }
    vk_img_barrier_all(cb);
}
static void clear_tex(const p_clear_tex *c) {
    vk_end_pass();
    vkimg_t *im = vk_tex(c->tex); if (!im) return;
    VkCommandBuffer cb = vk_cmd(); vk_tex_prepare(cb, im); vk_img_barrier_all(cb); im->gpu_w = 1;
    VkImageSubresourceLayers l = sub_of(im, &c->view, c->has_view, c->r.zoffset);
    VkImageSubresourceRange rg = {l.aspectMask, l.mipLevel, 1, l.baseArrayLayer, im->layers > 1 ? (uint32_t)(c->r.depth ? c->r.depth : 1) : 1};
    if (c->r.xoffset || c->r.yoffset || (uint32_t)c->r.width < (im->w >> l.mipLevel) || (uint32_t)c->r.height < (im->h >> l.mipLevel)) VKRED_ONCE(71, "partial texture clear: the whole level is cleared");
    if (im->aspect & VK_IMAGE_ASPECT_COLOR_BIT) { VkClearColorValue v; memcpy(&v, c->c.u, 16); vkCmdClearColorImage(cb, im->img, VK_IMAGE_LAYOUT_GENERAL, &v, 1, &rg); }
    else { VkClearDepthStencilValue v = {c->c.f[0], c->c.u[1]}; rg.aspectMask = im->aspect; vkCmdClearDepthStencilImage(cb, im->img, VK_IMAGE_LAYOUT_GENERAL, &v, 1, &rg); }
    vk_img_barrier_all(cb);
}

// ---------------- the record walker ----------------
static int rt_same(const p_set_rt *r) {   // the targets the open pass draws into already
    int n = r->n > 8 ? 8 : r->n; if (n != S.rt_n) return 0;
    for (int i = 0; i < n; i++) {
        int has = r->color[i] != NULL && r->color[i]->magic == MAGIC('T', 'E', 'X', 0);
        if (has != S.rt_has_col[i] || (has && memcmp(r->color[i], &S.rt_col[i], sizeof S.rt_col[i]))) return 0;
        if (r->has_cview[i] != S.rt_has_cview[i] || (r->has_cview[i] && memcmp(&r->cview[i], &S.rt_cview[i], sizeof S.rt_cview[i]))) return 0;
    }
    int hd = r->depth != NULL && r->depth->magic == MAGIC('T', 'E', 'X', 0);
    if (hd != S.rt_has_dep || (hd && memcmp(r->depth, &S.rt_dep, sizeof S.rt_dep))) return 0;
    if (r->has_dview != S.rt_has_dview || (r->has_dview && memcmp(&r->dview, &S.rt_dview, sizeof S.rt_dview))) return 0;
    return 1;
}
static void set_rt(const p_set_rt *r) {
    if (g_x_tracepass == -2) {   // (DM_VK_TRACE_PASS=-2 lists the render target changes too: each target's serial and memory)
        char t[512]; int o = 0; for (int i = 0; i < r->n && i < 8 && o < 400; i++) { const tex_t *c = r->color[i]; o += snprintf(t + o, sizeof t - (size_t)o, " c%d %s", i, c && c->magic == MAGIC('T', 'E', 'X', 0) ? "" : "-");
            if (c && c->magic == MAGIC('T', 'E', 'X', 0)) o += snprintf(t + o, sizeof t - (size_t)o, "#%llu@%p", (unsigned long long)c->serial, (void *)c->mem); }
        const tex_t *d = r->depth; VKLOG("--- set targets:%s depth %s#%llu@%p%s", t, d && d->magic == MAGIC('T', 'E', 'X', 0) ? "" : "(none) ", d ? (unsigned long long)d->serial : 0ull, d ? (void *)d->mem : NULL,
                                         S.pass_open && !g_x_nokeep && rt_same(r) ? " (the same: the pass goes on)" : ""); }
    if (S.pass_open && !g_x_nokeep && rt_same(r)) { g_vkst.rt_kept++; return; }   // the same targets again (Unity sets them before many draws): the pass goes on
    vk_end_pass();
    S.rt_n = r->n > 8 ? 8 : r->n;
    for (int i = 0; i < S.rt_n; i++) {
        S.rt_has_col[i] = r->color[i] != NULL && r->color[i]->magic == MAGIC('T', 'E', 'X', 0); if (S.rt_has_col[i]) S.rt_col[i] = *r->color[i];
        S.rt_bad[i] = r->color[i] && !S.rt_has_col[i] ? r->color[i]->magic | 1u : 0;
        S.rt_has_cview[i] = r->has_cview[i]; if (r->has_cview[i]) S.rt_cview[i] = r->cview[i];
    }
    S.rt_has_dep = r->depth != NULL && r->depth->magic == MAGIC('T', 'E', 'X', 0); if (S.rt_has_dep) S.rt_dep = *r->depth;
    S.rt_has_dview = r->has_dview; if (r->has_dview) S.rt_dview = r->dview;
}
static void bind_handles_rec(uint32_t op, const void *a) {
    p_bind_handles tmp; const p_bind_handles *b = bind_rec(op, a, &tmp);
    if (b->defer && b->src && b->n > 0 && b->n <= 32) memcpy((void *)b->h, (const void *)(uintptr_t)b->src, 8 * (size_t)b->n);
    int st = b->stage; if (st < 0 || st > 5) return;
    for (int i = 0; i < b->n; i++) {
        int slot = b->first + i;
        uint64_t *d = NULL;
        uint64_t *q = NULL;
        if (op == OP_BIND_TEX || op == OP_BIND_TEX1) { if (slot >= 0 && slot < 256) { d = &S.tex[st][slot]; q = &S.tex_sseq[st][slot]; } }
        else if (op == OP_BIND_SAMP) { if (slot >= 0 && slot < 128) { d = &S.samp[st][slot]; q = &S.samp_sseq[st][slot]; } }
        else if (op == OP_BIND_IMG) { if (slot >= 0 && slot < 32) d = &S.img[st][slot]; }
        if (d && (*d != b->h[i] || (q && *q != b->sseq))) { *d = b->h[i]; if (q) *q = b->sseq; g_bind_gen[st]++; }   // only a change counts: games bind the same handles again before many draws (stage_set's memo)
    }
}
uint32_t g_sub_ops[64];   // record ops replayed in the open submission, by op (logged when the driver refuses it)
static void exec_ops(const uint8_t *p);
// DM_VK_PROF=1 (start-up): a 1 ms CPU-time timer on the replay thread samples where it is; every 20 s the most frequent places are logged: offsets in this library,
// time in other libraries charged to the first caller of ours ("via:"), or with =2 the place itself (libc, the driver). Resolve offsets with llvm-symbolizer
// on the debug build's libdmhost.so. Simpleperf is refused on retail devices; this found the command pool resets and render pass begins that took half the replay
#include <signal.h>
#include <sys/syscall.h>
static _Atomic uint64_t g_prof_pc[16384], g_prof_leaf[16384], g_prof_lr[16384], g_prof_st[16384][6]; static _Atomic uint32_t g_prof_n; static uintptr_t g_prof_lo, g_prof_hi;   // this library's text, for the first frame of ours
static void prof_sig(int sig, siginfo_t *si, void *ucv) { (void)sig; (void)si; ucontext_t *u = ucv; uint64_t pc = u->uc_mcontext.pc;
    if (pc < g_prof_lo || pc >= g_prof_hi) {   // outside this library: charge it to our first caller on the frame chain (marked with bit 63)
        uint64_t lr = u->uc_mcontext.regs[30] & 0x7fffffffffull, fp = u->uc_mcontext.regs[29], sp = u->uc_mcontext.sp, c = 0;
        if (lr >= g_prof_lo && lr < g_prof_hi) c = lr;
        for (int k = 0; k < 12 && !c && fp && !(fp & 7) && fp >= sp && fp - sp < 0x400000; k++) { uint64_t r = ((const uint64_t *)(uintptr_t)fp)[1] & 0x7fffffffffull; if (r >= g_prof_lo && r < g_prof_hi) c = r; fp = ((const uint64_t *)(uintptr_t)fp)[0]; }
        if (c) pc = c | (1ull << 63); }
    uint32_t i = atomic_fetch_add(&g_prof_n, 1); if (i < 16384) { atomic_store(&g_prof_pc[i], pc); atomic_store(&g_prof_leaf[i], u->uc_mcontext.pc); atomic_store(&g_prof_lr[i], u->uc_mcontext.regs[30]);
        uint64_t fp = u->uc_mcontext.regs[29], sp = u->uc_mcontext.sp;   // the frame chain's return addresses (DM_VK_PROF_OUT)
        for (int k = 0; k < 6; k++) { uint64_t r = 0; if (fp && !(fp & 7) && fp >= sp && fp - sp < 0x400000) { r = ((const uint64_t *)(uintptr_t)fp)[1] & 0x7fffffffffull; fp = ((const uint64_t *)(uintptr_t)fp)[0]; } else fp = 0; atomic_store(&g_prof_st[i][k], r); } } }
static void prof_tick(void) {
    // armed by the first DM_VK_PROF=1 it sees, read every 64 frames: a ctl.txt env line can start it after loading (Xenoblade Chronicles 3 crashed loading a save with it on from start-up)
    static int on; static unsigned chk; static uint64_t last; if (!on && !(chk++ & 63)) { const char *e = getenv("DM_VK_PROF"); on = e && atoi(e);
        if (on) { Dl_info d0; if (dladdr((void *)prof_tick, &d0)) { g_prof_lo = (uintptr_t)d0.dli_fbase; g_prof_hi = g_prof_lo + 0x4000000; }
            struct sigaction sa; memset(&sa, 0, sizeof sa); sa.sa_sigaction = prof_sig; sa.sa_flags = SA_SIGINFO | SA_RESTART; sigaction(SIGPROF, &sa, NULL);
            // DM_VK_PROF_TID=<tid>: sample that thread of the process instead (its CPU-time clock; e.g. the game's main thread), and DM_VK_PROF_OUT=<file>: append the raw samples (leaf pc, lr) there
            const char *tt = getenv("DM_VK_PROF_TID"); pid_t tid = tt && atoi(tt) > 0 ? (pid_t)atoi(tt) : (pid_t)syscall(SYS_gettid);
            clockid_t clk = tt && atoi(tt) > 0 ? (clockid_t)((~(unsigned)tid << 3) | 6) : CLOCK_THREAD_CPUTIME_ID;   // Linux MAKE_THREAD_CPUCLOCK(tid, CPUCLOCK_SCHED)
            struct sigevent ev; memset(&ev, 0, sizeof ev); ev.sigev_notify = SIGEV_THREAD_ID; ev.sigev_signo = SIGPROF; ev._sigev_un._tid = tid;
            timer_t tm; if (!timer_create(clk, &ev, &tm)) { struct itimerspec it = {{0, 1000000}, {0, 1000000}}; timer_settime(tm, 0, &it, NULL); VKLOG("[prof] sampling thread %d every 1 ms of its CPU time", (int)tid); } else VKLOG("[RED] [prof] timer_create for thread %d failed", (int)tid); } }
    if (!on) return; uint64_t now = tnow(); if (!last) last = now; if (now - last < 20000000000ull) return; last = now;
    uint32_t n = atomic_exchange(&g_prof_n, 0); if (n > 16384) n = 16384;
    { const char *po = getenv("DM_VK_PROF_OUT"); FILE *pf = po ? fopen(po, "ab") : NULL;
      if (pf) { for (uint32_t i = 0; i < n; i++) { uint64_t v[8] = {atomic_load(&g_prof_leaf[i]), atomic_load(&g_prof_lr[i])}; for (int k = 0; k < 6; k++) v[2 + k] = atomic_load(&g_prof_st[i][k]); fwrite(v, sizeof v, 1, pf); } fclose(pf); } }
    Dl_info di; uintptr_t base = dladdr((void *)prof_tick, &di) ? (uintptr_t)di.dli_fbase : 0;
    static uint64_t pcs[16384]; static uint32_t cnt[16384]; int np = 0;
    static int leaves = -1; if (leaves < 0) { const char *e = getenv("DM_VK_PROF"); leaves = e && atoi(e) == 2; }   // DM_VK_PROF=2: where the time is, not who called
    for (uint32_t i = 0; i < n; i++) { uint64_t pc = atomic_load(leaves ? &g_prof_leaf[i] : &g_prof_pc[i]); int k = 0; for (; k < np; k++) if (pcs[k] == pc) break; if (k == np) { pcs[np] = pc; cnt[np++] = 0; } cnt[k]++; }
    char line[4000]; int o = snprintf(line, sizeof line, "[prof] %u samples, top pcs (library offset:count):", n);
    for (int t = 0; t < 40 && o < (int)sizeof line - 40; t++) { int b = -1; for (int k = 0; k < np; k++) if (cnt[k] && (b < 0 || cnt[k] > cnt[b])) b = k; if (b < 0) break;
        uint64_t pc = pcs[b]; Dl_info d2;
        if (pc >> 63) o += snprintf(line + o, sizeof line - o, " via:%llx:%u", (unsigned long long)((pc & ~(1ull << 63)) - base), cnt[b]);
        else if (pc - base < 0x10000000) o += snprintf(line + o, sizeof line - o, " %llx:%u", (unsigned long long)(pc - base), cnt[b]);
        else if (dladdr((void *)(uintptr_t)pc, &d2) && d2.dli_fname) o += snprintf(line + o, sizeof line - o, " %s(%s)+%llx:%u", strrchr(d2.dli_fname, '/') ? strrchr(d2.dli_fname, '/') + 1 : d2.dli_fname, d2.dli_sname ? d2.dli_sname : "?", (unsigned long long)(pc - (uintptr_t)(d2.dli_saddr ? d2.dli_saddr : d2.dli_fbase)), cnt[b]);
        else o += snprintf(line + o, sizeof line - o, " ?%llx:%u", (unsigned long long)pc, cnt[b]);
        cnt[b] = 0; }
    VKLOG("%s", line);
}
void mtl_exec(const uint8_t *p) {   // timed at the top level only (OP_CALL nests)
    static int depth; uint64_t t0 = depth ? 0 : tnow(); depth++; if (depth == 1) { prof_tick(); g_la_depth = 0; g_la_epoch++; }
    exec_ops(p);
    if (!--depth) g_vkst.ns_exec += tnow() - t0;
}
static void exec_ops(const uint8_t *p) {
    for (;;) {
        op_hdr h; memcpy(&h, p, sizeof h);
        if (h.op < 64) g_sub_ops[h.op]++;
        static uint64_t skip = ~0ull; if (skip == ~0ull) { skip = 0; const char *e = getenv("DM_VK_SKIP_OPS"); for (const char *q = e; q && *q;) { char *n; long v = strtol(q, &n, 10); if (n == q) break; if (v > 0 && v < 64) skip |= 1ull << v; q = *n ? n + 1 : n; } }   // debugging: record ops not replayed
        if (h.op < 64 && h.op > 3 && (skip >> h.op & 1)) { p += h.size; continue; }
        const uint8_t *a = p + 8; g_la_at = p + h.size;
        switch (h.op) {
        case OP_END: return;
        case OP_BEGIN: break;
        case OP_JUMP: { const uint8_t *t; memcpy(&t, a, 8); p = t; continue; }
        case OP_CALL: { uint64_t t; memcpy(&t, a, 8); if (g_la_depth < 16) g_la_ret[g_la_depth] = p + h.size; g_la_depth++; exec_ops((const uint8_t *)(uintptr_t)t); g_la_depth--; break; }
        case OP_SIGNAL_EVENT: { volatile uint32_t *val; uint32_t m, v; memcpy(&val, a, 8); memcpy(&m, a + 8, 4); memcpy(&v, a + 12, 4); mtl_sub_event(val, m, v); break; }
        case OP_REPORT_COUNTER: { uint64_t t, ad; memcpy(&t, a, 8); memcpy(&ad, a + 8, 8); if (DM_GAME_RENDER_COND) rc_note(ad, t == 0 || t == 0x10 ? 1 : 2); g_la_epoch++; mtl_sub_counter(ad, (uint32_t)t); break; }
        case OP_RESET_COUNTER: { uint64_t t; memcpy(&t, a, 8); g_la_epoch++; if (t == NVN_COUNTER_TYPE_SAMPLES_PASSED) vk_occ_reset(); break; }
        case OP_FENCE_SYNC: { sync_t *s; memcpy(&s, a, 8); s->fence = mtl_last_fence() + 1; s->fenced = 1; break; }
        // Upstream's queue waits (hle_nvn.c nvnCommandBufferWaitEvent / WaitSync, ea0c98df3): the Apple scheduler parks the queue on them; this replay runs
        // submissions in their order and does not wait, as when the calls were no-ops. An event wait whose predicate does not hold yet is named once (an
        // ordering the logical queues would change: Ocarina of Time's lighting producer and consumer)
        case OP_WAIT_EVENT: { volatile uint32_t *val; uint32_t m, v; memcpy(&val, a, 8); memcpy(&m, a + 8, 4); memcpy(&v, a + 12, 4);
            uint32_t cur = __atomic_load_n(val, __ATOMIC_ACQUIRE); if (!(m == 0 ? cur == v : (uint32_t)(cur - v) < 0x80000000u)) VKRED_ONCE(130, "event wait not satisfied at replay (word %#x, mode %u, target %#x): replayed in submission order", cur, m, v); break; }
        case OP_WAIT_SYNC: break;
        // (DM_VK_TRACE_PASS=-2 lists the buffer writes too: copies, clears and uniform updates, by address)
        case OP_COPY_B2B: { uint64_t s_, d, n; memcpy(&s_, a, 8); memcpy(&d, a + 8, 8); memcpy(&n, a + 16, 8); mtl_sub_copy(s_, d, n); if (g_rc_npend) rc_overwrite(d, n);
            if (g_x_tracepass == -2 || mdi_log_now()) VKLOG("--- copy %#llx → %#llx, %llu bytes, after pass %d", (unsigned long long)s_, (unsigned long long)d, (unsigned long long)n, g_sub_pass - 1); break; }
        case OP_CLEAR_BUF: { const p_clear_buf *c = (const p_clear_buf *)a; mtl_sub_fill(c->addr, c->size, c->value); if (g_rc_npend) rc_overwrite(c->addr, c->size);
            if (g_x_tracepass == -2 || mdi_log_now()) VKLOG("--- buffer clear %#llx, %llu bytes to %#x, after pass %d", (unsigned long long)c->addr, (unsigned long long)c->size, c->value, g_sub_pass - 1); break; }
        case OP_UPDATE_UB: { uint64_t d, n; memcpy(&d, a, 8); memcpy(&n, a + 8, 8); vk_sub_update(d, a + 16, n); if (g_rc_npend) rc_overwrite(d, n);
            if (g_x_tracepass == -2) VKLOG("--- uniform update %#llx, %llu bytes, after pass %d", (unsigned long long)d, (unsigned long long)n, g_sub_pass - 1); break; }
        case OP_SET_RT: set_rt((const p_set_rt *)a); break;
        case OP_CLEAR_COLOR: if (!g_rc_skip) clear_color((const p_clear_color *)a); break;
        case OP_CLEAR_DS: if (!g_rc_skip) clear_ds((const p_clear_ds *)a); break;
        case OP_VIEWPORTS: { const p_viewports *v = (const p_viewports *)a; for (int i = 0; i < v->n && v->first + i < 16; i++) memcpy(S.vp[v->first + i], v->v[i], 16); break; }
        case OP_SCISSOR: S.sc = *(const p_scissor *)a; S.sc_set = 1; break;
        case OP_DEPTH_RANGE: { const p_depth_range *d = (const p_depth_range *)a; S.dnear = d->n; S.dfar = d->f; break; }
        case OP_BIND_PROGRAM: bind_program((const p_bind_program *)a); break;
        case OP_BIND_STATE: { const p_bind_state *s = (const p_bind_state *)a;
            switch (s->kind) {
            case 0: { blend_st b; memcpy(&b, s->bytes, sizeof b); if (b.target < 8) S.blend[b.target] = b; break; }
            case 1: memcpy(&S.color, s->bytes, sizeof S.color); break;
            case 2: { chmask_st mask;memcpy(&mask,s->bytes,sizeof mask);bind_chmask(mask.mask);break; }
            case 3: memcpy(&S.ms, s->bytes, sizeof S.ms); break;
            case 4: memcpy(&S.poly, s->bytes, sizeof S.poly); break;
            case 5: memcpy(&S.ds, s->bytes, sizeof S.ds); break;
            }
            break; }
        case OP_BIND_VATTR: { const p_bind_vattr *v = (const p_bind_vattr *)a; memset(S.vattr, 0, sizeof S.vattr); for (int i = 0; i < v->n && i < 16; i++) S.vattr[v->first + i] = v->st[i]; break; }
        case OP_BIND_VSTREAM: { const p_bind_vstream *v = (const p_bind_vstream *)a; for (int i = 0; i < v->n && v->first + i < 16; i++) S.vstream[v->first + i] = v->st[i]; break; }
        case OP_BIND_VBUF: case OP_BIND_VBUF1: { p_bind_vbuf t1; const p_bind_vbuf *v = vbuf_rec(h.op, a, &t1); for (int i = 0; i < v->n && v->first + i < 16; i++) S.vbuf[v->first + i] = v->r[i]; break; }   // VBUF1: upstream's 24-byte one-buffer record
        case OP_BIND_UBO: case OP_BIND_UBO_SMALL: { p_bind_buf t1; const p_bind_buf *b = bind_buf_rec(h.op, a, &t1); if (b->stage >= 0 && b->stage < 6 && b->index >= 0 && b->index < 16 && memcmp(&S.ubo[b->stage][b->index], b, sizeof *b)) { S.ubo[b->stage][b->index] = *b; g_ubo_gen[b->stage]++; } break; }
        case OP_BIND_SSBO: { const p_bind_buf *b = (const p_bind_buf *)a; if (b->stage >= 0 && b->stage < 6 && b->index >= 0 && b->index < 16 && memcmp(&S.ssbo[b->stage][b->index], b, sizeof *b)) { S.ssbo[b->stage][b->index] = *b; g_ssbo_gen[b->stage]++; } break; }
        case OP_BIND_TEX: case OP_BIND_SAMP: case OP_BIND_IMG: case OP_BIND_TEX1: bind_handles_rec(h.op, a); break;
        case OP_STENCIL: { const p_stencil *s = (const p_stencil *)a;
            for (int f = 0; f < 2; f++) if (s->face & (1 << f)) { if (s->which == 0) S.smask[f] = s->value_mask; else if (s->which == 1) S.swmask[f] = s->mask; else S.sref[f] = s->ref; }
            break; }
        case OP_POLY_OFFSET: S.pofs = *(const p_poly_offset *)a; break;
        // ---- records upstream added (upsync 2026-11-07) ----
        case OP_ALPHA_REF: S.alpha_ref = ((const p_alpha_ref *)a)->ref; break;   // SetAlphaRef: the alpha test's reference (pkey atest / aref)
        case OP_PUSH_CONST: push_constants((const p_push_const *)a); break;
        case OP_MULTI_DRAW: {   // MultiDraw*IndirectCount with more than MDI_INLINE draws: one record (hle_nvn.c); one draw by the GPU's count at the range's first record (mdi_group), as the inline form's group
            const p_multi_draw *m = (const p_multi_draw *)a; if (g_rc_skip) break;
            if (m->itype < 0) { p_draw d = {m->prim, 0, 0, 1, 0, m->ind, m->cnt, 0, m->max, (int)m->stride}; draw(&d); }
            else { p_draw_idx d = {m->prim, m->itype, 0, 0, 0, 1, m->ibuf, m->ind, m->cnt, 0, m->max, (int)m->stride}; draw_idx(&d); }
            break; }
        case OP_DRAW_TEX: VKRED_ONCE(113, "DrawTexture: not replayed on Android yet (upstream records it since the upsync; before it the call was not taken over)"); break;
        case OP_DLSS: {   // an NVN2 DLSS evaluation (hle_nvn.c "NVN2's DLSS"): no neural network here, and none wanted: the colour input is scaled bilinearly into the output
            // texture, as the Metal replay does where MetalFX has no scaler. The game then composes from the output as it would; the picture is the render resolution's, upscaled
            const p_dlss *e = (const p_dlss *)a; if (g_rc_skip) break;
            if (!e->tex[DLSS_COLOR] || !e->tex[DLSS_OUTPUT]) { VKRED_ONCE(114, "DLSS evaluation without a colour input or an output texture: skipped"); break; }
            static int said; if (!said++) VKLOG("DLSS evaluation: %dx%d colour scaled into the %dx%d output (bilinear; no DLSS on Android)", e->iw, e->ih, e->ow, e->oh);
            p_copy_t2t k = {.src = e->tex[DLSS_COLOR], .dst = e->tex[DLSS_OUTPUT], .flags = NVN_COPY_FLAGS_LINEAR_FILTER_BIT};
            k.sr = (NVNcopyRegion){e->base[DLSS_COLOR][0], e->base[DLSS_COLOR][1], 0, e->iw, e->ih, 1};
            k.dr = (NVNcopyRegion){e->base[DLSS_OUTPUT][0], e->base[DLSS_OUTPUT][1], 0, e->ow, e->oh, 1};
            copy_t2t(&k); break; }
        case OP_BLEND_COLOR: memcpy(S.blendc, ((const p_blend_color *)a)->rgba, 16); break;
        case OP_DEPTH_CLAMP: S.depth_clamp = ((const p_depth_clamp *)a)->on; break;
        case OP_RAST_DISCARD: S.rast_discard = ((const p_rast_discard *)a)->on; break;
        case OP_PRIM_RESTART: S.prim_restart = ((const p_prim_restart *)a)->on; break;
        case OP_DRAW: { if (g_rc_skip) break; p_draw t; draw(draw_rec(a, h.size, &t)); break; }
        case OP_DRAW_IDX: { if (g_rc_skip) break; p_draw_idx t; draw_idx(draw_idx_rec(a, h.size, &t)); break; }
        case OP_DISPATCH: dispatch((const p_dispatch *)a); break;
        case OP_BARRIER: {   // the pass ends at every barrier: kept across XC3's fragment- and primitive-order barriers (0x1 0x2 0x4) its characters drew torn in some frames
            int32_t bits = *(const int32_t *)a;
            if (g_x_listpasses) VKLOG("    barrier %#x%s", bits, g_order_keep && S.pass_open == 1 && !(bits & ~0x7) ? " (order only: decided at the next draw)" : "");
            if (g_order_keep && S.pass_open == 1 && !(bits & ~0x7)) { g_order_pending = 1; break; }   // (g_order_keep)
            vk_end_pass(); vk_img_barrier_all(vk_cmd()); g_order_pending = 0; break; }
        case OP_SET_TEX_POOL: case OP_SET_SAMP_POOL: {   // bound handles were rebased when recorded (hle_nvn.c bind_handles); handles read from memory are rebased by these (mem_handle)
            const p_set_pool *sp = (const p_set_pool *)a; int base = sp->pool ? nvn_dpool_base(sp->pool, h.op == OP_SET_TEX_POOL) : 0;
            if (h.op == OP_SET_TEX_POOL) g_pool_tex_base = base; else g_pool_samp_base = base;
            break; }
        case OP_COPY_STRIDE: { const p_copy_stride *c = (const p_copy_stride *)a; if (c->image) S.image_stride = c->stride; else S.row_stride = c->stride; break; }
        case OP_COPY_B2T: copy_b2t((const p_copy_b2t *)a, 1); break;
        case OP_COPY_T2B: copy_b2t((const p_copy_b2t *)a, 0); break;
        case OP_COPY_T2T: copy_t2t((const p_copy_t2t *)a); break;
        case OP_CLEAR_TEX: clear_tex((const p_clear_tex *)a); break;
        case OP_RENDER_COND: render_cond((const p_render_cond *)a); break;   // recorded only with the profile key render_cond (hle_nvn.c)
        case OP_DRAW_TF: if (!g_rc_skip) draw_tf((const p_draw_tf *)a); break;
        case OP_TF_BIND: { const p_tf_bind *b = (const p_tf_bind *)a; if (b->index >= 0 && b->index < 4) g_tf[b->index] = *b; break; }
        case OP_TF_STATE: if (!V.has_xfb) VKRED_ONCE(80, "transform feedback: the device has none, its draws write nothing"); tf_state((const p_tf_state *)a); break;
        case OP_VP_SWIZZLE: { const p_vp_swizzle *b = (const p_vp_swizzle *)a; memcpy(S.vp_sw[b->first], b->sw, 4 * (size_t)b->n); break; }
        default: VKRED_ONCE(82, "unknown record op %u", h.op); break;
        }
        p += h.size;
    }
}
