// Android port: what both graphics backends (gfx_null.c, gfx_vk_*.c) report for the host's statistics and diagnostic hooks of the Metal backend
// (nvn_mtl.m) that have no counterpart yet: zeros.
#include "dm.h"
#include "nvn_obj.h"
#include "nvn_mtl.h"
#include <stdlib.h>
#include <string.h>

// ---------------- statistics and diagnostics: nothing to report ----------------
void mtl_vbuf_stats(uint64_t *a, uint64_t *b, uint64_t *c, uint64_t *d, uint64_t *e, uint64_t *f) { *a = *b = *c = *d = *e = *f = 0; }
void mtl_res_counts(uint32_t tick, const char *why) { (void)tick; (void)why; }
void mtl_res_snapshot(mtl_res_count *out) { memset(out, 0, sizeof *out); }
void mtl_alloc_stats(uint64_t *w, uint64_t *g, int *n) { *w = *g = 0; *n = 0; }
void mtl_args_stats(uint64_t *e, uint64_t *r, uint64_t *p) { *e = *r = *p = 0; }   // (Metal argument blocks encoded, reused, patched: upstream nxnative adaff7356 added the third)
void mtl_barrier_stats(uint64_t *s, uint64_t *f) { *s = *f = 0; }
uint64_t mtl_blit_splits(void) { return 0; }
void mtl_bw(uint64_t *l, uint64_t *s, uint64_t *f) { *l = *s = *f = 0; }
uint64_t mtl_copy_cut_count(void) { return 0; }
uint64_t mtl_cut_count(void) { return 0; }
void mtl_create_ns(uint64_t *l, uint64_t *p) { *l = *p = 0; }
void mtl_create_stats(uint64_t *l, uint64_t *p) { *l = *p = 0; }
uint64_t mtl_drawable_ns_take(void) { return 0; }
uint64_t mtl_ds_omit(void) { return 0; }
__attribute__((weak)) double mtl_gpu_ms(uint64_t f) { (void)f; return 0; }   // gfx_vk_core.c has the real one
int mtl_gpu_span(uint64_t f, uint64_t *t0, uint64_t *t1) { (void)f; *t0 = *t1 = 0; return 0; }
void mtl_lc_stats(uint64_t *t, uint64_t *l) { *t = *l = 0; }
void mtl_mem_stat(unsigned long long frame, unsigned long long rss_mb) { (void)frame; (void)rss_mb; }
uint64_t mtl_pkg_miss(void) { return 0; }
void mtl_pkg_stats(uint64_t *u, uint64_t *r) { *u = *r = 0; }
void mtl_res_stats(char *out, size_t n) { if (n) out[0] = 0; }
void mtl_stats(uint64_t *p, uint64_t *d) { *p = *d = 0; }
void mtl_ub_present(double iv) { (void)iv; }
void mtl_ubd_stats(uint64_t *d, uint64_t *c) { *d = *c = 0; }
void mtl_ubp_stats(uint64_t *m, uint64_t *r, uint64_t *f) { *m = *r = *f = 0; }
void mtl_ubp_why(uint64_t w[5]) { memset(w, 0, 5 * sizeof *w); }
uint64_t mtl_udc_saved_bytes(void) { return 0; }
uint64_t mtl_upload_bytes(void) { return 0; }
uint64_t mtl_upload_gpu_ns_take(void) { return 0; }
uint64_t mtl_upload_ns_take(void) { return 0; }
void mtl_use_stats(uint64_t *c, uint64_t *s) { *c = *s = 0; }
void dm_gfx_passdump_next(long n) { (void)n; }
void dm_gfx_shot_at(long k, const char *path) { (void)k; (void)path; }
void dm_gfx_texdump_next(const char *path) { (void)path; }
void vk_registry_texlist(int target) __attribute__((weak));   // Vulkan metadata-only listing; absent for the null backend
void dm_gfx_texlist(int target) { if (vk_registry_texlist) vk_registry_texlist(target); }
void dm_gfx_texraw(uint64_t serial, const char *path) { (void)serial; (void)path; }

// ---------------- upsync 2026-11-07: hooks the host gained from nxnative (Metal backend) that both Android backends answer the same way ----------------
// Statistics of Metal-only mechanisms (shader hoisting, mesh geometry, same-pass feedback, residency tables): nothing to report
void mtl_cshoist_stats(uint64_t *m, uint64_t *d, uint64_t *s) { *m = *d = 0; memset(s, 0, 8 * sizeof *s); }
void mtl_geohoist_stats(uint64_t *h, uint64_t *r, uint64_t *s) { *h = *r = 0; memset(s, 0, 6 * sizeof *s); }
void mtl_geomesh_stats(uint64_t *n, uint64_t *w, uint64_t *d, uint64_t *rn, uint64_t *e) { *n = *w = *d = *rn = 0; memset(e, 0, 4 * sizeof *e); }
void mtl_fb_stats(uint64_t *s, uint64_t *r, uint64_t *t, uint64_t *w) { *s = *r = *t = *w = 0; }
void mtl_hybrid_stats(uint64_t *draws, uint64_t *made, uint64_t *none) { *draws = *made = *none = 0; }
void mtl_sm_stats(uint64_t *p, uint64_t *c, uint64_t *w) { *p = *c = *w = 0; }
void mtl_pool_stats(uint64_t s[6]) { memset(s, 0, 6 * sizeof *s); }
unsigned mtl_cpu_tex_count(void) { return 0; }
uint64_t mtl_pkg_edge_skips(void) { return 0; }
uint64_t mtl_pkg_copy_absorbs(void) { return 0; }   // (the Metal backend's packaged-texture copy count, hle_nvn.c's stats line; upstream nxnative bbea4c90f)
void mtl_gsb_stats(uint64_t *n, uint64_t *b) { *n = *b = 0; }   // GPU textures' writes stored into their storage (Metal's gsb_store, hle_nvn.c's stats line; upstream nxnative 6c3aeb376): none counted here
void mtl_pso_line(char *out, size_t n) { if (n) out[0] = 0; }
int mtl_pso_prebuild_tool(void) { return 0; }   // DM_PSO_PREBUILD=tool is a Metal pipeline-archive step: no such tool here, the game runs
void mtl_ab_gpu(uint64_t f, double gpu_ms, double frame_ms) { (void)f; (void)gpu_ms; (void)frame_ms; }   // DM_AB phases
void dm_gfx_restop(void) {}   // DM_CTL "restop": the Metal residency set
// The display link (CVDisplayLink / CADisplayLink pacing): Android presents through the swapchain (gfx_vk_present.c) and mtl_link_active is 0, so these are never consulted for pacing
int mtl_link_rate(void) { return 60; }
int mtl_link_stalled(void) { return 0; }
int mtl_link_tick(uint64_t n, double *target, double *lead_ms, int *ticks, double *gap_ms) { (void)n; *target = *lead_ms = *gap_ms = 0; *ticks = 0; return 0; }
int mtl_link_wait_ticks(int ticks, uint64_t timeout_ns) { (void)ticks; (void)timeout_ns; return 0; }
int mtl_snap_on(void) { return 0; }   // DM_PRESENT_SNAP (present snapshots into a ring the display link shows) is Metal-only
uint64_t mtl_snap_acquire_wait(void) { return 0; }
int mtl_probe_ramp_on(void) { static int on = -1; if (on < 0) { const char *e = getenv("DM_FSTEP_PROBE_RAMP"); on = e ? atoi(e) != 0 : 1; } return on; }   // as nvn_mtl.m: default on
// Null-backend queue fallbacks; Vulkan supplies state restoration in logical mode.
// Vulkan builds pipelines at first use instead of the Metal prewarm prepare step.
void mtl_prepare(void) {}
uint64_t g_nvn_active_queue;   // the NVN queue whose submission is being replayed (gfx_vk_exec.c: its render-enable state, DM_VK_RC_QUEUE)
__attribute__((weak)) void mtl_queue_activate(uint64_t serial) { g_nvn_active_queue = serial; }
__attribute__((weak)) void mtl_queue_forget(uint64_t serial) { (void)serial; }
void mtl_set_queue_origin_lower_left(int on) { mtl_set_origin_lower_left(on); }   // the raster flip is per queue upstream; here the one flag, set under the submit lock at each submit
// Memory pools: every pool is resident and reachable by address (vk_pool_of), so no residency hints are needed; a pool's physical storage going away is handled by mtl_pool_del
void mtl_sp_phys_gone(const void *mem, uint64_t size) { (void)mem; (void)size; }
int mtl_addr_known(uint64_t addr, uint64_t len) { return mtl_addr(addr, len, NULL, NULL); }
// Programs: the Vulkan backend has no per-queue binding tables to drop and loads shader modules at first use
void mtl_prog_forget(const prog_t *p) { (void)p; }
void mtl_prog_prefetch(const prog_t *p) { (void)p; }
// Textures: image (storage) use comes from the builder's IMAGE flag at creation (gfx_vk_tex.c), and a GPU write is seen through nvn_ft_written before a storage copy.
// TODO(upsync): a RegisterImage of a texture built without IMAGE needs the image re-created with STORAGE usage. GPU-written storage and mtl_tex_cpu_rewritten: gfx_vk_tex.c / gfx_vk_cpu_storage.c
// RegisterImage uses the texture descriptor index space and explicitly declares
// image usage. Preserve that intent in our snapshot; storage preflight adds native
// usage and compatible integer views while retaining any existing GPU contents.
void mtl_reg_image(int id, const tex_t *t, const view_t *v) {
    if (!t) { mtl_reg_tex(id, NULL, v); return; }
    tex_t image = *t; image.d.flags |= NVN_TEXTURE_FLAGS_IMAGE;
    mtl_reg_tex(id, &image, v);
}
// Acquire syncs (WindowAcquireTexture's sync, fenced 2) wait like any fence; the wait ledger that tells them apart is nvn_mtl.m's
int mtl_wait_acq(uint64_t fence, uint64_t timeout_ns) { return mtl_wait(fence, timeout_ns); }
// The picture's shape (a sized vi layer, a taller screen): the swapchain presenter letterboxes to the window texture's own size. TODO(upsync): honour these as present_shape does
void mtl_present_layer(uint32_t w, uint32_t h) { (void)w; (void)h; }
void mtl_present_screen(uint32_t w, uint32_t h) { (void)w; (void)h; }
// DM_NVN_CAPTURE: gfx_vk_capture.inc (included by gfx_vk_tex.c)

// Android shader modules and serial→key records live for the process. A queued
// bind owns descriptor bytes, and materializes keys before the guest can reuse
// its program object/control section. No compiled library reference is borrowed.
void mtl_prog_snapshot(prog_t *copy, const prog_t *source) {
    *copy = *source;
    if (!copy->serial) copy->serial = nvn_new_serial();
    for (unsigned i=0; i<copy->nstage && i<6; i++) { uint8_t key[8]; prog_key(copy,i,key); }
}
void mtl_prog_unpin(prog_t *p) { (void)p; }

// Null backend fallbacks; Vulkan retains registry versions/native images.
__attribute__((weak)) void vk_registry_job_pin(uint64_t id) { (void)id; }
__attribute__((weak)) void vk_registry_job_unpin(uint64_t id) { (void)id; }
__attribute__((weak)) void mtl_tex_pin(uint64_t serial) { (void)serial; }
__attribute__((weak)) void mtl_tex_unpin(uint64_t serial) { (void)serial; }
__attribute__((weak)) void mtl_tex_snapshot(tex_t *copy, const tex_t *source) { *copy=*source; mtl_tex_pin(copy->serial); }

__attribute__((weak)) uint64_t mtl_completed_fence(void) { return mtl_last_fence(); }
