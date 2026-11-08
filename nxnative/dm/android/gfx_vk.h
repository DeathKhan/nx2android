// Android port: the Vulkan backend behind the replay interface of nvn_mtl.h (what nvn_mtl.m implements on Metal).
// Files: gfx_vk_core.c (device, memory pools, submissions, completion), gfx_vk_tex.c (formats, textures, samplers, registration tables),
// gfx_vk_shader.c (shader bundle, modules, layouts), gfx_vk_exec.c (command replay: passes, pipelines, descriptors, draws, copies), gfx_vk_present.c (window, swapchain).
// Version 1 is correct-first: whole-pipeline barriers between passes, full pipelines with a small dynamic state set, descriptor sets per draw.
#pragma once
#define VK_NO_PROTOTYPES
#define VK_USE_PLATFORM_ANDROID_KHR
#include <android/hardware_buffer.h>
#include <android/native_window.h>
#include <vulkan/vulkan.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include "dm.h"
#include "nvn_obj.h"
#include "nvn_mtl.h"

#define VKLOG(...) LOG("[vk] " __VA_ARGS__)
#define VKRED_ONCE(id, ...) do { static int once_##id; if (!once_##id++) LOG("[RED] [vk] " __VA_ARGS__); } while (0)
#define VKCHK(x) do { VkResult r_ = (x); if (r_ < 0) die("[vk] %s failed: %d", #x, r_); } while (0)

// ---- Vulkan entry points (loaded from libvulkan.so) ----
#define VK_FNS(X) \
    X(vkCreateInstance) X(vkEnumeratePhysicalDevices) X(vkGetPhysicalDeviceProperties) X(vkGetPhysicalDeviceProperties2) X(vkGetPhysicalDeviceFeatures2) \
    X(vkGetPhysicalDeviceMemoryProperties) X(vkGetPhysicalDeviceQueueFamilyProperties) X(vkGetPhysicalDeviceFormatProperties) X(vkCreateDevice) X(vkGetDeviceProcAddr) \
    X(vkEnumerateDeviceExtensionProperties) X(vkGetPhysicalDeviceSurfaceCapabilitiesKHR) X(vkGetPhysicalDeviceSurfaceFormatsKHR) X(vkGetPhysicalDeviceSurfacePresentModesKHR) \
    X(vkCreateAndroidSurfaceKHR) X(vkDestroySurfaceKHR) X(vkGetPhysicalDeviceSurfaceSupportKHR)
#define VK_DEV_FNS(X) \
    X(vkGetDeviceQueue) X(vkCreateBuffer) X(vkDestroyBuffer) X(vkAllocateMemory) X(vkFreeMemory) X(vkBindBufferMemory) X(vkBindImageMemory) X(vkGetBufferMemoryRequirements) \
    X(vkGetImageMemoryRequirements) X(vkCreateImage) X(vkDestroyImage) X(vkCreateImageView) X(vkDestroyImageView) X(vkCreateSampler) X(vkDestroySampler) X(vkMapMemory) X(vkUnmapMemory) X(vkFreeCommandBuffers) X(vkFlushMappedMemoryRanges) X(vkInvalidateMappedMemoryRanges) \
    X(vkCreateCommandPool) X(vkResetCommandPool) X(vkResetCommandBuffer) X(vkAllocateCommandBuffers) X(vkBeginCommandBuffer) X(vkEndCommandBuffer) X(vkQueueSubmit) X(vkQueueWaitIdle) X(vkDeviceWaitIdle) \
    X(vkCreateSemaphore) X(vkWaitSemaphores) X(vkGetSemaphoreCounterValue) X(vkCreateFence) X(vkWaitForFences) X(vkResetFences) X(vkDestroyFence) \
    X(vkCmdPipelineBarrier) X(vkCmdCopyBuffer) X(vkCmdCopyBufferToImage) X(vkCmdCopyImageToBuffer) X(vkCmdCopyImage) X(vkCmdBlitImage) X(vkCmdFillBuffer) X(vkCmdUpdateBuffer) \
    X(vkCmdClearColorImage) X(vkCmdClearDepthStencilImage) X(vkCmdClearAttachments) X(vkCmdResolveImage) \
    X(vkCreateRenderPass) X(vkCreateRenderPass2) X(vkDestroyRenderPass) X(vkCreateFramebuffer) X(vkDestroyFramebuffer) X(vkCmdBeginRenderPass) X(vkCmdEndRenderPass) X(vkCmdBeginRendering) X(vkCmdEndRendering) X(vkCmdBindPipeline) X(vkCmdBindDescriptorSets) X(vkCmdBindVertexBuffers) X(vkCmdBindIndexBuffer) \
    X(vkCmdSetViewport) X(vkCmdSetScissor) X(vkCmdSetBlendConstants) X(vkCmdSetStencilReference) X(vkCmdSetStencilCompareMask) X(vkCmdSetStencilWriteMask) \
    X(vkCmdSetDepthBias) X(vkCmdSetLineWidth) X(vkCmdDraw) X(vkCmdPushConstants) X(vkCmdDrawIndexed) X(vkCmdDrawIndirect) X(vkCmdDrawIndexedIndirect) X(vkCmdDispatch) X(vkCmdDispatchIndirect) \
    X(vkCreateShaderModule) X(vkDestroyShaderModule) X(vkCreateDescriptorSetLayout) X(vkCreatePipelineLayout) X(vkCreateGraphicsPipelines) X(vkDestroyPipeline) X(vkCreateComputePipelines) X(vkCreatePipelineCache) \
    X(vkGetPipelineCacheData) X(vkCreateDescriptorPool) X(vkDestroyDescriptorPool) X(vkResetDescriptorPool) X(vkAllocateDescriptorSets) X(vkUpdateDescriptorSets) \
    X(vkCreateBufferView) X(vkDestroyBufferView) X(vkGetAndroidHardwareBufferPropertiesANDROID) X(vkGetBufferDeviceAddress) \
    X(vkCreateQueryPool) X(vkCmdResetQueryPool) X(vkCmdBeginQuery) X(vkCmdEndQuery) X(vkCmdWriteTimestamp) X(vkGetQueryPoolResults) \
    X(vkCreateSwapchainKHR) X(vkDestroySwapchainKHR) X(vkGetSwapchainImagesKHR) X(vkAcquireNextImageKHR) X(vkQueuePresentKHR) \
    X(vkCmdBindTransformFeedbackBuffersEXT) X(vkCmdBeginTransformFeedbackEXT) X(vkCmdEndTransformFeedbackEXT) X(vkCmdDrawIndirectByteCountEXT) X(vkCmdDrawIndirectCount) X(vkCmdDrawIndexedIndirectCount) \
    X(vkGetRefreshCycleDurationGOOGLE) X(vkGetPastPresentationTimingGOOGLE) \
    X(vkCreateSamplerYcbcrConversion) X(vkDestroySamplerYcbcrConversion)   // zero-copy movie pictures (movie_zc_vk.c): the decoder's Y'CbCr buffers sampled as RGB
#define VK_DECL(f) extern PFN_##f f;
VK_FNS(VK_DECL) VK_DEV_FNS(VK_DECL)

// ---- device ----
typedef struct {
    VkInstance inst; VkPhysicalDevice pd; VkDevice dev; VkQueue q; uint32_t qf;
    VkPhysicalDeviceProperties props; VkPhysicalDeviceMemoryProperties mem;
    VkPipelineCache pcache;
    int has_bc, has_astc, has_gs, has_depth_clip_ctl, has_custom_border, has_robust2, has_xfb, has_imageless, has_dtiming, has_dynrender, has_timeline, has_draw_count;   // has_draw_count: drawIndirectCount (multi-draws by the GPU's count)
    int has_filter_cubic; // EXT filter_cubic enabled; each blit still checks the source format capability
    int has_occ_precise;  // exact samples-passed query counts (used as a ratio by Bananza, not only a visibility Boolean)
    int has_barycentric;   // VK_KHR_fragment_shader_barycentric: per-vertex fragment inputs (Ampere LDTRAM)
    int dyn_ssbo;          // DM_VK_DYN_SSBO=1 at startup: direct graphics storage buffers use dynamic offsets within device limits
    int has_bda;           // bufferDeviceAddress (Vulkan 1.2 or VK_KHR_buffer_device_address): the pools' buffers get GPU addresses for raw global memory (vk_gmem_map)
    float max_aniso;
    // subgroups (vk_init): VkPhysicalDeviceSubgroupProperties' size, stages, operations and quadOperationsInAllStages; VkPhysicalDeviceSubgroupSizeControlProperties'
    // range and maxComputeWorkgroupSubgroups; DM_VK_SUBGROUP_PIN's required size (sg_pin) and the stages it is given to (vk_sg_pin)
    uint32_t sg_size, sg_stages, sg_ops, sg_quad_all, sg_min, sg_max, sg_maxwgs, sg_pin_stages; VkPipelineShaderStageRequiredSubgroupSizeCreateInfo sg_pin;
} vkdev_t;
extern vkdev_t V;
const void *vk_sg_pin(VkShaderStageFlagBits stage, const uint32_t *local);   // DM_VK_SUBGROUP_PIN: a game pipeline stage's pNext, NULL when the driver picks its size (local: a compute shader's workgroup size)
// gfx_vk_subgroup.c (no Vulkan): the module as subgroups of one invocation when it uses a subgroup class not in ops (its stage's; 0 when the device has none there).
// NULL when there is nothing the stage lacks (why empty) or it uses something not covered (why names it); the module then stays as it is
uint32_t *spv_one_lane(const uint32_t *w, uint32_t nw, uint32_t ops, uint32_t *onw, char *why, size_t nwhy);
uint32_t vk_mem_type(uint32_t bits, VkMemoryPropertyFlags want, VkMemoryPropertyFlags avoid);
void vk_init(void);   // once, from mtl_init or the first use

// ---- memory pools (gfx_vk_core.c): the guest's pool memory is GPU memory (AHardwareBuffer mapped over it) ----
typedef struct { uint64_t lo, hi; VkBuffer buf; VkDeviceMemory mem; void *ahb; int zero_copy, noncoherent; uint8_t *shadow, *baseline; uint64_t serial, base, end; int flags, synced, baseline_complete; uint64_t stamp, stamp0; const void *span; uint64_t gpu; int backed, addressed; uint64_t boff; void *slab; uint32_t slot; } vkpool_t;   // boff: [lo, hi) is at this offset of buf (a slab slot, upstream nx2android 98bee61d2; 0 otherwise), so a binding's offset is addr - lo + boff and gpu includes it; backed: has its VkBuffer (lazy pools get one at their first GPU reference, vk_pool_of); addressed: the game took a GPU address in it (mtl_pool_addressed); serial: unique per pool made (caches of views over a pool's buffer key on it); [base, end): the game's pool, of which [lo, hi) is one chunk (ZC_CHUNK); flags: the NVN pool's (vk_pool_set_flags); baseline: last synchronized GPU-copy contents; baseline_complete: refreshed by a full completion merge, consumed by the next sync; synced, stamp, stamp0, span: a copied pool's incremental sync
vkpool_t *vk_pool_of(uint64_t addr, uint64_t len);   // the pool containing [addr, addr+len), NULL if none; backed (its buffer made) if it was not yet
vkpool_t *vk_pool_peek(uint64_t addr, uint64_t len);   // the same without backing it: for CPU-side range checks (p->buf may be VK_NULL_HANDLE)
vkpool_t *vk_pool_holding(uint64_t addr, uint64_t len);   // the chunk addr lies in when [addr, addr+len) lies in its pool (across chunks too): for CPU reads of the pool's memory, no buffer
extern __thread const char *vk_pool_tag;   // what the lookup is for (set by the caller: "uniform", "storage", "vertex", "index", "copy"...), named when a range across two chunks has no buffer
void vk_pool_stats(uint64_t *pools, uint64_t *backed, uint64_t *bytes, uint64_t *backed_bytes);   // chunks made and backed so far, and their bytes (lazy pools: gfx_vk_core.c)
// gpu (vkpool_t): the buffer's device address (0 without has_bda); its delta from the chunk's guest address lo is what the raw global memory table carries
VkBuffer vk_gmem_map(VkDeviceSize *len);   // the memory pool table for raw global memory (dmsh_abi.h DMSH_GMEM_MAP_*; RK_GMEM_MAP), rebuilt when the pools change; *len: its bytes
int vk_buf_at(uint64_t addr, uint64_t len, VkBuffer *buf, VkDeviceSize *off);   // GPU buffer + offset of a pool address range
// Pools that could not be made zero-copy keep a GPU copy refreshed before each submission (vk_pools_sync, called from mtl_sub_begin)
void vk_pools_sync(VkCommandBuffer cb);

// ---- submissions (gfx_vk_core.c) ----
typedef struct vksub vksub_t;
VkCommandBuffer vk_cmd(void);          // the current submission's main command buffer (recording)
int vk_cmd_split(void);               //1 only if a new command-buffer part began
int vk_cmd_parts_left(void);
void vk_cmd_split_refused(int capacity);
int vk_readback_ranges(void);
void vk_readback_unknown(void);   // force full cached-pool merge for this submission
int vk_sub_open(void);                 // a submission is being recorded
VkCommandBuffer vk_upload_cmd(void);   // the current submission's upload command buffer (runs before the main one)
void *vk_scratch(VkDeviceSize n, VkDeviceSize align, VkBuffer *buf, VkDeviceSize *off);   // host-visible memory that lives until this submission completes
// Bounded staging for a texture's first upload once this submission's scratch is over DM_VK_SCRATCH_MAX_MB (gfx_vk_tex.c pkg_upload): one 128 MB buffer and a command
// buffer of its own, submitted and waited for when full and before this submission goes to the queue (gfx_vk_core.c)
int vk_stage_wanted(void);                            // this submission's scratch is over the cap: upload through the staging buffer
VkCommandBuffer vk_stage_cmd(void);                   // the staging command buffer, begun
void *vk_stage_alloc(VkDeviceSize n, VkDeviceSize align, VkBuffer *buf, VkDeviceSize *off);   // room in it (flushed first when full; a request over its size falls back to vk_scratch)
void vk_stage_flush(void);                            // submit what it holds and wait
extern __thread const char *vk_mem_tag;              // device memory accounting (gfx_vk_core.c mem_acct): the tag the next vkAllocateMemory on this thread is counted under
void vk_mem_acct_report(const char *why);             // the per-tag table, logged
void vk_defer_free_buffer(VkBuffer b, VkDeviceMemory m);   // destroyed once the GPU has passed the current submission
void vk_defer_free_image(VkImage i, VkDeviceMemory m);
VkDeviceMemory vk_image_memory(const VkMemoryRequirements *req, uint32_t type, VkDeviceSize *off, void **sub);   // NULL on failure; *sub: a slot of a shared block, or NULL (memory of its own)
void vk_defer_free_image_sub(VkImage i, void *sub);
void vk_defer_free_view(VkImageView v);
void vk_defer_free_bview(VkBufferView v);   // a texel buffer view, destroyed when this submission completes
VkDescriptorSet vk_alloc_set(VkDescriptorSetLayout l);    // from this submission's descriptor pools
uint64_t vk_descriptor_lease(VkDescriptorSet set, int rewrite); // current slot's immutable-content token; rewrite invalidates old tokens
int vk_descriptor_claim(VkDescriptorSet set, uint64_t lease); // exact completed-slot token, once per submission
int vk_registry_snapshot(uint64_t *textures, uint64_t *samplers); // semantic identity visible to this replay job, or 0 if unproven
void vk_table_report(void); // per-600-present table build/reuse costs

// Samples-passed is cumulative across passes/submissions until ResetCounter. Only guest draws are counted.
int vk_occ_counting(void); // a skipped rasterizing draw may change a later samples-passed report
void vk_occ_draw(void);   // immediately before an actual guest draw, inside its render pass
void vk_occ_end(void);    // before ending a render pass (also used by reports/resets inside a pass)
void vk_occ_reset(void);  // NVN samples-passed ResetCounter, preserving command order
uint64_t vk_reports_fence_done(void);   // report/readback writes landed, before event release
uint64_t vk_fence_done(void);   // the highest submission fence the GPU has completed
uint64_t vk_frame_current(void); // frame identity of the job currently replaying
void vk_perf_frame_seal(uint64_t frame); // after the frame's last submission; no later submission may use this identity
// Calibrated CLOCK_MONOTONIC endpoints, only after every sealed submission completes.
// 1: complete valid timing; 0: pending/unknown/expired; -1: dropped or invalid timing.
int vk_perf_frame_gpu(uint64_t frame, uint64_t *begin, uint64_t *end, uint64_t *busy);

// ---- formats and textures (gfx_vk_tex.c) ----
// dec: a BC format the device lacks, decoded on the CPU (gfx_vk_tex.c bc_decode) into vk, dbpp bytes a texel; bw, bh and bpb stay the guest's blocks
typedef struct { VkFormat vk; VkComponentMapping swz; uint8_t kind; uint8_t depth, stencil, compressed, is_int, is_sint, bw, bh, bpb, dec, dbpp; } vkfmt_t;
const vkfmt_t *vk_fmt(int nvn_format);   // NULL when the format has no Vulkan counterpart on this device
void vk_bc_decode(const vkfmt_t *f, const uint8_t *src, size_t pitch, uint32_t w, uint32_t h, uint32_t z, uint8_t *dst);   // (f->dec) gfx_vk_tex.c bc_decode
typedef struct vkimg vkimg_t;
struct vkimg {
    VkImageUsageFlags usage; VkImageCreateFlags cflags;   // as created (an imageless framebuffer names them)
    VkImage img; VkDeviceMemory mem; void *sub; VkFormat fmt; VkImageViewType vtype; VkImageAspectFlags aspect;
    uint32_t w, h, d, layers, levels, samples; int target, nvn_format;
    VkImageLayout layout;           // version 1 keeps every image in GENERAL after its first use
    int packaged;                   // filled from guest memory (deswizzled upload), re-uploaded when its pages change
    uint64_t pkg_stamp; const void *pkg_span; uint64_t pkg_seen;
    uint64_t pkg_hash;              // hash of the guest bytes it was last uploaded from: pages written around it with its own bytes unchanged keep what the GPU wrote (gfx_vk_tex.c vk_tex)
    const uint8_t *pkg_mem; size_t pkg_len; uint64_t pkg_used; int pkg_released;   // its storage, the present it was last drawn at, protection given up (vk_tex_sweep)
    uint64_t used;                  // the present it was last looked up at (vk_tex: evict_overlaps)
    int looked;                     // looked up inside a submission once (vk_tex: its first use, when its content is decided; a lookup outside one records nothing)
    uint32_t pool_ck;               // the pool generation (vk_pool_gen) at which its packaged storage was last found mapped for the pool tables (vk_tex_pool_ok)
    uint64_t used_bind;   // the present of its last lookup other than by a pool table fill (gfx_vk_tex.c evict_overlaps)
    uint64_t rt_last, rt_frames;     // as an attachment: 1 + the present it was last drawn in, and in how many presents in a row up to it (gfx_vk_exec.c async_pipe)
    uint64_t fdz_frame;             // depth images: 1 + the present at which a fragment shader that replaces depth last wrote it (gfx_vk_exec.c fdz)
    int gpu_w;                      // the GPU wrote it: an attachment, a storage image, a copy or clear destination (gfx_vk_exec.c); lin_copy keeps that until the CPU writes its storage
    uint64_t lin_stamp; const void *lin_span; const uint8_t *lin_mem; size_t lin_len;   // lin_copy: the storage watched since the GPU wrote it
    uint64_t lin_gen;               // linear textures: the upload command buffer (g_up_gen) that last copied it from its pool (vk_tex: lin_copy)
    uint64_t depth_sub;   // the submission (gfx_vk_exec.c g_set_epoch) whose passes last used it as their depth attachment
    uint64_t depth_cut;   // and the command buffer within it (gfx_vk_exec.c g_cuts)
    int fixed_fmt;   // created with a format list (DM_VK_UBWC): views only in its format or its sRGB twin
    int promoted;    // created in a storage-capable float format (R16G16B16A16_SFLOAT) in place of one the device cannot store to (E5B9G9R9, B10G11R11; gfx_vk_tex.c storage_promote): views, copies
                     // and attachments take its format; its bytes are never the guest's, so raw uploads and buffer copies of it are skipped (the image is the host's own, as every vkimg)
    VkFormat fmt2;   // with fixed_fmt: the other format of its list (its sRGB twin), VK_FORMAT_UNDEFINED when the list is its format alone
    VkImageView full;               // a view of the whole image as created
    VkBufferView tbuf;              // buffer textures: a texel buffer view of the pool
    uint32_t pins; int orphan;      // queued texture snapshots; a detached image survives until its final pin
    uint64_t last_use;              // submission fence of the last use
    uint64_t pool; int64_t off, size; int flags;   // its storage (pool, offset, layout size) and NVN texture flags: images over the same memory find each other
    struct vkimg *ovl[8]; int novl;   // the other images over some of the same memory (a texture of another description over it: on the Switch it sees these texels)
    uint64_t wseq, rseen;             // when it was last rendered to (gfx_vk_tex.c g_wseq); the newest overlapping write already reported
    uint64_t vkbytes;                 // the image's device memory (vkGetImageMemoryRequirements size): what the texture budget counts (gfx_vk_budget.c)
    uint64_t zc_gen;                  // a texture over a movie picture slot (movie_zc_vk.c zc_tex_refresh): the slot's change count last converted into it
};
vkimg_t *vk_tex(const tex_t *t);                                 // the image of a texture object (created, uploaded or refreshed as needed)
int vk_tex_storage_prepare(const tex_t *t, const view_t *view);                    // before descriptor collection/render pass: preserve contents when adding storage usage
void vk_lrz_forget(vkimg_t *im);                                 // an image is being dropped: the pass code forgets it (gfx_vk_exec.c)
int vk_window_bottom_up(void);                                   // the game draws in lower-left origin mode: its window image is stored bottom-up (gfx_vk_exec.c)
VkImageView vk_tex_view(const tex_t *t, const view_t *v, int dim, int arrayed, int view_kind, vkimg_t **img_out);   // 0 sampled, 1 attachment, 2 storage (identity, one mip)
void vk_watch(const vkimg_t *im, const char *what);   // DM_VK_WATCH (gfx_vk_exec.c)
vkimg_t *vk_tex_by_id(int tid, const view_t **view_out, const tex_t **tex_out);   // registered texture ID → image (NULL if none)
VkSampler vk_samp_by_id(int sid);
extern __thread VkFormat vk_view_fmt; extern __thread uint32_t vk_view_layers;   // the format and layer count of the view vk_tex_view last returned on this thread
VkImageView vk_dummy_view(int dim, int arrayed, int ms, int storage, int is_int);
VkSampler vk_dummy_sampler(void);
VkBufferView vk_dummy_tbuf(int storage);
void vk_end_pass_land(void);   // vk_end_pass, then the deferred uniform updates land (before copies, dispatches, the submission's end)
void vk_img_barrier_all(VkCommandBuffer cb);   // version 1: one global memory barrier (all images stay in GENERAL)
void vk_tex_prepare(VkCommandBuffer cb, vkimg_t *im);   // first use: transition from UNDEFINED to GENERAL (and upload a packaged texture)
// ---- texture memory budget (gfx_vk_budget.c): the images' device memory kept under DM_VK_IMAGE_BUDGET_MB by dropping the least recently drawn ----
typedef struct { void *h; uint32_t i; vkimg_t *im; } vk_img_ref;   // an image and where its table holds it (gfx_vk_tex.c)
size_t vk_img_list(vk_img_ref **out, size_t *cap);   // every live image into *out (realloc'd as needed); the caller holds the image lock (called from vk_tex)
void vk_img_evict(const vk_img_ref *r);               // drops it (img_drop; the memory goes once the GPU is past this submission); under the image lock
uint64_t vk_img_vkbytes(void);                        // device memory of the live images
void vk_img_budget_make_room(uint64_t incoming);      // before a new image of incoming bytes: evicts down to the budget (gfx_vk_budget.c); under the image lock, inside a submission
void vk_img_budget_stats(uint64_t *budget, uint64_t *evicted, uint64_t *evicted_bytes, uint64_t *sweeps);

// ---- shaders (gfx_vk_shader.c) ----
enum { RK_SUPPORT, RK_CBUF, RK_C1, RK_DRV, RK_SBUF, RK_UBO_PTR, RK_SBUF_UBO, RK_TEX, RK_TBUF, RK_IMG, RK_IMG_TBUF, RK_TEX_C2A, RK_TEX_UBO, RK_TBUF_UBO, RK_IMG_UBO, RK_TEXD, RK_SAMPD,
       RK_GMEM_MAP, RK_GMEM_LOG,   // raw global memory (dmsh_abi.h DMSH_KIND_GMEM_MAP / _LOG): the pool table (vk_gmem_map) and the write log of the stores through it (stage_set); ids in step with shader_vk.py KIND
       RK_MESH_CTL, RK_MESH_TASKOUT, RK_MESH_LAUNCH, RK_MESH_VTX, RK_MESH_IDX,   // the mesh emulation's buffers (shader_meshwrap.py dm_mesh_*; gfx_vk_exec.c mesh_draw sets them before stage_set)
       RK_TEXP, RK_SAMPP,   // the texture and sampler pools indexed by handle ID (shader_vk.py classify pools): separate sampled image arrays and a sampler array in set 0's pool bindings (vk_tbl_dsl; gfx_vk_exec.c tbl_set)
       RK_GS_CLAIM = 240,        // synthetic runtime-only per-draw geometry side-effect claim storage; not an on-disk KIND
       RK_UNKNOWN = 255 };
enum { RD_UBO, RD_SSBO, RD_SAMPLED, RD_STORAGE, RD_TBUF, RD_STORAGE_TBUF, RD_IMAGE, RD_SAMPLER };
enum { RS_NONE, RS_TEX, RS_SLOT, RS_ID, RS_UBO };
typedef struct __attribute__((packed)) { uint8_t kind, desc, dim, flags; uint16_t binding, count; int16_t reg; uint8_t samp, fmt; int32_t off, samp_arg; } dmsv_bind;
_Static_assert(sizeof(dmsv_bind) == 20, "dmsv_bind");
typedef struct {
    uint64_t key; int stage;          // NVN stage: 0 vertex 1 fragment 2 geometry 3 tess control 4 tess eval 5 compute
    VkShaderModule mod;
    VkDescriptorSetLayout dsl; int set;   // the set this stage's bindings were moved to (vertex 1, fragment 2, geometry 3; compute 0: set 0 of a graphics layout is the texture table, vk_tbl_dsl)
    int nbind; const dmsv_bind *b;
    uint8_t tbl[96]; int ntbl, tbl_reg, tbl_off;   // bindings moved to the texture table set (1 + slot: 2D, 2D array, cube; gfx_vk_exec.c tbl_set), their count, and the uniform buffer (register, byte offset) the handles come from
    int npool;                        // of those, the pool arrays (RK_TEXP / RK_SAMPP; tbl[i] = 1 + TBL_SLOTS + pool slot: 2D, 2D array, cube, 3D, samplers), bound by texture / sampler ID (tbl_set)
    const uint8_t *c1; uint32_t c1_len; VkBuffer c1_buf; VkDeviceSize c1_off;
    uint32_t local[3];
    uint32_t ubo_extent[16];           // proven base-module read extents by guest uniform slot; zero means unknown (graphics shadow copies only)
    uint8_t dyn[96]; int ndyn;          // 0 static, 1 dynamic uniform, 2 dynamic storage; ndyn includes both, sorted by binding at each bind
    int xfb; uint32_t xfb_stride[4];    // a vertex shader that captures transform feedback (its SPIR-V has the Xfb execution mode), and each buffer's stride
    uint32_t *spv, nspv;                // its SPIR-V as the module was made from it, for the variants below
    struct { uint64_t um, im; uint32_t xo; int fd; VkShaderModule mod; } var[8]; int nvar;   // variants whose sampled bindings (bit = index in b) of an integer view read uint (um) / int (im) texels,
                                                                                    // that declare the vec4 outputs xo (bit = location) the next stage reads, and (fd) that write gl_FragDepth = gl_FragCoord.z
    int depth_repl;                     // a fragment shader that writes gl_FragDepth (DepthReplacing / DepthGreater / DepthLess)
    uint32_t in_locs, out_locs;         // the vec4 (or per-vertex vec4) inputs and outputs it declares, by location
    uint32_t omap; int omap_known;      // a fragment shader's output map (SPH word 18, shader_vk.py ps_omap): 4 bits per colour output; the components it writes (gfx_vk_exec.c wmask)
    uint32_t gs_claim_invocations;     // transformed geometry shader: claim slots per input primitive, zero when unchanged
    int storage_images;                 // replay preflight: 0 unknown, 1 has reflected storage images, 2 none
    int setmemo;                        // gfx_vk_exec.c stage_set: 1 its set may be memoized per stage, 2 not (it reads handles or addresses from uniform buffer memory), 0 not known yet
    int pv_gs;   // a geometry shader shader_vk.py generated for a fragment shader that reads inputs per vertex (shader_pvgs.py): bound in place of a program's missing geometry stage
    // an NVN2 task (1) or mesh (2) kernel (bundle stage 6 / 7; shader_meshwrap.facts in its header words): its workgroup's threads, its payload's words, whether the mesh shader has a task,
    // a primitive's vertices, the most vertices and primitives a workgroup, and the words a vertex has in the emulation's vertex buffer (gfx_vk_exec.c mesh_draw)
    int mesh_kind; uint32_t mesh_threads, mesh_pay, mesh_task, mesh_verts, mesh_max_v, mesh_max_p, mesh_ncomp;
} vkshader_t;
vkshader_t *vk_shader(uint64_t key, int stage);   // NULL when the bundle has no such shader
// Texture tables: a shader's arrays of TBL_MIN or more combined image samplers whose handles are consecutive words of one of the game's uniform buffers (element k = the handle at
// off + 8k; Bezel Engine's material texture lists in Nintendo Switch 2 Welcome Tour: three arrays of 8192 in 420 fragment shaders, one each for 2D, 2D array and cube reads). They go
// to set 0, whose layout (vk_tbl_dsl: TBL_SLOTS bindings of vk_tbl_n() each) is the same in every graphics pipeline layout, so a table set stays bound across pipelines and is bound
// again only when the handles change; the arrays' lengths in the SPIR-V are cut to vk_tbl_n() (the UMin clamps on their indices with them). Adreno copies every bound set into
// command memory at each vkCmdBindDescriptorSets: with the tables in the per-draw fragment sets a Welcome Tour scene submission mapped 2.4-3.3 GB. DM_VK_TBL_N: the length (0: off)
enum { VK_NSETS = 6, TBL_SLOTS = 3, TBL_MIN = 256 };
// Texture and sampler pools (RK_TEXP / RK_SAMPP): the translator's arrays over the whole ID ranges (2^20 textures, 4096 samplers) for a handle it could trace to no constant-buffer
// word (Ocarina of Time's gsys materials read theirs from records in storage buffers). Set 0 holds, after the texture tables, POOL_TEX_SLOTS sampled-image arrays of vk_pool_n()
// (element k = the registered texture of ID k through a 2D, 2D-array, cube or 3D view; the arrays declare floating-point texels) and a sampler array of vk_pool_samp_n() (element
// k = sampler ID k), rebuilt when a registration changes (gfx_vk_exec.c tbl_set) and bound with the table set; the SPIR-V arrays and their index clamps are cut to those lengths
// (gfx_vk_shader.c). DM_VK_POOL_N: the texture length (65536 unless set; 0: such shaders are not drawn), bounded by the device's per-stage descriptor limits
enum { POOL_TEX_SLOTS = 4, POOL_B0 = TBL_SLOTS, POOL_SAMP_B = TBL_SLOTS + POOL_TEX_SLOTS, SET0_BINDINGS = TBL_SLOTS + POOL_TEX_SLOTS + 1 };
VkDescriptorSetLayout vk_tbl_dsl(void);
int vk_tbl_n(void);
int vk_pool_n(void); int vk_pool_samp_n(void);
int vk_tex_id_max(void); int vk_samp_id_max(void);   // the highest texture / sampler ID registered so far (gfx_vk_tex.c)
int vk_tex_pool_ok(const tex_t *t);   // may the pool tables bind this registered texture: 2 its image has its contents, 1 the table may give it its first use (small, storage mapped), 0 the dummy (gfx_vk_tex.c)
void vk_tex_pool_fill(int on);        // around a pool-table fill: first uses made by it do not bump g_tex_gen (a real bind's first use does, so the tables are built again with it)
vkshader_t *vk_shader_gs_topo(vkshader_t *gs, VkPrimitiveTopology topo);   // the geometry shader translated for the draw's input primitive (shader_vk.py gs_variant_key); gs when there is none
enum { VK_ATEST_SPEC_FUNC = 0x7a00, VK_ATEST_SPEC_REF = 0x7a01 };   // the alpha test variant's specialization constants (gfx_vk_shader.c spv_alpha_test)
int vk_shader_observer_pair(vkshader_t *vs, vkshader_t *fs, VkShaderModule modules[2]); // opt-in atomic diagnostic pair
VkShaderModule vk_shader_module(vkshader_t *s, uint64_t um, uint64_t im, uint32_t xo, int fd);   // the module, or its variant (integer views; extra outputs; fragment depth written)
void vk_shaders_open(const char *path);
enum { VK_SUPPORT_BYTES = 65536 };
void vk_support_values(void *p);   // exact host support contents, also used by isolated diagnostic snapshots
VkBuffer vk_support_buffer(void);
VkBuffer vk_zero_buffer(void);
VkBuffer vk_gmem_log_disabled(void);   // all bits set: the translated log guard skips its atomic; real stores still run

// ---- replay (gfx_vk_exec.c) ----
void vk_queue_preserve(void);   // logical-queue bindings retained at each submission end
void vk_exec_reset(void);   // per submission
void vk_end_pass(void);     // close an open render pass

// ---- presentation (gfx_vk_present.c) ----
void vk_present_init(void);
uint64_t vk_present(const tex_t *t, double period, const int crop[4]);
enum { PT_DISPATCH = 0xd15 };   // vk_pass_ts: a dispatch timed as a pass (its format field)
vkshader_t *vk_shader_pv_gs(vkshader_t *fs);   // the generated pass-through geometry shader of a fragment shader with per-vertex inputs, NULL when the bundle has none
vkshader_t *vk_shader_mesh_vs(vkshader_t *ms);   // the generated pass-through vertex shader of a mesh kernel (shader_meshwrap.mesh_vs_key), NULL when the bundle has none
