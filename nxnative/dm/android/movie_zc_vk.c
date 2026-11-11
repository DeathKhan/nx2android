// Zero-copy movie pictures, the Android and Vulkan side (movie_zc.h; the platform-free parts are movie_zc.c). The codec renders into an AImageReader (AIMAGE_FORMAT_PRIVATE,
// GPU-sampled: the decoder's own layout, UBWC on Adreno); each picture's AHardwareBuffer is imported once (per buffer id, the reader recycles them) as a Vulkan image in the
// format vkGetAndroidHardwareBufferPropertiesANDROID reports (an external format, or a multi-planar one), sampled through a Y'CbCr conversion in an immutable sampler, and
// one compute dispatch a frame writes RGBA into the game's texture over the pool's picture slot (gfx_vk_tex.c vk_tex: zc_tex_refresh), recorded in the submission's upload
// command buffer so the game's draws of that submission read the new picture. The game's shader reads an ordinary RGBA8 texture, so no game pipeline changes
#include "gfx_vk.h"
#include "movie_zc.h"
#include <media/NdkImageReader.h>
#include <media/NdkImage.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "movie_yuv_comp.spv.h"

zc_frame *zc_frame_new(void);   // movie_zc.c
uint64_t mtl_last_fence(void);  // gfx_vk_core.c: the last committed submission's fence (the one being recorded commits as the next)
extern uint64_t g_tex_frame;    // gfx_vk_tex.c: presents so far (the import cache's age)
#define ZC_SETS 64   // descriptor sets in a ring: one per dispatch, reused 64 dispatches later (the GPU is long done with it)

struct zc_reader { AImageReader *rd; ANativeWindow *win; int refs, w, h, max; pthread_mutex_t mu; int64_t full_ns; int full_red; };   // full_ns: since when every image is out (0: not)
static pthread_mutex_t g_rd_mu = PTHREAD_MUTEX_INITIALIZER;   // readers' reference counts

static int zc_on(void) { static int v = -1; if (v < 0) { const char *e = getenv("DM_MOVIE_ZEROCOPY"); v = !e || atoi(e) != 0; } return v; }
int zc_gpu_ok(void) {
    static int ok = -1;
    if (ok >= 0) return ok;
    if (!zc_on() || !V.dev) return 0;   // (not cached: the device may come up later)
    VkPhysicalDeviceSamplerYcbcrConversionFeatures yf = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SAMPLER_YCBCR_CONVERSION_FEATURES};
    VkPhysicalDeviceFeatures2 f2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &yf}; vkGetPhysicalDeviceFeatures2(V.pd, &f2);
    uint32_t ne = 0; vkEnumerateDeviceExtensionProperties(V.pd, NULL, &ne, NULL); VkExtensionProperties *ex = calloc(ne, sizeof *ex); vkEnumerateDeviceExtensionProperties(V.pd, NULL, &ne, ex);
    int ahb = 0; for (uint32_t i = 0; i < ne; i++) if (!strcmp(ex[i].extensionName, "VK_ANDROID_external_memory_android_hardware_buffer")) ahb = 1;
    free(ex);
    ok = yf.samplerYcbcrConversion && ahb && vkCreateSamplerYcbcrConversion && vkGetAndroidHardwareBufferPropertiesANDROID;
    if (!ok) LOG("[RED] [movie] zero-copy pictures need Y'CbCr sampler conversion (%d) and the AHardwareBuffer import (%d): the CPU path stands", yf.samplerYcbcrConversion, ahb);
    return ok;
}
uint64_t zc_fence_done(void) { return V.dev ? vk_fence_done() : ~0ull; }

// ---- the reader ----
zc_reader *zc_reader_new(int w, int h, int max_images) {
    if (w <= 0 || h <= 0) { w = 1920; h = 1080; }
    zc_reader *r = calloc(1, sizeof *r); if (!r) return NULL;
    media_status_t st = AImageReader_newWithUsage(w, h, AIMAGE_FORMAT_PRIVATE, AHARDWAREBUFFER_USAGE_GPU_SAMPLED_IMAGE, max_images, &r->rd);
    if (st != AMEDIA_OK || !r->rd) { LOG("[RED] [movie] zero-copy: AImageReader %dx%d x%d refused (%d): the CPU path stands", w, h, max_images, (int)st); free(r); return NULL; }
    if (AImageReader_getWindow(r->rd, &r->win) != AMEDIA_OK || !r->win) { LOG("[RED] [movie] zero-copy: the AImageReader has no window"); AImageReader_delete(r->rd); free(r); return NULL; }
    r->refs = 1; r->w = w; r->h = h; r->max = max_images; pthread_mutex_init(&r->mu, NULL);
    LOG("[movie] zero-copy: AImageReader %dx%d, %d images, GPU-sampled private format", w, h, max_images);
    return r;
}
static void reader_unref(zc_reader *r) {
    pthread_mutex_lock(&g_rd_mu); int last = --r->refs == 0; pthread_mutex_unlock(&g_rd_mu);
    if (!last) return;
    AImageReader_delete(r->rd); pthread_mutex_destroy(&r->mu); free(r);
}
void zc_reader_free(zc_reader *r) { if (r) reader_unref(r); }
void *zc_reader_window(zc_reader *r) { return r ? r->win : NULL; }
zc_frame *zc_reader_acquire(zc_reader *r) {
    AImage *img = NULL;
    pthread_mutex_lock(&r->mu); media_status_t st = AImageReader_acquireNextImage(r->rd, &img); pthread_mutex_unlock(&r->mu);
    // Every image out is the codec's back-pressure, as designed (mavc_set_zero_copy): the codec decodes ahead of the game, the game holds up to 12 pictures (hle_codec_video.c) and
    // the codec waits for one back. Welcome Tour's 29.97 fps 4K intro reaches it in some runs on the base build too and plays its 660 pictures in the same time either way. Red only
    // when no image comes back for 2 s: the game holds pictures it does not release
    if (st == AMEDIA_IMGREADER_MAX_IMAGES_ACQUIRED) {
        struct timespec now; clock_gettime(CLOCK_MONOTONIC, &now); int64_t ns = (int64_t)now.tv_sec * 1000000000 + now.tv_nsec;
        if (!r->full_ns) { r->full_ns = ns; static int n; if (!n++) LOG("[movie] zero-copy: every one of the reader's %d images is out: the codec waits for the game to release pictures (its back-pressure)", r->max); }
        else if (ns - r->full_ns > 2000000000ll && !r->full_red) { r->full_red = 1; LOG("[RED] [movie] zero-copy: every one of the reader's %d images has been out for %.1f s: the game holds pictures it does not release", r->max, (ns - r->full_ns) / 1e9); }
    } else if (st == AMEDIA_OK && img) r->full_ns = 0;
    if (st != AMEDIA_OK || !img) return NULL;
    zc_frame *f = zc_frame_new(); if (!f) { AImage_delete(img); return NULL; }
    AHardwareBuffer *ahb = NULL; AImage_getHardwareBuffer(img, &ahb);
    int32_t w = 0, h = 0; AImage_getWidth(img, &w); AImage_getHeight(img, &h);
    AImageCropRect c = {0, 0, 0, 0}; AImage_getCropRect(img, &c);
    int64_t ts = 0; AImage_getTimestamp(img, &ts);
    f->img = img; f->ahb = ahb; if (ahb) AHardwareBuffer_getId(ahb, &f->id); f->w = w; f->h = h; f->ts_ns = ts;
    if (c.right > c.left && c.bottom > c.top) { f->cx = c.left; f->cy = c.top; f->cw = c.right - c.left; f->ch = c.bottom - c.top; }
    f->r = r; pthread_mutex_lock(&g_rd_mu); r->refs++; pthread_mutex_unlock(&g_rd_mu);
    return f;
}
void zc_frame_destroy(zc_frame *f) {   // (movie_zc.c, with its lock held: nothing here takes it)
    if (f->img) AImage_delete(f->img);
    zc_reader *r = f->r; free(f);
    if (r) reader_unref(r);
}

// ---- the Vulkan side ----
typedef struct { uint32_t w, h; float ox, oy, sx, sy; int32_t swap_rb; } zc_pc;   // the shader's push constants (movie_yuv.comp)
typedef struct {   // a conversion and its pipeline: per source format (and colour model)
    VkFormat fmt; uint64_t ext; VkSamplerYcbcrModelConversion model; VkSamplerYcbcrRange range; VkChromaLocation xo, yo; VkComponentMapping comp;
    VkSamplerYcbcrConversion conv; VkSampler samp; VkDescriptorSetLayout dsl; VkPipelineLayout pl; VkPipeline pipe; VkDescriptorPool dp; VkDescriptorSet sets[ZC_SETS]; uint32_t next;
} zc_conv;
static zc_conv g_conv[4]; static int g_nconv;
typedef struct { uint64_t id; VkImage img; VkDeviceMemory mem; VkImageView view; zc_conv *cv; uint32_t w, h; uint64_t last; } zc_imp;   // an imported buffer (per AHardwareBuffer id)
static zc_imp g_imp[24]; static int g_nimp;
static VkShaderModule g_zc_mod;
static int g_foreign = -1;   // VK_EXT_queue_family_foreign is on: the acquire barrier names VK_QUEUE_FAMILY_FOREIGN_EXT (else EXTERNAL)

static zc_conv *conv_get(VkFormat fmt, uint64_t ext, const VkAndroidHardwareBufferFormatPropertiesANDROID *fp, int bt709) {
    // an external format takes the driver's suggested model, range and chroma location (the only ones allowed for it); a known multi-planar format takes the stream's matrix
    VkSamplerYcbcrModelConversion model = ext ? fp->suggestedYcbcrModel : bt709 ? VK_SAMPLER_YCBCR_MODEL_CONVERSION_YCBCR_709 : VK_SAMPLER_YCBCR_MODEL_CONVERSION_YCBCR_601;
    VkSamplerYcbcrRange range = ext ? fp->suggestedYcbcrRange : VK_SAMPLER_YCBCR_RANGE_ITU_NARROW;
    VkChromaLocation xo = fp->suggestedXChromaOffset, yo = fp->suggestedYChromaOffset;
    VkComponentMapping comp = ext ? fp->samplerYcbcrConversionComponents : (VkComponentMapping){0, 0, 0, 0};
    for (int i = 0; i < g_nconv; i++) { zc_conv *c = &g_conv[i]; if (c->fmt == fmt && c->ext == ext && c->model == model && c->range == range && c->xo == xo && c->yo == yo) return c; }
    if (g_nconv == 4) { VKRED_ONCE(300, "zero-copy movie: more than 4 source formats"); return NULL; }
    zc_conv *c = &g_conv[g_nconv]; memset(c, 0, sizeof *c);
    c->fmt = fmt; c->ext = ext; c->model = model; c->range = range; c->xo = xo; c->yo = yo; c->comp = comp;
    VkExternalFormatANDROID xf = {VK_STRUCTURE_TYPE_EXTERNAL_FORMAT_ANDROID, NULL, ext};
    // nearest filters: the picture is converted 1:1, and a chroma filter equal to the sampler's needs no separate-reconstruction feature
    VkSamplerYcbcrConversionCreateInfo ci = {VK_STRUCTURE_TYPE_SAMPLER_YCBCR_CONVERSION_CREATE_INFO, ext ? &xf : NULL, fmt, model, range, comp, xo, yo, VK_FILTER_NEAREST, VK_FALSE};
    VkResult r = vkCreateSamplerYcbcrConversion(V.dev, &ci, NULL, &c->conv);
    if (r) { VKRED_ONCE(301, "zero-copy movie: vkCreateSamplerYcbcrConversion failed (%d) for format %d external %llu", r, fmt, (unsigned long long)ext); return NULL; }
    VkSamplerYcbcrConversionInfo cvi = {VK_STRUCTURE_TYPE_SAMPLER_YCBCR_CONVERSION_INFO, NULL, c->conv};
    VkSamplerCreateInfo si = {VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO, &cvi, 0, VK_FILTER_NEAREST, VK_FILTER_NEAREST, VK_SAMPLER_MIPMAP_MODE_NEAREST,
        VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, 0, VK_FALSE, 1, VK_FALSE, VK_COMPARE_OP_ALWAYS, 0, 0.25f,
        VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK, VK_FALSE};
    if ((r = vkCreateSampler(V.dev, &si, NULL, &c->samp))) { VKRED_ONCE(302, "zero-copy movie: sampler with conversion failed (%d)", r); return NULL; }
    VkDescriptorSetLayoutBinding b[2] = {{0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, &c->samp}, {1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL}};
    VkDescriptorSetLayoutCreateInfo li = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, NULL, 0, 2, b};
    VkPushConstantRange pr = {VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(zc_pc)};
    if ((r = vkCreateDescriptorSetLayout(V.dev, &li, NULL, &c->dsl))) { VKRED_ONCE(303, "zero-copy movie: descriptor set layout failed (%d)", r); return NULL; }
    VkPipelineLayoutCreateInfo pl = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, NULL, 0, 1, &c->dsl, 1, &pr};
    if ((r = vkCreatePipelineLayout(V.dev, &pl, NULL, &c->pl))) { VKRED_ONCE(304, "zero-copy movie: pipeline layout failed (%d)", r); return NULL; }
    if (!g_zc_mod) { VkShaderModuleCreateInfo mi = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, NULL, 0, sizeof movie_yuv_comp_spv, movie_yuv_comp_spv};
        if ((r = vkCreateShaderModule(V.dev, &mi, NULL, &g_zc_mod))) { VKRED_ONCE(305, "zero-copy movie: shader module failed (%d)", r); return NULL; } }
    VkComputePipelineCreateInfo pi = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO, NULL, 0, {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, NULL, 0, VK_SHADER_STAGE_COMPUTE_BIT, g_zc_mod, "main", NULL}, c->pl, VK_NULL_HANDLE, -1};
    r = vkCreateComputePipelines(V.dev, V.pcache, 1, &pi, NULL, &c->pipe); vk_pipeline_cache_changed();   // (gfx_vk_pipeline_cache.inc: a new entry to save; upstream nx2android #213)
    if (r) { VKRED_ONCE(306, "zero-copy movie: compute pipeline failed (%d)", r); return NULL; }
    // a combined sampler with a conversion may take several descriptors (combinedImageSamplerDescriptorCount, 3 at most): the pool is sized for that
    VkDescriptorPoolSize ps[2] = {{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 3 * ZC_SETS}, {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, ZC_SETS}};
    VkDescriptorPoolCreateInfo dpi = {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, NULL, 0, ZC_SETS, 2, ps};
    if ((r = vkCreateDescriptorPool(V.dev, &dpi, NULL, &c->dp))) { VKRED_ONCE(307, "zero-copy movie: descriptor pool failed (%d)", r); return NULL; }
    VkDescriptorSetLayout ls[ZC_SETS]; for (int i = 0; i < ZC_SETS; i++) ls[i] = c->dsl;
    VkDescriptorSetAllocateInfo ai = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, NULL, c->dp, ZC_SETS, ls};
    if ((r = vkAllocateDescriptorSets(V.dev, &ai, c->sets))) { VKRED_ONCE(308, "zero-copy movie: descriptor sets failed (%d)", r); return NULL; }
    g_nconv++;
    VKLOG("zero-copy movie: conversion for Vulkan format %d external %llu, model %d range %d chroma %d/%d (%s)", fmt, (unsigned long long)ext, model, range, xo, yo, ext ? "the driver's suggestion" : bt709 ? "BT.709 narrow" : "BT.601 narrow");
    return c;
}
static void imp_drop(zc_imp *m) { if (m->view) vk_defer_free_view(m->view); if (m->img) vk_defer_free_image(m->img, m->mem); memset(m, 0, sizeof *m); }
static zc_imp *import_frame(zc_frame *f) {
    for (int i = 0; i < g_nimp; i++) if (g_imp[i].id == f->id && g_imp[i].img) { g_imp[i].last = g_tex_frame; return &g_imp[i]; }
    zc_imp *m = NULL;
    if (g_nimp < 24) m = &g_imp[g_nimp++];
    else { m = &g_imp[0]; for (int i = 1; i < 24; i++) if (g_imp[i].last < m->last) m = &g_imp[i]; imp_drop(m); }   // (a reader recycles at most its image count; the oldest goes)
    AHardwareBuffer_Desc d; AHardwareBuffer_describe(f->ahb, &d);
    VkAndroidHardwareBufferFormatPropertiesANDROID fp = {VK_STRUCTURE_TYPE_ANDROID_HARDWARE_BUFFER_FORMAT_PROPERTIES_ANDROID};
    VkAndroidHardwareBufferPropertiesANDROID ap = {VK_STRUCTURE_TYPE_ANDROID_HARDWARE_BUFFER_PROPERTIES_ANDROID, &fp};
    VkResult r = vkGetAndroidHardwareBufferPropertiesANDROID(V.dev, f->ahb, &ap);
    if (r) { VKRED_ONCE(310, "zero-copy movie: vkGetAndroidHardwareBufferPropertiesANDROID failed (%d)", r); return NULL; }
    VkFormat fmt = fp.format; uint64_t ext = fmt == VK_FORMAT_UNDEFINED ? fp.externalFormat : 0;
    if (fmt == VK_FORMAT_UNDEFINED && !ext) { VKRED_ONCE(311, "zero-copy movie: the buffer has neither a Vulkan format nor an external one"); return NULL; }
    static int once; if (!once++) VKLOG("zero-copy movie: AHardwareBuffer %ux%u format %#x usage %#llx: Vulkan format %d external %llu, features %#x, suggested model %d range %d chroma %d/%d",
        d.width, d.height, d.format, (unsigned long long)d.usage, fp.format, (unsigned long long)fp.externalFormat, fp.formatFeatures, fp.suggestedYcbcrModel, fp.suggestedYcbcrRange, fp.suggestedXChromaOffset, fp.suggestedYChromaOffset);
    zc_conv *cv = conv_get(fmt, ext, &fp, d.height >= 720); if (!cv) return NULL;
    VkExternalFormatANDROID xf = {VK_STRUCTURE_TYPE_EXTERNAL_FORMAT_ANDROID, NULL, ext};
    VkExternalMemoryImageCreateInfo em = {VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO, ext ? (void *)&xf : NULL, VK_EXTERNAL_MEMORY_HANDLE_TYPE_ANDROID_HARDWARE_BUFFER_BIT_ANDROID};
    VkImageCreateInfo ci = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, &em, 0, VK_IMAGE_TYPE_2D, ext ? VK_FORMAT_UNDEFINED : fmt, {d.width, d.height, 1}, 1, 1, VK_SAMPLE_COUNT_1_BIT, VK_IMAGE_TILING_OPTIMAL,
        VK_IMAGE_USAGE_SAMPLED_BIT, VK_SHARING_MODE_EXCLUSIVE, 0, NULL, VK_IMAGE_LAYOUT_UNDEFINED};
    if ((r = vkCreateImage(V.dev, &ci, NULL, &m->img))) { VKRED_ONCE(312, "zero-copy movie: vkCreateImage of the imported buffer failed (%d)", r); m->img = VK_NULL_HANDLE; return NULL; }
    VkMemoryDedicatedAllocateInfo ded = {VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO, NULL, m->img, VK_NULL_HANDLE};
    VkImportAndroidHardwareBufferInfoANDROID imp = {VK_STRUCTURE_TYPE_IMPORT_ANDROID_HARDWARE_BUFFER_INFO_ANDROID, &ded, f->ahb};
    VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, &imp, ap.allocationSize, vk_mem_type(ap.memoryTypeBits, 0, 0)};
    if ((r = vkAllocateMemory(V.dev, &ai, NULL, &m->mem))) { VKRED_ONCE(313, "zero-copy movie: importing the buffer's memory failed (%d)", r); vkDestroyImage(V.dev, m->img, NULL); m->img = VK_NULL_HANDLE; return NULL; }
    if ((r = vkBindImageMemory(V.dev, m->img, m->mem, 0))) { VKRED_ONCE(314, "zero-copy movie: vkBindImageMemory failed (%d)", r); vkDestroyImage(V.dev, m->img, NULL); vkFreeMemory(V.dev, m->mem, NULL); m->img = VK_NULL_HANDLE; return NULL; }
    VkSamplerYcbcrConversionInfo cvi = {VK_STRUCTURE_TYPE_SAMPLER_YCBCR_CONVERSION_INFO, NULL, cv->conv};
    VkImageViewCreateInfo vi = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, &cvi, 0, m->img, VK_IMAGE_VIEW_TYPE_2D, ci.format, {0, 0, 0, 0}, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
    if ((r = vkCreateImageView(V.dev, &vi, NULL, &m->view))) { VKRED_ONCE(315, "zero-copy movie: the imported image's view failed (%d)", r); vkDestroyImage(V.dev, m->img, NULL); vkFreeMemory(V.dev, m->mem, NULL); m->img = VK_NULL_HANDLE; return NULL; }
    m->id = f->id; m->cv = cv; m->w = d.width; m->h = d.height; m->last = g_tex_frame;
    return m;
}
int zc_tex_wants(const void *mem, uint64_t len) { return zc_slot_find(mem, len, NULL, NULL); }
// The game's texture over a slot: the slot's frame is converted into its image when the slot changed since the last conversion into this image (im->zc_gen)
// The conversion writes 8-bit RGBA. A R8G8B8A8_UNORM texture with storage usage (img_new gives it to textures over a slot) is written directly; any other 4-byte format (BGRA8, sRGB,
// a texture made before its slot existed) goes through a staging image of the picture's size and one vkCmdCopyImage (the bytes are copied as they are: BGRA8 gets its channels swapped in the shader)
static struct { VkImage img; VkDeviceMemory mem; VkImageView view; uint32_t w, h; } g_stage;
static int stage_get(uint32_t w, uint32_t h) {
    if (g_stage.img && g_stage.w == w && g_stage.h == h) return 1;
    if (g_stage.img) { vk_defer_free_view(g_stage.view); vk_defer_free_image(g_stage.img, g_stage.mem); memset(&g_stage, 0, sizeof g_stage); }
    VkImageCreateInfo ci = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, NULL, 0, VK_IMAGE_TYPE_2D, VK_FORMAT_R8G8B8A8_UNORM, {w, h, 1}, 1, 1, VK_SAMPLE_COUNT_1_BIT, VK_IMAGE_TILING_OPTIMAL,
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, VK_SHARING_MODE_EXCLUSIVE, 0, NULL, VK_IMAGE_LAYOUT_UNDEFINED};
    VkResult r = vkCreateImage(V.dev, &ci, NULL, &g_stage.img);
    if (r) { VKRED_ONCE(319, "zero-copy movie: the staging image failed (%d)", r); g_stage.img = VK_NULL_HANDLE; return 0; }
    VkMemoryRequirements req; vkGetImageMemoryRequirements(V.dev, g_stage.img, &req);
    VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL, req.size, vk_mem_type(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)};
    if ((r = vkAllocateMemory(V.dev, &ai, NULL, &g_stage.mem)) || (r = vkBindImageMemory(V.dev, g_stage.img, g_stage.mem, 0))) { VKRED_ONCE(320, "zero-copy movie: the staging image's memory failed (%d)", r); vkDestroyImage(V.dev, g_stage.img, NULL); g_stage.img = VK_NULL_HANDLE; return 0; }
    VkImageViewCreateInfo vi = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, NULL, 0, g_stage.img, VK_IMAGE_VIEW_TYPE_2D, VK_FORMAT_R8G8B8A8_UNORM, {0, 0, 0, 0}, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
    if ((r = vkCreateImageView(V.dev, &vi, NULL, &g_stage.view))) { VKRED_ONCE(321, "zero-copy movie: the staging image's view failed (%d)", r); vkDestroyImage(V.dev, g_stage.img, NULL); vkFreeMemory(V.dev, g_stage.mem, NULL); g_stage.img = VK_NULL_HANDLE; return 0; }
    g_stage.w = w; g_stage.h = h;
    VkImageMemoryBarrier b = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, NULL, 0, VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
        VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, g_stage.img, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
    vkCmdPipelineBarrier(vk_upload_cmd(), VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, NULL, 0, NULL, 1, &b);
    return 1;
}
// The game's texture over a slot: the slot's frame is converted into its image when the slot changed since the last conversion into this image (im->zc_gen)
int zc_tex_refresh(vkimg_t *im, const void *mem, uint64_t len) {
    zc_frame *f; uint64_t gen;
    if (!zc_slot_find(mem, len, &f, &gen)) return 0;
    if (!f) return 1;   // nothing shown yet: the image keeps what it has (zeros)
    if (im->zc_gen == gen) { zc_frame_unref(f); return 1; }
    const vkfmt_t *tf = vk_fmt(im->nvn_format);
    if (im->levels != 1 || im->layers != 1 || im->aspect != VK_IMAGE_ASPECT_COLOR_BIT || !tf || tf->compressed || tf->bpb != 4) {
        VKRED_ONCE(316, "zero-copy movie: the texture over a picture slot (%ux%u nvn format %#x, vk %d) is not a 4-byte colour image: the conversion cannot write it", im->w, im->h, im->nvn_format, im->fmt); zc_frame_unref(f); return 1; }
    int direct = im->fmt == VK_FORMAT_R8G8B8A8_UNORM && (im->usage & VK_IMAGE_USAGE_STORAGE_BIT);
    int swap = im->fmt == VK_FORMAT_B8G8R8A8_UNORM || im->fmt == VK_FORMAT_B8G8R8A8_SRGB;
    zc_imp *m = import_frame(f);
    if (!m) { zc_frame_unref(f); return 1; }
    VkCommandBuffer cb = vk_upload_cmd();
    if (im->layout != VK_IMAGE_LAYOUT_GENERAL) vk_tex_prepare(cb, im);
    vk_img_barrier_all(cb);   // after the reads of earlier command buffers
    if (!direct && !stage_get(im->w, im->h)) { zc_frame_unref(f); return 1; }
    if (g_foreign < 0) { uint32_t ne = 0; vkEnumerateDeviceExtensionProperties(V.pd, NULL, &ne, NULL); VkExtensionProperties *ex = calloc(ne, sizeof *ex); vkEnumerateDeviceExtensionProperties(V.pd, NULL, &ne, ex);
        g_foreign = 0; for (uint32_t i = 0; i < ne; i++) if (!strcmp(ex[i].extensionName, "VK_EXT_queue_family_foreign")) g_foreign = 1; free(ex); }
    // the codec (a foreign queue) wrote the buffer: acquire it, with its contents, for sampling; given back to the foreign queue after the dispatch
    uint32_t ext_q = g_foreign ? VK_QUEUE_FAMILY_FOREIGN_EXT : VK_QUEUE_FAMILY_EXTERNAL;
    VkImageMemoryBarrier acq = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, NULL, 0, VK_ACCESS_SHADER_READ_BIT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL, ext_q, V.qf, m->img, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, NULL, 0, NULL, 1, &acq);
    zc_conv *cv = m->cv; VkDescriptorSet set = cv->sets[cv->next++ % ZC_SETS];
    VkImageView dv = g_stage.view;
    if (direct) {
        VkImageViewCreateInfo dvi = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, NULL, 0, im->img, VK_IMAGE_VIEW_TYPE_2D, VK_FORMAT_R8G8B8A8_UNORM, {0, 0, 0, 0}, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
        VkResult r = vkCreateImageView(V.dev, &dvi, NULL, &dv);
        if (r) { VKRED_ONCE(318, "zero-copy movie: the storage view of the game's texture failed (%d)", r); zc_frame_unref(f); return 1; }
        vk_defer_free_view(dv);
    }
    VkDescriptorImageInfo si = {VK_NULL_HANDLE, m->view, VK_IMAGE_LAYOUT_GENERAL}, di = {VK_NULL_HANDLE, dv, VK_IMAGE_LAYOUT_GENERAL};
    VkWriteDescriptorSet ws[2] = {{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, NULL, set, 0, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &si, NULL, NULL},
                                  {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, NULL, set, 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &di, NULL, NULL}};
    vkUpdateDescriptorSets(V.dev, 2, ws, 0, NULL);
    int cx, cy, cw, ch; zc_frame_crop(f, &cx, &cy, &cw, &ch);
    zc_pc pc = {im->w, im->h, (float)cx / m->w, (float)cy / m->h, (float)cw / m->w / im->w, (float)ch / m->h / im->h, swap};   // the crop rectangle of the buffer over the whole texture
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, cv->pipe);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, cv->pl, 0, 1, &set, 0, NULL);
    vkCmdPushConstants(cb, cv->pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof pc, &pc);
    vkCmdDispatch(cb, (im->w + 15) / 16, (im->h + 15) / 16, 1);
    VkImageMemoryBarrier rel = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, NULL, VK_ACCESS_SHADER_READ_BIT, 0, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL, V.qf, ext_q, m->img, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
    VkMemoryBarrier mb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, NULL, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT};
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &mb, 0, NULL, 1, &rel);
    if (!direct) {   // the staging image's bytes into the game's texture
        VkImageCopy rc = {{VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, {0, 0, 0}, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, {0, 0, 0}, {im->w, im->h, 1}};
        vkCmdCopyImage(cb, g_stage.img, VK_IMAGE_LAYOUT_GENERAL, im->img, VK_IMAGE_LAYOUT_GENERAL, 1, &rc);
        VkMemoryBarrier tb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, NULL, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT};
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &tb, 0, NULL, 0, NULL);
    }
    im->zc_gen = gen; im->gpu_w = 1;   // (the GPU holds the picture: nothing uploads the slot's bytes over it)
    zc_frame_used(f, mtl_last_fence() + 1);   // the submission being recorded commits as the next fence
    zc_frame_unref(f);
    static int n; if (n++ < 3) VKLOG("zero-copy movie: picture %ux%u (crop %d,%d %dx%d) converted into the %ux%u texture at %p (vk format %d, %s)", m->w, m->h, cx, cy, cw, ch, im->w, im->h, mem, im->fmt, direct ? "written directly" : "through the staging image");
    return 1;
}
