// Movie fill (movie_fill_vk.h): the decoded NV12 picture, converted on the GPU into the storage image a skipped movie post-processing stage would have written.
// Ocarina of Time's NDVE network (22 compute shaders Adreno 830 cannot build: DM_VK_SKIP_CS) takes the decoded picture and ssbo activations and writes the enhanced
// picture into a storage image the final draw samples; with the network out, that image stayed black (the passthrough blit's source is a skipped layer's output). Here
// the picture the CPU decoder wrote into the game's image pool (hle_codec_video.c pool_native_write, tight NV12) is read straight out of the pool's VkBuffer by
// movie_fill.comp into an RGBA8 image of the picture's size, and that is blitted over the stage's output (any colour format, any size: linear). The conversion
// runs once per decoded picture (dm_codec_last_nv12's generation); later dispatches of the same picture only blit
#include "gfx_vk.h"
#include "movie_fill_vk.h"
#include "../host/hle_codec_video.h"
#include <stdlib.h>
#include <string.h>
#include "movie_fill_comp.spv.h"

typedef struct { uint32_t w, h, word0, flags; } mf_pc;   // the shader's push constants (movie_fill.comp)
typedef struct { VkImage img; VkDeviceMemory mem; VkImageView view; uint32_t w, h; int srgb; } mf_image;
static struct {
    VkShaderModule mod; VkDescriptorSetLayout dsl; VkPipelineLayout pl; VkPipeline pipe; int failed;
    mf_image stage, scaled;   // matrix-converted picture and, only for a raw sRGB copy where the stage cannot be an sRGB image (direct_ok), its output-sized UNORM image
    uint64_t gen;             // picture generation converted into stage (0: none)
} g;

static int pipeline_get(void) {
    if (g.pipe) return 1;
    if (g.failed) return 0;
    g.failed = 1;
    VkResult r;
    VkShaderModuleCreateInfo mi = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, NULL, 0, sizeof movie_fill_comp_spv, movie_fill_comp_spv};
    if ((r = vkCreateShaderModule(V.dev, &mi, NULL, &g.mod))) { VKRED_ONCE(330, "movie fill: shader module failed (%d)", r); return 0; }
    VkDescriptorSetLayoutBinding b[2] = {{0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL}, {1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL}};
    VkDescriptorSetLayoutCreateInfo li = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, NULL, 0, 2, b};
    if ((r = vkCreateDescriptorSetLayout(V.dev, &li, NULL, &g.dsl))) { VKRED_ONCE(331, "movie fill: descriptor set layout failed (%d)", r); return 0; }
    void vk_dsl_register(VkDescriptorSetLayout l, const VkDescriptorSetLayoutBinding *bindings, uint32_t n);
    vk_dsl_register(g.dsl, b, 2);   // vk_alloc_set owns every written set until its submission completes
    VkPushConstantRange pr = {VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(mf_pc)};
    VkPipelineLayoutCreateInfo pl = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, NULL, 0, 1, &g.dsl, 1, &pr};
    if ((r = vkCreatePipelineLayout(V.dev, &pl, NULL, &g.pl))) { VKRED_ONCE(332, "movie fill: pipeline layout failed (%d)", r); return 0; }
    VkComputePipelineCreateInfo pi = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO, NULL, 0, {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, NULL, 0, VK_SHADER_STAGE_COMPUTE_BIT, g.mod, "main", NULL}, g.pl, VK_NULL_HANDLE, -1};
    r = vkCreateComputePipelines(V.dev, V.pcache, 1, &pi, NULL, &g.pipe); vk_pipeline_cache_changed();   // (gfx_vk_pipeline_cache.inc: a new entry to save; upstream nx2android #213)
    if (r) { VKRED_ONCE(333, "movie fill: compute pipeline failed (%d)", r); g.pipe = VK_NULL_HANDLE; return 0; }
    g.failed = 0;
    return 1;
}
// Reuse intermediate images on the one graphics queue. Replacement allocations and views are deferred through the
// current submission, so previously recorded conversions/copies keep their images. A failed allocation leaves the old one intact.
// srgb: the image's own format is sRGB and the shader stores through a UNORM view (direct_ok)
static int image_get(VkCommandBuffer cb, mf_image *image, uint32_t w, uint32_t h, int storage, int srgb) {
    if (image->img && image->w == w && image->h == h && image->srgb == srgb) return 1;
    mf_image fresh = {.w = w, .h = h, .srgb = srgb};
    VkImageCreateInfo ci = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, NULL, srgb ? VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT | VK_IMAGE_CREATE_EXTENDED_USAGE_BIT : 0, VK_IMAGE_TYPE_2D, srgb ? VK_FORMAT_R8G8B8A8_SRGB : VK_FORMAT_R8G8B8A8_UNORM, {w, h, 1}, 1, 1, VK_SAMPLE_COUNT_1_BIT, VK_IMAGE_TILING_OPTIMAL,
        VK_IMAGE_USAGE_TRANSFER_SRC_BIT | (storage ? VK_IMAGE_USAGE_STORAGE_BIT : VK_IMAGE_USAGE_TRANSFER_DST_BIT), VK_SHARING_MODE_EXCLUSIVE, 0, NULL, VK_IMAGE_LAYOUT_UNDEFINED};
    VkResult r = vkCreateImage(V.dev, &ci, NULL, &fresh.img);
    if (r) goto fail;
    VkMemoryRequirements req; vkGetImageMemoryRequirements(V.dev, fresh.img, &req);
    VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL, req.size, vk_mem_type(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)};
    if ((r = vkAllocateMemory(V.dev, &ai, NULL, &fresh.mem)) || (r = vkBindImageMemory(V.dev, fresh.img, fresh.mem, 0))) goto fail;
    if (storage) {
        VkImageViewCreateInfo vi = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, NULL, 0, fresh.img, VK_IMAGE_VIEW_TYPE_2D, VK_FORMAT_R8G8B8A8_UNORM, {0}, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
        if ((r = vkCreateImageView(V.dev, &vi, NULL, &fresh.view))) goto fail;
    }
    VkImageMemoryBarrier barrier = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, NULL, 0, VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
        VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, fresh.img, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, NULL, 0, NULL, 1, &barrier);
    if (image->view) vk_defer_free_view(image->view);
    if (image->img) vk_defer_free_image(image->img, image->mem);
    *image = fresh; return 1;
fail:
    VKRED_ONCE(336, "movie fill: intermediate image allocation failed (%d)", r);
    if (fresh.view) vkDestroyImageView(V.dev, fresh.view, NULL);
    if (fresh.img) vkDestroyImage(V.dev, fresh.img, NULL);
    if (fresh.mem) vkFreeMemory(V.dev, fresh.mem, NULL);
    return 0;
}
// A raw sRGB copy without the copy: the stage image is sRGB with a UNORM view for the shader's stores (the same bytes), and one blit from it to the sRGB output decodes,
// filters and encodes again, so the output holds the stored bytes. vkCmdCopyImage from a UNORM image into the sRGB output took Adreno 830's driver 15 s of CPU in 37 s
// of the intro movie, on the replay thread, in pages it allocated and cleared for every copy: the movie ran at 30 fps (Odin 3, 2026-11-10).
static int direct_ok(void) {
    static int ok = -1; if (ok >= 0) return ok;
    VkFormatProperties fp; vkGetPhysicalDeviceFormatProperties(V.pd, VK_FORMAT_R8G8B8A8_SRGB, &fp); VkImageFormatProperties ip;
    VkFormatFeatureFlags need = VK_FORMAT_FEATURE_BLIT_SRC_BIT | VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT | VK_FORMAT_FEATURE_BLIT_DST_BIT;
    ok = !getenv("DM_VK_MOVIE_FILL_COPY") && (fp.optimalTilingFeatures & need) == need &&
         !vkGetPhysicalDeviceImageFormatProperties(V.pd, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_TYPE_2D, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_STORAGE_BIT,
                                                  VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT | VK_IMAGE_CREATE_EXTENDED_USAGE_BIT, &ip);
    VKLOG("movie fill: a raw sRGB output is %s", ok ? "blitted from an sRGB stage image" : "copied from a scaled UNORM image");
    return ok;
}
// Image-store output uses the selected view's encoding. Only the observed compatible RGBA8 pair can be copied raw:
// matrix conversion produces encoded RGB bytes in UNORM; a direct blit into native sRGB would encode them a second time.
static int output_contract(const vkimg_t *out, VkFormat view_format, uint32_t level, uint32_t layer,
                           uint32_t *w, uint32_t *h, int *raw_srgb) {
    if (!out || out->aspect != VK_IMAGE_ASPECT_COLOR_BIT || out->samples != VK_SAMPLE_COUNT_1_BIT || out->d != 1 ||
        (out->vtype != VK_IMAGE_VIEW_TYPE_2D && out->vtype != VK_IMAGE_VIEW_TYPE_2D_ARRAY) ||
        !(out->usage & VK_IMAGE_USAGE_TRANSFER_DST_BIT) || level >= out->levels || level >= 32 || layer >= out->layers) return 0;
    *raw_srgb = view_format == VK_FORMAT_R8G8B8A8_UNORM && out->fmt == VK_FORMAT_R8G8B8A8_SRGB;
    if (view_format != out->fmt && !*raw_srgb) return 0;
    *w = out->w >> level; if (!*w) *w = 1;
    *h = out->h >> level; if (!*h) *h = 1;
    return out->w && out->h && *w <= INT32_MAX && *h <= INT32_MAX;
}
int vk_movie_fill(vkimg_t *out, uint32_t view_format, uint32_t level, uint32_t layer) {
    uint32_t ow, oh; int raw_srgb;
    if (!V.dev || !output_contract(out, (VkFormat)view_format, level, layer, &ow, &oh, &raw_srgb)) return 0;
    uint64_t addr, gen; uint32_t w, h; int cs;
    if (!dm_codec_last_nv12(&addr, &w, &h, &gen, &cs) || !w || !h || (w & 1) || (h & 1) || w > INT32_MAX || h > INT32_MAX) return 0;
    uint64_t len = (uint64_t)w * h * 3 / 2;
    if (len > UINT32_MAX) return 0;   // the conversion shader indexes NV12 bytes with uint32
    int convert = g.gen != gen || !g.stage.img || g.stage.w != w || g.stage.h != h;
    if (convert && ((len + 3) & ~(VkDeviceSize)3) > V.props.limits.maxStorageBufferRange) return 0;
    // Format capability queries only when the actual blit destination changes. A raw copy has no sRGB conversion.
    int direct = raw_srgb && direct_ok(), scaled = raw_srgb && !direct;
    VkFormat blit_format = scaled ? VK_FORMAT_R8G8B8A8_UNORM : out->fmt;
    static VkFormat last_out = VK_FORMAT_UNDEFINED; static int formats_ok;
    if (blit_format != last_out) {
        VkFormatProperties src, dst; last_out = blit_format;
        vkGetPhysicalDeviceFormatProperties(V.pd, VK_FORMAT_R8G8B8A8_UNORM, &src);
        vkGetPhysicalDeviceFormatProperties(V.pd, blit_format, &dst);
        VkFormatFeatureFlags need = VK_FORMAT_FEATURE_BLIT_SRC_BIT | VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT | VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT;
        formats_ok = (src.optimalTilingFeatures & need) == need && (dst.optimalTilingFeatures & VK_FORMAT_FEATURE_BLIT_DST_BIT);
    }
    if (!formats_ok) { VKRED_ONCE(340, "movie fill: the output's effective format %d cannot take the converted picture", view_format); return 0; }
    if (!pipeline_get()) return 0;
    vk_end_pass_land(); VkCommandBuffer cb = vk_cmd();
    vk_tex_prepare(cb, out); vk_img_barrier_all(cb);
    if (g.stage.w != w || g.stage.h != h || g.stage.srgb != direct) g.gen = 0;
    if (!image_get(cb, &g.stage, w, h, 1, direct) || (scaled && !image_get(cb, &g.scaled, ow, oh, 0, 0))) return 0;
    if (g.gen != gen) {
        // A codec image pool is ordinary CPU memory, not necessarily a registered NVN pool. When it has no GPU buffer,
        // snapshot this selected generation into completion-owned scratch once; later stages reuse the converted image.
        VkBuffer buf = VK_NULL_HANDLE; VkDeviceSize off = 0;
        VkDeviceSize align = V.props.limits.minStorageBufferOffsetAlignment ? V.props.limits.minStorageBufferOffsetAlignment : 256;
        VkDeviceSize padded = (len + 3) & ~(VkDeviceSize)3;
        if (padded > V.props.limits.maxStorageBufferRange) return 0;
        vk_pool_tag = "movie fill";
        if (!vk_buf_at(addr, len, &buf, &off) || !buf) {
            uint8_t *pixels = vk_scratch(padded, align, &buf, &off);
            if (!dm_codec_copy_last_nv12(gen, pixels, len)) return 0;   // the decoder advanced/finalized; retry a newly selected generation next dispatch
            if (padded > len) memset(pixels + len, 0, (size_t)(padded - len));
            static int n; if (n++ < 3) VKLOG("movie fill: CPU NV12 generation %llu staged (%llu bytes); the decoder pool has no GPU buffer",
                (unsigned long long)gen, (unsigned long long)len);
        }
        VkDeviceSize aoff = off - off % align, range = (off - aoff + len + 3) & ~(VkDeviceSize)3;
        if (range > V.props.limits.maxStorageBufferRange) return 0;
        if ((off - aoff) & 3) { VKRED_ONCE(341, "movie fill: the picture at %#llx is not word-aligned in its pool", (unsigned long long)addr); return 0; }
        // Fresh descriptor storage for every conversion, reclaimed with the owning submission. A fixed ring can wrap while older dispatches still reference it.
        VkDescriptorSet set = vk_alloc_set(g.dsl);
        VkDescriptorBufferInfo bi = {buf, aoff, range}; VkDescriptorImageInfo di = {VK_NULL_HANDLE, g.stage.view, VK_IMAGE_LAYOUT_GENERAL};
        VkWriteDescriptorSet ws[2] = {{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, NULL, set, 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, NULL, &bi, NULL},
                                      {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, NULL, set, 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &di, NULL, NULL}};
        vkUpdateDescriptorSets(V.dev, 2, ws, 0, NULL);
        mf_pc pc = {w, h, (uint32_t)((off - aoff) / 4), (uint32_t)cs};
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, g.pipe);
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, g.pl, 0, 1, &set, 0, NULL);
        vkCmdPushConstants(cb, g.pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof pc, &pc);
        vkCmdDispatch(cb, (w + 15) / 16, (h + 15) / 16, 1);
        VkMemoryBarrier mb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, NULL, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT};
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
        g.gen = gen;
        static int n; if (n++ < 3) VKLOG("movie fill: picture %ux%u at %#llx (%s, %s range) converted (generation %llu)", w, h, (unsigned long long)addr, cs & DM_CODEC_CS_BT709 ? "BT.709" : "BT.601", cs & DM_CODEC_CS_FULL ? "full" : "limited", (unsigned long long)gen);
    }
    VkImageBlit bl = {.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, .srcOffsets = {{0, 0, 0}, {(int32_t)w, (int32_t)h, 1}},
                      .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, scaled ? 0 : level, scaled ? 0 : layer, 1}, .dstOffsets = {{0, 0, 0}, {(int32_t)ow, (int32_t)oh, 1}}};
    vkCmdBlitImage(cb, g.stage.img, VK_IMAGE_LAYOUT_GENERAL, scaled ? g.scaled.img : out->img, VK_IMAGE_LAYOUT_GENERAL, 1, &bl, VK_FILTER_LINEAR);
    if (scaled) {
        VkMemoryBarrier mb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, NULL, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT};
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
        VkImageCopy cp = {.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
                          .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, level, layer, 1}, .extent = {ow, oh, 1}};
        vkCmdCopyImage(cb, g.scaled.img, VK_IMAGE_LAYOUT_GENERAL, out->img, VK_IMAGE_LAYOUT_GENERAL, 1, &cp);
    }
    out->gpu_w = 1; vk_img_barrier_all(cb); return 1;
}
// ---- DM_VK_MOVIE_FILL_CS ----
static uint64_t g_keys[64]; static uint8_t g_noted[64]; static int g_n = -1;
int vk_movie_fill_kernel(uint64_t key) {
    if (g_n < 0) { g_n = 0; const char *e = getenv("DM_VK_MOVIE_FILL_CS");
        for (const char *q = e; q && *q && g_n < 64; ) { char *end; uint64_t k = strtoull(q, &end, 16); if (end == q) break; g_keys[g_n++] = k; q = *end == ',' ? end + 1 : end; }
        if (g_n) VKLOG("[RED] %d compute shaders' output images take the decoded movie picture instead (DM_VK_MOVIE_FILL_CS)", g_n); }
    for (int i = 0; i < g_n; i++) if (g_keys[i] == key) return 1;
    return 0;
}
int vk_movie_fill_note(uint64_t key) { for (int i = 0; i < g_n; i++) if (g_keys[i] == key) { int first = !g_noted[i]; g_noted[i] = 1; return first; } return 0; }
