// Encoded-RGB spatial reconstruction for an explicitly audited compute upscaler contract.
// Filter in the original encoded domain, then apply signed-square into an RGBA16F intermediate.
// A 1:1 nearest blit converts that result to the guest RGB output, whose native format may lack storage support.
#include "gfx_vk.h"
#include "upscale_square_vk.h"
#include "upscale_square_comp.spv.h"
#include <stdlib.h>

typedef struct { uint32_t w, h; } square_pc;
typedef struct { VkImage img; VkDeviceMemory mem; VkImageView view; uint32_t w, h; } square_image;
static struct {
    VkShaderModule mod; VkSampler sampler; VkDescriptorSetLayout dsl; VkPipelineLayout pl; VkPipeline pipe; int failed;
    square_image image;
} g_square;

static int square_pipeline(void) {
    if (g_square.pipe) return 1;
    if (g_square.failed) return 0;
    g_square.failed = 1;
    VkResult r;
    VkShaderModuleCreateInfo mi = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, NULL, 0, sizeof upscale_square_comp_spv, upscale_square_comp_spv};
    if ((r = vkCreateShaderModule(V.dev, &mi, NULL, &g_square.mod))) goto fail;
    VkSamplerCreateInfo si = {.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO, .magFilter = VK_FILTER_LINEAR, .minFilter = VK_FILTER_LINEAR,
        .mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST, .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, .maxLod = 0.0f};
    if ((r = vkCreateSampler(V.dev, &si, NULL, &g_square.sampler))) goto fail;
    VkDescriptorSetLayoutBinding b[2] = {
        {0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL},
        {1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL}};
    VkDescriptorSetLayoutCreateInfo li = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, NULL, 0, 2, b};
    if ((r = vkCreateDescriptorSetLayout(V.dev, &li, NULL, &g_square.dsl))) goto fail;
    void vk_dsl_register(VkDescriptorSetLayout l, const VkDescriptorSetLayoutBinding *bindings, uint32_t n);
    vk_dsl_register(g_square.dsl, b, 2);
    VkPushConstantRange pr = {VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(square_pc)};
    VkPipelineLayoutCreateInfo pl = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, NULL, 0, 1, &g_square.dsl, 1, &pr};
    if ((r = vkCreatePipelineLayout(V.dev, &pl, NULL, &g_square.pl))) goto fail;
    VkComputePipelineCreateInfo pi = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO, NULL, 0,
        {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, NULL, 0, VK_SHADER_STAGE_COMPUTE_BIT, g_square.mod, "main", NULL},
        g_square.pl, VK_NULL_HANDLE, -1};
    r = vkCreateComputePipelines(V.dev, V.pcache, 1, &pi, NULL, &g_square.pipe); vk_pipeline_cache_changed();   // (gfx_vk_pipeline_cache.inc: a new entry to save; upstream nx2android #213)
    if (r) { g_square.pipe = VK_NULL_HANDLE; goto fail; }
    g_square.failed = 0; return 1;
fail:
    VKRED_ONCE(342, "signed-square upscaler pipeline unavailable (%d); preserving native shader", r);
    return 0;
}
static int square_image_get(VkCommandBuffer cb, uint32_t w, uint32_t h) {
    square_image *old = &g_square.image;
    if (old->img && old->w == w && old->h == h) return 1;
    square_image fresh = {.w = w, .h = h}; VkResult r;
    VkImageCreateInfo ci = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, NULL, 0, VK_IMAGE_TYPE_2D, VK_FORMAT_R16G16B16A16_SFLOAT,
        {w, h, 1}, 1, 1, VK_SAMPLE_COUNT_1_BIT, VK_IMAGE_TILING_OPTIMAL,
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, VK_SHARING_MODE_EXCLUSIVE, 0, NULL, VK_IMAGE_LAYOUT_UNDEFINED};
    if ((r = vkCreateImage(V.dev, &ci, NULL, &fresh.img))) goto fail;
    VkMemoryRequirements req; vkGetImageMemoryRequirements(V.dev, fresh.img, &req);
    VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL, req.size,
        vk_mem_type(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)};
    if ((r = vkAllocateMemory(V.dev, &ai, NULL, &fresh.mem)) || (r = vkBindImageMemory(V.dev, fresh.img, fresh.mem, 0))) goto fail;
    VkImageViewCreateInfo vi = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, NULL, 0, fresh.img, VK_IMAGE_VIEW_TYPE_2D,
        VK_FORMAT_R16G16B16A16_SFLOAT, {0}, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
    if ((r = vkCreateImageView(V.dev, &vi, NULL, &fresh.view))) goto fail;
    VkImageMemoryBarrier barrier = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, NULL, 0, VK_ACCESS_SHADER_WRITE_BIT,
        VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL, VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
        fresh.img, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, NULL, 0, NULL, 1, &barrier);
    if (old->view) vk_defer_free_view(old->view);
    if (old->img) vk_defer_free_image(old->img, old->mem);
    *old = fresh; return 1;
fail:
    VKRED_ONCE(343, "signed-square upscaler intermediate unavailable (%d); preserving native shader", r);
    if (fresh.view) vkDestroyImageView(V.dev, fresh.view, NULL);
    if (fresh.img) vkDestroyImage(V.dev, fresh.img, NULL);
    if (fresh.mem) vkFreeMemory(V.dev, fresh.mem, NULL);
    return 0;
}
static int square_region(const vkimg_t *im, const VkImageSubresourceLayers *sub, const VkOffset3D offsets[2],
                         uint32_t *w, uint32_t *h) {
    if (!im || !im->img || im->aspect != VK_IMAGE_ASPECT_COLOR_BIT || im->d != 1 || im->samples != 1 ||
        (im->vtype != VK_IMAGE_VIEW_TYPE_2D && im->vtype != VK_IMAGE_VIEW_TYPE_2D_ARRAY) ||
        sub->aspectMask != VK_IMAGE_ASPECT_COLOR_BIT || sub->layerCount != 1 ||
        sub->mipLevel >= im->levels || sub->mipLevel >= 32 || sub->baseArrayLayer >= im->layers) return 0;
    *w = im->w >> sub->mipLevel; if (!*w) *w = 1;
    *h = im->h >> sub->mipLevel; if (!*h) *h = 1;
    return im->w && im->h && *w <= INT32_MAX && *h <= INT32_MAX &&
        offsets[0].x == 0 && offsets[0].y == 0 && offsets[0].z == 0 &&
        offsets[1].x == (int32_t)*w && offsets[1].y == (int32_t)*h && offsets[1].z == 1;
}
int vk_upscale_square(vkimg_t *src, VkImageView source_view, vkimg_t *dst, const VkImageBlit *region) {
    uint32_t iw, ih, ow, oh;
    if (!V.dev || !source_view || !region || !square_region(src, &region->srcSubresource, region->srcOffsets, &iw, &ih) ||
        !square_region(dst, &region->dstSubresource, region->dstOffsets, &ow, &oh) || src->img == dst->img ||
        ow < iw || oh < ih || !(src->usage & VK_IMAGE_USAGE_SAMPLED_BIT) || !(dst->usage & VK_IMAGE_USAGE_TRANSFER_DST_BIT)) return 0;
    // Initial audited output is floating RGB. Alpha is deliberately ignored, and there is no implicit sRGB encoding or UNORM clamp.
    if ((src->fmt != VK_FORMAT_E5B9G9R9_UFLOAT_PACK32 && src->fmt != VK_FORMAT_B10G11R11_UFLOAT_PACK32 &&
         src->fmt != VK_FORMAT_R16G16B16A16_SFLOAT) || dst->fmt != VK_FORMAT_B10G11R11_UFLOAT_PACK32) return 0;
    if ((ow + 15u) / 16u > V.props.limits.maxComputeWorkGroupCount[0] || (oh + 7u) / 8u > V.props.limits.maxComputeWorkGroupCount[1]) return 0;
    static VkFormat last_src, last_dst; static int formats_ok;
    if (last_src != src->fmt || last_dst != dst->fmt) {
        VkFormatProperties a, b, c; last_src = src->fmt; last_dst = dst->fmt;
        vkGetPhysicalDeviceFormatProperties(V.pd, src->fmt, &a);
        vkGetPhysicalDeviceFormatProperties(V.pd, dst->fmt, &b);
        vkGetPhysicalDeviceFormatProperties(V.pd, VK_FORMAT_R16G16B16A16_SFLOAT, &c);
        VkFormatFeatureFlags sample = VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT | VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT;
        VkFormatFeatureFlags intermediate = VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT | VK_FORMAT_FEATURE_BLIT_SRC_BIT;
        formats_ok = (a.optimalTilingFeatures & sample) == sample &&
            (b.optimalTilingFeatures & VK_FORMAT_FEATURE_BLIT_DST_BIT) && (c.optimalTilingFeatures & intermediate) == intermediate;
    }
    if (!formats_ok || !square_pipeline()) return 0;
    vk_end_pass_land(); VkCommandBuffer cb = vk_cmd();
    vk_tex_prepare(cb, src); vk_tex_prepare(cb, dst); vk_img_barrier_all(cb);
    if (!square_image_get(cb, ow, oh)) return 0;
    // Completion-owned descriptor sets: never rewrite a set still referenced by queued dispatches.
    VkDescriptorSet set = vk_alloc_set(g_square.dsl); if (!set) return 0;
    VkDescriptorImageInfo images[2] = {
        {g_square.sampler, source_view, VK_IMAGE_LAYOUT_GENERAL},
        {VK_NULL_HANDLE, g_square.image.view, VK_IMAGE_LAYOUT_GENERAL}};
    VkWriteDescriptorSet ws[2] = {
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, NULL, set, 0, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &images[0], NULL, NULL},
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, NULL, set, 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &images[1], NULL, NULL}};
    vkUpdateDescriptorSets(V.dev, 2, ws, 0, NULL);
    square_pc pc = {ow, oh};
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, g_square.pipe);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, g_square.pl, 0, 1, &set, 0, NULL);
    vkCmdPushConstants(cb, g_square.pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof pc, &pc);
    vkCmdDispatch(cb, (ow + 15) / 16, (oh + 7) / 8, 1);
    VkMemoryBarrier mb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, NULL, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT};
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
    VkImageBlit bl = *region;
    bl.srcSubresource = (VkImageSubresourceLayers){VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    bl.srcOffsets[1] = (VkOffset3D){(int32_t)ow, (int32_t)oh, 1};
    vkCmdBlitImage(cb, g_square.image.img, VK_IMAGE_LAYOUT_GENERAL, dst->img, VK_IMAGE_LAYOUT_GENERAL, 1, &bl, VK_FILTER_NEAREST);
    dst->gpu_w = 1; vk_img_barrier_all(cb); return 1;
}

// Explicit shader color metadata; selecting a key in the ordinary upscaler list alone does not infer its transfer.
static uint64_t g_square_keys[64]; static int g_square_count = -1;
int vk_upscale_square_key(uint64_t key) {
    if (g_square_count < 0) {
        g_square_count = 0; const char *p = getenv("DM_VK_UPSCALER_SQUARE_CS");
        while (p && *p) {
            uint64_t value = 0; int i;
            if (g_square_count == 64) { g_square_count = 0; break; }
            for (i = 0; i < 16; i++) {
                unsigned char c = (unsigned char)p[i]; unsigned digit;
                if (c >= '0' && c <= '9') digit = c - '0';
                else if (c >= 'a' && c <= 'f') digit = c - 'a' + 10;
                else if (c >= 'A' && c <= 'F') digit = c - 'A' + 10;
                else break;
                value = (value << 4) | digit;
            }
            if (i != 16 || (p[16] && p[16] != ',') || (p[16] == ',' && !p[17])) { g_square_count = 0; break; }
            g_square_keys[g_square_count++] = value; p += 16; if (*p == ',') p++;
        }
        if (g_square_count) VKLOG("upscaler network: %d audited signed-square variants (DM_VK_UPSCALER_SQUARE_CS)", g_square_count);
    }
    for (int i = 0; i < g_square_count; i++) if (g_square_keys[i] == key) return 1;
    return 0;
}
