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

#define MF_SETS 8   // descriptor sets in a ring: a new one when the picture's buffer range changes (the one in flight is left alone)
typedef struct { uint32_t w, h, word0, flags; } mf_pc;   // the shader's push constants (movie_fill.comp)
static struct {
    VkShaderModule mod; VkDescriptorSetLayout dsl; VkPipelineLayout pl; VkPipeline pipe; VkDescriptorPool dp; VkDescriptorSet sets[MF_SETS]; uint32_t next; int failed;
    VkImage img; VkDeviceMemory mem; VkImageView view; uint32_t w, h;   // the intermediate RGBA8 image, per picture size
    VkBuffer set_buf; VkDeviceSize set_off, set_range; VkImageView set_view; int set_ok;   // what the current descriptor set names
    uint64_t gen;   // the picture generation converted into the intermediate image (0: none)
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
    VkPushConstantRange pr = {VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(mf_pc)};
    VkPipelineLayoutCreateInfo pl = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, NULL, 0, 1, &g.dsl, 1, &pr};
    if ((r = vkCreatePipelineLayout(V.dev, &pl, NULL, &g.pl))) { VKRED_ONCE(332, "movie fill: pipeline layout failed (%d)", r); return 0; }
    VkComputePipelineCreateInfo pi = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO, NULL, 0, {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, NULL, 0, VK_SHADER_STAGE_COMPUTE_BIT, g.mod, "main", NULL}, g.pl, VK_NULL_HANDLE, -1};
    if ((r = vkCreateComputePipelines(V.dev, V.pcache, 1, &pi, NULL, &g.pipe))) { VKRED_ONCE(333, "movie fill: compute pipeline failed (%d)", r); g.pipe = VK_NULL_HANDLE; return 0; }
    VkDescriptorPoolSize ps[2] = {{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, MF_SETS}, {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, MF_SETS}};
    VkDescriptorPoolCreateInfo dpi = {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, NULL, 0, MF_SETS, 2, ps};
    if ((r = vkCreateDescriptorPool(V.dev, &dpi, NULL, &g.dp))) { VKRED_ONCE(334, "movie fill: descriptor pool failed (%d)", r); return 0; }
    VkDescriptorSetLayout ls[MF_SETS]; for (int i = 0; i < MF_SETS; i++) ls[i] = g.dsl;
    VkDescriptorSetAllocateInfo ai = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, NULL, g.dp, MF_SETS, ls};
    if ((r = vkAllocateDescriptorSets(V.dev, &ai, g.sets))) { VKRED_ONCE(335, "movie fill: descriptor sets failed (%d)", r); return 0; }
    g.failed = 0;
    return 1;
}
// The intermediate image of the picture's size (STORAGE for the conversion, TRANSFER_SRC for the blit), in GENERAL from its creation
static int stage_get(VkCommandBuffer cb, uint32_t w, uint32_t h) {
    if (g.img && g.w == w && g.h == h) return 1;
    if (g.img) { vk_defer_free_view(g.view); vk_defer_free_image(g.img, g.mem); g.img = VK_NULL_HANDLE; g.view = VK_NULL_HANDLE; g.mem = VK_NULL_HANDLE; g.w = g.h = 0; }
    g.gen = 0; g.set_ok = 0;   // (the set named the old view; the picture goes through the conversion again)
    VkImageCreateInfo ci = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, NULL, 0, VK_IMAGE_TYPE_2D, VK_FORMAT_R8G8B8A8_UNORM, {w, h, 1}, 1, 1, VK_SAMPLE_COUNT_1_BIT, VK_IMAGE_TILING_OPTIMAL,
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, VK_SHARING_MODE_EXCLUSIVE, 0, NULL, VK_IMAGE_LAYOUT_UNDEFINED};
    VkResult r = vkCreateImage(V.dev, &ci, NULL, &g.img);
    if (r) { VKRED_ONCE(336, "movie fill: the intermediate image failed (%d)", r); g.img = VK_NULL_HANDLE; return 0; }
    VkMemoryRequirements req; vkGetImageMemoryRequirements(V.dev, g.img, &req);
    VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL, req.size, vk_mem_type(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)};
    if ((r = vkAllocateMemory(V.dev, &ai, NULL, &g.mem)) || (r = vkBindImageMemory(V.dev, g.img, g.mem, 0))) { VKRED_ONCE(337, "movie fill: the intermediate image's memory failed (%d)", r); vkDestroyImage(V.dev, g.img, NULL); g.img = VK_NULL_HANDLE; return 0; }
    VkImageViewCreateInfo vi = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, NULL, 0, g.img, VK_IMAGE_VIEW_TYPE_2D, VK_FORMAT_R8G8B8A8_UNORM, {0, 0, 0, 0}, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
    if ((r = vkCreateImageView(V.dev, &vi, NULL, &g.view))) { VKRED_ONCE(338, "movie fill: the intermediate image's view failed (%d)", r); vkDestroyImage(V.dev, g.img, NULL); vkFreeMemory(V.dev, g.mem, NULL); g.img = VK_NULL_HANDLE; return 0; }
    g.w = w; g.h = h;
    VkImageMemoryBarrier b = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, NULL, 0, VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
        VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, g.img, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, NULL, 0, NULL, 1, &b);
    return 1;
}
int vk_movie_fill(vkimg_t *out) {
    uint64_t addr, gen; uint32_t w, h; int cs;
    if (!out || !V.dev || !dm_codec_last_nv12(&addr, &w, &h, &gen, &cs) || !w || !h) return 0;
    uint64_t len = (uint64_t)w * h * 3 / 2;
    VkBuffer buf; VkDeviceSize off; vk_pool_tag = "movie fill";
    if (!vk_buf_at(addr, len, &buf, &off) || !buf) { VKRED_ONCE(339, "movie fill: the picture at %#llx (%ux%u) is in no GPU-backed pool", (unsigned long long)addr, w, h); return 0; }
    // the output must take the blit (the intermediate R8G8B8A8_UNORM image is a blit source everywhere)
    static VkFormat last_out = VK_FORMAT_UNDEFINED; static int out_ok;
    if (out->fmt != last_out) { VkFormatProperties p; last_out = out->fmt; vkGetPhysicalDeviceFormatProperties(V.pd, last_out, &p); out_ok = (p.optimalTilingFeatures & VK_FORMAT_FEATURE_BLIT_DST_BIT) != 0; }
    if (!out_ok || !(out->usage & VK_IMAGE_USAGE_TRANSFER_DST_BIT)) { VKRED_ONCE(340, "movie fill: the stage's output image (vk format %d, usage %#x) cannot take a blit", out->fmt, out->usage); return 0; }
    if (!pipeline_get()) return 0;
    vk_end_pass_land(); VkCommandBuffer cb = vk_cmd();
    vk_tex_prepare(cb, out); vk_img_barrier_all(cb);   // (the CPU's picture writes and earlier command buffers' reads are behind this)
    if (!stage_get(cb, w, h)) return 0;
    if (g.gen != gen) {   // a new picture: converted into the intermediate image
        // the storage buffer binding is aligned down to the device's offset alignment; the shader adds the rest in words (the picture is 2-byte aligned at worst: word0 counts whole words, the byte remainder is 0 for the 4-aligned slots pool_native_write makes)
        VkDeviceSize align = V.props.limits.minStorageBufferOffsetAlignment ? V.props.limits.minStorageBufferOffsetAlignment : 256, aoff = off & ~(align - 1);
        VkDeviceSize range = (off - aoff + len + 3) & ~(VkDeviceSize)3;
        if ((off - aoff) & 3) { VKRED_ONCE(341, "movie fill: the picture at %#llx is not word-aligned in its pool", (unsigned long long)addr); return 0; }
        if (!g.set_ok || g.set_buf != buf || g.set_off != aoff || g.set_range != range || g.set_view != g.view) {
            VkDescriptorSet set = g.sets[g.next++ % MF_SETS];
            VkDescriptorBufferInfo bi = {buf, aoff, range}; VkDescriptorImageInfo di = {VK_NULL_HANDLE, g.view, VK_IMAGE_LAYOUT_GENERAL};
            VkWriteDescriptorSet ws[2] = {{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, NULL, set, 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, NULL, &bi, NULL},
                                          {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, NULL, set, 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &di, NULL, NULL}};
            vkUpdateDescriptorSets(V.dev, 2, ws, 0, NULL);
            g.set_buf = buf; g.set_off = aoff; g.set_range = range; g.set_view = g.view; g.set_ok = 1;
        }
        VkDescriptorSet set = g.sets[(g.next - 1) % MF_SETS];
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
                      .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, .dstOffsets = {{0, 0, 0}, {(int32_t)out->w, (int32_t)out->h, 1}}};
    vkCmdBlitImage(cb, g.img, VK_IMAGE_LAYOUT_GENERAL, out->img, VK_IMAGE_LAYOUT_GENERAL, 1, &bl, VK_FILTER_LINEAR);
    out->gpu_w = 1; vk_img_barrier_all(cb);
    return 1;
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
