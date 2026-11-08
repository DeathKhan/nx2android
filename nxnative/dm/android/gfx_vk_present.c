// Android port, Vulkan backend: presentation (see gfx_vk.h).
// With a window (the app's ANativeWindow, dm_front_window): the game's swap chain texture is scaled into a Vulkan swapchain image (aspect kept, black borders) and
// presented with FIFO (vsync) pacing. Headless (dmhost from adb shell): frames are counted, and DM_VK_SHOT=n1,n2,... writes those presented frames as PPM files to
// DM_VK_SHOT_DIR (default DM_ROOT), which is how a run without a screen is checked.
#include "gfx_vk.h"
#include <android/native_window.h>
#include <jni.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "overlay_layout.h"
#include "font5x7.h"
#include "overlay_font.h"
#include "overlay_vert.spv.h"
#include "overlay_frag.spv.h"
#include "rot_vert.spv.h"
#include "rot_frag.spv.h"

__attribute__((weak)) ANativeWindow *dm_front_window(void) { return NULL; }
__attribute__((weak)) int dm_external_active(void) { return 0; }
static uint64_t g_presents;
// Vulkan present IDs include overlay redraws; only the game's tickets have a time for the NVN frame statistics.
enum { PRES_TIMES = 64 };
static struct { uint64_t ticket, ns; } g_pres_t[PRES_TIMES];
static struct { uint32_t id; uint64_t ticket; } g_pres_ids[PRES_TIMES];
static pthread_mutex_t g_pres_mu = PTHREAD_MUTEX_INITIALIZER;
static uint64_t g_pres_recorded;   // display times recorded so far (present_time_record): none after 240 presents with display timing on, the hand-over time stands in
static void present_time_begin(uint64_t ticket, uint64_t ns) {   // ns: the time to report now (no display times on this device), or 0 for the display's
    pthread_mutex_lock(&g_pres_mu);
    g_pres_t[ticket % PRES_TIMES] = (typeof(g_pres_t[0])){ticket, ns};
    pthread_mutex_unlock(&g_pres_mu);
}
static void present_time_map(uint32_t id, uint64_t ticket) {   // the present thread, under g_present_mu
    g_pres_ids[id % PRES_TIMES] = (typeof(g_pres_ids[0])){id, ticket};
}
static void present_time_record(uint32_t id, uint64_t ns) {   // the present thread, under g_present_mu
    typeof(g_pres_ids[0]) *m = &g_pres_ids[id % PRES_TIMES];
    if (m->id != id || !m->ticket || !ns) return;
    pthread_mutex_lock(&g_pres_mu);
    typeof(g_pres_t[0]) *p = &g_pres_t[m->ticket % PRES_TIMES];
    if (p->ticket == m->ticket) p->ns = ns;
    pthread_mutex_unlock(&g_pres_mu); g_pres_recorded++;
}
static void present_time_reset(void) {   // the old swapchain's pending results cannot describe the new one
    memset(g_pres_ids, 0, sizeof g_pres_ids);
    pthread_mutex_lock(&g_pres_mu); memset(g_pres_t, 0, sizeof g_pres_t); pthread_mutex_unlock(&g_pres_mu);
}
static double present_time_get(uint64_t ticket) {
    pthread_mutex_lock(&g_pres_mu);
    typeof(g_pres_t[0]) *p = &g_pres_t[ticket % PRES_TIMES];
    double t = ticket && p->ticket == ticket ? p->ns / 1e9 : 0;
    pthread_mutex_unlock(&g_pres_mu);
    return t;
}
static VkSurfaceKHR g_surf; static VkSwapchainKHR g_swap; static VkImage g_simg[8]; static uint32_t g_nsimg; static VkExtent2D g_sext; static VkFormat g_sfmt;
// The swapchain's format follows the window texture's sRGB-ness: a blit from an sRGB image into a UNORM one decodes to linear and stores that unencoded, which darkened the whole
// picture (Donkey Kong Country Returns HD presents RGBA8_SRGB, NVN format 0x38; issue #4). The Switch scans the window texture's bytes out as they are, and an SRGB -> SRGB blit
// decodes and encodes them again. UNORM games keep the UNORM swapchain. The overlay drawn after the blit is encoded too in an sRGB swapchain, so it shows a little lighter there
static VkFormat g_want = VK_FORMAT_R8G8B8A8_UNORM;
static VkSemaphore g_acq[8], g_done_sem[8]; static VkFence g_pfence[8]; static VkCommandPool g_ppool; static VkCommandBuffer g_pcb[8]; static uint32_t g_pi;
static ANativeWindow *g_win;
static VkImageView g_sview[8];
// Where the game's picture lies in the window (fractions: x, y, width, height; aspect kept, black borders around it), for the front end's touch points, which hle_touch.c
// takes relative to the picture: normalized by the whole window, a tap in Unpacking's 3040x1904 window (97 px borders) was off by up to 5% of the height
static _Atomic uint32_t g_pic[4] = {0, 0, 1u << 20, 1u << 20};
static void vk_pic_set(float x, float y, float w, float h) { atomic_store(&g_pic[0], (uint32_t)(x * (1 << 20))); atomic_store(&g_pic[1], (uint32_t)(y * (1 << 20))); atomic_store(&g_pic[2], (uint32_t)(w * (1 << 20))); atomic_store(&g_pic[3], (uint32_t)(h * (1 << 20))); }
void dm_picture_rect(float r[4]) { for (int i = 0; i < 4; i++) r[i] = (float)atomic_load(&g_pic[i]) / (1 << 20); }
__attribute__((weak)) int dm_overlay_visible(void) { return 0; }
__attribute__((weak)) float dm_overlay_scale(void) { return 1; }
__attribute__((weak)) void dm_overlay_knob(int stick, float *dx, float *dy) { (void)stick; *dx = *dy = 0; }
__attribute__((weak)) int64_t dm_overlay_pressed(int ctl, uint64_t bit) { (void)ctl; (void)bit; return 0; }
__attribute__((weak)) int dm_menu_state(void) { return -1; }
__attribute__((weak)) int dm_overlay_fps(void) { return 0; }
__attribute__((weak)) int dm_overlay_fps_pos(void) { return 0; }   // front_android.c: the counter's corner (bit 0 right, bit 1 bottom)   // front_android.c: the pause menu's Show FPS
__attribute__((weak)) int dm_save_note(char *out, int cap) { if (cap > 0) out[0] = 0; return OVX_IDLE; }   // front_android.c: the save manager's rows
__attribute__((weak)) int dm_save_rows(ov_srow rows[OVM_ROWS]) { (void)rows; return 0; }
__attribute__((weak)) int dm_overlay_custom(int layout, int ctl, float *x, float *y, float *s) { (void)layout; (void)ctl; (void)x; (void)y; *s = 1; return 0; }
__attribute__((weak)) int dm_overlay_custom_ver(void) { return 0; }
__attribute__((weak)) float dm_overlay_opacity(void) { return 0.6f; }
__attribute__((weak)) int dm_overlay_layout(void) { return 0; }
__attribute__((weak)) int dm_menu_value(int item) { (void)item; return 0; }   // front_android.c: the pause menu's rows' values, colour and header
__attribute__((weak)) unsigned dm_menu_accent(void) { return OVM_ACCENT_DEFAULT; }
__attribute__((weak)) const char *dm_menu_choice_name(int item, int v) {
    if (item < 0 || item >= OVM_N) return "";
    return v == OVM_NA ? (OVM_ROW[item].na ? OVM_ROW[item].na : "") : v >= 0 && v < OVM_ROW[item].n ? OVM_ROW[item].choices[v] : "Custom";
}
__attribute__((weak)) int dm_pad_labels(void) { return 1; }
__attribute__((weak)) int dm_game_title(char *out, int cap) { if (cap > 0) out[0] = 0; return 0; }
__attribute__((weak)) int dm_text_entry(char *title, int tcap, char *text, int xcap, int *sel, int *shift) { (void)title; (void)tcap; (void)text; (void)xcap; (void)sel; (void)shift; return 0; }
static void pace_reset(void); static void ov_fb_drop(void);
// Pre-rotation. A landscape window on a panel whose own orientation is portrait (the Retroid Pocket 5: 1080x1920, the window 1920x1080, currentTransform
// ROTATE_90) left the turn to Android's compositor, and its display hardware does not rotate such a layer: SurfaceFlinger composited it on the GPU every
// frame ("CLIENT", "ROT_90" in its layer table) and showed a new frame only every second vsync, 30 fps on screen whatever the game made. With the surface's
// rotation as the swapchain's preTransform the images are in the panel's orientation and the display shows them as they are: the frame is composed as before
// (blit and overlay) into an image in the window's orientation (g_rot.img), and one draw (shaders/rot.*) copies it turned into the swapchain image.
// DM_VK_PRETRANSFORM=0 (at the next swapchain) leaves the turn to the compositor; it needs dynamic rendering, else the compositor turns it as before
static VkSurfaceTransformFlagBitsKHR g_pre = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR;   // the swapchain's preTransform
static VkExtent2D g_next;   // the swapchain images' extent: the window's (g_sext), or the panel's orientation when pre-rotated
static struct { VkImage img; VkDeviceMemory mem; VkImageView view; VkDescriptorSetLayout dsl; VkPipelineLayout layout; VkDescriptorPool pool; VkDescriptorSet set;
                VkSampler samp; VkPipeline pipe; VkFormat pfmt; } g_rot;
static void rot_drop_image(void) {
    if (g_rot.view) vkDestroyImageView(V.dev, g_rot.view, NULL);
    if (g_rot.img) vkDestroyImage(V.dev, g_rot.img, NULL);
    if (g_rot.mem) vkFreeMemory(V.dev, g_rot.mem, NULL);
    g_rot.view = VK_NULL_HANDLE; g_rot.img = VK_NULL_HANDLE; g_rot.mem = VK_NULL_HANDLE;
}
static void swap_destroy(void) {
    pace_reset();
    present_time_reset();
    if (g_swap) { vkDeviceWaitIdle(V.dev); ov_fb_drop(); for (uint32_t i = 0; i < g_nsimg; i++) if (g_sview[i]) { vkDestroyImageView(V.dev, g_sview[i], NULL); g_sview[i] = VK_NULL_HANDLE; }
                  vkDestroySwapchainKHR(V.dev, g_swap, NULL); g_swap = VK_NULL_HANDLE; rot_drop_image(); g_pre = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR; }
    if (g_surf) { vkDestroySurfaceKHR(V.inst, g_surf, NULL); g_surf = VK_NULL_HANDLE; }
}
// The objects of the rotated copy: the pipeline (once per swapchain format) and the image the frame is composed into (the window's size); 0 = not available
static int rot_setup(void) {
    if (!g_rot.dsl) {
        VkDescriptorSetLayoutBinding b = {0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, NULL};
        VkDescriptorSetLayoutCreateInfo li = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, NULL, 0, 1, &b};
        VkPushConstantRange pr = {VK_SHADER_STAGE_VERTEX_BIT, 0, 6 * sizeof(float)};
        VkDescriptorPoolSize ps = {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1}; VkDescriptorPoolCreateInfo pi = {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, NULL, 0, 1, 1, &ps};
        VkSamplerCreateInfo si = {VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO, NULL, 0, VK_FILTER_NEAREST, VK_FILTER_NEAREST, VK_SAMPLER_MIPMAP_MODE_NEAREST,
            VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE};
        VkResult r = vkCreateDescriptorSetLayout(V.dev, &li, NULL, &g_rot.dsl);
        VkPipelineLayoutCreateInfo pl = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, NULL, 0, 1, &g_rot.dsl, 1, &pr};
        if (!r) r = vkCreatePipelineLayout(V.dev, &pl, NULL, &g_rot.layout);
        if (!r) r = vkCreateDescriptorPool(V.dev, &pi, NULL, &g_rot.pool);
        VkDescriptorSetAllocateInfo ai = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, NULL, g_rot.pool, 1, &g_rot.dsl};
        if (!r) r = vkAllocateDescriptorSets(V.dev, &ai, &g_rot.set);
        if (!r) r = vkCreateSampler(V.dev, &si, NULL, &g_rot.samp);
        if (r) { VKRED_ONCE(110, "pre-rotation: descriptor set, layout or sampler: %d; the compositor turns the picture", r); return 0; }
    }
    if (g_rot.pfmt != g_sfmt) {
        if (g_rot.pipe) { vkDestroyPipeline(V.dev, g_rot.pipe, NULL); g_rot.pipe = VK_NULL_HANDLE; }
        VkShaderModule vm = VK_NULL_HANDLE, fm = VK_NULL_HANDLE;
        VkShaderModuleCreateInfo vc = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, NULL, 0, sizeof rot_vert_spv, rot_vert_spv};
        VkShaderModuleCreateInfo fc = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, NULL, 0, sizeof rot_frag_spv, rot_frag_spv};
        VkResult r = vkCreateShaderModule(V.dev, &vc, NULL, &vm); if (!r) r = vkCreateShaderModule(V.dev, &fc, NULL, &fm);
        VkPipelineShaderStageCreateInfo ss[2] = {{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, NULL, 0, VK_SHADER_STAGE_VERTEX_BIT, vm, "main", NULL},
                                                {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, NULL, 0, VK_SHADER_STAGE_FRAGMENT_BIT, fm, "main", NULL}};
        VkPipelineVertexInputStateCreateInfo vi = {VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
        VkPipelineInputAssemblyStateCreateInfo ia = {VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO, NULL, 0, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, 0};
        VkPipelineViewportStateCreateInfo vs = {VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO, NULL, 0, 1, NULL, 1, NULL};
        VkPipelineRasterizationStateCreateInfo rs = {VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO, NULL, 0, 0, 0, VK_POLYGON_MODE_FILL, VK_CULL_MODE_NONE, VK_FRONT_FACE_COUNTER_CLOCKWISE, 0, 0, 0, 0, 1};
        VkPipelineMultisampleStateCreateInfo ms = {VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO, NULL, 0, VK_SAMPLE_COUNT_1_BIT};
        VkPipelineColorBlendAttachmentState ba = {0, 0, 0, 0, 0, 0, 0, 0xf};
        VkPipelineColorBlendStateCreateInfo cbs = {VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO, NULL, 0, 0, 0, 1, &ba};
        VkDynamicState dyn[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR}; VkPipelineDynamicStateCreateInfo dy = {VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO, NULL, 0, 2, dyn};
        VkPipelineRenderingCreateInfo rci = {VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO, NULL, 0, 1, &g_sfmt};
        VkGraphicsPipelineCreateInfo ci = {VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, &rci, 0, 2, ss, &vi, &ia, NULL, &vs, &rs, &ms, NULL, &cbs, &dy, g_rot.layout, VK_NULL_HANDLE};
        if (!r) r = vkCreateGraphicsPipelines(V.dev, V.pcache, 1, &ci, NULL, &g_rot.pipe);
        if (vm) vkDestroyShaderModule(V.dev, vm, NULL); if (fm) vkDestroyShaderModule(V.dev, fm, NULL);
        if (r) { g_rot.pipe = VK_NULL_HANDLE; VKRED_ONCE(111, "pre-rotation: pipeline for format %d: %d; the compositor turns the picture", g_sfmt, r); return 0; }
        g_rot.pfmt = g_sfmt;
    }
    rot_drop_image();
    VkImageCreateInfo ic = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, NULL, 0, VK_IMAGE_TYPE_2D, g_sfmt, {g_sext.width, g_sext.height, 1}, 1, 1, VK_SAMPLE_COUNT_1_BIT, VK_IMAGE_TILING_OPTIMAL,
        VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_SHARING_MODE_EXCLUSIVE, 0, NULL, VK_IMAGE_LAYOUT_UNDEFINED};
    VkResult r = vkCreateImage(V.dev, &ic, NULL, &g_rot.img);
    VkMemoryRequirements rq = {0}; if (!r) vkGetImageMemoryRequirements(V.dev, g_rot.img, &rq);
    VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL, rq.size, vk_mem_type(rq.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0)};
    if (!r) r = vkAllocateMemory(V.dev, &ai, NULL, &g_rot.mem);
    if (!r) r = vkBindImageMemory(V.dev, g_rot.img, g_rot.mem, 0);
    VkImageViewCreateInfo vi = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, NULL, 0, g_rot.img, VK_IMAGE_VIEW_TYPE_2D, g_sfmt, {0}, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
    if (!r) r = vkCreateImageView(V.dev, &vi, NULL, &g_rot.view);
    if (r) { rot_drop_image(); VKRED_ONCE(112, "pre-rotation: the %ux%u image: %d; the compositor turns the picture", g_sext.width, g_sext.height, r); return 0; }
    VkDescriptorImageInfo ii = {g_rot.samp, g_rot.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    VkWriteDescriptorSet wr = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, NULL, g_rot.set, 0, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &ii};
    vkUpdateDescriptorSets(V.dev, 1, &wr, 0, NULL);
    return 1;
}
static int swap_make(void) {
    ANativeWindow *w = dm_front_window();
    if (w != g_win) { swap_destroy(); g_win = w; }
    if (!w) return 0;
    int ww = ANativeWindow_getWidth(w), wh = ANativeWindow_getHeight(w);
    if (g_swap && ww > 0 && wh > 0 && ((uint32_t)ww != g_sext.width || (uint32_t)wh != g_sext.height)) swap_destroy();
    if (g_swap) return 1;
    VkAndroidSurfaceCreateInfoKHR si = {VK_STRUCTURE_TYPE_ANDROID_SURFACE_CREATE_INFO_KHR, NULL, 0, w};
    if (vkCreateAndroidSurfaceKHR(V.inst, &si, NULL, &g_surf)) return 0;
    VkSurfaceCapabilitiesKHR caps; vkGetPhysicalDeviceSurfaceCapabilitiesKHR(V.pd, g_surf, &caps);
    uint32_t nf = 0; vkGetPhysicalDeviceSurfaceFormatsKHR(V.pd, g_surf, &nf, NULL); VkSurfaceFormatKHR sf[32]; if (nf > 32) nf = 32; vkGetPhysicalDeviceSurfaceFormatsKHR(V.pd, g_surf, &nf, sf);
    g_sfmt = sf[0].format; for (uint32_t i = 0; i < nf; i++) if (sf[i].format == VK_FORMAT_R8G8B8A8_UNORM) g_sfmt = sf[i].format;
    for (uint32_t i = 0; i < nf; i++) if (sf[i].format == g_want) g_sfmt = g_want;   // the window texture's sRGB-ness (g_want), so the blit copies its bytes as they are
    if (g_sfmt != g_want) VKRED_ONCE(95, "the surface offers no format %d for the window texture: its picture is %s", g_want, g_want == VK_FORMAT_R8G8B8A8_SRGB ? "darker" : "lighter");
    g_sext = (VkExtent2D){(uint32_t)ANativeWindow_getWidth(w), (uint32_t)ANativeWindow_getHeight(w)};   // the window as it is now; Android's compositor rotates (identity transform)
    if (!g_sext.width || !g_sext.height) g_sext = caps.currentExtent;
    uint32_t n = caps.minImageCount < 3 ? 3 : caps.minImageCount; if (caps.maxImageCount && n > caps.maxImageCount) n = caps.maxImageCount;
    int want = 1; { const char *e = getenv("DM_VK_PRETRANSFORM"); if (e && *e) want = atoi(e) != 0; }
    VkSurfaceTransformFlagBitsKHR turn = (VkSurfaceTransformFlagBitsKHR)(caps.currentTransform & (VK_SURFACE_TRANSFORM_ROTATE_90_BIT_KHR | VK_SURFACE_TRANSFORM_ROTATE_270_BIT_KHR));
    g_pre = want && turn && (caps.supportedTransforms & turn) && V.has_dynrender && rot_setup() ? turn : VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR;
    g_next = g_pre == VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR ? g_sext : (VkExtent2D){g_sext.height, g_sext.width};
    VkSwapchainCreateInfoKHR ci = {VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR, NULL, 0, g_surf, n, g_sfmt, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR, g_next, 1,
        VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, VK_SHARING_MODE_EXCLUSIVE, 0, NULL, g_pre,
        (caps.supportedCompositeAlpha & VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR) ? VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR : VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR, VK_PRESENT_MODE_FIFO_KHR, 1, VK_NULL_HANDLE};
    VkResult r = vkCreateSwapchainKHR(V.dev, &ci, NULL, &g_swap);
    if (r) { VKLOG("[RED] swapchain creation failed: %d", r); swap_destroy(); return 0; }
    g_nsimg = 8; vkGetSwapchainImagesKHR(V.dev, g_swap, &g_nsimg, g_simg);
    for (uint32_t i = 0; i < g_nsimg; i++) { VkImageViewCreateInfo vi = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, NULL, 0, g_simg[i], VK_IMAGE_VIEW_TYPE_2D, g_sfmt, {0}, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
        VKCHK(vkCreateImageView(V.dev, &vi, NULL, &g_sview[i])); }
    if (!g_ppool) {
        VkCommandPoolCreateInfo pci = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, NULL, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, V.qf}; VKCHK(vkCreateCommandPool(V.dev, &pci, NULL, &g_ppool));
        VkCommandBufferAllocateInfo ai = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, NULL, g_ppool, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 8}; VKCHK(vkAllocateCommandBuffers(V.dev, &ai, g_pcb));
        for (int i = 0; i < 8; i++) {
            VkSemaphoreCreateInfo s = {VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO}; VKCHK(vkCreateSemaphore(V.dev, &s, NULL, &g_acq[i])); VKCHK(vkCreateSemaphore(V.dev, &s, NULL, &g_done_sem[i]));
            VkFenceCreateInfo f = {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, NULL, VK_FENCE_CREATE_SIGNALED_BIT}; VKCHK(vkCreateFence(V.dev, &f, NULL, &g_pfence[i]));
        }
    }
    if (g_pre != VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR) VKLOG("swapchain %ux%u format %d, %u images, pre-rotated (transform %#x; the window %ux%u)", g_next.width, g_next.height, g_sfmt, g_nsimg, (unsigned)g_pre, g_sext.width, g_sext.height);
    else VKLOG("swapchain %ux%u format %d, %u images%s", g_sext.width, g_sext.height, g_sfmt, g_nsimg, turn ? " (the compositor turns the picture)" : "");
    return 1;
}
static void barrier(VkCommandBuffer cb, VkImage img, VkImageLayout from, VkImageLayout to) {
    VkImageMemoryBarrier b = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, NULL, VK_ACCESS_MEMORY_WRITE_BIT, VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT, from, to,
        VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, img, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, NULL, 0, NULL, 1, &b);
}
// ---- the on-screen gamepad: rasterized on the CPU once per window size (half resolution), blended over each frame ----
// The texture (tw texels wide: the static layer's or the panel's width) is the static layer (w x h, half the window's size), a strip of sprites under it
// (sh rows) and the pause menu's panel under that (pw x ph, at the window's own resolution, so its letters are sharp). The sprites: each stick's
// knob, drawn every frame where the finger holds it; the menu button, drawn on its own at OV_MENU_ALPHA whether the gamepad is shown or not; a white disc, the
// glow over a pressed round button (a wider one glows in its own shape); a black block, the dimming behind the menu. sp: each sprite in the strip (centre x, y,
// half width and height, texels; 0 = none). The panel is rasterized again whenever the menu's state (dm_menu_state) changes. Under the panel, the frame rate
// counter's characters (OV_FPS_CHARS, gh rows: one cell each), from which the counter is put together on every present, so a changing number needs no upload.
// With a theme in the APK (overlay_theme.c), the gamepad is its sprites instead: an atlas per layout in a texture of its own (g_tt), uploaded once, every
// control drawn from it on each present; the procedural layer is then only the menu's panel, the dimming and the counter.
enum { SP_DISC = OV_N, SP_BLACK, SP_WHITE, SP_ACCENT, SP_N };   // (black, white, accent: solid blocks, the dimming and the layout editor's frames)
#define OV_FPS_CHARS "0123456789 FPS-CBATO"   // (the temperature rows: "SOC 52C", "BAT 38C")
static struct { VkImage img; VkDeviceMemory mem; VkImageView view; VkSampler samp; VkDescriptorSetLayout dsl; VkPipelineLayout layout; VkDescriptorPool pool; VkDescriptorSet set;
                VkPipeline pipe; VkFormat pfmt; uint32_t w, h, sh, pw, ph, gh, gs, tw; unsigned acc; VkBuffer up; VkDeviceMemory upm; uint8_t *px; int uploaded, panel, lay, cver; float sp[SP_N][4];
                VkRenderPass rp; VkFramebuffer fb[8]; VkImageView fbview[8]; VkExtent2D fbext[8]; } g_ov;
static struct { const ov_theme *t; VkImage img; VkDeviceMemory mem; VkImageView view; VkDescriptorSet set; VkBuffer up; VkDeviceMemory upm; uint32_t upk; int ready; } g_tt[2];   // per layout
// Without dynamic rendering (a Vulkan 1.1 / 1.2 device, V.has_dynrender) the overlay draws in a render pass object over the swapchain image: one pass for the swapchain's
// format (load and store, the image already in COLOR_ATTACHMENT_OPTIMAL), and a framebuffer per swapchain image, made again when its view or the extent changes
static VkRenderPass ov_rp(void) {
    static VkFormat fmt; if (g_ov.rp && fmt == g_sfmt) return g_ov.rp;
    if (g_ov.rp) { vkDestroyRenderPass(V.dev, g_ov.rp, NULL); g_ov.rp = VK_NULL_HANDLE; }   // the swapchain's format changed (g_want); swap_destroy dropped its framebuffers
    fmt = g_sfmt;
    VkAttachmentDescription a = {0, g_sfmt, VK_SAMPLE_COUNT_1_BIT, VK_ATTACHMENT_LOAD_OP_LOAD, VK_ATTACHMENT_STORE_OP_STORE, VK_ATTACHMENT_LOAD_OP_DONT_CARE,
                                 VK_ATTACHMENT_STORE_OP_DONT_CARE, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    VkAttachmentReference r = {0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    VkSubpassDescription sp = {0, VK_PIPELINE_BIND_POINT_GRAPHICS, 0, NULL, 1, &r, NULL, NULL, 0, NULL};
    VkRenderPassCreateInfo ci = {VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO, NULL, 0, 1, &a, 1, &sp, 0, NULL};
    if (vkCreateRenderPass(V.dev, &ci, NULL, &g_ov.rp)) { VKRED_ONCE(93, "overlay render pass failed"); g_ov.rp = VK_NULL_HANDLE; }
    return g_ov.rp;
}
// every overlay framebuffer goes with the views or the pass it was made for: a view handle made again later may have the same value (ov_fb would take it for its own)
static void ov_fb_drop(void) { for (int i = 0; i < 8; i++) { if (g_ov.fb[i]) vkDestroyFramebuffer(V.dev, g_ov.fb[i], NULL); g_ov.fb[i] = VK_NULL_HANDLE; g_ov.fbview[i] = VK_NULL_HANDLE; g_ov.fbext[i] = (VkExtent2D){0, 0}; } }
static VkFramebuffer ov_fb(uint32_t idx) {
    if (idx >= 8 || !ov_rp()) return VK_NULL_HANDLE;
    if (g_ov.fb[idx] && g_ov.fbview[idx] == g_sview[idx] && g_ov.fbext[idx].width == g_sext.width && g_ov.fbext[idx].height == g_sext.height) return g_ov.fb[idx];
    if (g_ov.fb[idx]) vkDestroyFramebuffer(V.dev, g_ov.fb[idx], NULL);   // (swap_destroy drops them all with the views; here only an extent change)
    VkFramebufferCreateInfo ci = {VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO, NULL, 0, g_ov.rp, 1, &g_sview[idx], g_sext.width, g_sext.height, 1};
    if (vkCreateFramebuffer(V.dev, &ci, NULL, &g_ov.fb[idx])) { VKRED_ONCE(94, "overlay framebuffer failed"); g_ov.fb[idx] = VK_NULL_HANDLE; return VK_NULL_HANDLE; }
    g_ov.fbview[idx] = g_sview[idx]; g_ov.fbext[idx] = g_sext;
    return g_ov.fb[idx];
}
static _Atomic uint64_t g_fps_frames;   // the game's frames shown (mtl_present), which the counter counts
static int ov_round(const ov_ctl *c) { return c->w == c->h; }
static uint32_t ov_layout_sprites(uint32_t w, uint32_t h) {   // places the sprites, returns the strip's height
    float x = 4, top = (float)h, mr = 0, disc = 0, b[4];
    for (int i = 0; i < OV_N; i++) { ov_box(&OV_CTLS[i], (float)w, (float)h, b); if (OV_CTLS[i].kind == OV_BUTTON && ov_round(&OV_CTLS[i]) && b[2] > disc) disc = b[2]; }
    for (int i = 0; i < SP_N; i++) {
        const ov_ctl *c = i < OV_N ? &OV_CTLS[i] : NULL; float hw = 0, hh = 0;
        if (!c) hw = hh = i == SP_DISC ? disc : 6;
        else { ov_box(c, (float)w, (float)h, b);
            if (c->kind == OV_STICK_L || c->kind == OV_STICK_R) hw = hh = b[2] * OV_KNOB;
            else if (c->kind == OV_MENU || (c->kind == OV_BUTTON && !ov_round(c))) { hw = b[2]; hh = b[3]; } }   // a wider button's glow is its own shape
        g_ov.sp[i][2] = hw; g_ov.sp[i][3] = hh; if (!hw) continue; g_ov.sp[i][0] = x + hw + 2; x += 2 * hw + 8; if (hh > mr) mr = hh;
    }
    for (int i = 0; i < SP_N; i++) g_ov.sp[i][1] = top + mr + 4;
    return (uint32_t)(2 * mr + 8);
}
static void ov_plot(uint8_t *px, uint32_t w, uint32_t h, int x, int y, uint8_t a, uint8_t v) {
    if (x < 0 || y < 0 || (uint32_t)x >= w || (uint32_t)y >= h) return;
    uint8_t *p = px + 4 * ((size_t)y * g_ov.tw + (size_t)x); if (p[3] >= a) return;   // (rows g_ov.tw texels apart)
    p[0] = p[1] = p[2] = v; p[3] = a;
}
static void ov_circle(uint8_t *px, uint32_t w, uint32_t h, float cx, float cy, float r, float ring) {
    for (int y = (int)(cy - r - 2); y <= (int)(cy + r + 2); y++) for (int x = (int)(cx - r - 2); x <= (int)(cx + r + 2); x++) {
        float d = sqrtf((x - cx) * (x - cx) + (y - cy) * (y - cy));
        if (d <= r && d >= r - ring) ov_plot(px, w, h, x, y, 150, 255);   // the ring
        else if (d < r - ring) ov_plot(px, w, h, x, y, 50, 30);           // a dark, see-through inside
    }
}
static void ov_box_ring(uint8_t *px, uint32_t w, uint32_t h, const ov_ctl *c, const float b[4], float ring) {   // ov_circle for any control's shape (b: ov_box)
    for (int y = (int)(b[1] - b[3] - 2); y <= (int)(b[1] + b[3] + 2); y++) for (int x = (int)(b[0] - b[2] - 2); x <= (int)(b[0] + b[2] + 2); x++) {
        float d = ov_sdf(c, b, x - b[0], y - b[1]);
        if (d <= 0 && d >= -ring) ov_plot(px, w, h, x, y, 150, 255);
        else if (d < -ring) ov_plot(px, w, h, x, y, 50, 30);
    }
}
static void ov_text(uint8_t *px, uint32_t w, uint32_t h, float cx, float cy, const char *s, int scale) {
    int n = (int)strlen(s), x0 = (int)(cx - n * 6 * scale / 2.0f), y0 = (int)(cy - 7 * scale / 2.0f);
    for (int i = 0; i < n; i++) {
        unsigned c = (unsigned char)s[i]; const unsigned char *g = FONT5X7[(c >= 32 && c < 127 ? c : 63) - 32];
        for (int col = 0; col < 5; col++) for (int row = 0; row < 7; row++) if (g[col] >> row & 1)
            for (int dy = 0; dy < scale; dy++) for (int dx = 0; dx < scale; dx++) ov_plot(px, w, h, x0 + i * 6 * scale + col * scale + dx, y0 + row * scale + dy, 220, 255);
    }
}
// colour drawing for the menu button and the panel: straight alpha, "over" blending, shapes antialiased by their distance
typedef struct { uint8_t *px; uint32_t stride, x0, y0, w, h; } ov_canvas;   // a w x h window of rows `stride` texels wide, at (x0, y0)
static inline void px_over(uint8_t *p, const uint8_t c[4]) {   // cv_blend at full coverage, in integers (a shape's inside)
    unsigned a = c[3], da = p[3];
    if (!da || a == 255) { p[0] = c[0]; p[1] = c[1]; p[2] = c[2]; p[3] = (uint8_t)a; return; }
    unsigned w = da * (255 - a), oa = a * 255 + w, inv = (1u << 24) / oa, ca = a * 255;   // (alpha x 255; one division)
    for (int k = 0; k < 3; k++) p[k] = (uint8_t)(((c[k] * ca + p[k] * w) * (uint64_t)inv + (1u << 23)) >> 24);
    p[3] = (uint8_t)((oa + 127) / 255);
}
static void cv_blend(const ov_canvas *cv, int x, int y, const uint8_t c[4], float cov) {
    if (x < 0 || y < 0 || (uint32_t)x >= cv->w || (uint32_t)y >= cv->h || cov <= 0) return;
    uint8_t *p = cv->px + 4 * ((size_t)(cv->y0 + y) * cv->stride + cv->x0 + x);
    if (cov >= 1) { px_over(p, c); return; }
    float a = c[3] / 255.0f * (cov > 1 ? 1 : cov);
    if (!p[3] || a >= 1) { p[0] = c[0]; p[1] = c[1]; p[2] = c[2]; p[3] = (uint8_t)(a * 255 + 0.5f); return; }   // (onto nothing, or opaque: no mixing)
    float da = p[3] / 255.0f, oa = a + da * (1 - a);
    for (int k = 0; k < 3; k++) p[k] = (uint8_t)((c[k] * a + p[k] * da * (1 - a)) / oa + 0.5f);
    p[3] = (uint8_t)(oa * 255 + 0.5f);
}
// a filled rounded rectangle, its colour from c0 at the top to c1 at the bottom; the distance is worked out only near the edges and in the corners
static void cv_rrect2(const ov_canvas *cv, float x0, float y0, float x1, float y1, float rad, const uint8_t c0[4], const uint8_t c1[4]) {
    float cx = (x0 + x1) / 2, cy = (y0 + y1) / 2, hx = (x1 - x0) / 2 - rad, hy = (y1 - y0) / 2 - rad;
    if (hx < 0) { rad += hx; hy -= hx; hx = 0; } if (hy < 0) { rad += hy; hx -= hy; hy = 0; } if (rad < 0) rad = 0;
    int ya = (int)floorf(y0) - 1, yb = (int)ceilf(y1) + 1, xa = (int)floorf(x0) - 1, xb = (int)ceilf(x1) + 1;
    if (ya < 0) ya = 0; if (xa < 0) xa = 0; if (yb >= (int)cv->h) yb = (int)cv->h - 1; if (xb >= (int)cv->w) xb = (int)cv->w - 1;
    for (int y = ya; y <= yb; y++) {
        float f = y1 > y0 ? fminf(fmaxf((y + 0.5f - y0) / (y1 - y0), 0), 1) : 0; uint8_t c[4];
        for (int k = 0; k < 4; k++) c[k] = (uint8_t)(c0[k] + (c1[k] - c0[k]) * f + 0.5f);
        float qy = fabsf(y + 0.5f - cy) - hy;
        int xs = (int)ceilf(cx - hx), xe = (int)floorf(cx + hx - 1);   // the columns well inside (|x + 0.5 - cx| <= hx - 0.5)
        if (qy > -0.5f) { xs = 1; xe = 0; }
        if (xs < xa) xs = xa; if (xe > xb) xe = xb;
        uint8_t *row = cv->px + 4 * ((size_t)(cv->y0 + y) * cv->stride + cv->x0);
        for (int x = xs; x <= xe; x++) px_over(row + 4 * x, c);
        for (int x = xa; x <= xb; x++) {
            if (x == xs && xe >= xs) { x = xe; continue; }
            float qx = fabsf(x + 0.5f - cx) - hx;
            float ox = qx > 0 ? qx : 0, oy = qy > 0 ? qy : 0, d = sqrtf(ox * ox + oy * oy) + fminf(fmaxf(qx, qy), 0) - rad;
            if (d < 0.5f) cv_blend(cv, x, y, c, 0.5f - d);
        }
    }
}
static void cv_rrect(const ov_canvas *cv, float x0, float y0, float x1, float y1, float rad, const uint8_t c[4]) { cv_rrect2(cv, x0, y0, x1, y1, rad, c, c); }
static void cv_disc(const ov_canvas *cv, float cx, float cy, float r, const uint8_t c[4]) { cv_rrect(cv, cx - r, cy - r, cx + r, cy + r, r, c); }
// a rounded rectangle's outline, t texels thick inside its edge
static void cv_rring(const ov_canvas *cv, float x0, float y0, float x1, float y1, float rad, float t, const uint8_t c[4]) {
    float cx = (x0 + x1) / 2, cy = (y0 + y1) / 2, hx = (x1 - x0) / 2 - rad, hy = (y1 - y0) / 2 - rad;
    int xs = (int)ceilf(cx - hx + rad + t + 1), xe = (int)floorf(cx + hx - rad - t - 2);   // the columns well inside, skipped on the rows well inside
    for (int y = (int)floorf(y0) - 1; y <= (int)ceilf(y1) + 1; y++) for (int x = (int)floorf(x0) - 1; x <= (int)ceilf(x1) + 1; x++) {
        float qx = fabsf(x + 0.5f - cx) - hx, qy = fabsf(y + 0.5f - cy) - hy;
        if (qy < -rad - t - 1 && x >= xs && x <= xe) { x = xe; continue; }
        float ox = qx > 0 ? qx : 0, oy = qy > 0 ? qy : 0, d = sqrtf(ox * ox + oy * oy) + fminf(fmaxf(qx, qy), 0) - rad;
        cv_blend(cv, x, y, c, fminf(0.5f - d, 0.5f + d + t));
    }
}
static void cv_box(const ov_canvas *cv, const ov_ctl *c, const float b[4], float inset, const uint8_t col[4]) {   // a control's shape filled (b: ov_box), inset texels in
    for (int y = (int)(b[1] - b[3]) - 1; y <= (int)(b[1] + b[3]) + 1; y++) for (int x = (int)(b[0] - b[2]) - 1; x <= (int)(b[0] + b[2]) + 1; x++)
        cv_blend(cv, x, y, col, 0.5f - (ov_sdf(c, b, x + 0.5f - b[0], y + 0.5f - b[1]) + inset));
}
static void cv_ring(const ov_canvas *cv, float cx, float cy, float r, float t, const uint8_t c[4]) {
    for (int y = (int)(cy - r) - 1; y <= (int)(cy + r) + 1; y++) for (int x = (int)(cx - r) - 1; x <= (int)(cx + r) + 1; x++) {
        float d = sqrtf((x + 0.5f - cx) * (x + 0.5f - cx) + (y + 0.5f - cy) * (y + 0.5f - cy));
        cv_blend(cv, x, y, c, fminf(0.5f - (d - r), 0.5f + (d - (r - t))));
    }
}
static void cv_line(const ov_canvas *cv, float ax, float ay, float bx, float by, float t, const uint8_t c[4]) {   // a round-ended stroke t wide
    float x0 = fminf(ax, bx) - t, x1 = fmaxf(ax, bx) + t, y0 = fminf(ay, by) - t, y1 = fmaxf(ay, by) + t, dx = bx - ax, dy = by - ay, l2 = dx * dx + dy * dy;
    for (int y = (int)floorf(y0); y <= (int)ceilf(y1); y++) for (int x = (int)floorf(x0); x <= (int)ceilf(x1); x++) {
        float px = x + 0.5f - ax, py = y + 0.5f - ay, h = l2 > 0 ? fminf(fmaxf((px * dx + py * dy) / l2, 0), 1) : 0;
        float ex = px - dx * h, ey = py - dy * h; cv_blend(cv, x, y, c, t / 2 + 0.5f - sqrtf(ex * ex + ey * ey));
    }
}
static float cv_text_w(const char *s, int scale) { return (float)strlen(s) * 6 * scale - scale; }
static void cv_text(const ov_canvas *cv, float x, float cy, const char *s, int scale, const uint8_t c[4]) {   // the 5x7 letters; x: the left edge
    int x0 = (int)(x + 0.5f), y0 = (int)(cy - 7 * scale / 2.0f + 0.5f);
    for (int i = 0; s[i]; i++) {
        unsigned ch = (unsigned char)s[i]; const unsigned char *g = FONT5X7[(ch >= 32 && ch < 127 ? ch : 63) - 32];
        for (int col = 0; col < 5; col++) for (int row = 0; row < 7; row++) if (g[col] >> row & 1)
            for (int dy = 0; dy < scale; dy++) for (int dx = 0; dx < scale; dx++) cv_blend(cv, x0 + i * 6 * scale + col * scale + dx, y0 + row * scale + dy, c, 1);
    }
}
// Text: the TrueType font (overlay_font.c) at px pixels to the em, or, without one, the 5x7 letters at the whole scale nearest the same capital height
static int tx_scale(float px) { int s = (int)lroundf(px * 0.72f / 7); return s < 1 ? 1 : s; }
static float tx_w(const char *s, float px, int bold) { return ovf_ready() ? ovf_width(s, px, bold) : cv_text_w(s, tx_scale(px)); }
typedef struct { const ov_canvas *cv; const uint8_t *c; } tx_ctx;
static void tx_put(void *ctx, int x, int y, float cov) { const tx_ctx *t = ctx; cv_blend(t->cv, x, y, t->c, cov); }
static void tx(const ov_canvas *cv, float x, float cy, const char *s, float px, int bold, const uint8_t c[4]) {   // x: the left edge, cy: the capitals' middle
    if (ovf_ready()) { tx_ctx t = {cv, c}; ovf_draw(s, x, cy, px, bold, tx_put, &t); } else cv_text(cv, x, cy, s, tx_scale(px), c);
}
static void tx_c(const ov_canvas *cv, float cx, float cy, const char *s, float px, int bold, const uint8_t c[4]) { tx(cv, cx - tx_w(s, px, bold) / 2, cy, s, px, bold, c); }   // centred
static void tx_r(const ov_canvas *cv, float x1, float cy, const char *s, float px, int bold, const uint8_t c[4]) { tx(cv, x1 - tx_w(s, px, bold), cy, s, px, bold, c); }   // right-aligned
static float tx_fit(const char *s, float px, int bold, float room) {   // px, smaller (to 60%) until s fits in room
    for (float k = 1; k > 0.6f; k -= 0.05f) if (tx_w(s, px * k, bold) <= room) return px * k;
    return px * 0.6f;
}
static void tx_clip(char *s, float px, int bold, float room) {   // s cut short with an ellipsis until it fits
    size_t n = strlen(s); if (tx_w(s, px, bold) <= room) return;
    while (n > 0) { while (n > 0 && ((unsigned char)s[n - 1] & 0xc0) == 0x80) n--; if (n) n--; memcpy(s + n, "...", 4); if (tx_w(s, px, bold) <= room) return; }
}
// The colours. The accent is the player's (front_android.c dm_menu_accent, settings.txt accent=): the selection, the switches, the slider, the open tab,
// Resume, the layout editor's chosen control and the frame rate counter's letters. On a light accent (yellow, white) the words on it are dark
static const uint8_t C_WHITE[4] = {255, 255, 255, 255};
static uint8_t C_ACCENT[4] = {OVM_ACCENT_DEFAULT >> 16 & 255, OVM_ACCENT_DEFAULT >> 8 & 255, OVM_ACCENT_DEFAULT & 255, 255}, C_ON_ACCENT[4] = {255, 255, 255, 255}, C_ACC_SEL[4], C_ACC_PRESS[4];
static unsigned g_acc_rgb = OVM_ACCENT_DEFAULT;
static void ov_accent_update(void) {
    unsigned c = dm_menu_accent() & 0xffffff; g_acc_rgb = c;
    C_ACCENT[0] = c >> 16 & 255; C_ACCENT[1] = c >> 8 & 255; C_ACCENT[2] = c & 255; C_ACCENT[3] = 255;
    float l = (0.2126f * C_ACCENT[0] + 0.7152f * C_ACCENT[1] + 0.0722f * C_ACCENT[2]) / 255;
    static const uint8_t dark[4] = {18, 18, 22, 255}; memcpy(C_ON_ACCENT, l > 0.62f ? dark : C_WHITE, 4);
    memcpy(C_ACC_SEL, C_ACCENT, 3); C_ACC_SEL[3] = 44; memcpy(C_ACC_PRESS, C_ACCENT, 3); C_ACC_PRESS[3] = 84;
}
static const uint8_t C_PANEL0[4] = {34, 35, 43, 246}, C_PANEL1[4] = {19, 20, 25, 250}, C_RIM[4] = {255, 255, 255, 30}, C_TEXT[4] = {244, 244, 248, 255},
                     C_SUB[4] = {146, 149, 162, 255}, C_DIM[4] = {186, 188, 199, 255}, C_FILL[4] = {255, 255, 255, 20}, C_FILL2[4] = {255, 255, 255, 34},
                     C_SEP[4] = {255, 255, 255, 16}, C_OFF[4] = {255, 255, 255, 52}, C_KNOB_OFF[4] = {222, 223, 230, 255}, C_SHADOW[4] = {0, 0, 0, 255},
                     C_BADGE[4] = {226, 227, 233, 255}, C_BADGE_TX[4] = {24, 24, 30, 255}, C_RED[4] = {255, 105, 97, 255}, C_HOT[4] = {255, 64, 48, 255};
static void ov_menu_icon(const ov_canvas *cv, const ov_ctl *c, const float b[4]) {   // the menu button: a dark pill with a ring and three bars
    static const uint8_t fill[4] = {20, 20, 26, 170}, ring[4] = {255, 255, 255, 210};
    float r = b[3], t = fmaxf(2, r * 0.07f);
    for (int y = (int)(b[1] - b[3]) - 1; y <= (int)(b[1] + b[3]) + 1; y++) for (int x = (int)(b[0] - b[2]) - 1; x <= (int)(b[0] + b[2]) + 1; x++) {
        float d = ov_sdf(c, b, x + 0.5f - b[0], y + 0.5f - b[1]);
        cv_blend(cv, x, y, fill, 0.5f - d); cv_blend(cv, x, y, ring, fminf(0.5f - d, 0.5f + d + t));
    }
    float bw = r * 0.50f, bt = fmaxf(2, r * 0.11f), gap = r * 0.30f, cx = b[0], cy = b[1];
    for (int k = -1; k <= 1; k++) cv_rrect(cv, cx - bw, cy + k * gap - bt / 2, cx + bw, cy + k * gap + bt / 2, bt / 2, C_WHITE);
}
// The panel's card: a soft shadow in the margin m around it (the raster's edge to the card's), the card's fill (lighter at the top) and a faint rim.
// Returns the card's canvas (the raster less the margin)
static ov_canvas ov_card(const ov_canvas *cv, float m, float rad) {
    // The card with its shadow is the same for every redraw of one size: drawn once into g_bg, copied in after that (one pass over the panel's texels)
    static pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER; static uint8_t *g_bg; static uint32_t bw, bh; static float bm, brad;
    float w = cv->w - 2 * m, h = cv->h - 2 * m, cx = cv->w / 2.0f, cy = cv->h / 2.0f, dy = m * 0.30f;
    pthread_mutex_lock(&mu);
    if (!g_bg || bw != cv->w || bh != cv->h || bm != m || brad != rad) {
        free(g_bg); g_bg = calloc((size_t)cv->w * cv->h, 4); bw = cv->w; bh = cv->h; bm = m; brad = rad;
        ov_canvas bg = {g_bg, cv->w, 0, 0, cv->w, cv->h};
        for (int y = 0; g_bg && y < (int)cv->h; y++) for (int x = 0; x < (int)cv->w; x++) {   // the shadow: only outside the card, falling off over the margin
            float qx = fabsf(x + 0.5f - cx) - (w / 2 - rad), qy = fabsf(y + 0.5f - cy) - (h / 2 - rad);
            float ox = qx > 0 ? qx : 0, oy = qy > 0 ? qy : 0, d = sqrtf(ox * ox + oy * oy) + fminf(fmaxf(qx, qy), 0) - rad;
            if (d < 0) continue;
            float qy2 = fabsf(y + 0.5f - cy - dy) - (h / 2 - rad), oy2 = qy2 > 0 ? qy2 : 0, d2 = sqrtf(ox * ox + oy2 * oy2) + fminf(fmaxf(qx, qy2), 0) - rad;   // offset down
            float f = 1 - fmaxf(d2, 0) / (m * 0.95f); if (f <= 0) continue;
            cv_blend(&bg, x, y, C_SHADOW, 0.42f * f * f);
        }
        ov_canvas in = {g_bg, cv->w, (uint32_t)m, (uint32_t)m, (uint32_t)w, (uint32_t)h};
        if (g_bg) { cv_rrect2(&in, 0, 0, (float)in.w, (float)in.h, rad, C_PANEL0, C_PANEL1); cv_rring(&in, 0, 0, (float)in.w, (float)in.h, rad, 1.2f, C_RIM); }
    }
    for (uint32_t y = 0; g_bg && y < cv->h; y++) memcpy(cv->px + 4 * ((size_t)(cv->y0 + y) * cv->stride + cv->x0), g_bg + 4 * (size_t)y * cv->w, (size_t)cv->w * 4);
    pthread_mutex_unlock(&mu);
    ov_canvas in = {cv->px, cv->stride, cv->x0 + (uint32_t)m, cv->y0 + (uint32_t)m, (uint32_t)w, (uint32_t)h};
    return in;
}
// a controller key's badge: a light disc with the letter (A, B, X, Y), or a rounded tab (L, R, +, -); returns its width
static float ov_badge(const ov_canvas *cv, float x, float cy, const char *k, float px) {
    float h = px * 1.18f, w = strlen(k) > 1 || k[0] == 'L' || k[0] == 'R' || k[0] == 'Z' ? fmaxf(h * 1.25f, tx_w(k, px * 0.72f, 1) + h * 0.55f) : h;
    cv_rrect(cv, x, cy - h / 2, x + w, cy + h / 2, w == h ? h / 2 : h * 0.32f, C_BADGE);
    tx_c(cv, x + w / 2, cy, k, px * 0.72f, 1, C_BADGE_TX);
    return w;
}
// The letters on the buttons that do A and B: Nintendo's, or on a pad labelled the Xbox way (DM_PAD_LABELS unset: A is the right button, labelled B) swapped
static const char *ov_key_a(void) { return dm_pad_labels() ? "A" : "B"; }
static const char *ov_key_b(void) { return dm_pad_labels() ? "B" : "A"; }
// the key hints' line, centred in r: pairs of a key and what it does ("A", "Select", ...), n of them
static void ov_hints(const ov_canvas *cv, const float r[4], const char *const *h, int n, float px) {
    float gap = px * 0.45f, sep = px * 1.5f, total = 0, cy = r[1] + r[3] / 2;
    for (int pass = 0; pass < 2; pass++) {
        float x = pass ? r[0] + (r[2] - total) / 2 : 0;
        for (int i = 0; i < n; i++) {
            const char *keys = h[2 * i], *word = h[2 * i + 1]; char one[4];
            for (const char *k = keys; *k; ) {   // keys: one or more, by spaces ("L R")
                size_t l = strcspn(k, " "); snprintf(one, sizeof one, "%.*s", (int)(l < 3 ? l : 3), k);
                float bw = pass ? ov_badge(cv, x, cy, one, px) : px * 1.18f * (l > 1 || one[0] == 'L' || one[0] == 'R' || one[0] == 'Z' ? 1.25f : 1);
                x += bw + px * 0.22f; k += l; while (*k == ' ') k++;
            }
            x += gap - px * 0.22f;
            if (pass) tx(cv, x, cy, word, px, 0, C_DIM);
            x += tx_w(word, px, 0) + (i < n - 1 ? sep : 0);
        }
        total = x;
    }
}
// a row's selection: an accent-tinted rounded bar with the accent's edge on its left; stronger while a finger presses it
static void ov_select(const ov_canvas *cv, const float r[4], int pressed) {
    float rad = r[3] * 0.22f, e = fmaxf(3, r[3] * 0.045f);
    cv_rrect(cv, r[0], r[1], r[0] + r[2], r[1] + r[3], rad, pressed ? C_ACC_PRESS : C_ACC_SEL);
    cv_rrect(cv, r[0] + e * 0.6f, r[1] + r[3] * 0.22f, r[0] + e * 1.6f, r[1] + r[3] * 0.78f, e / 2, C_ACCENT);
}
// a rounded button with its word: filled with the accent (fill), or a quiet one; held: the accent, ringed white
static void cv_button(const ov_canvas *cv, const float b[4], const char *l, int fill, int held) {
    float rad = fminf(b[3] * 0.5f, b[2] * 0.5f), px = tx_fit(l, b[3] * 0.42f, 1, b[2] * 0.84f);
    cv_rrect(cv, b[0], b[1], b[0] + b[2], b[1] + b[3], rad, held || fill ? C_ACCENT : C_FILL2);
    if (held) cv_rring(cv, b[0], b[1], b[0] + b[2], b[1] + b[3], rad, fmaxf(2, b[3] * 0.05f), C_WHITE);
    tx_c(cv, b[0] + b[2] / 2, b[1] + b[3] / 2, l, px, 1, held || fill ? C_ON_ACCENT : C_TEXT);
}
// a chevron, < (d -1) or > (d 1), s high, centred on (cx, cy)
static void ov_chevron(const ov_canvas *cv, float cx, float cy, float s, int d, const uint8_t c[4]) {
    float w = s * 0.36f, t = fmaxf(2.5f, s * 0.20f);
    cv_line(cv, cx + d * w * -0.5f, cy - s / 2, cx + d * w * 0.5f, cy, t, c); cv_line(cv, cx + d * w * 0.5f, cy, cx + d * w * -0.5f, cy + s / 2, t, c);
}
// the text entry (front_android.c front_text_input), in the pause menu's panel: the game's title for it, the field with the text and a caret, the keyboard
// (overlay_layout.h ov_kb_*: the same rectangles the touches are tested against) with the highlighted key, and the controller's keys
static void ov_text_panel(const ov_canvas *cv) {
    char title[128], text[256]; int sel, shift; if (!dm_text_entry(title, sizeof title, text, sizeof text, &sel, &shift)) return;
    float W = (float)cv->w, H = (float)cv->h, p[4] = {0, 0, W, H}, pad = W * 0.03f;
    float tp = H * 0.045f, xp = H * 0.055f, kp = H * 0.046f, hp = H * 0.030f, room = W - 4 * pad;
    tx_clip(title, tp, 1, room);
    tx_c(cv, W / 2, H * 0.07f, title, tp, 1, C_TEXT);
    cv_rrect(cv, pad, H * 0.14f, W - pad, H * 0.27f, H * 0.03f, C_FILL);
    cv_rring(cv, pad, H * 0.14f, W - pad, H * 0.27f, H * 0.03f, fmaxf(2, H * 0.004f), C_ACCENT);
    char shown[260]; snprintf(shown, sizeof shown, "%s", text);
    const char *s = shown; while (*s && tx_w(s, xp, 0) > room - xp) s++;   // the end of a long text
    tx(cv, 2 * pad, H * 0.205f, s, xp, 0, C_TEXT);
    float cx = 2 * pad + tx_w(s, xp, 0) + xp * 0.08f; cv_rrect(cv, cx, H * 0.205f - xp * 0.45f, cx + fmaxf(2, xp * 0.06f), H * 0.205f + xp * 0.45f, 1, C_ACCENT);   // the caret
    static const char *wide[4] = {"Shift", "Space", "Del", "OK"};
    for (int i = 0; i < OVK_N; i++) {
        float r[4]; ov_kb_key(i, p, r);
        int on = i == OVK_SHIFT && shift, hi = i == sel;
        cv_rrect(cv, r[0], r[1], r[0] + r[2], r[1] + r[3], r[3] * 0.22f, hi ? C_ACCENT : on || i == OVK_OK ? C_FILL2 : C_FILL);
        char one[2] = {(char)ov_kb_char(i, shift), 0}; const char *l = i < 40 ? one : wide[i - 40];
        float lp = tx_fit(l, i < 40 ? kp : kp * 0.8f, i >= 40, r[2] * 0.85f);
        tx_c(cv, r[0] + r[2] / 2, r[1] + r[3] / 2, l, lp, i >= 40, hi ? C_ON_ACCENT : i == OVK_OK || on ? C_ACCENT : C_TEXT);
    }
    const char *const hints[] = {ov_key_a(), "Key", ov_key_b(), "Delete", dm_pad_labels() ? "Y" : "X", "Shift", "+", "OK", "-", "Cancel"};
    float f[4] = {pad, H * 0.91f, W - 2 * pad, H * 0.08f}; ov_hints(cv, f, hints, 5, tx_fit("A Key  B Delete  Y Shift  + OK  - Cancel", hp, 0, W * 0.80f));
}
// The pause menu (overlay_layout.h: the header, the tabs, the rows by their kind, Resume, the key hints), in the card's canvas. Sizes follow the panel's
// height (u: the window's height at the menu's scale), so the letters are as large on every screen relative to the rows
static void ov_menu_panel(const ov_canvas *cv, int st) {
    float p[4] = {0, 0, (float)cv->w, (float)cv->h}, u = cv->h / OV_MENU_H, r[4];
    int tab = st >> 8 & 3, sel = (st >> 4 & 15) - 1, pressed = (st & OVM_PRESSED) != 0;
    if (tab >= OVM_TABS) tab = 0;
    float lp = 0.0315f * u, sp = 0.0215f * u, vp = 0.029f * u;
    // the header: the game's name, and "Paused"
    ov_menu_headrect(p, r);
    char title[128]; if (!dm_game_title(title, sizeof title)) snprintf(title, sizeof title, "Paused");
    float tp = 0.040f * u, pw = tx_w("PAUSED", 0.020f * u, 1) + 0.03f * u;
    tx_clip(title, tp, 1, r[2] - pw - 0.06f * u);
    float hy = r[1] + r[3] * 0.48f;
    cv_disc(cv, r[0] + 0.012f * u, hy, 0.0085f * u, C_ACCENT);
    tx(cv, r[0] + 0.034f * u, hy, title, tp, 1, C_TEXT);
    if (strcmp(title, "Paused")) {
        float bx = r[0] + r[2] - pw, bh = 0.036f * u;
        cv_rrect(cv, bx, hy - bh / 2, r[0] + r[2], hy + bh / 2, bh / 2, C_FILL2);
        tx_c(cv, bx + pw / 2, hy, "PAUSED", 0.020f * u, 1, C_DIM);
    }
    for (int row = 0; row < OVM_ROWS; row++) {
        int it = ov_menu_item(tab, row), k = ov_menu_kind(it); if (it < 0) continue;
        ov_menu_rowrect(p, row, r); float rh = r[3], rw = r[2], cy = r[1] + rh / 2, in = ov_menu_inset(r), x1 = r[0] + rw - in;
        int hi = row == sel;
        if (k == OVR_TABS) {   // a segmented control: the open tab a pill of the accent; the controller's highlight rings it
            float tw = rw / OVM_TABS, rad = rh / 2, g = fmaxf(3, rh * 0.07f);
            cv_rrect(cv, r[0], r[1], r[0] + rw, r[1] + rh, rad, C_FILL);
            if (hi) cv_rring(cv, r[0], r[1], r[0] + rw, r[1] + rh, rad, fmaxf(2, rh * 0.04f), C_ACCENT);
            cv_rrect(cv, r[0] + tab * tw + g, r[1] + g, r[0] + (tab + 1) * tw - g, r[1] + rh - g, rad - g, C_ACCENT);
            for (int t = 0; t < OVM_TABS; t++)
                tx_c(cv, r[0] + (t + 0.5f) * tw, cy, OVM_TAB_NAMES[t], tx_fit(OVM_TAB_NAMES[t], 0.029f * u, 1, tw * 0.86f), 1, t == tab ? C_ON_ACCENT : C_DIM);
            continue;
        }
        if (k == OVR_RESUME) {   // a wide accent button
            float m = rh * 0.10f, b[4] = {r[0], r[1] + m, rw, rh - 2 * m}, rad = b[3] / 2;
            cv_rrect(cv, b[0], b[1], b[0] + b[2], b[1] + b[3], rad, C_ACCENT);
            if (hi) cv_rring(cv, b[0] - 0, b[1], b[0] + b[2], b[1] + b[3], rad, fmaxf(3, rh * 0.045f), C_WHITE);
            if (hi && pressed) { static const uint8_t sh[4] = {0, 0, 0, 50}; cv_rrect(cv, b[0], b[1], b[0] + b[2], b[1] + b[3], rad, sh); }
            float rp = 0.033f * u, ww = tx_w("Resume", rp, 1), ts = rp * 0.62f, x = r[0] + (rw - ww - ts * 1.6f) / 2;
            for (int y = (int)(cy - ts / 2) - 1; y <= (int)(cy + ts / 2) + 1; y++) for (int xx = (int)x - 1; xx <= (int)(x + ts * 0.9f) + 1; xx++) {   // a play triangle
                float fx = (xx + 0.5f - x) / (ts * 0.9f), fy = fabsf(y + 0.5f - cy) / (ts / 2);
                float d = fminf(fminf(xx + 0.5f - x, (1 - fx - fy * 1.0f) * ts * 0.45f), ts);
                cv_blend(cv, xx, y, C_ON_ACCENT, d + 0.5f);
            }
            tx(cv, x + ts * 1.6f, cy, "Resume", rp, 1, C_ON_ACCENT);
            continue;
        }
        if (row > 1 && row != sel && row - 1 != sel) cv_rrect(cv, r[0] + in, r[1] - 0.5f, r[0] + rw - in, r[1] + 0.5f, 0, C_SEP);   // a line between unselected rows
        if (hi) ov_select(cv, r, pressed);
        const ov_mrow *d = &OVM_ROW[it];
        float room = (k == OVR_SLIDER ? r[2] * 0.52f : k == OVR_CHOICE ? r[2] * 0.58f : r[2] * 0.62f) - in * 1.3f;
        float lpx = tx_fit(d->label, lp, 0, room), lx = r[0] + in;
        int v = dm_menu_value(it);
        const char *subt = d->swatches ? (v >= 0 && v < d->n ? d->choices[v] : "Custom") : d->sub;   // a swatch row: the chosen colour's name
        if (subt) {
            char sub[96]; snprintf(sub, sizeof sub, "%s", subt); tx_clip(sub, sp, 0, room);
            tx(cv, lx, r[1] + rh * 0.37f, d->label, lpx, 0, hi ? C_WHITE : C_TEXT);
            tx(cv, lx, r[1] + rh * 0.69f, sub, sp, 0, d->swatches ? C_DIM : C_SUB);
        } else tx(cv, lx, cy, d->label, lpx, 0, hi ? C_WHITE : C_TEXT);
        if (k == OVR_TOGGLE) {   // a switch
            float sh = rh * 0.36f, sw = sh * 1.72f, sx0 = x1 - sw, kr = sh / 2 - fmaxf(2, sh * 0.11f);
            cv_rrect(cv, sx0, cy - sh / 2, x1, cy + sh / 2, sh / 2, v ? C_ACCENT : C_OFF);
            float kx = v ? x1 - sh / 2 : sx0 + sh / 2; static const uint8_t ks[4] = {0, 0, 0, 60};
            cv_disc(cv, kx, cy + kr * 0.10f, kr + 1, ks); cv_disc(cv, kx, cy, kr, v ? C_WHITE : C_KNOB_OFF);
        } else if (k == OVR_SLIDER) {   // the track, filled with the accent to the value, the knob and the percent
            float t[3]; ov_menu_track(p, row, t); float f = (float)(v - OV_OPACITY_MIN) / (OV_OPACITY_MAX - OV_OPACITY_MIN), th = fmaxf(4, rh * 0.075f), kr = rh * 0.15f;
            f = f < 0 ? 0 : f > 1 ? 1 : f;
            cv_rrect(cv, t[0], t[1] - th / 2, t[0] + t[2], t[1] + th / 2, th / 2, C_OFF);
            cv_rrect(cv, t[0], t[1] - th / 2, t[0] + f * t[2], t[1] + th / 2, th / 2, C_ACCENT);
            static const uint8_t ks[4] = {0, 0, 0, 70}; cv_disc(cv, t[0] + f * t[2], t[1] + kr * 0.12f, kr + 1.5f, ks);
            cv_disc(cv, t[0] + f * t[2], t[1], kr, C_WHITE);
            if (hi) cv_ring(cv, t[0] + f * t[2], t[1], kr + fmaxf(3, kr * 0.35f), fmaxf(2, kr * 0.18f), C_ACCENT);
            char s[8]; snprintf(s, sizeof s, "%d%%", v); tx_r(cv, x1, cy, s, vp, 1, C_TEXT);
        } else if (k == OVR_CHOICE && d->swatches) {   // a row of colours, the chosen one larger and ringed
            float c[3]; ov_menu_choice(p, row, c); float cell = c[2] / d->n, sr = fminf(cell * 0.30f, rh * 0.17f);
            for (int i = 0; i < d->n; i++) {
                float sx = ov_menu_swatch_x(p, row, i, d->n); unsigned rgb = d->swatches[i]; uint8_t col[4] = {rgb >> 16 & 255, rgb >> 8 & 255, rgb & 255, 255};
                if (i == v) { cv_ring(cv, sx, c[1], sr * 1.62f, fmaxf(2.5f, sr * 0.22f), C_WHITE); cv_disc(cv, sx, c[1], sr * 1.18f, col); }
                else cv_disc(cv, sx, c[1], sr, col);
            }
        } else if (k == OVR_CHOICE && v == OVM_NA) {   // the game cannot take it: its note, greyed, in place of the value box
            const char *name = dm_menu_choice_name(it, v); float npx = tx_fit(name, vp, 1, r[2] * 0.40f);
            tx_r(cv, x1, cy, name, npx, 1, C_DIM);
        } else if (k == OVR_CHOICE) {   // "< value >": the arrows the accent's when the row is chosen; the colour row shows its swatch
            float c[3]; ov_menu_choice(p, row, c); float bh = rh * 0.56f, ch = bh * 0.40f, cx0 = c[0] + bh * 0.42f, cx1 = c[0] + c[2] - bh * 0.42f;
            cv_rrect(cv, c[0], c[1] - bh / 2, c[0] + c[2], c[1] + bh / 2, bh / 2, hi ? C_FILL2 : C_FILL);
            ov_chevron(cv, cx0, c[1], ch, -1, hi ? C_ACCENT : C_DIM); ov_chevron(cv, cx1, c[1], ch, 1, hi ? C_ACCENT : C_DIM);
            const char *name = dm_menu_choice_name(it, v);
            float vroom = c[2] - bh * 1.6f, sw = it == OVM_ACCENT ? bh * 0.42f : 0, gap = sw ? bh * 0.22f : 0;
            float npx = tx_fit(name, vp, 1, vroom - sw - gap), nw = tx_w(name, npx, 1), nx = c[0] + (c[2] - nw - sw - gap) / 2;
            if (sw) { cv_disc(cv, nx + sw / 2, c[1], sw / 2, C_ACCENT); cv_ring(cv, nx + sw / 2, c[1], sw / 2, 1.2f, C_RIM); }
            tx(cv, nx + sw + gap, c[1], name, npx, 1, C_TEXT);
        } else if (it == OVM_EXPORT && (st >> 20 & 15) != OVX_IDLE) {   // the export's outcome in place of its button: its name, or why there is none
            char nm[64], s[80]; int x = dm_save_note(nm, sizeof nm);
            snprintf(s, sizeof s, "%s", x == OVX_BUSY ? "Exporting..." : x == OVX_DONE ? nm : x == OVX_NOSAVE ? "No save yet" : "Failed");
            float spx = tx_fit(s, vp, 1, r[2] * 0.40f); tx_clip(s, spx, 1, r[2] * 0.40f);
            tx_r(cv, x1, cy, s, spx, 1, x == OVX_FAILED || x == OVX_NOSAVE ? C_RED : x == OVX_DONE ? C_ACCENT : C_DIM);
        } else if (k == OVR_ACTION && d->button) {   // a button
            float bpx = vp * 0.95f, bw = tx_w(d->button, bpx, 1) + rh * 0.62f, b[4] = {x1 - bw, cy - rh * 0.24f, bw, rh * 0.48f};
            cv_button(cv, b, d->button, hi, 0);
        }
    }
    ov_menu_footrect(p, r);
    const char *const h1[] = {ov_key_a(), "Select", ov_key_b(), "Back", "L R", "Tabs", "+", "Resume"};
    ov_hints(cv, r, h1, 4, 0.023f * u);
}
// The layout editor's toolbar and prompt (overlay_layout.h ov_ed_*), at k of the window's scale (1: the Vulkan raster, 0.5: the external display's) in the
// raster's panel rectangle: the toolbar at its top, the prompt under it (gfx_vk_present.c ov_render draws each where overlay_layout.h puts it)
static void ov_ed_rects_for(float W, float H, float pw, float ph, float k, float tb[4], float pr[4]) {
    float a[4], b[4]; ov_ed_rect(W, H, a); ov_ed_prompt_rect(W, H, b);
    tb[0] = 0; tb[1] = 0; tb[2] = fminf(a[2] * k, pw); tb[3] = a[3] * k;
    pr[0] = 0; pr[1] = tb[3] + 2; pr[2] = fminf(b[2] * k, pw); pr[3] = fminf(b[3] * k, ph - pr[1]);
}
static void ov_ed_rects(float tb[4], float pr[4]) { ov_ed_rects_for((float)g_sext.width, (float)g_sext.height, (float)g_ov.pw, (float)g_ov.ph, 1, tb, pr); }
static const char *ov_ctl_name(const ov_ctl *c) {
    return c->kind == OV_STICK_L ? "Left stick" : c->kind == OV_STICK_R ? "Right stick" : c->kind == OV_DPAD ? "D-pad" : c->kind == OV_MENU ? "Menu button"
         : !strcmp(c->label, "-") ? "Minus" : !strcmp(c->label, "+") ? "Plus" : c->label;
}
static void ov_edit_panel(const ov_canvas *cv, int st, float W, float H, float k) {
    float tb[4], pr[4]; ov_ed_rects_for(W, H, (float)cv->w, (float)cv->h, k, tb, pr);
    int held = (st >> 4 & 15) - 1, sel = (st >> 8 & 255) - 1;
    static const char *const words[OVE_ALL] = {"-", "+", "Reset", "Cancel", "Save", "Snap to grid: Off", "Discard", "Keep editing"};
    cv_rrect2(cv, tb[0], tb[1], tb[0] + tb[2], tb[1] + tb[3], tb[3] * 0.12f, C_PANEL0, C_PANEL1);
    cv_rring(cv, tb[0], tb[1], tb[0] + tb[2], tb[1] + tb[3], tb[3] * 0.12f, 1.2f, C_RIM);
    char hint[96];
    if (sel >= 0 && sel < OV_N) snprintf(hint, sizeof hint, "%s: drag, pinch, - +", ov_ctl_name(&OV_CTLS[sel]));
    else snprintf(hint, sizeof hint, "Drag to move, pinch to resize");
    float hp = tx_fit(hint, tb[3] * 0.10f, sel >= 0, tb[2] * 0.94f);
    tx_c(cv, tb[0] + tb[2] / 2, tb[1] + tb[3] * 0.14f, hint, hp, sel >= 0, sel >= 0 ? C_TEXT : C_DIM);
    for (int i = 0; i < OVE_N; i++) { float b[4]; ov_ed_button(i, tb, b); int grid = i == OVE_GRID && (st & OVE_GRID_ON);
        cv_button(cv, b, grid ? "Snap to grid: On" : words[i], i == OVE_SAVE || grid, i == held); }
    if (!(st & 1)) return;
    cv_rrect2(cv, pr[0], pr[1], pr[0] + pr[2], pr[1] + pr[3], pr[3] * 0.10f, C_PANEL0, C_PANEL1);
    cv_rring(cv, pr[0], pr[1], pr[0] + pr[2], pr[1] + pr[3], pr[3] * 0.10f, 1.2f, C_RIM);
    const char *q = "Discard your changes?";
    tx_c(cv, pr[0] + pr[2] / 2, pr[1] + pr[3] * 0.27f, q, tx_fit(q, pr[3] * 0.12f, 1, pr[2] * 0.9f), 1, C_TEXT);
    for (int i = OVE_DISCARD; i <= OVE_KEEP; i++) { float b[4]; ov_ed_button(i, pr, b); cv_button(cv, b, words[i], i == OVE_DISCARD, i == held); }
}
// The save picker (overlay_layout.h OVS_*): the menu's card and rows, as front_android.c fills them (dm_save_rows): the title in the tabs' place, the saves
// as rows (name, size), the notes, the import as an accent button and Back as Resume is drawn, quiet
static void ov_save_panel(const ov_canvas *cv, int st) {
    ov_srow rows[OVM_ROWS]; if (!dm_save_rows(rows)) return;
    float p[4] = {0, 0, (float)cv->w, (float)cv->h}, u = cv->h / OV_MENU_H, r[4];
    int sel = (st >> 4 & 15) - 1;
    ov_menu_headrect(p, r);
    char title[128]; if (!dm_game_title(title, sizeof title)) title[0] = 0;
    float hy = r[1] + r[3] * 0.48f, tp = 0.040f * u;
    cv_disc(cv, r[0] + 0.012f * u, hy, 0.0085f * u, C_ACCENT);
    tx_clip(title, tp, 1, r[2] * 0.9f); tx(cv, r[0] + 0.034f * u, hy, title[0] ? title : "Saves", tp, 1, C_TEXT);
    for (int row = 0; row < OVM_ROWS; row++) {
        const ov_srow *q = &rows[row]; ov_menu_rowrect(p, row, r);
        float rh = r[3], rw = r[2], cy = r[1] + rh / 2, in = ov_menu_inset(r), x0 = r[0] + in, x1 = r[0] + rw - in;
        if (q->style == OVS_EMPTY) continue;
        char left[80], right[40]; snprintf(left, sizeof left, "%s", q->left); snprintf(right, sizeof right, "%s", q->right);
        if (q->style == OVS_TITLE) {
            float tpx = 0.031f * u; tx(cv, r[0] + in * 0.4f, cy, left, tpx, 1, C_TEXT);
            if (right[0]) tx_r(cv, r[0] + rw - in * 0.4f, cy, right, 0.022f * u, 0, C_SUB);
            cv_rrect(cv, r[0], r[1] + rh - 1, r[0] + rw, r[1] + rh, 0, C_SEP);
            continue;
        }
        if (q->style == OVS_BACK || q->style == OVS_BUTTON) {
            float m = rh * 0.12f, b[4], w = q->style == OVS_BACK ? rw : fminf(rw, tx_w(left, 0.03f * u, 1) + rh * 1.6f);
            b[0] = r[0] + (rw - w) / 2; b[1] = r[1] + m; b[2] = w; b[3] = rh - 2 * m;
            cv_button(cv, b, left, q->style == OVS_BUTTON, 0);
            if (row == sel) cv_rring(cv, b[0], b[1], b[0] + b[2], b[1] + b[3], b[3] / 2, fmaxf(3, rh * 0.045f), q->style == OVS_BUTTON ? C_WHITE : C_ACCENT);
            continue;
        }
        if (row == sel) ov_select(cv, r, 0);
        if (q->style == OVS_NOTE) { float npx = tx_fit(left, 0.024f * u, 0, rw - 2 * in); tx(cv, x0, cy, left, npx, 0, C_DIM); continue; }
        float rpx = 0.023f * u, rwid = right[0] ? tx_w(right, rpx, 0) + in : 0, lpx = 0.030f * u;
        tx_clip(left, lpx, 0, x1 - x0 - rwid);
        tx(cv, x0, cy, left, lpx, 0, row == sel ? C_WHITE : C_TEXT);
        if (right[0]) tx_r(cv, x1, cy, right, rpx, 0, C_SUB);
    }
    ov_menu_footrect(p, r);
    const char *const h[] = {ov_key_a(), "Select", ov_key_b(), "Back", "L R", "Page"};
    ov_hints(cv, r, h, 3, 0.023f * u);
}
// A panel for menu state st (dm_menu_state) into a canvas of the menu's raster: the text entry, the layout editor (k: its scale) or, in a card with a
// shadow margin of m texels, the pause menu or the save picker
static void ov_panel_into(const ov_canvas *cv, int st, float W, float H, float k, float m) {
    ov_accent_update();
    if (st & OVE_ST && !(st >> 16 & 1)) { ov_edit_panel(cv, st, W, H, k); return; }
    ov_canvas in = ov_card(cv, m, (cv->h - 2 * m) * 0.045f);
    if (st >> 16 & 1) ov_text_panel(&in);   // a text entry is open (dm_menu_state)
    else if (st & OVS_ST) ov_save_panel(&in, st);
    else ov_menu_panel(&in, st);
}
static float ov_panel_margin(float H) { return roundf(H * 0.035f); }   // the card's shadow margin, texels
static void ov_card_size(float W, float H, uint32_t *w, uint32_t *h) {   // the card's raster with its margin, texels (= the window's pixels)
    float r[4], m = ov_panel_margin(H); ov_menu_rect(W, H, r); *w = (uint32_t)ceilf(r[2]) + 2 * (uint32_t)m; *h = (uint32_t)ceilf(r[3]) + 2 * (uint32_t)m;
}
static void ov_panel(int st) {   // into the panel's rows of the upload buffer
    struct timespec t0, t1; clock_gettime(CLOCK_MONOTONIC, &t0);
    ov_canvas cv = {g_ov.px, g_ov.tw, 0, g_ov.h + g_ov.sh, g_ov.pw, g_ov.ph};
    uint32_t cw = 0, ch = 0; if (!(st & OVE_ST) || (st >> 16 & 1)) ov_card_size((float)g_sext.width, (float)g_sext.height, &cw, &ch);   // (the editor's panels: the whole area)
    for (uint32_t y = 0; y < g_ov.ph; y++) { uint8_t *row = g_ov.px + 4 * ((size_t)(cv.y0 + y) * g_ov.tw);   // cleared where the card does not cover
        if (y >= ch) memset(row, 0, (size_t)g_ov.pw * 4); else if (cw < g_ov.pw) memset(row + 4 * cw, 0, (size_t)(g_ov.pw - cw) * 4); }
    if (cw) { cv.w = cw; cv.h = ch; }
    ov_panel_into(&cv, st, (float)g_sext.width, (float)g_sext.height, 1, ov_panel_margin((float)g_sext.height));
    clock_gettime(CLOCK_MONOTONIC, &t1); double ms = (t1.tv_sec - t0.tv_sec) * 1e3 + (t1.tv_nsec - t0.tv_nsec) / 1e6;
    static int logged; if (logged < 3 || ms > 12) { logged++; VKLOG("[menu] panel %ux%u rasterized in %.1f ms (font: %s)", g_ov.pw, g_ov.ph, ms, ovf_path() ? ovf_path() : "bitmap letters"); }
}
// the panel's redraw key: its state and every row's value (a value that is not in the state bits, as a new choice row's, still redraws it)
static int ov_panel_key(int st) {
    if (st < 0 || (st >> 16 & 1) || (st & (OVE_ST | OVS_ST))) return st;
    uint32_t h = (uint32_t)st * 2654435761u; for (int i = 0; i < OVM_N; i++) h = (h ^ (uint32_t)dm_menu_value(i)) * 16777619u;
    return (int)((h ^ (dm_menu_accent() & 0xffffff)) & 0x7fffffff);
}
// The phone window uses the same menu raster as Vulkan, independently of the external swapchain: at full size with the card's margin (externalMenuRect: where
// it goes), the layout editor's at half size (ExternalActivity.java drawEditor's rectangles)
JNIEXPORT jint JNICALL Java_org_nx2a_ExternalActivity_externalMenuState(JNIEnv *env, jclass cls) { (void)env; (void)cls; return dm_menu_state(); }
JNIEXPORT jint JNICALL Java_org_nx2a_ExternalActivity_externalMenuKey(JNIEnv *env, jclass cls) { (void)env; (void)cls; return ov_panel_key(dm_menu_state()); }
JNIEXPORT jfloatArray JNICALL Java_org_nx2a_ExternalActivity_externalMenuRect(JNIEnv *env, jclass cls, jint W, jint H) {
    (void)cls; if (W <= 0 || H <= 0) return NULL;
    float r[4]; ov_menu_rect((float)W, (float)H, r); float m = ov_panel_margin((float)H);
    jfloat v[4] = {r[0] - m, r[1] - m, ceilf(r[2]) + 2 * m, ceilf(r[3]) + 2 * m};
    jfloatArray out = (*env)->NewFloatArray(env, 4); if (out) (*env)->SetFloatArrayRegion(env, out, 0, 4, v); return out;
}
JNIEXPORT jintArray JNICALL Java_org_nx2a_ExternalActivity_externalMenuPixels(JNIEnv *env, jclass cls, jint W, jint H, jint st) {
    (void)cls;
    if (W <= 0 || H <= 0 || W > 8192 || H > 8192 || st < 0) return NULL;
    float r[4]; ov_menu_rect((float)W, (float)H, r);
    int edit = (st & OVE_ST) && !(st & 0x10000); float m = edit ? 0 : ov_panel_margin((float)H);
    uint32_t w = edit ? (uint32_t)(r[2] / 2) + 1 : (uint32_t)ceilf(r[2]) + 2 * (uint32_t)m, h = edit ? (uint32_t)(r[3] / 2) + 1 : (uint32_t)ceilf(r[3]) + 2 * (uint32_t)m;
    if (edit && w > (uint32_t)W / 2) w = (uint32_t)W / 2;
    if (!w || !h || (size_t)w * h > 16 * 1024 * 1024) return NULL;
    uint8_t *pixels = calloc((size_t)w * h, 4); if (!pixels) return NULL;
    ov_canvas cv = {pixels, w, 0, 0, w, h};
    ov_panel_into(&cv, st, (float)W, (float)H, 0.5f, m);
    size_t count = (size_t)w * h; jint *argb = malloc((count + 2) * sizeof *argb);
    if (!argb) { free(pixels); return NULL; }
    argb[0] = (jint)w; argb[1] = (jint)h;
    for (size_t i = 0; i < count; i++) { uint8_t *p = pixels + i * 4; argb[i + 2] = (jint)((uint32_t)p[3] << 24 | (uint32_t)p[0] << 16 | (uint32_t)p[1] << 8 | p[2]); }
    jintArray out = (*env)->NewIntArray(env, (jsize)(count + 2));
    if (out) (*env)->SetIntArrayRegion(env, out, 0, (jsize)(count + 2), argb);
    free(argb); free(pixels); return out;
}
static void ov_raster(uint8_t *px, uint32_t w, uint32_t h, uint32_t sh) {
    memset(px, 0, (size_t)g_ov.tw * (h + sh) * 4);
    int ts = h > 600 ? 3 : 2;
    for (int i = 0; i < OV_N && g_tt[g_ov.lay].ready != 1; i++) {   // (with a theme the controls are its sprites: the static layer stays empty)
        const ov_ctl *c = &OV_CTLS[i]; float b[4]; ov_box(c, (float)w, (float)h, b);
        if (c->kind == OV_MENU) continue;   // a sprite only
        if (c->kind == OV_DPAD) {
            for (int k = 0; k < 4; k++) { float ox, oy, dr; ov_dpad_dir(k, &ox, &oy, &dr); ov_circle(px, w, h, b[0] + ox * b[2], b[1] + oy * b[2], dr * b[2], 2.5f); }
            continue;
        }
        if (ov_round(c)) ov_circle(px, w, h, b[0], b[1], b[2], c->kind == OV_BUTTON ? 2.5f : 3.0f);
        else ov_box_ring(px, w, h, c, b, 2.5f);
        int ls = ts; while (ls > 1 && 7 * ls > fminf(b[2], b[3])) ls--;   // smaller on a small button (-, +)
        if (c->label[0]) ov_text(px, w, h, b[0], b[1], c->label, ls);
    }
    ov_canvas strip = {px, g_ov.tw, 0, 0, w, h + sh};
    for (int i = 0; i < SP_N; i++) {   // the sprites (the knobs are drawn only as sprites, where the finger has them)
        float x = g_ov.sp[i][0], y = g_ov.sp[i][1], r = g_ov.sp[i][2], b[4] = {x, y, r, g_ov.sp[i][3]}; if (!r) continue;
        if (i == SP_DISC) cv_disc(&strip, x, y, r, C_WHITE);
        else if (i == SP_BLACK || i == SP_WHITE || i == SP_ACCENT) { static const uint8_t k[4] = {0, 0, 0, 255}; cv_rrect(&strip, x - r, y - r, x + r, y + r, 0, i == SP_BLACK ? k : i == SP_WHITE ? C_WHITE : C_ACCENT); }
        else if (OV_CTLS[i].kind == OV_MENU) ov_menu_icon(&strip, &OV_CTLS[i], b);
        else if (OV_CTLS[i].kind == OV_BUTTON) cv_box(&strip, &OV_CTLS[i], b, fminf(b[2], b[3]) * 0.07f, C_WHITE);   // inset as the disc's 0.93
        else ov_circle(px, w, h + sh, x, y, r, 2.5f);
    }
}
typedef struct { float pos[4], uv[4], alpha; } ov_quad;
static void ov_draw(VkCommandBuffer cb, const ov_quad *q) { vkCmdPushConstants(cb, g_ov.layout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof *q, q); vkCmdDraw(cb, 6, 1, 0, 0); }
static void ov_rect(VkCommandBuffer cb, float x0, float y0, float x1, float y1, float u0, float v0, float u1, float v1, float alpha) {   // screen pixels, texels
    float W = (float)g_sext.width, H = (float)g_sext.height, tw = (float)g_ov.tw, th = (float)(g_ov.h + g_ov.sh + g_ov.ph + g_ov.gh);
    ov_quad q = {{x0 / W * 2 - 1, y0 / H * 2 - 1, x1 / W * 2 - 1, y1 / H * 2 - 1}, {u0 / tw, v0 / th, u1 / tw, v1 / th}, alpha}; ov_draw(cb, &q);
}
static void ov_sprite(VkCommandBuffer cb, int i, float sx, float sy, float r, float alpha) {   // sprite i centred on screen pixel (sx, sy), r screen pixels in half width
    float k = r / g_ov.sp[i][2], hx = g_ov.sp[i][2] + 3, hy = g_ov.sp[i][3] + 3;
    ov_rect(cb, sx - hx * k, sy - hy * k, sx + hx * k, sy + hy * k, g_ov.sp[i][0] - hx, g_ov.sp[i][1] - hy, g_ov.sp[i][0] + hx, g_ov.sp[i][1] + hy, alpha);
}
// The theme's sprites (overlay_layout.h ov_tsprite): the design's CSS pixels scaled by the window's height over the design's (and dm_overlay_scale), each at its
// offset from its control's centre. Decorations first, then each control in the table's order: a held button, d-pad direction or stick in its held sprite,
// a stick's knob where the finger has it (as far as the design lets it travel); the menu button last (ov_theme_menu), over everything
static void ov_tsp(VkCommandBuffer cb, const ov_theme *t, const ov_tsprite *s, float cx, float cy, float k, float alpha) {
    float W = (float)g_sext.width, H = (float)g_sext.height, x0 = cx + s->ox * k, y0 = cy + s->oy * k, x1 = x0 + s->w / t->scale * k, y1 = y0 + s->h / t->scale * k;
    ov_quad q = {{x0 / W * 2 - 1, y0 / H * 2 - 1, x1 / W * 2 - 1, y1 / H * 2 - 1}, {s->u / t->w, s->v / t->h, (s->u + s->w) / t->w, (s->v + s->h) / t->h}, alpha}; ov_draw(cb, &q);
}
static float ov_glow(int64_t t, int64_t now) {   // a held button's glow: bright at the press, fading to a steady one
    if (!t) return 0;
    float f = (float)(now - t) / OV_FLASH_NS; return f >= 1 ? 0.22f : 0.22f + 0.48f * (1 - f) * (1 - f);
}
// The frame rate counter: the game's frames shown per second, measured over half a second, in the window's top left corner on a dark box. Its characters
// (OV_FPS_CHARS) are rasterized at the window's resolution into the rows under the panel, one cell each (the digits white, the letters the accent's); the counter is put together from them on every present, so a changing number needs no upload
static float ov_fps_cell(void) { return (ovf_ready() ? 7.0f : 6.0f) * g_ov.gs + 2; }   // a character cell's width in texels
static void ov_fps_glyphs(void) {   // the characters into the rows under the panel
    ov_canvas cv = {g_ov.px, g_ov.tw, 0, g_ov.h + g_ov.sh + g_ov.ph, g_ov.tw, g_ov.gh};
    for (uint32_t y = 0; y < g_ov.gh; y++) memset(g_ov.px + 4 * ((size_t)(cv.y0 + y) * g_ov.tw), 0, (size_t)g_ov.tw * 4);
    float px = ovf_ready() ? 7.0f * g_ov.gs / ovf_cap(1) : 0, cell = ov_fps_cell();
    int nc = (int)strlen(OV_FPS_CHARS);   // the characters, then the same ones again in red after them (a temperature too hot: ov_fps)
    for (int i = 0; i < 2 * nc; i++) {
        char one[2] = {OV_FPS_CHARS[i % nc], 0}; const uint8_t *c = i >= nc ? C_HOT : one[0] >= 'A' && one[0] <= 'Z' ? C_ACCENT : C_WHITE;
        if (px > 0) tx_c(&cv, i * cell + cell / 2, g_ov.gh / 2.0f, one, px, 1, c); else cv_text(&cv, i * cell + 1 + g_ov.gs * 0.5f, g_ov.gh / 2.0f, one, (int)g_ov.gs, c);
    }
}
__attribute__((weak)) void dm_android_temps(int *soc, int *bat) { *soc = *bat = -1000; }   // (front_android.c temp_watch; absent in other builds)
// One row of the counter at y0: its box and its characters, in the corner the player chose (dm_overlay_fps_pos); on the right a box is placed as wide as "999 FPS", so it stays
// put as the number changes (the bottom corners are clear of the gesture bar in the middle, the right ones of the system handle half way down the right edge). hot: the red copies
static void ov_fps_row(VkCommandBuffer cb, const char *s, int hot, float y0) {
    float cw = ov_fps_cell() * (ovf_ready() ? 0.80f : 1), ch = (float)g_ov.gh, m = (float)g_sext.height * 0.02f, r = ch / 2, x = m + r * 0.9f, gy = (float)(g_ov.h + g_ov.sh + g_ov.ph);
    float bx = g_ov.sp[SP_BLACK][0], by = g_ov.sp[SP_BLACK][1], x1 = x + (strlen(s) - (strchr(s, ' ') ? 0.55f : 0)) * cw + r * 0.1f;
    int pos = dm_overlay_fps_pos() & 3; float bw = (7 - 0.55f) * cw + 1.9f * r, dx = pos & 1 ? (float)g_sext.width - 2 * m - bw : 0;
    x += dx; x1 += dx;
    ov_rect(cb, m + dx, y0, x1 + r * 0.9f, y0 + ch, bx - 2, by - 2, bx + 2, by + 2, 0.50f);   // the box: square-cornered, as much room each side
    size_t nc = strlen(OV_FPS_CHARS);
    for (int i = 0; s[i]; x += s[i] == ' ' ? cw * 0.45f : cw, i++) {   // (a narrower space)
        const char *c = strchr(OV_FPS_CHARS, s[i]); if (!c || s[i] == ' ') continue;
        float u = (float)((size_t)(c - OV_FPS_CHARS) + (hot ? nc : 0)) * ov_fps_cell(), o = (ov_fps_cell() - cw) / 2;
        ov_rect(cb, x - o, y0, x - o + ov_fps_cell(), y0 + ch, u, gy, u + ov_fps_cell(), gy + ch, 1);
    }
}
// The counter: the frame rate, then the chip's and the battery's temperature where the device gives them (front_android.c dm_android_temps, upstream nx2android #150), one box
// a row; in a bottom corner the rows grow upwards and the frame rate stays on top. Red from where it is dangerously hot: the chip at 80 C (phones throttle hard in the high 80s),
// the battery at 45 C (Android's warnings and charging limits)
static void ov_fps(VkCommandBuffer cb, int64_t now) {
    static uint64_t n0; static int64_t t0; static int fps = -1;
    uint64_t n = atomic_load(&g_fps_frames);
    if (!t0 || now - t0 > 2000000000ll) { t0 = now; n0 = n; fps = -1; }   // the first frame, or after a pause in the presents (the menu, the background): measure anew
    else if (now - t0 >= 500000000ll) { fps = (int)((double)(n - n0) * 1e9 / (double)(now - t0) + 0.5); t0 = now; n0 = n; }
    char rows[3][16]; int nr = 0, hot[3] = {0};
    if (fps < 0) snprintf(rows[nr++], 16, "-- FPS"); else snprintf(rows[nr++], 16, "%d FPS", fps > 999 ? 999 : fps);
    int soc = -1000, bat = -1000; dm_android_temps(&soc, &bat);   // whole degrees C, -1000 when not known
    if (soc > -100 && soc < 1000) { hot[nr] = soc >= 80; snprintf(rows[nr++], 16, "SOC %dC", soc); }
    if (bat > -100 && bat < 1000) { hot[nr] = bat >= 45; snprintf(rows[nr++], 16, "BAT %dC", bat); }
    float ch = (float)g_ov.gh, m = (float)g_sext.height * 0.02f, gap = ch * 0.15f; int bottom = (dm_overlay_fps_pos() & 2) != 0;
    for (int i = 0; i < nr; i++) ov_fps_row(cb, rows[i], hot[i], bottom ? (float)g_sext.height - m - ch - (float)(nr - 1 - i) * (ch + gap) : m + (float)i * (ch + gap));
}
static void ov_theme_draw(VkCommandBuffer cb, const ov_theme *t, float alpha) {   // (alpha: the player's opacity; each control at its own size factor)
    float W = (float)g_sext.width, H = (float)g_sext.height, k = H / t->dh * dm_overlay_scale(), b[4]; const ov_ctl *L = OV_LAYOUTS[t->layout];
    for (int j = 0; j < t->n; j++) if (t->sp[j].state == OVT_DECOR) { const ov_ctl *c = &L[t->sp[j].ctl]; ov_box(c, W, H, b); ov_tsp(cb, t, &t->sp[j], b[0], b[1], k * ov_cscale(c), alpha); }
    for (int i = 0; i < OV_N; i++) {
        const ov_ctl *c = &L[i]; if (c->kind == OV_MENU) continue;
        ov_box(c, W, H, b); float kc = k * ov_cscale(c); int stick = c->kind == OV_STICK_L || c->kind == OV_STICK_R, held = c->kind != OV_DPAD && dm_overlay_pressed(i, 0) != 0;
        float kx = 0, ky = 0; if (stick) { dm_overlay_knob(c->kind == OV_STICK_R, &kx, &ky); kx *= t->travel * kc; ky *= t->travel * kc; }
        for (int j = 0; j < t->n; j++) {
            const ov_tsprite *s = &t->sp[j]; if (s->ctl != i || s->state == OVT_DECOR) continue;
            if (s->state == OVT_KNOB) { ov_tsp(cb, t, s, b[0] + kx, b[1] + ky, kc, alpha); continue; }
            int on = c->kind == OV_DPAD ? s->dir >= 0 && dm_overlay_pressed(i, OV_DPAD_BITS[s->dir]) != 0 : held;
            if (s->state == (on ? OVT_HELD : OVT_IDLE)) ov_tsp(cb, t, s, b[0], b[1], kc, alpha);
        }
    }
}
static void ov_theme_menu(VkCommandBuffer cb, const ov_theme *t, int i, float alpha) {
    const ov_ctl *c = &OV_LAYOUTS[t->layout][i]; float W = (float)g_sext.width, H = (float)g_sext.height, b[4]; ov_box(c, W, H, b);
    int held = dm_overlay_pressed(i, 0) != 0;
    for (int j = 0; j < t->n; j++) if (t->sp[j].ctl == i && t->sp[j].state == (held ? OVT_HELD : OVT_IDLE)) ov_tsp(cb, t, &t->sp[j], b[0], b[1], H / t->dh * dm_overlay_scale() * ov_cscale(c), alpha);
}
static void ov_block(VkCommandBuffer cb, int sp, float x0, float y0, float x1, float y1, float alpha) {   // a solid rectangle in a block sprite's colour
    float bx = g_ov.sp[sp][0], by = g_ov.sp[sp][1]; ov_rect(cb, x0, y0, x1, y1, bx - 2, by - 2, bx + 2, by + 2, alpha);
}
static void ov_frame(VkCommandBuffer cb, int sp, const float b[4], float t, float alpha) {   // a rectangle's outline around a control's box (b: ov_box), t pixels thick
    float x0 = b[0] - b[2] - t, x1 = b[0] + b[2] + t, y0 = b[1] - b[3] - t, y1 = b[1] + b[3] + t;
    ov_block(cb, sp, x0, y0, x1, y0 + t, alpha); ov_block(cb, sp, x0, y1 - t, x1, y1, alpha); ov_block(cb, sp, x0, y0 + t, x0 + t, y1 - t, alpha); ov_block(cb, sp, x1 - t, y0 + t, x1, y1 - t, alpha);
}
static void ov_render(VkCommandBuffer cb, int mode, int menu, int fps) {
    float W = (float)g_sext.width, H = (float)g_sext.height, op = dm_overlay_opacity();
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); int64_t now = (int64_t)ts.tv_sec * 1000000000 + ts.tv_nsec;
    int edit = menu >= 0 && !(menu >> 16 & 1) && (menu & OVE_ST);   // the layout editor: the gamepad over the held frame, every control (shown or not)
    if (edit) mode = 1;
    float bx = g_ov.sp[SP_BLACK][0], by = g_ov.sp[SP_BLACK][1];
    if (menu >= 0 && !edit) {   // paused: the frame dimmed, the panel, and the menu button (which closes it)
        ov_rect(cb, 0, 0, W, H, bx - 2, by - 2, bx + 2, by + 2, 0.55f);
        float r[4], m = ov_panel_margin(H); uint32_t cw, ch; ov_menu_rect(W, H, r); ov_card_size(W, H, &cw, &ch);   // the card and its shadow's margin, texel for pixel
        ov_rect(cb, r[0] - m, r[1] - m, r[0] - m + cw, r[1] - m + ch, 0, (float)(g_ov.h + g_ov.sh), (float)cw, (float)(g_ov.h + g_ov.sh + ch), 1);
    } else if (edit) {
        ov_rect(cb, 0, 0, W, H, bx - 2, by - 2, bx + 2, by + 2, 0.30f);
        if (menu & OVE_GRID_ON) {
            float step = H / OVE_GRID_ROWS, t = fmaxf(1, H * 0.001f);
            for (int i = 1; i * step < W; i++) { float x = i * step; ov_block(cb, SP_WHITE, x - t / 2, 0, x + t / 2, H, 0.20f); }
            for (int i = 1; i < OVE_GRID_ROWS; i++) { float y = i * step; ov_block(cb, SP_WHITE, 0, y - t / 2, W, y + t / 2, 0.20f); }
        }
    }
    const ov_theme *th = g_tt[g_ov.lay].ready == 1 ? g_tt[g_ov.lay].t : NULL; int tbound = 0;   // (the theme's texture bound)
    if ((menu < 0 || edit) && mode == 1 && th) {
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, g_ov.layout, 0, 1, &g_tt[g_ov.lay].set, 0, NULL); tbound = 1;
        ov_theme_draw(cb, th, op);
    } else if ((menu < 0 || edit) && mode == 1) {   // without a theme: the outlines (their opacity: 60% looks as they always did), the knobs and the glows
        float fo = fminf(1, op / 0.6f);
        ov_rect(cb, 0, 0, W, H, 0, 0, (float)g_ov.w, (float)g_ov.h, fo);
        for (int i = 0; i < OV_N; i++) { const ov_ctl *c = &OV_CTLS[i]; float b[4]; ov_box(c, W, H, b);
            if (c->kind == OV_STICK_L || c->kind == OV_STICK_R) {
                float dx, dy; dm_overlay_knob(c->kind == OV_STICK_R, &dx, &dy); float travel = b[2] * (1 - OV_KNOB);   // the knob's edge reaches the ring
                ov_sprite(cb, i, b[0] + dx * travel, b[1] + dy * travel, b[2] * OV_KNOB, fo);
            } else if (c->kind == OV_BUTTON) { float g = ov_glow(dm_overlay_pressed(i, 0), now); if (g <= 0) continue;
                if (!ov_round(c)) ov_sprite(cb, i, b[0], b[1], b[2], g); else ov_sprite(cb, SP_DISC, b[0], b[1], b[2] * 0.93f, g); }
            else if (c->kind == OV_DPAD) for (int k = 0; k < 4; k++) {
                float g = ov_glow(dm_overlay_pressed(i, OV_DPAD_BITS[k]), now), ox, oy, dr; if (g <= 0) continue;
                ov_dpad_dir(k, &ox, &oy, &dr); ov_sprite(cb, SP_DISC, b[0] + ox * b[2], b[1] + oy * b[2], dr * b[2] * 0.93f, g);
            }
        }
    }
    for (int i = 0; i < OV_N; i++) if (OV_CTLS[i].kind == OV_MENU && mode) {   // the menu button, on top of the controls, where the player put it
        float ma = mode == 1 && (menu < 0 || edit) ? fmaxf(op, 0.5f) : OV_MENU_ALPHA;   // (findable at any opacity)
        if (th) { if (!tbound) { vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, g_ov.layout, 0, 1, &g_tt[g_ov.lay].set, 0, NULL); tbound = 1; }
                  ov_theme_menu(cb, th, i, ma); continue; }
        const ov_ctl *c = &OV_CTLS[i]; float b[4], g = ov_glow(dm_overlay_pressed(i, 0), now); ov_box(c, W, H, b);
        ov_sprite(cb, i, b[0], b[1], b[2], ma);
        if (g > 0) ov_sprite(cb, i, b[0], b[1], b[2], g);
    }
    if (tbound) vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, g_ov.layout, 0, 1, &g_ov.set, 0, NULL);   // the rest is the procedural texture's
    if (edit) {   // every control's box faintly, the chosen one in the accent colour; the toolbar; the prompt over a darker frame
        int sel = (menu >> 8 & 255) - 1; float t = fmaxf(2, H * 0.003f);
        for (int i = 0; i < OV_N; i++) { float b[4]; ov_box(&OV_CTLS[i], W, H, b); if (i == sel) ov_frame(cb, SP_ACCENT, b, 2 * t, 1); else ov_frame(cb, SP_WHITE, b, t, 0.35f); }
        float tb[4], pr[4], a[4], p[4], y0 = (float)(g_ov.h + g_ov.sh); ov_ed_rects(tb, pr); ov_ed_rect(W, H, a); ov_ed_prompt_rect(W, H, p);
        ov_rect(cb, a[0], a[1], a[0] + a[2], a[1] + a[3], tb[0], y0 + tb[1], tb[0] + tb[2], y0 + tb[1] + tb[3], 1);
        if (menu & 1) {
            ov_rect(cb, 0, 0, W, H, bx - 2, by - 2, bx + 2, by + 2, 0.45f);
            ov_rect(cb, p[0], p[1], p[0] + p[2], p[1] + p[3], pr[0], y0 + pr[1], pr[0] + pr[2], y0 + pr[1] + pr[3], 1);
        }
    }
    if (fps && menu < 0) ov_fps(cb, now);
}
// the theme's atlas for a layout into its texture, once (overlay_theme.c reads it at the first present that shows the gamepad in that layout): copied in this
// present's command buffer; its staging buffer goes at a later present, once the copy has run (its slot's fence signalled, or the slot came round again)
static void ov_theme_upload(VkCommandBuffer cb, uint32_t k, int lay) {
    typeof(g_tt[0]) *T = &g_tt[lay];
    if (T->up && (k == T->upk || vkWaitForFences(V.dev, 1, &g_pfence[T->upk], 1, 0) == VK_SUCCESS)) { vkDestroyBuffer(V.dev, T->up, NULL); vkFreeMemory(V.dev, T->upm, NULL); T->up = VK_NULL_HANDLE; }
    if (T->ready) return;
    const ov_theme *t = dm_overlay_theme(lay); T->ready = -1;
    if (!t || !t->px) return;
    VkImageCreateInfo ic = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, NULL, 0, VK_IMAGE_TYPE_2D, VK_FORMAT_R8G8B8A8_UNORM, {(uint32_t)t->w, (uint32_t)t->h, 1}, 1, 1, VK_SAMPLE_COUNT_1_BIT,
        VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, VK_SHARING_MODE_EXCLUSIVE, 0, NULL, VK_IMAGE_LAYOUT_UNDEFINED};
    VKCHK(vkCreateImage(V.dev, &ic, NULL, &T->img)); VkMemoryRequirements rq; vkGetImageMemoryRequirements(V.dev, T->img, &rq);
    VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL, rq.size, vk_mem_type(rq.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0)};
    VKCHK(vkAllocateMemory(V.dev, &ai, NULL, &T->mem)); VKCHK(vkBindImageMemory(V.dev, T->img, T->mem, 0));
    VkImageViewCreateInfo vi = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, NULL, 0, T->img, VK_IMAGE_VIEW_TYPE_2D, VK_FORMAT_R8G8B8A8_UNORM, {0}, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
    VKCHK(vkCreateImageView(V.dev, &vi, NULL, &T->view));
    VkDeviceSize n = (VkDeviceSize)t->w * t->h * 4;
    VkBufferCreateInfo bc = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, NULL, 0, n, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_SHARING_MODE_EXCLUSIVE};
    VKCHK(vkCreateBuffer(V.dev, &bc, NULL, &T->up)); VkMemoryRequirements bq; vkGetBufferMemoryRequirements(V.dev, T->up, &bq);
    VkMemoryAllocateInfo bi = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL, bq.size, vk_mem_type(bq.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 0)};
    VKCHK(vkAllocateMemory(V.dev, &bi, NULL, &T->upm)); VKCHK(vkBindBufferMemory(V.dev, T->up, T->upm, 0));
    void *px; VKCHK(vkMapMemory(V.dev, T->upm, 0, VK_WHOLE_SIZE, 0, &px)); memcpy(px, t->px, (size_t)n);   // (freeing the memory unmaps it)
    dm_overlay_theme_done(t);
    VkDescriptorSetAllocateInfo da = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, NULL, g_ov.pool, 1, &g_ov.dsl};
    if (vkAllocateDescriptorSets(V.dev, &da, &T->set)) { VKRED_ONCE(95, "overlay theme descriptor set failed: the gamepad is drawn without its theme"); return; }
    VkDescriptorImageInfo ii = {g_ov.samp, T->view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    VkWriteDescriptorSet wr = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, NULL, T->set, 0, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &ii};
    vkUpdateDescriptorSets(V.dev, 1, &wr, 0, NULL);
    VkImageMemoryBarrier b = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, NULL, 0, VK_ACCESS_TRANSFER_WRITE_BIT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, T->img, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, NULL, 0, NULL, 1, &b);
    VkBufferImageCopy c = {0, (uint32_t)t->w, 0, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, {0, 0, 0}, {(uint32_t)t->w, (uint32_t)t->h, 1}};
    vkCmdCopyBufferToImage(cb, T->up, T->img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &c);
    b.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT; b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT; b.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL; b.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, NULL, 0, NULL, 1, &b);
    T->t = t; T->upk = k; T->ready = 1;
}
static int overlay_ready(VkCommandBuffer cb, uint32_t k, int menu) {   // k: this present's slot (its fence is not signalled yet)
    uint32_t w = g_sext.width / 2, h = g_sext.height / 2; int lay = dm_overlay_layout();
    if (!g_ov.dsl) {
        VkDescriptorSetLayoutBinding b = {0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, NULL};
        VkDescriptorSetLayoutCreateInfo li = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, NULL, 0, 1, &b}; VKCHK(vkCreateDescriptorSetLayout(V.dev, &li, NULL, &g_ov.dsl));
        VkPushConstantRange pr = {VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(ov_quad)};
        VkPipelineLayoutCreateInfo pl = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, NULL, 0, 1, &g_ov.dsl, 1, &pr}; VKCHK(vkCreatePipelineLayout(V.dev, &pl, NULL, &g_ov.layout));
        VkDescriptorPoolSize ps = {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 3}; VkDescriptorPoolCreateInfo pi = {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, NULL, 0, 3, 1, &ps};   // the procedural texture, a theme atlas per layout
        VKCHK(vkCreateDescriptorPool(V.dev, &pi, NULL, &g_ov.pool));
        VkDescriptorSetAllocateInfo ai = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, NULL, g_ov.pool, 1, &g_ov.dsl}; VKCHK(vkAllocateDescriptorSets(V.dev, &ai, &g_ov.set));
        VkSamplerCreateInfo si = {VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO, NULL, 0, VK_FILTER_LINEAR, VK_FILTER_LINEAR, VK_SAMPLER_MIPMAP_MODE_NEAREST,
            VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE};
        VKCHK(vkCreateSampler(V.dev, &si, NULL, &g_ov.samp));
    }
    if (g_ov.pfmt != g_sfmt) {   // the pipeline renders into the swapchain's format
        VkShaderModule vm, fm;
        VkShaderModuleCreateInfo vc = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, NULL, 0, sizeof overlay_vert_spv, overlay_vert_spv}; VKCHK(vkCreateShaderModule(V.dev, &vc, NULL, &vm));
        VkShaderModuleCreateInfo fc = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, NULL, 0, sizeof overlay_frag_spv, overlay_frag_spv}; VKCHK(vkCreateShaderModule(V.dev, &fc, NULL, &fm));
        VkPipelineShaderStageCreateInfo ss[2] = {{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, NULL, 0, VK_SHADER_STAGE_VERTEX_BIT, vm, "main", NULL},
                                                {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, NULL, 0, VK_SHADER_STAGE_FRAGMENT_BIT, fm, "main", NULL}};
        VkPipelineVertexInputStateCreateInfo vi = {VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
        VkPipelineInputAssemblyStateCreateInfo ia = {VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO, NULL, 0, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, 0};
        VkPipelineViewportStateCreateInfo vs = {VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO, NULL, 0, 1, NULL, 1, NULL};
        VkPipelineRasterizationStateCreateInfo rs = {VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO, NULL, 0, 0, 0, VK_POLYGON_MODE_FILL, VK_CULL_MODE_NONE, VK_FRONT_FACE_COUNTER_CLOCKWISE, 0, 0, 0, 0, 1};
        VkPipelineMultisampleStateCreateInfo ms = {VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO, NULL, 0, VK_SAMPLE_COUNT_1_BIT};
        VkPipelineColorBlendAttachmentState ba = {1, VK_BLEND_FACTOR_SRC_ALPHA, VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA, VK_BLEND_OP_ADD, VK_BLEND_FACTOR_ZERO, VK_BLEND_FACTOR_ONE, VK_BLEND_OP_ADD, 0xf};
        VkPipelineColorBlendStateCreateInfo cbs = {VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO, NULL, 0, 0, 0, 1, &ba};
        VkDynamicState dyn[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR}; VkPipelineDynamicStateCreateInfo dy = {VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO, NULL, 0, 2, dyn};
        VkPipelineRenderingCreateInfo rci = {VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO, NULL, 0, 1, &g_sfmt};
        if (!V.has_dynrender && g_ov.rp) { ov_fb_drop(); vkDestroyRenderPass(V.dev, g_ov.rp, NULL); g_ov.rp = VK_NULL_HANDLE; }   // the pass follows the swapchain's format, as the pipeline does
        VkRenderPass orp = V.has_dynrender ? VK_NULL_HANDLE : ov_rp();
        if (!V.has_dynrender && !orp) return 0;
        VkGraphicsPipelineCreateInfo ci = {VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, orp ? NULL : &rci, 0, 2, ss, &vi, &ia, NULL, &vs, &rs, &ms, NULL, &cbs, &dy, g_ov.layout, orp};
        VkResult r = vkCreateGraphicsPipelines(V.dev, V.pcache, 1, &ci, NULL, &g_ov.pipe);
        vkDestroyShaderModule(V.dev, vm, NULL); vkDestroyShaderModule(V.dev, fm, NULL);
        if (r) { VKRED_ONCE(92, "overlay pipeline failed: %d", r); return 0; }
        g_ov.pfmt = g_sfmt;
    }
    ov_theme_upload(cb, k, lay);
    if (g_ov.w != w || g_ov.h != h || g_ov.lay != lay || (g_tt[lay].ready != 1 && g_ov.cver != dm_overlay_custom_ver())) {   // (re)rasterize for this window size and layout
        g_ov.cver = dm_overlay_custom_ver();   // (and, without a theme, the player's own layout: the outlines are in the static layer)
        vkDeviceWaitIdle(V.dev);
        if (g_ov.view) { vkDestroyImageView(V.dev, g_ov.view, NULL); vkDestroyImage(V.dev, g_ov.img, NULL); vkFreeMemory(V.dev, g_ov.mem, NULL); vkDestroyBuffer(V.dev, g_ov.up, NULL); vkFreeMemory(V.dev, g_ov.upm, NULL); }
        g_ov.lay = lay; uint32_t sh = ov_layout_sprites(w, h); float mr[4]; ov_menu_rect((float)g_sext.width, (float)g_sext.height, mr);
        (void)mr; ov_card_size((float)g_sext.width, (float)g_sext.height, &g_ov.pw, &g_ov.ph);
        { float a[4], b[4]; ov_ed_rect((float)g_sext.width, (float)g_sext.height, a); ov_ed_prompt_rect((float)g_sext.width, (float)g_sext.height, b);   // (the editor's two panels)
          if (g_ov.pw < (uint32_t)ceilf(fmaxf(a[2], b[2]))) g_ov.pw = (uint32_t)ceilf(fmaxf(a[2], b[2]));
          if (g_ov.ph < (uint32_t)ceilf(a[3] + b[3]) + 4) g_ov.ph = (uint32_t)ceilf(a[3] + b[3]) + 4; }
        g_ov.tw = g_ov.pw > w ? g_ov.pw : w;
        g_ov.gs = (uint32_t)fmaxf(2, roundf(g_sext.height * 0.024f / 7)); g_ov.gh = 7 * g_ov.gs + 8;   // the counter's capitals: 2.4% of the window's height
        if (g_ov.tw < (uint32_t)((2 * strlen(OV_FPS_CHARS) + 2) * ov_fps_cell() + g_ov.gh)) g_ov.tw = (uint32_t)((2 * strlen(OV_FPS_CHARS) + 2) * ov_fps_cell() + g_ov.gh);   // (twice: the red copies)
        ov_accent_update(); g_ov.acc = g_acc_rgb;
        uint32_t th = h + sh + g_ov.ph + g_ov.gh;
        VkImageCreateInfo ic = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, NULL, 0, VK_IMAGE_TYPE_2D, VK_FORMAT_R8G8B8A8_UNORM, {g_ov.tw, th, 1}, 1, 1, VK_SAMPLE_COUNT_1_BIT, VK_IMAGE_TILING_OPTIMAL,
            VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, VK_SHARING_MODE_EXCLUSIVE, 0, NULL, VK_IMAGE_LAYOUT_UNDEFINED};
        VKCHK(vkCreateImage(V.dev, &ic, NULL, &g_ov.img)); VkMemoryRequirements rq; vkGetImageMemoryRequirements(V.dev, g_ov.img, &rq);
        VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL, rq.size, vk_mem_type(rq.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0)};
        VKCHK(vkAllocateMemory(V.dev, &ai, NULL, &g_ov.mem)); VKCHK(vkBindImageMemory(V.dev, g_ov.img, g_ov.mem, 0));
        VkImageViewCreateInfo vi = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, NULL, 0, g_ov.img, VK_IMAGE_VIEW_TYPE_2D, VK_FORMAT_R8G8B8A8_UNORM, {0}, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
        VKCHK(vkCreateImageView(V.dev, &vi, NULL, &g_ov.view));
        VkBufferCreateInfo bc = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, NULL, 0, (VkDeviceSize)g_ov.tw * th * 4, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_SHARING_MODE_EXCLUSIVE};
        VKCHK(vkCreateBuffer(V.dev, &bc, NULL, &g_ov.up)); VkMemoryRequirements bq; vkGetBufferMemoryRequirements(V.dev, g_ov.up, &bq);
        VkMemoryAllocateInfo bi = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL, bq.size, vk_mem_type(bq.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 0)};
        VKCHK(vkAllocateMemory(V.dev, &bi, NULL, &g_ov.upm)); VKCHK(vkBindBufferMemory(V.dev, g_ov.up, g_ov.upm, 0));
        void *px; VKCHK(vkMapMemory(V.dev, g_ov.upm, 0, VK_WHOLE_SIZE, 0, &px)); g_ov.px = px;
        g_ov.w = w; g_ov.h = h; g_ov.sh = sh; ov_raster(g_ov.px, w, h, sh); memset(g_ov.px + (size_t)g_ov.tw * (h + sh) * 4, 0, (size_t)g_ov.tw * (g_ov.ph + g_ov.gh) * 4); ov_fps_glyphs();
        VkDescriptorImageInfo ii = {g_ov.samp, g_ov.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
        VkWriteDescriptorSet wr = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, NULL, g_ov.set, 0, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &ii};
        vkUpdateDescriptorSets(V.dev, 1, &wr, 0, NULL);
        g_ov.uploaded = 0; g_ov.panel = -1;
    }
    uint32_t y0 = 0, rows = g_ov.h + g_ov.sh + g_ov.ph + g_ov.gh;   // the rows to upload
    int key = ov_panel_key(menu);
    if (g_ov.uploaded && menu >= 0 && key != g_ov.panel) {   // only the sprites, the panel and the counter (the accent), after the last copy out of the buffer has run
        VkFence f[8]; uint32_t n = 0; for (uint32_t i = 0; i < 8; i++) if (i != k) f[n++] = g_pfence[i];
        vkWaitForFences(V.dev, n, f, 1, 1000000000ull);
        y0 = g_ov.h; rows = g_ov.sh + g_ov.ph + g_ov.gh; g_ov.uploaded = 0;
    }
    if (menu >= 0 && key != g_ov.panel) {
        ov_panel(menu); g_ov.panel = key;
        if (g_ov.acc != g_acc_rgb) {   // a new highlight colour: the editor's frame sprite and the counter's letters too
            g_ov.acc = g_acc_rgb; ov_fps_glyphs();
            float *q = g_ov.sp[SP_ACCENT]; ov_canvas strip = {g_ov.px, g_ov.tw, 0, 0, g_ov.w, g_ov.h + g_ov.sh};
            cv_rrect(&strip, q[0] - q[2], q[1] - q[2], q[0] + q[2], q[1] + q[2], 0, C_ACCENT);
        }
    }
    if (!g_ov.uploaded) {
        VkImageMemoryBarrier b = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, NULL, VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, y0 ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL : VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, g_ov.img, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, NULL, 0, NULL, 1, &b);
        VkBufferImageCopy c = {(VkDeviceSize)y0 * g_ov.tw * 4, g_ov.tw, 0, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, {0, (int32_t)y0, 0}, {g_ov.tw, rows, 1}};
        vkCmdCopyBufferToImage(cb, g_ov.up, g_ov.img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &c);
        b.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT; b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT; b.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL; b.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, NULL, 0, NULL, 1, &b);
        g_ov.uploaded = 1;
    }
    return 1;
}
// ---- frame pacing on the display (VK_GOOGLE_display_timing) ----
// Without a time of its own a frame goes on screen at the first vsync after it is latched. Mario vs. Donkey Kong at 60 fps on the Legion Y700's 120 Hz panel was ready about
// 4.6 ms before that latch (SurfaceFlinger's latency: 12.9 ms from ready to shown, GPU 9.5-10.5 ms a frame), and about one frame in ten missed it by a millisecond or so: shown
// for three vsyncs, the next one for one (11 of 125 frames each way, a visible stutter; Hades II's title, ready far earlier, showed every frame for two). Each frame now gets a
// desired present time: the last one's plus the game's period, on the display's vsync grid (anchored at a past frame's actual present time), half a vsync before that vsync so
// it cannot go out early. A frame that would not be ready in time moves the cadence to the first vsync it can make (a "resync": one longer frame, and from then on a vsync of
// slack). The pause menu's redraws carry no time. DM_VK_PACE=0 (live) presents without times
static uint64_t g_refresh_ns, g_pace_target, g_pace_ideal, g_pace_anchor, g_pace_n, g_pace_resync, g_pace_slip; static uint32_t g_pace_id; static int g_pace_on = 1;
static uint64_t mono_ns(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return (uint64_t)t.tv_sec * 1000000000ull + (uint64_t)t.tv_nsec; }
static void pace_reset(void) { g_pace_target = g_pace_ideal = g_pace_anchor = g_refresh_ns = 0; }
static uint64_t pace_snap(uint64_t t) {   // the first vsync at or after t on the grid through the anchor
    uint64_t r = g_refresh_ns, a = g_pace_anchor;
    if (t <= a) return a - (a - t) / r * r;
    return a + (t - a + r - 1) / r * r;
}
static uint64_t pace_desired(double period) {   // the frame's desired present time (0 = none)
    if (!V.has_dtiming || !g_swap) return 0;
    if (g_pace_n % 30 == 0) { const char *e = getenv("DM_VK_PACE"); g_pace_on = !(e && !atoi(e)); }
    if (!g_refresh_ns || g_pace_n % 120 == 0) {
        VkRefreshCycleDurationGOOGLE rc; uint64_t was = g_refresh_ns;
        if (vkGetRefreshCycleDurationGOOGLE(V.dev, g_swap, &rc) == VK_SUCCESS && rc.refreshDuration > 1000000) g_refresh_ns = rc.refreshDuration;
        if (g_refresh_ns && g_refresh_ns != was) { VKLOG("[perf] display timing: refresh %.3f ms", g_refresh_ns / 1e6); g_pace_target = 0; }
    }
    VkPastPresentationTimingGOOGLE pt[16]; uint32_t np = 16;
    if (vkGetPastPresentationTimingGOOGLE(V.dev, g_swap, &np, pt) >= 0)
        for (uint32_t i = 0; i < np; i++) {
            present_time_record(pt[i].presentID, pt[i].actualPresentTime);
            if (pt[i].actualPresentTime > g_pace_anchor) g_pace_anchor = pt[i].actualPresentTime;
            if (pt[i].desiredPresentTime && pt[i].actualPresentTime > pt[i].desiredPresentTime + g_refresh_ns) g_pace_slip++;   // went out a vsync or more after its own
        }
    g_pace_n++;
    if (g_pace_n % 600 == 0) VKLOG("[perf] display timing: %llu frames, %llu resyncs, %llu shown a vsync or more after their time (cumulative)%s",
                                   (unsigned long long)g_pace_n, (unsigned long long)g_pace_resync, (unsigned long long)g_pace_slip, g_pace_on ? "" : "; off (DM_VK_PACE=0)");
    if (!g_pace_on || !g_refresh_ns || !g_pace_anchor || period <= 0) { g_pace_target = 0; return 0; }
    // The earliest vsync this frame can make: its GPU work (submitted just before), the blit, then the latch a vsync before it is shown. When the cadence's next vsync is
    // earlier, the cadence restarts half a vsync further out than that, so the next frames have room for the jitter: restarting at the earliest itself made 31 long frames
    // in 4,200 (5 shown late), and going on until the target was under a vsync away made 54 in 3,600 go out late
    double vk_gpu_ms_recent(void); double g = vk_gpu_ms_recent(); uint64_t gpu = (uint64_t)((g > 4 ? g : 4) * 1e6) + 3000000;
    // The cadence advances unsnapped (g_pace_ideal) and only the frame's vsync is snapped: snapping each target and adding the period to it lost the fraction,
    // so a period that is not a whole number of vsyncs always rounded up (40 fps on a 60 Hz panel: 25 ms came out as 2 vsyncs every frame, 30 fps, on a Retroid Pocket 5)
    uint64_t now = mono_ns(), t = 0;
    if (g_pace_target) { g_pace_ideal += (uint64_t)(period * 1e9); t = pace_snap(g_pace_ideal - g_refresh_ns / 2); }
    if (t < now + gpu + g_refresh_ns) { if (g_pace_target) g_pace_resync++; t = pace_snap(now + gpu + g_refresh_ns * 3 / 2); g_pace_ideal = t; }
    g_pace_target = t;
    return t - g_refresh_ns / 2;
}
static void present_window(vkimg_t *im, const int crop[4], double period, uint64_t ticket) {   // period: the game's frame period (s); < 0 for a redraw (ticket 0)
    uint32_t k = g_pi++ % 8;
    vkWaitForFences(V.dev, 1, &g_pfence[k], 1, UINT64_MAX);
    VkFormat want = im->fmt == VK_FORMAT_R8G8B8A8_SRGB || im->fmt == VK_FORMAT_B8G8R8A8_SRGB || im->fmt == VK_FORMAT_A8B8G8R8_SRGB_PACK32 ? VK_FORMAT_R8G8B8A8_SRGB : VK_FORMAT_R8G8B8A8_UNORM;
    if (want != g_want) { g_want = want; swap_destroy(); if (!swap_make()) return; }   // once, at the first present of the other kind
    uint32_t idx; VkResult r = vkAcquireNextImageKHR(V.dev, g_swap, UINT64_MAX, g_acq[k], VK_NULL_HANDLE, &idx);
    if (r == VK_ERROR_OUT_OF_DATE_KHR) { swap_destroy(); if (!swap_make()) return; r = vkAcquireNextImageKHR(V.dev, g_swap, UINT64_MAX, g_acq[k], VK_NULL_HANDLE, &idx); }
    if (r < 0) { VKRED_ONCE(90, "vkAcquireNextImageKHR: %d", r); return; }
    vkResetFences(V.dev, 1, &g_pfence[k]);
    int turned = g_pre != VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR;   // composed into g_rot.img, then copied turned into the swapchain image
    VkImage tgt = turned ? g_rot.img : g_simg[idx]; VkImageView tview = turned ? g_rot.view : g_sview[idx];
    VkCommandBuffer cb = g_pcb[k];
    VkCommandBufferBeginInfo bi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, NULL, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT}; vkBeginCommandBuffer(cb, &bi);
    vk_img_barrier_all(cb);   // after everything submitted before on this queue
    barrier(cb, tgt, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    VkClearColorValue black = {{0, 0, 0, 1}}; VkImageSubresourceRange rg = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCmdClearColorImage(cb, tgt, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &black, 1, &rg);
    int cx = crop ? crop[0] : 0, cy = crop ? crop[1] : 0, cw = crop && crop[2] ? crop[2] : (int)im->w, ch = crop && crop[3] ? crop[3] : (int)im->h;
    double s = fmin((double)g_sext.width / cw, (double)g_sext.height / ch); int dw = (int)(cw * s), dh = (int)(ch * s), dx = ((int)g_sext.width - dw) / 2, dy = ((int)g_sext.height - dh) / 2;
    vk_pic_set((float)dx / g_sext.width, (float)dy / g_sext.height, (float)dw / g_sext.width, (float)dh / g_sext.height);
    VkImageBlit b = {{VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, {{cx, cy, 0}, {cx + cw, cy + ch, 1}}, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, {{dx, dy, 0}, {dx + dw, dy + dh, 1}}};
    if (vk_window_bottom_up()) { b.srcOffsets[0].y = cy + ch; b.srcOffsets[1].y = cy; }   // lower-left origin: the window image is stored bottom-up (gfx_vk_exec.c)
    if (im->layout == VK_IMAGE_LAYOUT_UNDEFINED) { barrier(cb, im->img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL); im->layout = VK_IMAGE_LAYOUT_GENERAL; }   // an imported frame buffer, first shown (gl_android.c)
    vkCmdBlitImage(cb, im->img, VK_IMAGE_LAYOUT_GENERAL, tgt, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &b, VK_FILTER_LINEAR);
    VkImageLayout lay = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    int external = dm_external_active();
    int ovm = external ? 0 : dm_overlay_visible(), menu = external ? -1 : dm_menu_state(), fps = dm_overlay_fps();
    if ((ovm || menu >= 0 || fps) && overlay_ready(cb, k, menu) && (V.has_dynrender || ov_fb(idx))) {   // without dynamic rendering, only with the image's framebuffer
        barrier(cb, tgt, lay, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL); lay = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        VkRenderingAttachmentInfo ca = {VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, NULL, tview, lay, 0, NULL, VK_IMAGE_LAYOUT_UNDEFINED, VK_ATTACHMENT_LOAD_OP_LOAD, VK_ATTACHMENT_STORE_OP_STORE};
        VkRenderingInfo ri = {VK_STRUCTURE_TYPE_RENDERING_INFO, NULL, 0, {{0, 0}, g_sext}, 1, 0, 1, &ca, NULL, NULL};
        VkFramebuffer ofb = V.has_dynrender ? VK_NULL_HANDLE : ov_fb(idx);
        if (V.has_dynrender) vkCmdBeginRendering(cb, &ri);
        else { VkRenderPassBeginInfo rbi = {VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO, NULL, g_ov.rp, ofb, {{0, 0}, g_sext}, 0, NULL};
               vkCmdBeginRenderPass(cb, &rbi, VK_SUBPASS_CONTENTS_INLINE); }
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, g_ov.pipe);
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, g_ov.layout, 0, 1, &g_ov.set, 0, NULL);
        VkViewport vp = {0, 0, (float)g_sext.width, (float)g_sext.height, 0, 1}; VkRect2D sc = {{0, 0}, g_sext};
        vkCmdSetViewport(cb, 0, 1, &vp); vkCmdSetScissor(cb, 0, 1, &sc);
        ov_render(cb, ovm, menu, fps);
        if (V.has_dynrender) vkCmdEndRendering(cb); else vkCmdEndRenderPass(cb);
    }
    if (turned) {
        barrier(cb, g_rot.img, lay, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        barrier(cb, g_simg[idx], VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
        VkRenderingAttachmentInfo ra = {VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, NULL, g_sview[idx], VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, 0, NULL, VK_IMAGE_LAYOUT_UNDEFINED,
                                        VK_ATTACHMENT_LOAD_OP_DONT_CARE, VK_ATTACHMENT_STORE_OP_STORE};
        VkRenderingInfo rri = {VK_STRUCTURE_TYPE_RENDERING_INFO, NULL, 0, {{0, 0}, g_next}, 1, 0, 1, &ra, NULL, NULL};
        vkCmdBeginRendering(cb, &rri);
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, g_rot.pipe);
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, g_rot.layout, 0, 1, &g_rot.set, 0, NULL);
        VkViewport vp = {0, 0, (float)g_next.width, (float)g_next.height, 0, 1}; VkRect2D sc = {{0, 0}, g_next};
        vkCmdSetViewport(cb, 0, 1, &vp); vkCmdSetScissor(cb, 0, 1, &sc);
        // Where each point (u, v) of the panel's image reads the window's picture: the turn the compositor would have made, a quarter turn clockwise for
        // ROTATE_90 (u, v) -> (v, 1 - u), anticlockwise for ROTATE_270 (u, v) -> (1 - v, u)
        static const float cw[6] = {0, 1, -1, 0, 0, 1}, ccw[6] = {0, -1, 1, 0, 1, 0};
        vkCmdPushConstants(cb, g_rot.layout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof cw, g_pre == VK_SURFACE_TRANSFORM_ROTATE_90_BIT_KHR ? cw : ccw);
        vkCmdDraw(cb, 3, 1, 0, 0);
        vkCmdEndRendering(cb);
        barrier(cb, g_simg[idx], VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);
    } else barrier(cb, g_simg[idx], lay, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);
    vkEndCommandBuffer(cb);
    VkPipelineStageFlags ws = VK_PIPELINE_STAGE_TRANSFER_BIT;
    VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO, NULL, 1, &g_acq[k], &ws, 1, &cb, 1, &g_done_sem[k]};
    extern pthread_mutex_t g_vk_queue_mu;
    pthread_mutex_lock(&g_vk_queue_mu);
    if (vkQueueSubmit(V.q, 1, &si, g_pfence[k])) { pthread_mutex_unlock(&g_vk_queue_mu); VKRED_ONCE(91, "present submit failed"); return; }
    VkPresentInfoKHR pi = {VK_STRUCTURE_TYPE_PRESENT_INFO_KHR, NULL, 1, &g_done_sem[k], 1, &g_swap, &idx, NULL};
    VkPresentTimeGOOGLE ptm = {++g_pace_id, period >= 0 ? pace_desired(period) : 0};
    VkPresentTimesInfoGOOGLE pti = {VK_STRUCTURE_TYPE_PRESENT_TIMES_INFO_GOOGLE, NULL, 1, &ptm};
    if (V.has_dtiming) pi.pNext = &pti;
    r = vkQueuePresentKHR(V.q, &pi);
    if (V.has_dtiming && r >= 0) present_time_map(ptm.presentID, ticket);
    pthread_mutex_unlock(&g_vk_queue_mu);
    { void dm_renderdoc_poll(void); dm_renderdoc_poll(); }   // inactive unless DM_CTL requested a RenderDoc capture
    if (r == VK_ERROR_OUT_OF_DATE_KHR) swap_destroy();
    else if (r == VK_SUBOPTIMAL_KHR && g_pre != VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR) {   // the device turned over (ROTATE_90 <-> ROTATE_270): a swapchain for the new turn
        VkSurfaceCapabilitiesKHR c; vkGetPhysicalDeviceSurfaceCapabilitiesKHR(V.pd, g_surf, &c);
        if ((c.currentTransform & (VK_SURFACE_TRANSFORM_ROTATE_90_BIT_KHR | VK_SURFACE_TRANSFORM_ROTATE_270_BIT_KHR)) != (VkSurfaceTransformFlagsKHR)g_pre) swap_destroy();
    }
}
// headless frame shots
static int shot_wanted(uint64_t n) {
    static uint64_t until; const char *k = getenv("DM_VK_SHOTS");   // a DM_CTL "env DM_VK_SHOTS=N" line: each of the next N presents (a transition caught frame by frame)
    if (k && atoi(k) > 0) { until = n + (uint64_t)atoi(k); setenv("DM_VK_SHOTS", "0", 1); VKLOG("frame shots: presents %llu to %llu", (unsigned long long)n, (unsigned long long)until - 1); }
    if (n < until) return 1;
    static const char *e; static int init; if (!init) { init = 1; e = getenv("DM_VK_SHOT"); }
    if (!e) return 0;
    for (const char *p = e; *p;) { char *end; unsigned long long v = strtoull(p, &end, 10); if (end == p) break; if (v == n) return 1; p = *end == ',' ? end + 1 : end; }
    return 0;
}
static void shot(vkimg_t *im, uint64_t n) {
    void mtl_sub_begin(void); uint64_t mtl_sub_commit(void); int mtl_wait(uint64_t, uint64_t);
    uint32_t w = im->w, h = im->h; VkBuffer sb; VkDeviceSize so;
    mtl_sub_begin();
    uint8_t *p = vk_scratch((VkDeviceSize)w * h * 4, 16, &sb, &so);
    VkCommandBuffer cb = vk_cmd(); vk_img_barrier_all(cb);
    VkBufferImageCopy c = {so, 0, 0, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, {0, 0, 0}, {w, h, 1}};
    vkCmdCopyImageToBuffer(cb, im->img, VK_IMAGE_LAYOUT_GENERAL, sb, 1, &c);
    mtl_wait(mtl_sub_commit(), ~0ull);
    const char *dir = getenv("DM_VK_SHOT_DIR"); if (!dir) dir = getenv("DM_ROOT"); if (!dir) dir = ".";
    char path[1024]; snprintf(path, sizeof path, "%s/frame_%06llu.ppm", dir, (unsigned long long)n);
    FILE *f = fopen(path, "wb");
    if (!f) { VKLOG("[RED] cannot write %s", path); return; }
    fprintf(f, "P6\n%u %u\n255\n", w, h);
    int bgra = im->fmt == VK_FORMAT_B8G8R8A8_UNORM || im->fmt == VK_FORMAT_B8G8R8A8_SRGB;
    uint8_t *row = malloc((size_t)w * 3);
    for (uint32_t y = 0; y < h; y++) { const uint8_t *s = p + (size_t)y * w * 4; for (uint32_t x = 0; x < w; x++) { row[3 * x] = s[4 * x + (bgra ? 2 : 0)]; row[3 * x + 1] = s[4 * x + 1]; row[3 * x + 2] = s[4 * x + (bgra ? 0 : 2)]; } fwrite(row, 1, (size_t)w * 3, f); }
    free(row); fclose(f);
    VKLOG("frame %llu written to %s (%ux%u)", (unsigned long long)n, path, w, h);
}
static pthread_mutex_t g_present_mu = PTHREAD_MUTEX_INITIALIZER;   // the swapchain against the window going away (vk_window_lost, on the main thread)
void vk_window_lost(void) { pthread_mutex_lock(&g_present_mu); swap_destroy(); g_win = NULL; pthread_mutex_unlock(&g_present_mu); }
// The present thread: mtl_present only queues the frame and returns (as on Apple), so the game thread never waits in vkAcquireNextImageKHR or on the
// swapchain's fences (a FIFO swapchain blocks there until a vsync); the window texture is not drawn again before its blit, because NVN's acquire waits for
// its ticket (mtl_present_wait), and the blit's submission comes after the frame's own on the one queue. DM_VK_SYNC_PRESENT=1 presents on the game thread.
#define PQ_N 4
static struct { vkimg_t *im; int crop[4], has_crop; uint64_t n; double period; } g_pq[PQ_N];
static uint64_t g_pq_head, g_pq_tail, g_pq_done;   // queued [tail, head), presents up to ticket g_pq_done are blitted
static pthread_mutex_t g_pq_mu = PTHREAD_MUTEX_INITIALIZER; static pthread_cond_t g_pq_cv = PTHREAD_COND_INITIALIZER;
// While the game is held (the pause menu) no frame comes: dm_overlay_redraw presents the last one again with what the overlay now shows.
static _Atomic int g_redraw;
void dm_overlay_redraw(void) { atomic_store(&g_redraw, 1); pthread_mutex_lock(&g_pq_mu); pthread_cond_broadcast(&g_pq_cv); pthread_mutex_unlock(&g_pq_mu); }
static void *present_thread(void *arg) {
    (void)arg; pthread_setname_np(pthread_self(), "vk present");
    typeof(g_pq[0]) last = {0};
    for (;;) {
        pthread_mutex_lock(&g_pq_mu);
        while (g_pq_tail == g_pq_head && !atomic_load(&g_redraw)) pthread_cond_wait(&g_pq_cv, &g_pq_mu);
        atomic_store(&g_redraw, 0);
        if (g_pq_tail == g_pq_head) {   // a redraw, no new frame
            pthread_mutex_unlock(&g_pq_mu);
            if (last.im) { pthread_mutex_lock(&g_present_mu); if (swap_make()) present_window(last.im, last.has_crop ? last.crop : NULL, -1, 0); pthread_mutex_unlock(&g_present_mu); }
            continue;
        }
        typeof(g_pq[0]) e = g_pq[g_pq_tail % PQ_N];
        pthread_mutex_unlock(&g_pq_mu);
        if (e.im) { last = e; pthread_mutex_lock(&g_present_mu); if (swap_make()) present_window(e.im, e.has_crop ? e.crop : NULL, e.period, e.n); pthread_mutex_unlock(&g_present_mu); }
        pthread_mutex_lock(&g_pq_mu); g_pq_tail++; g_pq_done = e.n; pthread_cond_broadcast(&g_pq_cv); pthread_mutex_unlock(&g_pq_mu);
    }
    return NULL;
}
static void present_thread_start(void) { pthread_t t; pthread_create(&t, NULL, present_thread, NULL); pthread_detach(t); }
static uint64_t present_enqueue(vkimg_t *im, const int crop[4], uint64_t n, double period) {   // im NULL: nothing to show, the ticket still completes in order
    static int sync = -1; static pthread_once_t once = PTHREAD_ONCE_INIT;
    if (sync < 0) sync = getenv("DM_VK_SYNC_PRESENT") != NULL;
    if (sync || !dm_front_window()) {   // headless: nothing to show; or presenting on this thread
        if (sync && im) { pthread_mutex_lock(&g_present_mu); if (swap_make()) present_window(im, crop, period, n); pthread_mutex_unlock(&g_present_mu); }
        pthread_mutex_lock(&g_pq_mu); if (g_pq_tail == g_pq_head) g_pq_done = n; pthread_mutex_unlock(&g_pq_mu);   // done at once (after any queued ones)
        return n;
    }
    pthread_once(&once, present_thread_start);
    pthread_mutex_lock(&g_pq_mu);
    while (g_pq_head - g_pq_tail >= PQ_N) pthread_cond_wait(&g_pq_cv, &g_pq_mu);   // never more than PQ_N frames behind
    typeof(g_pq[0]) *e = &g_pq[g_pq_head % PQ_N]; e->im = im; e->n = n; e->period = period; e->has_crop = crop != NULL; if (crop) memcpy(e->crop, crop, sizeof e->crop);
    g_pq_head++; pthread_cond_broadcast(&g_pq_cv);
    pthread_mutex_unlock(&g_pq_mu);
    return n;
}
void dm_perf_frame(double period); void dm_perf_frame_id(uint64_t frame, double period);
void dm_perf_window(ANativeWindow *w, double fps); void dm_perf_frame_cost(int64_t replay_ns, int64_t gpu_ns);
static void present_perf_end(double period) {
    uint64_t frame = vk_frame_current();
    vk_perf_frame_seal(frame);   // Includes any screenshot submission; presentation blits have their own queue fences.
    dm_perf_frame_id(frame, period);
}   // perf_android.c
uint64_t mtl_present(const tex_t *t, double period, const int crop[4]) {
    vk_init();
    { uint64_t vk_replay_ns(void); double vk_gpu_ms_recent(void);   // what else the frame costs: its replay and its GPU time
      static uint64_t last_exec; uint64_t ex = vk_replay_ns(), rp = ex - last_exec; last_exec = ex; double g = vk_gpu_ms_recent(); int64_t gn = g > 0 ? (int64_t)(g * 1e6) : 0;
      dm_perf_frame_cost((int64_t)rp, gn); }
    // A longer period goes to the window only once it has held for 60 presents; a shorter one at once. Super Mario Bros. Wonder's own controller (main+0xcfd24) asks
    // for interval 2 for a frame or a few and then 1 again: each ask was a vote (176 in 3 minutes of its first level on the Y700, 27-60 a run in issue #64's log of an
    // Adreno 750 phone with a 60 Hz display), and where the display changes its mode for a vote, the switch itself lengthens the next frames (issue #45: 30-40 fps)
    static ANativeWindow *rate_win; static double rate_period, want_period; static uint64_t want_since;
    if (period != want_period) { want_period = period; want_since = g_presents; }
    if (g_win && (g_win != rate_win || (period != rate_period && (period < rate_period || g_presents - want_since >= 60)))) { rate_win = g_win; rate_period = period; dm_perf_window(g_win, period > 0 ? 1.0 / period : 60); }
    uint64_t n = ++g_presents;
    // without display times (no VK_GOOGLE_display_timing, no window, or none came back) the hand-over time stands in, as before: the governor's late check needs readings
    { int disp = V.has_dtiming && dm_front_window() && (n <= 240 || g_pres_recorded); struct timespec ts; if (!disp) clock_gettime(CLOCK_MONOTONIC, &ts);
      present_time_begin(n, disp ? 0 : (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec); }
    vkimg_t *im = vk_tex(t);
    static uint64_t no_img, undrawn, shown;
    if (n == 600) { void crash_handlers_check(const char *when); crash_handlers_check("after 600 presents"); }
    if (n % 600 == 0) {
        extern struct { uint64_t draws, skip_prog, skip_rt, skip_pipe, passes, folded, dispatches, pipes_built, rt_kept, ns_exec, ns_desc, ns_pipe, max_pass_draws; } g_vkst; extern uint64_t g_ubd_landed, g_ubd_superseded, g_ubd_shadows, g_ubd_bytes, g_ns_submit, g_gpu_ns, g_pkg_n, g_pkg_bytes, g_pkg_ns, g_set_hits, g_set_new, g_ubd_racy, g_ns_set_build, g_ns_set_hash, g_ns_set_new, g_set_memo, g_set_memo_bad;
        static uint64_t psb, psh, psn, pe, pd, pp, ps, pb, pdr, pg, pkn, pkb, pkt, psm;
        extern uint64_t g_desc_alloc_ns, g_desc_alloc_calls, g_desc_alloc_sets; extern int g_desc_timing;
        static uint64_t pda, pdc, pds;
        VKLOG("replay up to present %llu: draws %llu, skipped (no program %llu, no target %llu, no pipeline %llu), passes %llu, clears folded %llu, dispatches %llu, pipelines %llu; presents shown %llu, no image %llu, never drawn %llu; uniform updates landed %llu (superseded %llu, shadows %llu, CPU reads racing a GPU write %llu), same targets kept %llu, most draws in one pass %llu",
              (unsigned long long)n, (unsigned long long)g_vkst.draws, (unsigned long long)g_vkst.skip_prog, (unsigned long long)g_vkst.skip_rt, (unsigned long long)g_vkst.skip_pipe,
              (unsigned long long)g_vkst.passes, (unsigned long long)g_vkst.folded, (unsigned long long)g_vkst.dispatches, (unsigned long long)g_vkst.pipes_built,
              (unsigned long long)shown, (unsigned long long)no_img, (unsigned long long)undrawn, (unsigned long long)g_ubd_landed, (unsigned long long)g_ubd_superseded, (unsigned long long)g_ubd_shadows, (unsigned long long)g_ubd_racy, (unsigned long long)g_vkst.rt_kept, (unsigned long long)g_vkst.max_pass_draws);
        double nd = g_vkst.draws > pdr ? (double)(g_vkst.draws - pdr) / 600 : 0;
        VKLOG("replay per frame (last 600): %.0f draws, replay %.2f ms (descriptor sets %.2f, pipeline keys %.2f), queue submit %.2f ms, uniform shadows %.0f KB; GPU busy %.2f ms; texture uploads %.1f (%.2f MB, %.2f ms); descriptor sets reused %llu, new %llu (collecting %.2f ms, hashing %.2f, new ones %.2f), kept per stage %.0f a frame (memo differences %llu)",
              nd, (g_vkst.ns_exec - pe) / 6e8, (g_vkst.ns_desc - pd) / 6e8, (g_vkst.ns_pipe - pp) / 6e8, (g_ns_submit - ps) / 6e8, (g_ubd_bytes - pb) / 600.0 / 1024, (g_gpu_ns - pg) / 6e8, (g_pkg_n - pkn) / 600.0, (g_pkg_bytes - pkb) / 600.0 / 1048576, (g_pkg_ns - pkt) / 6e8, (unsigned long long)g_set_hits, (unsigned long long)g_set_new, (g_ns_set_build - psb) / 6e8, (g_ns_set_hash - psh) / 6e8, (g_ns_set_new - psn) / 6e8,
              (g_set_memo - psm) / 600.0, (unsigned long long)g_set_memo_bad);
        VKLOG("descriptor allocation per frame (last 600, timing %s): %.3f ms, %.1f driver calls, %.1f sets allocated (includes absent-stage sets)",
              g_desc_timing ? "on" : "off; DM_VK_DESC_TIMING=1", (g_desc_alloc_ns - pda) / 6e8, (g_desc_alloc_calls - pdc) / 600.0, (g_desc_alloc_sets - pds) / 600.0);
        vk_table_report();
        pda = g_desc_alloc_ns; pdc = g_desc_alloc_calls; pds = g_desc_alloc_sets;
        { extern uint64_t g_vk_nbar, g_vk_nlrz; static uint64_t pbar, plrz, ppa, pdi, pfo;   // what costs the GPU a fixed amount per pass: barriers drain it, LRZ touches are a copy and barriers each
          VKLOG("pass work per frame (last 600): %.1f passes, %.1f dispatches, %.1f clears folded, %.1f full barriers, %.1f LRZ touches", (g_vkst.passes - ppa) / 600.0, (g_vkst.dispatches - pdi) / 600.0, (g_vkst.folded - pfo) / 600.0, (g_vk_nbar - pbar) / 600.0, (g_vk_nlrz - plrz) / 600.0);
          pbar = g_vk_nbar; plrz = g_vk_nlrz; ppa = g_vkst.passes; pdi = g_vkst.dispatches; pfo = g_vkst.folded; }
        void vk_pass_report(void); vk_pass_report();
        pe = g_vkst.ns_exec; pd = g_vkst.ns_desc; pp = g_vkst.ns_pipe; ps = g_ns_submit; pb = g_ubd_bytes; pdr = g_vkst.draws; pg = g_gpu_ns; pkn = g_pkg_n; pkb = g_pkg_bytes; pkt = g_pkg_ns; psb = g_ns_set_build; psh = g_ns_set_hash; psn = g_ns_set_new; psm = g_set_memo;
    }
    if (!im) { no_img++; present_perf_end(period); return present_enqueue(NULL, NULL, n, period); }
    if (im->layout != VK_IMAGE_LAYOUT_GENERAL) { undrawn++; present_perf_end(period); return present_enqueue(NULL, NULL, n, period); }   // never drawn into: nothing to show yet
    shown++; atomic_fetch_add(&g_fps_frames, 1);
    if (shot_wanted(n)) shot(im, n);
    { void vk_pass_dump_poll(uint64_t done); uint64_t vk_fence_done(void); vk_pass_dump_poll(vk_fence_done()); void vk_pass_frame(void); vk_pass_frame(); void vk_tex_sweep(void); vk_tex_sweep(); }
    if (n % 1800 == 0) {   // every ~30 s: the pipeline cache to disk on a thread of its own (an Android app can be killed without any exit path)
        void vk_pipeline_cache_save(void); pthread_t th; pthread_attr_t a; pthread_attr_init(&a); pthread_attr_setdetachstate(&a, PTHREAD_CREATE_DETACHED);
        pthread_create(&th, &a, (void *(*)(void *))(void *)vk_pipeline_cache_save, NULL); pthread_attr_destroy(&a);
    }
    present_perf_end(period);
    return present_enqueue(im, crop, n, period);
}
// OpenGL ES games (gl_android.c): their frames are AHardwareBuffers the device's GL driver drew, imported as images and shown as an NVN window texture is
vkimg_t *vk_image_from_ahb(AHardwareBuffer *ahb, uint32_t w, uint32_t h) {
    vk_init();
    VkAndroidHardwareBufferPropertiesANDROID ap = {VK_STRUCTURE_TYPE_ANDROID_HARDWARE_BUFFER_PROPERTIES_ANDROID};
    if (vkGetAndroidHardwareBufferPropertiesANDROID(V.dev, ahb, &ap)) { VKRED_ONCE(92, "vkGetAndroidHardwareBufferPropertiesANDROID failed for a GL frame buffer"); return NULL; }
    VkExternalMemoryImageCreateInfo ei = {VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO, NULL, VK_EXTERNAL_MEMORY_HANDLE_TYPE_ANDROID_HARDWARE_BUFFER_BIT_ANDROID};
    VkImageCreateInfo ci = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, &ei, 0, VK_IMAGE_TYPE_2D, VK_FORMAT_R8G8B8A8_UNORM, {w, h, 1}, 1, 1, VK_SAMPLE_COUNT_1_BIT, VK_IMAGE_TILING_OPTIMAL,
                            VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_SHARING_MODE_EXCLUSIVE, 0, NULL, VK_IMAGE_LAYOUT_UNDEFINED};
    vkimg_t *im = calloc(1, sizeof *im);
    if (vkCreateImage(V.dev, &ci, NULL, &im->img)) { VKRED_ONCE(93, "vkCreateImage for a GL frame buffer failed"); free(im); return NULL; }
    VkImportAndroidHardwareBufferInfoANDROID imp = {VK_STRUCTURE_TYPE_IMPORT_ANDROID_HARDWARE_BUFFER_INFO_ANDROID, NULL, ahb};
    VkMemoryDedicatedAllocateInfo ded = {VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO, &imp, im->img, VK_NULL_HANDLE};
    uint32_t mt = 0; while (mt < 32 && !(ap.memoryTypeBits & (1u << mt))) mt++;
    VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, &ded, ap.allocationSize, mt};
    if (vkAllocateMemory(V.dev, &ai, NULL, &im->mem) || vkBindImageMemory(V.dev, im->img, im->mem, 0)) { VKRED_ONCE(94, "importing a GL frame buffer into Vulkan failed"); free(im); return NULL; }
    im->fmt = VK_FORMAT_R8G8B8A8_UNORM; im->w = w; im->h = h; im->d = im->layers = im->levels = im->samples = 1; im->aspect = VK_IMAGE_ASPECT_COLOR_BIT; im->layout = VK_IMAGE_LAYOUT_UNDEFINED;
    VKLOG("GL frame buffer %ux%u imported (memory type %u, %llu bytes)", w, h, mt, (unsigned long long)ap.allocationSize);
    return im;
}
uint64_t vk_present_vkimg(vkimg_t *im, double period) {
    vk_init();
    dm_perf_frame(period);
    uint64_t n = ++g_presents;
    { int disp = V.has_dtiming && dm_front_window() && (n <= 240 || g_pres_recorded); struct timespec ts; if (!disp) clock_gettime(CLOCK_MONOTONIC, &ts);
      present_time_begin(n, disp ? 0 : (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec); }
    atomic_fetch_add(&g_fps_frames, 1);
    if (shot_wanted(n) && im->layout == VK_IMAGE_LAYOUT_GENERAL) shot(im, n);
    return present_enqueue(im, NULL, n, period);
}
void mtl_present_wait(uint64_t tk) {   // NVN acquire: the window texture's last present has been blitted (the game may draw into it again)
    if (!tk) return;
    pthread_mutex_lock(&g_pq_mu);
    struct timespec dl; clock_gettime(CLOCK_REALTIME, &dl); dl.tv_sec += 1;
    while (g_pq_done < tk && pthread_cond_timedwait(&g_pq_cv, &g_pq_mu, &dl) == 0) {}
    pthread_mutex_unlock(&g_pq_mu);
}
double mtl_presented_time(uint64_t n) { return present_time_get(n); }
double vk_refresh_ms(void) { return V.has_dtiming && g_pres_recorded && g_refresh_ns ? g_refresh_ns / 1e6 : 0; }   // perf_android.c's late check: the display's refresh, when display times are what it gets
uint32_t mtl_present_qos(void) { return 0; }
int mtl_link_active(void) { return 0; }
int mtl_link_wait_tick(int vsyncs, uint64_t timeout_ns) { (void)vsyncs; (void)timeout_ns; return 0; }
int mtl_has_window(void) { return dm_front_window() != NULL; }
int mtl_display_hz(void) { return 60; }
int mtl_fps_target(void) { return DM_GAME_FPS_TARGET > 0 ? DM_GAME_FPS_TARGET : 60; }
void mtl_xv_window_tex(uint64_t serial) { (void)serial; }
void mtl_prog_release(prog_t *p) { (void)p; }

// hle_vi.c's frame history: presents queued and shown so far. The swapchain hands a frame on when mtl_present returns, so nothing stays queued
void mtl_present_progress(uint64_t *queued, uint64_t *done) { *queued = *done = g_presents; }
