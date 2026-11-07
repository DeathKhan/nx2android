// Android port, Vulkan backend: formats, textures, views, samplers and the texture / sampler ID tables (see gfx_vk.h).
//   A texture object's image is found by its storage: textures over the same pool, offset and description share one image (as nvn_mtl.m's storage-sharing
//   table), which is how a render target and the texture a later pass samples it through meet. A packaged texture (packaged layout, CPU-written) is filled
//   from the guest's block-linear bytes (nvn_deswizzle on the CPU into scratch memory, copied in by the upload command buffer) and filled again when the
//   dirty-page tracker saw a write to its pages; any other texture is the GPU's (render targets, images), filled only by GPU work.
//   Version 1 keeps every image in VK_IMAGE_LAYOUT_GENERAL after its first use, ordered by whole-pipeline barriers.
#include "gfx_vk.h"
#include "movie_zc.h"   // zero-copy movie pictures: textures over a decoder pool's picture slot (zc_tex_wants, zc_tex_refresh)
#include "nvn_tex.h"
#include "../gfx/tex/nvn_deswizzle.h"
#include "nvn_hmap.h"
#include "nvn_queue.h"
#include "gfx_vk_cpu_storage.h"   // storage the GPU writes, uncompressed textures taken from storage (nvn_mtl.m gws_span / cpu_unc)
#include <string.h>
#include <sys/mman.h>
#include "../gfx/tex/nvn_bc.h"

int nvn_dirty_since_span(const void *span, const void *addr, size_t len, uint64_t stamp);
const void *nvn_dirty_span(const void *addr, size_t len);
void nvn_dirty_protect(const void *addr, size_t len); void nvn_dirty_release(const void *addr, size_t len); uint64_t nvn_dirty_released(void); int nvn_dirty_runs(void);
uint64_t nvn_dirty_now(void);

// ---------------- formats ----------------
enum { K_NONE, K_FLOAT, K_INT, K_SINT };
typedef struct { uint16_t vk; uint8_t noalpha, rb, kind; } fent;   // vk: VkFormat (16 bits suffice for the core formats; ASTC / packed extras computed)
#define F(v) {v, 0, 0, K_FLOAT}
#define FX(v) {v, 1, 0, K_FLOAT}
#define U(v) {v, 0, 0, K_INT}
#define UX(v) {v, 1, 0, K_INT}
#define I(v) {v, 0, 0, K_SINT}
#define IX(v) {v, 1, 0, K_SINT}
static const fent FT[0x9D] = {
    [0x01] = F(VK_FORMAT_R8_UNORM), F(VK_FORMAT_R8_SNORM), U(VK_FORMAT_R8_UINT), I(VK_FORMAT_R8_SINT),
    [0x05] = F(VK_FORMAT_R16_SFLOAT), F(VK_FORMAT_R16_UNORM), F(VK_FORMAT_R16_SNORM), U(VK_FORMAT_R16_UINT), I(VK_FORMAT_R16_SINT),
    [0x0A] = F(VK_FORMAT_R32_SFLOAT), U(VK_FORMAT_R32_UINT), I(VK_FORMAT_R32_SINT),
    [0x0D] = F(VK_FORMAT_R8G8_UNORM), F(VK_FORMAT_R8G8_SNORM), U(VK_FORMAT_R8G8_UINT), I(VK_FORMAT_R8G8_SINT),
    [0x11] = F(VK_FORMAT_R16G16_SFLOAT), F(VK_FORMAT_R16G16_UNORM), F(VK_FORMAT_R16G16_SNORM), U(VK_FORMAT_R16G16_UINT), I(VK_FORMAT_R16G16_SINT),
    [0x16] = F(VK_FORMAT_R32G32_SFLOAT), U(VK_FORMAT_R32G32_UINT), I(VK_FORMAT_R32G32_SINT),
    // 0x19-0x24 three-component formats: not sampleable on mobile GPUs (vertex input only, see vk_vertex_format)
    [0x25] = F(VK_FORMAT_R8G8B8A8_UNORM), F(VK_FORMAT_R8G8B8A8_SNORM), U(VK_FORMAT_R8G8B8A8_UINT), I(VK_FORMAT_R8G8B8A8_SINT),
    [0x29] = F(VK_FORMAT_R16G16B16A16_SFLOAT), F(VK_FORMAT_R16G16B16A16_UNORM), F(VK_FORMAT_R16G16B16A16_SNORM), U(VK_FORMAT_R16G16B16A16_UINT), I(VK_FORMAT_R16G16B16A16_SINT),
    [0x2E] = F(VK_FORMAT_R32G32B32A32_SFLOAT), U(VK_FORMAT_R32G32B32A32_UINT), I(VK_FORMAT_R32G32B32A32_SINT),
    [0x31] = U(VK_FORMAT_S8_UINT), F(VK_FORMAT_D16_UNORM), F(VK_FORMAT_X8_D24_UNORM_PACK32), F(VK_FORMAT_D32_SFLOAT), F(VK_FORMAT_D24_UNORM_S8_UINT), F(VK_FORMAT_D32_SFLOAT_S8_UINT),
    [0x37] = FX(VK_FORMAT_R8G8B8A8_SRGB), F(VK_FORMAT_R8G8B8A8_SRGB),
    // NVN packed names run from low bits to high bits; Vulkan PACK16 names
    // run from high bits to low bits. RGB565 has R in bits 0..4, B in 11..15.
    [0x39] = F(VK_FORMAT_A4B4G4R4_UNORM_PACK16), FX(VK_FORMAT_A1B5G5R5_UNORM_PACK16_KHR), F(VK_FORMAT_A1B5G5R5_UNORM_PACK16_KHR), F(VK_FORMAT_B5G6R5_UNORM_PACK16),
    [0x3D] = F(VK_FORMAT_A2B10G10R10_UNORM_PACK32), U(VK_FORMAT_A2B10G10R10_UINT_PACK32), F(VK_FORMAT_B10G11R11_UFLOAT_PACK32), F(VK_FORMAT_E5B9G9R9_UFLOAT_PACK32),
    [0x41] = F(VK_FORMAT_BC1_RGB_UNORM_BLOCK), F(VK_FORMAT_BC1_RGBA_UNORM_BLOCK), F(VK_FORMAT_BC2_UNORM_BLOCK), F(VK_FORMAT_BC3_UNORM_BLOCK),
             F(VK_FORMAT_BC1_RGB_SRGB_BLOCK), F(VK_FORMAT_BC1_RGBA_SRGB_BLOCK), F(VK_FORMAT_BC2_SRGB_BLOCK), F(VK_FORMAT_BC3_SRGB_BLOCK),
    [0x49] = F(VK_FORMAT_BC4_UNORM_BLOCK), F(VK_FORMAT_BC4_SNORM_BLOCK), F(VK_FORMAT_BC5_UNORM_BLOCK), F(VK_FORMAT_BC5_SNORM_BLOCK),
             F(VK_FORMAT_BC7_UNORM_BLOCK), F(VK_FORMAT_BC7_SRGB_BLOCK), F(VK_FORMAT_BC6H_SFLOAT_BLOCK), F(VK_FORMAT_BC6H_UFLOAT_BLOCK),
    // *_UI2F / *_I2F: integer storage read as float without normalization = Vulkan's scaled formats
    [0x51] = F(VK_FORMAT_R8_USCALED), F(VK_FORMAT_R8_SSCALED), F(VK_FORMAT_R16_USCALED), F(VK_FORMAT_R16_SSCALED), U(VK_FORMAT_R32_UINT), I(VK_FORMAT_R32_SINT),
    [0x57] = F(VK_FORMAT_R8G8_USCALED), F(VK_FORMAT_R8G8_SSCALED), F(VK_FORMAT_R16G16_USCALED), F(VK_FORMAT_R16G16_SSCALED), U(VK_FORMAT_R32G32_UINT), I(VK_FORMAT_R32G32_SINT),
    [0x63] = F(VK_FORMAT_R8G8B8A8_USCALED), F(VK_FORMAT_R8G8B8A8_SSCALED), F(VK_FORMAT_R16G16B16A16_USCALED), F(VK_FORMAT_R16G16B16A16_SSCALED), U(VK_FORMAT_R32G32B32A32_UINT), I(VK_FORMAT_R32G32B32A32_SINT),
    [0x69] = F(VK_FORMAT_A2B10G10R10_SNORM_PACK32), I(VK_FORMAT_A2B10G10R10_SINT_PACK32), F(VK_FORMAT_A2B10G10R10_USCALED_PACK32), F(VK_FORMAT_A2B10G10R10_SSCALED_PACK32),
    [0x6D] = FX(VK_FORMAT_R8G8B8A8_UNORM), FX(VK_FORMAT_R8G8B8A8_SNORM), UX(VK_FORMAT_R8G8B8A8_UINT), IX(VK_FORMAT_R8G8B8A8_SINT),
    [0x71] = FX(VK_FORMAT_R16G16B16A16_SFLOAT), FX(VK_FORMAT_R16G16B16A16_UNORM), FX(VK_FORMAT_R16G16B16A16_SNORM), UX(VK_FORMAT_R16G16B16A16_UINT), IX(VK_FORMAT_R16G16B16A16_SINT),
    [0x76] = FX(VK_FORMAT_R32G32B32A32_SFLOAT), UX(VK_FORMAT_R32G32B32A32_UINT), IX(VK_FORMAT_R32G32B32A32_SINT),
    [0x95] = F(VK_FORMAT_R5G6B5_UNORM_PACK16), FX(VK_FORMAT_A1R5G5B5_UNORM_PACK16), F(VK_FORMAT_A1R5G5B5_UNORM_PACK16), F(VK_FORMAT_R5G5B5A1_UNORM_PACK16),
    [0x99] = FX(VK_FORMAT_B8G8R8A8_UNORM), F(VK_FORMAT_B8G8R8A8_UNORM), FX(VK_FORMAT_B8G8R8A8_SRGB), F(VK_FORMAT_B8G8R8A8_SRGB),
};
static vkfmt_t g_fmt[0x9D]; static uint8_t g_fmt_done[0x9D];
// BC decoded on the CPU: what each BC format decodes into (dec 1 BC1 RGB, 2 BC1, 3 BC2, 4 BC3, 5 / 6 BC4 unsigned / signed, 7 / 8 BC5, 9 / 10 BC6H unsigned / signed, 11 BC7)
static void bc_target(vkfmt_t *o) {
    int srgb = o->vk == VK_FORMAT_BC1_RGB_SRGB_BLOCK || o->vk == VK_FORMAT_BC1_RGBA_SRGB_BLOCK || o->vk == VK_FORMAT_BC2_SRGB_BLOCK || o->vk == VK_FORMAT_BC3_SRGB_BLOCK || o->vk == VK_FORMAT_BC7_SRGB_BLOCK;
    VkFormat rgba = srgb ? VK_FORMAT_R8G8B8A8_SRGB : VK_FORMAT_R8G8B8A8_UNORM;
    switch (o->vk) {
    case VK_FORMAT_BC1_RGB_UNORM_BLOCK: case VK_FORMAT_BC1_RGB_SRGB_BLOCK: o->dec = 1; o->vk = rgba; o->dbpp = 4; break;
    case VK_FORMAT_BC1_RGBA_UNORM_BLOCK: case VK_FORMAT_BC1_RGBA_SRGB_BLOCK: o->dec = 2; o->vk = rgba; o->dbpp = 4; break;
    case VK_FORMAT_BC2_UNORM_BLOCK: case VK_FORMAT_BC2_SRGB_BLOCK: o->dec = 3; o->vk = rgba; o->dbpp = 4; break;
    case VK_FORMAT_BC3_UNORM_BLOCK: case VK_FORMAT_BC3_SRGB_BLOCK: o->dec = 4; o->vk = rgba; o->dbpp = 4; break;
    case VK_FORMAT_BC4_UNORM_BLOCK: o->dec = 5; o->vk = VK_FORMAT_R8_UNORM; o->dbpp = 1; break;
    case VK_FORMAT_BC4_SNORM_BLOCK: o->dec = 6; o->vk = VK_FORMAT_R8_SNORM; o->dbpp = 1; break;
    case VK_FORMAT_BC5_UNORM_BLOCK: o->dec = 7; o->vk = VK_FORMAT_R8G8_UNORM; o->dbpp = 2; break;
    case VK_FORMAT_BC5_SNORM_BLOCK: o->dec = 8; o->vk = VK_FORMAT_R8G8_SNORM; o->dbpp = 2; break;
    case VK_FORMAT_BC6H_UFLOAT_BLOCK: o->dec = 9; o->vk = VK_FORMAT_R16G16B16A16_SFLOAT; o->dbpp = 8; break;
    case VK_FORMAT_BC6H_SFLOAT_BLOCK: o->dec = 10; o->vk = VK_FORMAT_R16G16B16A16_SFLOAT; o->dbpp = 8; break;
    default: o->dec = 11; o->vk = rgba; o->dbpp = 4; break;   // BC7
    }
}
// Decodes BC blocks (rows of whole blocks pitch bytes apart, (h + 3) / 4 rows a slice, z slices) into w x h x z texels of f->vk, rows of w * f->dbpp bytes
static void bc_decode(const vkfmt_t *f, const uint8_t *src, size_t pitch, uint32_t w, uint32_t h, uint32_t z, uint8_t *dst) {
    // Retain the renderer's tight-output interface; the shared decoder also
    // accepts explicit row/slice pitches and buffer bounds for copy paths.
    static const uint8_t formats[] = {0, 0x41, 0x42, 0x43, 0x44, 0x49, 0x4a, 0x4b, 0x4c, 0x50, 0x4f, 0x4d};
    if (!w || !h || !z || !f->dbpp || h > UINT32_MAX - 3 || f->dec >= sizeof formats || !formats[f->dec])
        die("[vk] invalid CPU BC decode dimensions or format");
    size_t rows = (h + 3) / 4;
    if (pitch > SIZE_MAX / rows || pitch * rows > SIZE_MAX / z ||
        (size_t)w > SIZE_MAX / f->dbpp || (size_t)w * f->dbpp > SIZE_MAX / h ||
        (size_t)w * f->dbpp * h > SIZE_MAX / z) die("[vk] CPU BC decode size overflow");
    size_t row = (size_t)w * f->dbpp, slice = pitch * rows;
    if (nvn_bc_decode(formats[f->dec], w, h, z, src, slice * z, pitch, slice,
                      dst, row * h * z, row, row * h)) die("[vk] invalid CPU BC decode layout");
}
void vk_bc_decode(const vkfmt_t *f, const uint8_t *src, size_t pitch, uint32_t w, uint32_t h, uint32_t z, uint8_t *dst) { bc_decode(f, src, pitch, w, h, z, dst); }
static pthread_mutex_t g_fmt_mu = PTHREAD_MUTEX_INITIALIZER;
const vkfmt_t *vk_fmt(int f) {
    if (f <= 0 || f >= 0x9D) return NULL;
    if (__atomic_load_n(&g_fmt_done[f], __ATOMIC_ACQUIRE)) return g_fmt[f].kind ? &g_fmt[f] : NULL;   // made already: no lock (the replay looks formats up for every texture it binds)
    pthread_mutex_lock(&g_fmt_mu);
    if (!g_fmt_done[f]) {
        vkfmt_t *o = &g_fmt[f]; memset(o, 0, sizeof *o);
        fent e = FT[f];
        if (f >= 0x79 && f <= 0x94) { int k = (f - 0x79) % 14; e = (fent){(uint16_t)(VK_FORMAT_ASTC_4x4_UNORM_BLOCK + 2 * k + (f >= 0x87)), 0, 0, K_FLOAT}; }
        if (e.vk) {
            int bw, bh; o->bpb = (uint8_t)nvn_fmt_bpb(f, &bw, &bh); o->bw = (uint8_t)bw; o->bh = (uint8_t)bh;
            o->vk = (VkFormat)(e.vk == (uint16_t)VK_FORMAT_A1B5G5R5_UNORM_PACK16_KHR ? VK_FORMAT_A1B5G5R5_UNORM_PACK16_KHR : e.vk);
            if (f == 0x33) {   // DEPTH24: X8_D24 where the device renders to it, else D32
                VkFormatProperties fp; vkGetPhysicalDeviceFormatProperties(V.pd, VK_FORMAT_X8_D24_UNORM_PACK32, &fp);
                if (!(fp.optimalTilingFeatures & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT)) o->vk = VK_FORMAT_D32_SFLOAT;
            }
            if (f == 0x31) {   // STENCIL8 alone: a combined format where the device has no S8 attachments
                VkFormatProperties fp; vkGetPhysicalDeviceFormatProperties(V.pd, VK_FORMAT_S8_UINT, &fp);
                if (!(fp.optimalTilingFeatures & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT)) o->vk = VK_FORMAT_D24_UNORM_S8_UINT;
            }
            o->depth = f >= 0x32 && f <= 0x36; o->stencil = f == 0x31 || f == 0x35 || f == 0x36 || (f == 0x31);
            o->compressed = o->bw > 1 || o->bh > 1;
            o->is_int = e.kind == K_INT || e.kind == K_SINT; o->is_sint = e.kind == K_SINT; o->kind = 1;
            o->swz = (VkComponentMapping){VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY, e.noalpha ? VK_COMPONENT_SWIZZLE_ONE : VK_COMPONENT_SWIZZLE_IDENTITY};
            if (o->compressed && o->vk >= VK_FORMAT_BC1_RGB_UNORM_BLOCK && o->vk <= VK_FORMAT_BC7_SRGB_BLOCK && !V.has_bc) {
                // no BC on the device (the Adreno 618 / 619 drivers of 2021, 512.530): an uncompressed image the CPU decodes into (bc_decode), 4 to 8 times the memory
                VkFormat bc = o->vk; bc_target(o); o->compressed = 0;
                VKLOG("texture format %#x (VkFormat %d): the device has no BC textures, decoded on the CPU into VkFormat %d", f, bc, o->vk);
            }
            if (o->vk >= VK_FORMAT_ASTC_4x4_UNORM_BLOCK && o->vk <= VK_FORMAT_ASTC_12x12_SRGB_BLOCK && !V.has_astc) {
                o->kind = 0;   // said once per format: these textures went missing with nothing in the log
                VKLOG("[RED] texture format %#x (VkFormat %d) needs ASTC (textureCompressionASTC_LDR), which the device lacks: textures in it are not drawn", f, o->vk);
            }
        }
        __atomic_store_n(&g_fmt_done[f], 1, __ATOMIC_RELEASE);   // published once complete
    }
    pthread_mutex_unlock(&g_fmt_mu);
    return g_fmt[f].kind ? &g_fmt[f] : NULL;
}
static VkImageAspectFlags aspect_of(const vkfmt_t *f) {
    if (f->vk == VK_FORMAT_S8_UINT) return VK_IMAGE_ASPECT_STENCIL_BIT;
    if (f->depth || f->stencil) return VK_IMAGE_ASPECT_DEPTH_BIT | (f->vk == VK_FORMAT_D24_UNORM_S8_UINT || f->vk == VK_FORMAT_D32_SFLOAT_S8_UINT ? VK_IMAGE_ASPECT_STENCIL_BIT : 0);
    return VK_IMAGE_ASPECT_COLOR_BIT;
}

// ---------------- images ----------------
static unsigned layers_of(const nvn_texdesc *d) { return d->target == 4 || d->target == 9 ? (d->depth ? d->depth : 1) : d->target == 8 ? 6 : d->target == 3 ? (d->height ? d->height : 1) : 1; }
static VkImageViewType vtype_of(int target) {
    switch (target) {
    case 0: return VK_IMAGE_VIEW_TYPE_1D; case 3: return VK_IMAGE_VIEW_TYPE_1D_ARRAY; case 2: return VK_IMAGE_VIEW_TYPE_3D;
    case 4: case 6: return VK_IMAGE_VIEW_TYPE_2D_ARRAY; case 8: return VK_IMAGE_VIEW_TYPE_CUBE; case 9: return VK_IMAGE_VIEW_TYPE_CUBE_ARRAY;
    default: return VK_IMAGE_VIEW_TYPE_2D;
    }
}
// storage key: textures with the same pool, offset and description share one image
typedef struct { uint64_t pool; int64_t off; nvn_texdesc d; } akey;
static nvn_hmap g_by_key = {.klen = sizeof(akey)};     // akey → vkimg_t*
static nvn_hmap g_by_serial = {.klen = sizeof(uint64_t)};  // serial → vkimg_t* (images with no storage key: no pool memory)
static pthread_mutex_t g_img_mu = PTHREAD_MUTEX_INITIALIZER;
// The sRGB / UNORM twin of a format (the one reinterpretation games make of uncompressed colour targets), VK_FORMAT_UNDEFINED when it has none
static VkFormat srgb_twin(VkFormat f) {
    switch (f) {
    case VK_FORMAT_R8G8B8A8_UNORM: return VK_FORMAT_R8G8B8A8_SRGB; case VK_FORMAT_R8G8B8A8_SRGB: return VK_FORMAT_R8G8B8A8_UNORM;
    case VK_FORMAT_B8G8R8A8_UNORM: return VK_FORMAT_B8G8R8A8_SRGB; case VK_FORMAT_B8G8R8A8_SRGB: return VK_FORMAT_B8G8R8A8_UNORM;
    case VK_FORMAT_A8B8G8R8_UNORM_PACK32: return VK_FORMAT_A8B8G8R8_SRGB_PACK32; case VK_FORMAT_A8B8G8R8_SRGB_PACK32: return VK_FORMAT_A8B8G8R8_UNORM_PACK32;
    default: return VK_FORMAT_UNDEFINED;
    }
}
// DM_VK_UBWC (start-up, bits): 1, uncompressed colour images get a format list (their format and its sRGB twin) instead of a free MUTABLE_FORMAT, which Adreno
// needs to keep UBWC framebuffer compression; views in another format then use the image's own (vk_tex_view). 2, depth / stencil images are made without
// MUTABLE_FORMAT and EXTENDED_USAGE (their views always take the image's own format). Unicorn Overlord on an Adreno 730: 1 cut GPU time by 35-40%. Both are the
// default (3): Super Mario Bros. Wonder's world map on the Adreno 830 took 8.7-8.9 ms of GPU a frame instead of 11.8, the same picture (1 alone: 9.5-10.0 ms).
// Xenoblade Chronicles 3's field took 22.9-23.0 ms against 23.0-23.6 (its own shading dominates), with clouds in every frame of a sky turn and no idle flicker;
// Graveyard Keeper 2, Advance Wars and DKCR drew the same pictures. A colour view in another format than the image's and its twin is logged [RED] (vk_tex_view):
// such a game wants DM_VK_UBWC=2 in its profile
static int ubwc_mode(void) { static int v = -1; if (v < 0) { const char *e = getenv("DM_VK_UBWC"); v = e ? atoi(e) : 3; } return v; }
// A float colour format the device cannot store to from a shader (Adreno: E5B9G9R9 and B10G11R11 have no STORAGE_IMAGE feature) is promoted to R16G16B16A16_SFLOAT when the image needs
// numeric image stores through that same format. A different-format storage view can instead write packed bits (for example R32F over RGB9E5).
// Defer the choice until storage preflight knows the effective view; the builder IMAGE flag alone cannot distinguish numeric stores from bit reinterpretation.
static VkFormat storage_promote(const vkfmt_t *f) {
    if (!f || (f->vk != VK_FORMAT_E5B9G9R9_UFLOAT_PACK32 && f->vk != VK_FORMAT_B10G11R11_UFLOAT_PACK32)) return VK_FORMAT_UNDEFINED;
    VkFormatProperties fp; vkGetPhysicalDeviceFormatProperties(V.pd, f->vk, &fp); if (fp.optimalTilingFeatures & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT) return VK_FORMAT_UNDEFINED;
    vkGetPhysicalDeviceFormatProperties(V.pd, VK_FORMAT_R16G16B16A16_SFLOAT, &fp); return fp.optimalTilingFeatures & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT ? VK_FORMAT_R16G16B16A16_SFLOAT : VK_FORMAT_UNDEFINED;
}
// Uncompressed colour views of the same texel size are Vulkan-compatible. Decoded/compressed/depth views need separate rules.
static int storage_view_compatible(const vkfmt_t *base, const vkfmt_t *view) {
    return base && view && !base->compressed && !view->compressed && !base->dec && !view->dec &&
        !base->depth && !view->depth && !base->stencil && !view->stencil &&
        base->bw == 1 && base->bh == 1 && view->bw == 1 && view->bh == 1 && base->bpb && base->bpb == view->bpb;
}
static vkimg_t *img_new(const tex_t *t, const vkfmt_t *f, VkImageUsageFlags extra_usage, VkFormat storage_format) {
    vkimg_t *im = calloc(1, sizeof *im);
    const nvn_texdesc *d = &t->d;
    int storage_reinterpret = (extra_usage & VK_IMAGE_USAGE_STORAGE_BIT) && storage_format != VK_FORMAT_UNDEFINED && storage_format != f->vk;
    vkfmt_t pf; VkFormat pv = (extra_usage & VK_IMAGE_USAGE_STORAGE_BIT) && !storage_reinterpret ? storage_promote(f) : VK_FORMAT_UNDEFINED;
    if (pv) { pf = *f; pf.vk = pv; pf.swz = (VkComponentMapping){0}; f = &pf; im->promoted = 1; extra_usage |= VK_IMAGE_USAGE_STORAGE_BIT;
        static int nl; if (nl++ < 8) VKLOG("image %ux%ux%u nvn format %#x (vk %d) promoted to R16G16B16A16_SFLOAT for image stores (flags %#x, target %d)", d->width, d->height, d->depth, d->format, pv, d->flags, d->target); }
    im->w = d->width ? d->width : 1; im->h = d->target == 0 || d->target == 3 ? 1 : (d->height ? d->height : 1);
    im->d = d->target == 2 ? (d->depth ? d->depth : 1) : 1; im->layers = layers_of(d);
    im->levels = d->levels ? d->levels : 1; im->samples = d->samples > 1 ? d->samples : 1; im->target = d->target; im->nvn_format = d->format;
    im->fmt = f->vk; im->vtype = vtype_of(d->target); im->aspect = aspect_of(f);
    VkFormatProperties fp; vkGetPhysicalDeviceFormatProperties(V.pd, f->vk, &fp);
    VkImageUsageFlags u = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | extra_usage;
    if (fp.optimalTilingFeatures & VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT) u |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    if (fp.optimalTilingFeatures & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT) u |= VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
    if ((d->flags & NVN_TEXTURE_FLAGS_IMAGE) && (fp.optimalTilingFeatures & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT)) u |= VK_IMAGE_USAGE_STORAGE_BIT;
    // a texture over a movie decoder's picture slot (movie_zc.h): the zero-copy conversion writes it as a storage image (zc_tex_refresh; without the usage it goes through a staging copy)
    if (t->mem && (fp.optimalTilingFeatures & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT) && zc_tex_wants(t->mem, t->L.size)) u |= VK_IMAGE_USAGE_STORAGE_BIT;
    // a block-compressed texture can be drawn into through a view in the uncompressed format of its block size (Mario Kart 8 Deluxe writes BC3 blocks into its
    // reflection cubemaps through RGBA32UI views): with EXTENDED_USAGE the image may carry a usage its own format lacks, and an attachment view needs it on the image
    if (f->compressed && !f->dec && (f->bpb == 16 || f->bpb == 8)) { VkFormatProperties up; vkGetPhysicalDeviceFormatProperties(V.pd, f->bpb == 16 ? VK_FORMAT_R32G32B32A32_UINT : VK_FORMAT_R32G32_UINT, &up);
        if (up.optimalTilingFeatures & VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT) u |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT; }
    VkImageCreateInfo ci = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, NULL,
        // a square 2D array of 6n layers too: shaders read one as a cube (Pokémon Legends: Arceus's reflection probes; the view was refused by the validation layer without the flag)
        VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT | VK_IMAGE_CREATE_EXTENDED_USAGE_BIT | (d->target == 8 || d->target == 9 || (d->target == 4 && im->layers >= 6 && im->layers % 6 == 0 && im->w == im->h && im->samples <= 1) ? VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT : 0) |
        (d->target == 2 ? VK_IMAGE_CREATE_2D_ARRAY_COMPATIBLE_BIT : 0) | (f->compressed ? VK_IMAGE_CREATE_BLOCK_TEXEL_VIEW_COMPATIBLE_BIT : 0),
        d->target == 2 ? VK_IMAGE_TYPE_3D : d->target == 0 || d->target == 3 ? VK_IMAGE_TYPE_1D : VK_IMAGE_TYPE_2D, f->vk,
        {im->w, im->h, im->d}, im->levels, im->layers, (VkSampleCountFlagBits)im->samples, VK_IMAGE_TILING_OPTIMAL, u, VK_SHARING_MODE_EXCLUSIVE, 0, NULL, VK_IMAGE_LAYOUT_UNDEFINED};
    if (im->samples > 1) ci.flags &= ~(VkImageCreateFlags)VK_IMAGE_CREATE_2D_ARRAY_COMPATIBLE_BIT;
    VkFormat fl[2] = {f->vk, srgb_twin(f->vk)}; VkImageFormatListCreateInfo fli = {VK_STRUCTURE_TYPE_IMAGE_FORMAT_LIST_CREATE_INFO, NULL, fl[1] ? 2u : 1u, fl};
    if (!storage_reinterpret && !f->compressed && (ubwc_mode() & (im->aspect == VK_IMAGE_ASPECT_COLOR_BIT ? 1 : 2))) {   // (a depth format has no twin: a list of itself, no MUTABLE_FORMAT)
        im->fixed_fmt = 1; im->fmt2 = fl[1]; ci.pNext = &fli;
        ci.flags &= ~(VkImageCreateFlags)VK_IMAGE_CREATE_EXTENDED_USAGE_BIT; if (!fl[1]) ci.flags &= ~(VkImageCreateFlags)VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT;
    }
    im->usage = ci.usage; im->cflags = ci.flags;
    VkResult r = vkCreateImage(V.dev, &ci, NULL, &im->img);
    if (r && f->compressed && (ci.usage & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT)) {   // the driver refused the attachment usage on a compressed image: as before, without it
        VKRED_ONCE(96, "compressed image format %#x refused colour-attachment usage (%d): created without it", d->format, r);
        ci.usage &= ~(VkImageUsageFlags)VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT; im->usage = ci.usage; r = vkCreateImage(V.dev, &ci, NULL, &im->img); }
    if (r) { VKRED_ONCE(10, "vkCreateImage failed (%d): %ux%ux%u format %#x (vk %d) target %d levels %u layers %u samples %u", r, im->w, im->h, im->d, d->format, f->vk, d->target, im->levels, im->layers, im->samples); free(im); return NULL; }
    VkMemoryRequirements req; vkGetImageMemoryRequirements(V.dev, im->img, &req);
    if ((ci.usage & (VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT)) && im->w >= 256) { static int n; if (n++ < 16)   // UBWC check: a compressed target takes room for its metadata
        VKLOG("target %ux%u format %d%s: %llu bytes", im->w, im->h, f->vk, im->fixed_fmt ? " (format-listed)" : "", (unsigned long long)req.size); }
    VkDeviceSize moff;
    if (!(im->mem = vk_image_memory(&req, vk_mem_type(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT), &moff, &im->sub))) {
        VKRED_ONCE(11, "image memory allocation failed (%llu bytes)", (unsigned long long)req.size); vkDestroyImage(V.dev, im->img, NULL); free(im); return NULL; }
    VKCHK(vkBindImageMemory(V.dev, im->img, im->mem, moff));
    im->layout = VK_IMAGE_LAYOUT_UNDEFINED; im->vkbytes = req.size;
    return im;
}
void vk_tex_prepare(VkCommandBuffer cb, vkimg_t *im) {
    if (im->layout == VK_IMAGE_LAYOUT_GENERAL) return;
    VkImageMemoryBarrier b = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, NULL, 0, VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
        VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, im->img, {im->aspect, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS}};
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, NULL, 0, NULL, 1, &b);
    im->layout = VK_IMAGE_LAYOUT_GENERAL;
    // A new image's contents are undefined in Vulkan, where new Switch memory reads 0 (nxnative's Metal backend zeroes new textures too). Xenoblade Chronicles 3's water keeps
    // a texture that builds on its previous frame: made again by the switch to docked mode, it started from leftover memory, and blocks of it stayed on the water from then on.
    // So an image starts as zeros, before an upload or a draw replaces them. Block-compressed images can't be cleared; they are always uploaded
    const vkfmt_t *f = vk_fmt(im->nvn_format);
    if (!(im->usage & VK_IMAGE_USAGE_TRANSFER_DST_BIT) || (f && f->compressed)) return;
    VkImageSubresourceRange r = {im->aspect, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS};
    if (im->aspect & (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT)) { VkClearDepthStencilValue z = {0, 0}; vkCmdClearDepthStencilImage(cb, im->img, VK_IMAGE_LAYOUT_GENERAL, &z, 1, &r); }
    else { VkClearColorValue z = {{0, 0, 0, 0}}; vkCmdClearColorImage(cb, im->img, VK_IMAGE_LAYOUT_GENERAL, &z, 1, &r); }
    VkMemoryBarrier mb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, NULL, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT};
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
}
// Upload a packaged texture from guest memory: each level and layer deswizzled into scratch memory, copied in by the upload command buffer
uint64_t g_pkg_n, g_pkg_bytes, g_pkg_ns;   // packaged texture uploads: count, bytes, CPU time (the replay's per-frame line)
static void pkg_upload_(vkimg_t *im, const tex_t *t, const vkfmt_t *f);
static uint64_t pkg_hash(const uint8_t *m, size_t n) {   // FNV-1a over 8-byte words
    uint64_t h = 1469598103934665603ull; size_t i = 0;
    for (; i + 8 <= n; i += 8) { uint64_t w; memcpy(&w, m + i, 8); h = (h ^ w) * 1099511628211ull; }
    for (; i < n; i++) h = (h ^ m[i]) * 1099511628211ull;
    return h;
}
static void pkg_upload(vkimg_t *im, const tex_t *t, const vkfmt_t *f) {
    struct timespec a, b; clock_gettime(CLOCK_MONOTONIC, &a);
    pkg_upload_(im, t, f);
    clock_gettime(CLOCK_MONOTONIC, &b); g_pkg_ns += (uint64_t)((b.tv_sec - a.tv_sec) * 1000000000ll + (b.tv_nsec - a.tv_nsec)); g_pkg_n++; g_pkg_bytes += t->L.size;
}
static void pkg_upload_(vkimg_t *im, const tex_t *t, const vkfmt_t *f) {
    { static int n; const char *e = getenv("DM_VK_LOG_UPLOADS"); if (e && atoi(e) && n < 60) { n++;   // which packaged textures are (re-)uploaded
        VKLOG("upload: serial %llu %ux%ux%u nvn format %#x (vk %d) target %d levels %u layers %u flags %#x stride %u size %llu mem %p, tileH*100+pkg_block_h %d%s",
              (unsigned long long)t->serial, t->d.width, t->d.height, t->d.depth, t->d.format, f->vk, t->d.target, im->levels, im->layers, t->d.flags, t->d.stride, (unsigned long long)t->L.size,
              (const void *)t->mem, (int)t->L.tileH * 100 + t->d.pkg_block_h, im->packaged > 1 ? " (again)" : ""); } }
    if (im->packaged < 100) im->packaged++;
    if (im->promoted) { VKRED_ONCE(171, "a packaged upload of an image promoted for image stores (nvn format %#x, %ux%u): skipped, its texels are not the guest's layout", t->d.format, t->d.width, t->d.height); return; }
    int staged = im->layout != VK_IMAGE_LAYOUT_GENERAL && vk_stage_wanted();   // a first upload with the submission's scratch over its cap: through the bounded staging buffer (gfx_vk_core.c)
    VkCommandBuffer cb = staged ? vk_stage_cmd() : vk_upload_cmd();
    if (im->layout != VK_IMAGE_LAYOUT_GENERAL) vk_tex_prepare(cb, im);
    nvn_dirty_protect(t->mem, t->L.size);
    im->pkg_stamp = nvn_dirty_now(); im->pkg_span = nvn_dirty_span(t->mem, t->L.size); im->pkg_seen = im->pkg_stamp;
    im->pkg_mem = t->mem; im->pkg_len = t->L.size; im->pkg_released = 0; im->pkg_hash = pkg_hash(t->mem, t->L.size);
    int linear = (t->d.flags & (NVN_TEXTURE_FLAGS_LINEAR | NVN_TEXTURE_FLAGS_LINEAR_RENDER_TARGET)) != 0;
    unsigned bw = f->bw ? f->bw : 1, bh = f->bh ? f->bh : 1, bpb = f->bpb;
    if (!bpb) { VKRED_ONCE(12, "packaged texture format %#x has no block size", t->d.format); return; }
    for (unsigned lv = 0; lv < im->levels; lv++) {
        uint32_t w = im->w >> lv ? im->w >> lv : 1, h = im->h >> lv ? im->h >> lv : 1, z = im->d >> lv ? im->d >> lv : 1;
        uint32_t bx = (w + bw - 1) / bw, by = (h + bh - 1) / bh; size_t pitch = (size_t)bx * bpb, slab = pitch * by * z;
        for (unsigned ly = 0; ly < im->layers; ly++) {
            VkBuffer sb; VkDeviceSize so; uint8_t *blocks = f->dec ? malloc(slab) : NULL, *dst = blocks ? blocks : staged ? vk_stage_alloc(slab, 16, &sb, &so) : vk_scratch(slab, 16, &sb, &so);
            if (f->dec && !blocks) return;
            if (linear) {   // linear storage: rows of the texture's stride, one level, one layer
                size_t stride = t->d.stride ? t->d.stride : pitch;
                for (uint32_t y = 0; y < by * z; y++) memcpy(dst + y * pitch, t->mem + (size_t)y * stride, pitch);
            } else if (nvn_deswizzle(&t->d, lv, ly, t->mem, dst, pitch)) { free(blocks); VKRED_ONCE(13, "deswizzle refused texture format %#x target %d", t->d.format, t->d.target); return; }
            if (blocks) { dst = staged ? vk_stage_alloc((size_t)w * h * z * f->dbpp, 16, &sb, &so) : vk_scratch((size_t)w * h * z * f->dbpp, 16, &sb, &so); bc_decode(f, blocks, pitch, w, h, z, dst); free(blocks); }   // (BC the device lacks)
            VkBufferImageCopy r = {so, 0, 0, {im->aspect & VK_IMAGE_ASPECT_COLOR_BIT ? VK_IMAGE_ASPECT_COLOR_BIT : VK_IMAGE_ASPECT_DEPTH_BIT, lv, ly, 1}, {0, 0, 0}, {w, h, z}};
            vkCmdCopyBufferToImage(cb, sb, im->img, VK_IMAGE_LAYOUT_GENERAL, 1, &r);
        }
    }
}
static int packaged_tex(const tex_t *t);
static int cpu_content_tex(const tex_t *t, const vkfmt_t *f);
static int unc_rule(void); static void cpu_unc_dropped(uint64_t serial);
static int lin_copy(vkimg_t *im, const tex_t *t, const vkfmt_t *f); static int linear_storage(const tex_t *t);
static void vk_tex_view_drop_all(vkimg_t *im);
// Images over the same memory with different descriptions (the storage key differs) are separate Vulkan images: on the Switch a texture sees what another
// one over its memory rendered, here it does not. They are linked both ways at creation (up to 8 each), and a sampled image older than an overlapping
// image's last rendering is logged ([alias] stale read), to find what a game reinterprets
static uint64_t g_wseq;
static void alias_link(vkimg_t *im) {
    for (uint32_t i = 0; i < g_by_key.cap; i++) {
        vkimg_t *o = g_by_key.vals[i]; if (!o || o == im || o->pool != im->pool || o->off >= im->off + im->size || im->off >= o->off + o->size) continue;
        if (im->novl < 8 && o->novl < 8) { im->ovl[im->novl++] = o; o->ovl[o->novl++] = im; }
    }
}
static void alias_unlink(vkimg_t *im) {
    for (int i = 0; i < im->novl; i++) { vkimg_t *o = im->ovl[i]; for (int j = 0; j < o->novl; j++) if (o->ovl[j] == im) { o->ovl[j] = o->ovl[--o->novl]; break; } }
    im->novl = 0;
}
static void alias_check(vkimg_t *im) {   // im is about to be sampled
    for (int i = 0; i < im->novl; i++) {
        vkimg_t *o = im->ovl[i]; if (o->wseq <= im->wseq || o->wseq <= im->rseen) continue;
        im->rseen = o->wseq;
        static int n; if (n++ < 200) VKLOG("[alias] stale read: %ux%u fmt %#x levels %u at %+lld..%+lld sampled, but %ux%u fmt %#x levels %u at %+lld..%+lld rendered over its memory later",
            im->w, im->h, im->nvn_format, im->levels, (long long)im->off, (long long)(im->off + im->size), o->w, o->h, o->nvn_format, o->levels, (long long)o->off, (long long)(o->off + o->size));
    }
}
extern uint64_t g_tex_frame;   // presents so far, for the packaged textures' last use (vk_tex_sweep)
// Images are kept per storage key until their pool goes, so a texture the game streams into memory another texture had before gets an image of its own and the old one
// stayed: Xenoblade Chronicles 3 streams its field textures through pools that live for good, the device's GPU memory grew by about 470 MB a minute while walking (6.3 GB
// after 20 minutes) and Android's low-memory killer ended the game. A new image takes over: the images of the same pool over its memory that were not looked up in the
// last EVICT_KEEP presents go (their memory now holds another texture; one drawn again is made and filled again). Called with g_img_mu held
enum { EVICT_KEEP = 60 };
static uint64_t g_img_n, g_img_bytes, g_img_evicted, g_img_vkbytes;   // live images and their bytes (guest layout; vkbytes: device memory), images evicted so far (vk_tex_sweep's line)
extern uint64_t g_tex_gen;
#include "gfx_vk_texture_diag.h"
static void img_drop(vkimg_t *im) { if (im->pins) {
    // The native image survives a finalized pool, but its CPU watches must stop before that
    // address can be reused. Orphan lookup never reads the former guest storage again.
    if (im->lin_mem) nvn_dirty_release(im->lin_mem, im->lin_len);
    if (im->packaged && !im->pkg_released && im->pkg_mem) nvn_dirty_release(im->pkg_mem, im->pkg_len);
    im->lin_mem = NULL; im->lin_len = 0; im->pkg_mem = NULL; im->pkg_len = 0; im->pkg_released = 1;
    im->orphan = 1; alias_unlink(im); g_tex_gen++; { extern uint64_t g_tex_gen_why[4]; g_tex_gen_why[0]++; } vk_tex_diag_native(TEX_DIAG_ORPHAN, NULL, im); return;
} vk_lrz_forget(im); if (im->lin_mem) nvn_dirty_release(im->lin_mem, im->lin_len); if (im->packaged && !im->pkg_released && im->pkg_mem) nvn_dirty_release(im->pkg_mem, im->pkg_len); g_img_n--; g_img_bytes -= im->size > 0 ? (uint64_t)im->size : 0; g_img_vkbytes -= im->vkbytes; alias_unlink(im); vk_tex_view_drop_all(im); if (im->sub) vk_defer_free_image_sub(im->img, im->sub); else vk_defer_free_image(im->img, im->mem); free(im); }
// A render target is also taken over when only the pool tables looked it up in that time. The tables' fill looks up every registered texture ID, so a target whose
// texture the game leaves registered never aged: Ocarina of Time places each frame's targets at changing offsets of one pool and leaves the old textures registered,
// and their images stayed (some 40 more a second while the camera turned, 2.1 GB within 15 seconds, and Android's low-memory killer ended the game). Such a target
// was neither drawn to nor bound for RT_KEEP presents and a new texture is being made over part of its memory at another place, so the game has reused the memory
// (on the console the new target's rendering overwrites the old one's texels; a frame's targets do not overlap each other, so one unused since the last present is dead).
// Its description is remembered (g_reclaimed) so that the next fill does not make the image again from the registration (vk_tex_pool_ok); a bind or a draw to it
// forgets that. A lookup by a fill never evicts (the table being filled may hold the view already)
static __thread int t_pool_fill;
static nvn_hmap g_reclaimed = {.klen = sizeof(akey)}; static uint64_t g_img_reclaimed;   // g_img_mu
enum { RT_KEEP = 2 };
static int rt_stale(const vkimg_t *o, int64_t off, int64_t size) {
    return !t_pool_fill && o->rt_last && !o->packaged && !(o->off == off && o->size == size) && g_tex_frame - o->used_bind >= RT_KEEP && g_tex_frame + 1 - o->rt_last >= RT_KEEP;
}
static void evict_overlaps(uint64_t pool, int64_t off, int64_t size) {
    akey ks[64]; int n = 0;
    for (uint32_t i = 0; i < g_by_key.cap && n < 64; i++) {
        vkimg_t *o = g_by_key.vals[i]; if (!o || o->pins || o->pool != pool || o->off >= off + size || off >= o->off + o->size || (g_tex_frame - o->used < EVICT_KEEP && !rt_stale(o, off, size))) continue;
        memcpy(&ks[n++], g_by_key.keys + (size_t)i * g_by_key.klen, sizeof(akey));
    }
    for (int j = 0; j < n; j++) { vkimg_t *o = nvn_hm_get(&g_by_key, &ks[j]); if (!o) continue;
        if (g_tex_frame - o->used < EVICT_KEEP) {   // (kept until now by the fills alone)
            if (g_reclaimed.n >= 32768) { free(g_reclaimed.keys); free(g_reclaimed.vals); nvn_hm_init(&g_reclaimed, sizeof(akey)); }
            nvn_hm_put(&g_reclaimed, &ks[j], (void *)1); g_img_reclaimed++; }
        nvn_hm_del(&g_by_key, &ks[j]); img_drop(o); g_img_evicted++; }
}
// Snapshot pins are keyed by serial plus the exact storage description. A serial can have
// aliases with different IMAGE flags; releasing its final pin releases every resolved alias.
typedef struct tex_pin_image { akey key; vkimg_t *im; struct tex_pin_image *next; } tex_pin_image;
typedef struct { uint64_t refs; tex_pin_image *images; } tex_pin;
static nvn_hmap g_tex_pins = {.klen = sizeof(uint64_t)};   // g_img_mu
static void tex_storage_key(akey *k, const tex_t *t, const vkfmt_t *f) {
    memset(k, 0, sizeof *k); k->pool = (uint64_t)(uintptr_t)t->pool; k->off = t->off; k->d = t->d;
    (void)f;
    // IMAGE is usage intent, not a different storage allocation. Actual storage use upgrades
    // the shared native image in vk_tex_storage_prepare while preserving its rendered contents.
    k->d.flags &= NVN_TEXTURE_FLAGS_LINEAR | NVN_TEXTURE_FLAGS_SPARSE | NVN_TEXTURE_FLAGS_LINEAR_RENDER_TARGET | NVN_TEXTURE_FLAGS_MINIMAL_LAYOUT;
}
static tex_pin *tex_pin_get(uint64_t serial) {
    tex_pin *p = nvn_hm_get(&g_tex_pins, &serial);
    if (!p) { p = calloc(1, sizeof *p); if (!p) die("[vk] texture pin allocation failed"); nvn_hm_put(&g_tex_pins, &serial, p); }
    return p;
}
static tex_pin_image *tex_pin_key(tex_pin *p, const akey *key) {
    for (tex_pin_image *i = p->images; i; i = i->next) if (!memcmp(&i->key, key, sizeof *key)) return i;
    tex_pin_image *i = calloc(1, sizeof *i); if (!i) die("[vk] texture pin image allocation failed");
    i->key = *key; i->next = p->images; p->images = i; return i;
}
// g_img_mu held. Pinned old pool contents take precedence over a new image reusing its key.
static vkimg_t *tex_image_find(const tex_t *t, const akey *key, tex_pin_image **pin_out) {
    tex_pin *pin = nvn_hm_get(&g_tex_pins, &t->serial);
    tex_pin_image *p = pin ? tex_pin_key(pin, key) : NULL;
    if (pin_out) *pin_out = p;
    return p && p->im ? p->im : t->pool ? nvn_hm_get(&g_by_key, key) : nvn_hm_get(&g_by_serial, &t->serial);
}
static void tex_image_pin_resolved(tex_pin_image *pin, vkimg_t *im) {
    if (pin && !pin->im) { pin->im = im; im->pins++; }
}
void mtl_tex_snapshot(tex_t *copy, const tex_t *source) {
    *copy = *source; if (!copy->serial) return;
    akey key; tex_storage_key(&key, copy, vk_fmt(copy->d.format));
    pthread_mutex_lock(&g_img_mu); tex_pin *p = tex_pin_get(copy->serial); p->refs++;
    tex_pin_image *i = tex_pin_key(p, &key);
    if (!i->im) { i->im = copy->pool ? nvn_hm_get(&g_by_key, &key) : nvn_hm_get(&g_by_serial, &copy->serial); if (i->im) i->im->pins++; }
    pthread_mutex_unlock(&g_img_mu);
}
void mtl_tex_pin(uint64_t serial) {
    if (!serial) return; pthread_mutex_lock(&g_img_mu);
    tex_pin_get(serial)->refs++; pthread_mutex_unlock(&g_img_mu);
}
void mtl_tex_unpin(uint64_t serial) {
    if (!serial) return; pthread_mutex_lock(&g_img_mu);
    tex_pin *p = nvn_hm_get(&g_tex_pins, &serial);
    if (p && !--p->refs) {
        nvn_hm_del(&g_tex_pins, &serial);
        for (tex_pin_image *i = p->images, *next; i; i = next) { next = i->next;
            if (i->im && !--i->im->pins && i->im->orphan) img_drop(i->im);
            free(i);
        }
        free(p);
    }
    pthread_mutex_unlock(&g_img_mu);
}
static vkimg_t *vk_tex_(const tex_t *t);
// The pool tables (gfx_vk_exec.c tbl_set) bind every registered texture ID, so they must not make a texture's first use out of its being registered: Ocarina of Time registers
// 1,475 packaged textures at a scene load (one 12288x12288 of 290 MB) and frees or remaps the storage of some before their IDs are rewritten; a table fill that looked one up
// uploaded it from pages that were gone (pkg_hash, SIGSEGV on the replay thread). 2: its image has its contents (a real bind looked it up inside a submission; vk_tex bumps
// g_tex_gen at that first use, so the tables are built again with it) and, if packaged, its storage is in a live pool chunk and mapped (mincore over the range, checked again
// when the pools change); 1: no contents yet, and the fill may give it its first use: storage in a live pool chunk, mapped, at most DM_VK_POOL_UPLOAD_MB (16) so no fill hashes
// and deswizzles a huge texture (the 290 MB one waits for a real bind); 0: the dummy. The fill bounds how many first uses it makes (tbl_set), the rest wait for the next submission
static __thread int t_pool_fill; extern uint64_t g_tex_gen; uint32_t vk_pool_gen(void);
void vk_tex_pool_fill(int on) { t_pool_fill = on; }
static int range_mapped(const uint8_t *mem, size_t len) {
    uintptr_t a = (uintptr_t)mem & ~(uintptr_t)4095, e = ((uintptr_t)mem + len + 4095) & ~(uintptr_t)4095; size_t np = (e - a) / 4096;
    static __thread unsigned char *vec; static __thread size_t cap;
    if (np > cap) { free(vec); vec = malloc(np); cap = vec ? np : 0; if (!vec) return 0; }
    return mincore((void *)a, e - a, vec) == 0;
}
int vk_tex_pool_ok(const tex_t *t) {
    if (!t || !t->mem || !t->L.size || t->magic != MAGIC('T', 'E', 'X', 0) || t->d.target == NVN_TEXTURE_TARGET_BUFFER) return 0;
    const vkfmt_t *f = vk_fmt(t->d.format); if (!f) return 0;
    akey k; tex_storage_key(&k, t, f);
    pthread_mutex_lock(&g_img_mu);
    vkimg_t *im = tex_image_find(t, &k, NULL);
    int looked = im && __atomic_load_n(&im->looked, __ATOMIC_ACQUIRE), pk = im && im->packaged, orphan = im && im->orphan; uint32_t ck = im ? im->pool_ck : 0;
    int reclaimed = !im && t->pool && g_reclaimed.n && nvn_hm_get(&g_reclaimed, &k);
    pthread_mutex_unlock(&g_img_mu);
    if (reclaimed) return 0;   // a render target whose memory a newer texture took (evict_overlaps): the dummy until it is bound or drawn to again
    if (orphan) return looked ? 2 : 0;   // a queued snapshot has native contents, not CPU storage
    if (!vk_pool_holding((uint64_t)(uintptr_t)t->mem, t->L.size)) return 0;   // CPU reads may cross chunks of the live logical pool
    uint32_t g = vk_pool_gen();
    if (looked) {
        if (!pk || ck == g) return 2;
        if (!range_mapped(t->mem, t->L.size)) return 0;
        pthread_mutex_lock(&g_img_mu); if ((t->pool ? nvn_hm_get(&g_by_key, &k) : nvn_hm_get(&g_by_serial, &t->serial)) == im) im->pool_ck = g; pthread_mutex_unlock(&g_img_mu);
        return 2;
    }
    static long cap = -1; if (cap < 0) { const char *e = getenv("DM_VK_POOL_UPLOAD_MB"); cap = (e ? atol(e) : 16) << 20; }
    if ((long)t->L.size > cap) return 0;
    return range_mapped(t->mem, t->L.size) ? 1 : 0;
}
vkimg_t *vk_tex(const tex_t *t) { extern int g_prof_on; extern uint64_t g_prof_ns[8]; struct timespec a, b; if (g_prof_on) clock_gettime(CLOCK_MONOTONIC, &a);
    vkimg_t *im = vk_tex_(t); if (g_prof_on) { clock_gettime(CLOCK_MONOTONIC, &b); g_prof_ns[1] += (uint64_t)((b.tv_sec - a.tv_sec) * 1000000000ll + (b.tv_nsec - a.tv_nsec)); } return im; }
static vkimg_t *vk_tex_(const tex_t *t) {
    if (!t || !t->serial || t->magic != MAGIC('T', 'E', 'X', 0)) return NULL;
    const vkfmt_t *f = vk_fmt(t->d.format);
    if (!f) { VKRED_ONCE(14, "texture format %#x has no Vulkan counterpart here (first case %ux%u target %d)", t->d.format, t->d.width, t->d.height, t->d.target); return NULL; }
    if (t->d.target == NVN_TEXTURE_TARGET_BUFFER) return NULL;   // buffer textures: texel buffer views of the pool (vk_tex_view)
    akey k; tex_storage_key(&k, t, f);   // every use of the same storage shares rendered contents
    int keyed = t->pool != NULL;
    pthread_mutex_lock(&g_img_mu);
    tex_pin_image *pinned = NULL;
    vkimg_t *im = tex_image_find(t, &k, &pinned);
    if (!im) {
        if (keyed) evict_overlaps(k.pool, t->off, (int64_t)t->L.size);
        if (keyed && !t_pool_fill && g_reclaimed.n && nvn_hm_get(&g_reclaimed, &k)) nvn_hm_del(&g_reclaimed, &k);
        if (vk_sub_open()) vk_img_budget_make_room(t->L.size);   // (the texture budget: the least recently drawn go once the GPU is past this submission)
        im = img_new(t, f, 0, VK_FORMAT_UNDEFINED); if (!im) { pthread_mutex_unlock(&g_img_mu); return NULL; }
        im->pool = k.pool; im->off = t->off; im->size = (int64_t)t->L.size; im->flags = t->d.flags; g_img_n++; g_img_bytes += t->L.size; g_img_vkbytes += im->vkbytes;
        if (keyed) { alias_link(im); nvn_hm_put(&g_by_key, &k, im); } else nvn_hm_put(&g_by_serial, &t->serial, im);
    }
    tex_image_pin_resolved(pinned, im);
    im->used = g_tex_frame; if (!t_pool_fill) im->used_bind = g_tex_frame; int orphan = im->orphan;
    pthread_mutex_unlock(&g_img_mu);
    // A finalized pool no longer supplies CPU bytes. A queued snapshot retains its native contents.
    if (orphan) return im;
    if (!vk_sub_open()) return __atomic_load_n(&im->looked, __ATOMIC_ACQUIRE) ? im : NULL;   // looked up outside a submission (present): nothing can be recorded, a new image has no contents yet
    if (unc_rule()) vcs_unc_drop(cpu_unc_dropped);   // taken uncompressed textures over storage the GPU wrote since: GPU textures from now on (nvn_mtl.m cpu_unc_drop)
    int first = !__atomic_exchange_n(&im->looked, 1, __ATOMIC_ACQ_REL);   // its first use is the first lookup inside a submission, not its creation (which may have been outside one)
    if (first && !t_pool_fill) { g_tex_gen++; { extern uint64_t g_tex_gen_why[4]; g_tex_gen_why[1]++; } vk_tex_diag_native(TEX_DIAG_FIRST_USE, t, im); }   // the pool tables (gfx_vk_exec.c tbl_set) bind only images with contents: a real bind's first use has them built again with this one
    // a texture's first use: written by the CPU since it was registered (profile tex_dirty_upload), it is CPU-written from now on, as at nvn_mtl.m's first resolution;
    // otherwise only a later nvn_ft_poll pass finds it, and its first draws read zeros
    int nvn_ft_written(const tex_t *);
    // a texture over a movie decoder's picture slot: the GPU converts the decoded picture into it (movie_zc_vk.c); nothing is taken from its storage (the host no longer writes it)
    // DM_VK_LIN_DIAG=1 (start-up): which path each linear texture takes at its first use (Ocarina of Time's movie planes stayed black)
    static int lin_diag = -1; if (lin_diag < 0) { const char *e = getenv("DM_VK_LIN_DIAG"); lin_diag = e && atoi(e); }
    int zc_hit = t->mem && zc_tex_refresh(im, t->mem, t->L.size);
    int lin_hit = !zc_hit && t->mem && lin_copy(im, t, f);
    if (lin_diag && first && t->mem && (linear_storage(t) || (t->d.width >= 960 && t->d.height >= 540))) { VkBuffer b_; VkDeviceSize o_;
        VKLOG("linear texture ser=%llu %ux%u nvn format %#x flags %#x stride %u size %llu mem %p pool flags %#x: %s (zc slot %s, pool buffer %s, levels %u layers %u bpb %u)",
              (unsigned long long)t->serial, t->d.width, t->d.height, t->d.format, t->d.flags, t->d.stride, (unsigned long long)t->L.size, (const void *)t->mem, t->pool ? t->pool->flags : 0,
              zc_hit ? "zero-copy slot" : lin_hit ? "lin_copy from the pool buffer" : packaged_tex(t) ? "packaged (CPU re-upload)" : nvn_ft_written(t) ? "dirty at first use" : "GPU-only image (nothing uploads it)", zc_tex_wants(t->mem, t->L.size) ? "yes" : "no",
              vk_buf_at((uint64_t)(uintptr_t)t->mem, t->L.size, &b_, &o_) ? "yes" : "NO", im->levels, im->layers, f ? f->bpb : 0); }
    if (zc_hit) { if (first && im->layout != VK_IMAGE_LAYOUT_GENERAL) vk_tex_prepare(vk_upload_cmd(), im); }
    else if (lin_hit) {}
    else if (t->mem && (packaged_tex(t) || (first && (nvn_ft_written(t) || cpu_content_tex(t, f))))) {
        im->pkg_used = g_tex_frame;
        if (!im->packaged) { im->packaged = 1; pkg_upload(im, t, f); }
        else {
            uint64_t now = nvn_dirty_now();
            if (now != im->pkg_seen && nvn_dirty_since_span(im->pkg_span, t->mem, t->L.size, im->pkg_stamp)) {   // pages written since the upload
                // The GPU wrote the image since (a compute shader's image stores, a copy, a render) and its storage still holds the bytes of the upload: the writes were around
                // it on shared pages, and an upload would put the CPU's bytes over the GPU's. Monster Hunter Rise's spot light profiles (512x1x64 R16F, written by a compute
                // shader into memory the game had zeroed) were uploaded again as zeros at presents 60 and 1079: every spot light's profile read 0 and the faces lit only by
                // those lights (the character maker, cutscenes) came out black
                if (im->gpu_w && pkg_hash(t->mem, t->L.size) == im->pkg_hash) { nvn_dirty_protect(t->mem, t->L.size); im->pkg_stamp = nvn_dirty_now(); im->pkg_released = 0;
                    static int nl; if (nl++ < 16) VKLOG("texture %ux%u nvn format %#x at %p: its pages were written, its bytes are as uploaded and the GPU wrote it: kept", t->d.width, t->d.height, t->d.format, (const void *)t->mem); }
                else pkg_upload(im, t, f);
            }
            im->pkg_seen = now;
        }
    } else if (first) vk_tex_prepare(vk_upload_cmd(), im);
    return im;
}
// Binding as an image establishes storage usage even when the NVN texture builder omitted IMAGE.
// Keep the storage-key object and all CPU/GPU ownership metadata. Only its native allocation/views change;
// prior descriptors retain the old handles until the submission fence, and subsequent lookups see the copy.
int vk_tex_storage_prepare(const tex_t *t, const view_t *view) {
    tex_t snapshot = *t; view_t selected; if (view) { selected = *view; view = &selected; }
    t = &snapshot;   // registrations are thread-local scratch; ending a pass may perform other lookups
    const vkfmt_t *f = vk_fmt(t->d.format), *vf = vk_fmt(view && view->set_format ? view->format : t->d.format);
    if (!f || !vf || (vf->vk != f->vk && !storage_view_compatible(f, vf))) {
        VKRED_ONCE(174, "storage view format %#x is incompatible with texture format %#x", view && view->set_format ? view->format : t->d.format, t->d.format); return 0;
    }
    vkimg_t *im = vk_tex(t);
    if (!im) return 0;
    VkFormatProperties vp; vkGetPhysicalDeviceFormatProperties(V.pd, vf->vk, &vp);
    // A supported compatible view uses the original bits. Only an unsupported numeric base-format view takes the float promotion.
    VkFormat pv = vf->vk == f->vk ? storage_promote(f) : VK_FORMAT_UNDEFINED;
    int numeric_promoted = pv && im->promoted && im->fmt == pv;
    if (!(vp.optimalTilingFeatures & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT) && !pv) {
        VKRED_ONCE(161, "storage binding view format %d cannot support image stores (base %d)", vf->vk, f->vk); return 0;
    }
    int native = !im->promoted && im->fmt == f->vk && (!im->fixed_fmt || vf->vk == im->fmt || vf->vk == im->fmt2) &&
        (vf->vk == im->fmt || (im->cflags & VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT));
    if ((im->usage & VK_IMAGE_USAGE_STORAGE_BIT) && (numeric_promoted || (native && (vp.optimalTilingFeatures & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT)))) return 1;
    if (im->aspect != VK_IMAGE_ASPECT_COLOR_BIT || (im->fmt != f->vk && !im->promoted)) {
        VKRED_ONCE(161, "storage binding format %d cannot support requested view %d", im->fmt, vf->vk); return 0;
    }
    VkFormatProperties fp; vkGetPhysicalDeviceFormatProperties(V.pd, im->fmt, &fp);
    // Called in a preflight for every stage before any descriptors are collected or a new render pass begins.
    // The main command buffer places the copy after earlier rendering; the upload buffer would copy too soon.
    vk_end_pass_land();
    vkimg_t *next = img_new(t, f, VK_IMAGE_USAGE_STORAGE_BIT, vf->vk);
    if (!next) return 0;
    VkCommandBuffer cb = vk_cmd();
    vk_tex_prepare(cb, im); vk_tex_prepare(cb, next); vk_img_barrier_all(cb);
    int blit = next->fmt != im->fmt; VkFormatProperties np; vkGetPhysicalDeviceFormatProperties(V.pd, next->fmt, &np);
    if (blit && (!(fp.optimalTilingFeatures & VK_FORMAT_FEATURE_BLIT_SRC_BIT) || !(np.optimalTilingFeatures & VK_FORMAT_FEATURE_BLIT_DST_BIT) || im->samples > 1)) { blit = -1; VKRED_ONCE(173, "image promoted for image stores (format %d → %d): its contents could not be blitted over, it starts from zeros", im->fmt, next->fmt); }
    for (uint32_t level = 0; level < im->levels && blit >= 0; level++) {
        uint32_t w = im->w >> level, h = im->h >> level, d = im->d >> level; w = w ? w : 1; h = h ? h : 1; d = d ? d : 1;
        if (blit) { VkImageBlit r = {{im->aspect, level, 0, im->layers}, {{0, 0, 0}, {(int32_t)w, (int32_t)h, (int32_t)d}}, {im->aspect, level, 0, im->layers}, {{0, 0, 0}, {(int32_t)w, (int32_t)h, (int32_t)d}}};
            vkCmdBlitImage(cb, im->img, VK_IMAGE_LAYOUT_GENERAL, next->img, VK_IMAGE_LAYOUT_GENERAL, 1, &r, VK_FILTER_NEAREST); continue; }
        VkImageCopy copy = {{im->aspect, level, 0, im->layers}, {0, 0, 0},
                           {im->aspect, level, 0, im->layers}, {0, 0, 0}, {w, h, d}};
        vkCmdCopyImage(cb, im->img, VK_IMAGE_LAYOUT_GENERAL, next->img, VK_IMAGE_LAYOUT_GENERAL, 1, &copy);
    }
    vk_img_barrier_all(cb);
    vk_tex_view_drop_all(im);
    if (im->full) { vk_defer_free_view(im->full); im->full = VK_NULL_HANDLE; }
    if (im->sub) vk_defer_free_image_sub(im->img, im->sub); else vk_defer_free_image(im->img, im->mem);
    g_img_vkbytes -= im->vkbytes; g_img_vkbytes += next->vkbytes;
    im->img = next->img; im->mem = next->mem; im->sub = next->sub; im->usage = next->usage; im->cflags = next->cflags;
    im->layout = next->layout; im->fixed_fmt = next->fixed_fmt; im->fmt2 = next->fmt2; im->vkbytes = next->vkbytes; im->fmt = next->fmt; im->promoted = next->promoted;
    free(next);
    static int noted; if (noted++ < 16) VKLOG("storage binding promoted %ux%u format %d%s (texture serial %llu, nvn format %#x): preserved %u levels, %u layers; usage %#x",
        im->w, im->h, im->fmt, im->promoted ? " (format promoted for image stores)" : "", (unsigned long long)t->serial, t->d.format, im->levels, im->layers, im->usage);
    return 1;
}
// Packaged textures give up their write protection (nvn_dirty_release; drawn again, they are uploaded and protected again) when not drawn for PKG_IDLE presents, and the
// least recently drawn ones beyond that while the protected runs, each splitting its pool's mapping, are more than PKG_RUNS (textures stay protected until the game writes them).
// Textures drawn in the last PKG_KEEP presents keep theirs (released, they would be uploaded again at once). Swept every 300 presents, with the pools whose mapping the
// protection has split most mapped again in one piece (nvn_dirty_defrag: Xenoblade Chronicles 3 went from 43,000 mappings after 25 minutes of walking, toward the kernel's
// limit of 65,530, to about 17,500)
enum { PKG_IDLE = 1800, PKG_KEEP = 60, PKG_RUNS = 8000 };   // about 60 s and 2 s at 30 fps; at most about 16,000 mappings for the protection
uint64_t g_tex_frame;
typedef struct { uint64_t used; vkimg_t *im; } pkg_age;
static int pkg_age_cmp(const void *a, const void *b) { uint64_t x = ((const pkg_age *)a)->used, y = ((const pkg_age *)b)->used; return x < y ? -1 : x > y; }
void vk_tex_sweep(void) {
    vk_tex_diag_poll();
    if (++g_tex_frame % 300) return;
    uint64_t idle = 0, lru = 0, before = nvn_dirty_released();
    pthread_mutex_lock(&g_img_mu);
    pkg_age *v = NULL; size_t nv = 0, cap = 0;
    for (int m = 0; m < 2; m++) {
        nvn_hmap *h = m ? &g_by_serial : &g_by_key;
        for (uint32_t i = 0; i < h->cap; i++) {
            vkimg_t *im = h->vals[i];
            if (!im || !im->packaged || im->pkg_released || !im->pkg_mem) continue;
            if (g_tex_frame - im->pkg_used >= PKG_IDLE) { nvn_dirty_release(im->pkg_mem, im->pkg_len); im->pkg_released = 1; idle++; continue; }
            if (g_tex_frame - im->pkg_used < PKG_KEEP) continue;
            if (nv == cap) { cap = cap ? cap * 2 : 4096; pkg_age *nw = realloc(v, cap * sizeof *v); if (!nw) break; v = nw; }
            v[nv++] = (pkg_age){im->pkg_used, im};
        }
    }
    { int vk_pool_remap_ro(void *mem, size_t len); int nvn_dirty_defrag(int (*)(void *, size_t)); struct timespec t0, t1; clock_gettime(CLOCK_MONOTONIC, &t0); int d = nvn_dirty_defrag(vk_pool_remap_ro); clock_gettime(CLOCK_MONOTONIC, &t1); static uint64_t nd;
        if (d && (nd += d) <= 8) VKLOG("dirty-page tracking: %d pool(s) mapped again in one piece in %.1f ms (%llu so far)", d, (t1.tv_sec - t0.tv_sec) * 1e3 + (t1.tv_nsec - t0.tv_nsec) / 1e6, (unsigned long long)nd); }
    int runs = nvn_dirty_runs();
    if (runs > PKG_RUNS && nv) {   // the least recently drawn first, in batches, until the runs are well under the budget
        qsort(v, nv, sizeof *v, pkg_age_cmp);
        for (size_t i = 0; i < nv && runs > PKG_RUNS * 3 / 4; ) {
            for (size_t e = i + 256; i < nv && i < e; i++) { nvn_dirty_release(v[i].im->pkg_mem, v[i].im->pkg_len); v[i].im->pkg_released = 1; lru++; }
            runs = nvn_dirty_runs();
        }
    }
    pthread_mutex_unlock(&g_img_mu);
    free(v);
    static uint64_t total; total += idle + lru;
    if (g_tex_frame % 3000 == 0) { uint64_t bu, be, bb, bs, pn, pb, pby, pbb; vk_img_budget_stats(&bu, &be, &bb, &bs); vk_pool_stats(&pn, &pb, &pby, &pbb);
        VKLOG("images: %llu live, %.0f MB of textures (%.0f MB of device memory, budget %.0f MB: %llu evicted, %.0f MB, in %llu sweeps), %llu evicted so far (a newer texture over their memory; %llu of them render targets only the pool tables still looked up); memory pools: %llu chunks (%.2f GB), %llu backed (%.2f GB)",
              (unsigned long long)g_img_n, g_img_bytes / 1048576.0, g_img_vkbytes / 1048576.0, bu / 1048576.0, (unsigned long long)be, bb / 1048576.0, (unsigned long long)bs, (unsigned long long)g_img_evicted, (unsigned long long)g_img_reclaimed,
              (unsigned long long)pn, pby / 1073741824.0, (unsigned long long)pb, pbb / 1073741824.0); }
    if ((idle || lru) && (g_tex_frame % 3000 == 0 || lru)) VKLOG("packaged textures: %llu not drawn for %d presents and %llu least recently drawn released their write protection (%llu pages; %llu so far); %d protected runs",
                    (unsigned long long)idle, PKG_IDLE, (unsigned long long)lru, (unsigned long long)(nvn_dirty_released() - before), (unsigned long long)total, runs);
}
// The keys are gathered first, 64 a pass, then deleted (a delete rehashes the table): a rescan from the start after each delete went over the whole table once per image,
// under the lock the replay's texture lookups take, for each of the pools Pokémon Scarlet drops while a capture loads
void vk_tex_drop_pool(const void *pool) {   // the pool is finalized: its textures' images go once the GPU is done with them
    pthread_mutex_lock(&g_img_mu);
    for (int n = 64; n == 64;) {
        akey ks[64]; n = 0;
        for (uint32_t i = 0; i < g_by_key.cap && n < 64; i++) {
            if (!g_by_key.vals[i]) continue;
            memcpy(&ks[n], g_by_key.keys + (size_t)i * g_by_key.klen, sizeof ks[n]);
            if (ks[n].pool == (uint64_t)(uintptr_t)pool) n++;
        }
        for (int j = 0; j < n; j++) { vkimg_t *im = nvn_hm_get(&g_by_key, &ks[j]); if (!im) continue; nvn_hm_del(&g_by_key, &ks[j]); img_drop(im); }
    }
    pthread_mutex_unlock(&g_img_mu);
}
// CPU-written textures (WriteTexels / a dirty first use): packaged from now on
static nvn_hmap g_cpu_written = {.klen = sizeof(uint64_t)}; static pthread_mutex_t g_cw_mu = PTHREAD_MUTEX_INITIALIZER;
void mtl_tex_cpu_written(const tex_t *t) { if (!t->serial) return; pthread_mutex_lock(&g_cw_mu); nvn_hm_put(&g_cpu_written, &t->serial, (void *)1); pthread_mutex_unlock(&g_cw_mu); }
// Linear textures (LINEAR) in a pool are their storage on the Switch: the CPU writes the pool memory and the GPU reads it (Metal's tex_linear makes them on the pool buffer).
// Hades II rasterizes its glyphs on the CPU into a 1024x1216 R8 IMAGE|LINEAR atlas (pool CPU_UNCACHED|GPU_CACHED, no flush, no WriteTexels) and decodes its Bink movies
// into linear planes: as GPU-only images of zeros all text, the title art and the movies were blank. A Vulkan image cannot be bound to the pool's memory (a dedicated
// AHardwareBuffer import, bound to its buffer), so each upload command buffer that a submission's draws need it in copies it from the pool buffer on the GPU (lin_copy).
// The packaged path (CPU upload, re-upload on written pages) is the fallback; as the only path the page faults of the CPU's writes cost Hades II about 120 a frame and
// 2 million while loading. Not LINEAR_RENDER_TARGET: the GPU draws into those, and a copy from storage would replace what it drew
static int linear_tex(const tex_t *t) { return t->pool && (t->d.flags & (NVN_TEXTURE_FLAGS_LINEAR | NVN_TEXTURE_FLAGS_LINEAR_RENDER_TARGET)) == NVN_TEXTURE_FLAGS_LINEAR; }
// lin_copy takes LINEAR_RENDER_TARGET textures too: Super Mario Bros. Wonder's CPU writes its sky's four 64x1 RGBA16F gradients straight into such textures (NVN flags 0x40)
// that nothing draws into, and as GPU-only images of zeros the sky drew diagonal bands of saturated colour over every level. What the GPU draws into one (gpu_w) stays, as for a
// cleared or copied linear texture, until the CPU writes its storage again. The packaged path (linear_tex) still leaves them out: a re-upload there would replace a drawing
static int linear_storage(const tex_t *t) { return t->pool && (t->d.flags & (NVN_TEXTURE_FLAGS_LINEAR | NVN_TEXTURE_FLAGS_LINEAR_RENDER_TARGET)); }
extern uint64_t g_up_gen;
static int lin_copy(vkimg_t *im, const tex_t *t, const vkfmt_t *f) {   // 1 = the image holds its storage as of this upload buffer
    if (im->promoted || !linear_storage(t) || im->levels != 1 || im->d != 1 || im->layers != 1 || f->bw > 1 || f->bh > 1 || !f->bpb || im->aspect != VK_IMAGE_ASPECT_COLOR_BIT) return 0;
    size_t stride = t->d.stride ? t->d.stride : (size_t)im->w * f->bpb; VkBuffer b; VkDeviceSize o;
    if (stride % f->bpb || stride < (size_t)im->w * f->bpb || !vk_buf_at((uint64_t)(uintptr_t)t->mem, (uint64_t)stride * (im->h - 1) + (uint64_t)im->w * f->bpb, &b, &o) || o % f->bpb) return 0;
    // the GPU wrote it (a clear, a copy, a storage write): it holds that until the CPU writes its storage again, which its pages, set read-only, tell
    size_t len = stride * (im->h - 1) + (size_t)im->w * f->bpb;
    if (im->gpu_w && !im->lin_mem) { nvn_dirty_protect(t->mem, len); im->lin_stamp = nvn_dirty_now(); im->lin_span = nvn_dirty_span(t->mem, len); im->lin_mem = t->mem; im->lin_len = len; return 1; }
    if (im->gpu_w) { if (!nvn_dirty_since_span(im->lin_span, t->mem, len, im->lin_stamp)) return 1; nvn_dirty_release(im->lin_mem, im->lin_len); im->lin_mem = NULL; im->gpu_w = 0; im->lin_gen = 0; }
    VkCommandBuffer cb = vk_upload_cmd();
    if (im->lin_gen == g_up_gen) return 1;
    im->lin_gen = g_up_gen;
    vk_img_barrier_all(cb);   // after the reads of earlier submissions
    if (im->layout != VK_IMAGE_LAYOUT_GENERAL) vk_tex_prepare(cb, im);
    VkBufferImageCopy r = {o, (uint32_t)(stride / f->bpb), 0, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, {0, 0, 0}, {im->w, im->h, 1}};
    vkCmdCopyBufferToImage(cb, b, im->img, VK_IMAGE_LAYOUT_GENERAL, 1, &r);
    return 1;
}
// A texture the CPU filled before its first use, as nxnative's tex_cpu_content (nvn_mtl.m) takes it: a sampled block-compressed texture (BC and ASTC are not colour-renderable, so
// data in its storage at first use is the CPU's), or an uncompressed 3D texture with none of the flags below (upstream 6a16b7d3, #1050: Xenoblade Chronicles 3's particle steering
// fields, 10x10x10 RGBA8 and RGBA16F, flags 0; as GPU textures of zeros every send-off particle moved along (-1, -1, -1)). It is made CPU-written: uploaded now, again when its pages
// change. The compressed case was missing here: Xenoblade Chronicles 3's particle sprites (the husks' 128x128 BC7 sRGB, flags 0) were images of zeros, so the particle
// pixel shader read alpha 0 and discarded every pixel (no red husk particles, no send-off ones). A pool the CPU cannot access is filled from its storage once in the same way, as the
// Mac's tex_init_fill does (the game cannot write it again). DM_NVN_STORAGE_FIRST_USE=0 turns this off
static int mem_any(const uint8_t *p, uint64_t n) {   // whether any byte is non-zero (stops at the first)
    uint64_t i = 0, v;
    for (; i + 8 <= n; i += 8) { memcpy(&v, p + i, 8); if (v) return 1; }
    for (; i < n; i++) if (p[i]) return 1;
    return 0;
}
// Uncompressed 2D textures (port of nvn_mtl.m tex_cpu_content / mtl_tex_cpu_rewritten, upstream #1390 / e2b76d4e via #1604, #1558): NVN has no flag that tells one from a render
// target, but storage the GPU has not written holds only what the CPU wrote, so there such a texture is taken from its storage at first use too, and once the GPU writes that storage
// it becomes a GPU texture again (gfx_vk_cpu_storage.c). Ocarina of Time's terrain tiles' height (260x260 R32F) and blend / hole (RGBA8) maps, which the CPU fills in the pool its
// normal maps are drawn into, were images of zeros (holes, wrong materials); Mario Kart World's terrain patches likewise. On the Mac this holds for every game; here it is on for the
// profiles that ask the host to follow CPU texture writes (tex_dirty_upload or tex_flush_upload), so other games (Donkey Kong Bananza, Welcome Tour) keep the rule they were tuned
// with, as the GPU-written spans it needs protect render-target pages. DM_NVN_STORAGE_UNC=1 / 0 (start-up) turns it on / off for any game
static int unc_rule(void) {
    static int on = -1;
    if (on < 0) { const char *e = getenv("DM_NVN_STORAGE_UNC"); on = e && *e ? *e != '0' : (DM_GAME_TEX_DIRTY_UPLOAD || DM_GAME_TEX_FLUSH_UPLOAD);
                  if (on) VKLOG("uncompressed textures over storage the GPU has not written are taken from their storage (tex_dirty_upload / tex_flush_upload, DM_NVN_STORAGE_UNC)"); }
    return on;
}
static int tex_unc(const tex_t *t, const vkfmt_t *f) { return f && !f->compressed && t->d.target != NVN_TEXTURE_TARGET_3D; }   // the uncompressed 2D kind, which the GPU can render into
// Whether the texture's storage can be what the host takes its content from (nvn_mtl.m tex_storage_ok)
static int tex_storage_ok(const tex_t *t, const vkfmt_t *f, int unc) {
    if (!t->mem || !t->pool || t->has_layout || t->d.samples > 1 || t->d.target == NVN_TEXTURE_TARGET_BUFFER) return 0;
    if (unc && (!f || f->depth || f->stencil || (t->d.flags & (NVN_TEXTURE_FLAGS_DISPLAY | NVN_TEXTURE_FLAGS_VIDEO_DECODE)))) return 0;   // not what only the GPU or a decoder fills
    if (t->pool->flags & (NVN_MEMORY_POOL_FLAGS_CPU_NO_ACCESS | NVN_MEMORY_POOL_FLAGS_VIRTUAL)) return 0;
    return !((t->d.flags & (NVN_TEXTURE_FLAGS_COMPRESSIBLE | NVN_TEXTURE_FLAGS_LINEAR | NVN_TEXTURE_FLAGS_LINEAR_RENDER_TARGET | NVN_TEXTURE_FLAGS_SPARSE))
             || ((t->d.flags & NVN_TEXTURE_FLAGS_IMAGE) && !(f && f->compressed)));
}
static void cpu_unc_take(const tex_t *t) { vcs_unc_take(t->serial, (uintptr_t)t->mem, (uintptr_t)t->mem + t->L.size); }
static int cpu_content_unc(const tex_t *t, const vkfmt_t *f) {   // first use, uncompressed 2D (above)
    if (!unc_rule() || !tex_storage_ok(t, f, 1)) return 0;
    if (vcs_owned((uintptr_t)t->mem, (uintptr_t)t->mem + t->L.size) || !mem_any(t->mem, t->L.size)) return 0;
    static int n; if (n++ < 8) VKLOG("texture %ux%u nvn format %#x at %p (pool flags %#x): uncompressed, written by the CPU before its first use in storage the GPU has not written: uploaded from its storage",
                                     t->d.width, t->d.height, t->d.format, (const void *)t->mem, t->pool->flags);
    mtl_tex_cpu_written(t); cpu_unc_take(t);
    return 1;
}
static int cpu_content_tex(const tex_t *t, const vkfmt_t *f) {
    static int on = -1; if (on < 0) { const char *e = getenv("DM_NVN_STORAGE_FIRST_USE"); on = !e || atoi(e) != 0; }
    if (on && tex_unc(t, f) && t->d.target != 2) return cpu_content_unc(t, f);
    if (!on || !t->pool || t->has_layout || !(f->compressed || t->d.target == 2) || t->d.samples > 1) return 0;
    if (t->pool->flags & NVN_MEMORY_POOL_FLAGS_VIRTUAL) return 0;
    if (t->d.flags & (NVN_TEXTURE_FLAGS_IMAGE | NVN_TEXTURE_FLAGS_COMPRESSIBLE | NVN_TEXTURE_FLAGS_LINEAR | NVN_TEXTURE_FLAGS_LINEAR_RENDER_TARGET | NVN_TEXTURE_FLAGS_SPARSE)) return 0;
    if (!mem_any(t->mem, t->L.size)) return 0;
    static int n; if (n++ < 8) VKLOG("texture %ux%ux%u nvn format %#x at %p (pool flags %#x): the CPU filled it before its first use: uploaded from its storage", t->d.width, t->d.height, t->d.depth, t->d.format, (const void *)t->mem, t->pool->flags);
    mtl_tex_cpu_written(t);
    return 1;
}
static int packaged_tex(const tex_t *t) {
    if (t->has_layout || linear_tex(t)) return 1;
    pthread_mutex_lock(&g_cw_mu); int r = nvn_hm_get(&g_cpu_written, &t->serial) != NULL; pthread_mutex_unlock(&g_cw_mu);
    return r;
}
// A GPU write into a texture's storage is recorded (render target, clear, copy, shader image): uncompressed textures over it are no longer taken from storage (nvn_mtl.m)
void mtl_tex_gpu_written(const tex_t *t) {
    if (!unc_rule() || !t || !t->mem || !t->L.size || !t->pool || (t->pool->flags & (NVN_MEMORY_POOL_FLAGS_CPU_NO_ACCESS | NVN_MEMORY_POOL_FLAGS_VIRTUAL))) return;
    vcs_gpu_written((uintptr_t)t->mem, (uintptr_t)t->mem + t->L.size);
}
static void cpu_unc_dropped(uint64_t serial) {   // vcs_unc_drop: the GPU wrote its storage; it keeps the content it has and CPU writes are no longer uploaded
    pthread_mutex_lock(&g_cw_mu); nvn_hm_del(&g_cpu_written, &serial); pthread_mutex_unlock(&g_cw_mu);
    static int nl; if (nl++ < 8) VKLOG("storage the GPU writes: an uncompressed texture taken from it is a GPU texture from now on (serial %llu)", (unsigned long long)serial);
}
// nvn_ft_poll (hle_nvn.c, tex_dirty_upload): the CPU wrote the pages of a registered texture after its first use (nvn_mtl.m mtl_tex_cpu_rewritten). A texture whose storage can hold
// its content is taken from it; an uncompressed 2D one not while the GPU has written that storage and the CPU has not rewritten it since (MTL_REWRITE_GPU_NOW: asked again at the
// CPU's next write). Any other texture is left to the GPU when the GPU can render into it (uncompressed), else uploaded
int mtl_tex_cpu_rewritten(const tex_t *t) {
    const vkfmt_t *f = vk_fmt(t->d.format); int unc = tex_unc(t, f);
    if (tex_storage_ok(t, f, unc)) {
        if (unc && vcs_owned((uintptr_t)t->mem, (uintptr_t)t->mem + t->L.size)) return MTL_REWRITE_GPU_NOW;
        mtl_tex_cpu_written(t);
        if (unc) cpu_unc_take(t);
        return MTL_REWRITE_UPLOADED;
    }
    // nvn_mtl.m asks the host texture's usage (render target or shader write). Here the question is answered by what the GPU has done: storage the GPU wrote (vcs: a render, a clear,
    // a copy, an image store) keeps the texture the GPU's until the CPU rewrites it; storage the GPU never wrote holds only what the CPU put there, whatever the texture's flags.
    // Ocarina of Time's movie planes (1920x1080 R8 + 960x540 RG8, NVN flags COMPRESSIBLE, so outside tex_storage_ok) are filled by nn::codec's CPU reads of the decoded pictures and
    // never rendered into: as "GPU-owned" they stayed black
    if (!(f && f->compressed) && vcs_owned((uintptr_t)t->mem, (uintptr_t)t->mem + t->L.size)) return MTL_REWRITE_GPU_NOW;
    mtl_tex_cpu_written(t);
    return MTL_REWRITE_UPLOADED;
}
void vk_cpu_storage_gone(uint64_t lo, uint64_t hi) { vcs_range_gone((uintptr_t)lo, (uintptr_t)hi); }   // a pool chunk goes (gfx_vk_core.c mtl_pool_del)
void mtl_tex_init(const tex_t *t) { (void)t; }
void mtl_tex_release(tex_t *t) { (void)t; }   // images belong to their storage key, dropped with the pool
int mtl_tex_gpu_owned(uint64_t serial) { (void)serial; return 0; }

// ---------------- views ----------------
typedef struct vview { VkImageView v; VkImageViewType type; VkFormat fmt; uint32_t lv0, nlv, ly0, nly; VkComponentMapping swz; VkImageAspectFlags aspect; int attach; struct vview *next; } vview;   // attach: an attachment view (its usage differs from a sampling view's)
static nvn_hmap g_views = {.klen = sizeof(vkimg_t *)};   // vkimg_t* → vview list
static pthread_mutex_t g_view_mu = PTHREAD_MUTEX_INITIALIZER;
extern uint64_t g_tex_gen;
// The images whose views went, in order (gfx_vk_exec.c tbl_ring rewrites only the pool table entries that held one). vk_droplog_take: those after position from, -1 when
// more went than the log or the caller holds; *to is the log's end either way
enum { DROPLOG_N = 1024 }; static const void *g_droplog[DROPLOG_N]; static uint64_t g_droplog_seq;
uint64_t vk_droplog_seq(void) { return __atomic_load_n(&g_droplog_seq, __ATOMIC_ACQUIRE); }
int vk_droplog_take(uint64_t from, const void **out, int max, uint64_t *to) {
    pthread_mutex_lock(&g_view_mu); uint64_t end = g_droplog_seq; int n = -1;
    if (end - from <= DROPLOG_N && end - from <= (uint64_t)max) { n = 0; for (uint64_t i = from; i < end; i++) out[n++] = g_droplog[i % DROPLOG_N]; }
    pthread_mutex_unlock(&g_view_mu); *to = end; return n;
}
static void vk_tex_view_drop_all(vkimg_t *im) {
    pthread_mutex_lock(&g_view_mu);
    g_droplog[g_droplog_seq % DROPLOG_N] = im; __atomic_store_n(&g_droplog_seq, g_droplog_seq + 1, __ATOMIC_RELEASE);
    vview *v = nvn_hm_get(&g_views, &im); g_tex_gen++; { extern uint64_t g_tex_gen_why[4]; g_tex_gen_why[2]++; } vk_tex_diag_native(TEX_DIAG_VIEW_DROP, NULL, im);
    while (v) { vview *n = v->next; vk_defer_free_view(v->v); free(v); v = n; }
    nvn_hm_del(&g_views, &im);
    pthread_mutex_unlock(&g_view_mu);
}
static VkComponentSwizzle swz_comp(int nvn) {   // NVN swizzle enum: 0 zero 1 one 2 R 3 G 4 B 5 A
    switch (nvn) { case 0: return VK_COMPONENT_SWIZZLE_ZERO; case 1: return VK_COMPONENT_SWIZZLE_ONE; case 2: return VK_COMPONENT_SWIZZLE_R; case 3: return VK_COMPONENT_SWIZZLE_G; case 4: return VK_COMPONENT_SWIZZLE_B; default: return VK_COMPONENT_SWIZZLE_A; }
}
static VkComponentSwizzle apply(VkComponentSwizzle outer, const VkComponentMapping *inner, VkComponentSwizzle self) {   // outer picks a channel of what inner produces
    if (outer == VK_COMPONENT_SWIZZLE_IDENTITY) outer = self;
    switch (outer) {
    case VK_COMPONENT_SWIZZLE_R: return inner->r == VK_COMPONENT_SWIZZLE_IDENTITY ? VK_COMPONENT_SWIZZLE_R : inner->r;
    case VK_COMPONENT_SWIZZLE_G: return inner->g == VK_COMPONENT_SWIZZLE_IDENTITY ? VK_COMPONENT_SWIZZLE_G : inner->g;
    case VK_COMPONENT_SWIZZLE_B: return inner->b == VK_COMPONENT_SWIZZLE_IDENTITY ? VK_COMPONENT_SWIZZLE_B : inner->b;
    case VK_COMPONENT_SWIZZLE_A: return inner->a == VK_COMPONENT_SWIZZLE_IDENTITY ? VK_COMPONENT_SWIZZLE_A : inner->a;
    default: return outer;
    }
}
static VkComponentMapping sample_swizzle(const VkComponentMapping *format, const tex_t *t, const view_t *v, int attachment) {
    if (attachment) return (VkComponentMapping){0};
    // NVN view selectors override texture selectors; both refer to the NVN
    // format's channels. Applying the texture first composes two selectors
    // and can turn a requested mask channel into a different component.
    int sw[4];
    if (v && v->set_swizzle) for (int c=0;c<4;c++) sw[c]=v->swz[c];
    else if (t->set_swz) for (int c=0;c<4;c++) sw[c]=(t->swz>>(3*c))&7;
    else return *format;
    return (VkComponentMapping){
        apply(swz_comp(sw[0]),format,VK_COMPONENT_SWIZZLE_R),
        apply(swz_comp(sw[1]),format,VK_COMPONENT_SWIZZLE_G),
        apply(swz_comp(sw[2]),format,VK_COMPONENT_SWIZZLE_B),
        apply(swz_comp(sw[3]),format,VK_COMPONENT_SWIZZLE_A)};
}
// Passive diagnostic: first sixteen distinct normally selected views of one native size.
// Called with g_view_mu held; it neither resolves resources nor changes cache behavior.
static int view_log_size_parse(const char *s, uint32_t *w, uint32_t *h) {
    uint32_t v[2]={0};
    if(!s||!*s)return 0;
    for(int i=0;i<2;i++){
        if(*s<'0'||*s>'9')return 0;
        while(*s>='0'&&*s<='9'){
            unsigned d=(unsigned)(*s++-'0');
            if(v[i]>(UINT32_MAX-d)/10u)return 0;
            v[i]=v[i]*10u+d;
        }
        if(!v[i])return 0;
        if(i==0&&*s++!='x')return 0;
    }
    if(*s)return 0;
    *w=v[0];*h=v[1];return 1;
}
static void view_log_selected(const tex_t *t,const view_t *v,const vkimg_t *im,
                              int dim,int arrayed,int kind,const vview *native,int cached) {
    static int init;static uint32_t width,height;
    static VkImageView seen[16];static unsigned nseen;
    if(!init){init=1;view_log_size_parse(getenv("DM_VK_VIEW_LOG_SIZE"),&width,&height);}
    if(!width||nseen>=16||im->w!=width||im->h!=height)return;
    for(unsigned i=0;i<nseen;i++)if(seen[i]==native->v)return;
    seen[nseen++]=native->v;
    unsigned flags=v?((unsigned)!!v->set_format|((unsigned)!!v->set_target<<1)|((unsigned)!!v->set_levels<<2)|
        ((unsigned)!!v->set_layers<<3)|((unsigned)!!v->set_swizzle<<4)|((unsigned)!!v->set_dsmode<<5)):0;
    VKLOG("view selected: cached=%d serial=%llu pool=%p off=%#llx texture_target=%d image=%p extent=%ux%ux%u native_format=%u native_type=%u native_target=%d native_layers=%u native_levels=%u samples=%u request_dim=%d request_arrayed=%d view_kind=%d view=%p type=%u format=%u mip=%u+%u layers=%u+%u aspect=%#x view_flags=%#x requested_target=%d requested_format=%d requested_mip=%u+%u requested_layers=%u+%u",
        cached,(unsigned long long)t->serial,(void*)t->pool,(unsigned long long)t->off,t->d.target,
        (void*)im->img,im->w,im->h,im->d,(unsigned)im->fmt,(unsigned)im->vtype,im->target,im->layers,im->levels,im->samples,
        dim,arrayed,kind,(void*)native->v,(unsigned)native->type,(unsigned)native->fmt,
        native->lv0,native->nlv,native->ly0,native->nly,(unsigned)native->aspect,flags,
        v?v->target:0,v?v->format:0,v?v->base_level:0,v?v->nlevels:0,v?v->min_layer:0,v?v->nlayers:0);
}
__thread VkFormat vk_view_fmt; __thread uint32_t vk_view_layers;
VkImageView vk_tex_view(const tex_t *t, const view_t *vw, int dim, int arrayed, int view_kind, vkimg_t **img_out) {
    int as_attachment = view_kind == 1, storage = view_kind == 2;
    vkimg_t *im = vk_tex(t); if (img_out) *img_out = im;
    if (!im) return VK_NULL_HANDLE;
    if (storage && !(im->usage & VK_IMAGE_USAGE_STORAGE_BIT)) { VKRED_ONCE(162, "storage image used without successful preflight"); return VK_NULL_HANDLE; }
    if (as_attachment) { im->wseq = ++g_wseq; im->gpu_w = 1; } else if (im->novl) alias_check(im);
    vk_watch(im, as_attachment ? "attachment" : "sampled or image");
    if (!as_attachment) { void vk_texdump_note(vkimg_t *im); vk_texdump_note(im); }
    const vkfmt_t *f = vk_fmt(vw && vw->set_format ? vw->format : t->d.format); if (!f) f = vk_fmt(t->d.format);
    uint32_t lv0 = vw && vw->set_levels ? vw->base_level : 0, nlv = vw && vw->set_levels ? vw->nlevels : im->levels - lv0;
    if (lv0 >= im->levels) lv0 = im->levels - 1; if (!nlv || lv0 + nlv > im->levels) nlv = im->levels - lv0;
    // a 3D image drawn into: the view's layers are depth slices of its level (the image is 2D-array compatible). Xenoblade Chronicles 3 renders its 16x16x16 colour grading
    // LUT one slice per pass through 2D views at layers 0-15; clamped to the image's one layer, every slice went to slice 0, and the graded title sky came out flat blue
    uint32_t nlay = as_attachment && im->vtype == VK_IMAGE_VIEW_TYPE_3D ? (im->d >> lv0 ? im->d >> lv0 : 1) : im->layers;
    uint32_t ly0 = vw && vw->set_layers ? vw->min_layer : 0, nly = vw && vw->set_layers ? vw->nlayers : nlay - ly0;
    if (ly0 >= nlay) ly0 = nlay - 1; if (!nly || ly0 + nly > nlay) nly = nlay - ly0;
    int target = vw && vw->set_target ? vw->target : im->target;
    VkImageViewType type = vtype_of(target);
    // what the shader declares wins over the object's target (a 2D render target sampled as an array, a cube face read as 2D)
    if (dim != 0xff && !as_attachment) {
        VkImageViewType want = dim == 0 ? (arrayed ? VK_IMAGE_VIEW_TYPE_1D_ARRAY : VK_IMAGE_VIEW_TYPE_1D) : dim == 1 ? (arrayed ? VK_IMAGE_VIEW_TYPE_2D_ARRAY : VK_IMAGE_VIEW_TYPE_2D) :
                               dim == 2 ? VK_IMAGE_VIEW_TYPE_3D : dim == 3 ? (arrayed ? VK_IMAGE_VIEW_TYPE_CUBE_ARRAY : VK_IMAGE_VIEW_TYPE_CUBE) : type;
        if (want == VK_IMAGE_VIEW_TYPE_2D && type == VK_IMAGE_VIEW_TYPE_2D_ARRAY) nly = 1;
        if ((want == VK_IMAGE_VIEW_TYPE_CUBE || want == VK_IMAGE_VIEW_TYPE_CUBE_ARRAY) && (im->layers < 6 || nly % 6)) want = type;   // not a cube: keep it as it is
        if (want == VK_IMAGE_VIEW_TYPE_CUBE) nly = 6;
        if (want == VK_IMAGE_VIEW_TYPE_3D && im->d == 1) want = type;
        if ((want == VK_IMAGE_VIEW_TYPE_1D || want == VK_IMAGE_VIEW_TYPE_1D_ARRAY) != (type == VK_IMAGE_VIEW_TYPE_1D || type == VK_IMAGE_VIEW_TYPE_1D_ARRAY)) want = type;
        type = want;
    }
    if (as_attachment) { type = im->layers > 1 && nly > 1 ? VK_IMAGE_VIEW_TYPE_2D_ARRAY : (im->vtype == VK_IMAGE_VIEW_TYPE_3D ? VK_IMAGE_VIEW_TYPE_2D_ARRAY : VK_IMAGE_VIEW_TYPE_2D); nlv = 1; }
    if (storage) nlv = 1;   // a storage image view addresses one mip, independent of its sampled-view range
    if (type == VK_IMAGE_VIEW_TYPE_2D || type == VK_IMAGE_VIEW_TYPE_1D || type == VK_IMAGE_VIEW_TYPE_3D) nly = 1;
    VkImageAspectFlags aspect = im->aspect;
    if (aspect == (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT) && !as_attachment)
        aspect = (!f->depth || t->stencil || (vw && vw->set_dsmode && vw->dsmode == 1)) ? VK_IMAGE_ASPECT_STENCIL_BIT : VK_IMAGE_ASPECT_DEPTH_BIT;   // (STENCIL8 alone may be stored combined)
    VkComponentMapping swz = sample_swizzle(&f->swz,t,vw,as_attachment || storage);
    VkFormat fmt = im->promoted ? im->fmt : f->vk;   // (a promoted image: its own float format, whatever the view asks; the formats' classes differ)
    const vkfmt_t *imf = vk_fmt(im->nvn_format);   // (a BC format decoded on the CPU counts as compressed: its view keeps the decoded image's format)
    if (im->promoted && (!imf || f->vk != imf->vk)) {
        VKRED_ONCE(175, "a promoted numeric image cannot supply a bit-reinterpreted view (base %d, requested %d)", imf ? imf->vk : 0, f->vk); return VK_NULL_HANDLE;
    }
    if ((im->aspect & ~VK_IMAGE_ASPECT_COLOR_BIT) || (f->compressed || f->dec) != (imf->compressed || imf->dec) || (f->dec && f->dbpp != imf->dbpp)) fmt = im->fmt;   // depth views and block-size changes keep the image's format
    if (im->fixed_fmt && fmt != im->fmt && fmt != srgb_twin(im->fmt)) { fmt = im->fmt; static int n; if (n++ < 8) VKLOG("[RED] view format %d of a format-listed image (%d, DM_VK_UBWC): the image's own used", f->vk, im->fmt); }
    pthread_mutex_lock(&g_view_mu);
    vview *head = nvn_hm_get(&g_views, &im);
    for (vview *v = head; v; v = v->next)
        if (v->type == type && v->fmt == fmt && v->lv0 == lv0 && v->nlv == nlv && v->ly0 == ly0 && v->nly == nly && v->aspect == aspect && v->attach == view_kind && !memcmp(&v->swz, &swz, sizeof swz)) { view_log_selected(t,vw,im,dim,arrayed,view_kind,v,1); pthread_mutex_unlock(&g_view_mu); vk_view_fmt = fmt; vk_view_layers = nly; return v->v; }
    VkImageViewUsageCreateInfo uci = {VK_STRUCTURE_TYPE_IMAGE_VIEW_USAGE_CREATE_INFO, NULL, 0};
    VkFormatProperties fp; vkGetPhysicalDeviceFormatProperties(V.pd, fmt, &fp);
    uci.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    if (storage) { if (!(fp.optimalTilingFeatures & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT)) { pthread_mutex_unlock(&g_view_mu); VKRED_ONCE(163, "storage view format %d is unsupported", fmt); return VK_NULL_HANDLE; } uci.usage = VK_IMAGE_USAGE_STORAGE_BIT; }
    else if (as_attachment) uci.usage = (fp.optimalTilingFeatures & VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT ? VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT : 0) | (fp.optimalTilingFeatures & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT ? VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT : 0);
    else if ((im->usage & VK_IMAGE_USAGE_STORAGE_BIT) && (t->d.flags & NVN_TEXTURE_FLAGS_IMAGE) && (fp.optimalTilingFeatures & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT)) uci.usage |= VK_IMAGE_USAGE_STORAGE_BIT;
    VkImageViewCreateInfo ci = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, &uci, 0, im->img, type, fmt, swz, {aspect, lv0, nlv, ly0, nly}};
    VkImageView view; VkResult r = vkCreateImageView(V.dev, &ci, NULL, &view);
    if (r) { pthread_mutex_unlock(&g_view_mu); VKRED_ONCE(15, "vkCreateImageView failed (%d): type %d format %d levels %u+%u layers %u+%u of image %ux%ux%u", r, type, fmt, lv0, nlv, ly0, nly, im->w, im->h, im->layers); return VK_NULL_HANDLE; }
    vview *nv = calloc(1, sizeof *nv); *nv = (vview){view, type, fmt, lv0, nlv, ly0, nly, swz, aspect, view_kind, head};
    nvn_hm_put(&g_views, &im, nv);
    view_log_selected(t,vw,im,dim,arrayed,view_kind,nv,0);
    pthread_mutex_unlock(&g_view_mu);
    vk_view_fmt = fmt; vk_view_layers = nly;
    return view;
}

// ---------------- texture / sampler ID tables ----------------
// A registration as the replay of each submission sees it. NVN's texture pool is memory the GPU reads as it runs, and a game rewrites an ID once the work
// that used it is done; with the replay on its own thread (DM_NVN_ASYNC) a submission is replayed after the game has gone on, and Xenoblade Chronicles 3
// rewrites about 90 IDs a second with other textures: a frame replayed late drew with the next frame's textures (whole-screen garbage while the camera
// turned). So each registration keeps the number of replay jobs queued when it was made, and a job sees the newest one made before it was queued
// (nvn_jobs_queued / nvn_replaying_job, gfx/nvn/hle_nvn.c); without a replay thread that is always the newest. TREG_OLD older ones per ID, kept for IDs rewritten
enum { TREG_OLD = 3 };
typedef struct { tex_t t; view_t v; uint8_t ok, has_v; uint64_t seq; } treg_ver;
typedef struct treg_history { treg_ver v; uint64_t until; struct treg_history *next; } treg_history;
typedef struct { treg_ver cur; treg_ver *old; treg_history *history; uint32_t history_next; } treg_t;   // old: TREG_OLD earlier versions, newest first (allocated at the first rewrite)
static treg_t *g_treg; static int g_ntreg, g_tid_max, g_sid_max;   // the registry, and the highest IDs registered (the pool tables' logs, gfx_vk_exec.c tbl_set)
int vk_tex_id_max(void) { return __atomic_load_n(&g_tid_max, __ATOMIC_RELAXED); }
int vk_samp_id_max(void) { return __atomic_load_n(&g_sid_max, __ATOMIC_RELAXED); }
uint64_t g_tex_gen = 1; uint64_t g_tex_gen_why[4];   // bumps by cause: orphan, first use, view drop, registration (logged with the texture table stats)   // changes whenever a texture ID is registered again or views go (gfx_vk_exec.c's per-submission view cache)
uint64_t g_samp_gen = 1;  // changes whenever a sampler ID is registered with other parameters (gfx_vk_exec.c samp_cached and the descriptor set memo)
static samp_par *g_sreg; static uint8_t *g_sreg_ok; static VkSampler *g_samp; static int g_nsreg;
// Samplers as a bind saw them. Super Mario Odyssey hands out sampler IDs from a cache of slots and registers a slot again with other parameters while a command
// buffer it has not submitted yet still binds that ID: the first pass of its bloom (a 400x228 chain, the level read chosen by the sampler's LOD range 0..0) bound a
// sampler that was then registered again as LOD 1..1 to 3..3, and read the chain's previous upsampled level. The bloom came out 2.7 times too strong for that frame:
// the glow around the moon flashed while the camera turned, in about 1% of frames (3 of 332 first passes; issue #42). Every registration that changes an ID's
// parameters counts (g_sregs), each bind records the count when it was made (mtl_samp_seq, hle_nvn.c), and a bind resolves its sampler to the newest version
// registered before it (vk_samp_at): SREG_OLD earlier versions are kept per ID. DM_NVN_SAMP_SNAP=0 (live, gfx_vk_exec.c): the newest, as before
enum { SREG_OLD = 3 };
typedef struct { samp_par p; uint64_t seq; } sreg_ver;
static uint64_t *g_sseq; static sreg_ver *g_sold; static uint64_t g_sregs;   // the current version's count, SREG_OLD older ones per ID (newest first)
// Logical queues can retain a bind recorded before enqueue, then keep it in a
// queue after its job finishes. Job IDs cannot bound these registration sequences.
// Keep immutable CPU parameter history for that opt-in mode until process exit;
// identical registrations below remain a no-op, and VkSamplers still deduplicate
// by parameters. Ascending vectors make old saved bindings logarithmic to resolve.
typedef struct { sreg_ver *v; size_t n, cap; } sreg_history;
static sreg_history *g_shistory;
static void samp_history_append(int sid) {   // g_rmu held, current version still installed
    sreg_history *h = &g_shistory[sid];
    if (h->n == h->cap) {
        size_t cap = h->cap ? h->cap * 2 : 4;
        if (cap < h->cap || cap > SIZE_MAX / sizeof *h->v) die("[vk] sampler history too large");
        sreg_ver *v = realloc(h->v, cap * sizeof *h->v);
        if (!v) die("[vk] sampler history allocation failed");
        h->v = v; h->cap = cap;
    }
    h->v[h->n++] = (sreg_ver){g_sreg[sid], g_sseq[sid]};
}
uint64_t mtl_samp_seq(void) { return __atomic_load_n(&g_sregs, __ATOMIC_ACQUIRE); }
static pthread_mutex_t g_rmu = PTHREAD_MUTEX_INITIALIZER;
// DM_CTL texlist: copy current CPU registration metadata only. The control thread must
// not create native views, resolve replay history, or touch texture storage/GPU work.
void vk_registry_texlist(int target) {
    enum { LIMIT = 256 };
    struct record { int tid; treg_ver value; };
    if (target < NVN_TEXTURE_TARGET_1D || target > NVN_TEXTURE_TARGET_BUFFER) {
        VKLOG("[texlist] unsupported NVN target %d", target); return;
    }
    struct record *records = calloc(LIMIT, sizeof *records);
    if (!records) { VKLOG("[texlist] snapshot allocation failed"); return; }
    unsigned n = 0; int truncated = 0;
    pthread_mutex_lock(&g_rmu);
    for (int tid = 1; tid < g_ntreg; tid++) {
        const treg_ver *v = &g_treg[tid].cur;
        if (!v->ok || v->t.d.target != target) continue;
        if (n == LIMIT) { truncated = 1; break; }
        records[n++] = (struct record){tid, *v};
    }
    pthread_mutex_unlock(&g_rmu);
    VKLOG("[texlist] current registry target %d: %u entries%s (CPU metadata snapshot; no native image lookup)",
        target, n, truncated ? ", truncated at 256" : "");
    for (unsigned i = 0; i < n; i++) {
        const treg_ver *r = &records[i].value; const tex_t *t = &r->t; const view_t *v = &r->v;
        VKLOG("[texlist] tid=%d ser=%llu seq=%llu target %d %ux%ux%u format %#x levels %u layers %u flags %#x pool %p+%#llx storage %p size %#llx layer_size %#llx block height/depth log2 %u/%u swizzle %d/%#x stencil %d",
            records[i].tid, (unsigned long long)t->serial, (unsigned long long)r->seq, t->d.target,
            t->d.width, t->d.height, t->d.depth, t->d.format, t->d.levels, t->L.layers, t->d.flags,
            (void *)t->pool, (unsigned long long)t->off, (void *)t->mem, (unsigned long long)t->L.size,
            (unsigned long long)t->L.layer_size, t->L.tileH, t->L.tileD, t->set_swz, t->swz, t->stencil);
        if (r->has_v) VKLOG("[texlist] tid=%d view target %d/%u format %d/%#x levels %d/%u+%u layers %d/%u+%u swizzle %d/%u,%u,%u,%u depth-stencil %d/%u",
            records[i].tid, v->set_target, v->target, v->set_format, v->format,
            v->set_levels, v->base_level, v->nlevels, v->set_layers, v->min_layer, v->nlayers,
            v->set_swizzle, v->swz[0], v->swz[1], v->swz[2], v->swz[3], v->set_dsmode, v->dsmode);
    }
    free(records);
}
__attribute__((weak)) uint64_t nvn_jobs_queued(void) { return 0; }
__attribute__((weak)) uint64_t nvn_replaying_job(void) { return 0; }
// Logical queues can replay out of enqueue order or park for arbitrarily many later jobs.
// Keep precisely the registration intervals visible to active jobs; never truncate them to TREG_OLD.
// Job pin/unpin and registration are serialized by the scheduler's submit lock; g_rmu also protects
// lookup and this retention state. Image pins are nested only g_rmu -> g_img_mu.
static struct { uint64_t id, refs, sampler_seq, texture_seq; } *g_registry_jobs;
static uint64_t g_texture_registrations;
static size_t g_registry_njobs, g_registry_capjobs;
static int g_registry_logical;
static uint32_t g_registry_history;   // IDs with retained intervals, encoded as tid+1
static int registry_interval_needed(uint64_t after, uint64_t through) {
    for (size_t i = 0; i < g_registry_njobs; i++) if (g_registry_jobs[i].id > after && g_registry_jobs[i].id <= through) return 1;
    return 0;
}
void vk_registry_job_pin(uint64_t id) {
    if (!id) return; pthread_mutex_lock(&g_rmu); g_registry_logical = 1;
    for (size_t i = 0; i < g_registry_njobs; i++) if (g_registry_jobs[i].id == id) { g_registry_jobs[i].refs++; pthread_mutex_unlock(&g_rmu); return; }
    if (g_registry_njobs == g_registry_capjobs) {
        size_t cap = g_registry_capjobs ? 2 * g_registry_capjobs : 16;
        void *next = realloc(g_registry_jobs, cap * sizeof *g_registry_jobs); if (!next) die("[vk] registry job retention allocation failed");
        g_registry_jobs = next; g_registry_capjobs = cap;
    }
    g_registry_jobs[g_registry_njobs++] = (__typeof__(*g_registry_jobs)){id, 1, g_sregs, g_texture_registrations}; pthread_mutex_unlock(&g_rmu);
}
// Enqueue sequence is not a registration identity: many rewrites can happen between jobs.
// Unpinned legacy asynchronous jobs have no complete semantic snapshot and cannot reuse tables.
int vk_registry_snapshot(uint64_t *textures, uint64_t *samplers) {
    uint64_t job = nvn_replaying_job(); int ok = 0;
    pthread_mutex_lock(&g_rmu);
    if (!job) { *textures = g_texture_registrations; *samplers = g_sregs; ok = 1; }
    else if (g_registry_logical) for (size_t i = 0; i < g_registry_njobs; i++) if (g_registry_jobs[i].id == job) {
        *textures = g_registry_jobs[i].texture_seq; *samplers = g_registry_jobs[i].sampler_seq; ok = 1; break;
    }
    pthread_mutex_unlock(&g_rmu); return ok;
}
// g_rmu held. Pool samplers have no per-bind sequence, so use their enclosing job's snapshot.
// A valid snapshot of zero registrations is distinct from the public API's zero-means-latest.
static int registry_job_sampler_seq(uint64_t *seq) {
    uint64_t job = nvn_replaying_job(); if (!g_registry_logical || !job) return 0;
    for (size_t i = 0; i < g_registry_njobs; i++) if (g_registry_jobs[i].id == job) { *seq = g_registry_jobs[i].sampler_seq; return 1; }
    *seq = 0; return 1;   // an unpinned logical job must not silently see future registrations
}
void vk_registry_job_unpin(uint64_t id) {
    if (!id) return; pthread_mutex_lock(&g_rmu);
    for (size_t i = 0; i < g_registry_njobs; i++) if (g_registry_jobs[i].id == id) {
        if (!--g_registry_jobs[i].refs) g_registry_jobs[i] = g_registry_jobs[--g_registry_njobs];
        break;
    }
    uint32_t *link = &g_registry_history;
    while (*link) {
        treg_t *r = &g_treg[*link - 1]; treg_history **p = &r->history;
        while (*p) { treg_history *h = *p;
            if (registry_interval_needed(h->v.seq, h->until)) p = &h->next;
            else { *p = h->next; if (h->v.ok) mtl_tex_unpin(h->v.t.serial); free(h); }
        }
        if (r->history) link = &r->history_next;
        else { *link = r->history_next; r->history_next = 0; }
    }
    pthread_mutex_unlock(&g_rmu);
}
static void registry_retain(treg_t *r, int tid, uint64_t until) {
    if (!registry_interval_needed(r->cur.seq, until)) return;
    treg_history *h = calloc(1, sizeof *h); if (!h) die("[vk] texture registry history allocation failed");
    h->v = r->cur; h->until = until;
    if (h->v.ok) mtl_tex_snapshot(&h->v.t, &r->cur.t);
    if (!r->history) { r->history_next = g_registry_history; g_registry_history = (uint32_t)tid + 1; }
    h->next = r->history; r->history = h;
}
// The IDs whose registration changed, in order, with the job sequence each was queued at (the replay sees a registration from the jobs after it): gfx_vk_exec.c's ring of
// pool tables rewrites only these in a table that was up to date before them. Written under g_rmu
enum { RLOG_N = 1 << 15 };
static struct { int tid; uint64_t seq; } g_rlog[RLOG_N]; static uint64_t g_rlog_head;
uint64_t vk_reglog_head(void) { pthread_mutex_lock(&g_rmu); uint64_t h = g_rlog_head; pthread_mutex_unlock(&g_rmu); return h; }
// Whether an entry after position p is visible to a job (its sequence below `before`)
int vk_reglog_pending(uint64_t p, uint64_t before) { pthread_mutex_lock(&g_rmu); int r = p < g_rlog_head && (g_rlog_head - p > RLOG_N || g_rlog[p % RLOG_N].seq < before); pthread_mutex_unlock(&g_rmu); return r; }
// The IDs of the entries after `from` visible to a job (sequence below `before`), at most max: the position reached, or UINT64_MAX when the log no longer holds `from`
// The position up to which every entry is visible to a job (sequence below `before`): entries are in sequence order, so the first one that is not ends them
uint64_t vk_reglog_visible_end(uint64_t before) {
    pthread_mutex_lock(&g_rmu); uint64_t lo = g_rlog_head > RLOG_N ? g_rlog_head - RLOG_N : 0, p = g_rlog_head;
    while (p > lo && g_rlog[(p - 1) % RLOG_N].seq >= before) p--;
    pthread_mutex_unlock(&g_rmu); return p;
}
uint64_t vk_reglog_take(uint64_t from, uint64_t before, int *tids, int max, int *n) {
    pthread_mutex_lock(&g_rmu); *n = 0;
    if (g_rlog_head - from > RLOG_N) { pthread_mutex_unlock(&g_rmu); return UINT64_MAX; }
    uint64_t p = from; while (p < g_rlog_head && *n < max && g_rlog[p % RLOG_N].seq < before) tids[(*n)++] = g_rlog[p++ % RLOG_N].tid;
    pthread_mutex_unlock(&g_rmu); return p;
}
void mtl_reg_tex(int tid, const tex_t *t, const view_t *v) {
    if (tid < 0) return;
    uint64_t seq = nvn_jobs_queued();
    pthread_mutex_lock(&g_rmu);
    if (tid >= g_ntreg) { int n = tid + 1024; g_treg = realloc(g_treg, n * sizeof *g_treg); memset(g_treg + g_ntreg, 0, (n - g_ntreg) * sizeof *g_treg); g_ntreg = n; }
    treg_t *r = &g_treg[tid];
    // Registering byte-identical texture/view snapshots changes no descriptor. Keep the original
    // seq too: advancing it or pushing a duplicate would hide meaningful history from queued jobs.
    // A full comparison is conservative; differing padding only misses this optimization.
    if (t && r->cur.ok && r->cur.has_v == (v != NULL) && !memcmp(&r->cur.t, t, sizeof *t) && (!v || !memcmp(&r->cur.v, v, sizeof *v))) {
        pthread_mutex_unlock(&g_rmu); return;
    }
    g_tex_gen++; { extern uint64_t g_tex_gen_why[4]; g_tex_gen_why[3]++; } g_texture_registrations++; if (tid > g_tid_max) g_tid_max = tid;
    vk_tex_diag_registration(tid, r->cur.ok ? &r->cur.t : NULL, r->cur.has_v ? &r->cur.v : NULL, t, v, r->cur.seq, seq, g_texture_registrations);
    if (g_registry_logical) registry_retain(r, tid, seq);
    else if (r->cur.ok && seq) {   // keep the one replaced for legacy FIFO jobs queued before this registration
        if (!r->old) r->old = calloc(TREG_OLD, sizeof *r->old);
        if (r->old) { memmove(r->old + 1, r->old, sizeof *r->old * (TREG_OLD - 1)); r->old[0] = r->cur; }
    }
    r->cur.ok = t != NULL; if (t) r->cur.t = *t; r->cur.has_v = v != NULL; if (v) r->cur.v = *v; r->cur.seq = seq;
    g_rlog[g_rlog_head % RLOG_N].tid = tid; g_rlog[g_rlog_head % RLOG_N].seq = seq; g_rlog_head++;
    pthread_mutex_unlock(&g_rmu);
}
void mtl_set_tex_pool_n(int n) { (void)n; }
static __thread treg_ver t_treg;   // the version a lookup found, copied (the table may be rewritten or grow meanwhile)
vkimg_t *vk_tex_by_id(int tid, const view_t **view_out, const tex_t **tex_out) {
    uint64_t job = nvn_replaying_job();
    pthread_mutex_lock(&g_rmu);
    const treg_ver *r = tid > 0 && tid < g_ntreg ? &g_treg[tid].cur : NULL;
    if (r && job && r->seq >= job) {
        if (g_registry_logical) {
            r = NULL;
            for (const treg_history *h = g_treg[tid].history; h; h = h->next) if (h->v.seq < job && job <= h->until) { r = &h->v; break; }
            // No retained interval means the ID did not exist, or the caller failed to pin its job.
            // Never substitute a future registration for a parked job.
        } else if (g_treg[tid].old)
            for (int k = 0; k < TREG_OLD; k++) { const treg_ver *o = &g_treg[tid].old[k]; if (!o->ok) break; r = o; if (o->seq < job) break; }
    }
    int ok = r && r->ok; if (ok) t_treg = *r;
    pthread_mutex_unlock(&g_rmu);
    if (!ok) return NULL;
    if (view_out) *view_out = t_treg.has_v ? &t_treg.v : NULL; if (tex_out) *tex_out = &t_treg.t;
    return (vkimg_t *)1;   // presence only: the caller makes the view with vk_tex_view
}
static VkSamplerAddressMode wrap(int w) {
    switch (w) { case 1: return VK_SAMPLER_ADDRESS_MODE_REPEAT; case 6: return VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT; case 2: case 3: return VK_SAMPLER_ADDRESS_MODE_MIRROR_CLAMP_TO_EDGE;
                 case 4: case 5: return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER; default: return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE; }
}
static VkCompareOp cmp_op(int f) { return f >= 1 && f <= 8 ? (VkCompareOp)(f - 1) : VK_COMPARE_OP_ALWAYS; }   // NVN NEVER..ALWAYS = 1..8, Vulkan 0..7 in the same order
static VkSampler make_sampler(const samp_par *p) {
    VkSamplerCustomBorderColorCreateInfoEXT cb = {VK_STRUCTURE_TYPE_SAMPLER_CUSTOM_BORDER_COLOR_CREATE_INFO_EXT};
    VkSamplerReductionModeCreateInfo rd = {VK_STRUCTURE_TYPE_SAMPLER_REDUCTION_MODE_CREATE_INFO, NULL, p->reduction == 1 ? VK_SAMPLER_REDUCTION_MODE_MIN : p->reduction == 2 ? VK_SAMPLER_REDUCTION_MODE_MAX : VK_SAMPLER_REDUCTION_MODE_WEIGHTED_AVERAGE};
    VkSamplerCreateInfo ci = {VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO, p->reduction ? &rd : NULL};
    ci.magFilter = p->magf ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;
    ci.minFilter = p->minf == 1 || p->minf == 3 || p->minf == 5 ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;
    ci.mipmapMode = p->minf == 4 || p->minf == 5 ? VK_SAMPLER_MIPMAP_MODE_LINEAR : VK_SAMPLER_MIPMAP_MODE_NEAREST;
    ci.addressModeU = wrap(p->ws); ci.addressModeV = wrap(p->wt); ci.addressModeW = wrap(p->wr);
    ci.mipLodBias = p->lod_bias;
    ci.anisotropyEnable = p->aniso > 1.0f; ci.maxAnisotropy = p->aniso > V.max_aniso ? V.max_aniso : p->aniso;
    ci.compareEnable = p->cmp_mode == 1; ci.compareOp = cmp_op(p->cmp_func);
    ci.minLod = p->lod_min; ci.maxLod = p->minf <= 1 ? 0.25f : p->lod_max;   // no mipmap filter: level 0 only (GL's rule)
    int border = ci.addressModeU == VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER || ci.addressModeV == VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER || ci.addressModeW == VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    ci.borderColor = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK;
    if (border) {
        const float *b = p->border;
        if (b[0] == 0 && b[1] == 0 && b[2] == 0 && b[3] == 1) ci.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK;
        else if (b[0] == 1 && b[1] == 1 && b[2] == 1 && b[3] == 1) ci.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
        else if ((b[0] || b[1] || b[2] || b[3]) && V.has_custom_border) {
            ci.borderColor = VK_BORDER_COLOR_FLOAT_CUSTOM_EXT; memcpy(cb.customBorderColor.float32, b, 16); cb.format = VK_FORMAT_UNDEFINED;
            cb.pNext = (void *)ci.pNext; ci.pNext = &cb;
        }
    }
    VkSampler s; if (vkCreateSampler(V.dev, &ci, NULL, &s)) return VK_NULL_HANDLE;
    return s;
}
volatile int g_vk_samp_log;   // DM_VK_SAMP_LOG=<n> (live, gfx_vk_exec.c x_read): the next n sampler registrations, in time order with the replay's lines
void mtl_reg_samp(int sid, const samp_par *sp) {
    if (sid < 0) return;
    if (g_vk_samp_log > 0) { extern uint64_t g_tex_frame; g_vk_samp_log--; LOG("[vk] sampler registered: id %d lod %g..%g min %d (present %llu, thread %d)", sid, sp->lod_min, sp->lod_max, sp->minf, (unsigned long long)g_tex_frame, (int)gettid()); }
    pthread_mutex_lock(&g_rmu);
    if (sid >= g_nsreg) { int n = sid + 256; g_sreg = realloc(g_sreg, n * sizeof *g_sreg); g_sreg_ok = realloc(g_sreg_ok, n); g_samp = realloc(g_samp, n * sizeof *g_samp);
        g_sseq = realloc(g_sseq, n * sizeof *g_sseq); g_sold = realloc(g_sold, (size_t)n * SREG_OLD * sizeof *g_sold);
        memset(g_sreg_ok + g_nsreg, 0, n - g_nsreg); memset(g_samp + g_nsreg, 0, (n - g_nsreg) * sizeof *g_samp); memset(g_sseq + g_nsreg, 0, (n - g_nsreg) * sizeof *g_sseq);
        memset(g_sold + (size_t)g_nsreg * SREG_OLD, 0, (size_t)(n - g_nsreg) * SREG_OLD * sizeof *g_sold);
        if (nvn_logical_queues_on()) {
            sreg_history *history = realloc(g_shistory, (size_t)n * sizeof *history);
            if (!history) die("[vk] sampler history table allocation failed");
            g_shistory = history; memset(g_shistory + g_nsreg, 0, (size_t)(n - g_nsreg) * sizeof *g_shistory);
        }
        g_nsreg = n; }
    if (sid > g_sid_max) g_sid_max = sid;
    if (!g_sreg_ok[sid] || memcmp(&g_sreg[sid], sp, sizeof *sp)) {
        sreg_ver *o = g_sold + (size_t)sid * SREG_OLD;
        if (g_sreg_ok[sid]) {
            if (nvn_logical_queues_on()) samp_history_append(sid);
            memmove(o + 1, o, sizeof *o * (SREG_OLD - 1)); o[0] = (sreg_ver){g_sreg[sid], g_sseq[sid]};
        }
        g_sreg[sid] = *sp; g_sreg_ok[sid] = 1; g_samp[sid] = VK_NULL_HANDLE; g_sseq[sid] = __atomic_add_fetch(&g_sregs, 1, __ATOMIC_RELEASE); __atomic_fetch_add(&g_samp_gen, 1, __ATOMIC_RELEASE); }   // samplers are cached by parameters below
    pthread_mutex_unlock(&g_rmu);
}
static nvn_hmap g_samp_cache = {.klen = sizeof(samp_par)};   // samp_par → VkSampler (VkSamplers are never destroyed: few distinct parameter sets)
static const samp_par *samp_at(int sid, uint64_t seq);
static const samp_par *samp_job_at(int sid);
void vk_samp_describe_at(int sid, uint64_t seq, char *out, size_t n) {   // the registered parameters a bind made after seq registrations sees (traces; 0: the newest)
    pthread_mutex_lock(&g_rmu);
    const samp_par *p = seq ? samp_at(sid, seq) : samp_job_at(sid);
    if (p) { snprintf(out, n, "sampler %d: wrap %d %d %d min %d mag %d lod %g..%g bias %g aniso %g cmp %d/%d red %d border %g %g %g %g", sid, p->ws, p->wt, p->wr,
        p->minf, p->magf, p->lod_min, p->lod_max, p->lod_bias, p->aniso, p->cmp_mode, p->cmp_func, p->reduction, p->border[0], p->border[1], p->border[2], p->border[3]); }
    else snprintf(out, n, "sampler %d: not registered (dummy)", sid);
    pthread_mutex_unlock(&g_rmu);
}
void vk_samp_describe(int sid, char *out, size_t n) { vk_samp_describe_at(sid, 0, out, n); }
static const samp_par *samp_at(int sid, uint64_t seq) {   // under g_rmu: the version a bind made after seq registrations sees; NULL when not registered
    if (sid <= 0 || sid >= g_nsreg || !g_sreg_ok[sid]) return NULL;
    if (!seq || g_sseq[sid] <= seq) return &g_sreg[sid];
    if (nvn_logical_queues_on()) {
        const sreg_history *h = &g_shistory[sid];
        size_t lo = 0, hi = h->n;
        while (lo < hi) { size_t mid = lo + (hi - lo) / 2; if (h->v[mid].seq <= seq) lo = mid + 1; else hi = mid; }
        return lo ? &h->v[lo - 1].p : NULL;   // this ID was not registered yet
    }
    const sreg_ver *o = g_sold + (size_t)sid * SREG_OLD;
    for (int k = 0; k < SREG_OLD; k++) if (o[k].seq && o[k].seq <= seq) return &o[k].p;
    return &g_sreg[sid];   // older than what is kept: the newest
}
// Implicit and bindless sampler IDs have no explicit bind-time sequence. Resolve
// them at this logical job's immutable enqueue snapshot, never a later registration.
static const samp_par *samp_job_at(int sid) {   // g_rmu held
    uint64_t seq = 0;
    if (registry_job_sampler_seq(&seq)) return seq ? samp_at(sid, seq) : NULL;
    return samp_at(sid, 0);
}
VkSampler vk_samp_at(int sid, uint64_t seq, int nearest) {   // the sampler a bind made after seq registrations sees (gfx_vk_exec.c samp_for)
    pthread_mutex_lock(&g_rmu);
    VkSampler s = VK_NULL_HANDLE; const samp_par *q = samp_at(sid, seq);
    if (q) { samp_par p = *q;
        if (nearest) { p.magf = 0; p.minf = p.minf == 1 ? 0 : p.minf == 3 || p.minf == 4 || p.minf == 5 ? 2 : p.minf; p.aniso = 1; p.reduction = 0; }
        if (!(s = nvn_hm_get(&g_samp_cache, &p)) && (s = make_sampler(&p))) nvn_hm_put(&g_samp_cache, &p, s); }
    pthread_mutex_unlock(&g_rmu);
    return s;
}
VkSampler vk_samp_by_id(int sid) {
    pthread_mutex_lock(&g_rmu);
    VkSampler s = VK_NULL_HANDLE; const samp_par *p = samp_job_at(sid);
    if (p) {
        int current = p == &g_sreg[sid];
        if (current) s = g_samp[sid];
        if (!s) { s = nvn_hm_get(&g_samp_cache, p); if (!s) { s = make_sampler(p); if (s) nvn_hm_put(&g_samp_cache, p, s); } if (current) g_samp[sid] = s; }
    }
    pthread_mutex_unlock(&g_rmu);
    return s;
}
// The same sampler without filtering, for integer views: Maxwell reads integer texels unfiltered whatever the sampler says, while Vulkan forbids linear filtering of
// formats without SAMPLED_IMAGE_FILTER_LINEAR (Pokémon Legends: Arceus samples R8_UINT and R32_UINT views with trilinear samplers; the validation layer named them)
VkSampler vk_samp_by_id_nearest(int sid) {   // cached by its parameters, as the samplers themselves
    pthread_mutex_lock(&g_rmu);
    VkSampler s = VK_NULL_HANDLE; const samp_par *q = samp_job_at(sid);
    if (q) {
        samp_par p = *q; p.magf = 0; p.minf = p.minf == 1 ? 0 : p.minf == 3 || p.minf == 4 || p.minf == 5 ? 2 : p.minf; p.aniso = 1; p.reduction = 0;
        if (!(s = nvn_hm_get(&g_samp_cache, &p)) && (s = make_sampler(&p))) nvn_hm_put(&g_samp_cache, &p, s);
    }
    pthread_mutex_unlock(&g_rmu);
    return s;
}
VkSampler vk_dummy_sampler(void) {
    static VkSampler s; if (s) return s;
    samp_par p = {0}; p.magf = 1; p.minf = 1; p.lod_max = 1000; p.aniso = 1; s = make_sampler(&p); return s;
}
// ---------------- dummies: what an unbound slot reads (zeros) ----------------
VkImageView vk_dummy_view(int dim, int arrayed, int ms, int storage, int is_int) {
    static VkImageView cache[4][2][2][2][2]; static vkimg_t *imgs[4][2][2][2]; static pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;
    int d = dim == 0 ? 0 : dim == 2 ? 2 : dim == 3 ? 3 : 1; arrayed = !!arrayed; ms = !!ms && d == 1; storage = !!storage; is_int = !!is_int;
    pthread_mutex_lock(&mu);
    VkImageView v = cache[d][arrayed][ms][storage][is_int];
    if (!v) {
        vkimg_t **pi = &imgs[d][ms][storage][is_int];
        if (!*pi) {
            tex_t t; memset(&t, 0, sizeof t); t.magic = MAGIC('T', 'E', 'X', 0); t.serial = 1;
            t.d.format = is_int ? 0x27 : 0x25; t.d.width = t.d.height = 1; t.d.depth = d == 3 ? 12 : 6; t.d.levels = 1; t.d.samples = ms ? 4 : 0;
            t.d.target = d == 0 ? 3 : d == 2 ? 2 : d == 3 ? 9 : ms ? 6 : 4; t.d.flags = storage ? NVN_TEXTURE_FLAGS_IMAGE : 0;
            if (d == 2) t.d.depth = 1;
            *pi = img_new(&t, vk_fmt(t.d.format), 0, VK_FORMAT_UNDEFINED);
            VkCommandBuffer cb = vk_upload_cmd(); vk_tex_prepare(cb, *pi);
            VkClearColorValue z = {{0}}; VkImageSubresourceRange rg = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, VK_REMAINING_ARRAY_LAYERS};
            if (!ms) vkCmdClearColorImage(cb, (*pi)->img, VK_IMAGE_LAYOUT_GENERAL, &z, 1, &rg);
        }
        vkimg_t *im = *pi;
        VkImageViewType type = d == 0 ? (arrayed ? VK_IMAGE_VIEW_TYPE_1D_ARRAY : VK_IMAGE_VIEW_TYPE_1D) : d == 2 ? VK_IMAGE_VIEW_TYPE_3D : d == 3 ? (arrayed ? VK_IMAGE_VIEW_TYPE_CUBE_ARRAY : VK_IMAGE_VIEW_TYPE_CUBE) : (arrayed ? VK_IMAGE_VIEW_TYPE_2D_ARRAY : VK_IMAGE_VIEW_TYPE_2D);
        uint32_t nly = type == VK_IMAGE_VIEW_TYPE_CUBE ? 6 : type == VK_IMAGE_VIEW_TYPE_CUBE_ARRAY ? 12 : arrayed ? im->layers : 1;
        VkImageViewCreateInfo ci = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, NULL, 0, im->img, type, im->fmt, {0}, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, nly}};
        VKCHK(vkCreateImageView(V.dev, &ci, NULL, &v)); cache[d][arrayed][ms][storage][is_int] = v;
    }
    pthread_mutex_unlock(&mu);
    return v;
}
VkBufferView vk_dummy_tbuf(int storage) {
    static VkBufferView v[2]; static VkBuffer b; static VkDeviceMemory m; storage = !!storage;
    if (v[storage]) return v[storage];
    if (!b) {
        VkBufferCreateInfo bi = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, NULL, 0, 65536, VK_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_TEXEL_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_SHARING_MODE_EXCLUSIVE};
        VKCHK(vkCreateBuffer(V.dev, &bi, NULL, &b)); VkMemoryRequirements req; vkGetBufferMemoryRequirements(V.dev, b, &req);
        VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL, req.size, vk_mem_type(req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 0)};
        VKCHK(vkAllocateMemory(V.dev, &ai, NULL, &m)); VKCHK(vkBindBufferMemory(V.dev, b, m, 0));
        void *p; VKCHK(vkMapMemory(V.dev, m, 0, VK_WHOLE_SIZE, 0, &p)); memset(p, 0, 65536);
    }
    VkBufferViewCreateInfo ci = {VK_STRUCTURE_TYPE_BUFFER_VIEW_CREATE_INFO, NULL, 0, b, VK_FORMAT_R32_UINT, 0, 65536};
    VKCHK(vkCreateBufferView(V.dev, &ci, NULL, &v[storage]));
    return v[storage];
}
// a buffer texture's texel buffer view over its pool (cached by texture contents: the registered copy). One at an offset that is not a multiple of
// minTexelBufferOffsetAlignment cannot have a view there: a read-only one (storage 0) is given a copy of its bytes as the CPU sees them, at an aligned place in
// this submission's scratch memory, made at each bind and freed with the submission. Final Fantasy Resonance (Unreal Engine 5.6) binds its sprite particles'
// parameters (R32F, 44 elements: colours, sizes) at offsets 36 bytes past a multiple of Adreno's 64, and they read the zero dummy (its battle fires stay missing with
// the copy: another cause). A buffer the GPU writes in the same submission would be read as last copied back; a storage one keeps the dummy (its writes would go to the copy)
VkBufferView vk_tbuf_view(const tex_t *t, int storage) {
    static nvn_hmap cache = {.klen = 24}; static pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;
    // keyed by the pool too: a pool finalized and another made at the same address has a new VkBuffer, and the old one's memory is freed once the GPU is past it. Keyed by address
    // alone, the view over the old buffer was handed out again and shaders read and wrote freed memory, which the kernel gives to newer buffers (Xenoblade Chronicles 3: the
    // allocator kept inside one of its memory pools had its tree bins overwritten during the opening movie, and malloc crashed at the next scene)
    vkpool_t *pl = t->mem ? vk_pool_of((uint64_t)(uintptr_t)t->mem, 1) : NULL;
    struct { uint64_t mem, pool; uint32_t fmt, n; } k = {(uint64_t)(uintptr_t)t->mem, pl ? pl->serial : 0, (uint32_t)t->d.format, t->d.width};
    pthread_mutex_lock(&mu);
    VkBufferView v = nvn_hm_get(&cache, &k);
    if (!v) {
        const vkfmt_t *f = vk_fmt(t->d.format); VkBuffer b; VkDeviceSize off = 0;
        uint32_t al = (uint32_t)V.props.limits.minTexelBufferOffsetAlignment;
        int in = f && t->mem && vk_buf_at((uint64_t)(uintptr_t)t->mem, (uint64_t)t->d.width * f->bpb, &b, &off);
        if (in && off % al == 0) {
            VkBufferViewCreateInfo ci = {VK_STRUCTURE_TYPE_BUFFER_VIEW_CREATE_INFO, NULL, 0, b, f->vk, off, (VkDeviceSize)t->d.width * f->bpb};
            if (vkCreateBufferView(V.dev, &ci, NULL, &v)) v = VK_NULL_HANDLE; else nvn_hm_put(&cache, &k, v);
        } else if (in && !storage) {
            VkDeviceSize n = (VkDeviceSize)t->d.width * f->bpb, so; VkBuffer sb; void vk_ubd_read(uint64_t a, uint64_t n, uint8_t *dst);
            uint8_t *p = vk_scratch(n, al, &sb, &so); vk_ubd_read((uint64_t)(uintptr_t)t->mem, n, p);
            VkBufferViewCreateInfo ci = {VK_STRUCTURE_TYPE_BUFFER_VIEW_CREATE_INFO, NULL, 0, sb, f->vk, so, n};
            if (vkCreateBufferView(V.dev, &ci, NULL, &v)) v = VK_NULL_HANDLE; else vk_defer_free_bview(v);
            static unsigned nc; if (nc++ < 4) VKLOG("buffer texture at a misaligned offset (%llu, not a multiple of %u; format %#x, %u elements): read from a copy", (unsigned long long)off, al, t->d.format, t->d.width);
            pthread_mutex_unlock(&mu); return v;   // (not cached: a copy of this bind's bytes)
        } else if (in) VKRED_ONCE(16, "storage buffer texture at a misaligned offset: %llu, not a multiple of %u (format %#x, %u elements): zeros", (unsigned long long)off, al, t->d.format, t->d.width);
        else VKRED_ONCE(16, "buffer texture not in a pool (format %#x, %u elements)", t->d.format, t->d.width);
    }
    pthread_mutex_unlock(&mu);
    return v;
}
// CPU readback of texels (ReadTexels): the image copied to scratch memory by a submission of its own, waited for
void mtl_read_texels(const tex_t *t, unsigned level, unsigned layer0, const NVNcopyRegion *r, uint8_t *dst, size_t row, size_t img) {
    void mtl_sub_begin(void); uint64_t mtl_sub_commit(void); int mtl_wait(uint64_t, uint64_t);
    mtl_sub_begin();
    vkimg_t *im = vk_tex(t); const vkfmt_t *f = vk_fmt(t->d.format);
    size_t depth = r->depth ? r->depth : 1;
    if (!im || !f) { mtl_sub_commit(); for (size_t z = 0; z < depth; z++) for (int y = 0; y < r->height; y++) memset(dst + z * img + (size_t)y * row, 0, row); return; }
    if (f->dec) { VKRED_ONCE(17, "ReadTexels of a BC texture decoded on the CPU: zeros returned"); mtl_sub_commit(); for (size_t z = 0; z < depth; z++) for (int y = 0; y < (r->height + 3) / 4; y++) memset(dst + z * img + (size_t)y * row, 0, row); return; }
    unsigned bw = f->bw ? f->bw : 1, bh = f->bh ? f->bh : 1;
    size_t pitch = (size_t)((r->width + bw - 1) / bw) * f->bpb, rows = (r->height + bh - 1) / bh;
    VkBuffer sb; VkDeviceSize so; uint8_t *p = vk_scratch(pitch * rows * depth, 16, &sb, &so);
    VkCommandBuffer cb = vk_cmd(); vk_tex_prepare(cb, im); vk_img_barrier_all(cb);
    int is3d = im->d > 1;
    VkBufferImageCopy c = {so, 0, 0, {im->aspect & VK_IMAGE_ASPECT_COLOR_BIT ? VK_IMAGE_ASPECT_COLOR_BIT : VK_IMAGE_ASPECT_DEPTH_BIT, level, is3d ? 0 : layer0 + (uint32_t)r->zoffset, is3d ? 1 : (uint32_t)depth},
                           {r->xoffset, r->yoffset, is3d ? r->zoffset : 0}, {(uint32_t)r->width, (uint32_t)r->height, is3d ? (uint32_t)depth : 1}};
    vkCmdCopyImageToBuffer(cb, im->img, VK_IMAGE_LAYOUT_GENERAL, sb, 1, &c);
    mtl_wait(mtl_sub_commit(), ~0ull);
    for (size_t z = 0; z < depth; z++) for (size_t y = 0; y < rows; y++) memcpy(dst + z * img + y * row, p + (z * rows + y) * pitch, pitch < row ? pitch : row);
}

// ---------------- texture memory budget (gfx_vk_budget.c): the live images and dropping one ----------------
// Both tables' images into *out (grown as needed); the caller holds g_img_mu (vk_tex calls vk_img_budget_make_room under it)
size_t vk_img_list(vk_img_ref **out, size_t *cap) {
    size_t n = 0;
    for (int m = 0; m < 2; m++) {
        nvn_hmap *h = m ? &g_by_serial : &g_by_key;
        for (uint32_t i = 0; i < h->cap; i++) {
            vkimg_t *im = h->vals[i]; if (!im || im->pins) continue;
            if (n == *cap) { *cap = *cap ? 2 * *cap : 4096; vk_img_ref *nw = realloc(*out, *cap * sizeof **out); if (!nw) return n; *out = nw; }
            (*out)[n++] = (vk_img_ref){h, i, im};
        }
    }
    return n;
}
void vk_img_evict(const vk_img_ref *r) {   // under g_img_mu, from a list taken under the same hold of the lock (a deletion shifts later entries back: the image is looked for again if its slot changed)
    if (r->im->pins) return;
    nvn_hmap *h = r->h; uint32_t i = r->i;
    if (i >= h->cap || h->vals[i] != r->im) { for (i = 0; i < h->cap && h->vals[i] != r->im; i++) {} if (i == h->cap) return; }
    uint8_t key[64]; _Static_assert(sizeof(akey) <= 64, "akey"); memcpy(key, h->keys + (size_t)i * h->klen, h->klen);
    nvn_hm_del(h, key); img_drop(r->im); g_img_evicted++;
}
uint64_t vk_img_vkbytes(void) { return g_img_vkbytes; }

#include "gfx_vk_capture.inc"   // DM_NVN_CAPTURE: frame captures over this backend's registries and images (dm/android/tools/nvn_replay_vk.c replays them)
