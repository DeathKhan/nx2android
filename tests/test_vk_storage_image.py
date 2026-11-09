"""Storage-image promotion must preserve prior texels and tracked ownership."""
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


def function(source, declaration):
    start = source.index(declaration)
    brace = source.index('{', start)
    pos, depth = brace + 1, 1
    while depth:
        depth += (source[pos] == '{') - (source[pos] == '}')
        pos += 1
    return source[start:pos]


class StorageImage(unittest.TestCase):
    @unittest.skipUnless(shutil.which('cc'), 'a host C compiler is required')
    def test_actual_promotion_preserves_resources_and_contents(self):
        header = (ROOT / 'nxnative/dm/android/gfx_vk.h').read_text()
        runtime = (ROOT / 'nxnative/dm/android/gfx_vk_tex.c').read_text()
        image_type = re.search(r'typedef struct vkimg vkimg_t;\nstruct vkimg \{.*?\n\};', header, re.S).group()
        helper = '\n'.join(function(runtime, n) for n in ['static VkFormat storage_promote(', 'static int storage_view_compatible(', 'int vk_tex_storage_prepare('])
        code = r'''
#define VK_NO_PROTOTYPES
#define VK_USE_64_BIT_PTR_DEFINES 0
#include <vulkan/vulkan_core.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdarg.h>
#include <assert.h>
IMAGE_TYPE
typedef struct { struct { int format, flags; } d; uint64_t serial; } tex_t;
typedef struct { int set_format,format; } view_t;
typedef struct { VkFormat vk; int compressed,dec,depth,stencil,bw,bh,bpb; } vkfmt_t;
static struct { VkPhysicalDevice pd; } V;
static vkimg_t image;
static vkfmt_t format={.vk=VK_FORMAT_B10G11R11_UFLOAT_PACK32,.bw=1,.bh=1,.bpb=4},raw={.vk=VK_FORMAT_R32_UINT,.bw=1,.bh=1,.bpb=4},wrong={.vk=VK_FORMAT_R16_SFLOAT,.bw=1,.bh=1,.bpb=2};
static int promotion_supported,raw_supported=1,blits;
static int supported=1,alloc_fail,pass_open,ended,copies,barriers,prepared,view_drops,deferred_images,deferred_views,allocations;
static uint64_t g_img_vkbytes,tex_gen=1;
static VkImageCopy regions[32];
static uint32_t old_pixels[32][8],new_pixels[32][8];
static int defer_sub;
static void log_ignore(const char *fmt,...){(void)fmt;}
#define VKLOG(...) log_ignore(__VA_ARGS__)
#define VKRED_ONCE(id,...) log_ignore(__VA_ARGS__)
static vkimg_t *vk_tex(const tex_t *t){assert(t->d.format==63);return &image;}
static const vkfmt_t *vk_fmt(int f){return f==63?&format:f==10?&raw:f==11?&wrong:NULL;}
static void vkGetPhysicalDeviceFormatProperties(VkPhysicalDevice pd,VkFormat f,VkFormatProperties *p){
 (void)pd;memset(p,0,sizeof *p);int ok=f==VK_FORMAT_R16G16B16A16_SFLOAT?promotion_supported:f==raw.vk?raw_supported:supported;
 p->optimalTilingFeatures=VK_FORMAT_FEATURE_BLIT_SRC_BIT|VK_FORMAT_FEATURE_BLIT_DST_BIT|(ok?VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT:0);
}
static void vk_end_pass_land(void){pass_open=0;ended++;}
static vkimg_t *img_new(const tex_t *t,const vkfmt_t *f,VkImageUsageFlags extra,VkFormat storage_format){
 assert(!pass_open&&t->d.format==63&&t->d.flags==0&&f==&format&&extra==VK_IMAGE_USAGE_STORAGE_BIT);allocations++;
 if(alloc_fail)return NULL;
 vkimg_t *n=calloc(1,sizeof *n);*n=image;n->img=200;n->mem=300;n->sub=(void *)(uintptr_t)400;
 n->usage=image.usage|extra;n->cflags=VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT|VK_IMAGE_CREATE_EXTENDED_USAGE_BIT;n->vkbytes=512;n->layout=VK_IMAGE_LAYOUT_UNDEFINED;
 n->fmt=f->vk;n->promoted=0;n->fixed_fmt=0;
 if(storage_format==f->vk&&!supported&&promotion_supported){n->fmt=VK_FORMAT_R16G16B16A16_SFLOAT;n->promoted=1;}
 return n;
}
static VkCommandBuffer vk_cmd(void){assert(!pass_open);return (VkCommandBuffer)(uintptr_t)5;}
static void vk_tex_prepare(VkCommandBuffer cb,vkimg_t *im){assert(cb==(VkCommandBuffer)(uintptr_t)5&&!pass_open);im->layout=VK_IMAGE_LAYOUT_GENERAL;prepared++;}
static void vk_img_barrier_all(VkCommandBuffer cb){assert(cb==(VkCommandBuffer)(uintptr_t)5&&!pass_open);barriers++;}
static void vkCmdCopyImage(VkCommandBuffer cb,VkImage src,VkImageLayout sl,VkImage dst,VkImageLayout dl,uint32_t n,const VkImageCopy *r){
 assert(cb==(VkCommandBuffer)(uintptr_t)5&&!pass_open&&barriers>=1&&src==100&&dst==200&&n==1);
 assert(sl==VK_IMAGE_LAYOUT_GENERAL&&dl==VK_IMAGE_LAYOUT_GENERAL);
 assert(r->srcSubresource.mipLevel==r->dstSubresource.mipLevel&&r->srcSubresource.layerCount==r->dstSubresource.layerCount);
 assert(!r->srcSubresource.baseArrayLayer&&!r->dstSubresource.baseArrayLayer);
 regions[copies++]=*r;
 for(uint32_t l=0;l<r->srcSubresource.layerCount;l++)new_pixels[r->srcSubresource.mipLevel][l]=old_pixels[r->srcSubresource.mipLevel][l];
}
static void vkCmdBlitImage(VkCommandBuffer cb,VkImage src,VkImageLayout sl,VkImage dst,VkImageLayout dl,uint32_t n,const VkImageBlit *r,VkFilter filter){
 assert(cb==(VkCommandBuffer)(uintptr_t)5&&!pass_open&&src==100&&dst==200&&sl==VK_IMAGE_LAYOUT_GENERAL&&dl==VK_IMAGE_LAYOUT_GENERAL&&n==1&&filter==VK_FILTER_NEAREST);
 assert(r->srcSubresource.layerCount==r->dstSubresource.layerCount);blits++;
}
static void vk_tex_view_drop_all(vkimg_t *im){assert(im==&image&&!pass_open&&barriers==2);view_drops++;tex_gen++;}
static void vk_defer_free_view(VkImageView view){assert(view==80&&view_drops==1);deferred_views++;}
static void vk_defer_free_image(VkImage img,VkDeviceMemory mem){assert(img==100&&mem==101&&view_drops==1);deferred_images++;}
static void vk_defer_free_image_sub(VkImage img,void *sub){assert(img==100&&sub==(void *)(uintptr_t)102&&view_drops==1);deferred_images++;defer_sub++;}
HELPER
static void setup(uint32_t w,uint32_t h,uint32_t d,uint32_t layers,uint32_t levels){
 image=(vkimg_t){.img=100,.mem=101,.fmt=VK_FORMAT_B10G11R11_UFLOAT_PACK32,.aspect=VK_IMAGE_ASPECT_COLOR_BIT,
  .w=w,.h=h,.d=d,.layers=layers,.levels=levels,.samples=1,.usage=0x17,.layout=VK_IMAGE_LAYOUT_GENERAL,
  .pool=7,.off=32,.size=128,.flags=0,.gpu_w=1,.wseq=45,.rseen=23,.packaged=3,.pkg_hash=123,.pkg_stamp=54,
  .pkg_mem=(void *)(uintptr_t)1000,.pkg_len=4096,.lin_mem=(void *)(uintptr_t)2000,.lin_stamp=76,.lin_gen=82,
  .last_use=8,.looked=1,.vkbytes=256,.full=80};
 image.ovl[0]=(void *)(uintptr_t)999;image.novl=1;
 pass_open=1;copies=barriers=prepared=ended=view_drops=deferred_images=deferred_views=allocations=defer_sub=0;
 memset(regions,0,sizeof regions);memset(new_pixels,0,sizeof new_pixels);
 for(int m=0;m<32;m++)for(int l=0;l<8;l++)old_pixels[m][l]=(uint32_t)(100*m+l+1);
 g_img_vkbytes=1024;supported=1;alloc_fail=0;promotion_supported=0;raw_supported=1;blits=0;
}
int main(void){
 tex_t t={.d={63,0}};
 setup(16,8,1,3,5);vkimg_t before=image;uint64_t generation=tex_gen;
 assert(vk_tex_storage_prepare(&t,NULL));assert(ended==1&&allocations==1&&copies==5&&barriers==2&&prepared==2);
 assert(image.img==200&&image.mem==300&&image.sub==(void *)(uintptr_t)400&&image.usage==0x1f);
 assert(g_img_vkbytes==1280&&tex_gen==generation+1&&deferred_images==1&&deferred_views==1&&!defer_sub&&!image.full);
 // Restore only the native fields in a copy: everything else must remain byte-for-byte unchanged.
 vkimg_t tracked=image;tracked.img=before.img;tracked.mem=before.mem;tracked.sub=before.sub;tracked.usage=before.usage;
 tracked.cflags=before.cflags;tracked.layout=before.layout;tracked.fixed_fmt=before.fixed_fmt;tracked.fmt2=before.fmt2;tracked.vkbytes=before.vkbytes;tracked.full=before.full;
 assert(!memcmp(&tracked,&before,sizeof before));
 for(int m=0;m<5;m++){assert(regions[m].extent.width==(16u>>m?16u>>m:1)&&regions[m].extent.height==(8u>>m?8u>>m:1)&&regions[m].extent.depth==1);
  for(int l=0;l<3;l++)assert(new_pixels[m][l]==old_pixels[m][l]);}
 assert(vk_tex_storage_prepare(&t,NULL));assert(allocations==1&&copies==5); // Subsequent storage binds do nothing.
 setup(8,4,4,1,4);image.sub=(void *)(uintptr_t)102;
 assert(vk_tex_storage_prepare(&t,NULL));assert(copies==4&&defer_sub==1);
 assert(regions[0].extent.depth==4&&regions[1].extent.depth==2&&regions[2].extent.depth==1&&regions[3].extent.depth==1);
 for(int m=0;m<4;m++)assert(regions[m].srcSubresource.layerCount==1);
 // Unsupported base format and allocation failure preserve the original allocation and data.
 setup(8,4,1,1,1);supported=0;before=image;assert(!vk_tex_storage_prepare(&t,NULL));
 assert(!memcmp(&before,&image,sizeof image)&&!ended&&!allocations&&!copies&&!deferred_images);
 setup(8,4,1,1,1);alloc_fail=1;before=image;assert(!vk_tex_storage_prepare(&t,NULL));
 assert(!memcmp(&before,&image,sizeof image)&&ended==1&&allocations==1&&!copies&&!deferred_images);
 // Packed R32_UINT writes preserve the original 32-bit backing even when its numeric format cannot support storage.
 view_t r32={1,10};setup(8,4,1,1,1);supported=0;promotion_supported=1;
 assert(vk_tex_storage_prepare(&t,&r32));assert(image.fmt==format.vk&&!image.promoted&&copies==1&&!blits);
 int saved_alloc=allocations;assert(vk_tex_storage_prepare(&t,&r32)&&allocations==saved_alloc);
 // A numeric store still takes the established RGBA16F conversion path.
 setup(8,4,1,1,1);supported=0;promotion_supported=1;
 assert(vk_tex_storage_prepare(&t,NULL));assert(image.fmt==VK_FORMAT_R16G16B16A16_SFLOAT&&image.promoted&&blits==1&&!copies);
 assert(vk_tex_storage_prepare(&t,NULL)&&allocations==1);
 // An existing promoted allocation must not bypass effective-view selection.
 setup(8,4,1,1,1);supported=0;promotion_supported=1;image.fmt=VK_FORMAT_R16G16B16A16_SFLOAT;image.promoted=1;image.usage|=VK_IMAGE_USAGE_STORAGE_BIT;
 assert(vk_tex_storage_prepare(&t,&r32));assert(image.fmt==format.vk&&!image.promoted&&blits==1);
 // A format-listed allocation with STORAGE still needs remaking before a different-format view.
 setup(8,4,1,1,1);image.usage|=VK_IMAGE_USAGE_STORAGE_BIT;image.fixed_fmt=1;
 assert(vk_tex_storage_prepare(&t,&r32)&&allocations==1&&!image.fixed_fmt);
 setup(8,4,1,1,1);view_t bad={1,11};before=image;
 assert(!vk_tex_storage_prepare(&t,&bad)&&!memcmp(&before,&image,sizeof image)&&!allocations);
 setup(8,4,1,1,1);raw_supported=0;before=image;
 assert(!vk_tex_storage_prepare(&t,&r32)&&!memcmp(&before,&image,sizeof image)&&!allocations);
 return 0;
}
'''
        code = code.replace('IMAGE_TYPE', image_type).replace('HELPER', helper)
        target = ROOT / 'target/tests'
        target.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='vk-storage-image-', dir=target) as folder:
            src, exe = Path(folder) / 'check.c', Path(folder) / 'check'
            src.write_text(code)
            subprocess.run(['cc', '-std=c11', '-Wall', '-Wextra', '-Werror', '-fsanitize=address,undefined',
                            '-I', str(ROOT / 'nxnative/dm/third_party/vulkan-headers/include'),
                            str(src), '-o', str(exe)], check=True)
            subprocess.run([str(exe)], check=True, timeout=15)


    @unittest.skipUnless(shutil.which('cc'), 'a host C compiler is required')
    def test_real_storage_views_have_identity_usage_and_one_mip(self):
        header = (ROOT / 'nxnative/dm/android/gfx_vk.h').read_text()
        runtime = (ROOT / 'nxnative/dm/android/gfx_vk_tex.c').read_text()
        image_type = re.search(r'typedef struct vkimg vkimg_t;\nstruct vkimg \{.*?\n\};', header, re.S).group()
        view_type = re.search(r'typedef struct vview \{[^\n]+', runtime).group()
        helpers = '\n'.join(function(runtime, name) for name in [
            'static VkComponentSwizzle swz_comp(', 'static VkComponentSwizzle apply(',
            'static VkComponentMapping sample_swizzle(', 'VkImageView vk_tex_view('])
        code = r'''
#define VK_NO_PROTOTYPES
#define VK_USE_64_BIT_PTR_DEFINES 0
#include <vulkan/vulkan_core.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdarg.h>
#include <pthread.h>
#include <assert.h>
IMAGE_TYPE
typedef struct { struct { int format,flags; } d; int set_swz; uint32_t swz; int stencil; } tex_t;
typedef struct { int set_format,format,set_levels,base_level,nlevels,set_layers,min_layer,nlayers,set_target,target,set_dsmode,dsmode,set_swizzle,swz[4]; } view_t;
typedef struct { VkFormat vk; VkComponentMapping swz; int depth,stencil,compressed,dec,dbpp; } vkfmt_t;
VIEW_TYPE
enum { NVN_TEXTURE_FLAGS_IMAGE=4 };
static struct { VkPhysicalDevice pd; VkDevice dev; } V;
static vkimg_t image;
static vkfmt_t fmt={.vk=VK_FORMAT_B10G11R11_UFLOAT_PACK32},rawfmt={.vk=VK_FORMAT_R32_UINT};
static pthread_mutex_t g_view_mu=PTHREAD_MUTEX_INITIALIZER;
static int g_views,creates,support=1;
static uint64_t g_wseq;
static vview *head;
static VkFormat vk_view_fmt;
static uint32_t vk_view_layers;
static VkImageViewCreateInfo last;
static VkImageUsageFlags last_usage;
static void log_ignore(const char *fmt,...){(void)fmt;}
#define VKLOG(...) log_ignore(__VA_ARGS__)
#define VKRED_ONCE(id,...) log_ignore(__VA_ARGS__)
static vkimg_t *vk_tex(const tex_t *t){(void)t;return &image;}
static void alias_check(vkimg_t *im){(void)im;}
static void vk_watch(const vkimg_t *im,const char *why){(void)im;(void)why;}
static void vk_texdump_note(vkimg_t *im){(void)im;}
static const vkfmt_t *vk_fmt(int f){return f==11?&rawfmt:&fmt;}
static VkImageViewType vtype_of(int target){return target==4?VK_IMAGE_VIEW_TYPE_2D_ARRAY:VK_IMAGE_VIEW_TYPE_2D;}
static VkFormat srgb_twin(VkFormat f){(void)f;return VK_FORMAT_UNDEFINED;}
static vview *nvn_hm_get(int *map,const vkimg_t **key){(void)map;assert(*key==&image);return head;}
static void nvn_hm_put(int *map,const vkimg_t **key,vview *v){(void)map;assert(*key==&image);head=v;}
static void vkGetPhysicalDeviceFormatProperties(VkPhysicalDevice pd,VkFormat f,VkFormatProperties *p){
 (void)pd;assert(f==fmt.vk||f==rawfmt.vk||f==VK_FORMAT_R16G16B16A16_SFLOAT);memset(p,0,sizeof *p);p->optimalTilingFeatures=support?VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT|VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT:0;
}
static VkResult vkCreateImageView(VkDevice d,const VkImageViewCreateInfo *ci,const void *a,VkImageView *out){
 (void)d;(void)a;last=*ci;last_usage=((const VkImageViewUsageCreateInfo *)ci->pNext)->usage;*out=(VkImageView)++creates;return VK_SUCCESS;
}
HELPERS
int main(void){
 image=(vkimg_t){.img=42,.usage=0x1f,.fmt=VK_FORMAT_B10G11R11_UFLOAT_PACK32,.aspect=VK_IMAGE_ASPECT_COLOR_BIT,
  .levels=4,.layers=3,.w=64,.h=32,.d=1,.target=4,.vtype=VK_IMAGE_VIEW_TYPE_2D_ARRAY,.nvn_format=63};
 tex_t tex={.d={63,0},.set_swz=1,.swz=0x31a};vkimg_t *got=NULL;
 VkImageView sampled=vk_tex_view(&tex,NULL,1,1,0,&got);
 assert(sampled&&got==&image&&last.components.a==VK_COMPONENT_SWIZZLE_ONE&&last.subresourceRange.levelCount==4);
 VkImageView storage=vk_tex_view(&tex,NULL,1,1,2,&got);
 assert(storage&&storage!=sampled&&last_usage==VK_IMAGE_USAGE_STORAGE_BIT&&last.subresourceRange.levelCount==1);
 assert(!last.components.r&&!last.components.g&&!last.components.b&&!last.components.a);
 assert(last.subresourceRange.layerCount==3);
 int n=creates;assert(vk_tex_view(&tex,NULL,1,1,2,&got)==storage&&creates==n);
 assert(vk_tex_view(&tex,NULL,1,1,0,&got)==sampled&&creates==n);
 view_t view={.set_levels=1,.base_level=2,.nlevels=2,.set_layers=1,.min_layer=1,.nlayers=2,
              .set_swizzle=1,.swz={4,3,2,0}};
 assert(vk_tex_view(&tex,&view,1,0,2,&got));assert(last.subresourceRange.baseMipLevel==2&&last.subresourceRange.levelCount==1);
 assert(last.subresourceRange.baseArrayLayer==1&&last.subresourceRange.layerCount==1&&!last.components.r&&!last.components.a);
 // A raw integer storage view keeps the same backing and its integer type; no numeric promotion is allowed to disguise it.
 view_t raw_view={.set_format=1,.format=11};
 assert(vk_tex_view(&tex,&raw_view,1,0,2,&got));assert(last.format==VK_FORMAT_R32_UINT&&got==&image&&last_usage==VK_IMAGE_USAGE_STORAGE_BIT);
 image.promoted=1;image.fmt=VK_FORMAT_R16G16B16A16_SFLOAT;
 assert(!vk_tex_view(&tex,&raw_view,1,0,2,&got));
 assert(vk_tex_view(&tex,NULL,1,1,2,&got));assert(last.format==VK_FORMAT_R16G16B16A16_SFLOAT);
 image.promoted=0;image.fmt=fmt.vk;
 // RegisterImage adds usage intent to the snapshot, not to an already allocated image.
 // Sampling that shared old backing must not request unsupported STORAGE view usage.
 tex.d.flags=NVN_TEXTURE_FLAGS_IMAGE;image.usage=0x17;
 view.base_level=1;view.nlevels=1;assert(vk_tex_view(&tex,&view,1,0,0,&got));
 assert(!(last_usage&VK_IMAGE_USAGE_STORAGE_BIT));
 // Storage itself still requires successful preflight; after upgrade a new sampled view
 // may include STORAGE only because the real allocation now supports it.
 assert(!vk_tex_view(&tex,NULL,1,0,2,&got));image.usage=0x1f;
 view.base_level=2;assert(vk_tex_view(&tex,&view,1,0,0,&got));
 assert(last_usage&VK_IMAGE_USAGE_STORAGE_BIT);
 // An unsupported view format is refused even when the allocation supports storage.
 support=0;view.base_level=3;assert(!vk_tex_view(&tex,&view,1,0,2,&got));
 while(head){vview *next=head->next;free(head);head=next;}
 return 0;
}
'''
        code = code.replace('IMAGE_TYPE', image_type).replace('\nVIEW_TYPE\n', '\n' + view_type + '\n').replace('HELPERS', helpers)
        # The real hmap takes untyped keys; keep this mock equivalent for C's pointer qualifiers.
        code = code.replace('const vkimg_t **key', 'vkimg_t **key')
        target = ROOT / 'target/tests'
        target.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='vk-storage-view-', dir=target) as folder:
            src, exe = Path(folder) / 'check.c', Path(folder) / 'check'
            src.write_text(code)
            subprocess.run(['cc', '-std=c11', '-Wall', '-Wextra', '-Werror',
                            '-Wno-misleading-indentation', '-Wno-sign-compare', '-fsanitize=address,undefined',
                            '-I', str(ROOT / 'nxnative/dm/third_party/vulkan-headers/include'),
                            str(src), '-pthread', '-o', str(exe)], check=True)
            subprocess.run([str(exe)], check=True, timeout=15)

    def test_preflight_and_cache_distinguish_storage(self):
        runtime = (ROOT / 'nxnative/dm/android/gfx_vk_exec.c').read_text()
        draw = function(runtime, 'static int prepare_draw(')
        self.assertLess(draw.index('stage_storage_prepare('), draw.index('if (!begin_pass()'))
        dispatch = function(runtime, 'static void dispatch(')
        self.assertLess(dispatch.index('stage_storage_prepare('), dispatch.index('stage_set('))
        texture = function(runtime, 'static int tex_desc_for(')
        self.assertIn('g_tvc[ck].storage == storage', texture)
        self.assertIn('g_tvc[ck].storage = storage', texture)
        self.assertIn('storage ? 2 : 0', texture)
        self.assertIn('if (storage) g_tvc[ck].im->gpu_w = 1;', texture)


if __name__ == '__main__':
    unittest.main()
