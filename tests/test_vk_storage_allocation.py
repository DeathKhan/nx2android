"""Actual image creation must distinguish packed storage aliases from numeric stores."""
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest
from tests.test_vk_storage_image import function

ROOT = Path(__file__).resolve().parents[1]


class StorageAllocation(unittest.TestCase):
    @unittest.skipUnless(shutil.which('cc'), 'C compiler required')
    def test_actual_image_creation_contract(self):
        header = (ROOT / 'nxnative/dm/android/gfx_vk.h').read_text()
        runtime = (ROOT / 'nxnative/dm/android/gfx_vk_tex.c').read_text()
        image = re.search(r'typedef struct vkimg vkimg_t;\nstruct vkimg \{.*?\n\};', header, re.S).group()
        # img_new calls vk_int_fmt (DM_VK_UBWC bit 4: integer storage views, FINAL FANTASY TACTICS); it only needs V.pd and the
        # format-properties stub below, so the production function is cut in rather than stubbed
        helpers = '\n'.join(function(runtime, n) for n in ['VkFormat vk_int_fmt(', 'static VkFormat storage_promote(', 'static vkimg_t *img_new('])
        code = r'''
#define VK_NO_PROTOTYPES
#define VK_USE_64_BIT_PTR_DEFINES 0
#include <vulkan/vulkan_core.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
IMAGE
typedef struct {uint32_t width,height,depth,levels,samples,flags;int format,target;} nvn_texdesc;
typedef struct {nvn_texdesc d;void *mem;struct {uint64_t size;} L;} tex_t;
typedef struct {VkFormat vk;VkComponentMapping swz;int compressed,dec,bpb;} vkfmt_t;
enum {NVN_TEXTURE_FLAGS_IMAGE=8};
static struct {VkPhysicalDevice pd;VkDevice dev;} V;
static VkImageCreateInfo last;
static int ubwc=3,create_count;
#define VKLOG(...) ((void)0)
#define VKRED_ONCE(...) ((void)0)
#define VKCHK(x) assert((x)==VK_SUCCESS)
static int ubwc_mode(void){return ubwc;}
static uint32_t layers_of(const nvn_texdesc *d){(void)d;return 1;}
static VkImageViewType vtype_of(int d){(void)d;return VK_IMAGE_VIEW_TYPE_2D;}
static VkImageAspectFlags aspect_of(const vkfmt_t *f){(void)f;return VK_IMAGE_ASPECT_COLOR_BIT;}
static int zc_tex_wants(void *p,uint64_t n){(void)p;(void)n;return 0;}
static VkFormat srgb_twin(VkFormat f){(void)f;return VK_FORMAT_UNDEFINED;}
static void vkGetPhysicalDeviceFormatProperties(VkPhysicalDevice pd,VkFormat f,VkFormatProperties *p){
 (void)pd;memset(p,0,sizeof *p);p->optimalTilingFeatures=VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT;
 if(f==VK_FORMAT_R32_UINT||f==VK_FORMAT_R16G16B16A16_SFLOAT)p->optimalTilingFeatures|=VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT;
}
static VkResult vkCreateImage(VkDevice d,const VkImageCreateInfo *ci,const void *a,VkImage *out){
 (void)d;(void)a;last=*ci;*out=++create_count;return VK_SUCCESS;
}
static void vkGetImageMemoryRequirements(VkDevice d,VkImage image,VkMemoryRequirements *r){(void)d;(void)image;*r=(VkMemoryRequirements){256,16,1};}
static int vk_mem_type(uint32_t mask,VkMemoryPropertyFlags flags,VkMemoryPropertyFlags avoid){(void)mask;(void)flags;(void)avoid;return 0;}
static VkDeviceMemory vk_image_memory(const VkMemoryRequirements *r,int type,VkDeviceSize *off,void **sub){(void)r;(void)type;*off=0;*sub=NULL;return 42;}
static VkResult vkBindImageMemory(VkDevice d,VkImage i,VkDeviceMemory m,VkDeviceSize o){(void)d;(void)i;(void)m;(void)o;return VK_SUCCESS;}
static void vkDestroyImage(VkDevice d,VkImage i,const void *a){(void)d;(void)i;(void)a;}
HELPERS
int main(void){
 tex_t t={.d={.width=64,.height=32,.depth=1,.levels=1,.samples=1,.flags=NVN_TEXTURE_FLAGS_IMAGE,.format=64,.target=1}};
 vkfmt_t f={.vk=VK_FORMAT_E5B9G9R9_UFLOAT_PACK32,.bpb=4};
 // IMAGE alone provides no evidence whether guest stores packed bits or numeric RGB.
 vkimg_t *im=img_new(&t,&f,0,VK_FORMAT_UNDEFINED);assert(im&&im->fmt==f.vk&&!im->promoted);
 assert(!(last.usage&VK_IMAGE_USAGE_STORAGE_BIT));free(im);
 for(ubwc=0;ubwc<=3;ubwc+=3){
  im=img_new(&t,&f,VK_IMAGE_USAGE_STORAGE_BIT,VK_FORMAT_R32_UINT);
  assert(im&&im->fmt==f.vk&&!im->promoted&&!im->fixed_fmt&&(last.usage&VK_IMAGE_USAGE_STORAGE_BIT));
  assert((last.flags&(VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT|VK_IMAGE_CREATE_EXTENDED_USAGE_BIT))==(VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT|VK_IMAGE_CREATE_EXTENDED_USAGE_BIT));
  assert(!last.pNext);free(im);
 }
 ubwc=3;im=img_new(&t,&f,VK_IMAGE_USAGE_STORAGE_BIT,f.vk);
 assert(im&&im->promoted&&im->fmt==VK_FORMAT_R16G16B16A16_SFLOAT&&(last.usage&VK_IMAGE_USAGE_STORAGE_BIT));free(im);
 // An already supported native storage format stays native.
 f.vk=VK_FORMAT_R32_UINT;im=img_new(&t,&f,VK_IMAGE_USAGE_STORAGE_BIT,f.vk);
 assert(im&&!im->promoted&&im->fmt==f.vk&&(last.usage&VK_IMAGE_USAGE_STORAGE_BIT));free(im);
 return 0;
}
'''
        code = code.replace('IMAGE\n', image + '\n').replace('HELPERS', helpers)
        scratch = ROOT / 'target/tests'
        scratch.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='storage-allocation-', dir=scratch) as folder:
            p = Path(folder)
            (p / 'check.c').write_text(code)
            subprocess.run(['cc', '-std=gnu11', '-Wall', '-Wextra', '-Werror', '-fsanitize=address,undefined',
                            '-I', str(ROOT / 'nxnative/dm/third_party/vulkan-headers/include'),
                            str(p / 'check.c'), '-o', str(p / 'check')], check=True)
            subprocess.run([str(p / 'check')], check=True)


if __name__ == '__main__':
    unittest.main()
