"""Bounded image-view diagnostics execute the production resolver without GPU operations."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
import sys; sys.path.insert(0, str(Path(__file__).resolve().parent))
import host_cc   # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

def extract(source, declaration):
    start = source.index(declaration)
    pos = source.index("{", start) + 1
    depth = 1
    while depth:
        depth += (source[pos] == "{") - (source[pos] == "}")
        pos += 1
    return source[start:pos]

class UpscalerImageDiagnostic(unittest.TestCase):
    @unittest.skipUnless(shutil.which("cc"), "host compiler required")
    @unittest.skipUnless(host_cc.posix() and host_cc.sanitizers(), 'a POSIX host C compiler with ASan and UBSan is required (MinGW has no setenv, libasan or libubsan)')
    def test_actual_resolver_reasons_and_log_budget(self):
        source = (ROOT / "nxnative/dm/android/gfx_vk_exec.c").read_text(encoding='utf-8')
        helpers = "\n".join(extract(source, x) for x in (
            "static void upscaler_image_diag(", "static int upscaler_image_at(", "static int upscaler_alpha_compatible("))
        poll = extract(source, '{ const char *e = getenv("DM_VK_UPSCALER_DIAG")')
        decl = next(x for x in source.splitlines() if x.startswith("typedef struct ") and x.endswith(" upscaler_image;"))
        code = r'''
#define _POSIX_C_SOURCE 200809L
#include <vulkan/vulkan_core.h>
#include <assert.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>
#include <string.h>
enum {NVN_TEXTURE_TARGET_2D=1,NVN_TEXTURE_TARGET_2D_ARRAY=4};
typedef struct {int format,target;uint32_t width,height;} desc_t;
typedef struct {uint64_t serial,off;void *pool;desc_t d;int set_swz;uint16_t swz;} tex_t;
typedef struct {int set_format,format,set_target,target,set_levels,set_layers,set_swizzle;
 uint32_t base_level,nlevels,min_layer,nlayers;uint8_t swz[4];} view_t;
typedef struct {VkImage img;VkFormat fmt;VkImageUsageFlags usage;VkImageAspectFlags aspect;VkImageViewType vtype;
 uint32_t w,h,d,levels,layers,samples;int promoted,nvn_format;} vkimg_t;
typedef struct {VkFormat vk;int is_int,compressed,dec;VkComponentMapping swz;} vkfmt_t;
typedef struct {uint64_t key;} shader_t;
DECL
static int g_up_diag_left,g_up_diag_request,logs,lookups,images,formats,registered=1,has_view=1;
static uint64_t g_tex_frame=91;
static struct {uint64_t tex[6][256],img[6][32];shader_t *sh[6];} S;
static tex_t texture;static view_t view;static vkimg_t native;static vkfmt_t format;
static char last[4096];
static uint64_t vk_tex_slot_handle(const uint64_t *tex,const uint64_t *img,unsigned r){(void)img;return tex[r];}
static int vk_tex_by_id(int id,const view_t **v,const tex_t **t){assert(id==55);lookups++;*v=has_view?&view:NULL;*t=&texture;return registered;}
static vkimg_t *vk_tex(const tex_t *t){assert(t->serial==texture.serial);images++;return &native;}
static const vkfmt_t *vk_fmt(int f){assert(f==(has_view&&view.set_format?view.format:(int)texture.d.format));formats++;return &format;}
static void logger(const char *fmt,...){va_list a;va_start(a,fmt);vsnprintf(last,sizeof last,fmt,a);va_end(a);logs++;}
#define VKLOG(...) logger(__VA_ARGS__)
HELPERS
static void poll(void) POLL
int main(void){
 shader_t sh={UINT64_C(0x4f41ccf24f163e0d)};S.sh[5]=&sh;S.img[5][1]=55;S.tex[5][4]=55;
 texture=(tex_t){.serial=6485,.pool=(void *)(uintptr_t)0x1234,.off=0x348400,.d={.format=0x38,.target=1,.width=2560,.height=1440}};
 native=(vkimg_t){.img=(VkImage)(uintptr_t)7,.fmt=VK_FORMAT_R8G8B8A8_SRGB,.usage=VK_IMAGE_USAGE_TRANSFER_DST_BIT,
 .aspect=VK_IMAGE_ASPECT_COLOR_BIT,.vtype=VK_IMAGE_VIEW_TYPE_2D,.w=2560,.h=1440,.d=1,.levels=1,.layers=1,.samples=1};
 view=(view_t){.set_format=1,.format=0x25};format.vk=VK_FORMAT_R8G8B8A8_UNORM;
 upscaler_image u={0};poll();assert(!upscaler_image_at(55,0,&u)&&!logs&&lookups==1&&images==1&&formats==1);
 setenv("DM_VK_UPSCALER_DIAG","1",1);poll();u=(upscaler_image){0};
 assert(!upscaler_image_at(55,0,&u)&&logs==1&&!g_up_diag_left);
 assert(strstr(last,"key 4f41ccf24f163e0d")&&strstr(last,"output slots 0x2")&&strstr(last,"view/native format reinterpretation"));
 assert(strstr(last,"requested VkFormat 37 native VkFormat 43")&&strstr(last,"serial 6485")&&strstr(last,"0x348400"));
 u=(upscaler_image){0};assert(!upscaler_image_at(55,0,&u)&&logs==1);poll();assert(!g_up_diag_left);
 setenv("DM_VK_UPSCALER_DIAG","0",1);poll();setenv("DM_VK_UPSCALER_DIAG","4",1);poll();assert(g_up_diag_left==4);
 native.fmt=format.vk;u=(upscaler_image){0};assert(upscaler_image_at(55,0,&u)&&u.w==2560&&u.h==1440&&logs==2&&strstr(last,"accepted"));
 texture.set_swz=1;u=(upscaler_image){0};assert(!upscaler_image_at(55,1,&u)&&logs==3&&strstr(last,"sampled slots 0x10")&&strstr(last,"texture/view channel swizzle"));
 texture.set_swz=0;view.set_levels=1;view.base_level=3;u=(upscaler_image){0};
 assert(!upscaler_image_at(55,0,&u)&&logs==4&&strstr(last,"view mip/layer outside native image"));
 registered=0;u=(upscaler_image){0};int old_images=images;assert(!upscaler_image_at(55,0,&u)&&logs==5&&images==old_images&&strstr(last,"texture ID not registered"));
 setenv("DM_VK_UPSCALER_DIAG","999",1);poll();assert(g_up_diag_left==256);
 for(int i=0;i<300;i++){u=(upscaler_image){0};assert(!upscaler_image_at(0,0,&u));}
 assert(logs==261&&!g_up_diag_left);setenv("DM_VK_UPSCALER_DIAG","-1",1);poll();assert(!g_up_diag_left);
 // The narrow movie-copy opt-in cannot relax ordinary or sampled blit contracts.
 registered=1;native.fmt=VK_FORMAT_R8G8B8A8_SRGB;native.levels=4;native.layers=3;
 view=(view_t){.set_format=1,.format=0x25,.set_levels=1,.base_level=2,.nlevels=1,.set_layers=1,.min_layer=1,.nlayers=1};
 format.vk=VK_FORMAT_R8G8B8A8_UNORM;
 u=(upscaler_image){0};assert(!upscaler_image_at(55,0,&u));
 u=(upscaler_image){.movie_output=1};assert(upscaler_image_at(55,0,&u)&&u.format==VK_FORMAT_R8G8B8A8_UNORM);
 assert(u.level==2&&u.layer==1&&u.w==640&&u.h==360);
 u=(upscaler_image){.movie_output=1};assert(!upscaler_image_at(55,1,&u));
 format.vk=VK_FORMAT_B8G8R8A8_UNORM;
 u=(upscaler_image){.movie_output=1};assert(!upscaler_image_at(55,0,&u));
 format.vk=VK_FORMAT_R8G8B8A8_SRGB;native.fmt=VK_FORMAT_R8G8B8A8_UNORM;
 u=(upscaler_image){.movie_output=1};assert(!upscaler_image_at(55,0,&u));
 format.vk=VK_FORMAT_R8G8B8A8_UNORM;native.fmt=VK_FORMAT_R8G8B8A8_SRGB;view.base_level=32;
 u=(upscaler_image){.movie_output=1};assert(!upscaler_image_at(55,0,&u));
 view.base_level=2;view.min_layer=3;
 u=(upscaler_image){.movie_output=1};assert(!upscaler_image_at(55,0,&u));
 // RGB storage promotion preserves RGB but its RGBA backing has no guaranteed alpha.
 view=(view_t){0};texture.d.format=0x40;texture.set_swz=1;texture.swz=0x31a;
 native.promoted=1;native.nvn_format=0x40;native.fmt=VK_FORMAT_R16G16B16A16_SFLOAT;format.vk=VK_FORMAT_E5B9G9R9_UFLOAT_PACK32;
 u=(upscaler_image){0};assert(!upscaler_image_at(55,1,&u)); // ordinary passthrough remains strict
 u=(upscaler_image){.allow_promotion=1};assert(upscaler_image_at(55,1,&u)&&u.needs_opaque_alpha);
 vkimg_t target={.fmt=VK_FORMAT_B10G11R11_UFLOAT_PACK32};upscaler_image out={.im=&target};
 assert(upscaler_alpha_compatible(&u,&out));target.fmt=VK_FORMAT_E5B9G9R9_UFLOAT_PACK32;assert(upscaler_alpha_compatible(&u,&out));
 target.fmt=VK_FORMAT_R16G16B16A16_SFLOAT;assert(!upscaler_alpha_compatible(&u,&out));target.fmt=VK_FORMAT_R8G8B8A8_UNORM;assert(!upscaler_alpha_compatible(&u,&out));
 texture.set_swz=0;u=(upscaler_image){.allow_promotion=1};assert(upscaler_image_at(55,1,&u)&&u.needs_opaque_alpha); // default alpha also unsafe
 view.set_format=1;view.format=0x3f;format.vk=VK_FORMAT_B10G11R11_UFLOAT_PACK32;
 u=(upscaler_image){.allow_promotion=1};assert(!upscaler_image_at(55,1,&u)); // different guest RGB view is a reinterpretation
 view.format=texture.d.format;format.vk=VK_FORMAT_E5B9G9R9_UFLOAT_PACK32;
 u=(upscaler_image){.allow_promotion=1};assert(upscaler_image_at(55,1,&u));
 native.promoted=0;u=(upscaler_image){.allow_promotion=1};assert(!upscaler_image_at(55,1,&u));
 native.promoted=1;native.fmt=VK_FORMAT_R8G8B8A8_UNORM;u=(upscaler_image){.allow_promotion=1};assert(!upscaler_image_at(55,1,&u));
 native.fmt=VK_FORMAT_R16G16B16A16_SFLOAT;format.vk=VK_FORMAT_R16G16B16A16_SFLOAT;view.format=0x55;
 u=(upscaler_image){.allow_promotion=1};assert(!upscaler_image_at(55,1,&u)); // native-format equality cannot disguise a guest reinterpretation
 view=(view_t){0};texture.d.format=0x3f;format.vk=VK_FORMAT_B10G11R11_UFLOAT_PACK32;
 u=(upscaler_image){.allow_promotion=1};assert(!upscaler_image_at(55,1,&u)); // alias has a different original guest format
 native.nvn_format=texture.d.format;texture.set_swz=1;texture.swz=0x31a;
 u=(upscaler_image){.allow_promotion=1};assert(upscaler_image_at(55,1,&u)&&u.needs_opaque_alpha);
 texture.swz=0x31b;u=(upscaler_image){.allow_promotion=1};assert(!upscaler_image_at(55,1,&u)); // nonidentity RGB selectors stay rejected
 native.promoted=0;native.fmt=VK_FORMAT_B10G11R11_UFLOAT_PACK32;texture.swz=0x31a;
 u=(upscaler_image){0};assert(upscaler_image_at(55,1,&u)&&!u.needs_opaque_alpha);assert(upscaler_alpha_compatible(&u,&out));
 return 0;
}
'''
        for key, value in (("DECL",decl),("HELPERS",helpers),("POLL",poll)):
            code=code.replace(key,value)
        target=ROOT/"target/tests";target.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="upscaler-diag-",dir=target) as folder:
            src,exe=Path(folder)/"check.c",Path(folder)/"check";src.write_text(code, encoding='utf-8')
            subprocess.run(["cc","-std=c11","-Wall","-Wextra","-Werror","-fsanitize=address,undefined",
                "-I"+str(ROOT/"nxnative/dm/third_party/vulkan-headers/include"),str(src),"-o",str(exe)],check=True)
            subprocess.run([str(exe)],check=True,timeout=15)

if __name__=="__main__":unittest.main()
