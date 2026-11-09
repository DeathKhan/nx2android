"""Exercise the actual one-shot compute trace and bounded image capture helpers."""
from pathlib import Path
import shutil
import json
import subprocess
import tempfile
import unittest
ROOT=Path(__file__).resolve().parents[1]
def extract(s, declaration):
    a=s.index(declaration);b=s.index("{",a)+1;depth=1
    while depth:
        depth+=(s[b]=="{")-(s[b]=="}");b+=1
    return s[a:b]
class ComputeTrace(unittest.TestCase):
    @unittest.skipUnless(shutil.which("cc"),"host compiler required")
    def test_rgb9e5_actual_decoder(self):
        src=(ROOT/"nxnative/dm/android/gfx_vk_exec.c").read_text()
        decoder=extract(src,"static void rgb9e5_floats(")
        code="#include <stdint.h>\n#include <math.h>\n#include <assert.h>\n"+decoder+r"""
int main(void){
 float rgb[3];rgb9e5_floats(0,rgb);assert(rgb[0]==0&&rgb[1]==0&&rgb[2]==0);
 rgb9e5_floats(0x7c020100u,rgb);assert(rgb[0]==.5f&&rgb[1]==.5f&&rgb[2]==.5f);
 rgb9e5_floats(0xffffffffu,rgb);assert(rgb[0]==65408&&rgb[1]==65408&&rgb[2]==65408);
 rgb9e5_floats(1,rgb);assert(rgb[0]==0x1p-24f&&rgb[1]==0&&rgb[2]==0);
 for(unsigned e=0;e<32;e++)for(unsigned m=0;m<512;m++){
  unsigned g=(m+7)&511,b=(m+37)&511;rgb9e5_floats((e<<27)|(b<<18)|(g<<9)|m,rgb);
  assert(rgb[0]==(float)ldexp((double)m,(int)e-24));
  assert(rgb[1]==(float)ldexp((double)g,(int)e-24)&&rgb[2]==(float)ldexp((double)b,(int)e-24));
  assert(isfinite(rgb[0])&&isfinite(rgb[1])&&isfinite(rgb[2]));
 }return 0;
}
"""
        self.assertIn("case VK_FORMAT_E5B9G9R9_UFLOAT_PACK32:",extract(src,"static uint32_t cs_trace_bpp("))
        writer=extract(src,"void vk_pass_dump_poll(")
        self.assertIn("rgb9e5_floats",writer)
        self.assertGreaterEqual(writer.count("d->f == VK_FORMAT_E5B9G9R9_UFLOAT_PACK32"),4)
        target=ROOT/"target/tests";target.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="rgb9e5-",dir=target) as folder:
            c,exe=Path(folder)/"test.c",Path(folder)/"test";c.write_text(code)
            subprocess.run(["cc","-std=c11","-Wall","-Wextra","-Werror","-fsanitize=address,undefined",str(c),"-lm","-o",str(exe)],check=True)
            subprocess.run([str(exe)],check=True,timeout=10)

    @unittest.skipUnless(shutil.which("cc"),"host compiler required")
    def test_actual_parser_and_capture_helpers(self):
        src=(ROOT/"nxnative/dm/android/gfx_vk_exec.c").read_text()
        functions="\n".join(extract(src,x) for x in (
            "static void cs_trace_poll(","static int cs_trace_match(","static uint32_t cs_trace_bpp(",
            "static uint64_t cs_trace_image_bytes(","static void cs_trace_image_note(","static void cs_trace_images_copy("))
        pd=next(x for x in src.splitlines() if x.startswith("typedef struct ") and x.endswith(" pdump_t;"))
        ci=next(x for x in src.splitlines() if x.startswith("typedef struct ") and x.endswith(" cs_trace_image;"))
        code=r"""
#define _POSIX_C_SOURCE 200809L
#include <vulkan/vulkan_core.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <assert.h>
typedef struct {VkImage img;VkFormat fmt;VkImageUsageFlags usage;VkImageAspectFlags aspect;VkImageViewType vtype;uint32_t samples,d,levels,layers,w,h;} vkimg_t;
typedef struct {uint32_t binding,flags;} dmsv_bind;
typedef struct {struct {int format;} d;} tex_t;
typedef struct {int set_levels,set_layers,set_format,format,set_target,target;uint32_t base_level,min_layer,nlayers;} view_t;
typedef struct {VkFormat vk;} vkfmt_t;
DECL_PD
DECL_CI
enum {CS_TRACE_KEYS=16,CS_TRACE_DISPATCHES=32,CS_TRACE_IMAGES=8,CS_TRACE_COPIES=16,PD_MAX=96};
static uint64_t g_cs_trace_keys[CS_TRACE_KEYS],g_cs_trace_bytes;
static unsigned g_cs_trace_n,g_cs_trace_seen,g_cs_trace_copies,g_cs_trace_wait,g_cs_trace_found,g_cs_buffer_copies,g_cs_images_n,g_cs_images_copies;
static int g_cs_trace_dump,g_cs_trace_collect,g_cs_trace_buffers;
static unsigned g_cs_trace_frames;static uint64_t g_cs_trace_start_present;
static cs_trace_image g_cs_images[CS_TRACE_IMAGES];
static pdump_t g_pd[PD_MAX];static int g_npd,g_pd_state,g_sub_pass=37;
static uint64_t g_tex_frame=419;
#define NVN_TEXTURE_TARGET_3D 3
static uint64_t vk_frame_current(void){return 501;}
static uint64_t mtl_last_fence(void){return 701;}
static struct {VkDevice dev;} V;
static int logs,copy_count,barriers,host_barriers,destroyed,freed,fail_alloc;
static uint64_t handle=10;static VkBufferImageCopy last_copy;
static char last[2048];
static void logger(const char *f,...){va_list a;va_start(a,f);vsnprintf(last,sizeof last,f,a);va_end(a);logs++;}
#define VKLOG(...) logger(__VA_ARGS__)
static const vkfmt_t *vk_fmt(int f){static vkfmt_t x;x.vk=(VkFormat)f;return &x;}
static uint32_t vk_mem_type(uint32_t b,VkMemoryPropertyFlags w,VkMemoryPropertyFlags a){(void)b;(void)w;(void)a;return 0;}
static void vk_img_barrier_all(VkCommandBuffer b){(void)b;barriers++;}
VkResult vkCreateBuffer(VkDevice d,const VkBufferCreateInfo *i,const VkAllocationCallbacks *a,VkBuffer *b){(void)d;(void)a;assert(i->size);*b=(VkBuffer)(uintptr_t)++handle;return VK_SUCCESS;}
void vkGetBufferMemoryRequirements(VkDevice d,VkBuffer b,VkMemoryRequirements *r){(void)d;(void)b;*r=(VkMemoryRequirements){4096,64,1};}
VkResult vkAllocateMemory(VkDevice d,const VkMemoryAllocateInfo *i,const VkAllocationCallbacks *a,VkDeviceMemory *m){(void)d;(void)i;(void)a;if(fail_alloc)return VK_ERROR_OUT_OF_HOST_MEMORY;*m=(VkDeviceMemory)(uintptr_t)++handle;return VK_SUCCESS;}
VkResult vkBindBufferMemory(VkDevice d,VkBuffer b,VkDeviceMemory m,VkDeviceSize o){(void)d;(void)b;(void)m;(void)o;return VK_SUCCESS;}
VkResult vkMapMemory(VkDevice d,VkDeviceMemory m,VkDeviceSize o,VkDeviceSize n,VkMemoryMapFlags f,void **p){(void)d;(void)m;(void)o;(void)n;(void)f;*p=(void *)(uintptr_t)1;return VK_SUCCESS;}
void vkDestroyBuffer(VkDevice d,VkBuffer b,const VkAllocationCallbacks *a){(void)d;(void)b;(void)a;destroyed++;}
void vkFreeMemory(VkDevice d,VkDeviceMemory m,const VkAllocationCallbacks *a){(void)d;(void)m;(void)a;freed++;}
void vkCmdCopyImageToBuffer(VkCommandBuffer c,VkImage i,VkImageLayout l,VkBuffer b,uint32_t n,const VkBufferImageCopy *p){(void)c;(void)i;(void)b;assert(l==VK_IMAGE_LAYOUT_GENERAL&&n==1);copy_count++;last_copy=*p;}
void vkCmdPipelineBarrier(VkCommandBuffer c,VkPipelineStageFlags src,VkPipelineStageFlags dst,VkDependencyFlags dep,uint32_t n,const VkMemoryBarrier *m,uint32_t bn,const VkBufferMemoryBarrier *b,uint32_t in,const VkImageMemoryBarrier *i){
 (void)c;(void)dep;(void)b;(void)i;assert(src==VK_PIPELINE_STAGE_TRANSFER_BIT&&dst==VK_PIPELINE_STAGE_HOST_BIT&&n==1&&!bn&&!in);
 assert(m->srcAccessMask==VK_ACCESS_TRANSFER_WRITE_BIT&&m->dstAccessMask==VK_ACCESS_HOST_READ_BIT);host_barriers++;
}

FUNCTIONS
static void reset_capture(void){g_cs_images_n=g_cs_images_copies=g_cs_trace_copies=0;g_cs_trace_bytes=0;g_npd=0;g_pd_state=1;g_cs_trace_collect=1;}
int main(void){
 cs_trace_poll();assert(!g_cs_trace_n&&!cs_trace_match(1));
 setenv("DM_VK_TRACE_CS","abc",1);cs_trace_poll();assert(g_cs_trace_n==1&&g_cs_trace_wait==600);
 for(int i=0;i<5;i++){assert(!cs_trace_match(0xdef));cs_trace_poll();assert(g_cs_trace_n==1&&!g_cs_trace_seen);}
 assert(cs_trace_match(0xabc));cs_trace_poll();assert(!g_cs_trace_n&&strstr(last,"completed requested-key window"));
 setenv("DM_VK_TRACE_CS","a,b",1);cs_trace_poll();assert(cs_trace_match(0xa));cs_trace_poll();
 assert(g_cs_trace_n==2&&g_cs_trace_found==1);assert(cs_trace_match(0xb));cs_trace_poll();assert(!g_cs_trace_n);
 setenv("DM_VK_TRACE_CS","123",1);cs_trace_poll();for(int i=0;i<600;i++)cs_trace_poll();
 assert(!g_cs_trace_n&&strstr(last,"expired"));
 setenv("DM_VK_TRACE_CS","123",1);cs_trace_poll();setenv("DM_VK_TRACE_CS","cancel",1);cs_trace_poll();assert(!g_cs_trace_n&&strstr(last,"cancelled"));
 setenv("DM_VK_TRACE_CS","0xfa, d17,fa",1);setenv("DM_VK_TRACE_CS_DUMP","1",1);cs_trace_poll();
 assert(g_cs_trace_n==2&&g_cs_trace_dump&&!strcmp(getenv("DM_VK_TRACE_CS"),"0"));
 assert(!cs_trace_match(4));for(int i=0;i<32;i++)assert(cs_trace_match(0xfa));
 assert(!cs_trace_match(0xd17)&&g_cs_trace_seen==33);cs_trace_poll();assert(!g_cs_trace_n&&!cs_trace_match(0xfa));
 setenv("DM_VK_TRACE_CS_BUFFERS","1",1);setenv("DM_VK_TRACE_CS","a,b",1);cs_trace_poll();assert(g_cs_trace_frames==2);
 assert(cs_trace_match(0xa)&&cs_trace_match(0xb));cs_trace_poll();assert(g_cs_trace_n==2);
 g_tex_frame++;cs_trace_poll();assert(g_cs_trace_n==2);g_tex_frame++;cs_trace_poll();assert(g_cs_trace_n==2);
 g_tex_frame++;cs_trace_poll();assert(!g_cs_trace_n);g_tex_frame=419;setenv("DM_VK_TRACE_CS_BUFFERS","0",1);
 setenv("DM_VK_TRACE_CS","12345678901234567",1);cs_trace_poll();assert(!g_cs_trace_n);
 setenv("DM_VK_TRACE_CS","a,z",1);cs_trace_poll();assert(g_cs_trace_n==1);
 setenv("DM_VK_TRACE_CS","1,2,3,4,5,6,7,8,9,a,b,c,d,e,f,10,11",1);cs_trace_poll();assert(g_cs_trace_n==16);
 vkimg_t im={.img=(VkImage)(uintptr_t)9,.fmt=VK_FORMAT_R8G8B8A8_SRGB,.usage=VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
 .aspect=VK_IMAGE_ASPECT_COLOR_BIT,.vtype=VK_IMAGE_VIEW_TYPE_2D_ARRAY,.samples=1,.d=1,.levels=4,.layers=3,.w=64,.h=32};
 tex_t t={.d={VK_FORMAT_R8G8B8A8_SRGB}};view_t v={.set_levels=1,.set_layers=1,.set_format=1,.format=VK_FORMAT_R8G8B8A8_UNORM,.base_level=2,.min_layer=1,.nlayers=1};
 dmsv_bind bind={.binding=7,.flags=1};VkDescriptorImageInfo ii={.imageView=(VkImageView)(uintptr_t)5};reset_capture();
 cs_trace_image_note(&bind,&ii,&im,&t,&v,0);assert(g_cs_images_n==1&&g_cs_trace_copies==1);
 cs_trace_image_note(&bind,&ii,&im,&t,&v,0);assert(g_cs_images_n==1);
 cs_trace_image_note(&bind,&ii,&im,&t,&v,1);assert(g_cs_images_n==2&&g_cs_trace_copies==3&&g_cs_trace_bytes==16*8*4*3);
 assert(g_cs_images[0].level==2&&g_cs_images[0].layer==1&&g_cs_images[0].view_format==VK_FORMAT_R8G8B8A8_UNORM);
 cs_trace_images_copy(VK_NULL_HANDLE,0xfa,0);assert(copy_count==2&&g_npd==2&&barriers==4);
 assert(last_copy.imageSubresource.mipLevel==2&&last_copy.imageSubresource.baseArrayLayer==1&&last_copy.imageExtent.width==16&&last_copy.imageExtent.height==8);
 cs_trace_images_copy(VK_NULL_HANDLE,0xfa,1);assert(copy_count==3&&g_npd==3&&g_pd[2].present==419);
 assert(strstr(last,"after storage binding 7")&&strstr(last,"native/view format 43/37"));
 reset_capture();g_pd_state=2;cs_trace_image_note(&bind,&ii,&im,&t,&v,1);assert(!g_cs_images_n);
 reset_capture();v.base_level=32;cs_trace_image_note(&bind,&ii,&im,&t,&v,1);assert(!g_cs_images_n);v.base_level=2;
 v.min_layer=3;cs_trace_image_note(&bind,&ii,&im,&t,&v,1);assert(!g_cs_images_n);v.min_layer=1;
 v.nlayers=2;cs_trace_image_note(&bind,&ii,&im,&t,&v,1);assert(!g_cs_images_n);v.nlayers=1;
 im.samples=4;cs_trace_image_note(&bind,&ii,&im,&t,&v,1);assert(!g_cs_images_n);im.samples=1;
 im.fmt=VK_FORMAT_R32G32B32A32_UINT;cs_trace_image_note(&bind,&ii,&im,&t,&v,1);assert(!g_cs_images_n);im.fmt=VK_FORMAT_R8G8B8A8_SRGB;
 im.w=im.h=UINT32_MAX;v.base_level=0;cs_trace_image_note(&bind,&ii,&im,&t,&v,1);assert(!g_cs_images_n);im.w=64;im.h=32;
 reset_capture();g_cs_trace_copies=15;cs_trace_image_note(&bind,&ii,&im,&t,&v,1);assert(!g_cs_images_n);
 cs_trace_image_note(&bind,&ii,&im,&t,&v,0);assert(g_cs_images_n==1&&g_cs_trace_copies==16);
 reset_capture();g_npd=95;cs_trace_image_note(&bind,&ii,&im,&t,&v,1);assert(!g_cs_images_n);
 reset_capture();g_cs_trace_bytes=256u<<20;cs_trace_image_note(&bind,&ii,&im,&t,&v,0);assert(!g_cs_images_n);
 reset_capture();cs_trace_image_note(&bind,&ii,&im,&t,&v,1);assert(g_cs_images_n==1);
 int old_copies=copy_count;fail_alloc=1;cs_trace_images_copy(VK_NULL_HANDLE,0xfa,0);assert(!g_npd&&copy_count==old_copies&&destroyed==1&&freed==0);
 fail_alloc=0;
 assert(cs_trace_image_bytes(64,32,8,4)==65536);
 assert(cs_trace_image_bytes(65536,512,1,4)==(128u<<20));
 assert(!cs_trace_image_bytes(65536,512,2,4)&&!cs_trace_image_bytes(UINT32_MAX,UINT32_MAX,UINT32_MAX,8));
 assert(!cs_trace_image_bytes(1,1,0,4)&&!cs_trace_image_bytes(0,1,1,4)&&!cs_trace_image_bytes(1,1,1,0));
 im=(vkimg_t){.img=(VkImage)(uintptr_t)9,.fmt=VK_FORMAT_B10G11R11_UFLOAT_PACK32,.usage=VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
  .aspect=VK_IMAGE_ASPECT_COLOR_BIT,.vtype=VK_IMAGE_VIEW_TYPE_3D,.samples=1,.d=8,.levels=5,.layers=1,.w=64,.h=32};
 v=(view_t){.set_levels=1,.base_level=2,.set_layers=1,.min_layer=0,.nlayers=1,.set_target=1,.target=NVN_TEXTURE_TARGET_3D};
 t.d.format=im.fmt;reset_capture();cs_trace_image_note(&bind,&ii,&im,&t,&v,1);
 assert(g_cs_images_n==1&&g_cs_images[0].depth==2&&g_cs_images[0].bytes==16*8*2*4&&g_cs_images[0].raw);
 assert(g_cs_trace_bytes==2*16*8*2*4);
 cs_trace_images_copy(VK_NULL_HANDLE,0x9e6,0);cs_trace_images_copy(VK_NULL_HANDLE,0x9e6,1);
 assert(g_npd==2&&last_copy.imageExtent.depth==2&&last_copy.imageSubresource.baseArrayLayer==0&&last_copy.imageSubresource.layerCount==1);
 assert(g_pd[0].raw_bytes==1024&&g_pd[0].depth==2&&g_pd[0].key==0x9e6&&g_pd[0].binding==7);
 assert(g_pd[0].frame==501&&g_pd[0].fence==702&&!g_pd[0].after&&g_pd[1].after&&host_barriers==copy_count);
 reset_capture();im.fmt=VK_FORMAT_R8G8_UNORM;t.d.format=im.fmt;cs_trace_image_note(&bind,&ii,&im,&t,&v,0);
 assert(g_cs_images_n==1&&g_cs_images[0].bytes==512&&g_cs_images[0].bpp==2);
 reset_capture();v.base_level=4;cs_trace_image_note(&bind,&ii,&im,&t,&v,0);assert(g_cs_images[0].depth==1&&g_cs_images[0].raw);
 reset_capture();v.target=2;cs_trace_image_note(&bind,&ii,&im,&t,&v,0);assert(!g_cs_images_n);v.target=NVN_TEXTURE_TARGET_3D;
 v.min_layer=1;cs_trace_image_note(&bind,&ii,&im,&t,&v,0);assert(!g_cs_images_n);v.min_layer=0;
 v.nlayers=2;cs_trace_image_note(&bind,&ii,&im,&t,&v,0);assert(!g_cs_images_n);v.nlayers=1;
 im.d=0;cs_trace_image_note(&bind,&ii,&im,&t,&v,0);assert(!g_cs_images_n);
 im.d=UINT32_MAX;im.w=im.h=UINT32_MAX;v.base_level=0;cs_trace_image_note(&bind,&ii,&im,&t,&v,1);assert(!g_cs_images_n);
 // Legacy RGBA16F 2D still enters the existing preview + RAW=2 path, with all bytes retained.
 im=(vkimg_t){.img=(VkImage)(uintptr_t)9,.fmt=VK_FORMAT_R16G16B16A16_SFLOAT,.usage=VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
  .aspect=VK_IMAGE_ASPECT_COLOR_BIT,.vtype=VK_IMAGE_VIEW_TYPE_2D,.samples=1,.d=1,.levels=1,.layers=1,.w=256,.h=256};
 v=(view_t){0};bind.flags=0;t.d.format=im.fmt;cs_trace_image_note(&bind,&ii,&im,&t,NULL,0);assert(g_cs_images_n==1&&!g_cs_images[0].raw);
 cs_trace_images_copy(VK_NULL_HANDLE,0x21d9,0);assert(!g_pd[0].raw_bytes&&g_pd[0].depth==1);

 return 0;
}
"""
        code=code.replace("DECL_PD",pd).replace("DECL_CI",ci).replace("FUNCTIONS",functions)
        dest=ROOT/"target/tests";dest.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="compute-trace-",dir=dest) as d:
            c=Path(d)/"check.c";exe=Path(d)/"check";c.write_text(code)
            subprocess.run(["cc","-std=c11","-Wall","-Wextra","-Werror","-fsanitize=address,undefined",
                "-I"+str(ROOT/"nxnative/dm/third_party/vulkan-headers/include"),str(c),"-o",str(exe)],check=True)
            subprocess.run([str(exe)],check=True,timeout=15)

    @unittest.skipUnless(shutil.which("cc"),"host compiler required")
    def test_native_volume_writer_preserves_bytes_and_metadata(self):
        src=(ROOT/"nxnative/dm/android/gfx_vk_exec.c").read_text()
        pd=next(x for x in src.splitlines() if x.startswith("typedef struct ") and x.endswith(" pdump_t;"))
        writer=extract(src,"static void pass_dump_raw_native(")
        code=r"""
#include <vulkan/vulkan_core.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <stdarg.h>
static void logger(const char *f,...){(void)f;}
#define VKLOG(...) logger(__VA_ARGS__)
DECL_PD
WRITER
int main(int argc,char **argv){
 assert(argc==2);uint8_t bytes[48];for(unsigned i=0;i<sizeof bytes;i++)bytes[i]=(uint8_t)(i*7);
 pdump_t d={.p=bytes,.w=3,.h=2,.depth=2,.f=VK_FORMAT_B10G11R11_UFLOAT_PACK32,
 .view_format=VK_FORMAT_B10G11R11_UFLOAT_PACK32,.raw_bytes=sizeof bytes,.key=0x9e6,
 .dispatch=3,.binding=4,.level=2,.layer=0,.storage=1,.after=1,.frame=501,.present=419,.fence=702};
 pass_dump_raw_native(&d,7,argv[1]);
 return 0;
}
"""
        code=code.replace("DECL_PD",pd).replace("WRITER",writer)
        parent=ROOT/"target/tests";parent.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="volume-writer-",dir=parent) as folder:
            d=Path(folder);c=d/"check.c";exe=d/"check";c.write_text(code)
            subprocess.run(["cc","-std=c11","-Wall","-Wextra","-Werror","-fsanitize=address,undefined",
                "-I"+str(ROOT/"nxnative/dm/third_party/vulkan-headers/include"),str(c),"-o",str(exe)],check=True)
            subprocess.run([str(exe),str(d)],check=True,timeout=15)
            self.assertEqual((d/"pass_07.raw").read_bytes(),bytes((i*7)&255 for i in range(48)))
            m=json.loads((d/"pass_07.raw.json").read_text())
            self.assertEqual((m["width"],m["height"],m["depth"]),(3,2,2))
            self.assertEqual((m["bytes"],m["written"],m["native_format"],m["view_format"]),(48,48,122,122))
            self.assertEqual((m["key"],m["binding"],m["phase"],m["mip"],m["layer"]),("00000000000009e6",4,"after",2,0))
            self.assertEqual((m["frame"],m["present"],m["fence"]),(501,419,702))
        poll=extract(src,"void vk_pass_dump_poll(")
        self.assertLess(poll.index("done_fence <"),poll.index("vkInvalidateMappedMemoryRanges"))
        self.assertLess(poll.index("vkInvalidateMappedMemoryRanges"),poll.index("pass_dump_raw_native("))
        self.assertLess(poll.index("pass_dump_raw_native("),poll.index('fopen(path, "wb")'))
        self.assertIn("memset(g_pd, 0, sizeof g_pd)",poll)

    def test_dispatch_and_completion_order(self):
        src=(ROOT/"nxnative/dm/android/gfx_vk_exec.c").read_text()
        dispatch=extract(src,"static void dispatch(")
        self.assertLess(dispatch.index("cs_trace_match("),dispatch.index("skip_kernel("))
        self.assertLess(dispatch.index("cs_trace_images_copy(cb, cs->key, 0)"),dispatch.index("vkCmdDispatch(cb"))
        self.assertGreater(dispatch.index("cs_trace_images_copy(cb, cs->key, 1)"),dispatch.index("vkCmdDispatch(cb"))
        poll=extract(src,"void vk_pass_dump_poll(")
        self.assertLess(poll.index("done_fence <"),poll.index("vkInvalidateMappedMemoryRanges"))
        self.assertLess(poll.index("vkInvalidateMappedMemoryRanges"),poll.index('fopen(path, "wb")'))
        self.assertIn("cs_trace_poll();",extract(src,"static void x_read("))
if __name__=="__main__":unittest.main()
