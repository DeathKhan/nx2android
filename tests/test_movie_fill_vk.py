"""Execute movie conversion/copy and descriptor ownership with a mocked Vulkan command stream."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
import sys; sys.path.insert(0, str(Path(__file__).resolve().parent))
import host_cc   # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


class MovieFill(unittest.TestCase):
    @unittest.skipUnless(shutil.which("cc"), "a host compiler is required")
    @unittest.skipUnless(host_cc.sanitizers(), 'a host C compiler with ASan and UBSan is required (MinGW has neither)')
    def test_actual_movie_commands_and_lifetimes(self):
        source = (ROOT / "nxnative/dm/android/movie_fill_vk.c").read_text(encoding='utf-8')
        source = source[source.index("typedef struct { uint32_t w, h, word0, flags; }"):
                        source.index("// ---- DM_VK_MOVIE_FILL_CS ----")]
        code = r"""
#include <vulkan/vulkan_core.h>
#include <stdint.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <stdio.h>
#define H(t,n) ((t)(uintptr_t)(n))
enum {DM_CODEC_CS_BT709=1,DM_CODEC_CS_FULL=2};
typedef struct {VkImage img;VkFormat fmt;VkImageAspectFlags aspect;VkImageUsageFlags usage;VkImageViewType vtype;
 uint32_t w,h,d,levels,layers,samples;int gpu_w;} vkimg_t;
static struct {VkDevice dev;VkPhysicalDevice pd;VkPipelineCache pcache;struct {VkPhysicalDeviceLimits limits;} props;} V;
static const uint32_t movie_fill_comp_spv[]={0x07230203};
static const char *vk_pool_tag;
static uint32_t pic_w=640,pic_h=360;static uint64_t pic_gen=1;static int picture=1,buffer_ok=1,fail_image;
static unsigned next_image=1,next_set=1,sets,dispatches,blits,copies,registered,deferred_images,deferred_views;
static unsigned buffer_lookups,stages,snapshots,snapshot_ok=1,source_word0=1;
static VkDeviceSize expected_offset,expected_range,stage_size;static uint8_t staged[8u<<20];
static unsigned set_written[256];static VkFormat image_format[1024];static VkImageBlit last_blit;static VkImageCopy last_copy;
static VkImage blit_dst,copy_src,copy_dst;static VkDescriptorSet last_bound;
static void logger(const char *s,...){(void)s;}
#define VKLOG(...) logger(__VA_ARGS__)
#define VKRED_ONCE(n,...) logger(__VA_ARGS__)
static int dm_codec_last_nv12(uint64_t *a,uint32_t *w,uint32_t *h,uint64_t *gen,int *cs){*a=0x1000;*w=pic_w;*h=pic_h;*gen=pic_gen;*cs=DM_CODEC_CS_BT709;return picture;}
static int vk_buf_at(uint64_t a,uint64_t len,VkBuffer *b,VkDeviceSize *off){assert(a==0x1000&&len==(uint64_t)pic_w*pic_h*3/2);buffer_lookups++;*b=H(VkBuffer,100+(pic_gen%3));*off=260;expected_offset=256;expected_range=(len+7)&~UINT64_C(3);source_word0=1;return buffer_ok;}
static void *vk_scratch(VkDeviceSize n,VkDeviceSize align,VkBuffer *b,VkDeviceSize *off){
 assert(align==256&&n<=sizeof staged&&!(n&3));stages++;stage_size=n;memset(staged,0xee,(size_t)n);
 *b=H(VkBuffer,900);*off=512;expected_offset=512;expected_range=n;source_word0=0;return staged;
}
static int dm_codec_copy_last_nv12(uint64_t expected,void *dst,uint64_t cap){
 assert(expected==pic_gen&&dst==staged&&cap==(uint64_t)pic_w*pic_h*3/2);
 if(!snapshot_ok){return 0;} snapshots++;memset(dst,0x5a,(size_t)cap);return 1;
}
static uint32_t vk_mem_type(uint32_t types,VkMemoryPropertyFlags req,VkMemoryPropertyFlags avoid){(void)types;(void)req;(void)avoid;return 0;}
static void vk_end_pass_land(void){}
// gfx_vk_pipeline_cache.inc (upstream nx2android #213): every vkCreateComputePipelines marks the on-disk pipeline cache dirty
static unsigned pcache_changes;static void vk_pipeline_cache_changed(void){pcache_changes++;}
static VkCommandBuffer vk_cmd(void){return H(VkCommandBuffer,1);}
static void vk_tex_prepare(VkCommandBuffer cb,vkimg_t *out){(void)cb;assert(out);}
static void vk_img_barrier_all(VkCommandBuffer cb){(void)cb;}
static void vk_defer_free_view(VkImageView v){assert(v);deferred_views++;}
static void vk_defer_free_image(VkImage i,VkDeviceMemory m){assert(i&&m);deferred_images++;}
static void vk_dsl_register(VkDescriptorSetLayout l,const VkDescriptorSetLayoutBinding *b,uint32_t n){
 assert(l&&n==2&&b[0].descriptorType==VK_DESCRIPTOR_TYPE_STORAGE_BUFFER&&b[1].descriptorType==VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);registered++;
}
static VkDescriptorSet vk_alloc_set(VkDescriptorSetLayout l){assert(l&&registered);sets++;return H(VkDescriptorSet,next_set++);}
static VkResult create_module(VkDevice d,const VkShaderModuleCreateInfo *i,const VkAllocationCallbacks *a,VkShaderModule *out){(void)d;(void)i;(void)a;*out=H(VkShaderModule,1);return VK_SUCCESS;}
static VkResult create_dsl(VkDevice d,const VkDescriptorSetLayoutCreateInfo *i,const VkAllocationCallbacks *a,VkDescriptorSetLayout *out){(void)d;(void)a;assert(i->bindingCount==2);*out=H(VkDescriptorSetLayout,1);return VK_SUCCESS;}
static VkResult create_pl(VkDevice d,const VkPipelineLayoutCreateInfo *i,const VkAllocationCallbacks *a,VkPipelineLayout *out){(void)d;(void)i;(void)a;*out=H(VkPipelineLayout,1);return VK_SUCCESS;}
static VkResult create_pipe(VkDevice d,VkPipelineCache c,uint32_t n,const VkComputePipelineCreateInfo *i,const VkAllocationCallbacks *a,VkPipeline *out){(void)d;(void)c;(void)n;(void)i;(void)a;*out=H(VkPipeline,1);return VK_SUCCESS;}
static VkResult create_image(VkDevice d,const VkImageCreateInfo *i,const VkAllocationCallbacks *a,VkImage *out){
 (void)d;(void)a;if(fail_image)return VK_ERROR_OUT_OF_DEVICE_MEMORY;
 assert(i->format==VK_FORMAT_R8G8B8A8_UNORM&&i->mipLevels==1&&i->arrayLayers==1);
 assert(i->usage&VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
 *out=H(VkImage,next_image++);image_format[(uintptr_t)*out]=i->format;return VK_SUCCESS;
}
static void image_req(VkDevice d,VkImage i,VkMemoryRequirements *r){(void)d;(void)i;*r=(VkMemoryRequirements){.size=4096,.alignment=256,.memoryTypeBits=1};}
static VkResult allocate_mem(VkDevice d,const VkMemoryAllocateInfo *i,const VkAllocationCallbacks *a,VkDeviceMemory *out){(void)d;(void)i;(void)a;*out=H(VkDeviceMemory,1);return VK_SUCCESS;}
static VkResult bind_mem(VkDevice d,VkImage i,VkDeviceMemory m,VkDeviceSize o){(void)d;(void)i;(void)m;(void)o;return VK_SUCCESS;}
static VkResult create_view(VkDevice d,const VkImageViewCreateInfo *i,const VkAllocationCallbacks *a,VkImageView *out){(void)d;(void)a;assert(i->format==VK_FORMAT_R8G8B8A8_UNORM);*out=H(VkImageView,(uintptr_t)i->image+100);return VK_SUCCESS;}
static void destroy_view(VkDevice d,VkImageView v,const VkAllocationCallbacks *a){(void)d;(void)v;(void)a;}
static void destroy_image(VkDevice d,VkImage i,const VkAllocationCallbacks *a){(void)d;(void)i;(void)a;}
static void free_mem(VkDevice d,VkDeviceMemory m,const VkAllocationCallbacks *a){(void)d;(void)m;(void)a;}
static void mock_barrier(VkCommandBuffer cb,VkPipelineStageFlags src,VkPipelineStageFlags dst,VkDependencyFlags f,uint32_t nm,const VkMemoryBarrier *m,uint32_t nb,const VkBufferMemoryBarrier *b,uint32_t ni,const VkImageMemoryBarrier *i){
 (void)cb;(void)src;(void)dst;(void)f;(void)nm;(void)m;(void)nb;(void)b;(void)ni;(void)i;
}
static void format_props(VkPhysicalDevice d,VkFormat f,VkFormatProperties *p){(void)d;(void)f;*p=(VkFormatProperties){.optimalTilingFeatures=VK_FORMAT_FEATURE_BLIT_SRC_BIT|VK_FORMAT_FEATURE_BLIT_DST_BIT|VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT|VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT};}
static void update_sets(VkDevice dev,uint32_t n,const VkWriteDescriptorSet *w,uint32_t nc,const VkCopyDescriptorSet *c){
 (void)dev;(void)nc;(void)c;assert(n==2&&w[0].dstSet==w[1].dstSet);uintptr_t set=(uintptr_t)w[0].dstSet;assert(set<256&&!set_written[set]++);
 assert(w[0].pBufferInfo->offset==expected_offset&&w[0].pBufferInfo->range==expected_range);
 assert(w[1].pImageInfo->imageView&&w[1].pImageInfo->imageLayout==VK_IMAGE_LAYOUT_GENERAL);
}
static void bind_pipe(VkCommandBuffer cb,VkPipelineBindPoint p,VkPipeline v){(void)cb;(void)p;(void)v;}
static void bind_sets(VkCommandBuffer cb,VkPipelineBindPoint p,VkPipelineLayout l,uint32_t first,uint32_t n,const VkDescriptorSet *s,uint32_t nd,const uint32_t *o){
 (void)cb;(void)p;(void)l;(void)first;(void)nd;(void)o;assert(n==1&&set_written[(uintptr_t)*s]);last_bound=*s;
}
static void push(VkCommandBuffer cb,VkPipelineLayout l,VkShaderStageFlags flags,uint32_t off,uint32_t n,const void *data){(void)cb;(void)l;(void)flags;(void)off;assert(n==16&&((const uint32_t*)data)[2]==source_word0);}
static void dispatch(VkCommandBuffer cb,uint32_t x,uint32_t y,uint32_t z){(void)cb;assert(x==(pic_w+15)/16&&y==(pic_h+15)/16&&z==1&&last_bound);dispatches++;}
static void blit(VkCommandBuffer cb,VkImage src,VkImageLayout sl,VkImage dst,VkImageLayout dl,uint32_t n,const VkImageBlit *r,VkFilter f){
 (void)cb;(void)sl;(void)dl;assert(n==1&&f==VK_FILTER_LINEAR&&image_format[(uintptr_t)src]==VK_FORMAT_R8G8B8A8_UNORM);
 assert(image_format[(uintptr_t)dst]!=VK_FORMAT_R8G8B8A8_SRGB); // No double encoding in the movie bridge.
 blits++;last_blit=*r;blit_dst=dst;
}
static void copy(VkCommandBuffer cb,VkImage src,VkImageLayout sl,VkImage dst,VkImageLayout dl,uint32_t n,const VkImageCopy *r){
 (void)cb;(void)sl;(void)dl;assert(n==1&&src==blit_dst&&image_format[(uintptr_t)src]==VK_FORMAT_R8G8B8A8_UNORM&&image_format[(uintptr_t)dst]==VK_FORMAT_R8G8B8A8_SRGB);
 copies++;last_copy=*r;copy_src=src;copy_dst=dst;
}
static VkResult image_format_props(VkPhysicalDevice d,VkFormat f,VkImageType t,VkImageTiling ti,VkImageUsageFlags u,VkImageCreateFlags c,VkImageFormatProperties *p){
 (void)d;(void)f;(void)t;(void)ti;(void)u;(void)c;memset(p,0,sizeof *p);return VK_SUCCESS;
}
#define vkCreateShaderModule create_module
#define vkCreateDescriptorSetLayout create_dsl
#define vkCreatePipelineLayout create_pl
#define vkCreateComputePipelines create_pipe
#define vkCreateImage create_image
#define vkGetImageMemoryRequirements image_req
#define vkAllocateMemory allocate_mem
#define vkBindImageMemory bind_mem
#define vkCreateImageView create_view
#define vkDestroyImageView destroy_view
#define vkDestroyImage destroy_image
#define vkFreeMemory free_mem
#define vkCmdPipelineBarrier mock_barrier
#define vkGetPhysicalDeviceFormatProperties format_props
#define vkGetPhysicalDeviceImageFormatProperties image_format_props
#define vkUpdateDescriptorSets update_sets
#define vkCmdBindPipeline bind_pipe
#define vkCmdBindDescriptorSets bind_sets
#define vkCmdPushConstants push
#define vkCmdDispatch dispatch
#define vkCmdBlitImage blit
#define vkCmdCopyImage copy
SOURCE
int main(void){
 setenv("DM_VK_MOVIE_FILL_COPY","1",1); // the scaled UNORM image and its raw copy, which a device without the sRGB stage image's formats takes (direct_ok)
 V.dev=H(VkDevice,1);V.props.limits.minStorageBufferOffsetAlignment=256;V.props.limits.maxStorageBufferRange=128u<<20;
 vkimg_t out={.img=H(VkImage,500),.fmt=VK_FORMAT_R8G8B8A8_SRGB,.aspect=VK_IMAGE_ASPECT_COLOR_BIT,.usage=VK_IMAGE_USAGE_TRANSFER_DST_BIT,
 .vtype=VK_IMAGE_VIEW_TYPE_2D_ARRAY,.w=2560,.h=1440,.d=1,.levels=3,.layers=2,.samples=1};
 image_format[500]=out.fmt;
 for(pic_gen=1;pic_gen<=12;pic_gen++)assert(vk_movie_fill(&out,VK_FORMAT_R8G8B8A8_UNORM,0,0));
 assert(sets==12&&dispatches==12&&copies==12&&blits==12&&registered==1&&out.gpu_w&&buffer_lookups==12&&!stages);
 assert(pcache_changes==1); // the one pipeline build marked the pipeline cache for saving
 pic_gen--;assert(vk_movie_fill(&out,VK_FORMAT_R8G8B8A8_UNORM,0,0));
 assert(sets==12&&dispatches==12&&copies==13&&blits==13); // Same picture only copies; no stale descriptor rebinding.
 assert(vk_movie_fill(&out,VK_FORMAT_R8G8B8A8_UNORM,1,1));
 assert(last_blit.dstSubresource.mipLevel==0&&last_blit.dstSubresource.baseArrayLayer==0);
 assert(last_blit.dstOffsets[1].x==1280&&last_blit.dstOffsets[1].y==720);
 assert(last_copy.dstSubresource.mipLevel==1&&last_copy.dstSubresource.baseArrayLayer==1);
 assert(last_copy.extent.width==1280&&last_copy.extent.height==720&&last_copy.srcSubresource.mipLevel==0);
 assert(copy_dst==out.img&&copy_src!=out.img&&deferred_images==1);
 unsigned old_sets=sets,old_copy=copies,old_blit=blits;
 assert(!vk_movie_fill(&out,VK_FORMAT_B8G8R8A8_UNORM,0,0));
 assert(!vk_movie_fill(&out,VK_FORMAT_R8G8B8A8_UNORM,3,0));
 assert(!vk_movie_fill(&out,VK_FORMAT_R8G8B8A8_UNORM,0,2));
 out.samples=4;assert(!vk_movie_fill(&out,VK_FORMAT_R8G8B8A8_UNORM,0,0));out.samples=1;
 out.usage=0;assert(!vk_movie_fill(&out,VK_FORMAT_R8G8B8A8_UNORM,0,0));out.usage=VK_IMAGE_USAGE_TRANSFER_DST_BIT;
 assert(sets==old_sets&&copies==old_copy&&blits==old_blit);
 picture=0;assert(!vk_movie_fill(&out,VK_FORMAT_R8G8B8A8_UNORM,0,0));picture=1;
 buffer_ok=0;unsigned old_lookups=buffer_lookups;assert(vk_movie_fill(&out,VK_FORMAT_R8G8B8A8_UNORM,1,1));assert(buffer_lookups==old_lookups&&!stages);buffer_ok=1;
 pic_w=641;assert(!vk_movie_fill(&out,VK_FORMAT_R8G8B8A8_UNORM,0,0));pic_w=640;
 pic_gen++;V.props.limits.maxStorageBufferRange=64;assert(!vk_movie_fill(&out,VK_FORMAT_R8G8B8A8_UNORM,0,0));V.props.limits.maxStorageBufferRange=128u<<20;
 VkImage old_stage=g.stage.img;pic_gen++;pic_w=1280;pic_h=720;fail_image=1;
 assert(!vk_movie_fill(&out,VK_FORMAT_R8G8B8A8_UNORM,1,1)&&g.stage.img==old_stage&&sets==old_sets);
 fail_image=0;assert(vk_movie_fill(&out,VK_FORMAT_R8G8B8A8_UNORM,1,1));
 assert(sets==old_sets+1&&g.stage.img!=old_stage&&deferred_views==1&&deferred_images==2);
 out.fmt=VK_FORMAT_A2B10G10R10_UNORM_PACK32;image_format[500]=out.fmt;old_copy=copies;
 assert(vk_movie_fill(&out,out.fmt,1,1));
 assert(copies==old_copy&&blit_dst==out.img&&last_blit.dstSubresource.mipLevel==1&&last_blit.dstSubresource.baseArrayLayer==1);
 // CPU-only decoder pools: snapshot once per new generation, then reuse converted image without touching the source.
 buffer_ok=0;pic_gen++;pic_w=pic_h=2;snapshot_ok=0;old_sets=sets;old_blit=blits;
 assert(!vk_movie_fill(&out,out.fmt,1,1)&&sets==old_sets&&blits==old_blit&&stages==1&&!snapshots&&g.gen!=pic_gen);
 snapshot_ok=1;assert(vk_movie_fill(&out,out.fmt,1,1)&&sets==old_sets+1&&snapshots==1&&stages==2&&g.gen==pic_gen);
 assert(stage_size==8&&staged[5]==0x5a&&staged[6]==0&&staged[7]==0);
 old_lookups=buffer_lookups;assert(vk_movie_fill(&out,out.fmt,1,1));assert(buffer_lookups==old_lookups&&stages==2&&snapshots==1);
 pic_gen++;assert(vk_movie_fill(&out,out.fmt,1,1)&&stages==3&&snapshots==2);
 puts("movie bridge: completion-owned descriptors, compatible raw copy, selected subresources, synchronized CPU staging and failures PASS");
 return 0;
}
""".replace("SOURCE", source)
        target = ROOT / "target/tests"
        target.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="movie-fill-", dir=target) as folder:
            c, exe = Path(folder) / "check.c", Path(folder) / "check"
            c.write_text(code, encoding='utf-8')
            subprocess.run(["cc", "-std=gnu11", "-Wall", "-Wextra", "-Werror",
                            "-Wno-missing-field-initializers", "-fsanitize=address,undefined",
                            "-I" + str(ROOT / "nxnative/dm/third_party/vulkan-headers/include"),
                            str(c), "-o", str(exe)], check=True)
            subprocess.run([str(exe)], check=True, timeout=15)


if __name__ == "__main__":
    unittest.main()
