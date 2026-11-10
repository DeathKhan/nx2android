"""Execute the actual signed-square recording helpers; validate the shader and its transfer expression."""
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]

class SquareUpscaler(unittest.TestCase):
    @unittest.skipUnless(shutil.which("cc"), "host compiler required")
    def test_commands_contract_lifetime(self):
        source = (ROOT / "nxnative/dm/android/upscale_square_vk.c").read_text()
        source = source[source.index("typedef struct { uint32_t w, h; }"):]
        code = PREAMBLE + source + MAIN
        target = ROOT / "target/tests"; target.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="square-upscale-", dir=target) as folder:
            c, exe = Path(folder) / "check.c", Path(folder) / "check"
            c.write_text(code)
            subprocess.run(["cc", "-std=gnu11", "-Wall", "-Wextra", "-Werror",
                "-Wno-missing-field-initializers", "-Wno-unused-variable", "-Wno-unused-function",
                "-fsanitize=address,undefined", "-I" + str(ROOT / "nxnative/dm/third_party/vulkan-headers/include"),
                str(c), "-o", str(exe)], check=True)
            subprocess.run([str(exe)], check=True, timeout=15)
    @unittest.skipUnless(shutil.which("glslc") and shutil.which("spirv-val") and shutil.which("cc"), "shader/compiler tools required")
    def test_actual_shader_transfer(self):
        shader = ROOT / "nxnative/dm/android/shaders/upscale_square.comp"
        s = shader.read_text()
        self.assertIn("textureLod(src, uv, 0.0)", s)
        self.assertNotIn("clamp(", s)
        expression = re.search(r"vec4\((encoded \* abs\(encoded\)), 1.0\)", s).group(1).replace("abs(", "fabsf(")
        code = "#include <math.h>\n#include <assert.h>\nstatic float decode(float encoded){return " + expression + ";}\n"
        code += "int main(void){assert(decode(4)==16&&decode(-2)==-4&&decode(0)==0); float filtered=(0.f+1.f)*.5f;assert(decode(filtered)==.25f&&decode(filtered)!=(decode(0)+decode(1))*.5f);return 0;}\n"
        target=ROOT/"target/tests";target.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="square-shader-",dir=target) as folder:
            p=Path(folder);(p/"math.c").write_text(code)
            subprocess.run(["glslc","--target-env=vulkan1.1","-O",str(shader),"-o",str(p/"shader.spv")],check=True)
            subprocess.run(["spirv-val","--target-env","vulkan1.1",str(p/"shader.spv")],check=True)
            subprocess.run(["cc","-std=c11","-fsanitize=address,undefined",str(p/"math.c"),"-lm","-o",str(p/"math")],check=True)
            subprocess.run([str(p/"math")],check=True,timeout=10)

PREAMBLE = r"""
#include <vulkan/vulkan_core.h>
#include <stdint.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <stdio.h>
#define H(t,n) ((t)(uintptr_t)(n))
typedef struct {VkImage img;VkFormat fmt;VkImageAspectFlags aspect;VkImageUsageFlags usage;VkImageViewType vtype;
 uint32_t w,h,d,levels,layers,samples;int gpu_w;} vkimg_t;
static struct {VkDevice dev;VkPhysicalDevice pd;VkPipelineCache pcache;struct {VkPhysicalDeviceLimits limits;} props;} V;
static const uint32_t upscale_square_comp_spv[]={0x07230203};
static uint32_t pic_w=1920,pic_h=1080;static int fail_image;
static int phase, fail_set; static VkImageView source_view; static unsigned next_image=1,next_set=1,sets,dispatches,blits,registered,deferred_images,deferred_views;
static unsigned set_written[256];static VkFormat image_format[1024];static VkImageBlit last_blit;
static VkImage blit_dst;static VkDescriptorSet last_bound;
static void logger(const char *s,...){(void)s;}
#define VKLOG(...) logger(__VA_ARGS__)
#define VKRED_ONCE(n,...) logger(__VA_ARGS__)
static uint32_t vk_mem_type(uint32_t types,VkMemoryPropertyFlags req,VkMemoryPropertyFlags avoid){(void)types;(void)req;(void)avoid;return 0;}
static void vk_end_pass_land(void){}
// gfx_vk_pipeline_cache.inc (upstream nx2android #213): every vkCreateComputePipelines marks the on-disk pipeline cache dirty
static unsigned pcache_changes;static void vk_pipeline_cache_changed(void){pcache_changes++;}
static VkCommandBuffer vk_cmd(void){return H(VkCommandBuffer,1);}
static void vk_tex_prepare(VkCommandBuffer cb,vkimg_t *out){(void)cb;assert(out);}
static void vk_img_barrier_all(VkCommandBuffer cb){(void)cb;phase=0;}
static void vk_defer_free_view(VkImageView v){assert(v);deferred_views++;}
static void vk_defer_free_image(VkImage i,VkDeviceMemory m){assert(i&&m);deferred_images++;}
static void vk_dsl_register(VkDescriptorSetLayout l,const VkDescriptorSetLayoutBinding *b,uint32_t n){
 assert(l&&n==2&&b[0].descriptorType==VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER&&b[1].descriptorType==VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);registered++;
}
static VkDescriptorSet vk_alloc_set(VkDescriptorSetLayout l){assert(l&&registered);if(fail_set)return VK_NULL_HANDLE;sets++;return H(VkDescriptorSet,next_set++);}
static VkResult create_module(VkDevice d,const VkShaderModuleCreateInfo *i,const VkAllocationCallbacks *a,VkShaderModule *out){(void)d;(void)i;(void)a;*out=H(VkShaderModule,1);return VK_SUCCESS;}
static VkResult create_dsl(VkDevice d,const VkDescriptorSetLayoutCreateInfo *i,const VkAllocationCallbacks *a,VkDescriptorSetLayout *out){(void)d;(void)a;assert(i->bindingCount==2);*out=H(VkDescriptorSetLayout,1);return VK_SUCCESS;}
static VkResult create_pl(VkDevice d,const VkPipelineLayoutCreateInfo *i,const VkAllocationCallbacks *a,VkPipelineLayout *out){(void)d;(void)i;(void)a;*out=H(VkPipelineLayout,1);return VK_SUCCESS;}
static VkResult create_pipe(VkDevice d,VkPipelineCache c,uint32_t n,const VkComputePipelineCreateInfo *i,const VkAllocationCallbacks *a,VkPipeline *out){(void)d;(void)c;(void)n;(void)i;(void)a;*out=H(VkPipeline,1);return VK_SUCCESS;}
static VkResult create_image(VkDevice d,const VkImageCreateInfo *i,const VkAllocationCallbacks *a,VkImage *out){
 (void)d;(void)a;if(fail_image)return VK_ERROR_OUT_OF_DEVICE_MEMORY;
 assert(i->format==VK_FORMAT_R16G16B16A16_SFLOAT&&i->mipLevels==1&&i->arrayLayers==1);
 assert(i->usage&VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
 *out=H(VkImage,next_image++);image_format[(uintptr_t)*out]=i->format;return VK_SUCCESS;
}
static void image_req(VkDevice d,VkImage i,VkMemoryRequirements *r){(void)d;(void)i;*r=(VkMemoryRequirements){.size=4096,.alignment=256,.memoryTypeBits=1};}
static VkResult allocate_mem(VkDevice d,const VkMemoryAllocateInfo *i,const VkAllocationCallbacks *a,VkDeviceMemory *out){(void)d;(void)i;(void)a;*out=H(VkDeviceMemory,1);return VK_SUCCESS;}
static VkResult bind_mem(VkDevice d,VkImage i,VkDeviceMemory m,VkDeviceSize o){(void)d;(void)i;(void)m;(void)o;return VK_SUCCESS;}
static VkResult create_view(VkDevice d,const VkImageViewCreateInfo *i,const VkAllocationCallbacks *a,VkImageView *out){(void)d;(void)a;assert(i->format==VK_FORMAT_R16G16B16A16_SFLOAT);*out=H(VkImageView,(uintptr_t)i->image+100);return VK_SUCCESS;}
static void destroy_view(VkDevice d,VkImageView v,const VkAllocationCallbacks *a){(void)d;(void)v;(void)a;}
static void destroy_image(VkDevice d,VkImage i,const VkAllocationCallbacks *a){(void)d;(void)i;(void)a;}
static void free_mem(VkDevice d,VkDeviceMemory m,const VkAllocationCallbacks *a){(void)d;(void)m;(void)a;}
static void mock_barrier(VkCommandBuffer cb,VkPipelineStageFlags src,VkPipelineStageFlags dst,VkDependencyFlags f,uint32_t nm,const VkMemoryBarrier *m,uint32_t nb,const VkBufferMemoryBarrier *b,uint32_t ni,const VkImageMemoryBarrier *i){
 (void)cb;(void)src;(void)dst;(void)f;(void)nm;(void)m;(void)nb;(void)b;(void)ni;(void)i;if(nm){assert(src==VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT&&dst==VK_PIPELINE_STAGE_TRANSFER_BIT&&m->srcAccessMask==VK_ACCESS_SHADER_WRITE_BIT&&m->dstAccessMask==VK_ACCESS_TRANSFER_READ_BIT&&phase==1);phase=2;}
}
static void format_props(VkPhysicalDevice d,VkFormat f,VkFormatProperties *p){(void)d;(void)f;*p=(VkFormatProperties){.optimalTilingFeatures=VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT|VK_FORMAT_FEATURE_BLIT_SRC_BIT|VK_FORMAT_FEATURE_BLIT_DST_BIT|VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT|VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT};}
static void update_sets(VkDevice dev,uint32_t n,const VkWriteDescriptorSet *w,uint32_t nc,const VkCopyDescriptorSet *c){
 (void)dev;(void)nc;(void)c;assert(n==2&&w[0].dstSet==w[1].dstSet);uintptr_t set=(uintptr_t)w[0].dstSet;assert(set<256&&!set_written[set]++);
 assert(w[0].pImageInfo->imageView==source_view&&w[0].pImageInfo->sampler);
 assert(w[1].pImageInfo->imageView&&w[1].pImageInfo->imageLayout==VK_IMAGE_LAYOUT_GENERAL);
}
static void bind_pipe(VkCommandBuffer cb,VkPipelineBindPoint p,VkPipeline v){(void)cb;(void)p;(void)v;}
static void bind_sets(VkCommandBuffer cb,VkPipelineBindPoint p,VkPipelineLayout l,uint32_t first,uint32_t n,const VkDescriptorSet *s,uint32_t nd,const uint32_t *o){
 (void)cb;(void)p;(void)l;(void)first;(void)nd;(void)o;assert(n==1&&set_written[(uintptr_t)*s]);last_bound=*s;
}
static void push(VkCommandBuffer cb,VkPipelineLayout l,VkShaderStageFlags flags,uint32_t off,uint32_t n,const void *data){(void)cb;(void)l;(void)flags;(void)off;assert(n==8&&((const uint32_t*)data)[0]==pic_w&&((const uint32_t*)data)[1]==pic_h);}
static void dispatch(VkCommandBuffer cb,uint32_t x,uint32_t y,uint32_t z){(void)cb;assert(x==(pic_w+15)/16&&y==(pic_h+7)/8&&z==1&&last_bound);assert(phase==0);phase=1;dispatches++;}
static void blit(VkCommandBuffer cb,VkImage src,VkImageLayout sl,VkImage dst,VkImageLayout dl,uint32_t n,const VkImageBlit *r,VkFilter f){
 (void)cb;(void)sl;(void)dl;assert(n==1&&f==VK_FILTER_NEAREST&&image_format[(uintptr_t)src]==VK_FORMAT_R16G16B16A16_SFLOAT);
 assert(image_format[(uintptr_t)dst]!=VK_FORMAT_R8G8B8A8_SRGB); // No double encoding in the movie bridge.
 assert(phase==2&&r->srcOffsets[1].x==(int32_t)pic_w&&r->srcOffsets[1].y==(int32_t)pic_h&&r->dstOffsets[1].x==(int32_t)pic_w&&r->dstOffsets[1].y==(int32_t)pic_h);phase=3;blits++;last_blit=*r;blit_dst=dst;
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
#define vkUpdateDescriptorSets update_sets
#define vkCmdBindPipeline bind_pipe
#define vkCmdBindDescriptorSets bind_sets
#define vkCmdPushConstants push
#define vkCmdDispatch dispatch
#define vkCmdBlitImage blit

static VkResult create_sampler(VkDevice d,const VkSamplerCreateInfo *i,const VkAllocationCallbacks *a,VkSampler *s){
 (void)d;(void)a;assert(i->magFilter==VK_FILTER_LINEAR&&i->minFilter==VK_FILTER_LINEAR&&i->maxLod==0);
 assert(i->addressModeU==VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE&&i->addressModeV==VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE);
 *s=H(VkSampler,1);return VK_SUCCESS;
}
#define vkCreateSampler create_sampler
"""
MAIN = r"""
int main(void) {
 V.dev=H(VkDevice,1);V.props.limits.maxComputeWorkGroupCount[0]=V.props.limits.maxComputeWorkGroupCount[1]=65535;
 vkimg_t in={.img=H(VkImage,500),.fmt=VK_FORMAT_E5B9G9R9_UFLOAT_PACK32,.aspect=VK_IMAGE_ASPECT_COLOR_BIT,
 .usage=VK_IMAGE_USAGE_SAMPLED_BIT,.vtype=VK_IMAGE_VIEW_TYPE_2D,.w=1440,.h=810,.d=1,.levels=1,.layers=1,.samples=1};
 vkimg_t out=in;out.img=H(VkImage,501);out.fmt=VK_FORMAT_B10G11R11_UFLOAT_PACK32;
 out.w=1920;out.h=1080;out.usage=VK_IMAGE_USAGE_TRANSFER_DST_BIT;image_format[501]=out.fmt;
 source_view=H(VkImageView,777);pic_w=out.w;pic_h=out.h;
 VkImageBlit r={.srcSubresource={VK_IMAGE_ASPECT_COLOR_BIT,0,0,1},.srcOffsets={{0,0,0},{1440,810,1}},
 .dstSubresource={VK_IMAGE_ASPECT_COLOR_BIT,0,0,1},.dstOffsets={{0,0,0},{1920,1080,1}}};
 for(int i=0;i<12;i++)assert(vk_upscale_square(&in,source_view,&out,&r));
 assert(dispatches==12&&sets==12&&blits==12&&registered==1&&out.gpu_w&&next_image==2);
 assert(pcache_changes==1); // the one pipeline build marked the pipeline cache for saving
 VkImage old=g_square.image.img;unsigned old_sets=sets,old_dispatch=dispatches,old_blits=blits;
 r.srcOffsets[0].x=1;assert(!vk_upscale_square(&in,source_view,&out,&r));r.srcOffsets[0].x=0;
 r.srcOffsets[1].x=1439;assert(!vk_upscale_square(&in,source_view,&out,&r));r.srcOffsets[1].x=1440;
 r.dstSubresource.layerCount=2;assert(!vk_upscale_square(&in,source_view,&out,&r));r.dstSubresource.layerCount=1;
 r.srcSubresource.mipLevel=32;assert(!vk_upscale_square(&in,source_view,&out,&r));r.srcSubresource.mipLevel=0;
 in.samples=4;assert(!vk_upscale_square(&in,source_view,&out,&r));in.samples=1;
 in.usage=0;assert(!vk_upscale_square(&in,source_view,&out,&r));in.usage=VK_IMAGE_USAGE_SAMPLED_BIT;
 out.fmt=VK_FORMAT_R8G8B8A8_SRGB;assert(!vk_upscale_square(&in,source_view,&out,&r));out.fmt=VK_FORMAT_B10G11R11_UFLOAT_PACK32;
 out.img=in.img;assert(!vk_upscale_square(&in,source_view,&out,&r));out.img=H(VkImage,501);
 assert(!vk_upscale_square(&in,VK_NULL_HANDLE,&out,&r));
 V.props.limits.maxComputeWorkGroupCount[0]=1;assert(!vk_upscale_square(&in,source_view,&out,&r));V.props.limits.maxComputeWorkGroupCount[0]=65535;
 assert(sets==old_sets&&dispatches==old_dispatch&&blits==old_blits);
 // A selected array mip has its own full extent; the conversion intermediate always starts at mip/layer zero.
 in.vtype=out.vtype=VK_IMAGE_VIEW_TYPE_2D_ARRAY;in.levels=out.levels=3;in.layers=out.layers=2;
 in.w*=2;in.h*=2;out.w*=2;out.h*=2;r.srcSubresource.mipLevel=r.dstSubresource.mipLevel=1;
 r.srcSubresource.baseArrayLayer=r.dstSubresource.baseArrayLayer=1;
 assert(vk_upscale_square(&in,source_view,&out,&r));
 assert(last_blit.srcSubresource.mipLevel==0&&last_blit.srcSubresource.baseArrayLayer==0&&last_blit.dstSubresource.mipLevel==1&&last_blit.dstSubresource.baseArrayLayer==1);
 // Failed allocation leaves the prior image and all recorded output untouched.
 out.w=4096;out.h=2304;pic_w=2048;pic_h=1152;r.dstOffsets[1]=(VkOffset3D){2048,1152,1};old_sets=sets;
 fail_image=1;assert(!vk_upscale_square(&in,source_view,&out,&r)&&g_square.image.img==old&&sets==old_sets);fail_image=0;
 assert(vk_upscale_square(&in,source_view,&out,&r)&&g_square.image.img!=old&&deferred_views==1&&deferred_images==1);
 fail_set=1;old_dispatch=dispatches;old_blits=blits;assert(!vk_upscale_square(&in,source_view,&out,&r)&&dispatches==old_dispatch&&blits==old_blits);fail_set=0;
 // Only explicit, well-formed metadata selects the transfer; malformed tails invalidate the entire list.
 unsetenv("DM_VK_UPSCALER_SQUARE_CS");g_square_count=-1;assert(!vk_upscale_square_key(0xfa3033d05c0a41d6ULL));
 setenv("DM_VK_UPSCALER_SQUARE_CS","fa3033d05c0a41d6,0000000000000001",1);g_square_count=-1;
 assert(vk_upscale_square_key(0xfa3033d05c0a41d6ULL)&&vk_upscale_square_key(1)&&!vk_upscale_square_key(2));
 const char *bad[]={"fa3033d05c0a41d6,","fa3033d05c0a41d6,gggggggggggggggg","fa3033","xfa3033d05c0a41d6","fa3033d05c0a41d60"};
 for(unsigned i=0;i<sizeof bad/sizeof *bad;i++){setenv("DM_VK_UPSCALER_SQUARE_CS",bad[i],1);g_square_count=-1;assert(!vk_upscale_square_key(0xfa3033d05c0a41d6ULL));}
 puts("signed-square spatial: full selected views, filter/transfer order, completion-owned descriptors, resize/failure and metadata PASS");return 0;
}
"""

if __name__ == "__main__":
    unittest.main()
