"""Execute the opt-in cubic filter contract and actual upscaler recording path."""
from pathlib import Path
import shutil
import os
import subprocess
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]


def function(source,declaration):
    start=source.index(declaration);pos=source.index('{',start)+1;depth=1
    while depth:
        depth+=(source[pos]=='{')-(source[pos]=='}');pos+=1
    return source[start:pos]


class CubicUpscaler(unittest.TestCase):
    @unittest.skipUnless(shutil.which('cc'),'a host C compiler is required')
    def test_live_filter_contract_and_recording(self):
        dm=ROOT/'nxnative/dm';source=(dm/'android/gfx_vk_exec.c').read_text()
        profile=(dm/'games/040074A01BF12000/android.tsv').read_text().splitlines()
        linear=next(row.split('=',1)[1] for row in profile if row.startswith('env\tDM_VK_UPSCALER_LINEAR_CS='))
        keys=linear.split(',')
        encoded='131d7dcc054f92b9,eb0af4c68c522092,0e351bd8ef324466,446761e211ec30d8,8df4d4ba8fa94292,1c144a902af468de,c8910045dc5b59af,d19ea940b265456f'.split(',')
        all_keys=next(row.split('=',1)[1].split(',') for row in profile if row.startswith('env\tDM_VK_UPSCALER_CS='))
        self.assertEqual(len(set(keys)),16)
        self.assertEqual(set(keys).intersection(encoded),set())
        self.assertEqual(set(keys+encoded),set(all_keys))
        code=r"""
#define _POSIX_C_SOURCE 200809L
#define VK_NO_PROTOTYPES
#include <vulkan/vulkan_core.h>
#include <assert.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <math.h>
typedef struct { VkImage img;VkFormat fmt;VkImageViewType vtype;uint32_t levels,layers,d,samples,w,h;VkImageUsageFlags usage;int gpu_w; } vkimg_t;
typedef struct { int set_levels,nlevels,set_layers,nlayers,set_target,target;uint32_t base_level,min_layer; } view_t;
enum { NVN_TEXTURE_TARGET_2D=1 };
typedef struct { vkimg_t *im;int tid,has_view,allow_promotion,needs_opaque_alpha;view_t view;int tex;uint32_t level,layer,w,h; } upscaler_image;
typedef struct { uint64_t key; } vkshader_t;
typedef struct { int x,y,z;uint64_t ind; } p_dispatch;
static struct { int has_filter_cubic;VkPhysicalDevice pd; } V;
static struct { uint32_t push[2][64];uint64_t tex[6][8],img[6][8]; } S;
static int contract_ok=1,image_ok=1,ends,prepares,barriers,blits,format_queries;
static VkFilter last_filter;static VkImageBlit last_region;
static VkFormatFeatureFlags source_features=VK_FORMAT_FEATURE_BLIT_SRC_BIT|VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT|VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_CUBIC_BIT_EXT;
static VkFormatFeatureFlags output_features=VK_FORMAT_FEATURE_BLIT_DST_BIT;
static upscaler_image input,output;
static int square_enabled,square_ok=1,square_view_ok=1,square_calls;
static int vk_upscale_square_key(uint64_t key){(void)key;return square_enabled;}
static VkImageView vk_tex_view(const int *tex,const view_t *v,int dim,int arrayed,int kind,vkimg_t **im){
 (void)tex;assert(dim==1&&!arrayed&&!kind&&v->set_levels&&v->nlevels==1&&v->set_layers&&v->nlayers==1);
 assert(v->base_level==input.level&&v->min_layer==input.layer&&v->set_target&&v->target==NVN_TEXTURE_TARGET_2D);
 *im=input.im;return square_view_ok?(VkImageView)(uintptr_t)42:VK_NULL_HANDLE;
}
static int vk_upscale_square(vkimg_t *src,VkImageView view,vkimg_t *dst,const VkImageBlit *r){
 assert(src==input.im&&dst==output.im&&view&&(r->srcSubresource.mipLevel==input.level));square_calls++;return square_ok;
}
static int upscaler_contract(const vkshader_t *cs){(void)cs;return contract_ok;}
static uint64_t vk_tex_slot_handle(const uint64_t *t,const uint64_t *i,int slot){(void)t;(void)i;assert(slot==0);return 1;}
static int upscaler_image_at(uint64_t handle,int sampled,upscaler_image *out){(void)handle;*out=sampled?input:output;return image_ok;}
static void vkGetPhysicalDeviceFormatProperties(VkPhysicalDevice pd,VkFormat fmt,VkFormatProperties *out){
 (void)pd;(void)fmt;memset(out,0,sizeof *out);out->optimalTilingFeatures=(format_queries++%2)?output_features:source_features;
}
static void vk_end_pass_land(void){ends++;}
static VkCommandBuffer vk_cmd(void){return (VkCommandBuffer)(uintptr_t)3;}
static void vk_tex_prepare(VkCommandBuffer cb,vkimg_t *im){assert(cb==vk_cmd()&&(im==input.im||im==output.im));prepares++;}
static void vk_img_barrier_all(VkCommandBuffer cb){assert(cb==vk_cmd());barriers++;}
static void vkCmdBlitImage(VkCommandBuffer cb,VkImage src,VkImageLayout sl,VkImage dst,VkImageLayout dl,uint32_t count,const VkImageBlit *region,VkFilter filter){
 assert(cb==vk_cmd()&&src==input.im->img&&dst==output.im->img&&sl==VK_IMAGE_LAYOUT_GENERAL&&dl==VK_IMAGE_LAYOUT_GENERAL&&count==1);
 blits++;last_filter=filter;last_region=*region;
}
static int upscaler_note(uint64_t key,uint8_t bit){(void)key;(void)bit;return 0;}
static void ignore(const char *fmt,...){(void)fmt;}
#define VKLOG(...) ignore(__VA_ARGS__)
CONTRACT
RECT
FILTER
ALPHA
SQUARE_COORD
REPLACE
int main(int argc,char **argv){
 if(argc>1){assert(upscaler_linear_contract(UINT64_C(0x14a8aaf3e0a9edbd))==atoi(argv[1]));return 0;}
 vkimg_t a={.img=(VkImage)(uintptr_t)1,.fmt=VK_FORMAT_B10G11R11_UFLOAT_PACK32,.vtype=VK_IMAGE_VIEW_TYPE_2D,
  .levels=1,.layers=1,.d=1,.samples=1,.w=1152,.h=648,.usage=VK_IMAGE_USAGE_TRANSFER_SRC_BIT};
 vkimg_t b=a;b.img=(VkImage)(uintptr_t)2;b.w=2304;b.h=1296;b.usage=VK_IMAGE_USAGE_TRANSFER_DST_BIT;
 input=(upscaler_image){.im=&a,.tid=1,.w=a.w,.h=a.h};output=(upscaler_image){.im=&b,.tid=2,.w=b.w,.h=b.h};
 uint32_t rect[8]={0,0,1151,647,0,0,2303,1295};
 memcpy(S.push[1],rect,sizeof rect);S.img[5][1]=2;
 vkshader_t cs={UINT64_C(0x14a8aaf3e0a9edbd)};p_dispatch d={48,22,1,0};const char *why;
 V.has_filter_cubic=1;
 assert(upscaler_filter(cs.key,0,source_features,&input,&output,rect,&why)==VK_FILTER_LINEAR);
 assert(upscaler_filter(cs.key,1,source_features,&input,&output,rect,&why)==VK_FILTER_CUBIC_EXT&&!why);
 // Both audited neural-luma clamp families write linear RGB; encoded siblings remain excluded.
 uint64_t linear_keys[]={LINEAR_KEYS},encoded_keys[]={ENCODED_KEYS};
 for(unsigned i=0;i<sizeof linear_keys/sizeof *linear_keys;i++)
  assert(upscaler_filter(linear_keys[i],1,source_features,&input,&output,rect,&why)==VK_FILTER_CUBIC_EXT&&!why);
 for(unsigned i=0;i<sizeof encoded_keys/sizeof *encoded_keys;i++)
  assert(upscaler_filter(encoded_keys[i],1,source_features,&input,&output,rect,&why)==VK_FILTER_LINEAR&&strstr(why,"colour"));
 // The newly observed DRS size uses the same proven encoding contract.
 a.w=input.w=1257;a.h=input.h=707;rect[2]=1256;rect[3]=706;
 assert(upscaler_filter(UINT64_C(0xa9fed0568668bf97),1,source_features,&input,&output,rect,&why)==VK_FILTER_CUBIC_EXT);
 a.w=input.w=1152;a.h=input.h=648;rect[2]=1151;rect[3]=647;
 assert(upscaler_filter(1,1,source_features,&input,&output,rect,&why)==VK_FILTER_LINEAR);
 V.has_filter_cubic=0;assert(upscaler_filter(cs.key,1,source_features,&input,&output,rect,&why)==VK_FILTER_LINEAR);V.has_filter_cubic=1;
 assert(upscaler_filter(cs.key,1,source_features&~VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_CUBIC_BIT_EXT,&input,&output,rect,&why)==VK_FILTER_LINEAR);
 // Cubic sampling must not bleed neighboring rectangle pixels or pick other mips/layers.
 for(int index=0;index<8;index++){
  uint32_t saved=rect[index];rect[index]=saved+1;
  assert(upscaler_filter(cs.key,1,source_features,&input,&output,rect,&why)==VK_FILTER_LINEAR);rect[index]=saved;
 }
 rect[2]=UINT32_MAX;assert(upscaler_filter(cs.key,1,source_features,&input,&output,rect,&why)==VK_FILTER_LINEAR);rect[2]=1151;
 for(int which=0;which<2;which++){
  upscaler_image *u=which?&output:&input;vkimg_t *im=u->im;
  im->levels=2;assert(upscaler_filter(cs.key,1,source_features,&input,&output,rect,&why)==VK_FILTER_LINEAR);im->levels=1;
  im->layers=2;assert(upscaler_filter(cs.key,1,source_features,&input,&output,rect,&why)==VK_FILTER_LINEAR);im->layers=1;
  im->vtype=VK_IMAGE_VIEW_TYPE_2D_ARRAY;assert(upscaler_filter(cs.key,1,source_features,&input,&output,rect,&why)==VK_FILTER_LINEAR);im->vtype=VK_IMAGE_VIEW_TYPE_2D;
  im->fmt=VK_FORMAT_R16G16B16A16_SFLOAT;assert(upscaler_filter(cs.key,1,source_features,&input,&output,rect,&why)==VK_FILTER_LINEAR);im->fmt=VK_FORMAT_B10G11R11_UFLOAT_PACK32;
  im->samples=4;assert(upscaler_filter(cs.key,1,source_features,&input,&output,rect,&why)==VK_FILTER_LINEAR);im->samples=1;
  im->d=2;assert(upscaler_filter(cs.key,1,source_features,&input,&output,rect,&why)==VK_FILTER_LINEAR);im->d=1;
  u->level=1;assert(upscaler_filter(cs.key,1,source_features,&input,&output,rect,&why)==VK_FILTER_LINEAR);u->level=0;
  u->layer=1;assert(upscaler_filter(cs.key,1,source_features,&input,&output,rect,&why)==VK_FILTER_LINEAR);u->layer=0;
  u->has_view=1;u->view=(view_t){.set_levels=1,.nlevels=1,.set_layers=1,.nlayers=1,.set_target=1,.target=NVN_TEXTURE_TARGET_2D};
  assert(upscaler_filter(cs.key,1,source_features,&input,&output,rect,&why)==VK_FILTER_CUBIC_EXT);
  u->view.nlevels=2;assert(upscaler_filter(cs.key,1,source_features,&input,&output,rect,&why)==VK_FILTER_LINEAR);u->view.nlevels=1;
  u->view.nlayers=0;assert(upscaler_filter(cs.key,1,source_features,&input,&output,rect,&why)==VK_FILTER_LINEAR);u->view.nlayers=1;
  u->view.target=99;assert(upscaler_filter(cs.key,1,source_features,&input,&output,rect,&why)==VK_FILTER_LINEAR);
  u->has_view=0;
 }
 // Execute the actual recording function: default, live on, and live off.
 unsetenv("DM_VK_UPSCALER_CUBIC");assert(upscaler_replace(&cs,&d)&&last_filter==VK_FILTER_LINEAR);
 setenv("DM_VK_UPSCALER_CUBIC","1",1);assert(upscaler_replace(&cs,&d)&&last_filter==VK_FILTER_CUBIC_EXT);
 assert(last_region.srcSubresource.mipLevel==0&&last_region.srcSubresource.baseArrayLayer==0&&last_region.srcSubresource.layerCount==1);
 assert(last_region.srcOffsets[1].x==1152&&last_region.srcOffsets[1].y==648&&last_region.dstOffsets[1].x==2304&&last_region.dstOffsets[1].y==1296);
 assert(b.gpu_w&&ends==2&&prepares==4&&barriers==4&&format_queries==2); // no extra capability queries or command/descriptor allocation per frame
 setenv("DM_VK_UPSCALER_CUBIC","0",1);assert(upscaler_replace(&cs,&d)&&last_filter==VK_FILTER_LINEAR);
 setenv("DM_VK_UPSCALER_CUBIC","1",1);cs.key=UINT64_C(0x1c144a902af468de);
 assert(upscaler_replace(&cs,&d)&&last_filter==VK_FILTER_LINEAR);cs.key=UINT64_C(0x14a8aaf3e0a9edbd);
 S.push[1][2]=1100;assert(upscaler_replace(&cs,&d)&&last_filter==VK_FILTER_LINEAR);S.push[1][2]=1151;
 int before=blits;
 d.ind=1;assert(!upscaler_replace(&cs,&d)&&blits==before);d.ind=0;
 d.z=2;assert(!upscaler_replace(&cs,&d)&&blits==before);d.z=1;
 contract_ok=0;assert(!upscaler_replace(&cs,&d)&&blits==before);contract_ok=1;
 image_ok=0;assert(!upscaler_replace(&cs,&d)&&blits==before);image_ok=1;
 b.usage=0;assert(!upscaler_replace(&cs,&d)&&blits==before);b.usage=VK_IMAGE_USAGE_TRANSFER_DST_BIT;
 S.push[1][2]=UINT32_MAX;assert(!upscaler_replace(&cs,&d)&&blits==before);S.push[1][2]=1151;
 // A promoted RGB source may have arbitrary backing alpha: reject RGBA before recording, allow actual RGB destinations.
 input.needs_opaque_alpha=1;a.fmt=VK_FORMAT_R16G16B16A16_SFLOAT;b.fmt=VK_FORMAT_R16G16B16A16_SFLOAT;
 assert(!upscaler_replace(&cs,&d)&&blits==before);
 b.fmt=VK_FORMAT_B10G11R11_UFLOAT_PACK32;assert(upscaler_replace(&cs,&d)&&last_filter==VK_FILTER_LINEAR&&blits==before+1);
 // Audited signed-square keys must use their transfer path, including when cubic is live.
 // Any helper/view failure falls back to native compute, never to the plain blit.
 before=blits;square_enabled=1;S.push[1][20]=S.push[1][21]=0x3f800000u;
 float source_recip[2]={1.f/input.w,1.f/input.h},out_recip[2]={1.f/output.w,1.f/output.h};
 memcpy(&S.push[1][8],source_recip,sizeof source_recip);memcpy(&S.push[1][26],out_recip,sizeof out_recip);
 S.push[1][24]=output.w;S.push[1][25]=output.h;input.has_view=1;input.view.nlevels=3;input.view.nlayers=2;
 assert(upscaler_replace(&cs,&d)&&square_calls==1&&blits==before);
 square_ok=0;assert(!upscaler_replace(&cs,&d)&&square_calls==2&&blits==before);square_ok=1;
 square_view_ok=0;assert(!upscaler_replace(&cs,&d)&&square_calls==2&&blits==before);square_view_ok=1;
 S.push[1][20]=0x40000000u;assert(!upscaler_replace(&cs,&d)&&square_calls==2&&blits==before);S.push[1][20]=0x3f800000u;
 S.push[1][22]=0x3f800000u;assert(!upscaler_replace(&cs,&d)&&square_calls==2&&blits==before);S.push[1][22]=0x80000000u;
 assert(upscaler_replace(&cs,&d)&&square_calls==3&&blits==before);
 S.push[1][24]++;assert(!upscaler_replace(&cs,&d)&&square_calls==3&&blits==before);S.push[1][24]--;
 S.push[1][8]=0x7fc00000u;assert(!upscaler_replace(&cs,&d)&&square_calls==3&&blits==before);
 source_recip[0]*=1.01f;memcpy(&S.push[1][8],source_recip,sizeof source_recip);
 assert(!upscaler_replace(&cs,&d)&&square_calls==3&&blits==before);
 source_recip[0]=1.f/input.w;memcpy(&S.push[1][8],source_recip,sizeof source_recip);
 S.push[1][8]++;assert(upscaler_replace(&cs,&d)&&square_calls==4&&blits==before);
 return 0;
}
"""
        for marker,decl in [('CONTRACT','static int upscaler_linear_contract('),('RECT','static int upscaler_rect('),('FILTER','static VkFilter upscaler_filter('),('ALPHA','static int upscaler_alpha_compatible('),('SQUARE_COORD','static int upscaler_square_coordinates('),('REPLACE','static int upscaler_replace(')]:
            code=code.replace('\n'+marker+'\n','\n'+function(source,decl)+'\n')
        code=code.replace('LINEAR_KEYS',','.join('UINT64_C(0x'+k+')' for k in keys))
        code=code.replace('ENCODED_KEYS',','.join('UINT64_C(0x'+k+')' for k in encoded))
        target=ROOT/'target/tests';target.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='vk-upscaler-cubic-',dir=target) as folder:
            src,exe=Path(folder)/'test.c',Path(folder)/'test';src.write_text(code)
            subprocess.run(['cc','-std=c11','-O1','-g','-Wall','-Wextra','-Werror','-fsanitize=address,undefined',
                '-I'+str(dm/'third_party/vulkan-headers/include'),str(src),'-o',str(exe)],check=True)
            env=dict(os.environ,DM_VK_UPSCALER_LINEAR_CS=linear)
            subprocess.run([str(exe)],env=env,check=True,timeout=15)
            # Each process reparses startup metadata, including malformed suffixes after a valid key.
            good='14a8aaf3e0a9edbd'
            for value,expected in [(None,0),('',0),(good,1),(good.upper(),1),
                (good+',',0),(good+',oops',0),('0x'+good,0),(good[:-1],0),
                (' '+good,0),(good+'0',0),(','.join([good]*64),1),(','.join([good]*65),0)]:
                e=dict(os.environ);e.pop('DM_VK_UPSCALER_LINEAR_CS',None)
                if value is not None:e['DM_VK_UPSCALER_LINEAR_CS']=value
                subprocess.run([str(exe),str(expected)],env=e,check=True,timeout=15)

    def test_extension_enablement_is_capability_gated(self):
        core=(ROOT/'nxnative/dm/android/gfx_vk_core.c').read_text()
        self.assertIn('V.has_filter_cubic = has_ext(ex, ne, "VK_EXT_filter_cubic");',core)
        self.assertIn('if (V.has_filter_cubic) on[non++] = "VK_EXT_filter_cubic";',core)
        self.assertNotIn('on[non++] = "VK_QCOM_filter_cubic_weights"',core)


if __name__=='__main__':unittest.main()
