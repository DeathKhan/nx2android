"""Actual-source isolated MRT selection, native attachment, immutable support and fence lifetime."""
from pathlib import Path
import shutil,subprocess,tempfile,unittest
from test_vk_compute_trace import extract
import sys; sys.path.insert(0, str(Path(__file__).resolve().parent))
import host_cc   # noqa: E402
ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'nxnative/dm/android/gfx_vk_exec.c'
HEADER=r"""
#define VK_NO_PROTOTYPES
#include <vulkan/vulkan_core.h>
#include <stdint.h>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define VKLOG(...) do {if(0)fprintf(stderr,__VA_ARGS__);}while(0)
enum {MRT_MODES=12,MRT_DUMPS=36,LIGHT_DIAG_OFFSET=1376,VK_SUPPORT_BYTES=1380};
static uint64_t g_mrt_key,g_mrt_first=UINT64_MAX,g_mrt_last=UINT64_MAX;
static uint32_t g_mrt_mode;static int g_mrt_active,g_mrt_clear,g_mrt_failed;
static unsigned g_mrt_ndump,g_mrt_done,g_mrt_fixed_mode;
static int g_mrt_observer;static uint32_t g_mrt_width=128,g_mrt_height=128;
static VkFormat g_mrt_format=VK_FORMAT_R32G32B32A32_SFLOAT;
typedef struct {VkImage img;VkDeviceMemory mem;VkFormat fmt;uint32_t w,h,layers,samples;} vkimg_t;
static vkimg_t g_mrt_image;static VkImageView g_mrt_view;
static struct {VkDevice dev;struct {struct {size_t minUniformBufferOffsetAlignment;} limits;} props;} V;
typedef struct {VkBuffer b;VkDeviceMemory m;void *p;uint32_t w,h;VkFormat f;uint64_t fence;int pass;uint64_t present,raw_bytes,key,frame;uint32_t depth,binding,dispatch,level,layer;VkFormat view_format;int storage,after;} pdump_t;
static pdump_t g_mrt_dumps[MRT_DUMPS];
"""
@unittest.skipUnless(shutil.which('cc'),'host compiler required')
class MRT(unittest.TestCase):
 def run_c(self,body):
  target=ROOT/'target/tests';target.mkdir(parents=True,exist_ok=True)
  with tempfile.TemporaryDirectory(prefix='mrt-',dir=target) as tmp:
   c,e=Path(tmp)/'t.c',Path(tmp)/'t';c.write_text(body, encoding='utf-8')
   subprocess.run(['cc','-std=gnu11','-O1','-g','-Wall','-Wextra','-Werror','-Wno-unused-variable','-Wno-missing-field-initializers','-fsanitize=address,undefined','-I'+str(ROOT/'nxnative/dm/third_party/vulkan-headers/include'),str(c),'-o',str(e)],check=True)
   subprocess.run([str(e)],check=True,timeout=10)
 @unittest.skipUnless(host_cc.sanitizers(), 'a host C compiler with ASan and UBSan is required (MinGW has neither)')
 def test_selection_attachment_support(self):
  src=SRC.read_text(encoding='utf-8')
  actual='\n'.join(extract(src,'static '+sig+('(void) {' if sig=='void mrt_attach' else '(')) for sig in ['uint32_t mrt_active_slots','int mrt_match','int mrt_choose','void mrt_attach','void mrt_detach','void mrt_support'])
  self.run_c(HEADER+r"""
typedef struct {uint64_t key;} vkshader_t;
static struct {vkimg_t *im;uint32_t fmt,layers,w,h;} P_att[9];
static VkRenderingAttachmentInfo P_ca[8],P_da;
static VkRenderingInfo P_ri;
static struct {int ncol;VkFormat cfmt[8];} S;
static uint8_t scratch[4][VK_SUPPORT_BYTES];static unsigned used;
static uint8_t *vk_scratch(size_t n,size_t align,VkBuffer *b,VkDeviceSize *o){
 assert(n==VK_SUPPORT_BYTES&&align==256&&used<4);*b=(VkBuffer)(uintptr_t)(used+1);*o=used*256;return scratch[used++];
}
static void vk_support_values(void *p){memset(p,0x5a,VK_SUPPORT_BYTES);}
"""+actual+r"""
static int match(uint32_t count){return mrt_match(9,count,1,4,0,1,1,128,128,VK_FORMAT_B10G11R11_UFLOAT_PACK32,1,6,0,0);}
int main(void){
 uint8_t slots[8]={1,1,0,0,0,0,0,1};
 assert(mrt_active_slots(1,slots)==1); // stale slot1/7 from an older multi-target pass are inactive
 assert((mrt_active_slots(2,slots)&~1u)==2); // active bound higher attachment still rejects
 assert(mrt_active_slots(8,slots)==0x83&&mrt_active_slots(99,slots)==0x83);
 assert(!mrt_active_slots(0,slots)&&!mrt_active_slots(-1,slots));
 assert(!match(396));g_mrt_key=9;assert(match(396)&&!match(75)&&!match(849));
 assert(!mrt_match(10,396,1,4,0,1,1,128,128,122,1,6,0,0));
 assert(!mrt_match(9,396,2,4,0,1,1,128,128,122,1,6,0,0));
 assert(!mrt_match(9,396,1,4,1,1,1,128,128,122,1,6,0,0));
 assert(!mrt_match(9,396,1,4,0,0,1,128,128,122,1,6,0,0));
 assert(!mrt_match(9,396,1,4,0,1,2,128,128,122,1,6,0,0));
 assert(!mrt_match(9,396,1,4,0,1,1,127,128,122,1,6,0,0));
 assert(!mrt_match(9,396,1,4,0,1,1,128,128,109,1,6,0,0));
 assert(!mrt_match(9,396,1,4,0,1,1,128,128,122,4,6,0,0));
 assert(!mrt_match(9,396,1,4,0,1,1,128,128,122,1,1,0,0));
 assert(!mrt_match(9,396,1,4,0,1,1,128,128,122,1,6,1,0));
 assert(!mrt_match(9,396,1,4,0,1,1,128,128,122,1,6,0,1));
 assert(!mrt_choose(700)&&!mrt_choose(700)&&!g_mrt_mode); // first entire matching presentation skipped
 for(unsigned mode=1;mode<=12;mode++){
  assert(mrt_choose(700+mode)&&g_mrt_mode==mode);assert(!mrt_choose(700+mode));g_mrt_done++;
 }
 assert(!match(396)&&!mrt_choose(900));g_mrt_done=0;
 g_mrt_fixed_mode=7;g_mrt_first=g_mrt_last=UINT64_MAX;assert(!mrt_choose(1000));
 assert(mrt_choose(1001)&&g_mrt_mode==7);g_mrt_done=1;assert(!match(396)&&!mrt_choose(1002));
 g_mrt_fixed_mode=g_mrt_done=0;
 memset(&P_ca[0],0x12,sizeof P_ca[0]);memset(&P_da,0x34,sizeof P_da);
 VkRenderingAttachmentInfo color=P_ca[0],depth=P_da;
 S.ncol=1;S.cfmt[0]=122;g_mrt_view=(VkImageView)(uintptr_t)55;g_mrt_active=g_mrt_clear=1;
 mrt_attach();assert(S.ncol==2&&S.cfmt[0]==122&&S.cfmt[1]==VK_FORMAT_R32G32B32A32_SFLOAT);
 assert(P_ca[1].loadOp==VK_ATTACHMENT_LOAD_OP_CLEAR&&P_ca[1].storeOp==VK_ATTACHMENT_STORE_OP_STORE);
 assert(P_att[1].im==&g_mrt_image&&P_ri.colorAttachmentCount==2);
 for(int i=0;i<4;i++)assert(P_ca[1].clearValue.color.uint32[i]==0x7fc00000);
 mrt_attach();assert(P_ca[1].loadOp==VK_ATTACHMENT_LOAD_OP_LOAD); // later diagnostic pass restart preserves it
 assert(!memcmp(&color,&P_ca[0],sizeof color)&&!memcmp(&depth,&P_da,sizeof depth));
 V.props.limits.minUniformBufferOffsetAlignment=256;vkshader_t shader={9};VkDescriptorBufferInfo bi={(VkBuffer)(uintptr_t)99,0,1380};
 mrt_support(0,&shader,&bi);assert(!used&&(uintptr_t)bi.buffer==99);
 shader.key=8;mrt_support(1,&shader,&bi);assert(!used);
 shader.key=9;g_mrt_mode=1;mrt_support(1,&shader,&bi);assert(used==1);
 uint32_t mode;memcpy(&mode,scratch[0]+1376,4);assert(mode==1);
 g_mrt_mode=12;mrt_support(1,&shader,&bi);memcpy(&mode,scratch[1]+1376,4);assert(mode==12);
 memcpy(&mode,scratch[0]+1376,4);assert(mode==1);for(int i=0;i<1376;i++)assert(scratch[0][i]==0x5a);
 mrt_detach();assert(!g_mrt_active&&S.ncol==1&&!S.cfmt[1]&&!P_att[1].im&&P_ri.colorAttachmentCount==1);
 mrt_support(1,&shader,&bi);assert(used==2);
 g_mrt_observer=1;g_mrt_active=g_mrt_clear=1;g_mrt_format=VK_FORMAT_R32G32B32A32_UINT;g_mrt_width=1440;g_mrt_height=810;
 VkDescriptorBufferInfo oldbi=bi;mrt_support(1,&shader,&bi);assert(used==2&&!memcmp(&oldbi,&bi,sizeof bi)); // observer must not modify support
 mrt_attach();assert(S.cfmt[1]==107&&P_att[1].w==1440&&P_att[1].h==810);
 for(int i=0;i<4;i++)assert(P_ca[1].clearValue.color.uint32[i]==0); // marker, not UINT NaN
 assert(!memcmp(&color,&P_ca[0],sizeof color)&&!memcmp(&depth,&P_da,sizeof depth));
 mrt_detach();assert(!g_mrt_active&&S.ncol==1&&S.cfmt[0]==122);return 0;
}
""")
 @unittest.skipUnless(host_cc.sanitizers(), 'a host C compiler with ASan and UBSan is required (MinGW has neither)')
 def test_fenced_native_copy_and_completion(self):
  src=SRC.read_text(encoding='utf-8')
  attempts='static unsigned g_mrt_attempts;\n'   # the observer's re-arm count, declared beside mrt_poll
  self.assertIn(attempts+'static void mrt_poll(uint64_t done_fence) {',src)
  actual=extract(src,'static int mrt_copy(')+'\n'+attempts+extract(src,'static void mrt_poll(uint64_t done_fence) {')
  self.run_c(HEADER+r"""
static uint64_t g_tex_frame=90,latest=10;static int g_sub_pass=4,g_draw_i=91,active;
static unsigned g_mrt_skip,g_mrt_qualifying; // the observer's gate state, which a re-arm resets
static int copies,barriers,hostbarriers,written,destroyed,freed,images,views,fail_alloc;
static uint64_t vk_frame_current(void){return 90;}static uint64_t mtl_last_fence(void){return latest;}
static VkCommandBuffer vk_cmd(void){assert(!active);return (VkCommandBuffer)(uintptr_t)1;}
static void vk_img_barrier_all(VkCommandBuffer cb){assert((uintptr_t)cb==1&&!active);barriers++;}
typedef struct {size_t bytes;} Buf;
static VkResult vkCreateBuffer(VkDevice d,const VkBufferCreateInfo *b,const void *a,VkBuffer *out){
 (void)d;(void)a;assert(!active&&b->usage==VK_BUFFER_USAGE_TRANSFER_DST_BIT);Buf *x=malloc(sizeof *x);x->bytes=b->size;*out=(VkBuffer)x;return VK_SUCCESS;
}
static void vkGetBufferMemoryRequirements(VkDevice d,VkBuffer b,VkMemoryRequirements *r){(void)d;r->size=((Buf*)b)->bytes;r->memoryTypeBits=1;}
static uint32_t vk_mem_type(uint32_t bits,uint32_t flags,uint32_t unused){(void)bits;(void)flags;(void)unused;return 0;}
static VkResult vkAllocateMemory(VkDevice d,const VkMemoryAllocateInfo *a,const void *p,VkDeviceMemory *out){
 (void)d;(void)p;if(fail_alloc)return VK_ERROR_OUT_OF_DEVICE_MEMORY;*out=(VkDeviceMemory)calloc(1,a->allocationSize);return VK_SUCCESS;
}
static VkResult vkBindBufferMemory(VkDevice d,VkBuffer b,VkDeviceMemory m,VkDeviceSize o){(void)d;(void)b;(void)m;assert(!o);return VK_SUCCESS;}
static VkResult vkMapMemory(VkDevice d,VkDeviceMemory m,VkDeviceSize o,VkDeviceSize n,VkMemoryMapFlags f,void **p){(void)d;(void)n;(void)f;assert(!o);*p=(void*)m;return VK_SUCCESS;}
static void vkCmdCopyImageToBuffer(VkCommandBuffer cb,VkImage im,VkImageLayout l,VkBuffer b,uint32_t n,const VkBufferImageCopy *r){
 (void)b;assert(!active&&(uintptr_t)cb==1&&im&&l==VK_IMAGE_LAYOUT_GENERAL&&n==1);
 assert(r->imageSubresource.mipLevel==0&&r->imageSubresource.baseArrayLayer==0&&r->imageExtent.width==g_mrt_width&&r->imageExtent.height==g_mrt_height);copies++;
}
static void vkCmdPipelineBarrier(VkCommandBuffer cb,VkPipelineStageFlags a,VkPipelineStageFlags b,VkDependencyFlags d,
 uint32_t mn,const VkMemoryBarrier *m,uint32_t bn,const VkBufferMemoryBarrier *bs,uint32_t in,const VkImageMemoryBarrier *is){
 (void)cb;(void)d;(void)m;(void)is;assert(a==VK_PIPELINE_STAGE_TRANSFER_BIT&&b==VK_PIPELINE_STAGE_HOST_BIT&&!mn&&bn==1&&!in);
 assert(bs->srcAccessMask==VK_ACCESS_TRANSFER_WRITE_BIT&&bs->dstAccessMask==VK_ACCESS_HOST_READ_BIT);hostbarriers++;
}
static void vkDestroyBuffer(VkDevice d,VkBuffer b,const void *a){(void)d;(void)a;destroyed++;free((void*)b);}
static void vkFreeMemory(VkDevice d,VkDeviceMemory m,const void *a){(void)d;(void)a;freed++;free((void*)m);}
static void vkDestroyImage(VkDevice d,VkImage im,const void *a){(void)d;(void)a;assert(im);images++;}
static void vkDestroyImageView(VkDevice d,VkImageView v,const void *a){(void)d;(void)a;assert(v);views++;}
static VkResult vkInvalidateMappedMemoryRanges(VkDevice d,uint32_t n,const VkMappedMemoryRange *r){(void)d;assert(n==1&&r->size==VK_WHOLE_SIZE);return VK_SUCCESS;}
static void pass_dump_raw_native_named(const pdump_t *d,int i,const char *root,const char *prefix){
 (void)root;assert(d->fence<=latest&&i>=0&&!strcmp(prefix,g_mrt_observer?"flat-observer":"mrt"));
 assert(d->raw_bytes==(g_mrt_observer?(d->binding?18662400u:4665600u):(d->binding?262144u:65536u)));written++;
}
"""+actual+r"""
int main(void){
 vkimg_t color={(VkImage)(uintptr_t)1,0,VK_FORMAT_B10G11R11_UFLOAT_PACK32};g_mrt_key=9;g_mrt_mode=1;
 g_mrt_image=(vkimg_t){(VkImage)(uintptr_t)2,(VkDeviceMemory)malloc(1),VK_FORMAT_R32G32B32A32_SFLOAT};g_mrt_view=(VkImageView)(uintptr_t)3;
 assert(mrt_copy(&color,0,0)&&mrt_copy(&color,0,1)&&mrt_copy(&g_mrt_image,1,1));
 assert(g_mrt_ndump==3&&copies==3&&hostbarriers==3&&barriers==6);
 assert(g_mrt_dumps[0].fence==11&&g_mrt_dumps[2].dispatch==1&&g_mrt_dumps[2].binding==1&&g_mrt_dumps[2].after);
 mrt_poll(10);assert(!written&&!destroyed&&!images);latest=11;mrt_poll(11);assert(written==3&&destroyed==3&&freed==3&&!images);
 g_mrt_mode=2;assert(mrt_copy(&color,0,0));g_mrt_done=12;
 mrt_poll(11);assert(written==3&&!images);latest=12;mrt_poll(12);assert(written==4&&images==1&&views==1&&!g_mrt_image.img);
 g_mrt_observer=1;g_mrt_width=1440;g_mrt_height=810;g_mrt_format=VK_FORMAT_R32G32B32A32_UINT;
 g_mrt_done=0;g_mrt_fixed_mode=1;
 g_mrt_image=(vkimg_t){(VkImage)(uintptr_t)2,(VkDeviceMemory)malloc(1),g_mrt_format};g_mrt_view=(VkImageView)(uintptr_t)3;
 assert(mrt_copy(&color,0,0)&&mrt_copy(&color,0,1)&&mrt_copy(&g_mrt_image,1,1));
 assert(g_mrt_dumps[6].f==107&&g_mrt_dumps[6].w==1440&&g_mrt_dumps[6].h==810&&g_mrt_dumps[6].fence==13);
 g_mrt_done=1;mrt_poll(12);assert(written==4&&images==1);g_mrt_qualifying=5;g_mrt_skip=2;latest=13;mrt_poll(13);
 // covered nothing (no marker in the observer's debug target): dropped unwritten, the observer re-arms and keeps its image
 assert(written==4&&destroyed==7&&freed==8&&images==1&&g_mrt_image.img&&g_mrt_attempts==1);
 assert(!g_mrt_ndump&&!g_mrt_done&&!g_mrt_qualifying&&!g_mrt_skip);
 assert(mrt_copy(&color,0,0)&&mrt_copy(&color,0,1)&&mrt_copy(&g_mrt_image,1,1));
 assert(g_mrt_dumps[2].binding==1&&g_mrt_dumps[2].fence==14);
 ((uint32_t*)g_mrt_dumps[2].p)[3]=0x43415252u; // a covered fragment's marker
 g_mrt_done=1;mrt_poll(13);assert(written==4&&images==1);latest=14;mrt_poll(14);
 assert(written==7&&images==2&&views==2&&!g_mrt_image.img&&g_mrt_attempts==1);
 unsigned saved=g_mrt_ndump;g_mrt_ndump=MRT_DUMPS;assert(!mrt_copy(&color,0,0));g_mrt_ndump=saved;
 fail_alloc=1;assert(!mrt_copy(&color,0,0)&&g_mrt_failed&&g_mrt_ndump==saved);
 return 0;
}
""")
 @unittest.skipUnless(host_cc.posix() and host_cc.sanitizers(), 'a POSIX host C compiler with ASan and UBSan is required (MinGW has no setenv, libasan or libubsan)')
 def test_observer_opt_in_exact_gate_skip_and_pair_refusal(self):
  src=SRC.read_text(encoding='utf-8')
  functions=['static int fs_buffer_number(', 'static int pass_dump_fs_parse(', 'static uint32_t mrt_active_slots(', 'static int mrt_match(', 'static int mrt_choose(', 'static int mrt_observer_match(', 'static int mrt_prepare(']
  actual='\n'.join(extract(src,x) for x in functions)
  self.run_c(HEADER+r"""
typedef struct {uint64_t key;} vkshader_t;
typedef struct {int set_levels,set_layers;uint32_t base_level,min_layer;} view_t;
static struct {int rt_n,rt_has_cview[8],rast_discard,ncol,pass_open;uint8_t rt_has_col[8];
 view_t rt_cview[8];vkshader_t *sh[5];uint32_t pass_w,pass_h,samples;
 struct {uint32_t blend_en,logic_op,alpha_func;} color;struct {uint64_t serial;}rt_col[8];} S;
static struct {vkimg_t *im;VkFormat fmt;} P_att[9];static VkRenderingInfo P_ri;
static int g_mrt_config=-1,g_spv_alt,g_tf_on,g_fs_buffer_mesh;
static uint64_t g_spv_key,g_tex_frame,g_set_epoch;static uint32_t g_light_diag_mask;
static unsigned g_mrt_skip,g_mrt_qualifying;static int g_sub_pass,g_draw_i;
static vkimg_t *g_mrt_color;
static int pair_ok=1,pairs,ends,creates,copies,begins;
static int vk_shader_observer_pair(vkshader_t *vs,vkshader_t *fs,void *out){(void)out;assert(vs&&fs);pairs++;return pair_ok;}
static void vk_end_pass(void){ends++;}
static int mrt_image_create(void){creates++;return 1;}
static int mrt_copy(vkimg_t *im,unsigned slot,int after){assert(im&&slot==0&&!after);copies++;return 1;}
static int begin_pass(void){begins++;return 1;}
static void mrt_detach(void){g_mrt_active=0;}
""" + actual + r"""
static vkshader_t vs={0xc8ab0e758c17df4aull},fs={0x96dbc0392691b437ull},mesh={0xedeefae4c37b0257ull};
static vkimg_t im={(VkImage)(uintptr_t)1,0,122,1440,810,1,1};
static void reset(void){
 unsetenv("DM_VK_MRT_FS");unsetenv("DM_VK_MRT_MODE");unsetenv("DM_VK_FLAT_OBSERVER");unsetenv("DM_VK_FLAT_OBSERVER_SKIP");
 memset(&S,0,sizeof S);memset(P_att,0,sizeof P_att);g_mrt_config=-1;g_mrt_key=0;
 g_mrt_observer=g_mrt_active=g_mrt_failed=0;g_mrt_done=g_mrt_fixed_mode=g_mrt_skip=g_mrt_qualifying=0;
 g_spv_alt=g_tf_on=g_light_diag_mask=0;g_fs_buffer_mesh=1;g_mrt_width=g_mrt_height=128;g_mrt_format=109;
 pairs=ends=creates=copies=begins=0;pair_ok=1;P_ri.layerCount=1;
 S.rt_n=S.ncol=1;S.rt_has_col[0]=1;S.rt_has_col[7]=1; // inactive stale attachment is not bound
 S.pass_w=1440;S.pass_h=810;S.samples=1;S.sh[0]=&vs;S.sh[1]=&fs;S.sh[4]=&mesh;
 P_att[0].im=&im;P_att[0].fmt=122;S.rt_col[0].serial=99;
}
static int prepare(void){return mrt_prepare(3,1,4,1,1);}
static int match(uint64_t v,uint64_t f,uint64_t m,int ismesh,int prim,int indexed,uint32_t ncol,uint32_t w,uint32_t h,uint32_t fmt,uint32_t samples,uint32_t layers,uint32_t level,uint32_t layer){
 return mrt_observer_match(v,f,m,ismesh,prim,indexed,ncol,w,h,fmt,samples,layers,level,layer);
}
int main(void){
 assert(match(vs.key,fs.key,mesh.key,1,4,1,1,1440,810,122,1,1,0,0));
 assert(!match(0,fs.key,mesh.key,1,4,1,1,1440,810,122,1,1,0,0));
 assert(!match(vs.key,0,mesh.key,1,4,1,1,1440,810,122,1,1,0,0));
 assert(!match(vs.key,fs.key,0,1,4,1,1,1440,810,122,1,1,0,0));
 assert(!match(vs.key,fs.key,mesh.key,0,4,1,1,1440,810,122,1,1,0,0));
 assert(!match(vs.key,fs.key,mesh.key,1,5,1,1,1440,810,122,1,1,0,0));
 assert(!match(vs.key,fs.key,mesh.key,1,4,0,1,1440,810,122,1,1,0,0));
 assert(!match(vs.key,fs.key,mesh.key,1,4,1,2,1440,810,122,1,1,0,0));
 assert(!match(vs.key,fs.key,mesh.key,1,4,1,1,128,128,122,1,1,0,0));
 assert(!match(vs.key,fs.key,mesh.key,1,4,1,1,1440,810,109,1,1,0,0));
 assert(!match(vs.key,fs.key,mesh.key,1,4,1,1,1440,810,122,4,1,0,0));
 assert(!match(vs.key,fs.key,mesh.key,1,4,1,1,1440,810,122,1,6,0,0));
 assert(!match(vs.key,fs.key,mesh.key,1,4,1,1,1440,810,122,1,1,1,0));
 assert(!match(vs.key,fs.key,mesh.key,1,4,1,1,1440,810,122,1,1,0,1));
 reset();for(int i=0;i<100;i++)assert(prepare());assert(!pairs&&!ends&&!creates&&!copies&&!g_mrt_observer);
 const char *invalid[]={"0","2","01","-1","true"};
 for(unsigned i=0;i<sizeof invalid/sizeof *invalid;i++){reset();setenv("DM_VK_FLAT_OBSERVER",invalid[i],1);assert(prepare()&&!g_mrt_config&&!pairs&&!ends);}
 const char *badskip[]={"33","-1","3x","999"};
 for(unsigned i=0;i<sizeof badskip/sizeof *badskip;i++){reset();setenv("DM_VK_FLAT_OBSERVER","1",1);setenv("DM_VK_FLAT_OBSERVER_SKIP",badskip[i],1);assert(prepare()&&!g_mrt_config&&!ends);}
 reset();setenv("DM_VK_FLAT_OBSERVER","1",1);setenv("DM_VK_MRT_FS","cca3264814bd7187",1);assert(prepare()&&!g_mrt_config&&!ends);
 reset();setenv("DM_VK_FLAT_OBSERVER","1",1);g_spv_alt=1;assert(prepare()&&!g_mrt_config&&!ends);
 reset();setenv("DM_VK_FLAT_OBSERVER","1",1);setenv("DM_VK_FLAT_OBSERVER_SKIP","32",1);
 S.pass_w=128;assert(prepare()&&!g_mrt_qualifying&&!pairs);S.pass_w=1440;
 for(unsigned i=0;i<32;i++)assert(prepare()&&g_mrt_qualifying==i+1&&!pairs&&!ends);
 assert(prepare()&&pairs==1&&ends==1&&copies==1&&creates==1&&begins==1&&g_mrt_active);
 assert(g_mrt_width==1440&&g_mrt_height==810&&g_mrt_format==107&&g_mrt_fixed_mode==1&&g_mrt_mode==1);
 g_mrt_active=0;g_mrt_done=1;assert(prepare()&&pairs==1&&ends==1&&copies==1); // one selected operation only
 reset();setenv("DM_VK_FLAT_OBSERVER","1",1);pair_ok=0;assert(prepare()&&g_mrt_failed&&pairs==1&&!ends&&!creates&&!copies);
 assert(prepare()&&pairs==1&&!ends); // unavailable or mismatched pair never changes native pass
 reset();setenv("DM_VK_FLAT_OBSERVER","1",1);S.rt_n=8;assert(prepare()&&!pairs&&!ends); // active extra guest attachment rejects
 return 0;
}
""")
 def test_actual_hooks_preserve_original_path(self):
  s=SRC.read_text(encoding='utf-8');prepare=extract(s,'static int prepare_draw(');stage=extract(s,'static VkDescriptorSet stage_set(')
  self.assertIn('!(g_mrt_active && st == 1)',stage)
  self.assertIn('light_diag_support(st, s, &bi[nw]); mrt_support(st, s, &bi[nw]);',stage)
  self.assertIn('k.wmask[1] = 0xf; k.blend_en &= ~2u;',prepare)
  self.assertLess(prepare.index('mrt_prepare('),prepare.index('pass_commit()'))
  idx=extract(s,'static void draw_idx(const p_draw_idx *d) {')
  self.assertLess(idx.rindex('vkCmdDrawIndexed('),idx.index('mrt_after_draw('))
  self.assertLess(idx.rindex('xfb_end('),idx.index('mrt_after_draw('))
  self.assertIn('if (g_mrt_active) mrt_attach();',extract(s,'static int begin_pass('))
  prep=extract(s,'static int mrt_prepare(')
  self.assertIn('g_spv_key==g_mrt_key',prep);self.assertIn('g_light_diag_mask',prep)
  self.assertIn('P_ri.layerCount==1',prep);self.assertIn('g_tf_on',prep);self.assertIn('mrt_active_slots(S.rt_n,S.rt_has_col)',prep);self.assertIn('!(active_slots&~1u)',prep)
  self.assertLess(prep.index('vk_end_pass()'),prep.index('mrt_image_create()'))
  self.assertIn('candidate_logs<3&&S.sh[1]&&S.sh[1]->key==g_mrt_key&&count==396',prep)
  self.assertIn('S.pass_w==128&&S.pass_h==128',prep)
  self.assertLess(prep.index('MRT eligibility:'),prep.index('!mrt_choose('))
  mesh=extract(s,'static void mesh_draw_one(const p_draw *d, uint32_t batch) {')
  self.assertLess(mesh.index('g_fs_buffer_mesh = 1'),mesh.index('prepare_draw('))
  self.assertLess(mesh.index('vkCmdDrawIndexedIndirect('),mesh.index('mrt_after_draw('))
  self.assertLess(mesh.index('mrt_after_draw('),mesh.index('if(g_mrt_active)mrt_abort();'))
  self.assertIn('k.alt=2;',prepare)
  self.assertIn('mrt_abort();return prepare_draw(prim,indexed);',prepare)
  self.assertIn('k->alt == 2',extract(s,'static int pipe_must_draw('))
  self.assertIn('k->alt || k->flat',extract(s,'static void pipe_record(const pkey *k) {'))

if __name__=='__main__':unittest.main()
