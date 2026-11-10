"""Exact compute-buffer readbacks keep source offsets and wait for their GPU fence."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from test_vk_compute_trace import extract

ROOT=Path(__file__).resolve().parents[1]

@unittest.skipUnless(shutil.which("cc"),"host C compiler required")
class ComputeBuffers(unittest.TestCase):
    def test_actual_gpu_copy_bounds_and_completion(self):
        s=(ROOT/"nxnative/dm/android/gfx_vk_exec.c").read_text(encoding='utf-8')
        start=s.index("enum { CS_BUFFER_BINDINGS")
        actual=s[start:s.index("// Bounded native-upscaler diagnostics",start)]
        code=r"""
#define _POSIX_C_SOURCE 200809L
#include <vulkan/vulkan_core.h>
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
typedef struct {int kind,reg;uint32_t binding;} dmsv_bind;
enum {RK_SBUF=1,RK_CBUF=2,RK_DRV=3,RK_C1=4,RK_SBUF_UBO=5,SB_ALIGN=256};
typedef struct { unsigned stage, reg; uint64_t offset, bytes; } fs_buffer_window;
static fs_buffer_window g_fs_buffer_windows[8];
static unsigned g_fs_buffer_windows_n,g_fs_buffer_copies;
enum {FS_BUFFER_BYTE_BUDGET=48<<20};
static uint32_t g_fs_buffer_sbuf_max=65536;static uint64_t g_fs_buffer_bytes;
static int g_fs_buffer_collect,g_draw_i=42,g_sub_pass=7;
static struct {struct {uint64_t addr;} ssbo[6][16],ubo[6][16];} S;
static struct {VkDevice dev;} V;
static uint64_t g_tex_frame=99;
static unsigned g_cs_trace_seen=3,g_cs_buffer_copies;
static int g_cs_buffer_collect=1;
static unsigned barriers,host_barriers,copies,invalidations,destroys,frees;
static int fail_alloc,fail_map,complete;
static uint64_t mtl_last_fence(void){return 6;}
static uint64_t vk_frame_current(void){return 123;}
static VkBuffer vk_zero_buffer(void){return (VkBuffer)(uintptr_t)0xdead;}
#define VKLOG(...) do{if(0)fprintf(stderr,__VA_ARGS__);}while(0)
typedef struct {uint8_t *p;VkDeviceSize size;} mock_buffer;
typedef struct {mock_buffer *src,*dst;VkBufferCopy copy;} pending_copy;
static pending_copy pending[128];static unsigned npending;
static uint32_t vk_mem_type(uint32_t bits,VkMemoryPropertyFlags want,VkMemoryPropertyFlags avoid){(void)bits;(void)want;(void)avoid;return 0;}
static void vk_img_barrier_all(VkCommandBuffer cb){(void)cb;barriers++;}
VkResult vkCreateBuffer(VkDevice d,const VkBufferCreateInfo *i,const VkAllocationCallbacks *a,VkBuffer *out){
 (void)d;(void)a;assert(i->usage==VK_BUFFER_USAGE_TRANSFER_DST_BIT);mock_buffer *b=calloc(1,sizeof *b);assert(b);b->size=i->size;*out=(VkBuffer)b;return VK_SUCCESS;
}
void vkGetBufferMemoryRequirements(VkDevice d,VkBuffer b,VkMemoryRequirements *r){(void)d;*r=(VkMemoryRequirements){((mock_buffer *)b)->size,4,1};}
VkResult vkAllocateMemory(VkDevice d,const VkMemoryAllocateInfo *i,const VkAllocationCallbacks *a,VkDeviceMemory *m){
 (void)d;(void)a;if(fail_alloc)return VK_ERROR_OUT_OF_HOST_MEMORY;void *p=malloc(i->allocationSize);assert(p);memset(p,0xcd,i->allocationSize);*m=(VkDeviceMemory)p;return VK_SUCCESS;
}
VkResult vkBindBufferMemory(VkDevice d,VkBuffer b,VkDeviceMemory m,VkDeviceSize off){(void)d;assert(!off);((mock_buffer *)b)->p=(void *)m;return VK_SUCCESS;}
VkResult vkMapMemory(VkDevice d,VkDeviceMemory m,VkDeviceSize off,VkDeviceSize size,VkMemoryMapFlags f,void **p){
 (void)d;(void)off;(void)size;(void)f;if(fail_map)return VK_ERROR_MEMORY_MAP_FAILED;*p=(void *)m;return VK_SUCCESS;
}
void vkDestroyBuffer(VkDevice d,VkBuffer b,const VkAllocationCallbacks *a){(void)d;(void)a;destroys++;free((void *)b);}
void vkFreeMemory(VkDevice d,VkDeviceMemory m,const VkAllocationCallbacks *a){(void)d;(void)a;frees++;free((void *)m);}
void vkCmdCopyBuffer(VkCommandBuffer cb,VkBuffer src,VkBuffer dst,uint32_t n,const VkBufferCopy *c){
 (void)cb;assert(n==1&&npending<128);assert(c->srcOffset+c->size<=((mock_buffer *)src)->size);
 pending[npending++]=(pending_copy){(void *)src,(void *)dst,*c};copies++;
}
void vkCmdPipelineBarrier(VkCommandBuffer cb,VkPipelineStageFlags src,VkPipelineStageFlags dst,VkDependencyFlags flags,
 uint32_t nm,const VkMemoryBarrier *m,uint32_t nb,const VkBufferMemoryBarrier *b,uint32_t ni,const VkImageMemoryBarrier *im){
 (void)cb;(void)flags;(void)m;(void)im;assert(src==VK_PIPELINE_STAGE_TRANSFER_BIT&&dst==VK_PIPELINE_STAGE_HOST_BIT);
 assert(!nm&&!ni&&nb==1&&b->srcAccessMask==VK_ACCESS_TRANSFER_WRITE_BIT&&b->dstAccessMask==VK_ACCESS_HOST_READ_BIT);
 assert(b->srcQueueFamilyIndex==VK_QUEUE_FAMILY_IGNORED&&b->dstQueueFamilyIndex==VK_QUEUE_FAMILY_IGNORED);host_barriers++;
}
VkResult vkInvalidateMappedMemoryRanges(VkDevice d,uint32_t n,const VkMappedMemoryRange *r){
 (void)d;assert(complete&&n==1&&!r->offset&&r->size==VK_WHOLE_SIZE);invalidations++;return VK_SUCCESS;
}
ACTUAL
static void gpu_complete(void){for(unsigned i=0;i<npending;i++){pending_copy *p=&pending[i];memcpy(p->dst->p+p->copy.dstOffset,p->src->p+p->copy.srcOffset,p->copy.size);}npending=0;complete=1;}
static void reset_notes(void){g_cs_buffers_n=g_cs_buffers_reserved=g_cs_buffer_copies=0;}
int main(int argc,char **argv){
 assert(argc==2);setenv("DM_ROOT",argv[1],1);uint8_t source[70000];for(unsigned i=0;i<sizeof source;i++)source[i]=(uint8_t)(i*17);
 mock_buffer buffer={source,sizeof source};VkDescriptorBufferInfo bi={(VkBuffer)&buffer,256,192};
 S.ssbo[5][0].addr=0x12340100;S.ubo[5][2].addr=0x887700;
 dmsv_bind storage={RK_SBUF,0,4},uniform={RK_CBUF,2,7},immutable={RK_C1,0,9};
 cs_trace_buffer_note(5,0x21d9,&immutable,&bi);assert(!g_cs_buffers_n);
 cs_trace_buffer_note(5,0x21d9,&storage,&bi);cs_trace_buffer_note(5,0x21d9,&uniform,&bi);
 assert(g_cs_buffers_n==2&&g_cs_buffer_copies==3&&g_cs_buffers[0].guest==0x12340100);
 cs_trace_buffers_copy(VK_NULL_HANDLE,0x21d9,0);assert(copies==2&&g_cs_buffer_dump[0].source_offset==256);
 cs_trace_buffers_copy(VK_NULL_HANDLE,0x21d9,1);assert(copies==3&&barriers==6&&host_barriers==3);
 assert(g_cs_buffer_dump[0].fence==7&&g_cs_buffer_dump[2].after);
 cs_trace_buffers_poll(6);assert(!invalidations&&!destroys&&!frees); // absolutely no CPU read before completed fence
 gpu_complete();cs_trace_buffers_poll(7);assert(invalidations==3&&destroys==3&&frees==3);
 char path[1024];snprintf(path,sizeof path,"%s/csbuf_0001_00000000000021d9_b4_before.bin",argv[1]);
 FILE *f=fopen(path,"rb");assert(f);uint8_t got[192];assert(fread(got,1,sizeof got,f)==sizeof got&&!memcmp(got,source+256,sizeof got));assert(fgetc(f)==EOF);fclose(f);
 for(unsigned i=0;i<CS_BUFFER_DUMPS;i++)assert(!g_cs_buffer_dump[i].b);
 reset_notes();g_cs_buffer_collect=0;cs_trace_buffer_note(5,0x21d9,&storage,&bi);assert(!g_cs_buffers_n);g_cs_buffer_collect=1;
 bi.buffer=vk_zero_buffer();cs_trace_buffer_note(5,0x21d9,&storage,&bi);assert(!g_cs_buffers_n);bi.buffer=(VkBuffer)&buffer;
 bi.offset=1;cs_trace_buffer_note(5,0x21d9,&storage,&bi);assert(!g_cs_buffers_n);bi.offset=256;
 bi.range=3;cs_trace_buffer_note(5,0x21d9,&storage,&bi);assert(!g_cs_buffers_n);bi.range=VK_WHOLE_SIZE;cs_trace_buffer_note(5,0x21d9,&storage,&bi);assert(!g_cs_buffers_n);
 bi.range=192;g_cs_buffer_copies=63;cs_trace_buffer_note(5,0x21d9,&storage,&bi);assert(!g_cs_buffers_n);cs_trace_buffer_note(5,0x21d9,&uniform,&bi);assert(g_cs_buffers_n==1&&g_cs_buffer_copies==64);
 reset_notes();for(unsigned i=0;i<63;i++)g_cs_buffer_dump[i].b=(VkBuffer)(uintptr_t)1;
 cs_trace_buffer_note(5,0x21d9,&storage,&bi);assert(!g_cs_buffers_n);cs_trace_buffer_note(5,0x21d9,&uniform,&bi);assert(g_cs_buffers_n==1);memset(g_cs_buffer_dump,0,sizeof g_cs_buffer_dump);
 reset_notes();bi.offset=0;bi.range=70000;cs_trace_buffer_note(5,0x21d9,&uniform,&bi);complete=0;
 cs_trace_buffers_copy(VK_NULL_HANDLE,0x9e6,0);assert(g_cs_buffer_dump[0].bytes==65536&&g_cs_buffer_dump[0].source_range==70000);
 gpu_complete();cs_trace_buffers_poll(7);
 reset_notes();bi.range=192;cs_trace_buffer_note(5,0x21d9,&storage,&bi);unsigned old=copies,old_d=destroys,old_f=frees;
 fail_alloc=1;cs_trace_buffers_copy(VK_NULL_HANDLE,1,0);assert(copies==old&&destroys==old_d+1&&frees==old_f);fail_alloc=0;
 fail_map=1;cs_trace_buffers_copy(VK_NULL_HANDLE,1,0);assert(copies==old&&destroys==old_d+2&&frees==old_f+1);fail_map=0;

 for(unsigned i=0;i<CS_BUFFER_DUMPS;i++)assert(!g_cs_buffer_dump[i].b);
 reset_notes();g_cs_buffer_collect=0;g_fs_buffer_collect=1;
 S.ssbo[1][0].addr=0x80000120;S.ssbo[0][0].addr=0x90000120;
 bi.offset=256;bi.range=66000;g_fs_buffer_windows_n=3;
 g_fs_buffer_windows[0]=(fs_buffer_window){1,0,0x10000,16};
 g_fs_buffer_windows[1]=(fs_buffer_window){1,0,UINT64_MAX-3,16}; // overflow refuses
 g_fs_buffer_windows[2]=(fs_buffer_window){1,0,65960,32}; // crosses actual descriptor end
 unsigned before=copies;
 cs_trace_buffer_note(1,0xabc,&storage,&bi);
 assert(g_cs_buffers_n==2&&g_fs_buffer_copies==2);
 assert(g_cs_buffers[0].guest==0x80000100&&g_cs_buffers[1].guest==0x80010120);
 assert(g_cs_buffers[1].source.offset==256+32+65536&&g_cs_buffers[1].source.range==16);
 assert(g_cs_buffers[1].descriptor_offset==256&&g_cs_buffers[1].descriptor_range==66000);
 assert(g_cs_buffers[1].window_offset==65568&&g_cs_buffers[1].stage==1);
 complete=0;cs_trace_buffers_copy(VK_NULL_HANDLE,0xdeadbeef,0);cs_trace_buffers_copy(VK_NULL_HANDLE,0xdeadbeef,1);
 assert(copies==before+2&&g_cs_buffer_dump[1].draw==42&&g_cs_buffer_dump[1].pass==7);
 uint64_t id=g_cs_buffer_dump[1].id;
 cs_trace_buffers_poll(6);assert(g_cs_buffer_dump[1].b);gpu_complete();cs_trace_buffers_poll(7);
 snprintf(path,sizeof path,"%s/fsbuf_%04llu_00000000deadbeef_s1_b4_draw42_before.bin",argv[1],(unsigned long long)id);
 f=fopen(path,"rb");assert(f);assert(fread(got,1,16,f)==16&&!memcmp(got,source+256+32+65536,16));assert(fgetc(f)==EOF);fclose(f);
 char meta[1100],json[2048];snprintf(meta,sizeof meta,"%s.json",path);f=fopen(meta,"rb");assert(f);size_t j=fread(json,1,sizeof json-1,f);json[j]=0;fclose(f);
 assert(strstr(json,"\"stage\":1")&&strstr(json,"\"shader_key\":\"0000000000000abc\"")&&strstr(json,"\"window_offset\":65568")&&strstr(json,"\"fence\":7"));
 reset_notes();g_fs_buffer_windows_n=0;bi.range=192;g_trace_sbuf_guest=0x66660000;
 dmsv_bind derived={RK_SBUF_UBO,3,11};cs_trace_buffer_note(0,0xa,&derived,&bi);cs_trace_buffer_note(0,0xa,&storage,&bi);
 assert(g_cs_buffers_n==2&&g_cs_buffers[0].guest==0x66660000&&g_cs_buffers[1].guest==0x90000100);
 // Extended graphics storage captures preserve the actual finite offset/range and remain fenced.
 reset_notes();g_fs_buffer_copies=0;g_fs_buffer_bytes=0;g_fs_buffer_sbuf_max=8u<<20;
 uint8_t *large=malloc((8u<<20)+256);assert(large);
 for(unsigned i=0;i<(8u<<20)+256;i++)large[i]=(uint8_t)(i*13+7);
 mock_buffer big={large,(8u<<20)+256};bi=(VkDescriptorBufferInfo){(VkBuffer)&big,256,5767200};
 cs_trace_buffer_note(1,0x396,&storage,&bi);assert(g_cs_buffers_n==1&&g_cs_buffers[0].bytes==5767200);
 complete=0;cs_trace_buffers_copy(VK_NULL_HANDLE,0x396,0);assert(g_cs_buffer_dump[0].bytes==5767200);
 cs_trace_buffers_poll(6);assert(g_cs_buffer_dump[0].b);gpu_complete();
 assert(!memcmp(g_cs_buffer_dump[0].p,large+256,5767200));cs_trace_buffers_poll(7);
 reset_notes();g_fs_buffer_copies=0;g_fs_buffer_bytes=0;
 cs_trace_buffer_note(1,0x396,&uniform,&bi);assert(g_cs_buffers[0].bytes==65536); // only SBUF gets the larger cap
 reset_notes();g_fs_buffer_collect=0;g_cs_buffer_collect=1;
 cs_trace_buffer_note(5,0x396,&storage,&bi);assert(g_cs_buffers[0].bytes==65536); // compute unchanged
 reset_notes();g_fs_buffer_collect=1;g_fs_buffer_copies=0;g_fs_buffer_bytes=0;bi.range=8u<<20;
 for(int i=0;i<6;i++)cs_trace_buffer_note(1,0x396,&storage,&bi);
 assert(g_cs_buffers_n==6&&g_fs_buffer_bytes==48u<<20);
 cs_trace_buffer_note(1,0x396,&storage,&bi);assert(g_cs_buffers_n==6); // request cap
 reset_notes();g_fs_buffer_copies=0;g_fs_buffer_bytes=0;
 for(int i=0;i<6;i++){g_cs_buffer_dump[i].b=(VkBuffer)(uintptr_t)1;g_cs_buffer_dump[i].graphics=1;g_cs_buffer_dump[i].bytes=8u<<20;}
 cs_trace_buffer_note(1,0x396,&storage,&bi);assert(!g_cs_buffers_n); // rearming cannot bypass in-flight cap
 memset(g_cs_buffer_dump,0,sizeof g_cs_buffer_dump);
 bi.range=VK_WHOLE_SIZE;cs_trace_buffer_note(1,0x396,&storage,&bi);assert(!g_cs_buffers_n);
 bi.range=8u<<20;bi.offset=UINT64_MAX-3;cs_trace_buffer_note(1,0x396,&storage,&bi);assert(!g_cs_buffers_n);
 free(large);
 return 0;
}
"""
        code=code.replace("ACTUAL",actual)
        target=ROOT/"target/tests";target.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="compute-buffers-",dir=target) as tmp:
            c,exe=Path(tmp)/"test.c",Path(tmp)/"test";c.write_text(code, encoding='utf-8')
            subprocess.run(["cc","-std=c11","-O1","-g","-Wall","-Wextra","-Werror","-fsanitize=address,undefined",
                            "-I"+str(ROOT/"nxnative/dm/third_party/vulkan-headers/include"),str(c),"-o",str(exe)],check=True)
            subprocess.run([str(exe),tmp],check=True,timeout=15)

    def test_actual_dispatch_and_present_order(self):
        s=(ROOT/"nxnative/dm/android/gfx_vk_exec.c").read_text(encoding='utf-8')
        dispatch=extract(s,"static void dispatch(")
        self.assertLess(dispatch.index("cs_trace_buffers_copy(cb, cs->key, 0)"),dispatch.index("vkCmdDispatch(cb"))
        self.assertGreater(dispatch.index("cs_trace_buffers_copy(cb, cs->key, 1)"),dispatch.index("vkCmdDispatch(cb"))
        stage=extract(s,"static VkDescriptorSet stage_set(")
        self.assertLess(stage.index("cs_trace_buffer_note(st, s->key, b, &bi[nw])"),stage.index("if (s->dyn[i])"))
        poll=extract(s,"void vk_pass_dump_poll(")
        self.assertLess(poll.index("cs_trace_buffers_poll(done_fence)"),poll.index("if (g_pd_state"))
if __name__=="__main__":unittest.main()
