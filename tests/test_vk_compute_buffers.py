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
        s=(ROOT/"nxnative/dm/android/gfx_vk_exec.c").read_text()
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
enum {RK_SBUF=1,RK_CBUF=2,RK_DRV=3,RK_C1=4};
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
 cs_trace_buffer_note(&immutable,&bi);assert(!g_cs_buffers_n);
 cs_trace_buffer_note(&storage,&bi);cs_trace_buffer_note(&uniform,&bi);
 assert(g_cs_buffers_n==2&&g_cs_buffer_copies==3&&g_cs_buffers[0].guest==0x12340100);
 cs_trace_buffers_copy(VK_NULL_HANDLE,0x21d9,0);assert(copies==2&&g_cs_buffer_dump[0].source_offset==256);
 cs_trace_buffers_copy(VK_NULL_HANDLE,0x21d9,1);assert(copies==3&&barriers==6&&host_barriers==3);
 assert(g_cs_buffer_dump[0].fence==7&&g_cs_buffer_dump[2].after);
 cs_trace_buffers_poll(6);assert(!invalidations&&!destroys&&!frees); // absolutely no CPU read before completed fence
 gpu_complete();cs_trace_buffers_poll(7);assert(invalidations==3&&destroys==3&&frees==3);
 char path[1024];snprintf(path,sizeof path,"%s/csbuf_0001_00000000000021d9_b4_before.bin",argv[1]);
 FILE *f=fopen(path,"rb");assert(f);uint8_t got[192];assert(fread(got,1,sizeof got,f)==sizeof got&&!memcmp(got,source+256,sizeof got));assert(fgetc(f)==EOF);fclose(f);
 for(unsigned i=0;i<CS_BUFFER_DUMPS;i++)assert(!g_cs_buffer_dump[i].b);
 reset_notes();g_cs_buffer_collect=0;cs_trace_buffer_note(&storage,&bi);assert(!g_cs_buffers_n);g_cs_buffer_collect=1;
 bi.buffer=vk_zero_buffer();cs_trace_buffer_note(&storage,&bi);assert(!g_cs_buffers_n);bi.buffer=(VkBuffer)&buffer;
 bi.offset=1;cs_trace_buffer_note(&storage,&bi);assert(!g_cs_buffers_n);bi.offset=256;
 bi.range=3;cs_trace_buffer_note(&storage,&bi);assert(!g_cs_buffers_n);bi.range=VK_WHOLE_SIZE;cs_trace_buffer_note(&storage,&bi);assert(!g_cs_buffers_n);
 bi.range=192;g_cs_buffer_copies=63;cs_trace_buffer_note(&storage,&bi);assert(!g_cs_buffers_n);cs_trace_buffer_note(&uniform,&bi);assert(g_cs_buffers_n==1&&g_cs_buffer_copies==64);
 reset_notes();for(unsigned i=0;i<63;i++)g_cs_buffer_dump[i].b=(VkBuffer)(uintptr_t)1;
 cs_trace_buffer_note(&storage,&bi);assert(!g_cs_buffers_n);cs_trace_buffer_note(&uniform,&bi);assert(g_cs_buffers_n==1);memset(g_cs_buffer_dump,0,sizeof g_cs_buffer_dump);
 reset_notes();bi.offset=0;bi.range=70000;cs_trace_buffer_note(&uniform,&bi);complete=0;
 cs_trace_buffers_copy(VK_NULL_HANDLE,0x9e6,0);assert(g_cs_buffer_dump[0].bytes==65536&&g_cs_buffer_dump[0].source_range==70000);
 gpu_complete();cs_trace_buffers_poll(7);
 reset_notes();bi.range=192;cs_trace_buffer_note(&storage,&bi);unsigned old=copies,old_d=destroys,old_f=frees;
 fail_alloc=1;cs_trace_buffers_copy(VK_NULL_HANDLE,1,0);assert(copies==old&&destroys==old_d+1&&frees==old_f);fail_alloc=0;
 fail_map=1;cs_trace_buffers_copy(VK_NULL_HANDLE,1,0);assert(copies==old&&destroys==old_d+2&&frees==old_f+1);fail_map=0;
 for(unsigned i=0;i<CS_BUFFER_DUMPS;i++)assert(!g_cs_buffer_dump[i].b);
 return 0;
}
"""
        code=code.replace("ACTUAL",actual)
        target=ROOT/"target/tests";target.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="compute-buffers-",dir=target) as tmp:
            c,exe=Path(tmp)/"test.c",Path(tmp)/"test";c.write_text(code)
            subprocess.run(["cc","-std=c11","-O1","-g","-Wall","-Wextra","-Werror","-fsanitize=address,undefined",
                            "-I"+str(ROOT/"nxnative/dm/third_party/vulkan-headers/include"),str(c),"-o",str(exe)],check=True)
            subprocess.run([str(exe),tmp],check=True,timeout=15)

    def test_actual_dispatch_and_present_order(self):
        s=(ROOT/"nxnative/dm/android/gfx_vk_exec.c").read_text()
        dispatch=extract(s,"static void dispatch(")
        self.assertLess(dispatch.index("cs_trace_buffers_copy(cb, cs->key, 0)"),dispatch.index("vkCmdDispatch(cb"))
        self.assertGreater(dispatch.index("cs_trace_buffers_copy(cb, cs->key, 1)"),dispatch.index("vkCmdDispatch(cb"))
        stage=extract(s,"static VkDescriptorSet stage_set(")
        self.assertLess(stage.index("cs_trace_buffer_note(b, &bi[nw])"),stage.index("if (s->dyn[i])"))
        poll=extract(s,"void vk_pass_dump_poll(")
        self.assertLess(poll.index("cs_trace_buffers_poll(done_fence)"),poll.index("if (g_pd_state"))
if __name__=="__main__":unittest.main()
