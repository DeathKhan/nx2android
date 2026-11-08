"""Exercise the actual GPU index diagnostic with a byte-copying Vulkan mock."""
from pathlib import Path
import os
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class IndexSnapshot(unittest.TestCase):
    @unittest.skipUnless(shutil.which('cc'), 'a host C compiler is required')
    def test_capture_timing_alignment_bounds_and_completion(self):
        source = (ROOT / 'nxnative/dm/android/gfx_vk_exec.c').read_text()
        start = source.index('// BEGIN original-index GPU snapshot diagnostic')
        helper = source[start:source.index('// END original-index GPU snapshot diagnostic', start)]
        code = r'''
#define _POSIX_C_SOURCE 200809L
#define VK_NO_PROTOTYPES
#include <vulkan/vulkan_core.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <unistd.h>
enum { OI_MAX = 1024 };
typedef struct {int count;uint64_t ibuf,ind;} p_draw_idx;
typedef struct {p_draw_idx draw;} original_index_entry;
static struct {VkDevice dev;} V;
static unsigned g_oi_refused,g_draw_i=17,g_sub_pass=8;
static uint64_t g_tex_frame=7,last_fence=100;
static unsigned char guest[16384];static void *mapped;
static int allocations,copies,ended,invalidated,destroyed,barriers;
#define VKCHK(x) assert((x)==VK_SUCCESS)
#define VKLOG(...) ((void)0)
static uint64_t mtl_last_fence(void){return last_fence;}
static int vk_buf_at(uint64_t a,uint64_t n,VkBuffer *b,VkDeviceSize *o){if(a>sizeof guest||n>sizeof guest-a)return 0;*b=(VkBuffer)(uintptr_t)1;*o=a;return 1;}
static uint32_t vk_mem_type(uint32_t b,VkMemoryPropertyFlags w,VkMemoryPropertyFlags a){(void)b;(void)w;(void)a;return 0;}
static VkResult vkCreateBuffer(VkDevice d,const VkBufferCreateInfo *i,const VkAllocationCallbacks *a,VkBuffer *b){(void)d;(void)a;assert(i->size==32*1024*1024);*b=(VkBuffer)(uintptr_t)2;return VK_SUCCESS;}
static void vkGetBufferMemoryRequirements(VkDevice d,VkBuffer b,VkMemoryRequirements *r){(void)d;(void)b;*r=(VkMemoryRequirements){.size=32*1024*1024,.memoryTypeBits=1};}
static VkResult vkAllocateMemory(VkDevice d,const VkMemoryAllocateInfo *i,const VkAllocationCallbacks *a,VkDeviceMemory *m){(void)d;(void)a;assert(i->allocationSize==32*1024*1024);*m=(VkDeviceMemory)(uintptr_t)3;allocations++;return VK_SUCCESS;}
static VkResult vkBindBufferMemory(VkDevice d,VkBuffer b,VkDeviceMemory m,VkDeviceSize o){(void)d;(void)b;(void)m;assert(!o);return VK_SUCCESS;}
static VkResult vkMapMemory(VkDevice d,VkDeviceMemory m,VkDeviceSize o,VkDeviceSize n,VkMemoryMapFlags f,void **p){(void)d;(void)m;(void)o;(void)n;(void)f;*p=mapped=calloc(1,32*1024*1024);assert(*p);return VK_SUCCESS;}
static void vk_end_pass(void){ended++;}
static VkCommandBuffer vk_cmd(void){return (VkCommandBuffer)(uintptr_t)4;}
static void vk_img_barrier_all(VkCommandBuffer b){(void)b;barriers++;}
static void vkCmdCopyBuffer(VkCommandBuffer b,VkBuffer s,VkBuffer d,uint32_t n,const VkBufferCopy *r){(void)b;assert(ended&&barriers&&s==(VkBuffer)(uintptr_t)1&&d==(VkBuffer)(uintptr_t)2&&n==1);assert(!(r->srcOffset%4)&&!(r->dstOffset%4)&&!(r->size%4));assert(r->srcOffset+r->size<=sizeof guest);memcpy((char*)mapped+r->dstOffset,guest+r->srcOffset,r->size);copies++;}
static void vkCmdPipelineBarrier(VkCommandBuffer c,VkPipelineStageFlags s,VkPipelineStageFlags d,VkDependencyFlags f,uint32_t n,const VkMemoryBarrier *m,uint32_t bn,const VkBufferMemoryBarrier *b,uint32_t in,const VkImageMemoryBarrier *i){(void)c;(void)f;(void)b;(void)i;assert(s==VK_PIPELINE_STAGE_TRANSFER_BIT&&d==VK_PIPELINE_STAGE_HOST_BIT&&n==1&&!bn&&!in);assert(m->srcAccessMask==VK_ACCESS_TRANSFER_WRITE_BIT&&m->dstAccessMask==VK_ACCESS_HOST_READ_BIT);}
static VkResult vkInvalidateMappedMemoryRanges(VkDevice d,uint32_t n,const VkMappedMemoryRange *m){(void)d;assert(n==1&&m->size==VK_WHOLE_SIZE);invalidated++;return VK_SUCCESS;}
static void vkDestroyBuffer(VkDevice d,VkBuffer b,const VkAllocationCallbacks *a){(void)d;(void)b;(void)a;destroyed++;}
static void vkFreeMemory(VkDevice d,VkDeviceMemory m,const VkAllocationCallbacks *a){(void)d;(void)m;(void)a;free(mapped);mapped=NULL;}
HELPER
int main(void){
 for(unsigned i=0;i<sizeof guest;i++)guest[i]=(unsigned char)(i*17+3);
 unsigned char before[sizeof guest];memcpy(before,guest,sizeof guest);
 original_index_entry r={.draw={.count=3,.ibuf=0x3002}};p_draw_idx d={.ind=0x1000,.ibuf=0x2002};
 setenv("DM_VK_ORIGINAL_INDEX_DUMP","11:22",1);setenv("DM_ROOT",".",1);
 original_index_snapshot(&r,&d);
 assert(g_oid.n==1&&g_oid.fence==101&&g_oid.used==36&&allocations==1&&copies==3);
 assert(!memcmp(guest,before,sizeof guest)); // Capturing never changes source memory.
 assert(g_oid.records[0].offset[0]==0&&g_oid.records[0].offset[1]==22&&g_oid.records[0].offset[2]==30);
 assert(!memcmp(g_oid.p+22,guest+0x2002,6)&&!memcmp(g_oid.p+30,guest+0x3002,6));
 assert(!strcmp(getenv("DM_VK_ORIGINAL_INDEX_DUMP"),"0"));
 original_index_dump_poll(100);assert(!invalidated&&!destroyed); // Never inspect/free in-flight bytes.
 last_fence=101;original_index_snapshot(&r,&d);assert(copies==3&&g_oid.n==1);last_fence=100;
 uint64_t used=g_oid.used;g_oid.used=OID_BYTES-4;original_index_snapshot(&r,&d);assert(copies==3&&g_oi_refused==1);g_oid.used=used;
 p_draw_idx bad=d;bad.ibuf=sizeof guest;original_index_snapshot(&r,&bad);assert(copies==3&&g_oi_refused==2);
 original_index_snapshot(&r,&d);assert(g_oid.n==2&&allocations==1&&copies==6&&g_oid.used==72);
 memset(guest,0,sizeof guest); // Later GPU writes must not change the earlier snapshot.
 original_index_dump_poll(101);assert(invalidated==1&&destroyed==1&&!mapped&&!g_oid.b);
 FILE *f=fopen("index_dump_7.bin","rb");assert(f);unsigned char got[72];assert(fread(got,1,sizeof got,f)==sizeof got&&fgetc(f)==EOF);fclose(f);
 assert(!memcmp(got,before+0x1000,20)&&!memcmp(got+22,before+0x2002,6)&&!memcmp(got+30,before+0x3002,6));
 f=fopen("index_dump_7.tsv","r");assert(f);char meta[4096];size_t z=fread(meta,1,sizeof meta-1,f);meta[z]=0;fclose(f);assert(strstr(meta,"source_count")&&strstr(meta,"0\t7\t8\t17\t3\t0x1000\t0x2002\t0x3002\t0\t22\t30\t6"));
 unlink("index_dump_7.bin");unlink("index_dump_7.tsv");return 0;
}
'''.replace('HELPER', helper)
        with tempfile.TemporaryDirectory(prefix='index-snapshot-') as tmp:
            p = Path(tmp); (p / 'test.c').write_text(code)
            subprocess.run(['cc', '-std=c11', '-O1', '-g', '-fsanitize=address,undefined',
                            '-I' + str(ROOT / 'nxnative/dm/third_party/vulkan-headers/include'),
                            str(p / 'test.c'), '-o', str(p / 'test')], check=True, text=True)
            subprocess.run([str(p / 'test')], cwd=p, check=True, text=True)


if __name__ == '__main__':
    unittest.main()
