"""Geometry claims use bounded, fresh per-draw submission scratch."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]


def function(source, declaration):
    start=source.index(declaration);brace=source.index('{',start);depth=1;pos=brace+1
    while depth:
        depth+=(source[pos]=='{')-(source[pos]=='}');pos+=1
    return source[start:pos]


class Claims(unittest.TestCase):
    @unittest.skipUnless(shutil.which('cc'),'a host C compiler is required')
    def test_bounds_disjoint_storage_and_memo(self):
        dm=ROOT/'nxnative/dm/android';source=(dm/'gfx_vk_exec.c').read_text();header=(dm/'gfx_vk.h').read_text()
        enums=[]
        for name in ('RK_SUPPORT','RD_UBO','RS_NONE'):
            start=header.index('enum { '+name);enums.append(header[start:header.index('};',start)+2])
        start=source.index('static struct { uint32_t count, instances; int prim, indirect; } g_claim_draw;')
        end=source.index('\n}',source.index('static void gs_claim_prepare(',start))+2
        helper=source[start:end]
        memo=function(source,'static int memo_buffer(')+'\n'+function(source,'static int set_memo_ok(')
        dynamic=function(source,'static uint32_t storage_dynamic_offset(')
        hash_function=function(source,'static inline uint64_t fnv(')
        code=r'''
#include <stdint.h>
#include <stddef.h>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>
#include <string.h>
#include "gfx_vk_gs_claim_runtime.h"
ENUMS
typedef struct { int kind,samp,reg; } dmsv_bind;
typedef struct { uint64_t key; uint32_t gs_claim_invocations; int setmemo,nbind; const dmsv_bind *b; uint8_t dyn[96],tbl[96]; } vkshader_t;
typedef uint64_t VkDeviceSize;
#define VK_WHOLE_SIZE UINT64_MAX
typedef struct { uintptr_t buffer; uint64_t offset,range; } VkDescriptorBufferInfo;
static uint64_t g_set_epoch=1;static int g_buffer_memo=1;
static struct { struct { struct { uint64_t minStorageBufferOffsetAlignment,maxStorageBufferRange; } limits; } props; } V;
static unsigned char arena[GS_CLAIM_BUDGET+4096];static uint64_t cursor;static int allocations;
static uintptr_t vk_zero_buffer(void){return 1;}
static void *vk_scratch(uint64_t n,uint64_t align,uintptr_t *buffer,uint64_t *offset){
 cursor=(cursor+align-1)&~(align-1);assert(cursor+n<=sizeof arena);*buffer=2;*offset=cursor;void *p=arena+cursor;cursor+=n;allocations++;return p;
}
static void log_ignore(const char *f,...){(void)f;}
#define VKLOG(...) log_ignore(__VA_ARGS__)
HELPER
enum { MEMO_CB=16 };
MEMO
DYNAMIC
HASH
int main(void){
 unsetenv("DM_VK_GS_CLAIM_BUCKET");
 int why;uint64_t n;
 n=gs_claim_size(24,1,1,4,0,256,GS_CLAIM_BUDGET,1u<<27,&why);assert(n==100&&why==GS_CLAIM_OK);
 assert(gs_claim_size(24,0,1,4,0,256,GS_CLAIM_BUDGET,1u<<27,&why)==100); // NVN zero instance count means one.
 assert(!gs_claim_size(24,2,1,4,0,256,GS_CLAIM_BUDGET,1u<<27,&why)&&why==GS_CLAIM_INSTANCES);
 assert(!gs_claim_size(24,1,1,4,1,256,GS_CLAIM_BUDGET,1u<<27,&why)&&why==GS_CLAIM_INDIRECT);
 assert(!gs_claim_size(24,1,1,5,0,256,GS_CLAIM_BUDGET,1u<<27,&why)&&why==GS_CLAIM_TOPOLOGY);
 assert(!gs_claim_size(0,1,1,4,0,256,GS_CLAIM_BUDGET,1u<<27,&why)&&why==GS_CLAIM_EMPTY);
 assert(!gs_claim_size(UINT32_MAX,1,UINT32_MAX,4,0,256,GS_CLAIM_BUDGET,UINT64_MAX,&why)&&why==GS_CLAIM_SIZE);
 assert(gs_claim_size(GS_CLAIM_MAX_SLOTS,1,1,4,0,256,GS_CLAIM_BUDGET,1u<<27,&why)==4ull*GS_CLAIM_MAX_SLOTS+4);
 assert(!gs_claim_size(24,1,1,4,0,256,GS_CLAIM_BUDGET,99,&why)&&why==GS_CLAIM_LIMIT);
 assert(!gs_claim_size(24,1,1,4,0,256,354,1u<<27,&why)&&why==GS_CLAIM_MEMORY);
 assert(gs_claim_size(24,1,1,4,0,256,355,1u<<27,&why)==100);
 assert(!gs_claim_size(24,1,1,4,0,UINT64_MAX,GS_CLAIM_BUDGET,1u<<27,&why)&&why==GS_CLAIM_MEMORY);

 // Bucketing preserves the exact primitive bound, pads only modest allocations, and
 // never makes an otherwise valid exact allocation fail its range/budget checks.
 assert(gs_claim_range(100,256,GS_CLAIM_BUDGET,1u<<27,1)==256);
 assert(gs_claim_range(256,256,GS_CLAIM_BUDGET,1u<<27,1)==256);
 assert(gs_claim_range(260,256,GS_CLAIM_BUDGET,1u<<27,1)==512);
 assert(gs_claim_range(65536,256,GS_CLAIM_BUDGET,1u<<27,1)==65536);
 assert(gs_claim_range(65540,256,GS_CLAIM_BUDGET,1u<<27,1)==65540);
 assert(gs_claim_range(260,256,767,1u<<27,1)==512);
 assert(gs_claim_range(260,256,766,1u<<27,1)==260);
 assert(gs_claim_range(260,256,GS_CLAIM_BUDGET,511,1)==260);
 assert(gs_claim_range(100,256,GS_CLAIM_BUDGET,1u<<27,0)==100);
 assert(gs_claim_range(0,256,GS_CLAIM_BUDGET,1u<<27,1)==0);
 for(unsigned count=1;count<20000;count++){
  uint64_t exact=4+4ull*count,range=gs_claim_range(exact,256,GS_CLAIM_BUDGET,1u<<27,1);
  assert(range>=exact&&range+255<=GS_CLAIM_BUDGET);
  if(exact<=65536)assert(range>=256&&!(range&(range-1)));
  else assert(range==exact);
 }

 V.props.limits.minStorageBufferOffsetAlignment=256;V.props.limits.maxStorageBufferRange=1u<<27;
 const dmsv_bind synthetic={.kind=RK_GS_CLAIM};
 vkshader_t shader={.key=1,.gs_claim_invocations=1,.nbind=1,.b=&synthetic,.dyn={2}};
 g_claim_draw=(typeof(g_claim_draw)){24,1,4,0};gs_claim_prepare(&shader);
 VkDescriptorBufferInfo first=g_draw_claim;assert(first.buffer==2&&first.offset%256==0&&first.range==256);
 uint32_t *a=(uint32_t *)(arena+first.offset);assert(a[0]==1);for(unsigned i=1;i<first.range/4;i++)assert(a[i]==0);a[1]=7;
 g_claim_draw.count=30;gs_claim_prepare(&shader);VkDescriptorBufferInfo second=g_draw_claim;
 assert(second.buffer==2&&second.offset>=first.offset+first.range&&second.offset%256==0&&second.range==256);
 uint32_t *b=(uint32_t *)(arena+second.offset);assert(a[1]==7&&b[0]==1&&b[1]==0); // Prior in-flight draw is never cleared.
 for(unsigned i=1;i<second.range/4;i++)assert(b[i]==0); // padding belongs to this allocation and is zero too.
 // Real dynamic-offset extraction + real descriptor hash: different draw counts
 // share a set key while the bind keeps each draw's disjoint effective address.
 VkDescriptorBufferInfo da=first,db=second;
 uint32_t oa=storage_dynamic_offset(&da,256),ob=storage_dynamic_offset(&db,256);
 assert(oa!=ob&&oa==first.offset&&ob==second.offset);
 assert(da.offset==0&&db.offset==0&&fnv(1,&da,sizeof da)==fnv(1,&db,sizeof db));
 da.range=100;db.range=124;assert(fnv(1,&da,sizeof da)!=fnv(1,&db,sizeof db)); // old exact ranges defeated reuse.
 int old=allocations;g_claim_draw.instances=2;gs_claim_prepare(&shader);
 assert(g_draw_claim.buffer==1&&g_draw_claim.offset==0&&g_draw_claim.range==4&&allocations==old);
 g_claim_draw.instances=1;g_claim_draw.indirect=1;gs_claim_prepare(&shader);assert(g_draw_claim.buffer==1&&allocations==old);
 g_claim_draw.indirect=0;g_claim_draw.prim=7;gs_claim_prepare(&shader);assert(g_draw_claim.buffer==1&&allocations==old);
 g_claim_draw.prim=4;g_claim_draw.count=GS_CLAIM_MAX_SLOTS;
 for(int i=0;i<6;i++)gs_claim_prepare(&shader);
 assert(g_draw_claim.buffer==1&&allocations==old+3&&cursor<=GS_CLAIM_BUDGET); // Budget is per submission, not per draw.
 g_set_epoch++;cursor=0; // Test driver simulates submission completion before scratch recycling.
 g_claim_draw.count=24;gs_claim_prepare(&shader);assert(g_draw_claim.buffer==2&&g_draw_claim.offset==0&&g_draw_claim.range==256);
 setenv("DM_VK_GS_CLAIM_BUCKET","0",1);g_set_epoch++;cursor=0;gs_claim_prepare(&shader);
 assert(g_draw_claim.buffer==2&&g_draw_claim.range==100); // live A/B takes effect on the next submission.
 unsetenv("DM_VK_GS_CLAIM_BUCKET");g_set_epoch++;cursor=0;
 vkshader_t static_shader={.key=2,.gs_claim_invocations=1,.nbind=1,.b=&synthetic};
 gs_claim_prepare(&static_shader);assert(g_draw_claim.range==100); // Static offsets cannot share sets, so avoid padding.
 gs_claim_prepare(&shader);assert(g_draw_claim.range==256); // Switching back refreshes the cached immutable layout.
 gs_claim_prepare(NULL);assert(!g_draw_claim.buffer);
 shader.gs_claim_invocations=0;gs_claim_prepare(&shader);assert(!g_draw_claim.buffer);

 const dmsv_bind plain[]={{.kind=RK_CBUF,.samp=RS_NONE}};
 vkshader_t ordinary={.nbind=1,.b=plain};assert(set_memo_ok(&ordinary));
 const dmsv_bind claim[]={{.kind=RK_CBUF},{.kind=RK_GS_CLAIM}};
 vkshader_t claimed={.nbind=2,.b=claim};assert(set_memo_ok(&claimed)&&claimed.setmemo==3); // the claim is a refreshed memo dependency
 g_buffer_memo=0;assert(!set_memo_ok(&claimed)&&claimed.setmemo==3);g_buffer_memo=1; // DM_VK_BUFFER_MEMO=0 keeps claim shaders out of the memo
 return 0;
}
'''
        code=code.replace('ENUMS','\n'.join(enums)).replace('HELPER',helper).replace('MEMO\n',memo+'\n').replace('DYNAMIC\n',dynamic+'\n').replace('HASH\n',hash_function+'\n')
        target=ROOT/'target/tests';target.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='vk-gs-claim-runtime-',dir=target) as folder:
            src,exe=Path(folder)/'check.c',Path(folder)/'check';src.write_text(code)
            subprocess.run(['cc','-std=gnu11','-Wall','-Wextra','-Werror','-fsanitize=address,undefined','-I',str(dm),str(src),'-o',str(exe)],check=True)
            subprocess.run([str(exe)],check=True,timeout=15)

    def test_runtime_binding_and_input_capture(self):
        source=(ROOT/'nxnative/dm/android/gfx_vk_exec.c').read_text()
        for decl in ('static void draw(const p_draw *d) {','static void draw_idx(const p_draw_idx *d) {'):
            draw=function(source,decl)
            self.assertLess(draw.index('g_claim_draw ='),draw.index('prepare_draw('))
        prepare=function(source,'static int prepare_draw(')
        self.assertLess(prepare.index('gs_claim_prepare(S.sh[2])'),prepare.index('stage_set(st,'))
        self.assertIn('case RK_GS_CLAIM: bi[nw] = g_draw_claim.buffer ? g_draw_claim',source)
        self.assertIn('d->ind || d->mdi_cnt || g_bc_slot >= 0',source)


if __name__=='__main__':unittest.main()
