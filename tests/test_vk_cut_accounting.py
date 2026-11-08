"""Actual command-buffer split accounting and bounded diagnostic lifecycle."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]

def function(source,declaration):
    start=source.index(declaration);pos=source.index('{',start)+1;depth=1
    while depth:
        depth+=(source[pos]=='{')-(source[pos]=='}');pos+=1
    return source[start:pos]

class CutAccounting(unittest.TestCase):
    @unittest.skipUnless(shutil.which('cc'),'a host C compiler is required')
    def test_actual_native_split_and_pass_accounting(self):
        dm=ROOT/'nxnative/dm'
        core=(dm/'android/gfx_vk_core.c').read_text()
        exe=(dm/'android/gfx_vk_exec.c').read_text()
        helpers=core[core.index('static int vk_cmd_split_(void);'):core.index('uint64_t g_up_gen;')]
        # sub_stats (the long-submission report) reads replay counters other units define: defined here instead
        helpers=helpers.replace('extern struct { uint64_t draws, skip_prog','static struct { uint64_t draws, skip_prog').replace('extern uint64_t g_set_new, g_pkg_n, g_pkg_bytes;','static uint64_t g_set_new, g_pkg_n, g_pkg_bytes;')
        code=r"""
#define _POSIX_C_SOURCE 200809L
#define VK_NO_PROTOTYPES
#include <vulkan/vulkan_core.h>
#include <assert.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdarg.h>
#include <time.h>
#include <pthread.h>
#define MAIN_CBS 32
typedef struct {
 VkCommandPool pool;VkCommandBuffer main,up,mains[MAIN_CBS],ups[MAIN_CBS];
 int nmain_used,nmain_alloc,nmain_sent,nup_used,nup_alloc,cb_err,up_used,cut_diag;
 uint32_t cut_success,cut_budget_refused,cut_capacity_refused;uint64_t cut_cpu_ns;
} vksub_t;
static vksub_t slot;
static vksub_t *cur(void){return &slot;}
static struct { VkDevice dev;VkQueue q; } V;
static int g_prof_on,clock_calls,end_calls,begin_calls,queue_calls,readbacks,lands,log_calls;
static uint64_t g_prof_ns[8],g_cmd_gen;
static pthread_mutex_t g_vk_queue_mu=PTHREAD_MUTEX_INITIALIZER;
static char log_line[1024];
static void log_message(const char *f,...){if(strncmp(f,"cut stats",9))return;va_list a;va_start(a,f);vsnprintf(log_line,sizeof log_line,f,a);va_end(a);log_calls++;}
#define VKLOG(...) log_message(__VA_ARGS__)
#define VKCHK(x) assert((x)==VK_SUCCESS)
static void die(const char *f,...){(void)f;abort();}
static void vk_crumbs_report(void){}
static void pools_readback_barrier(VkCommandBuffer cb){assert(cb==slot.main);readbacks++;}
static void vk_img_barrier_all(VkCommandBuffer cb){assert(cb==slot.up);}
static VkResult vkEndCommandBuffer(VkCommandBuffer cb){assert(cb);end_calls++;return VK_SUCCESS;}
static VkResult vkBeginCommandBuffer(VkCommandBuffer cb,const VkCommandBufferBeginInfo *b){assert(cb&&b);begin_calls++;return VK_SUCCESS;}
static VkResult vkQueueSubmit(VkQueue q,uint32_t n,const VkSubmitInfo *s,VkFence f){(void)q;assert(n==1&&s->commandBufferCount>0&&!f);queue_calls++;return VK_SUCCESS;}
static VkResult vkAllocateCommandBuffers(VkDevice d,const VkCommandBufferAllocateInfo *a,VkCommandBuffer *p){(void)d;assert(a->commandBufferCount==1);static uintptr_t next=100;*p=(VkCommandBuffer)next++;return VK_SUCCESS;}
static int mock_clock(clockid_t c,struct timespec *t){assert(c==CLOCK_MONOTONIC);clock_calls++;*t=(struct timespec){0,clock_calls*100};return 0;}
#define clock_gettime mock_clock
static uint64_t g_pipe_kgsl;static uint64_t kgsl_mapped(void){return 0;}static void vk_stage_flush(void){}
HELPERS
#undef clock_gettime
static int g_ord_parts,g_cb_draws,g_x_landcut;
static void vk_ubd_land(VkCommandBuffer cb){assert(cb==slot.main);lands++;}
PASS_CUT
static void reset_slot(void){
 memset(&slot,0,sizeof slot);slot.main=slot.mains[0]=(VkCommandBuffer)(uintptr_t)1;
 slot.up=slot.ups[0]=(VkCommandBuffer)(uintptr_t)2;
 slot.nmain_used=slot.nmain_alloc=slot.nup_used=slot.nup_alloc=1;
 g_ord_parts=0;g_cb_draws=128;g_x_landcut=0;
}
int main(void){
 unsetenv("DM_VK_CUT_STATS");reset_slot();cut_stats_begin(&slot);
 uint64_t cuts=9;
 assert(pass_cut(0,&cuts)==1&&cuts==10&&g_cb_draws==0&&g_ord_parts==1);
 assert(slot.nmain_used==2&&slot.nmain_sent==1&&queue_calls==1);
 assert(clock_calls==0&&slot.cut_success==0);cut_stats_report(&slot,4,5);assert(!log_calls);
 // Refused ordinary budget cannot generate a fictional depth/LRZ boundary or land deferred updates.
 g_ord_parts=15;g_cb_draws=999;g_x_landcut=1;
 uint64_t prior_depth_cut=cuts;int prior_queue=queue_calls;
 assert(!pass_cut(0,&cuts)&&cuts==prior_depth_cut&&g_cb_draws==999&&g_ord_parts==15);
 assert(!lands&&queue_calls==prior_queue);
 // Interior splits use the remaining native capacity despite ordinary budget exhaustion.
 assert(pass_cut(1,&cuts)&&cuts==prior_depth_cut+1&&g_ord_parts==15&&g_cb_draws==0&&lands==1);
 // Both native arrays bound capacity. No landing or draw reset when either is full.
 g_cb_draws=128;prior_depth_cut=cuts;int prior_lands=lands;
 slot.nmain_used=MAIN_CBS;
 assert(!pass_cut(1,&cuts)&&cuts==prior_depth_cut&&g_cb_draws==128&&lands==prior_lands);
 assert(!vk_cmd_split()&&slot.nmain_used==MAIN_CBS);
 slot.nmain_used=3;slot.nup_used=MAIN_CBS;
 assert(!pass_cut(1,&cuts)&&cuts==prior_depth_cut&&g_cb_draws==128&&lands==prior_lands);
 assert(!vk_cmd_split());
 // The live diagnostic includes successful cuts and both refusal classes, with real split timing.
 setenv("DM_VK_CUT_STATS","2",1);reset_slot();cut_stats_begin(&slot);assert(slot.cut_diag);
 cuts=0;g_cb_draws=256;assert(pass_cut(0,&cuts));
 g_ord_parts=15;assert(!pass_cut(0,&cuts));
 slot.nmain_used=MAIN_CBS;assert(!pass_cut(1,&cuts));assert(!vk_cmd_split());
 assert(slot.cut_success==1&&slot.cut_budget_refused==1&&slot.cut_capacity_refused==2);
 assert(slot.cut_cpu_ns==200&&clock_calls==4);
 cut_stats_report(&slot,123,456);
 assert(log_calls==1&&strstr(log_line,"frame 123 fence 456")&&strstr(log_line,"parts 32/32")&&strstr(log_line,"successful 1")&&strstr(log_line,"budget 1/capacity 2"));
 reset_slot();cut_stats_begin(&slot);assert(slot.cut_diag&&!slot.cut_success&&!slot.cut_cpu_ns);
 cut_stats_begin(&slot);assert(!slot.cut_diag); // budget consumed once, not rearmed every submission
 setenv("DM_VK_CUT_STATS","0",1);cut_stats_begin(&slot);assert(!slot.cut_diag);
 setenv("DM_VK_CUT_STATS","999",1);
 for(int i=0;i<64;i++){cut_stats_begin(&slot);assert(slot.cut_diag);}
 cut_stats_begin(&slot);assert(!slot.cut_diag);
 // Default path reaches exactly32 parts and leaves refused31st-boundary state unchanged.
 setenv("DM_VK_CUT_STATS","0",1);reset_slot();cut_stats_begin(&slot);cuts=0;
 prior_queue=queue_calls;int prior_clocks=clock_calls;
 for(int i=0;i<31;i++){g_cb_draws=128;assert(pass_cut(1,&cuts));}
 assert(cuts==31&&slot.nmain_used==32&&slot.nmain_sent==31&&queue_calls==prior_queue+31);
 g_cb_draws=777;assert(!pass_cut(1,&cuts)&&cuts==31&&g_cb_draws==777&&clock_calls==prior_clocks);
 // Deferred-submit mode still creates a real boundary, with no early queue submission.
 setenv("DM_VK_SUBMIT_SPLIT","0",1);reset_slot();cut_stats_begin(&slot);cuts=0;prior_queue=queue_calls;
 assert(pass_cut(0,&cuts)&&cuts==1&&slot.nmain_used==2&&slot.nmain_sent==0&&queue_calls==prior_queue);
 // Existing slow-profiler timing works independently when the bounded diagnostic is off.
 g_prof_on=1;assert(vk_cmd_split());assert(g_prof_ns[6]==100&&!slot.cut_cpu_ns);
 return 0;
}
"""
        code=code.replace('HELPERS',helpers).replace('PASS_CUT',function(exe,'static int pass_cut('))
        target=ROOT/'target/tests';target.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='cut-accounting-',dir=target) as t:
            src=Path(t)/'test.c';out=Path(t)/'test';src.write_text(code)
            subprocess.run(['cc','-std=c11','-g','-O1','-fsanitize=address,undefined','-fno-omit-frame-pointer','-pthread',
                            '-I'+str(dm/'third_party/vulkan-headers/include'),str(src),'-o',str(out)],check=True)
            subprocess.run([str(out)],check=True)

    def test_actual_callsite_and_submission_lifecycle(self):
        dm=ROOT/'nxnative/dm'
        core=(dm/'android/gfx_vk_core.c').read_text()
        exe=(dm/'android/gfx_vk_exec.c').read_text()
        begin=function(exe,'static int begin_pass(void)')
        self.assertIn('if (pass_cut(psplit, &g_cuts)) cb = vk_cmd();',begin)
        self.assertNotIn('g_cuts++',begin)
        self.assertNotIn('g_cb_draws = 0',begin)
        self.assertIn('} else vk_cmd_split_refused(1);',exe) # interior preflight saturations counted
        self.assertIn('cut_stats_begin(s);',function(core,'void mtl_sub_begin(void)'))
        commit=function(core,'uint64_t mtl_sub_commit(void)')
        self.assertIn('cut_stats_report(s, c->frame, c->fence);',commit)
        self.assertLess(commit.index('c->frame ='),commit.index('cut_stats_report('))
        self.assertLess(commit.index('cut_stats_report('),commit.index('g_cur = -1'))

if __name__=='__main__':
    unittest.main()
