"""Execute the real Vulkan samples counter helpers with ordered query mocks."""
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


def function(source, declaration):
    start = source.index(declaration)
    brace = source.index('{', start)
    depth, pos = 1, brace + 1
    while depth:
        depth += (source[pos] == '{') - (source[pos] == '}')
        pos += 1
    return source[start:pos]


class OcclusionCounter(unittest.TestCase):
    @unittest.skipUnless(shutil.which('cc'), 'a host C compiler is required')
    def test_real_query_lifetime_and_ordered_reports(self):
        runtime = (ROOT / 'nxnative/dm/android/gfx_vk_core.c').read_text(encoding='utf-8')
        start = runtime.index('enum { OCC_BLOCK')
        helpers = runtime[start:runtime.index('void mtl_init(', start)]
        trace_field = next(line.split('//')[0].strip() for line in runtime.splitlines() if '} occ[MAIN_CBS];' in line)
        post = function(runtime, 'static void post(')
        report = function(runtime, 'void mtl_sub_counter(')
        start = runtime.index('        for (int i = 0; i < c->n; i++) {')
        completion = runtime[start:runtime.index('        completion_events_publish(c);', start)]
        wait = function(runtime, 'static void slot_recycle(')
        wait = wait[wait.index('{') + 1:wait.index('    pthread_mutex_lock(&g_df_mu);')]
        code = r'''
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>
#include <string.h>
#include <assert.h>
#include <setjmp.h>
typedef uintptr_t VkQueryPool;
typedef uintptr_t VkCommandBuffer;
typedef int VkResult;
typedef struct { int sType; const void *pNext; unsigned flags; int queryType; unsigned queryCount; } VkQueryPoolCreateInfo;
enum { VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO=1,VK_QUERY_TYPE_OCCLUSION=0,
       VK_QUERY_CONTROL_PRECISE_BIT=1,VK_QUERY_RESULT_64_BIT=1,VK_QUERY_RESULT_WITH_AVAILABILITY_BIT=2,VK_SUCCESS=0,
       VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT=0,VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT=1,
       NVN_COUNTER_TYPE_TIMESTAMP=0,NVN_COUNTER_TYPE_TIMESTAMP_TOP=0x10,NVN_COUNTER_TYPE_SAMPLES_PASSED=1,CQ=256 };
#define DM_GAME_RENDER_COND 0
#define MAIN_CBS 32
typedef struct { int kind; volatile uint32_t *val; uint32_t mode,v; uint64_t addr; uint32_t type; int q; } post_op;
typedef struct { VkQueryPool *oq; int noq,capoq,nocc,nmain_used,ncq; VkCommandBuffer main,mains[MAIN_CBS]; int nmain_sent,nup_used; TRACE_FIELD post_op *post; int npost,cappost; uint64_t fence; } vksub_t;
static vksub_t g_slot[2];
static int g_cur, g_prereset, in_pass, begin_calls, end_calls, reset_calls, get_calls, last_flags, creates;
static int cpu_waits,gpu_waits,complete_on_wait,expect_die;
static jmp_buf death;
static VkQueryPool g_cq;
static void *g_lib_base;
static int diagnostic_reads,diagnostic_releases;
static struct { int dev,has_occ_precise; struct { struct { double timestampPeriod; } limits; } props; } V;
typedef struct { uint64_t values[256]; unsigned char available[256],reset[256]; } mock_pool;
static mock_pool pools[32];
typedef struct { VkCommandBuffer cb; int kind; VkQueryPool pool; uint32_t q; uint64_t value; } queued_query;
static queued_query queued[4096];static int nqueued,defer_queries;
static void enqueue_query(VkCommandBuffer cb,int kind,VkQueryPool pool,uint32_t q,uint64_t value){
 assert(nqueued<4096);queued[nqueued++]=(queued_query){cb,kind,pool,q,value};
}
static void execute_query_cb(VkCommandBuffer cb){
 // Queue execution, not recording, makes queries available. Upload reset executes first.
 for(int i=0;i<nqueued;i++){queued_query *x=&queued[i];if(x->cb!=cb)continue;
  mock_pool *p=&pools[x->pool];assert(x->pool&&x->q<256);
  if(x->kind==0){memset(p->reset,1,sizeof p->reset);memset(p->available,0,sizeof p->available);}
  else if(x->kind==1){assert(p->reset[x->q]&&!p->available[x->q]);p->reset[x->q]=0;p->values[x->q]=0;}
  else if(x->kind==2){assert(!p->reset[x->q]);p->available[x->q]=1;}
  else {assert(!p->reset[x->q]&&!p->available[x->q]);p->values[x->q]+=x->value;}
 }
}
static int failure_lines;static char failure_log[4096];
static void log_ignore(const char *fmt,...){
 if(!strstr(fmt,"query failure")&&!strstr(fmt,"query part")&&!strstr(fmt,"query draw")&&!strstr(fmt,"query availability"))return;
 failure_lines++;va_list ap;va_start(ap,fmt);size_t at=strlen(failure_log);
 vsnprintf(failure_log+at,sizeof failure_log-at,fmt,ap);va_end(ap);
}
#define VKLOG(...) log_ignore(__VA_ARGS__)
static void die(const char *fmt,...){(void)fmt;if(expect_die)longjmp(death,1);abort();}
#define VKCHK(expr) do { if((expr)!=VK_SUCCESS)die("Vulkan failure"); } while(0)
static vksub_t *cur(void){assert(g_cur>=0&&g_cur<2);return &g_slot[g_cur];}
static VkCommandBuffer vk_upload_cmd(void){return 99;}
static VkCommandBuffer vk_cmd(void){return cur()->main;}
static int vkCreateQueryPool(int dev,const VkQueryPoolCreateInfo *ci,const void *a,VkQueryPool *out){
 (void)dev;(void)a;assert(ci->queryType==VK_QUERY_TYPE_OCCLUSION&&ci->queryCount==256);
 assert(creates<31);*out=(VkQueryPool)++creates;return VK_SUCCESS;
}
static void vkCmdResetQueryPool(VkCommandBuffer cb,VkQueryPool pool,uint32_t first,uint32_t count){
 assert(cb==99&&pool&&first==0&&count==256);reset_calls++;
 if(defer_queries){enqueue_query(cb,0,pool,0,0);return;}
 memset(pools[pool].reset,1,sizeof pools[pool].reset);
 memset(pools[pool].available,0,sizeof pools[pool].available);
}
static void vkCmdBeginQuery(VkCommandBuffer cb,VkQueryPool pool,uint32_t q,unsigned flags){
 if(defer_queries){assert(in_pass&&cb==cur()->main);enqueue_query(cb,1,pool,q,0);begin_calls++;last_flags=(int)flags;return;}
 assert(in_pass&&cb==cur()->main&&q<256&&pools[pool].reset[q]&&!pools[pool].available[q]);
 pools[pool].reset[q]=0;pools[pool].values[q]=0;begin_calls++;last_flags=(int)flags;
}
static void vkCmdEndQuery(VkCommandBuffer cb,VkQueryPool pool,uint32_t q){
 if(defer_queries){assert(in_pass&&cb==cur()->main);enqueue_query(cb,2,pool,q,0);end_calls++;return;}
 assert(in_pass&&cb==cur()->main&&q<256&&!pools[pool].reset[q]);
 pools[pool].available[q]=1;end_calls++;
}
static VkResult vkGetQueryPoolResults(int dev,VkQueryPool pool,uint32_t q,uint32_t n,size_t len,void *out,size_t stride,unsigned flags){
 (void)dev;assert(q<256&&n==1);get_calls++;
 if(flags==(VK_QUERY_RESULT_64_BIT|VK_QUERY_RESULT_WITH_AVAILABILITY_BIT)){
  assert(len==16&&stride==16);diagnostic_reads++;
  if(diagnostic_releases)pools[pool].available[q]=1;
  ((uint64_t*)out)[1]=pools[pool].available[q];
 }else assert(len==8&&stride==8&&flags==VK_QUERY_RESULT_64_BIT);
 if(!pools[pool].available[q])return 1;
 memcpy(out,&pools[pool].values[q],8);return VK_SUCCESS;
}
static void vkCmdWriteTimestamp(VkCommandBuffer cb,int stage,VkQueryPool pool,uint32_t q){(void)cb;(void)stage;(void)pool;(void)q;abort();}
static uint64_t nvn_timestamp_from_ns(uint64_t ns){return ns;}
static uint64_t guest_ns_at_host_time(uint64_t ns){return ns;}
static int gpu_clock_host_ns(const int *pair,uint64_t query,uint64_t *host){(void)pair;(void)query;(void)host;return 0;}
POST
HELPERS
REPORT
static void complete(vksub_t *s,int submitted_parts){
 struct { int n,rslot,occ_parts; post_op *ops; int slot; uint64_t frame,fence; } pending={s->npost,(int)(s-g_slot),submitted_parts,s->post,0,123,456},*c=&pending;
 uint64_t now=12345,tend=0;int clock=0;
COMPLETION
 s->npost=0;
}
static int mtl_wait(uint64_t fence,uint64_t timeout){(void)fence;(void)timeout;cpu_waits++;if(complete_on_wait)complete(cur(),cur()->nmain_used);return 1;}
static void wait_fence(uint64_t fence){(void)fence;gpu_waits++;}
static void recycle_wait(vksub_t *s){
WAIT
}
static void open_pass(void){assert(!in_pass);in_pass=1;}
static void close_pass(void){vk_occ_end();in_pass=0;}
static void draw(uint64_t samples){
 vk_occ_draw();if(g_occ_active>=0){int q=g_occ_active;
 if(defer_queries)enqueue_query(cur()->main,3,cur()->oq[q/256],(uint32_t)(q%256),samples);
 else pools[cur()->oq[q/256]].values[q%256]+=samples;}
}
static void new_submission(int slot){
 assert(!in_pass&&g_occ_active<0);g_cur=slot;vksub_t *s=cur();
 s->nocc=0;s->npost=0;s->nmain_used=1;s->main=s->mains[0]=(VkCommandBuffer)(slot+1);
 s->nmain_sent=0;s->nup_used=1;memset(s->occ,0,sizeof s->occ);
}
static void report_to(uint64_t out[2]){mtl_sub_counter((uint64_t)(uintptr_t)out,NVN_COUNTER_TYPE_SAMPLES_PASSED);}
static void clear_state(void){
 assert(!in_pass);for(int i=0;i<2;i++){free(g_slot[i].oq);free(g_slot[i].post);}memset(g_slot,0,sizeof g_slot);
 memset(pools,0,sizeof pools);creates=begin_calls=end_calls=reset_calls=get_calls=0;nqueued=defer_queries=0;
 failure_lines=0;failure_log[0]=0;diagnostic_reads=diagnostic_releases=0;g_occ_counting=0;g_occ_active=-1;g_occ_completed=0;V.has_occ_precise=1;new_submission(0);
}
int main(void){
 uint64_t a[2]={0},b[2]={0},c[2]={0},d[2]={0};
 clear_state();
 // No eager query for unrelated drawing before the first counter reset.
 open_pass();draw(99);close_pass();assert(!begin_calls&&!creates);
 vk_occ_reset();report_to(a);complete(cur(),1);assert(a[0]==0&&a[1]==12345&&!creates);
 // Reports snapshot cumulative values without resetting. Multiple guest draws share one segment.
 open_pass();draw(3);draw(4);assert(begin_calls==1&&last_flags==VK_QUERY_CONTROL_PRECISE_BIT);
 report_to(a);report_to(b);assert(end_calls==1);
 assert(cur()->post[0].kind==3&&cur()->post[0].type==2&&cur()->post[0].addr&&(uintptr_t)cur()->post[0].val);
 draw(5);report_to(c);close_pass();complete(cur(),1);
 assert(a[0]==7&&b[0]==7&&c[0]==12&&get_calls==2);
 // Reset inside a pass closes the old segment, then resets in exact CPU-completion order.
 open_pass();draw(8);vk_occ_reset();draw(11);report_to(a);vk_occ_reset();report_to(b);close_pass();complete(cur(),1);
 assert(a[0]==11&&b[0]==0);
 // A pass boundary closes a segment, not the cumulative counter.
 open_pass();draw(13);close_pass();open_pass();draw(17);report_to(a);close_pass();complete(cur(),1);assert(a[0]==30);
 // Ordered completion carries the cumulative counter across different submission slots.
 new_submission(1);open_pass();draw(19);report_to(a);close_pass();complete(cur(),1);assert(a[0]==49);
 // The clear helper ends the query; its own implementation draw contributes no samples.
 open_pass();draw(2);vk_occ_end();assert(g_occ_active<0);draw(3);report_to(a);close_pass();complete(cur(),1);assert(a[0]==54);
 // 64-bit counts must not truncate.
 vk_occ_reset();open_pass();draw(1ull<<40);report_to(a);close_pass();complete(cur(),1);assert(a[0]==1ull<<40);
 // Unsupported precise feature never passes an unsupported Vulkan flag (documented approximate fallback).
 V.has_occ_precise=0;vk_occ_reset();open_pass();draw(1);assert(last_flags==0);report_to(a);close_pass();complete(cur(),1);assert(a[0]==1);
 V.has_occ_precise=1;
 // Reset-only/empty submissions and repeated resets require no query pool or dummy count.
 clear_state();vk_occ_reset();vk_occ_reset();report_to(a);complete(cur(),1);assert(a[0]==0&&!creates&&!get_calls);
 // Grow across block boundary: each pool is reset once, every segment remains distinct.
 open_pass();for(int i=0;i<257;i++){draw(1);vk_occ_end();}report_to(a);close_pass();
 assert(creates==2&&reset_calls==2&&cur()->nocc==257);complete(cur(),1);assert(a[0]==257&&get_calls==257);
 // Even with pre-reset disabled, reuse waits for CPU query consumption before overwriting old results.
 new_submission(1);open_pass();draw(7);report_to(b);close_pass();
 g_prereset=0;complete_on_wait=1;recycle_wait(cur());assert(cpu_waits==1&&!gpu_waits&&b[0]==264);
 complete_on_wait=0;new_submission(1);open_pass();draw(5);report_to(c);close_pass();complete(cur(),1);assert(c[0]==269);
 assert(creates==3); // Pool reused rather than recreated.
 cur()->nocc=0;recycle_wait(cur());assert(gpu_waits==1);
 g_prereset=1;recycle_wait(cur());assert(cpu_waits==2);
 // Command-buffer failure: only submitted parts affect accumulated state; dropped reset cannot erase prior counts.
 clear_state();vk_occ_reset();open_pass();draw(23);close_pass();cur()->nmain_used=2;
 vk_occ_reset();open_pass();draw(100);close_pass();report_to(a);complete(cur(),1);assert(a[0]==23&&get_calls==1);
 // Dropping the entire submission does not consume stale unavailable queries.
 clear_state();vk_occ_reset();open_pass();draw(100);close_pass();report_to(a);complete(cur(),0);assert(a[0]==0&&!get_calls);
 // Delayed Vulkan execution: 257 segments recorded across two native parts.
 // The two block resets are recorded after main-CB recording has started, but execute
 // on the upload CB before either main. Results must remain unavailable until End executes.
 clear_state();defer_queries=1;vk_occ_reset();open_pass();
 for(int i=0;i<256;i++){draw((uint64_t)i+1);vk_occ_end();}close_pass();
 cur()->nmain_used=2;cur()->main=cur()->mains[1]=3;
 open_pass();draw(1000);report_to(a);close_pass();
 assert(cur()->nocc==257&&reset_calls==2&&!pools[cur()->oq[0]].available[35]);
 assert(cur()->occ[0].begins==256&&cur()->occ[0].ends==256&&cur()->occ[1].begins==1);
 assert(cur()->occ[0].first_block==0&&cur()->occ[1].first_block==1);
 execute_query_cb(99);execute_query_cb(1);
 assert(pools[cur()->oq[0]].available[35]&&!pools[cur()->oq[1]].available[0]);
 execute_query_cb(3);complete(cur(),2);
 assert(a[0]==32896+1000&&get_calls==257&&!failure_lines);
 // If the final part is dropped, its recorded End must not be read and cannot reset
 // the accumulated counts of the submitted prefix.
 clear_state();defer_queries=1;vk_occ_reset();open_pass();draw(17);close_pass();
 cur()->nmain_used=2;cur()->main=cur()->mains[1]=3;
 vk_occ_reset();open_pass();draw(99);report_to(a);close_pass();
 execute_query_cb(99);execute_query_cb(1);complete(cur(),1);
 assert(a[0]==17&&get_calls==1&&!pools[cur()->oq[0]].available[1]&&!failure_lines);
 // An unavailable result is no longer fatal (8569eeb7b "non-fatal occlusion readback": a few queries per submission part never
 // become available on the Odin 3 in Ocarina of Time): completion logs the full diagnostics for the first failures and counts the
 // query as visible (1 sample), so a render condition reading it draws. It never publishes a diagnostic read as the guest's count.
 clear_state();vk_occ_reset();open_pass();draw(9);close_pass();
 pools[cur()->oq[0]].available[0]=0;expect_die=1;complete(cur(),1);expect_die=0;
 assert(failure_lines==4&&strstr(failure_log,"frame 123 fence 456 slot 0"));
 assert(strstr(failure_log,"queries 1 begins 1 ends 1 block resets 1"));
 assert(strstr(failure_log,"begin/reset/end parts 0/0/0"));
 assert(g_occ_completed==1&&diagnostic_reads==1); // counted as visible; unavailable diagnostics never become guest results
 assert(strstr(failure_log,"query draw calls 1")&&strstr(failure_log,"available 0 (failed query)"));
 // Even a second read that succeeds is evidence only: the failed query still counts as 1, not the 9 samples the diagnostic read saw.
 // (completion consumed the posts, so the same submission is recorded again)
 clear_state();vk_occ_reset();open_pass();draw(9);close_pass();
 pools[cur()->oq[0]].available[0]=0;diagnostic_releases=1;expect_die=1;complete(cur(),1);expect_die=0;
 assert(diagnostic_reads==1&&pools[cur()->oq[0]].available[0]);
 assert(g_occ_completed==1&&strstr(failure_log,"available 1 (failed query)"));
 diagnostic_releases=0;
 // A query ending in the wrong native part is identified independently of reset ownership.
 failure_lines=0;failure_log[0]=0;cur()->nmain_used=2;cur()->mains[1]=3;
 occ_failure(cur(),&(post_op){.kind=3,.v=1,.q=0},2,9,10,1);
 assert(failure_lines==5&&strstr(failure_log,"begin/reset/end parts 0/0/1"));
 // Diagnostic neighbors in a dropped part may never have been reset; do not read them.
 clear_state();defer_queries=1;vk_occ_reset();open_pass();
 for(int i=0;i<256;i++){draw(1);vk_occ_end();}close_pass();
 cur()->nmain_used=2;cur()->main=cur()->mains[1]=3;open_pass();draw(1);close_pass();
 execute_query_cb(99);execute_query_cb(1);
 diagnostic_reads=0;occ_failure(cur(),&cur()->post[256],1,9,10,1);
 assert(diagnostic_reads==3); // only253,254,255; never unsubmitted256
 clear_state();(void)d;
 for(int i=0;i<2;i++){free(g_slot[i].oq);free(g_slot[i].post);}
 return 0;
}
'''
        for marker, value in [('TRACE_FIELD', trace_field), ('POST', post), ('HELPERS', helpers), ('REPORT', report),
                              ('COMPLETION', completion), ('WAIT', wait)]:
            code = code.replace(marker, value)
        target = ROOT / 'target/tests'
        target.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='vk-occlusion-', dir=target) as folder:
            src, exe = Path(folder) / 'check.c', Path(folder) / 'check'
            src.write_text(code, encoding='utf-8')
            subprocess.run(['cc', '-std=gnu11', '-Wall', '-Wextra', '-Werror',
                            '-Wno-missing-field-initializers', '-fsanitize=address,undefined',
                            str(src), '-o', str(exe)], check=True)
            subprocess.run([str(exe)], check=True, timeout=15)

    @unittest.skipUnless(shutil.which('cc'), 'a host C compiler is required')
    def test_deferred_query_commands_preserve_draw_order(self):
        runtime = (ROOT / 'nxnative/dm/android/gfx_vk_dq.c').read_text(encoding='utf-8')
        commands = runtime[runtime.index('#define DQ_CMDS'):runtime.index('#define DQ_SYNC')]
        for name in ['vkCmdBeginQuery', 'vkCmdEndQuery']:
            self.assertIn(f'X({name})', commands)
        names = ['vkCmdBeginQuery', 'vkCmdDraw', 'vkCmdEndQuery', 'vkCmdResetQueryPool']
        wrappers = '\n'.join(function(runtime, f'static VKAPI_ATTR void VKAPI_CALL w_{name}(') for name in names)
        macros = '\n'.join(line for line in runtime.splitlines()
                           if line.startswith(('#define DIRECT(', '#define BEGIN(', '#define W_NOARG(')))
        wrappers += '\nW_NOARG(vkCmdEndRenderPass)\n'
        cases = '\n'.join(re.search(r'    case OP_' + name + r':[^\n]+', runtime).group()
                          for name in names + ['vkCmdEndRenderPass'])
        structs = '\n'.join(re.search(r'typedef struct \{[^\n]+\} ' + name + ';', runtime).group()
                            for name in ['a_query', 'a_draw', 'a_qreset', 'rec_hdr'])
        code = r'''
#include <stdint.h>
#include <stddef.h>
#include <assert.h>
#include <string.h>
typedef uintptr_t VkCommandBuffer;
typedef uintptr_t VkQueryPool;
typedef unsigned VkQueryControlFlags;
#define VKAPI_ATTR
#define VKAPI_CALL
enum { OP_vkCmdBeginQuery, OP_vkCmdDraw, OP_vkCmdEndQuery, OP_vkCmdResetQueryPool, OP_vkCmdEndRenderPass };
STRUCTS
static union { max_align_t align; unsigned char bytes[2048]; } queue;
static size_t written,read_pos;
static int fail_next,g_dq_direct,called,active;
static char order[32];
static void r_vkCmdResetQueryPool(VkCommandBuffer cb,VkQueryPool p,uint32_t q,uint32_t n){assert(cb==9&&p==71&&q==0&&n==256);order[called++]='R';}
static void r_vkCmdBeginQuery(VkCommandBuffer cb,VkQueryPool p,uint32_t q,VkQueryControlFlags flags){assert(cb==5&&p==71&&q==17&&flags==1&&!active);active=1;order[called++]='B';}
static void r_vkCmdDraw(VkCommandBuffer cb,uint32_t n,uint32_t i,uint32_t f,uint32_t fi){assert(cb==5&&n==123&&i==2&&f==3&&fi==4&&active);order[called++]='D';}
static void r_vkCmdEndQuery(VkCommandBuffer cb,VkQueryPool p,uint32_t q){assert(cb==5&&p==71&&q==17&&active);active=0;order[called++]='E';}
static void r_vkCmdEndRenderPass(VkCommandBuffer cb){assert(cb==5&&!active);order[called++]='P';}
static void run(const rec_hdr *h){const uint8_t *p=(const uint8_t *)(h+1);VkCommandBuffer cb=h->cb;switch(h->op){
CASES
default: assert(0);
}}
static void vk_dq_drain(void){while(read_pos<written){const rec_hdr *h=(const void *)(queue.bytes+read_pos);run(h);read_pos+=h->size;}}
static void *rec_begin(uint32_t op,VkCommandBuffer cb,size_t n){
 if(fail_next){fail_next=0;return NULL;}
 size_t size=(sizeof(rec_hdr)+n+15)&~(size_t)15;assert(written+size<=sizeof queue.bytes);
 rec_hdr *h=(void *)(queue.bytes+written);*h=(rec_hdr){op,(uint32_t)size,cb};return h+1;
}
static void rec_end(void *payload){rec_hdr *h=(rec_hdr *)payload-1;written+=h->size;}
MACROS
WRAPPERS
static void clear(void){assert(!active);written=read_pos=0;called=g_dq_direct=0;memset(order,0,sizeof order);}
int main(void){
 clear();w_vkCmdResetQueryPool(9,71,0,256);w_vkCmdBeginQuery(5,71,17,1);w_vkCmdDraw(5,123,2,3,4);w_vkCmdEndQuery(5,71,17);w_vkCmdEndRenderPass(5);
 assert(called==0);vk_dq_drain();assert(!strcmp(order,"RBDEP")&&g_dq_direct==0);
 // Failed allocation drains prior commands before the direct BeginQuery.
 clear();w_vkCmdResetQueryPool(9,71,0,256);fail_next=1;w_vkCmdBeginQuery(5,71,17,1);assert(!strcmp(order,"RB"));
 w_vkCmdDraw(5,123,2,3,4);w_vkCmdEndQuery(5,71,17);w_vkCmdEndRenderPass(5);vk_dq_drain();assert(!strcmp(order,"RBDEP")&&g_dq_direct==1);
 // The direct EndQuery cannot overtake a queued draw.
 clear();w_vkCmdBeginQuery(5,71,17,1);w_vkCmdDraw(5,123,2,3,4);fail_next=1;w_vkCmdEndQuery(5,71,17);
 assert(!strcmp(order,"BDE"));w_vkCmdEndRenderPass(5);vk_dq_drain();assert(!strcmp(order,"BDEP")&&g_dq_direct==1);
 return 0;
}
'''
        for marker, value in [('STRUCTS', structs), ('CASES', cases), ('MACROS', macros), ('WRAPPERS', wrappers)]:
            code = code.replace(marker, value)
        target = ROOT / 'target/tests'
        target.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='vk-occlusion-dq-', dir=target) as folder:
            src, exe = Path(folder) / 'check.c', Path(folder) / 'check'
            src.write_text(code, encoding='utf-8')
            subprocess.run(['cc', '-std=gnu11', '-Wall', '-Wextra', '-Werror', '-fsanitize=address,undefined',
                            str(src), '-o', str(exe)], check=True)
            subprocess.run([str(exe)], check=True, timeout=15)

    def test_exec_hooks_exclude_internal_clear(self):
        runtime = (ROOT / 'nxnative/dm/android/gfx_vk_exec.c').read_text(encoding='utf-8')
        end = function(runtime, 'void vk_end_pass(')
        self.assertLess(end.index('vk_occ_end();'), end.index('vkCmdEndRenderPass('))
        self.assertIn('if (t == NVN_COUNTER_TYPE_SAMPLES_PASSED) vk_occ_reset();', runtime)
        self.assertRegex(runtime, r'vk_occ_end\(\);\s*vkCmdDraw\(cb, 3, 1, 0, 0\);')
        # the indirect draws choose between the count and the plain indirect command after the one vk_occ_draw
        # (170571070, #151: indirect-count multi-draws honour the GPU's count): `vk_occ_draw(); if (mcb) vkCmdDraw...IndirectCount(`
        guest_calls = re.findall(r'vk_occ_draw\(\);\s*(?:if \(mcb\) )?vkCmdDraw\w*\(', runtime)
        self.assertEqual(len(guest_calls), 8)  # includes the raster draw of mesh emulation
        self.assertEqual(len(re.findall(r'vk_occ_draw\(\);', runtime)), 8)  # and no other guest draw counts


if __name__ == '__main__':
    unittest.main()
