"""Queued Android submits and presents keep captured frame identity while the guest advances."""
from pathlib import Path
import re, shutil, subprocess, tempfile, unittest
ROOT=Path(__file__).resolve().parents[1]

def function(s, signature):
    start=s.index(signature);p=s.index('{',start);depth=1;i=p+1
    while depth:
        depth+=(s[i]=='{')-(s[i]=='}');i+=1
    return s[start:i]

class AsyncFrameIdentity(unittest.TestCase):
    @unittest.skipUnless(shutil.which('cc'),'C compiler required')
    def test_actual_queued_replay(self):
        s=(ROOT/'nxnative/dm/gfx/nvn/hle_nvn.c').read_text()
        declaration=re.search(r'typedef struct \{ int kind, n;.*?\} nvn_job;',s).group()
        replay=function(s,'static void *replay_main(')
        push=function(s,'static uint64_t job_push(')
        submit=function(s,'static void submit_now(NVNqueue *q, int n, const NVNcommandHandle *h, uint64_t frame) {')
        self.assertIn('mtl_set_frame(frame);',submit)
        self.assertIn('dm_perf_replay_work(frame, t0, end);',submit)
        self.assertTrue('if (!nvn_async() && !nvn_logical_queues_on()) mtl_set_frame(g_frame);' in s,
                        'both asynchronous replay modes retain their captured frame ID')
        self.assertIn('dm_perf_frame_start_id(g_frame, pace_ns, pq_ns);',s)
        code=r'''
#define _GNU_SOURCE
#define __ANDROID__ 1
#include <assert.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sched.h>
typedef int NVNqueue;
typedef int tex_t;
typedef uint64_t NVNcommandHandle;
enum {JOB_SUBMIT,JOB_PRESENT,JOB_RING=4096};
DECL
static nvn_job *g_jobs;static int g_njob,g_capjob;
static uint64_t g_replaying,g_job_enq,g_job_done,g_job_fence[JOB_RING];
static uint64_t g_tl_enq,g_tl_start,g_tl_end;
static _Atomic uint64_t guest_frame,clock_ns,g_nvn_ns;
static _Atomic int allow;
static pthread_t worker;
static pthread_mutex_t g_job_mu=PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t g_job_cv=PTHREAD_COND_INITIALIZER,g_jobdone_cv=PTHREAD_COND_INITIALIZER;
static uint64_t observed[16],backend_frame;static int observed_n;
static uint64_t dm_nvn_frame_count(void){return atomic_load(&guest_frame);}
static uint64_t now_ns(void){return atomic_fetch_add(&clock_ns,1000)+1000;}
static void dm_perf_replay_thread(void){}
static void gfx_enter(void){}
static void gfx_leave(void){}
static void mtl_set_frame(uint64_t f){backend_frame=f;}
static uint64_t mtl_last_fence(void){return (uint64_t)observed_n;}
static void submit_now(NVNqueue*q,int n,const NVNcommandHandle*h,uint64_t frame){
 (void)q;(void)n;(void)h;mtl_set_frame(frame);
 // Every command-buffer cut of this job must retain its queued frame ID.
 for(int cut=0;cut<3;cut++)assert(backend_frame==frame);
 observed[observed_n++]=frame*2;
}
static uint64_t mtl_present(const tex_t*t,double period,const int crop[4]){
 (void)t;(void)period;(void)crop;
 observed[observed_n++]=backend_frame*2+1;return (uint64_t)observed_n;
}
REPLAY
static void *launch(void*x){while(!atomic_load(&allow))sched_yield();return replay_main(x);}
void replay_start(void){assert(!pthread_create(&worker,NULL,launch,NULL));}
PUSH
int main(void){
 uint64_t ticket10=0,ticket11=0;
 atomic_store(&guest_frame,10);
 assert(job_push((nvn_job){.kind=JOB_SUBMIT})==1);
 assert(job_push((nvn_job){.kind=JOB_SUBMIT})==2);
 assert(job_push((nvn_job){.kind=JOB_PRESENT,.ticket=&ticket10})==3);
 atomic_store(&guest_frame,11);
 assert(job_push((nvn_job){.kind=JOB_SUBMIT})==4);
 assert(job_push((nvn_job){.kind=JOB_PRESENT,.ticket=&ticket11})==5);
 atomic_store(&guest_frame,999); // The guest advances before any old job executes.
 atomic_store(&allow,1);
 pthread_mutex_lock(&g_job_mu);
 while(g_job_done<5)pthread_cond_wait(&g_jobdone_cv,&g_job_mu);
 pthread_mutex_unlock(&g_job_mu);
 pthread_cancel(worker);pthread_join(worker,NULL);
 const uint64_t want[]={20,20,21,22,23};
 assert(observed_n==5&&!memcmp(observed,want,sizeof want));
 assert(ticket10==3&&ticket11==5);
 for(int i=1;i<=5;i++)assert(g_job_fence[i]==(uint64_t)i);
 free(g_jobs);
 return 0;
}
'''
        for key,value in [('DECL',declaration),('REPLAY',replay),('PUSH',push)]:code=code.replace(key,value)
        parent=ROOT/'target/tests';parent.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='async-frames-',dir=parent) as tmp:
            p=Path(tmp);(p/'main.c').write_text(code)
            subprocess.run(['cc','-std=c11','-g','-O1','-pthread','-fsanitize=address,undefined',str(p/'main.c'),'-o',str(p/'test')],check=True,capture_output=True)
            subprocess.run([str(p/'test')],check=True,timeout=10)
if __name__=='__main__':unittest.main()

