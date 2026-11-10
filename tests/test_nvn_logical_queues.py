"""Execute the real logical queue scheduler with ordered backend/fence mocks."""
from pathlib import Path
import os
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]

C = r"""
#define _GNU_SOURCE
#define __ANDROID__ 1
#include <assert.h>
#include <pthread.h>
#include <stdint.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include "nvn_obj.h"
#include "nvn_queue.h"
#include "nvn_capture.h"
static pthread_mutex_t g_sub_mu=PTHREAD_MUTEX_INITIALIZER;
static _Atomic uint64_t g_nq_enqueued;
static uint64_t g_replaying;
static _Atomic uint64_t g_nvn_ns,g_sub_wall_ns,g_cum_sub_wall_ns,g_sub_cpu_ns,g_cum_sub_cpu_ns,g_sub_instr,g_cum_sub_instr,g_cum_sub_cyc,g_cum_sub_sys_ns;
static int dm_host_work, enabled, auto_complete=1;
static _Atomic uint64_t completed,last_fence,frame_now;
static pthread_mutex_t mock_mu=PTHREAD_MUTEX_INITIALIZER;
static int markers[128], nmarkers, tex_pins, prog_pins, registry_pins;
static uint64_t replay_ids[128], seen_prog, seen_tex, present_tex;
static int nids;
static uint8_t active_ids[128];
static uint64_t now_ns(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return (uint64_t)t.tv_sec*1000000000+t.tv_nsec; }
static uint64_t clock_gettime_nsec_np(clockid_t c) { struct timespec t; clock_gettime(c,&t); return (uint64_t)t.tv_sec*1000000000+t.tv_nsec; }
static int relative_wait(pthread_cond_t *cv,pthread_mutex_t *mu,const struct timespec *r) {
 struct timespec t; clock_gettime(CLOCK_REALTIME,&t); t.tv_sec+=r->tv_sec;t.tv_nsec+=r->tv_nsec;if(t.tv_nsec>=1000000000){t.tv_sec++;t.tv_nsec-=1000000000;}
 return pthread_cond_timedwait(cv,mu,&t);
}
#define pthread_cond_timedwait_relative_np relative_wait
#define pthread_setname_np(name) ((void)(name))
#define LOG(...) ((void)0)
#define die(...) do { fprintf(stderr,__VA_ARGS__); abort(); } while(0)
static uint64_t thr_sys_ns(void) { return 0; }
static void thr_counts(uint64_t *a,uint64_t *b) { *a=*b=0; }
int nvn_logical_queues_on(void) { return enabled; }
uint64_t dm_nvn_frame_count(void) { return atomic_load(&frame_now); }
static void dm_perf_replay_thread(void) {}
static void dm_perf_replay_work(uint64_t f,uint64_t a,uint64_t b) { assert(f==3 && b>=a); }
static void dm_host_mutex_lock(pthread_mutex_t *m) { pthread_mutex_lock(m); }
static void gfx_enter(void) {}
static void gfx_leave(void) {}
static void mtl_prepare(void) {}
void mtl_set_frame(uint64_t f) { assert(f==3); }
static void mtl_queue_activate(uint64_t s) { assert(s>=1 && s<=4); }
static void mtl_set_queue_origin_lower_left(int x) { (void)x; }
void mtl_prog_snapshot(prog_t *out,const prog_t *p) { *out=*p; prog_pins++; }
void mtl_prog_unpin(prog_t *p) { assert(p->serial); assert(prog_pins>0);prog_pins--; }
void mtl_tex_pin(uint64_t s) { assert(s);tex_pins++; }
void mtl_tex_unpin(uint64_t s) { assert(s && tex_pins>0);tex_pins--; }
void mtl_tex_snapshot(tex_t *out,const tex_t *p) { *out=*p;mtl_tex_pin(p->serial); }
void vk_registry_job_pin(uint64_t id) { assert(id<128 && !active_ids[id]);active_ids[id]=1;registry_pins++; }
void vk_registry_job_unpin(uint64_t id) { assert(id<128 && active_ids[id]);active_ids[id]=0;registry_pins--; }
static void nvn_ft_poll(void) {}
void nvn_cap_queue(uint64_t s,int w) { (void)s;(void)w; }
void nvn_cap_submit(int n,const uint64_t *h) { assert(n==1 && *h); }
void nvn_cap_present(const tex_t *t,const int c[4]) { (void)t;(void)c; }
static void mtl_sub_begin(void) {}
void mtl_sub_index(const uint64_t *h,int n) { (void)h;(void)n; }
void mtl_exec_rest(const uint64_t *h,int n) { assert(!h&&!n); }
static volatile uint32_t *pending_word;
static uint32_t pending_mode,pending_value;
void mtl_exec(const uint8_t *p) {
 pthread_mutex_lock(&mock_mu);
 assert(g_replaying && active_ids[g_replaying]);replay_ids[nids++]=g_replaying;
 for(;;) {
  op_hdr h;memcpy(&h,p,8);const uint8_t *a=p+8;if(h.op==OP_END)break;
  if(h.op==OP_CLEAR_COLOR) { const p_clear_color *c=(const void*)a;markers[nmarkers++]=(int)c->rgba[0]; }
  else if(h.op==OP_BIND_PROGRAM) seen_prog=((const p_bind_program*)a)->prog->serial;
  else if(h.op==OP_SET_RT) seen_tex=((const p_set_rt*)a)->color[0]->serial;
  else if(h.op==OP_SIGNAL_EVENT) { memcpy(&pending_word,a,8);memcpy(&pending_mode,a+8,4);memcpy(&pending_value,a+12,4); }
  else assert(0);
  p+=h.size;
 }
 pthread_mutex_unlock(&mock_mu);
}
static uint64_t mtl_sub_commit(void) {
 uint64_t f=atomic_fetch_add(&last_fence,1)+1;
 if(auto_complete) { if(pending_word) nvn_event_publish_gpu(pending_word,pending_mode,pending_value,f);atomic_store(&completed,f);nvn_queue_gpu_progress(); }
 pending_word=NULL;return f;
}
static uint64_t mtl_completed_fence(void) { return atomic_load(&completed); }
static int mtl_wait(uint64_t f,uint64_t timeout) {
 uint64_t start=now_ns();
 while(atomic_load(&completed)<f) { if(!timeout || now_ns()-start>=timeout) return 0;usleep(100); }
 return 1;
}
uint64_t mtl_present(const tex_t *t,double period,const int crop[4]) { (void)period;(void)crop;present_tex=t->serial;return 77; }
void mtl_present_wait(uint64_t t) { assert(t==77); }
// hle_nvn.c expands a compact record (nxnative f02d0474c); this scheduler test records canonical ones only, so an expansion would be a harness error
static size_t rec_expand(const uint8_t *rec,uint8_t *out) { (void)rec;(void)out; fprintf(stderr,"unexpected compact record\n"); abort(); }
#include "nvn_queue.inc"

HOST_SIGNAL

typedef struct { _Alignas(8) uint8_t bytes[2048];size_t n; } commands;
static void emit(commands *c,unsigned op,const void *data,size_t n) {
 size_t z=(n+15)&~(size_t)7;assert(c->n+z<sizeof c->bytes);op_hdr h={op,(uint32_t)z};memcpy(c->bytes+c->n,&h,8);
 if(n)memcpy(c->bytes+c->n+8,data,n);c->n+=z;
}
static void mark(commands *c,int x) { p_clear_color v={.rgba={(float)x,0,0,1}};emit(c,OP_CLEAR_COLOR,&v,sizeof v); }
static void event(commands *c,unsigned op,volatile uint32_t *p,unsigned v) { struct { volatile uint32_t *p;unsigned mode,value; } a={p,0,v};emit(c,op,&a,sizeof a); }
// A handle is the recording's control list (nvn_obj.h ctl_e, nxnative f02d0474c): here one piece, its records with no END (the list's CTL_LAST ends it)
static void submit(queue_t *q,commands *c,sync_t *s) { assert(c->n);uint64_t list=ctl_e(CTL_CHUNK,c->bytes,c->n)|CTL_LAST,h=(uintptr_t)&list;nvn_queue_submit(q,1,&h);if(s)nvn_queue_fence(q,s); }
static void finish(sync_t *s) { assert(nvn_sync_wait(s,2000000000ull));nvn_sync_release(s); }
static void wait_published(sync_t *s) {
 uint64_t start=now_ns();for(;;) { pthread_mutex_lock(&nq_mu);int p=s->fenced==3&&((nq_receipt*)(uintptr_t)s->fence)->published;pthread_mutex_unlock(&nq_mu);
 if(p)return;assert(now_ns()-start<2000000000ull);usleep(100); }
}
static void retired(void) {
 // Publication can precede releasing job-owned metadata; acquiring g_sub_mu then
 // checking under nq_mu gives the worker opportunity to complete that retirement.
 for(int i=0;i<20000;i++) {
  pthread_mutex_lock(&g_sub_mu);int left=registry_pins;pthread_mutex_unlock(&g_sub_mu);if(!left)return;usleep(100);
 }assert(0);
}
int main(void) {
 atomic_store(&frame_now,3);
 // Disabled hooks preserve physical ticket representation and release-ordered stores.
 uint32_t word=0;nvn_event_publish(&word,0,4);nvn_event_publish_gpu(&word,1,2,99);assert(word==6);
 sync_t physical={.fenced=2,.fence=45};nvn_sync_release(&physical);assert(physical.fenced==2&&physical.fence==45);
 nvn_queue_present_release(123);nvn_queue_gpu_progress();
 enabled=1;
 queue_t a={.serial=1},b={.serial=2},c={.serial=3},d={.serial=4};
 // A consumer submitted first parks; unrelated ready work and its later producer
 // progress. Original descriptors and nested CALL records are immutable copies.
 nvn_queue_test_hold(1);word=0;commands first={0},nested={0},producer={0};
 prog_t prog={.serial=81,.nstage=1};tex_t tex={.serial=91};
 p_bind_program bp={.prog=&prog};p_set_rt rt={.n=1};rt.color[0]=&tex;
 emit(&nested,OP_BIND_PROGRAM,&bp,sizeof bp);emit(&nested,OP_SET_RT,&rt,sizeof rt);mark(&nested,10);
 event(&first,OP_WAIT_EVENT,&word,1);memcpy(first.bytes+first.n,nested.bytes,nested.n);first.n+=nested.n;   // a call is copied in when it is recorded (f02d0474c), so the nested records are the recording's own
 sync_t sa={0},sb={0};submit(&a,&first,&sa);prog.serial=82;tex.serial=92;memset(nested.bytes,0xcc,sizeof nested.bytes);memset(first.bytes,0xcc,sizeof first.bytes);   // (the submitted recording itself: the queue must have copied it)
 mark(&producer,20);event(&producer,OP_SIGNAL_EVENT,&word,1);submit(&b,&producer,&sb);
 nvn_queue_test_hold(0);finish(&sa);finish(&sb);retired();
 assert(nmarkers==2&&markers[0]==20&&markers[1]==10);assert(seen_prog==81&&seen_tex==91);
 assert(nids==2&&replay_ids[0]>replay_ids[1]);assert(!tex_pins&&!prog_pins);
 // A queued exact event write may satisfy a wait without a CPU completion stall.
 // A wait-only tail still owns its producer fence and cannot signal prematurely.
 auto_complete=0;word=0;commands p={0};event(&p,OP_SIGNAL_EVENT,&word,7);sync_t sp={0};submit(&c,&p,&sp);wait_published(&sp);
 commands onlywait={0};event(&onlywait,OP_WAIT_EVENT,&word,7);sync_t sw={0};submit(&d,&onlywait,&sw);wait_published(&sw);
 assert(!nvn_sync_wait(&sw,0));
 pthread_mutex_lock(&nq_mu);assert(((nq_receipt*)(uintptr_t)sw.fence)->fence==((nq_receipt*)(uintptr_t)sp.fence)->fence);pthread_mutex_unlock(&nq_mu);
 atomic_store(&completed,atomic_load(&last_fence));nvn_event_publish_gpu(&word,0,7,atomic_load(&completed));nvn_queue_gpu_progress();finish(&sp);finish(&sw);retired();auto_complete=1;
 // Queue wait snapshots a receipt: re-fencing/finalizing that sync cannot release
 // a continuation still depending on the originally parked queue.
 word=0;commands parked={0};event(&parked,OP_WAIT_EVENT,&word,8);mark(&parked,30);sync_t old={0},waiter={0};submit(&a,&parked,&old);
 nvn_queue_wait_sync(&b,&old);commands after={0};mark(&after,40);submit(&b,&after,&waiter);
 nvn_queue_fence(&c,&old);nvn_sync_release(&old);assert(!nvn_sync_wait(&waiter,0));
 nvn_event_publish(&word,0,8);finish(&waiter);retired();assert(markers[2]==30&&markers[3]==40);
 // Present owns its copied texture until replay, then its receipt until acquire.
 nvn_queue_test_hold(1);tex.serial=101;uint64_t ticket=nvn_queue_present(&a,&tex,0,NULL);tex.serial=102;
 nvn_queue_test_hold(0);nvn_queue_present_wait(ticket);nvn_queue_present_release(ticket);retired();assert(present_tex==101&&!tex_pins);
 // Host/codec completion replaces a logical receipt with an immediate fence,
 // releasing only its own reference; already captured wait receipts stay valid.
 sync_t cpu={.magic=MAGIC('S','Y','N','C')};nvn_queue_fence(&a,&cpu);
 pthread_mutex_lock(&nq_mu);nq_receipt *old_cpu=(void*)(uintptr_t)cpu.fence;unsigned refs=old_cpu->refs;pthread_mutex_unlock(&nq_mu);
 assert(nvn_host_signal_sync(&cpu)==0 && cpu.fenced==1 && !cpu.fence);
 pthread_mutex_lock(&nq_mu);assert(old_cpu->refs==refs-1);pthread_mutex_unlock(&nq_mu);
 // Finalize retires each queue and its receipt without leaking parked ownership.
 nvn_queue_forget(&a);nvn_queue_forget(&b);nvn_queue_forget(&c);nvn_queue_forget(&d);
 assert(!registry_pins&&!prog_pins&&!tex_pins);
 puts("logical queues: ordering, receipt fences, snapshots and out-of-order retirement passed");
 return 0;
}
"""

class LogicalQueues(unittest.TestCase):
    @unittest.skipUnless(shutil.which("cc"), "host C compiler required")
    def test_actual_scheduler(self):
        out = ROOT / "target/tests"
        out.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=out, prefix="logical-queues-") as temp:
            c, exe = Path(temp)/"test.c", Path(temp)/"test"
            source=(ROOT/"nxnative/dm/gfx/nvn/hle_nvn.c").read_text()
            start=source.index("int nvn_host_signal_sync(")
            end=source.index("\n}",start)+2
            c.write_text(C.replace("HOST_SIGNAL",source[start:end]))
            subprocess.run(["cc", "-std=gnu11", "-O1", "-g", "-fsanitize=address,undefined",
                "-fno-omit-frame-pointer", "-pthread",
                "-I"+str(ROOT/"nxnative/dm/gfx/nvn"),
                "-I"+str(ROOT/"nxnative/dm/gfx/nvn/include"),
                "-I"+str(ROOT/"nxnative/dm/android/compat/include"),
                str(c), str(ROOT/"nxnative/dm/gfx/nvn/nvn_capture.c"), "-o", str(exe)], check=True)
            subprocess.run([str(exe)], check=True, timeout=15,
                           env=dict(os.environ, ASAN_OPTIONS="detect_leaks=0"))


    @unittest.skipUnless(shutil.which("cc"), "host C compiler required")
    def test_live_trace_budget(self):
        src=(ROOT/"nxnative/dm/gfx/nvn/nvn_queue.inc").read_text()
        start=src.index("static void nq_trace(")
        trace=src[start:src.index("\n}\n",start)+2]
        code=r"""
#define _GNU_SOURCE
#define __ANDROID__ 1
#include <stdint.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <stdarg.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
typedef struct { uint64_t serial; } nq_queue;
typedef struct { uint64_t id,frame; } nq_job;
typedef struct { unsigned kind,mode; volatile uint32_t *event; uint32_t value; } nq_node;
static int lines;
static char last[512];
static void log_line(const char *fmt,...) {
 va_list ap;va_start(ap,fmt);vsnprintf(last,sizeof last,fmt,ap);va_end(ap);lines++;
}
#define LOG log_line
TRACE
int main(void) {
 nq_queue q={7};nq_job j={19,23};volatile uint32_t word=9;nq_node n={1,0,&word,9};
 unsetenv("DM_NVN_QUEUE_DIAG");
 nq_trace("park",&q,&j,&n,29);assert(!lines);
 setenv("DM_NVN_QUEUE_DIAG","2",1);
 for(int i=0;i<5;i++) nq_trace("resume",&q,&j,&n,29);
 assert(lines==2&&strstr(last,"queue=7 job=19 frame=23")&&strstr(last,"current=9 fence=29"));
 // Exhausted budgets do not automatically renew; explicitly disarm/rearm does.
 setenv("DM_NVN_QUEUE_DIAG","0",1);nq_trace("off",&q,&j,&n,0);assert(lines==2);
 setenv("DM_NVN_QUEUE_DIAG","2",1);nq_trace("new",&q,&j,&n,30);assert(lines==3);
 setenv("DM_NVN_QUEUE_DIAG","-1",1);nq_trace("off",&q,&j,&n,0);assert(lines==3);
 setenv("DM_NVN_QUEUE_DIAG","1000",1);
 n.event=NULL;for(int i=0;i<70;i++)nq_trace("cap",&q,&j,&n,31);
 assert(lines==67&&strstr(last,"current=0 fence=31"));
 return 0;
}
""".replace("TRACE",trace)
        out=ROOT/"target/tests";out.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=out,prefix="queue-diag-") as temp:
            c,exe=Path(temp)/"test.c",Path(temp)/"test";c.write_text(code)
            subprocess.run(["cc","-std=gnu11","-O1","-g","-fsanitize=address,undefined",
                "-fno-omit-frame-pointer",str(c),"-o",str(exe)],check=True)
            subprocess.run([str(exe)],check=True,timeout=5)

    @unittest.skipUnless(shutil.which("cc"), "host C compiler required")
    def test_startup_gate_and_program_snapshot(self):
        hle = (ROOT/"nxnative/dm/gfx/nvn/hle_nvn.c").read_text()
        common = (ROOT/"nxnative/dm/android/gfx_common.c").read_text()
        gate = hle[hle.index("static pthread_once_t g_nq_config_once"):hle.index("\n#else\nint nvn_logical_queues_on")]
        start = common.index("void mtl_prog_snapshot(")
        end = common.index("\nvoid mtl_prog_unpin", start)
        snapshot = common[start:end]
        code = r"""
#define _GNU_SOURCE
#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include "nvn_obj.h"
#define NVN_LOGICAL_QUEUES 1
#define LOG(...) ((void)0)
GATE
static uint64_t next_serial=99, saved_serial, saved_word;
static int reads;
uint64_t nvn_new_serial(void) { return ++next_serial; }
int prog_key(const prog_t *p,unsigned stage,uint8_t key[8]) {
 assert(stage<p->nstage); if(saved_serial!=p->serial) {
  saved_serial=p->serial;saved_word=*(const uint64_t*)p->sd[stage].control;reads++;
 }
 memcpy(key,&saved_word,8);return 1;
}
SNAPSHOT
static void *thread(void *p) { for(int i=0;i<1000;i++) assert(nvn_logical_queues_on()==(int)(uintptr_t)p);return NULL; }
int main(int argc,char **argv) {
 assert(argc==2);int expected=atoi(argv[1]);assert(nvn_logical_queues_on()==expected);
 setenv("DM_NVN_LOGICAL_QUEUES",expected?"0":"1",1);
 pthread_t t[4];for(int i=0;i<4;i++)assert(!pthread_create(&t[i],NULL,thread,(void*)(uintptr_t)expected));
 for(int i=0;i<4;i++)pthread_join(t[i],NULL);
 uint64_t control=0x12345678;prog_t p={.serial=77,.nstage=1},copy;p.sd[0].control=&control;
 mtl_prog_snapshot(&copy,&p);assert(reads==1&&copy.serial==77&&copy.nstage==1);
 p.serial=78;p.nstage=0;control=0;uint8_t key[8];assert(prog_key(&copy,0,key));uint64_t word;memcpy(&word,key,8);assert(word==0x12345678&&reads==1);
 // Serial-less descriptors acquire their own process-lived key identity.
 p=(prog_t){.nstage=1};control=0xabcdef;p.sd[0].control=&control;mtl_prog_snapshot(&copy,&p);
 assert(!p.serial&&copy.serial==100&&reads==2);control=0;prog_key(&copy,0,key);memcpy(&word,key,8);assert(word==0xabcdef);
 return 0;
}
""".replace("GATE",gate).replace("SNAPSHOT",snapshot)
        out=ROOT/"target/tests"
        out.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=out,prefix="logical-gate-") as temp:
            c,exe=Path(temp)/"test.c",Path(temp)/"test"
            c.write_text(code)
            subprocess.run(["cc","-std=gnu11","-O1","-g","-fsanitize=address,undefined",
                "-fno-omit-frame-pointer","-pthread","-I"+str(ROOT/"nxnative/dm/gfx/nvn"),
                "-I"+str(ROOT/"nxnative/dm/gfx/nvn/include"),str(c),"-o",str(exe)],check=True)
            for value,expected in [("",0),("0",0),("1",1),("01",0),("2",0),("yes",0)]:
                subprocess.run([str(exe),str(expected)],check=True,timeout=5,
                               env=dict(os.environ,DM_NVN_LOGICAL_QUEUES=value))

if __name__ == "__main__":
    unittest.main()

