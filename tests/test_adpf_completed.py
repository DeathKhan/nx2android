"""Execute real completed-frame records and ADPF consumer with deterministic GPU reports."""
from pathlib import Path
import os
import subprocess
import tempfile
import unittest
import sys; sys.path.insert(0, str(Path(__file__).resolve().parent))
import host_cc   # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def function(source, name):
    start = source.index(name)
    pos = source.index('{', start) + 1
    depth = 1
    while depth:
        depth += (source[pos] == '{') - (source[pos] == '}')
        pos += 1
    return source[start:pos]


class CompletedAdpf(unittest.TestCase):
    @unittest.skipUnless(host_cc.posix() and host_cc.sanitizers(), 'a POSIX host C compiler with ASan and UBSan is required (MinGW has no setenv, libasan or libubsan)')
    def test_actual_records_consumer_and_lifecycle(self):
        source = (ROOT / 'nxnative/dm/android/perf_android.c').read_text(encoding='utf-8')
        helpers = '\n'.join(function(source, name) for name in (
            'static int perf_completed_enabled(',
            'static int hint_completed_frame(',
            'void dm_perf_frame_id('))
        code = r'''
#define _GNU_SOURCE
#include <assert.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdatomic.h>
#include <sched.h>
#include "perf_frame.h"
static perf_frame_state g_perf_frames = {.mutex=PTHREAD_MUTEX_INITIALIZER};
static void log_ignore(const char *fmt,...) { (void)fmt; }
#define LOG(...) log_ignore(__VA_ARGS__)
typedef struct {int status;uint64_t begin,end,busy;} gpu_record;
static gpu_record gpu[1024];
static int vk_perf_frame_gpu(uint64_t id,uint64_t *b,uint64_t *e,uint64_t *w) {
 if(id>=1024)return 0;*b=gpu[id].begin;*e=gpu[id].end;*w=gpu[id].busy;return gpu[id].status;
}
typedef struct {int64_t start,total,cpu,gpu,work,pace;} report;
static report reports[128];static int nr;
static void hint_report_values(int64_t start,int64_t total,int64_t cpu,int64_t g,
 int64_t work,int64_t pace,int64_t now,int64_t target) {
 (void)now;(void)target;assert(nr<128);assert(start>0&&total>0&&total<250000000);
 assert(cpu>=0&&cpu<=total&&g>=0&&g<=total);reports[nr++]=(report){start,total,cpu,g,work,pace};
}
static int64_t clock_now,g_work_start,g_prev_present,g_pace,g_pq,g_frame_gpu,g_hint_target=16666667;
static uint64_t g_game_period;
static int g_step,g_hint=1;
static int64_t perf_now(void){return clock_now;}
static int hint_target_fps(const char *e){(void)e;return 0;}
static int64_t hint_target_ns(int64_t p,uint64_t gp,int f){(void)gp;(void)f;return p;}
static void hint_init(int64_t p){(void)p;}
static void hint_update_target(int64_t p,int changed){(void)changed;g_hint_target=p;}
static void gpu_turbo(void){}
static void adapt_frame(int64_t w,int64_t t){(void)w;(void)t;}
static void hint_threads_scan(int32_t t){(void)t;}
static void hint_stats_report(void){}
static int gettid(void){return 123;}
static void hint_report(int64_t w,int64_t pq,int64_t pace,int64_t now,int64_t target) {
 int64_t c=w>pq?w-pq:0,t=w>g_frame_gpu?w:g_frame_gpu;
 hint_report_values(now-t,t,c,g_frame_gpu,w,pace,now,target);
}
HELPERS
static void closed(uint64_t id) {
 uint64_t t=1000000+id*1000;
 perf_frame_start_record(&g_perf_frames,id,t,0,0);
 perf_frame_close_record(&g_perf_frames,id,t+100,100,100,0,50);
}
static void ready(uint64_t id) {
 uint64_t t=1000000+id*1000;
 gpu[id]=(gpu_record){1,t+30,t+150,120};
}
static void activate(void) {
 closed(0);ready(0);assert(hint_completed_frame(1,2000000,16666667));assert(nr==1);
}
static void consumer(void) {
 closed(0);assert(!hint_completed_frame(1,2000000,16666667));assert(!nr);
 assert(perf_frame_mark_reported(&g_perf_frames,0)); // actual caller's bootstrap accounting
 ready(0);assert(hint_completed_frame(1,2000000,16666667));assert(!nr);
 closed(1);closed(2);ready(1);
 assert(hint_completed_frame(3,2000000,16666667));assert(nr==1);
 assert(reports[0].start==1001000&&reports[0].total==150&&reports[0].cpu==100&&reports[0].gpu==120);
 // Several GPU parts are represented by one exact completion copyout: keep pending until ready.
 assert(hint_completed_frame(4,2000000,16666667));assert(nr==1);
 ready(2);gpu[2].end+=50;gpu[2].busy=130;
 assert(hint_completed_frame(4,2000000,16666667));assert(nr==2&&reports[1].total==200&&reports[1].gpu==130);
 assert(hint_completed_frame(4,2000000,16666667)&&nr==2);
 closed(3);gpu[3].status=-1; // dropped/unsupported immediately falls back once
 assert(hint_completed_frame(4,2000000,16666667)&&nr==3);
 assert(reports[2].total==100&&reports[2].gpu==50);
 closed(4);assert(hint_completed_frame(11,2000000,16666667)&&nr==3);
 assert(hint_completed_frame(12,2000000,16666667)&&nr==4); // pending aged eight frames
 ready(4);assert(hint_completed_frame(13,2000000,16666667)&&nr==4); // no late duplicate
 closed(5);ready(5);gpu[5].begin=1; // incorrect epoch cannot reach ADPF
 assert(hint_completed_frame(13,2000000,16666667)&&nr==5&&reports[4].total==100);
 perf_frame_start_record(&g_perf_frames,6,1006000,0,0); // unclosed/missing record never borrowed
 assert(hint_completed_frame(20,2000000,16666667)&&nr==5);
 closed(7);closed(71);ready(7);ready(71); // bounded ring exact-ID check
 assert(hint_completed_frame(72,2000000,16666667)&&nr==6&&reports[5].start==1071000);
 assert(hint_completed_frame(UINT64_MAX,2000000,16666667)==0&&nr==6);
}
static void lifecycle(void) {
 // An older async present cleared the legacy global, but exact records still close/report.
 perf_frame_start_record(&g_perf_frames,0,1000000,0,0);g_work_start=0;clock_now=1000100;
 dm_perf_frame_id(0,1.0/60);perf_frame_record r;
 assert(perf_frame_copy_record(&g_perf_frames,0,&r)&&r.closed&&r.reported&&r.cpu==100&&nr==1);
 ready(0);
 perf_frame_start_record(&g_perf_frames,1,1001000,0,0);g_work_start=0;clock_now=1001100;
 dm_perf_frame_id(1,1.0/60);
 assert(nr==1&&perf_frame_copy_record(&g_perf_frames,1,&r)&&r.closed&&!r.reported);
 ready(1);clock_now=1002100;dm_perf_frame_id(2,1.0/60);
 assert(nr==2&&reports[1].start==1001000&&reports[1].total==150);
 // A stale legacy interval must not invalidate an otherwise matched CPU/GPU frame.
 perf_frame_start_record(&g_perf_frames,3,1003000,0,0);g_work_start=1;g_prev_present=2000000;clock_now=1003100;
 dm_perf_frame_id(3,1.0/60);
 assert(perf_frame_copy_record(&g_perf_frames,3,&r)&&!r.invalid&&r.work==100);
 ready(3);clock_now=1004100;dm_perf_frame_id(4,1.0/60);assert(nr==3);
}
static atomic_int published,release_writer;
static void *writer(void *unused) {
 (void)unused;
 for(int id=100;id<300;id++) {
  uint64_t t=(uint64_t)id*1000;
  // Replay may reach this ID before guest acquire publishes its own start.
  perf_frame_replay_record(&g_perf_frames,id,t+10,t+30);
  perf_frame_start_record(&g_perf_frames,id,t,7,11);
  perf_frame_replay_record(&g_perf_frames,id,t+40,t+60);
  perf_frame_close_record(&g_perf_frames,id,t+100,100,100,40,50);
  atomic_store(&published,id);
  while(atomic_load(&release_writer)!=id)sched_yield();
 }
 return NULL;
}
static void records(void) {
 perf_frame_record r;int64_t start,total;
 perf_frame_replay_record(&g_perf_frames,1,1010,1030);
 perf_frame_start_record(&g_perf_frames,1,1000,7,11);
 perf_frame_replay_record(&g_perf_frames,1,1040,1060);
 assert(!perf_frame_copy_record(&g_perf_frames,1,&r));
 assert(perf_frame_peek_record(&g_perf_frames,1,&r)&&r.start==1000&&r.replay==40);
 perf_frame_close_record(&g_perf_frames,1,1100,100,100,40,50);
 assert(perf_frame_copy_record(&g_perf_frames,1,&r)&&r.cpu==100&&r.pace==7&&r.pq==11);
 assert(perf_frame_values(&r,1020,1150,130,&start,&total)&&start==1000&&total==150);
 assert(!perf_frame_values(&r,999,1150,130,&start,&total));
 assert(!perf_frame_values(&r,1020,1150,151,&start,&total));
 assert(!perf_frame_values(&r,1150,1020,1,&start,&total));
 assert(!perf_frame_values(&r,1020,250001000,1,&start,&total));
 assert(perf_frame_mark_reported(&g_perf_frames,1)&&!perf_frame_mark_reported(&g_perf_frames,1));
 perf_frame_replay_record(&g_perf_frames,1,1110,1120);
 assert(perf_frame_copy_record(&g_perf_frames,1,&r)&&r.invalid);
 perf_frame_start_record(&g_perf_frames,65,65000,0,0);
 perf_frame_close_record(&g_perf_frames,65,65100,100,100,0,50);
 perf_frame_start_record(&g_perf_frames,1,1000,0,0);
 assert(!perf_frame_copy_record(&g_perf_frames,1,&r)&&!perf_frame_mark_reported(&g_perf_frames,1));
 assert(perf_frame_copy_record(&g_perf_frames,65,&r)&&r.start==65000);
 perf_frame_close_record(&g_perf_frames,66,66100,100,100,0,0);
 assert(perf_frame_copy_record(&g_perf_frames,66,&r)&&r.invalid); // empty
 // Concurrent writer/reader exercise publication and two full ring wraps.
 pthread_t thread;assert(!pthread_create(&thread,NULL,writer,NULL));
 for(int id=100;id<300;id++) {
  while(atomic_load(&published)!=id)sched_yield();
  assert(perf_frame_copy_record(&g_perf_frames,id,&r));
  assert(r.id==(uint64_t)id&&r.start==(uint64_t)id*1000&&r.end==r.start+100&&r.replay==40&&r.cpu==100);
  assert(perf_frame_mark_reported(&g_perf_frames,id));
  atomic_store(&release_writer,id);
 }
 assert(!pthread_join(thread,NULL));
}
int main(int argc,char **argv) {
 assert(argc==2);setenv("DM_ADPF_COMPLETE_DIAG","0",1);
 if(!strcmp(argv[1],"off")) {
  unsetenv("DM_ADPF_COMPLETE");closed(0);ready(0);
  assert(!hint_completed_frame(1,2000000,16666667)&&!nr);
  setenv("DM_ADPF_COMPLETE","1",1);assert(!hint_completed_frame(1,2000000,16666667)&&!nr);
  return 0;
 }
 setenv("DM_ADPF_COMPLETE","1",1);
 if(!strcmp(argv[1],"consumer"))consumer();
 else if(!strcmp(argv[1],"lifecycle"))lifecycle();
 else if(!strcmp(argv[1],"records"))records();
 else if(!strcmp(argv[1],"expired")) {
  activate();closed(1);gpu[1].status=0;
  assert(hint_completed_frame(9,2000000,16666667)&&nr==2);
 }
 else abort();
 return 0;
}
'''.replace('HELPERS', helpers)
        target=ROOT/'target/tests'
        target.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='adpf-completed-',dir=target) as folder:
            path=Path(folder);(path/'test.c').write_text(code, encoding='utf-8')
            subprocess.run([os.environ.get('CC','cc'),'-std=c11','-g','-O1',
                '-fsanitize=address,undefined','-pthread','-I',str(ROOT/'nxnative/dm/android'),
                str(path/'test.c'),'-o',str(path/'test')],check=True)
            for case in ('off','consumer','lifecycle','records','expired'):
                subprocess.run([str(path/'test'),case],check=True,timeout=15)


if __name__=='__main__':
    unittest.main()

