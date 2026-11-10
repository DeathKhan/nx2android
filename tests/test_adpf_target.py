"""ADPF targets, async work accounting and report failures exercise runtime helpers."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
import sys; sys.path.insert(0, str(Path(__file__).resolve().parent))
import host_cc   # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def function(source, name):
    start = source.index('static ', source.index(name) - 24)
    brace = source.index('{', start)
    depth = 1
    pos = brace + 1
    while depth:
        depth += (source[pos] == '{') - (source[pos] == '}')
        pos += 1
    return source[start:pos]


class AdpfTarget(unittest.TestCase):
    @unittest.skipUnless(shutil.which('cc'), 'a host C compiler is required')
    @unittest.skipUnless(host_cc.sanitizers(), 'a host C compiler with ASan and UBSan is required (MinGW has neither)')
    def test_actual_targets_and_report_failures(self):
        runtime = (ROOT / 'nxnative/dm/android/perf_android.c').read_text(encoding='utf-8')
        start = runtime.index('typedef struct APerformanceHintManager')
        state = runtime[start:runtime.index('static int64_t perf_now(', start)]
        start = runtime.index('static int hint_target_fps(')
        helpers = runtime[start:runtime.index('// Keep both known frame threads', start)]
        report = function(runtime, 'hint_report_values(') + function(runtime, 'hint_report(')
        start = runtime.index('static int64_t g_frame_cost,')
        costs = runtime[start:runtime.index('// The frame\'s work to the CPU hint', start)]
        code = r'''
#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdarg.h>
#include <assert.h>
#include <errno.h>
static void log_ignore(const char *fmt,...){(void)fmt;}
#define LOG(...) log_ignore(__VA_ARGS__)
STATE
struct APerformanceHintSession { int unused; };
struct AWorkDuration { int64_t start,total,cpu,gpu; };
COSTS
HELPERS
REPORT
static int updates,update_result,legacy,reports,report_result,legacy_result;
static int64_t requested,legacy_work;
static int mock_update(APerformanceHintSession *s,int64_t target){assert(s==g_hint);updates++;requested=target;return update_result;}
static int mock_report(APerformanceHintSession *s,int64_t work){assert(s==g_hint&&work>0);legacy_work=work;legacy++;return legacy_result;}
static int mock_report2(APerformanceHintSession *s,AWorkDuration *d){assert(s==g_hint&&d==g_wd&&d->start>0&&d->total>0&&d->cpu>=0&&d->gpu>=0&&(d->cpu||d->gpu));assert(d->total>=d->cpu&&d->total>=d->gpu);reports++;return report_result;}
static void set_start(AWorkDuration *d,int64_t n){d->start=n;}
static void set_total(AWorkDuration *d,int64_t n){d->total=n;}
static void set_cpu(AWorkDuration *d,int64_t n){d->cpu=n;}
static void set_gpu(AWorkDuration *d,int64_t n){d->gpu=n;}
int main(void){
 const int64_t thirty=33333333,sixty=16666667;
 int32_t tids[8];
 assert(hint_known_threads(101,tids)==1&&tids[0]==101);
 __atomic_store_n(&g_acquire_tid,202,__ATOMIC_RELAXED);
 assert(hint_known_threads(101,tids)==2&&tids[0]==101&&tids[1]==202);
 assert(hint_known_threads(202,tids)==1&&tids[0]==202); // Synchronous replay: no duplicate.
 __atomic_store_n(&g_acquire_tid,303,__ATOMIC_RELAXED);
 assert(hint_known_threads(101,tids)==2&&tids[1]==303); // A new acquire thread replaces the old one.
 assert(!hint_target_fps(NULL)&&!hint_target_fps(""));
 const char *bad[]={"0","-1","241","60fps","60.0","99999999999999999999999999999","x","60 "};
 for(size_t i=0;i<sizeof bad/sizeof *bad;i++)assert(!hint_target_fps(bad[i]));
 assert(hint_target_fps("1")==1&&hint_target_fps("60")==60&&hint_target_fps("240")==240);
 assert(hint_target_ns(thirty,thirty,0)==thirty);
 assert(hint_target_ns(thirty,sixty,0)==sixty);
 assert(hint_target_ns(sixty,thirty,0)==sixty);
 assert(hint_target_ns(thirty,0,0)==thirty);
 assert(hint_target_ns(thirty,thirty,60)==sixty);
 assert(hint_target_ns(thirty,thirty,240)==4166667);
 struct APerformanceHintSession session={0};struct AWorkDuration duration={0};
 H.update=mock_update;g_hint_target=thirty;
 hint_update_target(sixty,1);assert(!updates&&g_hint_target==thirty); // No session.
 g_hint=&session;H.update=NULL;hint_update_target(sixty,1);assert(!updates&&g_hint_target==thirty&&g_hint_stats.target_failed==1);
 memset(&g_hint_stats,0,sizeof g_hint_stats);
 H.update=mock_update;hint_update_target(thirty,0);assert(!updates);
 update_result=22;hint_update_target(sixty,1);
 assert(updates==1&&requested==sixty&&g_hint_target==thirty&&g_hint_stats.target_failed==1&&!g_hint_stats.target_ok);
 update_result=0;hint_update_target(sixty,0);
 assert(updates==2&&g_hint_target==sixty&&g_hint_stats.target_ok==1);
 hint_update_target(sixty,0);assert(updates==2);
 hint_update_target(hint_target_ns(thirty,thirty,hint_target_fps("0")),1);
 assert(updates==3&&g_hint_target==thirty&&g_hint_stats.target_ok==2); // Disabling restores normal target.
 H.report=mock_report;H.report2=mock_report2;g_wd=&duration;
 H.wd_start=set_start;H.wd_total=set_total;H.wd_cpu=set_cpu;H.wd_gpu=set_gpu;
 g_frame_gpu=20000000;hint_report(30000000,8000000,0,1000000000,thirty);
 assert(reports==1&&!legacy&&duration.start==970000000&&duration.total==30000000&&duration.cpu==22000000&&duration.gpu==20000000);
 assert(g_hint_stats.report2_calls==1&&!g_hint_stats.report2_failed&&g_hint_stats.report2_cpu_ns==22000000&&g_hint_stats.report2_gpu_ns==20000000);
 report_result=22;hint_report(30000000,8000000,0,1030000000,thirty);
 assert(reports==2&&legacy==1&&g_hint_stats.report2_failed==1&&g_hint_stats.fallback==1&&g_hint_stats.legacy_calls==1&&!g_hint_stats.legacy_failed);
 legacy_result=5;hint_report(30000000,8000000,0,1060000000,thirty);
 assert(g_hint_stats.report2_failed==2&&g_hint_stats.fallback==2&&g_hint_stats.legacy_failed==1);
 H.report2=NULL;legacy_result=0;hint_report(30000000,0,0,1090000000,thirty);
 assert(reports==3&&legacy==3&&g_hint_stats.legacy_calls==3&&g_hint_stats.legacy_failed==1);
 legacy_result=5;hint_report(30000000,0,0,1120000000,thirty);
 assert(legacy==4&&g_hint_stats.legacy_failed==2);
 H.report2=mock_report2;report_result=0;legacy_result=0;
 dm_perf_frame_cost(-1,-1);assert(!g_frame_replay&&!g_frame_gpu&&!g_frame_cost);
 hint_report(30000000,40000000,0,1150000000,thirty);
 assert(reports==3&&legacy==5&&legacy_work==30000000); // No measured component: valid legacy fallback.
 dm_perf_frame_cost(16000000,24000000);assert(g_frame_replay==16000000&&g_frame_cost==24000000);
 hint_report(30000000,26670000,0,1200000000,sixty);
 assert(duration.cpu==16000000&&duration.gpu==24000000&&duration.total==30000000); // Async replay cannot disappear in guest wait.
 hint_report(5000000,4000000,0,1230000000,sixty);
 assert(duration.cpu==16000000&&duration.total==24000000&&duration.start==1206000000); // Each measured component fits total; parallel work is not summed.
 dm_perf_frame_cost(16000000,0);
 hint_report(5000000,4000000,0,1260000000,sixty);
 assert(duration.cpu==16000000&&duration.total==16000000&&duration.start==1244000000);
 dm_perf_frame_cost(10000000,5000000);
 hint_report(30000000,8000000,8000000,1290000000,thirty);
 assert(duration.cpu==22000000&&duration.total==30000000); // Existing larger estimate and pacing treatment retained.
 H.report2=NULL;dm_perf_frame_cost(16000000,0);
 hint_report(5000000,4000000,0,1320000000,sixty);
 assert(legacy_work==16000000); // Older Android also receives known work lower bound.
 hint_stats_report();
 (void)g_hint_tids;(void)g_hint_n;
 return 0;
}
'''
        for marker, text in [('STATE', state), ('HELPERS', helpers), ('REPORT', report), ('COSTS', costs)]:
            code = code.replace(marker, text)
        target = ROOT / 'target/tests'; target.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='adpf-target-', dir=target) as folder:
            src, exe = Path(folder) / 'check.c', Path(folder) / 'check'
            src.write_text(code, encoding='utf-8')
            subprocess.run(['cc', '-std=c11', '-Wall', '-Wextra', '-Werror', '-fsanitize=address,undefined',
                            str(src), '-o', str(exe)], check=True)
            subprocess.run([str(exe)], check=True, timeout=15)


if __name__ == '__main__':
    unittest.main()
