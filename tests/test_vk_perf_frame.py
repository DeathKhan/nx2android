"""Completed frame GPU records require exact IDs, calibrated endpoints and a sealed fence."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
import sys; sys.path.insert(0, str(Path(__file__).resolve().parent))
import host_cc   # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def function(source, declaration):
    start = source.index(declaration)
    brace = source.index('{', start)
    depth, pos = 1, brace + 1
    while depth:
        depth += (source[pos] == '{') - (source[pos] == '}')
        pos += 1
    return source[start:pos]


class FrameTiming(unittest.TestCase):
    @unittest.skipUnless(shutil.which('cc'), 'a host C compiler is required')
    @unittest.skipUnless(host_cc.posix() and host_cc.sanitizers(), 'a POSIX host C compiler with ASan and UBSan is required (MinGW has no setenv, libasan or libubsan)')
    def test_completed_sealed_frames(self):
        core = (ROOT / 'nxnative/dm/android/gfx_vk_core.c').read_text(encoding='utf-8')
        start = core.index('static int perf_frame_on(')
        end = core.index('double mtl_gpu_ms(', start)
        helper = core[start:end]
        code = r"""
#define _POSIX_C_SOURCE 200809L
#include <stdlib.h>
#include <assert.h>
#include <pthread.h>
#include <stdint.h>
#include <stdatomic.h>
#include <string.h>
static uint64_t g_cur_frame;
HELPER
static uint64_t a,b,c;
static void reset(void) {
 memset(g_perf_frames,0,sizeof g_perf_frames);g_perf_completed_fence=0;
 a=b=c=999;
}
static int result(uint64_t frame) { return vk_perf_frame_gpu(frame,&a,&b,&c); }
static atomic_int stop_reader;
static void *reader(void *unused) {
 (void)unused;
 while(!atomic_load(&stop_reader)) {
  for(uint64_t f=1000;f<1200;f++) {
   uint64_t begin=0,end=0,busy=0;int status=vk_perf_frame_gpu(f,&begin,&end,&busy);
   if(status==1)assert(begin==f*100&&end==f*100+30&&busy==20);
  }
 }
 return NULL;
}
int main(int argc,char **argv) {
 (void)argv;
 if(argc>1) {
  unsetenv("DM_ADPF_COMPLETE");reset();assert(!perf_frame_on());
  perf_frame_submit(1,1);perf_frame_complete(1,1,1,100,110);vk_perf_frame_seal(1);
  assert(result(1)==-1&&!g_perf_frames[1].occupied&&!g_perf_completed_fence);
  setenv("DM_ADPF_COMPLETE","1",1);assert(!perf_frame_on()); // startup-only switch
  return 0;
 }
 setenv("DM_ADPF_COMPLETE","1",1);assert(perf_frame_on());
 reset();assert(!result(1)&&a==999&&b==999&&c==999);
 g_cur_frame=55;assert(vk_frame_current()==55);
 // Several command-buffer parts remain one submission/one completion record.
 perf_frame_submit(1,1);perf_frame_submit(1,2);vk_perf_frame_seal(1);
 assert(!result(1));perf_frame_complete(1,1,1,100,110);assert(!result(1));
 perf_frame_complete(1,2,1,130,150);
 assert(result(1)==1&&a==100&&b==150&&c==30);
 vk_perf_frame_seal(1);assert(result(1)==1); // idempotent seal
 // Completion may arrive before the presenter seals this frame.
 perf_frame_submit(2,3);perf_frame_complete(2,3,1,200,220);assert(!result(2));
 vk_perf_frame_seal(2);assert(result(2)==1&&a==200&&b==220&&c==20);
 // A completed neighbor must not publish the next frame's pending GPU work.
 perf_frame_submit(3,4);perf_frame_submit(4,5);vk_perf_frame_seal(3);vk_perf_frame_seal(4);
 perf_frame_complete(3,4,1,300,310);assert(result(3)==1&&!result(4));
 perf_frame_complete(4,5,1,400,410);assert(result(4)==1);
 // Any missing calibration or dropped submission invalidates the whole frame.
 perf_frame_submit(5,6);perf_frame_submit(5,7);vk_perf_frame_seal(5);
 perf_frame_complete(5,6,1,500,510);perf_frame_complete(5,7,0,0,0);
 assert(result(5)==-1);
 perf_frame_submit(6,8);vk_perf_frame_seal(6);perf_frame_complete(6,8,1,650,640);
 assert(result(6)==-1);
 vk_perf_frame_seal(7);assert(result(7)==-1); // empty: no borrowed prior fence
 perf_frame_complete(8,9,1,800,810);vk_perf_frame_seal(8);assert(result(8)==-1);
 // Post-seal submissions and repeated completions cannot silently extend a report.
 perf_frame_submit(9,10);vk_perf_frame_seal(9);perf_frame_submit(9,11);assert(result(9)==-1);
 perf_frame_submit(10,12);vk_perf_frame_seal(10);perf_frame_complete(10,12,1,1000,1010);
 assert(result(10)==1);perf_frame_complete(10,12,1,1000,1010);assert(result(10)==-1);
 // A slow old completion cannot clobber a newer frame occupying the same ring slot.
 reset();perf_frame_submit(20,1);vk_perf_frame_seal(20);
 perf_frame_submit(84,2);vk_perf_frame_seal(84);assert(!result(20)&&!result(84));
 perf_frame_complete(20,1,1,200,210);assert(!result(20)&&!result(84));
 perf_frame_complete(84,2,1,840,860);assert(result(84)==1&&a==840&&c==20);
 vk_perf_frame_seal(20);perf_frame_submit(20,3);assert(!result(20)&&result(84)==1);
 // A timestamp-quantized no-op submission may be zero length within a valid frame.
 reset();perf_frame_submit(1,1);perf_frame_submit(1,2);vk_perf_frame_seal(1);
 perf_frame_complete(1,1,1,100,100);assert(!result(1));
 perf_frame_complete(1,2,1,110,120);assert(result(1)==1&&a==100&&b==120&&c==10);
 perf_frame_submit(2,3);vk_perf_frame_seal(2);perf_frame_complete(2,3,1,200,200);
 assert(result(2)==-1); // an entirely zero-duration frame cannot report ADPF work
 // Checked accumulation refuses overflow; partial output never escapes on failure.
 reset();perf_frame_submit(1,1);perf_frame_submit(1,2);vk_perf_frame_seal(1);
 perf_frame_complete(1,1,1,1,UINT64_MAX-1);perf_frame_complete(1,2,1,1,10);
 assert(result(1)==-1&&a==999&&b==999&&c==999);
 // Concurrent readout sees complete consistent snapshots while the bounded ring wraps.
 reset();pthread_t thread;assert(!pthread_create(&thread,NULL,reader,NULL));
 for(uint64_t f=1000;f<1200;f++) {
  uint64_t fence=(f-1000)*2+1;
  perf_frame_submit(f,fence);perf_frame_submit(f,fence+1);
  perf_frame_complete(f,fence,1,f*100,f*100+10);vk_perf_frame_seal(f);
  perf_frame_complete(f,fence+1,1,f*100+20,f*100+30);
 }
 atomic_store(&stop_reader,1);assert(!pthread_join(thread,NULL));
 assert(result(1199)==1&&a==119900&&b==119930&&c==20);
 return 0;
}
""".replace('HELPER', helper)
        target = ROOT / 'target/tests'
        target.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='vk-perf-frame-', dir=target) as folder:
            src, exe = Path(folder) / 'test.c', Path(folder) / 'test'
            src.write_text(code, encoding='utf-8')
            subprocess.run(['cc', '-std=c11', '-g', '-O1', '-Wall', '-Wextra', '-Werror',
                            '-fsanitize=address,undefined', '-pthread', str(src), '-o', str(exe)], check=True)
            subprocess.run([str(exe)], check=True, timeout=15)
            subprocess.run([str(exe), 'off'], check=True, timeout=15)

    def test_completion_and_registration_order(self):
        core = (ROOT / 'nxnative/dm/android/gfx_vk_core.c').read_text(encoding='utf-8')
        completion = function(core, 'static void *completion_main(')
        commit = function(core, 'uint64_t mtl_sub_commit(')
        self.assertLess(commit.index('perf_frame_submit(c->frame, c->fence)'), commit.index('g_ct->next = c'))
        self.assertEqual(completion.count('gpu_clock_sample()'), 1)
        self.assertEqual(completion.count('vkGetQueryPoolResults(V.dev, g_tsq,'), 1)
        self.assertIn('gpu_clock_host_ns(&clock, submit_ts[0], &frame_begin)', completion)
        self.assertIn('gpu_clock_host_ns(&clock, submit_ts[1], &frame_end)', completion)
        self.assertIn('c->slot >= 0', completion)
        self.assertLess(completion.index('gpu_clock_sample()'), completion.index('pools_copy_back('))
        self.assertLess(completion.index('perf_frame_complete(c->frame, c->fence, frame_valid, frame_begin, frame_end)'),
                        completion.index('__atomic_store_n(&g_done,'))
        helpers = core[core.index('enum { PERF_FRAMES'):core.index('double mtl_gpu_ms(')]
        self.assertNotIn('gpu_ticks()', helpers)
        self.assertNotIn('guest_ns', helpers)


if __name__ == '__main__':
    unittest.main()
