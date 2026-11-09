"""Actual Vulkan logical-queue state and completion publication helpers."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "nxnative/dm/android/gfx_vk_core.c"
EXEC = ROOT / "nxnative/dm/android/gfx_vk_exec.c"


def function(source, declaration):
    start = source.index(declaration)
    pos = source.index("{", start) + 1
    depth = 1
    while depth:
        depth += (source[pos] == "{") - (source[pos] == "}")
        pos += 1
    return source[start:pos]


def run_c(code):
    target = ROOT / "target/tests"
    target.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=target, prefix="vk-queue-") as temp:
        src, exe = Path(temp) / "test.c", Path(temp) / "test"
        src.write_text(code)
        subprocess.run(["cc", "-std=gnu11", "-g", "-O1", "-fsanitize=address,undefined",
                        "-fno-omit-frame-pointer", str(src), "-o", str(exe)], check=True)
        subprocess.run([str(exe)], check=True)


@unittest.skipUnless(shutil.which("cc"), "a host C compiler is required")
class VulkanQueue(unittest.TestCase):
    def test_actual_queue_state_and_target_lifetime(self):
        source = EXEC.read_text()
        begin = source.index("typedef struct vk_queue_state {")
        end = source.index("static int tfc_ready(void) {", begin)
        helper = source[begin:end]
        code = r"""
#include <assert.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#define VK_NULL_HANDLE 0
typedef struct { uint64_t serial; } tex_t;
typedef struct { uint64_t addr,size; int index; } p_tf_bind;
static struct {
 const void *prog;const void *sh[6];int pass_open;uintptr_t bound_pipe;
 tex_t rt_col[8],rt_dep;uint8_t rt_has_dep,rt_has_col[8];int rt_n;
 uint64_t ubo[6][16],ssbo[6][16],tex[6][256],samp[6][128];
 uint8_t push[2][256];uint32_t raster;uint64_t tex_sseq[6][256],samp_sseq[6][128];
} S;
static p_tf_bind g_tf[4];static uint64_t g_tf_off[4];
static int g_tf_on,g_tf_in_pass,g_tfc_cur=-1,g_rc_skip,g_pool_tex_base,g_pool_samp_base,g_bc_stale;
static uint32_t g_tf_verts;static uint64_t g_set_epoch,g_nvn_active_queue;
static int logical;static int nvn_logical_queues_on(void){return logical;}
static void state_reset(void){memset(&S,0,sizeof S);S.raster=0xabc;}
static void die(const char *s){(void)s;abort();}
static int pins[128],pin_calls,unpin_calls;
static void mtl_tex_pin(uint64_t serial){if(serial){assert(serial<128);pins[serial]++;pin_calls++;}}
static void mtl_tex_snapshot(tex_t *dst,const tex_t *src){tex_t copy=*src;mtl_tex_pin(copy.serial);*dst=copy;}
static void mtl_tex_unpin(uint64_t serial){if(serial){assert(serial<128&&pins[serial]>0);pins[serial]--;unpin_calls++;}}
HELPER
int main(void){
 S.raster=3;mtl_queue_activate(10);assert(g_nvn_active_queue==10&&S.raster==3&&!g_vk_active);
 logical=1;mtl_queue_activate(1);assert(g_vk_active&&S.raster==0xabc&&!g_rc_skip);
 S.sh[0]=(void*)123;S.prog=(void*)456;S.ubo[2][7]=0x111;S.ssbo[4][3]=0x222;
 S.tex[1][200]=0x333;S.samp[2][100]=0x444;S.tex_sseq[1][200]=7;S.samp_sseq[2][100]=9;S.push[1][255]=0x55;S.raster=0x66;
 S.rt_n=1;S.rt_has_col[0]=1;S.rt_col[0].serial=11;S.rt_has_dep=1;S.rt_dep.serial=12;
 g_tf[2]=(p_tf_bind){0x777,0x888,2};g_tf_off[2]=0x999;g_tf_on=1;g_tf_verts=71;g_tfc_cur=9;
 g_rc_skip=1;g_pool_tex_base=1000;g_pool_samp_base=2000;
 vk_queue_preserve();assert(pins[11]==1&&pins[12]==1&&g_vk_active->state.prog==NULL);
 // Same-queue successive jobs retain new targets before job-owned snapshots disappear.
 mtl_queue_activate(1);assert(S.raster==0x66&&g_rc_skip);
 S.rt_col[0].serial=13;vk_queue_preserve();assert(!pins[11]&&pins[12]==1&&pins[13]==1);
 int saved_pin_calls=pin_calls;vk_queue_preserve();assert(pin_calls==saved_pin_calls);
 mtl_queue_activate(2);assert(S.raster==0xabc&&!S.sh[0]&&!S.ubo[2][7]&&!g_rc_skip);
 assert(!g_pool_tex_base&&!g_pool_samp_base&&!g_tf_on&&g_tfc_cur==-1);
 S.raster=0x22;S.rt_n=1;S.rt_has_col[0]=1;S.rt_col[0].serial=14;vk_queue_preserve();
 mtl_queue_activate(1);
 assert(S.sh[0]==(void*)123&&!S.prog&&S.ubo[2][7]==0x111&&S.ssbo[4][3]==0x222);
 assert(S.tex[1][200]==0x333&&S.samp[2][100]==0x444&&S.push[1][255]==0x55&&S.raster==0x66);
 assert(S.tex_sseq[1][200]==7&&S.samp_sseq[2][100]==9);
 assert(S.rt_col[0].serial==13&&S.rt_dep.serial==12&&g_rc_skip);
 assert(g_pool_tex_base==1000&&g_pool_samp_base==2000&&g_tf_on&&g_tf_verts==71&&g_tfc_cur==9);
 assert(g_tf[2].addr==0x777&&g_tf_off[2]==0x999&&!g_tf_in_pass&&!S.pass_open&&!S.bound_pipe&&g_bc_stale);
 mtl_queue_forget(1);assert(!g_vk_active&&!pins[12]&&!pins[13]&&pins[14]==1&&!g_rc_skip);
 mtl_queue_activate(2);assert(S.raster==0x22&&S.rt_col[0].serial==14);
 mtl_queue_forget(999);assert(pins[14]==1);
 mtl_queue_forget(2);assert(!g_vk_queues&&!pins[14]&&pin_calls==unpin_calls);
 mtl_queue_activate(1);assert(S.raster==0xabc&&!S.sh[0]&&!g_rc_skip);
 mtl_queue_forget(1);return 0;
}
""".replace("HELPER", helper)
        run_c(code)

    def test_actual_release_frontier_and_counter_expiry(self):
        core, exe = CORE.read_text(), EXEC.read_text()
        helper = function(core, "static void completion_events_publish(const cpend *c) {")
        value = function(exe, "static int rc_value(uint64_t a, uint64_t *v) {")
        code = r"""
#include <assert.h>
#include <stdint.h>
#include <stddef.h>
static uint64_t g_reports_done;
static uint64_t vk_reports_fence_done(void){return __atomic_load_n(&g_reports_done,__ATOMIC_ACQUIRE);}
typedef struct {int kind;volatile uint32_t *val;uint32_t mode,v;uint64_t addr;uint32_t type;int q;} post_op;
typedef struct {uint64_t fence;int n;post_op *ops;} cpend;
static uint64_t report;static int published;
static void nvn_event_publish_gpu(volatile uint32_t *word,unsigned mode,uint32_t value,uint64_t fence){
 assert(report==99&&vk_reports_fence_done()==fence);
 if(mode==1)__atomic_fetch_add(word,value,__ATOMIC_RELEASE);
 else __atomic_store_n(word,value,__ATOMIC_RELEASE);
 published++;
}
HELPER
static struct {uint64_t addr,fence;int kind;} g_rc_pend[4];
static int g_rc_npend;
static int rc_occ_has(uint64_t a){(void)a;return 0;}
VALUE
int main(void){
 volatile uint32_t event=0;post_op ops[]={{.kind=1,.val=&event,.mode=0,.v=3},{.kind=2},{.kind=1,.val=&event,.mode=1,.v=4}};
 cpend done={.fence=8,.n=3,.ops=ops};report=99;
 completion_events_publish(&done);assert(event==7&&published==2&&g_reports_done==8);
 uint64_t bytes=123,v=999;g_rc_npend=1;g_rc_pend[0]=(typeof(g_rc_pend[0])){(uintptr_t)&bytes,9,1};
 assert(rc_value((uintptr_t)&bytes,&v)&&v==0); // pending timestamp has known zero counter
 g_rc_pend[0].kind=2;assert(!rc_value((uintptr_t)&bytes,&v));
 g_rc_pend[0].kind=1;__atomic_store_n(&g_reports_done,9,__ATOMIC_RELEASE);
 bytes=456;assert(rc_value((uintptr_t)&bytes,&v)&&v==456); // completed report, then CPU rewrite
 g_rc_pend[0].kind=2;assert(rc_value((uintptr_t)&bytes,&v)&&v==456); // completed unknown GPU writer is now readable
 return 0;
}
""".replace("HELPER", helper).replace("VALUE", value)
        run_c(code)
        completion = function(core, "static void *completion_main(void *a) {")
        self.assertLess(completion.index("pools_copy_back("), completion.index("completion_events_publish(c)"))
        self.assertLess(completion.index("memcpy((void *)(uintptr_t)o->addr"), completion.index("completion_events_publish(c)"))
        self.assertLess(completion.index("completion_events_publish(c)"), completion.index("__atomic_store_n(&g_done"))
        self.assertLess(completion.index("__atomic_store_n(&g_done"), completion.index("nvn_queue_gpu_progress()"))
        commit = function(core, "uint64_t mtl_sub_commit(void) {")
        self.assertLess(commit.index("vk_end_pass_land()"), commit.index("vk_queue_preserve()"))
        self.assertLess(commit.index("vk_queue_preserve()"), commit.index("vkQueueSubmit("))


if __name__ == "__main__":
    unittest.main()
