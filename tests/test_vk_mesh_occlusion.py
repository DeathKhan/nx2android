"""The actual final raster draw of mesh emulation participates in sample queries."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
import sys; sys.path.insert(0, str(Path(__file__).resolve().parent))
import host_cc   # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


class MeshOcclusion(unittest.TestCase):
    @unittest.skipUnless(shutil.which("cc"), "C compiler required")
    @unittest.skipUnless(host_cc.sanitizers(), 'a host C compiler with ASan and UBSan is required (MinGW has neither)')
    def test_actual_final_draw_query_order(self):
        source = (ROOT / "nxnative/dm/android/gfx_vk_exec.c").read_text(encoding='utf-8')
        start = source.index("    if (ok) { cb = vk_cmd();", source.index("static void mesh_draw_one(const p_draw *d, uint32_t batch) {"))
        end = source.index("    if(g_mrt_active)mrt_abort();", start)   # the draw block ends where the observer is closed
        code = r"""
#include <stdint.h>
#include <assert.h>
#include <stddef.h>
typedef int VkCommandBuffer;
typedef struct {uintptr_t b;uint64_t off;} Buffer;
typedef struct {uint64_t key;} shader;
typedef struct {uint64_t ind;} p_draw;
static struct {shader *sh[5];} S;
static int journal_rows,checkpoint,observed,g_mrt_active,g_sub_pass=1,g_draw_i=1;
static uint64_t g_tex_frame,g_set_epoch;
static int pass_journal_row(void){return journal_rows;}
static int logged;
static void VKLOG(const char *f,...){(void)f;logged++;}
enum {VK_INDEX_TYPE_UINT32=1};
static int allowed,debug,sequence,query,draw;
static int vk_cmd(void){return 42;}
static int draw_allowed(void){return allowed;}
static int dbg_draw(void){return debug;}
static void bc_index(int cb,uintptr_t b,uint64_t off,int type){assert(cb==42&&b==7&&off==32&&type==1);}
static void vk_occ_draw(void){query=++sequence;}
static void vkCmdDrawIndexedIndirect(int cb,uintptr_t b,uint64_t off,int count,int stride){
 assert(cb==42&&b==8&&off==80&&count==1&&stride==20);draw=++sequence;assert(query==draw-1);
}
static void pass_dump_draw_checkpoint(const char *kind,uint32_t count,int inst,uint64_t ind,uint64_t v,uint64_t t,uint64_t m){
 (void)kind;(void)count;(void)inst;(void)ind;(void)v;(void)t;(void)m;assert(draw);checkpoint=++sequence;
}
static void mrt_after_draw(int a,int b,uint64_t index_buffer){(void)a;(void)b;assert(draw&&index_buffer==7);observed=++sequence;}
static void run(int ok){
 VkCommandBuffer cb=0;Buffer idx={7,32},launch={8,64};
 static shader vss={1},fss={2},mss={3};shader *vs=&vss,*ts=NULL,*ms=&mss;p_draw dd={0},*d=&dd;uint32_t count=3,ntask=0,nmesh=4;S.sh[1]=&fss;
 query=draw=sequence=checkpoint=observed=logged=0;
""" + source[start:end] + r"""
 assert((query!=0)==(ok&&allowed&&!(debug&1)));assert((draw!=0)==(query!=0));
 assert((checkpoint!=0)==(draw!=0)&&(!checkpoint||checkpoint>draw));assert((observed!=0)==(draw&&g_mrt_active));
 assert(logged==(draw?(journal_rows!=0)+(g_mrt_active!=0):0)); // the journal row and the observer line go with the issued draw only
}
int main(void){for(g_mrt_active=0;g_mrt_active<2;g_mrt_active++)for(journal_rows=0;journal_rows<2;journal_rows++)for(allowed=0;allowed<2;allowed++)for(debug=0;debug<2;debug++){run(0);run(1);}return 0;}
"""
        scratch = ROOT / "target/tests"
        scratch.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="mesh-query-", dir=scratch) as folder:
            p = Path(folder)
            (p / "check.c").write_text(code, encoding='utf-8')
            subprocess.run(["cc", "-std=c11", "-Wall", "-Wextra", "-Werror", "-fsanitize=address,undefined", str(p / "check.c"), "-o", str(p / "check")], check=True)
            subprocess.run([str(p / "check")], check=True)


if __name__ == "__main__":
    unittest.main()
