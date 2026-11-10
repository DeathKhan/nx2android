"""The actual final raster draw of mesh emulation participates in sample queries."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class MeshOcclusion(unittest.TestCase):
    @unittest.skipUnless(shutil.which("cc"), "C compiler required")
    def test_actual_final_draw_query_order(self):
        source = (ROOT / "nxnative/dm/android/gfx_vk_exec.c").read_text(encoding='utf-8')
        start = source.index("    if (ok) { cb = vk_cmd();", source.index("static void mesh_draw(const p_draw *d) {"))
        end = source.index("    if (g_mesh_draws++", start)
        code = r"""
#include <stdint.h>
#include <assert.h>
typedef int VkCommandBuffer;
typedef struct {int b;uint64_t off;} Buffer;
enum {VK_INDEX_TYPE_UINT32=1};
static int allowed,debug,sequence,query,draw;
static int vk_cmd(void){return 42;}
static int draw_allowed(void){return allowed;}
static int dbg_draw(void){return debug;}
static void bc_index(int cb,int b,uint64_t off,int type){assert(cb==42&&b==7&&off==32&&type==1);}
static void vk_occ_draw(void){query=++sequence;}
static void vkCmdDrawIndexedIndirect(int cb,int b,uint64_t off,int count,int stride){
 assert(cb==42&&b==8&&off==80&&count==1&&stride==20);draw=++sequence;assert(query==draw-1);
}
static void run(int ok){
 VkCommandBuffer cb=0;Buffer idx={7,32},launch={8,64};
 query=draw=sequence=0;
""" + source[start:end] + r"""
 assert((query!=0)==(ok&&allowed&&!(debug&1)));assert((draw!=0)==(query!=0));
}
int main(void){for(allowed=0;allowed<2;allowed++)for(debug=0;debug<2;debug++){run(0);run(1);}return 0;}
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
