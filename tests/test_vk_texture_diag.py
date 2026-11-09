"""Exercise bounded generation observations without touching rendering or registry state."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(shutil.which("cc"), "host C compiler required")
class TextureDiagnostics(unittest.TestCase):
    def test_bounds_causes_fields_and_concurrent_registration(self):
        code = r"""
#define _POSIX_C_SOURCE 200809L
#include <assert.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include "nvn_obj.h"
typedef struct {void *img;uint64_t pool,off;unsigned nvn_format,w,h,levels,pins;} vkimg_t;
static uint64_t g_tex_gen=20;
static char messages[256][2048];static unsigned notes;
static void note(const char *fmt,...){
 assert(notes<256);va_list ap;va_start(ap,fmt);vsnprintf(messages[notes++],sizeof messages[0],fmt,ap);va_end(ap);
}
#define VKLOG(...) note(__VA_ARGS__)
uint64_t nvn_jobs_queued(void){return 13;}
uint64_t nvn_replaying_job(void){return 11;}
#include "gfx_vk_texture_diag.h"
static vkimg_t image={.img=(void *)(uintptr_t)0x100,.pool=0x200,.off=16,.nvn_format=0x38,.w=64,.h=32,.levels=1};
static tex_t a={.serial=5,.d={.flags=0,.format=0x38,.target=1,.width=64,.height=32,.depth=1,.levels=1}};
static void *worker(void *unused){
 (void)unused;for(int i=0;i<1000;i++)vk_tex_diag_native(TEX_DIAG_FIRST_USE,&a,&image);return NULL;
}
int main(void){
 unsetenv("DM_VK_TEX_DIAG");vk_tex_diag_poll();
 vk_tex_diag_native(TEX_DIAG_FIRST_USE,&a,&image);
 vk_tex_diag_registration(7,&a,NULL,&a,NULL,10,12,99);
 assert(!notes&&!g_tex_diag_causes[0]&&!g_tex_diag_active);
 const char *bad[]={"-1","nonsense","1x",""};
 for(unsigned i=0;i<sizeof bad/sizeof *bad;i++){setenv("DM_VK_TEX_DIAG",bad[i],1);vk_tex_diag_poll();assert(!g_tex_diag_active);}
 setenv("DM_VK_TEX_DIAG","2",1);vk_tex_diag_poll();
 assert(g_tex_diag_active&&g_tex_diag_frames==64&&g_tex_diag_left==2&&!strcmp(getenv("DM_VK_TEX_DIAG"),"0"));
 vk_tex_diag_native(TEX_DIAG_FIRST_USE,&a,&image);
 tex_t b=a;b.serial=6;b.d.flags=4;b.off=32;b.L.size=1024;
 view_t av={0},bv={.set_format=1,.format=0xb,.set_levels=1,.base_level=2,.nlevels=1,.min_layer=3,.nlayers=1};
 vk_tex_diag_registration(133,&a,&av,&b,&bv,10,12,99);
 assert(notes==5&&strstr(messages[2],"registration id 133")&&strstr(messages[2],"registration 99"));
 assert(strstr(messages[3],"serial 5->6")&&strstr(messages[3],"flags 0->0x4")&&strstr(messages[4],"format 0->0xb"));
 vk_tex_diag_native(TEX_DIAG_VIEW_DROP,NULL,&image);
 vk_tex_diag_native(TEX_DIAG_ORPHAN,NULL,&image);
 assert(notes==5&&g_tex_diag_causes[0]==1&&g_tex_diag_causes[1]==1&&g_tex_diag_causes[2]==1&&g_tex_diag_causes[3]==1);
 for(int i=0;i<63;i++)vk_tex_diag_poll();
 assert(g_tex_diag_active);g_tex_gen=25;vk_tex_diag_poll();assert(!g_tex_diag_active&&notes==6);
 assert(strstr(messages[5],"gen 20->25")&&strstr(messages[5],"registration 1")&&strstr(messages[5],"orphan 1"));
 // A new request resets only diagnostics; immutable caller objects remain unchanged.
 assert(a.serial==5&&a.d.flags==0&&b.serial==6&&b.off==32);
 setenv("DM_VK_TEX_DIAG","999999999999999999999999999",1);vk_tex_diag_poll();
 assert(g_tex_diag_left==64);unsigned start=notes;
 pthread_t threads[8];for(unsigned i=0;i<8;i++)assert(!pthread_create(&threads[i],NULL,worker,NULL));
 for(unsigned i=0;i<8;i++)assert(!pthread_join(threads[i],NULL));
 assert(g_tex_diag_causes[TEX_DIAG_FIRST_USE]==8000&&g_tex_diag_left==0&&notes-start==64);
 for(int i=0;i<64;i++)vk_tex_diag_poll();assert(!g_tex_diag_active);
 start=notes;vk_tex_diag_native(TEX_DIAG_FIRST_USE,&a,&image);assert(notes==start&&g_tex_diag_causes[1]==8000);
 setenv("DM_VK_TEX_DIAG","1",1);vk_tex_diag_poll();
 vk_tex_diag_registration(9,&a,NULL,NULL,NULL,12,13,100);
 assert(strstr(messages[notes-3],"present 1->0"));for(int i=0;i<64;i++)vk_tex_diag_poll();
 puts("texture diagnostics: inactive guard, bounded events/window, actual differences and concurrent counters passed");
}
"""
        target = ROOT / "target/tests"
        target.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="texture-diag-", dir=target) as directory:
            source, exe = Path(directory) / "test.c", Path(directory) / "test"
            source.write_text(code)
            subprocess.run(["cc", "-std=c11", "-O1", "-g", "-Wall", "-Wextra", "-Werror",
                            "-Wno-misleading-indentation", "-fsanitize=address,undefined", "-pthread",
                            "-I" + str(ROOT / "nxnative/dm/gfx/nvn"),
                            "-I" + str(ROOT / "nxnative/dm/gfx/nvn/include"),
                            "-I" + str(ROOT / "nxnative/dm/android"), str(source), "-o", str(exe)], check=True)
            subprocess.run([str(exe)], check=True, timeout=15)

    def test_every_generation_writer_and_identical_registration_guard(self):
        source = (ROOT / "nxnative/dm/android/gfx_vk_tex.c").read_text()
        self.assertEqual(source.count("g_tex_gen++"), 4)
        for cause in ("TEX_DIAG_FIRST_USE", "TEX_DIAG_VIEW_DROP", "TEX_DIAG_ORPHAN"):
            self.assertEqual(source.count("vk_tex_diag_native(" + cause), 1)
        reg = source[source.index("void mtl_reg_tex("):source.index("void mtl_set_tex_pool_n(")]
        self.assertLess(reg.index("pthread_mutex_unlock(&g_rmu); return;"), reg.index("vk_tex_diag_registration("))
        self.assertIn("r->cur.seq, seq, g_texture_registrations", reg)
        self.assertIn("void vk_tex_sweep(void) {\n    vk_tex_diag_poll();", source)


if __name__ == "__main__":
    unittest.main()
