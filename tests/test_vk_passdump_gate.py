"""Actual pass-dump arming helpers: startup key gate and allocation-free menu wait."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from test_vk_compute_trace import extract
import sys; sys.path.insert(0, str(Path(__file__).resolve().parent))
import host_cc   # noqa: E402
ROOT=Path(__file__).resolve().parents[1]
@unittest.skipUnless(shutil.which("cc"),"host compiler required")
class PassDumpGate(unittest.TestCase):
    @unittest.skipUnless(host_cc.posix() and host_cc.sanitizers(), 'a POSIX host C compiler with ASan and UBSan is required (MinGW has no setenv, libasan or libubsan)')
    def test_actual_gate(self):
        s=(ROOT/"nxnative/dm/android/gfx_vk_exec.c").read_text(encoding='utf-8')
        a=s.index("static int g_pd_fs_mode")
        actual=s[a:s.index("void vk_pass_dump_poll(",a)]
        code=r"""
#define _POSIX_C_SOURCE 200809L
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
typedef unsigned VkFormat; typedef void *VkImage;
typedef struct {int set_levels,base_level,set_layers,min_layer;} view_t;
static struct {VkImage img;} image;
static struct {typeof(image)*im;} P_att[1]={{&image}};
static struct {int rt_has_col[1],rt_has_cview[1];view_t rt_cview[1];struct {uint64_t serial;} rt_col[1];} S={{1},{0},{{0}},{{1}}};
static int g_pd_draw_mode,g_pd_draw_emit,g_pd_draw_before;static unsigned g_pd_draw_ordinal;
static uint64_t g_pd_draw_present,g_pd_draw_serial,g_tex_frame;static VkImage g_pd_draw_image;
enum {PD_DRAW_MAX=256};
enum { VK_FORMAT_B10G11R11_UFLOAT_PACK32=122 };
static int g_pd_state,g_pd_skip,g_pd_frames,g_npd,g_pd_strict;
static void pass_journal_arm(void){}
static unsigned char g_pd[96];
static char g_pd_from[32],g_pd_size[128];
static uint64_t g_td_pool;static int64_t g_td_off;static void *g_td_im;static int g_td_done;
#define VKLOG(...) do { if(0)fprintf(stderr,__VA_ARGS__); } while(0)
""" + actual + r"""
static void hit(void){pass_dump_arm_for_draw(0xabc,128,128,122,6,0);}
int main(void) {
 uint64_t key;
 const char *bad[]={"-abc","+abc"," abc","abc ","abc:1","12345678901234567","0x","g"};
 for(unsigned i=0;i<sizeof bad/sizeof *bad;i++)assert(pass_dump_fs_parse(bad[i],&key)==-1&&!key);
 assert(pass_dump_fs_parse(NULL,&key)==0&&!key);
 assert(pass_dump_fs_parse("0xabc",&key)==1&&key==0xabc);
 setenv("DM_VK_PASSDUMP_FS","abc",1);setenv("DM_VK_PASSDUMP","1",1);
 setenv("DM_VK_PASSDUMP_FRAMES","2",1);memset(g_pd,0xa5,sizeof g_pd);
 for(int i=0;i<10000;i++){assert(pass_dump_start_requested());pass_dump_arm_for_draw(0xdef,128,128,122,6,0);}
 assert(!g_pd_state&&!g_npd&&g_pd[0]==0xa5&&!strcmp(getenv("DM_VK_PASSDUMP"),"1"));
 pass_dump_arm_for_draw(0xabc,64,64,122,6,0);pass_dump_arm_for_draw(0xabc,128,64,122,6,0);
 pass_dump_arm_for_draw(0xabc,128,128,97,6,0);pass_dump_arm_for_draw(0xabc,128,128,122,1,0);
 pass_dump_arm_for_draw(0xabc,128,128,122,6,1);assert(!g_pd_state);
 setenv("DM_VK_PASSDUMP","2",1);hit();assert(!g_pd_state);assert(pass_dump_start_requested()&&!strcmp(getenv("DM_VK_PASSDUMP"),"2"));
 setenv("DM_VK_PASSDUMP","1",1);hit();assert(g_pd_state==1&&g_pd_frames==2&&!g_npd&&!g_pd[0]&&!strcmp(g_pd_size,"128x128")&&!strcmp(getenv("DM_VK_PASSDUMP"),"0"));
 g_npd=10;hit();assert(g_npd==10&&!pass_dump_start_requested());g_pd_state=2;hit();assert(g_npd==10&&!pass_dump_start_requested());
 g_pd_state=0;setenv("DM_VK_PASSDUMP_FS","def",1);setenv("DM_VK_PASSDUMP","1",1);setenv("DM_VK_PASSDUMP_FRAMES","999",1);
 pass_dump_arm_for_draw(0xdef,128,128,122,6,0);assert(!g_pd_state);hit();assert(g_pd_state==1&&g_pd_frames==3); // startup key stays abc
 g_pd_state=0;g_tex_frame=100;setenv("DM_VK_PASSDUMP_DRAWS","1",1);setenv("DM_VK_PASSDUMP","1",1);
 S.rt_has_cview[0]=1;S.rt_cview[0].set_layers=1;S.rt_cview[0].min_layer=1;hit();assert(!g_pd_state);
 S.rt_cview[0].min_layer=0;hit();assert(g_pd_state==1&&g_pd_draw_mode&&g_pd_draw_present==101&&g_pd_draw_serial==1);
 unsetenv("DM_VK_PASSDUMP_DRAWS");
 g_pd_state=0;g_pd_fs_mode=-2;setenv("DM_VK_PASSDUMP_FS","bad:",1);setenv("DM_VK_PASSDUMP","1",1);assert(pass_dump_start_requested());hit();assert(!g_pd_state);
 g_pd_fs_mode=-2;unsetenv("DM_VK_PASSDUMP_FS");setenv("DM_VK_PASSDUMP","3",1);
 assert(pass_dump_start_requested()&&!strcmp(getenv("DM_VK_PASSDUMP"),"2")&&!g_pd_state);
 assert(pass_dump_start_requested()&&!strcmp(getenv("DM_VK_PASSDUMP"),"1")&&!g_pd_state);
 assert(pass_dump_start_requested()&&g_pd_state==1&&g_pd_frames==999); // normal behavior retained
 return 0;
}
"""
        target=ROOT/"target/tests";target.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="passdump-gate-",dir=target) as temp:
            c,exe=Path(temp)/"t.c",Path(temp)/"t";c.write_text(code, encoding='utf-8')
            subprocess.run(["cc","-std=gnu11","-O1","-g","-Wall","-Wextra","-Werror","-fsanitize=address,undefined",str(c),"-o",str(exe)],check=True)
            subprocess.run([str(exe)],check=True,timeout=10)
        prepare=extract(s,"static int prepare_draw(")
        self.assertGreater(prepare.index("pass_dump_arm_for_draw("),prepare.index("fs_buffers_before_draw("))
        self.assertGreater(prepare.index("pass_dump_arm_for_draw("),prepare.index("vkCmdBindPipeline("))
        dump=extract(s,"static void pass_dump(")
        self.assertGreater(dump.index('"%ux%u", w, h'),dump.index("uint32_t lv ="))
        self.assertNotIn("g_tracing",actual)
if __name__=="__main__":unittest.main()
