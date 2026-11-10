"""Actual bounded per-draw capture state and emitted-draw hook coverage."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from test_vk_compute_trace import extract
ROOT=Path(__file__).resolve().parents[1]

@unittest.skipUnless(shutil.which("cc"),"host compiler required")
class DrawCheckpoints(unittest.TestCase):
    def test_actual_checkpoint_helpers_and_fence_window(self):
        s=(ROOT/"nxnative/dm/android/gfx_vk_exec.c").read_text(encoding='utf-8')
        a=s.index("// Deliberately intrusive diagnostic:")
        helpers=s[a:s.index("static void pass_dump_draw_geometry",a)]
        poll=extract(s,"void vk_pass_dump_poll(")
        poll=poll[:poll.index('    const char *root = getenv("DM_ROOT");')]+"    completed++;\\n}".replace("\\n","\n")
        code=r"""
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
typedef void *VkImage;typedef void *VkCommandBuffer;
enum {PD_MAX=96,PD_DRAW_MAX=256,VK_FORMAT_B10G11R11_UFLOAT_PACK32=122};
typedef struct {int set_levels,base_level,set_layers,min_layer;} view_t;
static struct {uint64_t key;} shader={123};
static struct {int pass_open,ncol,rt_has_col[1],rast_discard,rt_has_cview[1];
 view_t rt_cview[1];struct {uint64_t serial;} rt_col[1];typeof(shader)*sh[2];} S;
static struct {VkImage img;} image={(void*)100};
static struct {typeof(image)*im;uint32_t w,h,fmt;} P_att[1]={{&image,128,128,122}};
static uint64_t g_tex_frame=100,g_set_epoch=4;static int g_sub_pass=2,g_draw_i=1;
static int g_pd_state=1,g_npd,g_pd_frames,ends,begins,completed;
static struct {uint64_t fence;} g_pd[PD_DRAW_MAX];
#define VKLOG(...) do {if(0)fprintf(stderr,__VA_ARGS__);}while(0)
static void vk_end_pass(void);
HELPERS
static uint64_t mtl_last_fence(void){return 10;}
static void vk_end_pass(void){
 assert(S.pass_open==1&&g_pd_draw_emit&&pass_dump_draw_ready());
 ends++;g_pd[g_npd++].fence=mtl_last_fence()+1;S.pass_open=0;
}
static int begin_pass(void){begins++;S.pass_open=2;return 1;}
static int pass_commit(void){assert(S.pass_open==2);S.pass_open=1;return 1;}
static VkCommandBuffer vk_cmd(void){return(void*)42;}
INITIAL
static void mrt_poll(uint64_t done){(void)done;}
static void cs_trace_buffers_poll(uint64_t done){(void)done;}
static int pass_dump_start_requested(void){return 0;}
POLL
int main(void){
 S.pass_open=1;S.ncol=1;S.rt_has_col[0]=1;S.rt_col[0].serial=7;S.sh[0]=S.sh[1]=&shader;
 VkCommandBuffer cb=NULL;
 for(int i=0;i<1000;i++)pass_dump_draw_checkpoint("direct",3,1,0,0,0,0);
 assert(!ends&&!g_npd); // off performs no GPU work
 g_pd_draw_mode=1;g_pd_draw_present=101;g_pd_draw_image=image.img;g_pd_draw_serial=7;
 for(int i=0;i<1000;i++)vk_pass_dump_poll(10);
 assert(g_pd_state==1&&!g_npd); // pre-target present never consumes capture
 g_tex_frame=101;
 assert(!pass_dump_draw_match(100,image.img,7,0,0,128,128,122));
 assert(!pass_dump_draw_match(101,(void*)101,7,0,0,128,128,122));
 assert(!pass_dump_draw_match(101,image.img,8,0,0,128,128,122));
 assert(!pass_dump_draw_match(101,image.img,7,1,0,128,128,122));
 assert(!pass_dump_draw_match(101,image.img,7,0,1,128,128,122));
 assert(!pass_dump_draw_match(101,image.img,7,0,0,64,128,122));
 assert(!pass_dump_draw_match(101,image.img,7,0,0,128,128,97));
 assert(pass_dump_draw_initial(&cb)&&cb==(void*)42&&g_npd==1&&begins==1);
 assert(pass_dump_draw_initial(&cb)&&g_npd==1&&begins==1);
 for(int i=0;i<225;i++){S.pass_open=1;g_draw_i++;pass_dump_draw_checkpoint("direct",3,1,0,0,0,0);}
 assert(g_npd==226&&g_pd_draw_ordinal==225&&ends==226);
 S.pass_open=1;S.rast_discard=1;pass_dump_draw_checkpoint("direct",3,1,0,0,0,0);assert(g_npd==226);S.rast_discard=0;
 for(int i=0;i<100;i++){S.pass_open=1;pass_dump_draw_checkpoint("direct",3,1,0,0,0,0);}
 assert(g_npd==256&&g_pd_draw_ordinal==255);
 g_tex_frame=102;assert(!pass_dump_draw_ready());
 vk_pass_dump_poll(10);assert(g_pd_state==2&&!completed);
 vk_pass_dump_poll(11);assert(completed==1); // exact recorded fence owns every copy
 return 0;
}
"""
        code=code.replace("HELPERS",helpers).replace("INITIAL",extract(s,"static int pass_dump_draw_initial(")).replace("POLL",poll)
        target=ROOT/"target/tests";target.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="draw-checkpoints-",dir=target) as tmp:
            c,exe=Path(tmp)/"t.c",Path(tmp)/"t";c.write_text(code, encoding='utf-8')
            subprocess.run(["cc","-std=gnu11","-O1","-g","-Wall","-Wextra","-Werror",
                            "-fsanitize=address,undefined",str(c),"-o",str(exe)],check=True)
            subprocess.run([str(exe)],check=True,timeout=10)

    def test_only_emitted_guest_draws_hook_and_preserve_readback_order(self):
        s=(ROOT/"nxnative/dm/android/gfx_vk_exec.c").read_text(encoding='utf-8')
        for name,count in [("static void draw(const p_draw *d) {",4),
                           ("static void draw_idx(",3),("static void mesh_draw(const p_draw *d) {",1)]:
            f=extract(s,name)
            self.assertEqual(f.count("pass_dump_draw_checkpoint("),count)
            self.assertEqual(f.count("vk_occ_draw();"),count)
            self.assertIn("pass_dump_draw_geometry(",f)
        prepare=extract(s,"static int prepare_draw(")
        self.assertLess(prepare.index("pass_dump_draw_initial("),prepare.index("vkCmdBindPipeline("))
        after=extract(s,"static void pass_dump_draw_checkpoint(")
        self.assertIn("vk_end_pass();",after);self.assertNotIn("vk_end_pass_land",after)
        end=extract(s,"void vk_end_pass(")
        self.assertLess(end.index("vkCmdEndRenderPass"),end.index("pass_dump();"))
        dump=extract(s,"static void pass_dump(")
        self.assertIn("d->fence = mtl_last_fence() + 1",dump)
        self.assertLess(dump.index("vkCmdCopyImageToBuffer"),dump.index("VK_ACCESS_HOST_READ_BIT"))
        self.assertIn("g_pd_draw_mode ? PD_DRAW_MAX : PD_MAX",dump)

if __name__=="__main__":unittest.main()
