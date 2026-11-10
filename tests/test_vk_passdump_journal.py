"""Actual bounded pass journal and strict cube selection, including mesh metadata."""
from pathlib import Path
import shutil, subprocess, tempfile, unittest
from test_vk_compute_trace import extract
ROOT=Path(__file__).resolve().parents[1]
@unittest.skipUnless(shutil.which("cc"),"host compiler required")
class PassJournal(unittest.TestCase):
 def test_actual_helpers(self):
  s=(ROOT/"nxnative/dm/android/gfx_vk_exec.c").read_text(encoding='utf-8')
  a=s.index("enum { PD_JOURNAL_MAX")
  actual=s[a:s.index("static void pass_dump_depth(void);",a)]
  # The intrusive draw capture is a separate subsystem with its own actual-helper tests.
  x=actual.index("// Deliberately intrusive diagnostic:")
  actual=actual[:x]+actual[actual.index("static int pass_dump_cube_match",x):]
  code=r"""
#define _POSIX_C_SOURCE 200809L
#include <vulkan/vulkan_core.h>
#include <assert.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
typedef struct {VkImage img;uint64_t pool;int64_t off;VkFormat fmt;} vkimg_t;
typedef struct {int set_levels,set_layers;uint32_t base_level,min_layer;} view_t;
typedef struct {uint64_t serial;} tex_t;
typedef struct {uint64_t key;} shader_t;
static struct {int pass_open,rp_pass,ncol,rt_has_cview[8],rt_has_dview,rast_discard;uint32_t pass_w,pass_h;tex_t rt_col[8],rt_dep;view_t rt_cview[8],rt_dview;VkPipeline bound_pipe;shader_t *sh[3];} S;
static struct {vkimg_t *im;VkImageView v;uint32_t fmt,layers,w,h;} P_att[9];
static VkRenderingAttachmentInfo P_ca[8],P_da,P_sa;
static VkRenderingInfo P_ri;
static int P_split,g_pd_state,g_pd_frames,g_sub_pass,g_draw_i;
static uint64_t g_tex_frame,g_set_epoch,g_pass_draws;
static char logs[32768];static unsigned nlog;
static void logmsg(const char *fmt,...) {va_list ap;va_start(ap,fmt);size_t n=strlen(logs);vsnprintf(logs+n,sizeof logs-n,fmt,ap);va_end(ap);nlog++;}
#define VKLOG(...) logmsg(__VA_ARGS__)
""" + actual + r"""
int main(void){
 assert(pass_dump_cube_match(0,64,32,97,97,1)); // legacy capture unchanged
 assert(pass_dump_cube_match(1,128,128,122,122,6));
 assert(pass_dump_cube_match(1,128,128,122,122,12));
 assert(!pass_dump_cube_match(1,128,128,122,122,1));
 assert(!pass_dump_cube_match(1,64,64,122,122,6)); // selected mip, not base extent
 assert(!pass_dump_cube_match(1,128,128,97,122,6));
 assert(!pass_dump_cube_match(1,128,128,122,97,6));
 g_pd_state=1;g_pd_frames=3;g_tex_frame=100;g_pd_strict=1;
 pass_journal_arm();for(int i=0;i<10000;i++)assert(!pass_journal_row());assert(!nlog);
 setenv("DM_VK_PASSDUMP_JOURNAL","01",1);g_pd_strict=0;pass_journal_arm();assert(!g_pd_journal&&!nlog);
 setenv("DM_VK_PASSDUMP_JOURNAL","1",1);g_pd_strict=0;pass_journal_arm();assert(g_pd_journal&&nlog==1&&pass_journal_row()); // ordinary capture opt-in
 unsetenv("DM_VK_PASSDUMP_JOURNAL");pass_journal_arm();assert(!g_pd_journal);setenv("DM_VK_PASSDUMP_JOURNAL","1",1);
 g_pd_strict=1;g_pd_state=0;pass_journal_arm();assert(!pass_journal_row());
 logs[0]=0;nlog=0;g_pd_state=1;S.pass_open=1;S.rp_pass=1;S.ncol=1;S.pass_w=S.pass_h=128;
 vkimg_t im={(VkImage)(uintptr_t)0x1234,0x2000,0x3000,122};
 P_att[0].im=&im;P_att[0].fmt=122;P_att[0].layers=1;P_att[0].w=P_att[0].h=128;
 P_ca[0].imageView=(VkImageView)(uintptr_t)0x4567;P_ca[0].loadOp=VK_ATTACHMENT_LOAD_OP_CLEAR;P_ca[0].storeOp=VK_ATTACHMENT_STORE_OP_STORE;
 P_ca[0].clearValue.color.uint32[0]=0x3f800000;S.rt_col[0].serial=99;S.rt_has_cview[0]=1;S.rt_cview[0]=(view_t){1,1,2,5};
 P_ri.layerCount=1;P_ri.colorAttachmentCount=1;g_set_epoch=42;g_sub_pass=7;g_draw_i=8;
 pass_journal_arm();assert(g_pd_journal&&g_pd_journal_first==100&&g_pd_journal_end==103);
 assert(strstr(logs,"event=active-at-arm")&&strstr(logs,"serial=99")&&strstr(logs,"mip=2 layer=5")&&strstr(logs,"clear=3f800000"));
 shader_t fs={0xabc},gs={0xdef};S.sh[1]=&fs;S.sh[2]=&gs;S.bound_pipe=(VkPipeline)(uintptr_t)0x9876;
 pass_journal_draw("mesh",12,1,0,0x111,0x222,0x333);
 assert(strstr(logs,"kind=mesh")&&strstr(logs,"task=0000000000000222 mesh=0000000000000333")&&strstr(logs,"count=12"));
 assert(strstr(logs,"vs=0000000000000111 fs=0000000000000abc"));
 S.sh[1]=NULL; // a subsequent program bind must not relabel the copied pass
 pass_journal_copy(7,0,&im,97,2,5,1440,810);
 assert(strstr(logs,"copy index=7 present=100 submission=42 pass=6 slot=0 serial=99"));
 assert(strstr(logs,"native=122 view_format=97 mip=2 layer=5 extent=1440x810 last_prepared_fs=0000000000000abc"));
 pass_journal_pass("begin");assert(!g_pd_journal_last_fs); // empty next pass has no inherited writer

 VkClearAttachment ca={.aspectMask=1,.colorAttachment=0};VkClearRect cr={.rect={{2,3},{4,5}},.baseArrayLayer=1,.layerCount=6};
 ca.clearValue.color.uint32[0]=0x7fc00001;pass_journal_clear("colour",&ca,&cr,15);assert(strstr(logs,"rect=2,3,4,5 layer=1 layers=6 value=7fc00001"));
 pass_journal_pass("begin");pass_journal_pass("end");assert(strstr(logs,"event=begin")&&strstr(logs,"event=end"));
 unsigned before=nlog;g_tex_frame=99;assert(!pass_journal_row());g_tex_frame=103;pass_journal_draw("indexed",1,1,0,0,0,0);assert(nlog==before);
 g_tex_frame=102;assert(pass_journal_row());g_pd_state=2;assert(!pass_journal_row());g_pd_state=1;
 before=nlog;g_pd_state=2;pass_journal_copy(8,0,&im,122,0,0,1440,810);assert(nlog==before);g_pd_state=1;

 g_pd_journal_rows=PD_JOURNAL_MAX;assert(!pass_journal_row());assert(nlog==before+1);assert(!pass_journal_row());assert(nlog==before+1);
 g_tex_frame=UINT64_MAX;pass_journal_arm();assert(!g_pd_journal);return 0;
}
"""
  target=ROOT/"target/tests";target.mkdir(parents=True,exist_ok=True)
  with tempfile.TemporaryDirectory(prefix="pass-journal-",dir=target) as temp:
   c,exe=Path(temp)/"t.c",Path(temp)/"t";c.write_text(code, encoding='utf-8')
   subprocess.run(["cc","-std=c11","-O1","-g","-Wall","-Wextra","-Werror","-fsanitize=address,undefined","-I"+str(ROOT/"nxnative/dm/third_party/vulkan-headers/include"),str(c),"-o",str(exe)],check=True)
   subprocess.run([str(exe)],check=True,timeout=10)
  self.assertNotIn("vkCmd",actual);self.assertNotIn("vk_tex(",actual);self.assertNotIn("g_tracing",actual)
 def test_recording_hooks(self):
  s=(ROOT/"nxnative/dm/android/gfx_vk_exec.c").read_text(encoding='utf-8')
  for name in ("draw","draw_idx"):
   f=extract(s,"static void "+name+("(const p_draw *d) {" if name=="draw" else "(const p_draw_idx *d) {"))
   self.assertGreater(f.index("pass_journal_draw("),f.index("if (!prepare_draw("))
   self.assertLess(f.index("pass_journal_draw("),f.index("draw_allowed()"))
  f=extract(s,"static void mesh_draw_one(const p_draw *d, uint32_t batch) {")
  self.assertIn('if (ok) pass_journal_draw("mesh", count, 1, d->ind, vs->key, ts ? ts->key : 0, ms->key)',f)
  self.assertLess(f.index("mesh-emitted"),f.index("vkCmdDrawIndexedIndirect"))
  end=extract(s,"void vk_end_pass(")
  self.assertLess(end.index('pass_journal_pass("end")'),end.index("vkCmdEndRenderPass"))
  begin=extract(s,"static int pass_commit(")
  self.assertGreater(begin.index('pass_journal_pass("begin")'),begin.index("g_sub_pass++"))
  dump=extract(s,"static void pass_dump(")
  self.assertLess(dump.index("pass_dump_cube_match("),dump.index("vkCreateBuffer("))
  self.assertIn("g_pd_strict, w, h",dump)
if __name__=="__main__":unittest.main()

