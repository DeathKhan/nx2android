"""Actual-source graphics GPU snapshot selection, ranges, and render-pass ordering."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from test_vk_compute_trace import extract
ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/"nxnative/dm/android/gfx_vk_exec.c"
@unittest.skipUnless(shutil.which("cc"),"host compiler required")
class GraphicsBuffers(unittest.TestCase):
    def compile_run(self,code):
        target=ROOT/"target/tests";target.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="graphics-buffers-",dir=target) as folder:
            c,exe=Path(folder)/"test.c",Path(folder)/"test";c.write_text(code, encoding='utf-8')
            subprocess.run(["cc","-std=c11","-O1","-g","-Wall","-Wextra","-Werror","-fsanitize=address,undefined",str(c),"-o",str(exe)],check=True)
            subprocess.run([str(exe)],check=True,timeout=10)
    def test_actual_selection_and_windows(self):
        s=SRC.read_text(encoding='utf-8');a=s.index("static uint64_t g_fs_buffer_key")
        actual=s[a:s.index("\nstatic void x_read(",a)]
        self.compile_run(r"""
#define _POSIX_C_SOURCE 200809L
#include <stdint.h>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static uint64_t g_tex_frame=100;
#define VKLOG(...) do{if(0)fprintf(stderr,__VA_ARGS__);}while(0)
"""+actual+r"""
static void arm(const char *key){setenv("DM_VK_TRACE_FS_BUFFERS",key,1);fs_buffer_poll();}
int main(void){
 (void)g_fs_buffer_collect;(void)g_draw_i;(void)g_fs_buffer_mesh;(void)g_fs_image_collect;
 fs_buffer_poll();assert(!g_fs_buffer_key&&!fs_buffer_match(1,128,128,75,0));
 const char *bad[]={"-1","+1"," a","g","12345678901234567","0x","0"};
 for(unsigned i=0;i<sizeof bad/sizeof *bad;i++){arm(bad[i]);assert(!g_fs_buffer_key);}
 arm("0xabc");assert(g_fs_buffer_key==0xabc&&g_fs_buffer_frames==3&&!strcmp(getenv("DM_VK_TRACE_FS_BUFFERS"),"0"));
 assert(!fs_buffer_match(0xabd,128,128,75,0)&&!fs_buffer_match(0xabc,127,128,75,0));
 assert(fs_buffer_match(0xabc,128,128,75,0)&&!fs_buffer_match(0xabc,128,128,75,0));
 g_tex_frame++;assert(fs_buffer_match(0xabc,128,128,75,0));g_tex_frame++;assert(fs_buffer_match(0xabc,128,128,75,0));
 g_tex_frame++;assert(!fs_buffer_match(0xabc,128,128,75,0));fs_buffer_poll();assert(!g_fs_buffer_key);
 setenv("DM_VK_TRACE_FS_FRAMES","999",1);arm("abc");assert(g_fs_buffer_frames==3);
 g_fs_buffer_copies=64;assert(!fs_buffer_match(0xabc,128,128,75,0));
 arm("abc");for(int i=0;i<10000;i++){g_tex_frame++;fs_buffer_poll();assert(!fs_buffer_match(0xdef,128,128,75,0));}
 assert(g_fs_buffer_key==0xabc&&!g_fs_buffer_seen&&!g_fs_buffer_copies);
 assert(fs_buffer_match(0xabc,128,128,75,0));uint64_t first=g_tex_frame;
 for(int i=0;i<10000;i++){fs_buffer_poll();}assert(g_fs_buffer_key==0xabc&&g_tex_frame==first);
 g_tex_frame=first+2;fs_buffer_poll();assert(g_fs_buffer_key==0xabc);
 g_tex_frame=first+3;fs_buffer_poll();assert(!g_fs_buffer_key);
 setenv("DM_VK_TRACE_FS_SIZE","0x128",1);arm("abc");assert(!g_fs_buffer_key);
 setenv("DM_VK_TRACE_FS_SIZE","16385x128",1);arm("abc");assert(!g_fs_buffer_key);
 setenv("DM_VK_TRACE_FS_SIZE","128x128junk",1);arm("abc");assert(!g_fs_buffer_key);
 setenv("DM_VK_TRACE_FS_SIZE","256x64",1);setenv("DM_VK_TRACE_FS_FRAMES","1",1);
 setenv("DM_VK_TRACE_FS_WINDOWS","1:1:2f4070:10,1:1:34060:20",1);arm("abc");
 assert(g_fs_buffer_key&&g_fs_buffer_windows_n==2&&g_fs_buffer_windows[0].offset==0x2f4070&&g_fs_buffer_windows[1].bytes==32);
 assert(fs_buffer_match(0xabc,256,64,75,0));g_tex_frame++;assert(!fs_buffer_match(0xabc,256,64,75,0));
 const char *badw[]={"3:1:0:10","1:16:0:10","1:1:0:0","1:1:0:10004","1:1:1:10","1:1:0:3","1:1:-4:10","1:1:0:10,",
  "1:1:fffffffffffffffff:10","1:1:0:10:2","1:1:0:4,1:1:0:4,1:1:0:4,1:1:0:4,1:1:0:4,1:1:0:4,1:1:0:4,1:1:0:4,1:1:0:4"};
 for(unsigned i=0;i<sizeof badw/sizeof *badw;i++){setenv("DM_VK_TRACE_FS_WINDOWS",badw[i],1);arm("abc");assert(!g_fs_buffer_key&&!g_fs_buffer_windows_n);}
 unsetenv("DM_VK_TRACE_FS_WINDOWS");setenv("DM_VK_TRACE_FS_SIZE","128x128",1);
 const char *badc[]={"-1","+396"," 396","396 ","396x","4294967296","18446744073709551616"};
 for(unsigned i=0;i<sizeof badc/sizeof *badc;i++){setenv("DM_VK_TRACE_FS_COUNT",badc[i],1);arm("abc");assert(!g_fs_buffer_key);}
 setenv("DM_VK_TRACE_FS_COUNT","396",1);arm("abc");assert(g_fs_buffer_count==396);
 for(int i=0;i<1000;i++){assert(!fs_buffer_match(0xabc,128,128,75,0));assert(!fs_buffer_match(0xabc,128,128,849,0));}
 assert(!g_fs_buffer_seen&&g_fs_buffer_last==UINT64_MAX);
 assert(!fs_buffer_match(0xabc,128,128,396,1)); // indirect args and mesh groups cannot masquerade as direct count
 assert(fs_buffer_match(0xabc,128,128,396,0)&&g_fs_buffer_seen==1);
 assert(!fs_buffer_match(0xabc,128,128,396,0)); // once per present remains
 setenv("DM_VK_TRACE_FS_COUNT","0",1);arm("abc");assert(!g_fs_buffer_count&&fs_buffer_match(0xabc,128,128,849,1));
 unsetenv("DM_VK_TRACE_FS_COUNT");arm("abc");assert(!g_fs_buffer_count&&fs_buffer_match(0xabc,128,128,75,0));
 const char *badmax[]={"-1","8M","65535","65537","8388609","18446744073709551616"};
 for(unsigned i=0;i<sizeof badmax/sizeof *badmax;i++){setenv("DM_VK_TRACE_FS_SBUF_MAX",badmax[i],1);arm("abc");assert(!g_fs_buffer_key);}
 setenv("DM_VK_TRACE_FS_SBUF_MAX","8388608",1);arm("abc");assert(g_fs_buffer_key&&g_fs_buffer_sbuf_max==8388608&&!g_fs_buffer_bytes);
 setenv("DM_VK_TRACE_FS_SBUF_MAX","0",1);arm("abc");assert(g_fs_buffer_key&&g_fs_buffer_sbuf_max==65536);
 unsetenv("DM_VK_TRACE_FS_SBUF_MAX");arm("abc");assert(g_fs_buffer_sbuf_max==65536);
 setenv("DM_VK_TRACE_FS_IMAGE_BINDING","1",1);arm("abc");assert(!g_fs_buffer_key); // requires exactcount
 setenv("DM_VK_TRACE_FS_COUNT","396",1);arm("abc");assert(g_fs_buffer_key&&g_fs_image_binding==1&&!g_fs_image_copies&&!g_fs_image_bytes);
 const char *badi[]={"-1","+1","1x","65536","18446744073709551616"};
 for(unsigned i=0;i<sizeof badi/sizeof *badi;i++){setenv("DM_VK_TRACE_FS_IMAGE_BINDING",badi[i],1);arm("abc");assert(!g_fs_buffer_key&&g_fs_image_binding==-1);}
 unsetenv("DM_VK_TRACE_FS_IMAGE_BINDING");arm("abc");assert(g_fs_buffer_key&&g_fs_image_binding==-1);

 return 0;
}
""")
    def test_actual_copy_outside_pass_and_resume(self):
        s=SRC.read_text(encoding='utf-8');actual=extract(s,"static int fs_buffers_before_draw(")
        self.compile_run(r"""
#include <stdint.h>
#include <assert.h>
typedef uintptr_t VkCommandBuffer;
static unsigned g_cs_buffers_n,g_cs_buffers_reserved,g_cs_images_n,g_cs_images_copies;
static int active=1,phase,fail_begin,fail_commit,copies,imagecopies;
static void vk_end_pass(void){assert(active);active=0;phase=1;}
static VkCommandBuffer vk_cmd(void){return (VkCommandBuffer)(100+phase);}
static void cs_trace_buffers_copy(VkCommandBuffer cb,uint64_t key,int after){assert(!active&&phase==1&&cb==101&&key==9&&!after);phase=2;copies++;}
static void cs_trace_images_copy(VkCommandBuffer cb,uint64_t key,int after){assert(!active&&phase==2&&cb==101&&key==9&&!after);imagecopies++;}
static int begin_pass(void){assert(!active&&phase==2);phase=3;return !fail_begin;}
static int pass_commit(void){assert(!active&&phase==3);phase=4;active=!fail_commit;return active;}
"""+actual+r"""
int main(void){
 VkCommandBuffer cb=100;assert(fs_buffers_before_draw(9,&cb)&&cb==100&&!copies);
 g_cs_buffers_n=3;g_cs_buffers_reserved=3;assert(fs_buffers_before_draw(9,&cb));assert(active&&copies==1&&cb==104&&!g_cs_buffers_n&&!g_cs_buffers_reserved);
 phase=0;g_cs_buffers_n=1;fail_begin=1;assert(!fs_buffers_before_draw(9,&cb));assert(!active&&copies==2&&!g_cs_buffers_n);
 active=1;phase=0;g_cs_buffers_n=1;fail_begin=0;fail_commit=1;assert(!fs_buffers_before_draw(9,&cb));assert(!active&&copies==3);
 active=1;phase=0;fail_commit=0;g_cs_images_n=g_cs_images_copies=1;assert(fs_buffers_before_draw(9,&cb));
 assert(imagecopies==1&&!g_cs_images_n&&!g_cs_images_copies&&active);
 return 0;
}
""")
        stage=extract(s,"static VkDescriptorSet stage_set(")
        self.assertIn("!g_fs_buffer_collect",stage)
        self.assertLess(stage.index("cs_trace_buffer_note(st, s->key, b, &bi[nw])"),stage.index("if (s->dyn[i])"))
        draw=extract(s,"static int prepare_draw(")
        self.assertLess(draw.index("if (!p)"),draw.index("fs_buffers_before_draw"))
        self.assertLess(draw.index("fs_buffers_before_draw"),draw.index("vkCmdBindPipeline"))
        self.assertLess(draw.index("g_fs_buffer_collect = 0"),draw.index("tbl_set("))
        self.assertNotIn("g_tracing",extract(s,"static int fs_buffers_before_draw("))
        self.assertIn("cs_trace_images_copy(*cb, key, 0)",extract(s,"static int fs_buffers_before_draw("))
        note=extract(s,"static void fs_trace_texture_note(")
        self.assertIn("st != 1",note);self.assertIn("b->binding != g_fs_image_binding",note)
        self.assertNotIn("vk_tex_view(",note);self.assertNotIn("g_tvc[",note)
        self.assertLess(stage.index("tex_desc_for(st, b, h, &ii[nii + k], 0)"),stage.index("fs_trace_texture_note("))
        self.assertIn("g_claim_draw.count, g_claim_draw.indirect || g_fs_buffer_mesh",draw)
        mesh=extract(s,"static void mesh_draw_one(const p_draw *d, uint32_t batch) {")
        self.assertLess(mesh.index("g_fs_buffer_mesh = 1"),mesh.index("ok = prepare_draw"))
        self.assertGreater(mesh.index("g_fs_buffer_mesh = 0"),mesh.index("ok = prepare_draw"))
if __name__=="__main__":unittest.main()
