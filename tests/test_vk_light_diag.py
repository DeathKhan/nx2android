"""Actual support-value initializer and opt-in fragment diagnostic descriptor behavior."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from test_vk_compute_trace import extract
import sys; sys.path.insert(0, str(Path(__file__).resolve().parent))
import host_cc   # noqa: E402
ROOT = Path(__file__).resolve().parents[1]

@unittest.skipUnless(shutil.which("cc"), "host compiler required")
class LightDiagnostic(unittest.TestCase):
    @unittest.skipUnless(host_cc.posix() and host_cc.sanitizers(), 'a POSIX host C compiler with ASan and UBSan is required (MinGW has no setenv, libasan or libubsan)')
    def test_actual_control_and_snapshot_lifetime(self):
        src = (ROOT/"nxnative/dm/android/gfx_vk_exec.c").read_text(encoding='utf-8')
        shader = (ROOT/"nxnative/dm/android/gfx_vk_shader.c").read_text(encoding='utf-8')
        start = src.index("static uint64_t g_light_diag_key")
        actual = src[start:src.index("static uint64_t g_fs_buffer_key", start)]
        fill = extract(shader, "void vk_support_values(")
        code = r"""
#define _POSIX_C_SOURCE 200809L
#include <assert.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
typedef uintptr_t VkBuffer;
typedef uint64_t VkDeviceSize;
typedef struct { VkBuffer buffer; VkDeviceSize offset, range; } VkDescriptorBufferInfo;
typedef struct { uint64_t key; } vkshader_t;
enum { VK_SUPPORT_BYTES=65536 };
static uint64_t g_set_epoch=1;
static struct { struct { struct { uint64_t minUniformBufferOffsetAlignment; } limits; } props; } V={{{256}}};
#define VKLOG(f,...) do { if (strstr(f,"diagnostic applied")) applied++; if(0)fprintf(stderr,f,__VA_ARGS__); } while(0)
static int applied,g_sub_pass=3;
static uint64_t g_tex_frame=5;
static uint64_t vk_frame_current(void){return 5;}
static unsigned allocated;
static uint8_t pages[8][VK_SUPPORT_BYTES];
static VkBuffer vk_support_buffer(void) { return 77; }
static void *vk_scratch(uint64_t n,uint64_t align,VkBuffer *b,VkDeviceSize *o) {
 assert(n==VK_SUPPORT_BYTES && align==256 && allocated<8);
 *b=100+allocated;*o=256*allocated;
 return pages[allocated++];
}
""" + fill + "\n" + actual + r"""
static void poll(const char *s) { if(s)setenv("DM_VK_LIGHT_DIAG",s,1);else unsetenv("DM_VK_LIGHT_DIAG");light_diag_poll(); }
static void check_base(const uint8_t *p) {
 uint8_t oracle[VK_SUPPORT_BYTES]={0};float one=1;
 for(unsigned i=0;i<73;i++)memcpy(oracle+192+16*i,&one,4);
 assert(!memcmp(p,oracle,sizeof oracle));
}
int main(void) {
 uint8_t normal[VK_SUPPORT_BYTES]; memset(normal,0xa5,sizeof normal);vk_support_values(normal);check_base(normal);
 uint64_t k;uint32_t m;
 const char *bad[]={"a","a:","0:1",":1","a:-1","a:+1","-a:1"," a:1","a:1 ","a:1:2","a:4294967296","a:0x100000000","12345678901234567:1","0x:1","a:0x","a:foo","a:99999999999999999999"};
 for(unsigned i=0;i<sizeof bad/sizeof *bad;i++){assert(!light_diag_parse(bad[i],&k,&m));assert(!k&&!m);}
 assert(light_diag_parse("0xffffffffffffffff:4294967295",&k,&m)&&k==UINT64_MAX&&m==UINT32_MAX);
 assert(light_diag_parse("abc:0x80000000",&k,&m)&&k==0xabc&&m==0x80000000u);
 assert(light_diag_parse("abc:08",&k,&m)&&m==8);
 assert(light_diag_parse("abc:0",&k,&m)&&!k&&!m);
 vkshader_t selected={0xabc},other={0xdef};
 VkDescriptorBufferInfo d,first;
 poll(NULL);light_diag_support(1,&selected,&d);assert(d.buffer==77&&!allocated&&d.offset==0&&d.range==65536);
 poll("abc:1");for(int stage=0;stage<6;stage++)if(stage!=1){light_diag_support(stage,&selected,&d);assert(d.buffer==77&&!allocated);}
 light_diag_support(1,&other,&d);assert(d.buffer==77&&!allocated);
 light_diag_support(1,&selected,&first);assert(!applied);light_diag_note_draw(0xdef,1);assert(!applied);light_diag_note_draw(selected.key,1);assert(allocated==1&&first.buffer==100&&first.range==65536&&applied==1);
 uint32_t value;memcpy(&value,pages[0]+1376,4);assert(value==1);
 uint8_t expected[VK_SUPPORT_BYTES];memcpy(expected,normal,sizeof expected);memcpy(expected+1376,&value,4);assert(!memcmp(expected,pages[0],sizeof expected));
 for(int i=0;i<1000;i++){light_diag_support(1,&selected,&d);assert(!memcmp(&first,&d,sizeof d)&&allocated==1&&applied==1);}
 poll("abc:2");light_diag_note_draw(selected.key,2);assert(applied==1);light_diag_support(1,&selected,&d);light_diag_note_draw(selected.key,2);assert(allocated==2&&d.buffer!=first.buffer&&applied==2);memcpy(&value,pages[0]+1376,4);assert(value==1);
 uint8_t old[VK_SUPPORT_BYTES];memcpy(old,pages[1],sizeof old);
 g_set_epoch++;light_diag_support(1,&selected,&d);assert(allocated==3&&d.buffer==102&&!memcmp(old,pages[2],sizeof old)&&applied==2);
 poll("def:2");light_diag_support(1,&selected,&d);assert(d.buffer==77&&allocated==3);light_diag_support(1,&other,&d);assert(allocated==4&&d.buffer==103);
 poll("0");light_diag_support(1,&other,&d);assert(d.buffer==77&&allocated==4&&!memcmp(old,pages[1],sizeof old));
 poll("abc:bad");light_diag_support(1,&selected,&d);assert(d.buffer==77&&allocated==4);
 return 0;
}
"""
        target=ROOT/"target/tests";target.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="light-diag-",dir=target) as temp:
            c,exe=Path(temp)/"t.c",Path(temp)/"t"
            c.write_text(code, encoding='utf-8')
            subprocess.run(["cc","-std=c11","-O1","-g","-Wall","-Wextra","-Werror","-fsanitize=address,undefined",str(c),"-o",str(exe)],check=True)
            subprocess.run([str(exe)],check=True,timeout=10)
        self.assertIn("case RK_SUPPORT: light_diag_support(st, s, &bi[nw]); mrt_support(st, s, &bi[nw]); break;", extract(src,"static VkDescriptorSet stage_set("))
        self.assertIn("light_diag_poll();",extract(src,"static void x_read("))
        self.assertIn("g_set_epoch++; x_read();",src) # controls only change after the memo epoch advances
        self.assertIn("if (which == 0) vk_support_values(p);",extract(shader,"static VkBuffer static_buf("))
if __name__=="__main__":unittest.main()
