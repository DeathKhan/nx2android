"""Bounded descriptor profiling executes production capture/report helpers."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class DescriptorProfile(unittest.TestCase):
    @unittest.skipUnless(shutil.which("cc"), "a host C compiler is required")
    def test_complete_windows_limits_and_auto_off(self):
        source = (ROOT / "nxnative/dm/android/gfx_vk_exec.c").read_text(encoding='utf-8')
        start = source.index("enum { SET_PROFILE_ROWS")
        helpers = source[start:source.index("// content hash of descriptor sets", start)]
        code = r"""
#define _POSIX_C_SOURCE 200809L
#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <assert.h>
typedef struct { uint64_t key; int nbind, ndyn; } vkshader_t;
static uint64_t g_tex_frame;
static char output[65536]; static size_t used;
static void log_line(const char *fmt, ...) {
    va_list ap; va_start(ap,fmt);
    int n=vsnprintf(output+used,sizeof output-used,fmt,ap); va_end(ap);
    assert(n>=0&&(size_t)n+1<sizeof output-used);
    used+=(size_t)n; output[used++]='\n'; output[used]=0;
}
#define VKLOG(...) log_line(__VA_ARGS__)
HELPERS
int main(void) {
    vkshader_t s={.key=0xabc,.nbind=6,.ndyn=3};
    unsetenv("DM_VK_SET_PROFILE"); g_tex_frame=100;
    set_profile_poll(); assert(!g_set_profile_end&&!set_profile_row(2,&s));
    setenv("DM_VK_SET_PROFILE","invalid",1); set_profile_poll(); assert(!g_set_profile_end);
    setenv("DM_VK_SET_PROFILE","1",1); set_profile_poll();
    assert(g_set_profile_first==101&&g_set_profile_end==165&&!set_profile_row(2,&s));
    set_profile_t *first=NULL;
    for(g_tex_frame=101;g_tex_frame<165;g_tex_frame++) {
        set_profile_poll();
        set_profile_t *r=set_profile_row(2,&s);
        if(!first)first=r;
        assert(r==first);
        r->collected++; r->eligible++; r->hits++; r->collect_ns+=1000000; r->hash_ns+=100000;
        set_profile_t *other=set_profile_row(1,&s); assert(other!=r);
        other->memo++; other->eligible++;
    }
    assert(!set_profile_row(2,&s)); set_profile_poll();
    assert(!g_set_profile_end&&!strcmp(getenv("DM_VK_SET_PROFILE"),"0"));
    assert(strstr(output,"64 frames from present 101, 2 stage/key pairs"));
    assert(strstr(output,"per frame collected 1.0 memo 1.0"));
    assert(strstr(output,"collect 1.000 hash 0.100"));
    const char *rank=strstr(output,"descriptor profile stage ");
    assert(rank&&strstr(rank,"stage 2 key 0000000000000abc")==rank+19);
    set_profile_poll(); assert(!g_set_profile_end);
    used=0; output[0]=0; g_tex_frame=200; setenv("DM_VK_SET_PROFILE","1",1); set_profile_poll();
    assert(!g_set_profile_overflow.collected);
    g_tex_frame=201;
    for(unsigned i=0;i<SET_PROFILE_ROWS;i++) {
        s.key=i;
        set_profile_t *r=set_profile_row(0,&s);
        assert(r!=&g_set_profile_overflow&&!r->collected);
        r->collected=1; r->fresh=1; r->new_ns=100;
    }
    s.key=999999; assert(set_profile_row(0,&s)==&g_set_profile_overflow);
    g_set_profile_overflow.collected=1;
    setenv("DM_VK_SET_PROFILE","0",1); set_profile_poll();
    assert(!g_set_profile_end&&strstr(output,"descriptor profile overflow: 1 collected"));
    assert(strstr(output,"1 partial-window frames"));
    // Cancel while armed, and reject a window that would wrap.
    used=0; output[0]=0; g_tex_frame=300; setenv("DM_VK_SET_PROFILE","1",1); set_profile_poll();
    setenv("DM_VK_SET_PROFILE","0",1); set_profile_poll();
    assert(strstr(output,"cancelled before its first frame"));
    g_tex_frame=UINT64_MAX; setenv("DM_VK_SET_PROFILE","1",1); set_profile_poll();
    assert(!g_set_profile_end&&!strcmp(getenv("DM_VK_SET_PROFILE"),"0"));
    return 0;
}
""".replace("HELPERS", helpers)
        target = ROOT / "target/tests"
        target.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="set-profile-", dir=target) as folder:
            src, exe = Path(folder) / "check.c", Path(folder) / "check"
            src.write_text(code, encoding='utf-8')
            subprocess.run(["cc", "-std=c11", "-Wall", "-Wextra", "-Werror",
                            "-fsanitize=address,undefined", str(src), "-o", str(exe)], check=True)
            subprocess.run([str(exe)], check=True, timeout=15)


if __name__ == "__main__":
    unittest.main()
