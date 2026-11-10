"""RenderDoc requests never inject a layer and complete only from actual capture metadata."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]

@unittest.skipUnless(shutil.which("cc"), "host C compiler required")
class RenderDocCapture(unittest.TestCase):
    def test_actual_module(self):
        src = (ROOT / "nxnative/dm/android/renderdoc_capture.c").read_text(encoding='utf-8')
        src = src.replace('#include "dm.h"', '').replace('#include "../third_party/renderdoc/renderdoc_app.h"', '')
        code = r"""
#define _GNU_SOURCE
#include <stdbool.h>
#include "renderdoc_app.h"
#include <assert.h>
#include <dlfcn.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <sys/stat.h>
static RENDERDOC_API_1_6_0 api;
static unsigned mode, opens, closes, resolves, gets, triggers, num, queried, pathmode, written;
static char last_log[2048], templ[1024];
static void mock_log(const char *fmt, ...) { va_list a; va_start(a,fmt); vsnprintf(last_log,sizeof last_log,fmt,a); va_end(a); }
#define LOG(...) mock_log(__VA_ARGS__)
static int getapi(RENDERDOC_Version v, void **out) { gets++; assert(v==eRENDERDOC_API_Version_1_6_0); *out=&api; return mode!=4; }
static void *mock_dlsym(void *h, const char *name) {
    resolves++; assert(!strcmp(name,"RENDERDOC_GetAPI"));
    return (mode==1 || (h==(void *)0x123 && mode!=3)) ? (void *)getapi : NULL;
}
static void *mock_dlopen(const char *name, int flags) {
    opens++; assert(!strcmp(name,"libVkLayer_GLES_RenderDoc.so") || !strcmp(name,"/loaded/layer.so"));
    assert((flags & RTLD_NOLOAD) && (flags & RTLD_NOW)); return mode ? (void *)0x123 : NULL;
}
static int mock_dladdr(const void *s, Dl_info *o) { assert(s==(void *)getapi); o->dli_fname="/loaded/layer.so";return 1; }
static int mock_dlclose(void *h) { assert(h==(void *)0x123); closes++; return 0; }
static int mock_stat(const char *p, struct stat *s) { assert(!strcmp(p,"/app/files/renderdoc/capture_frame12.rdc"));
    if(!written)return -1; memset(s,0,sizeof *s); s->st_mode=S_IFREG|0600; s->st_size=12345; return 0; }
static void setpath(const char *p) { snprintf(templ,sizeof templ,"%s",p); }
static uint32_t getnum(void) { return num; }
static void trigger(void) { triggers++; }
static uint32_t getcapture(uint32_t idx, char *p, uint32_t *n, uint64_t *t) {
    queried=idx; const char *path="/app/files/renderdoc/capture_frame12.rdc";
    if(pathmode==1)return 0;
    if(pathmode==2){*n=65537;return 1;}
    if(pathmode==3){*n=0;return 1;}
    if(p)strcpy(p,path); *n=(uint32_t)strlen(path)+1; *t=1700000000; return 1;
}
#define dlsym mock_dlsym
#define dlopen mock_dlopen
#define dlclose mock_dlclose
#define dladdr mock_dladdr
#define stat(...) mock_stat(__VA_ARGS__)
""" + src + r"""
#undef stat
static void reset(unsigned m) {
    mode=m;opens=closes=resolves=gets=triggers=num=queried=pathmode=written=0;last_log[0]=templ[0]=0;
    g_rdoc=NULL;g_rdoc_handle=NULL;atomic_store(&g_rdoc_pending,0);g_rdoc_before=0;
    memset(&api,0,sizeof api);api.SetCaptureFilePathTemplate=setpath;api.GetNumCaptures=getnum;api.GetCapture=getcapture;api.TriggerCapture=trigger;
    setenv("DM_ROOT","/app/files",1);
}
static void *request_thread(void *p) {(void)p;dm_renderdoc_capture();return NULL;}
#ifndef DM_RELEASE
int main(void) {
    reset(0);dm_renderdoc_poll();dm_renderdoc_status();assert(!opens&&!resolves&&!triggers);
    dm_renderdoc_capture();assert(opens==1&&!triggers&&!atomic_load(&g_rdoc_pending));
    reset(3);dm_renderdoc_capture();assert(opens==1&&closes==1&&!triggers&&!g_rdoc);
    reset(4);dm_renderdoc_capture();assert(opens==1&&closes==1&&!triggers&&!g_rdoc);
    reset(2);api.TriggerCapture=NULL;dm_renderdoc_capture();assert(closes==1&&!g_rdoc);
    reset(1);unsetenv("DM_ROOT");dm_renderdoc_capture();assert(!resolves&&!triggers);
    setenv("DM_ROOT","relative",1);dm_renderdoc_capture();assert(!resolves&&!triggers);
    char huge[1300];memset(huge,'a',sizeof huge);huge[0]='/';huge[sizeof huge-1]=0;
    setenv("DM_ROOT",huge,1);dm_renderdoc_capture();assert(!resolves&&!triggers);
    reset(1);num=3;dm_renderdoc_capture();assert(triggers==1&&opens==1&&g_rdoc_handle==(void *)0x123&&g_rdoc_before==3);
    assert(!strcmp(templ,"/app/files/renderdoc/capture"));assert(atomic_load(&g_rdoc_pending));
    dm_renderdoc_capture();dm_renderdoc_poll();assert(triggers==1&&atomic_load(&g_rdoc_pending));
    num=4;pathmode=1;dm_renderdoc_poll();assert(atomic_load(&g_rdoc_pending));
    pathmode=0;written=1;dm_renderdoc_poll();assert(!atomic_load(&g_rdoc_pending)&&queried==3);
    assert(strstr(last_log,"capture completed")&&strstr(last_log,"bytes 12345"));
    dm_renderdoc_poll();assert(triggers==1);dm_renderdoc_capture();assert(triggers==2&&gets==1);
    num=5;written=0;dm_renderdoc_poll();assert(!atomic_load(&g_rdoc_pending));
    assert(strstr(last_log,"file not accessible")&&!strstr(last_log,"capture completed"));
    reset(2);dm_renderdoc_capture();assert(opens==1&&!closes&&g_rdoc_handle==(void *)0x123);
    num=1;pathmode=2;dm_renderdoc_poll();assert(!atomic_load(&g_rdoc_pending));
    assert(strstr(last_log,"invalid path length"));
    reset(2);dm_renderdoc_capture();num=1;pathmode=3;dm_renderdoc_poll();assert(!atomic_load(&g_rdoc_pending));
    reset(2);pthread_t th[12];for(unsigned i=0;i<12;i++)assert(!pthread_create(&th[i],NULL,request_thread,NULL));
    for(unsigned i=0;i<12;i++)pthread_join(th[i],NULL);assert(triggers==1&&gets==1&&opens==1);
    puts("RenderDoc actual module: all cases passed");return 0;
}
#else
int main(void) { reset(1);dm_renderdoc_capture();dm_renderdoc_poll();dm_renderdoc_status();
    assert(!resolves&&!opens&&!gets&&!triggers&&!atomic_load(&g_rdoc_pending));return 0; }
#endif
"""
        tdroot = ROOT / "target/tests"
        tdroot.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="renderdoc-", dir=tdroot) as td:
            p=Path(td);(p/"test.c").write_text(code, encoding='utf-8')
            for flags in ([], ["-DDM_RELEASE=1"]):
                subprocess.run(["cc","-std=c11","-g","-O1","-fsanitize=address,undefined","-pthread"]+flags+[
                                "-I"+str(ROOT/"nxnative/dm/third_party/renderdoc"),str(p/"test.c"),"-o",str(p/"test")],check=True)
                subprocess.run([str(p/"test")],check=True)

    def test_command_and_present_hooks(self):
        hid=(ROOT/"nxnative/dm/host/hle_hid.c").read_text(encoding='utf-8')
        cap=hid.index('if (!strcmp(line, "rdoccapture"))')
        self.assertGreater(hid.rfind('#if defined(__ANDROID__)',0,cap),hid.rfind('#endif',0,cap))
        self.assertIn('dm_renderdoc_capture(); continue;',hid[cap:cap+180])
        present=(ROOT/"nxnative/dm/android/gfx_vk_present.c").read_text(encoding='utf-8')
        call=present.index('dm_renderdoc_poll();')
        vk=present.rfind('r = vkQueuePresentKHR',0,call)
        unlock=present.rfind('pthread_mutex_unlock(&g_vk_queue_mu);',0,call)
        self.assertGreater(vk,0);self.assertGreater(unlock,vk)
        self.assertLess(call,present.index('if (r == VK_ERROR_OUT_OF_DATE_KHR)',vk))
        build=(ROOT/"nxnative/dm/android/build.py").read_text(encoding='utf-8')
        self.assertIn("'android/*.c'",build)

if __name__ == '__main__':unittest.main()
