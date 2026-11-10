"""Execute the completed-readback native RGBA8 sidecar block unchanged."""
from pathlib import Path
import shutil, subprocess, tempfile, unittest
import sys; sys.path.insert(0, str(Path(__file__).resolve().parent))
import host_cc   # noqa: E402
ROOT=Path(__file__).resolve().parents[1]
@unittest.skipUnless(shutil.which("cc"), "host compiler required")
class NativeSidecar(unittest.TestCase):
 @unittest.skipUnless(host_cc.sanitizers(), 'a host C compiler with ASan and UBSan is required (MinGW has neither)')
 def test_native_rgba_bytes_and_opt_in(self):
  s=(ROOT/"nxnative/dm/android/gfx_vk_exec.c").read_text(encoding='utf-8')
  a=s.index("        // Keep all four native RGBA8/BGRA8 channels too:")
  b=s.index("        if (pr && atoi(pr) && (d->f ==",a)
  actual=s[a:b]
  code=r"""
#include <vulkan/vulkan_core.h>
#include <assert.h>
#include <stdint.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
static char logs[2048];
static void logmsg(const char *fmt,...) { va_list ap; va_start(ap,fmt); vsnprintf(logs,sizeof logs,fmt,ap); va_end(ap); }
#define VKLOG(...) logmsg(__VA_ARGS__)
typedef struct { uint32_t w,h; VkFormat f; const void *p; } dump;
static void write_native(const char *pr,const dump *d,const char *path) { int i=11;
""" + actual + r"""
}
int main(int argc,char **argv) {
 assert(argc==2);char path[1024],native[1100];snprintf(path,sizeof path,"%s/pass_11.ppm",argv[1]);snprintf(native,sizeof native,"%s.native",path);
 // Every channel differs, including alpha. Raw SRGB/BGRA bytes must not be decoded or reordered.
 const unsigned char payload[16]={0,71,212,3,255,129,1,217,9,7,5,0,181,43,91,255};
 dump d={2,2,VK_FORMAT_R8G8B8A8_UNORM,payload};
 write_native(NULL,&d,path);assert(access(native,F_OK)!=0&&!logs[0]);
 write_native("0",&d,path);write_native("1",&d,path);assert(access(native,F_OK)!=0&&!logs[0]);
 VkFormat formats[]={VK_FORMAT_R8G8B8A8_UNORM,VK_FORMAT_R8G8B8A8_SRGB,VK_FORMAT_B8G8R8A8_UNORM,VK_FORMAT_B8G8R8A8_SRGB};
 for(unsigned k=0;k<sizeof formats/sizeof *formats;k++){
  d.f=formats[k];write_native(k==3?"3":"2",&d,path);
  FILE *f=fopen(native,"rb");assert(f);unsigned char got[17];assert(fread(got,1,sizeof got,f)==16);assert(!memcmp(got,payload,16));fclose(f);
  char expected[80];snprintf(expected,sizeof expected,"native format %u extent 2x2 bytes 16/16",d.f);
  assert(strstr(logs,expected));assert(strstr(logs,"pass dump 11:"));assert(!unlink(native));
 }
 d.f=VK_FORMAT_R16G16B16A16_SFLOAT;logs[0]=0;write_native("2",&d,path);assert(access(native,F_OK)!=0&&!logs[0]);
 d.f=VK_FORMAT_R8G8B8A8_UNORM;snprintf(path,sizeof path,"%s/missing/pass_11.ppm",argv[1]);write_native("2",&d,path);assert(strstr(logs,"bytes 0/16"));
 return 0;
}
"""
  target=ROOT/"target/tests";target.mkdir(parents=True,exist_ok=True)
  with tempfile.TemporaryDirectory(prefix="pass-native-",dir=target) as tmp:
   c,exe=Path(tmp)/"test.c",Path(tmp)/"test";c.write_text(code, encoding='utf-8')
   subprocess.run(["cc","-std=c11","-O1","-g","-Wall","-Wextra","-Werror","-fsanitize=address,undefined","-I"+str(ROOT/"nxnative/dm/third_party/vulkan-headers/include"),str(c),"-o",str(exe)],check=True)
   subprocess.run([str(exe),tmp],check=True,timeout=10)
  self.assertNotIn("vkCmd",actual)
  # The block consumes already-fenced, invalidated native bytes.
  poll=s[s.index("void vk_pass_dump_poll("):s.index("// DM_VK_CRUMBS=")]
  self.assertLess(poll.index("done_fence < g_pd[i].fence"),poll.index("// Keep all four native RGBA8/"))
  self.assertLess(poll.index("vkInvalidateMappedMemoryRanges"),poll.index("// Keep all four native RGBA8/"))
if __name__=="__main__": unittest.main()
