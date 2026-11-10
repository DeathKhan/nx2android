"""Actual bundle-wide pool-kind scan, descriptor layout and limit accounting."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
import sys; sys.path.insert(0, str(Path(__file__).resolve().parent))
import host_cc   # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def fixture():
    source = (ROOT / "nxnative/dm/android/gfx_vk_shader.c").read_text(encoding='utf-8')
    header = (ROOT / "nxnative/dm/android/gfx_vk.h").read_text(encoding='utf-8')
    defs = header[header.index("enum { RK_SUPPORT"):header.index("_Static_assert(sizeof(dmsv_bind)")]
    bent = source[source.index("typedef struct __attribute__((packed)) { uint64_t key;"):
                  source.index("static const uint8_t *g_bundle;")]
    # ends at vk_shaders_open itself rather than its comment, which upstream nx2android #173 (stricter index checks) rewrote
    scan = source[source.index("static unsigned g_pool_kinds"):
                  source.index("void vk_shaders_open(")]
    layout = source[source.index("static int g_pool_nt"):
                    source.index("// Is the id used only")]
    return r"""
#include <vulkan/vulkan_core.h>
#include <assert.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
DEFS
BENT
enum { TBL_SLOTS=3, POOL_TEX_SLOTS=4, POOL_B0=3, POOL_SAMP_B=7, SET0_BINDINGS=8 };
SCAN
static struct { struct { VkPhysicalDeviceLimits limits; } props; } V;
static pthread_mutex_t g_sh_mu=PTHREAD_MUTEX_INITIALIZER;
static unsigned bundle_opens, layout_calls, desired_mask=3;
static VkDescriptorSetLayoutBinding captured[8];
static void bundle_auto(void){bundle_opens++;g_pool_kinds=desired_mask;}
static int vk_tbl_n(void){return 2048;}
#define VKLOG(...) ((void)0)
static VkDescriptorSetLayout shader_layout(const VkDescriptorSetLayoutBinding *b,uint32_t n){
 assert(n==8);layout_calls++;memcpy(captured,b,sizeof captured);
 return (VkDescriptorSetLayout)(uintptr_t)123;
}
LAYOUT
static uint8_t b[1024];
static void put32(unsigned off,uint32_t x){memcpy(b+off,&x,4);}
static void entry(unsigned i,unsigned stage,dmsv_bind r){
 bent e={.key=i+1,.stage=stage,.spv_off=900,.spv_len=20,.refl_off=200+i*40,.refl_len=40};
 memcpy(b+12+i*sizeof(e),&e,sizeof(e));put32(e.refl_off,1);
 memcpy(b+e.refl_off+20,&r,sizeof(r));
}
static void init(unsigned n){
 memset(b,0,sizeof b);memcpy(b,"DMSV",4);put32(4,2);put32(8,n);
}
static void scanner(void){
 dmsv_bind r={.kind=RK_TEXP,.desc=RD_IMAGE,.dim=1,.count=65535};
 assert(bundle_pool_kinds(NULL,0)==15);
 init(0);assert(bundle_pool_kinds(b,sizeof b)==15);
 init(2);entry(0,0,r);r.flags=1;entry(1,7,r);
 assert(bundle_pool_kinds(b,sizeof b)==3); // Includes later mesh/task/compute entries.
 for(unsigned stage=0;stage<8;stage++){
  init(1);r.flags=0;r.dim=2;entry(0,stage,r);assert(bundle_pool_kinds(b,sizeof b)==8);
 }
 init(4);r.dim=1;entry(0,0,r);r.flags=1;entry(1,1,r);
 r.flags=0;r.dim=3;entry(2,6,r);r.dim=2;entry(3,7,r);
 assert(bundle_pool_kinds(b,sizeof b)==15);
 init(1);r=(dmsv_bind){.kind=RK_CBUF,.desc=RD_UBO,.count=1};entry(0,5,r);
 assert(bundle_pool_kinds(b,sizeof b)==0);
 r=(dmsv_bind){.kind=RK_SAMPP,.desc=RD_SAMPLER,.dim=255,.count=4096};entry(0,1,r);
 assert(bundle_pool_kinds(b,sizeof b)==0);
 r.flags=2;entry(0,1,r);assert(bundle_pool_kinds(b,sizeof b)==15);
 r=(dmsv_bind){.kind=RK_TEXP,.desc=RD_IMAGE,.dim=1,.count=65535};entry(0,0,r);
 assert(bundle_pool_kinds(b,sizeof b)==1);
 uint8_t good[1024];memcpy(good,b,sizeof b);
 #define RESET() memcpy(b,good,sizeof b)
 put32(4,99);assert(bundle_pool_kinds(b,sizeof b)==15);RESET();
 assert(bundle_pool_kinds(b,11)==15);assert(bundle_pool_kinds(b,43)==15);
 put32(8,UINT32_MAX);assert(bundle_pool_kinds(b,sizeof b)==15);RESET();
 put32(20,8);assert(bundle_pool_kinds(b,sizeof b)==15);RESET(); // unknown stage
 put32(24,UINT32_MAX);assert(bundle_pool_kinds(b,sizeof b)==15);RESET(); // SPIR-V offset
 put32(32,UINT32_MAX);assert(bundle_pool_kinds(b,sizeof b)==15);RESET(); // reflection offset
 put32(36,19);assert(bundle_pool_kinds(b,sizeof b)==15);RESET();
 put32(200,UINT32_MAX);assert(bundle_pool_kinds(b,sizeof b)==15);RESET();
 put32(204,1);assert(bundle_pool_kinds(b,sizeof b)==15);RESET(); // missing constant payload
 b[220]=255;assert(bundle_pool_kinds(b,sizeof b)==15);RESET();
 b[221]=8;assert(bundle_pool_kinds(b,sizeof b)==15);RESET();
 b[221]=RD_STORAGE;assert(bundle_pool_kinds(b,sizeof b)==15);RESET();
 b[222]=0;assert(bundle_pool_kinds(b,sizeof b)==15);RESET();
 for(unsigned f=2;f<=128;f<<=1){b[223]=f;assert(bundle_pool_kinds(b,sizeof b)==15);RESET();}
 b[226]=b[227]=0;assert(bundle_pool_kinds(b,sizeof b)==15);RESET();
}
static void *get_layout(void *unused){
 (void)unused;assert(vk_tbl_dsl()==(VkDescriptorSetLayout)(uintptr_t)123);return NULL;
}
int main(int argc,char **argv){
 assert(argc==2);int mode=atoi(argv[1]);scanner();
 V.props.limits.maxPerStageDescriptorSampledImages=1000000;
 V.props.limits.maxDescriptorSetSampledImages=1000000;
 V.props.limits.maxPerStageDescriptorSamplers=1000000;
 V.props.limits.maxDescriptorSetSamplers=1000000;
 setenv("DM_VK_POOL_N","65536",1);
 if(mode==1){desired_mask=15;V.props.limits.maxDescriptorSetSampledImages=200000;}
 if(mode==2)desired_mask=0;
 if(mode==3)setenv("DM_VK_POOL_N","0",1);
 if(mode==4||mode==5)V.props.limits.maxPerStageDescriptorSampledImages=131074+6144+256-(mode==5);
 if(mode==6)V.props.limits.maxDescriptorSetSamplers=6144+256;
 if(mode==7)desired_mask=15;
 pthread_t threads[8];for(int i=0;i<8;i++)assert(!pthread_create(&threads[i],NULL,get_layout,NULL));
 for(int i=0;i<8;i++)assert(!pthread_join(threads[i],NULL));
 assert(bundle_opens==1&&layout_calls==1);
 unsigned np=mode==3||mode==6?1:mode==5?65280:mode==1?48384:65536;
 assert(vk_pool_n()==(mode==3||mode==6?0:(int)np));
 unsigned total=0;
 for(unsigned i=0;i<8;i++){
  assert(captured[i].binding==i&&captured[i].stageFlags==VK_SHADER_STAGE_ALL_GRAPHICS);
  if(i<3)assert(captured[i].descriptorType==VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER&&captured[i].descriptorCount==2048);
  else if(i<7){
   assert(captured[i].descriptorType==VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE);
   assert(captured[i].descriptorCount==((desired_mask&(1u<<(i-3)))?np:1));total+=captured[i].descriptorCount;
  }else assert(captured[i].descriptorType==VK_DESCRIPTOR_TYPE_SAMPLER&&captured[i].descriptorCount==(mode==3||mode==6?1:4096));
 }
 if(mode==0||mode==4)assert(total==131074); // Former layout reserved262144.
 if(mode==2)assert(total==4);
 if(mode==1){assert(total+6144+256<=200000);assert(total+4*256+6144+256>200000);}
 if(mode==7)assert(total==262144);
 return 0;
}
""".replace("DEFS", defs).replace("BENT", bent).replace("SCAN", scan).replace("LAYOUT", layout)


class PoolLayout(unittest.TestCase):
    @unittest.skipUnless(shutil.which("cc"), "host C compiler required")
    @unittest.skipUnless(host_cc.posix() and host_cc.sanitizers(), 'a POSIX host C compiler with ASan and UBSan is required (MinGW has no setenv, libasan or libubsan)')
    def test_bundle_union_limits_and_actual_layout(self):
        target = ROOT / "target/tests"
        target.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="pool-layout-", dir=target) as folder:
            folder = Path(folder)
            source, exe = folder / "fixture.c", folder / "fixture"
            source.write_text(fixture(), encoding='utf-8')
            subprocess.run(["cc", "-std=gnu11", "-Wall", "-Wextra", "-Werror",
                            "-Wno-misleading-indentation", "-fsanitize=address,undefined", "-pthread",
                            "-I" + str(ROOT / "nxnative/dm/third_party/vulkan-headers/include"),
                            str(source), "-o", str(exe)], check=True)
            for mode in range(8):
                subprocess.run([str(exe), str(mode)], check=True, timeout=15)


if __name__ == "__main__":
    unittest.main()
