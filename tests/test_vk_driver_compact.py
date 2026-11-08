"""Execute the real graphics-c0 cache key and driver segment construction."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


def function(source, declaration):
    start = source.index(declaration)
    pos = source.index("{", start) + 1
    depth = 1
    while depth:
        depth += (source[pos] == "{") - (source[pos] == "}")
        pos += 1
    return source[start:pos]


class CompactDriver(unittest.TestCase):
    @unittest.skipUnless(shutil.which("cc"), "a host C compiler is required")
    def test_content_dependencies_cache_and_legacy_paths(self):
        source = (ROOT / "nxnative/dm/android/gfx_vk_exec.c").read_text()
        code = r"""
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "dmsh_abi.h"
typedef uint64_t VkBuffer;
typedef uint64_t VkDeviceSize;
typedef struct { uint64_t addr,size; uint32_t unrelated; } binding;
static struct {
 binding ssbo[6][16],ubo[6][16];
 uint8_t push[2][256];
 uint64_t img[6][8],tex[6][256],samp[6][128];
} S;
static struct { struct { struct { uint32_t minUniformBufferOffsetAlignment; } limits; } props; } V;
static uint32_t g_draw_binst,g_dispatch[3];
static uint64_t g_set_epoch=1;
static int g_drv_compact=1,g_x_nodrvc,ounce,segment_calls,allocations;
enum { DRVC=1024,MAX_ALLOCATIONS=1024 };
static struct { uint64_t h,epoch; VkBuffer b; VkDeviceSize o; } g_drvc[DRVC];
static uint8_t scratch[MAX_ALLOCATIONS][4096];
static uint64_t c2_salt;
static int dm_sdk_ounce(void){return ounce;}
static uint64_t c2_handle(int st,int off){return c2_salt^((uint64_t)(unsigned)st<<48)^(uint32_t)off;}
static uint8_t *vk_scratch(uint64_t n,uint64_t align,VkBuffer *b,VkDeviceSize *o){
 assert(n==4096&&align==256&&allocations<MAX_ALLOCATIONS);
 *b=1;*o=(uint64_t)allocations*4096;
 memset(scratch[allocations],0xa5,4096);
 return scratch[allocations++];
}
SEGMENT
static void drv_segment(int st,int bank,uint8_t *d,int dyn){segment_calls++;actual_drv_segment(st,bank,d,dyn);}
FNV
KEY
SHARED
static uint8_t *bytes(VkBuffer b,VkDeviceSize o){
 assert(b==1&&o%4096==0&&o/4096<(uint64_t)allocations);return scratch[o/4096];
}
// Independent expected c0 layout, including its zero padding and fixed point-size word.
static void graphics_oracle(int st,uint8_t *expected){
 const unsigned bases[3]={DMSH_C0_SSBO_VS,DMSH_C0_SSBO_PS,DMSH_C0_SSBO_GS};
 memset(expected,0,4096);
 for(unsigned i=0;i<16;i++){
  uint64_t address=S.ssbo[st][i].addr;uint32_t size=(uint32_t)S.ssbo[st][i].size;
  memcpy(expected+bases[st]+16*i,&address,8);memcpy(expected+bases[st]+16*i+8,&size,4);
 }
 float point_size=2047.0f;memcpy(expected+0x610,&point_size,4);memcpy(expected+0x644,&g_draw_binst,4);
}
static uint64_t legacy_key(int st,int bank,int dyn){
 uint8_t expected[4096];actual_drv_segment(st,bank,expected,dyn);
 return fnv(UINT64_C(0xcbf29ce484222325),expected,bank==24?0x100:bank?0x968:0x648)^((uint64_t)bank<<56);
}
static void check_bytes(int st,int bank,int dyn,VkBuffer b,VkDeviceSize o){
 uint8_t expected[4096];actual_drv_segment(st,bank,expected,dyn);
 assert(!memcmp(bytes(b,o),expected,4096));
 if(st>=0&&st<3&&!bank){graphics_oracle(st,expected);assert(!memcmp(bytes(b,o),expected,4096));}
}
static void call_checked(int st,int bank,int dyn,VkBuffer *b,VkDeviceSize *o){
 drv_shared(st,bank,b,o,dyn);check_bytes(st,bank,dyn,*b,*o);
}
static void fresh_case(int st,int bank,int dyn){
 memset(g_drvc,0,sizeof g_drvc);
 VkBuffer a,b;VkDeviceSize x,y;int before=allocations,segments=segment_calls;
 call_checked(st,bank,dyn,&a,&x);assert(allocations==before+1&&segment_calls==segments+1);
 uint64_t h=g_drv_compact&&!bank&&st>=0&&st<3?drv_graphics_key(st):legacy_key(st,bank,dyn);
 assert(g_drvc[h%DRVC].h==h&&g_drvc[h%DRVC].epoch==g_set_epoch);
 call_checked(st,bank,dyn,&b,&y);assert(a==b&&x==y&&allocations==before+1);
 assert(segment_calls==segments+(g_drv_compact&&!bank&&st>=0&&st<3?1:2));
}
int main(void){
 V.props.limits.minUniformBufferOffsetAlignment=256;
 for(int st=0;st<6;st++){
  for(int i=0;i<16;i++){S.ssbo[st][i]=(binding){UINT64_C(0x7000000000)+i*0x12340,UINT64_C(0x200000000)+i*61+1,0};S.ubo[st][i]=(binding){UINT64_C(0x8000000000)+i*0x43210,1024+i,0};}
  for(int i=0;i<8;i++)S.img[st][i]=i+1;
  for(int i=0;i<256;i++)S.tex[st][i]=i+1;
  for(int i=0;i<128;i++)S.samp[st][i]=i+1;
 }
 g_draw_binst=91;
 // Equal stage data still occupies different table offsets.
 uint64_t h0=drv_graphics_key(0),h1=drv_graphics_key(1),h2=drv_graphics_key(2);
 assert(h0!=h1&&h0!=h2&&h1!=h2);
 for(int st=0;st<3;st++){
  fresh_case(st,0,0);
  assert(drv_graphics_key(st)!=legacy_key(st,0,0)); // compact/full domains do not alias these inputs
  VkBuffer b;VkDeviceSize o;uint64_t previous;int n;
  // Each of all16 address and low32 size fields is a real dependency.
  for(int i=0;i<16;i++){
   previous=drv_graphics_key(st);S.ssbo[st][i].addr^=UINT64_C(0x8000000000000001);
   assert(drv_graphics_key(st)!=previous);n=allocations;call_checked(st,0,0,&b,&o);assert(allocations==n+1);
   previous=drv_graphics_key(st);S.ssbo[st][i].size^=UINT64_C(0x80000001);
   assert(drv_graphics_key(st)!=previous);n=allocations;call_checked(st,0,0,&b,&o);assert(allocations==n+1);
   // The segment deliberately truncates sizes to uint32; unrelated CPU fields are absent.
   previous=drv_graphics_key(st);S.ssbo[st][i].size^=UINT64_C(1)<<44;S.ssbo[st][i].unrelated++;
   assert(drv_graphics_key(st)==previous);n=allocations;call_checked(st,0,1,&b,&o);assert(allocations==n);
  }
  previous=drv_graphics_key(st);g_draw_binst++;assert(drv_graphics_key(st)!=previous);
  n=allocations;call_checked(st,0,0,&b,&o);assert(allocations==n+1);
  // Other stages, handles, push banks, dispatch and compute UBOs do not affect graphics c0.
  previous=drv_graphics_key(st);S.ssbo[(st+1)%3][0].addr++;S.ubo[5][7].addr++;
  S.tex[st][0]++;S.img[st][0]++;S.samp[st][0]++;S.push[0][17]++;g_dispatch[0]++;
  assert(drv_graphics_key(st)==previous);n=allocations;call_checked(st,0,0,&b,&o);assert(allocations==n);
 }
 // Epoch and no-cache guard must never return a prior submission's or disabled entry.
 VkBuffer a,b;VkDeviceSize x,y;int before;
 call_checked(2,0,0,&a,&x);before=allocations;g_set_epoch++;call_checked(2,0,0,&b,&y);assert(allocations==before+1&&x!=y);
 g_x_nodrvc=1;before=allocations;call_checked(2,0,0,&a,&x);call_checked(2,0,0,&b,&y);assert(allocations==before+2&&x!=y);g_x_nodrvc=0;
 // Switching A/B changes only cache representation, never constructed bytes.
 g_drv_compact=0;for(int st=0;st<3;st++)fresh_case(st,0,0);
 g_drv_compact=1;call_checked(2,0,0,&a,&x);g_drv_compact=0;call_checked(2,0,0,&b,&y);assert(!memcmp(bytes(a,x),bytes(b,y),4096));
 g_drv_compact=1;
 // Compute c0 remains the full-byte path, including dispatch, compute UBOs and NVN2 header.
 for(ounce=0;ounce<2;ounce++){
  fresh_case(5,0,0);uint64_t old=legacy_key(5,0,0);g_dispatch[2]++;
  assert(legacy_key(5,0,0)!=old);before=allocations;call_checked(5,0,0,&a,&x);assert(allocations==before+1);
  old=legacy_key(5,0,0);S.ubo[5][15].addr++;assert(legacy_key(5,0,0)!=old);
  before=allocations;call_checked(5,0,0,&a,&x);assert(allocations==before+1);
 }
 for(int st=3;st<=4;st++)fresh_case(st,0,0);
 // Texture-table banks preserve both full handles and dynamic-slot forms; bank24 follows the correct push bank.
 for(int st=0;st<6;st++){fresh_case(st,2,0);fresh_case(st,2,1);fresh_case(st,24,0);}
 fresh_case(2,2,0);uint64_t old=legacy_key(2,2,0);c2_salt=UINT64_C(0x987654321);
 assert(legacy_key(2,2,0)!=old);before=allocations;call_checked(2,2,0,&a,&x);assert(allocations==before+1);
 for(int st=0;st<=5;st+=5){
  fresh_case(st,24,0);old=legacy_key(st,24,0);S.push[st==5][255]^=0x81;
  assert(legacy_key(st,24,0)!=old);before=allocations;call_checked(st,24,0,&a,&x);assert(allocations==before+1);
 }
 printf("PASS: compact graphics c0 dependencies, exact4096-byte outputs, cache hits/misses and legacy fallbacks (%d allocations)\n",allocations);
 return 0;
}
"""
        helpers = {
            "SEGMENT": function(source, "static void drv_segment(").replace(
                "static void drv_segment(", "static void actual_drv_segment(", 1),
            "FNV": function(source, "static inline uint64_t fnv("),
            "KEY": function(source, "static uint64_t drv_graphics_key("),
            "SHARED": function(source, "static void drv_shared("),
        }
        for marker, helper in helpers.items():
            code = code.replace("\n" + marker + "\n", "\n" + helper + "\n")
        target = ROOT / "target/tests"
        target.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="vk-driver-compact-", dir=target) as folder:
            src, exe = Path(folder) / "test.c", Path(folder) / "test"
            src.write_text(code)
            subprocess.run(["cc", "-std=c11", "-O1", "-g", "-Wall", "-Wextra", "-Werror",
                            "-fsanitize=address,undefined", "-I" + str(ROOT / "nxnative/dm/gfx/nvn"),
                            str(src), "-o", str(exe)], check=True)
            subprocess.run([str(exe)], check=True, timeout=15)


if __name__ == "__main__":
    unittest.main()
