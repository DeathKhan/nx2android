"""Exercise the real graphics UBO descriptor path with proven and unknown extents."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(shutil.which('cc'), 'host C compiler required')
class UniformShadowBounds(unittest.TestCase):
    def test_shadow_descriptor_and_full_range_fallbacks(self):
        source = (ROOT / 'nxnative/dm/android/gfx_vk_exec.c').read_text()
        start = source.index('static void cbuf_desc(')
        pos = source.index('{', start) + 1; depth = 1
        while depth:
            depth += (source[pos] == '{') - (source[pos] == '}'); pos += 1
        body = source[start:pos] + '\n'
        code = r"""
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <assert.h>
typedef uintptr_t VkBuffer;
typedef uint64_t VkDeviceSize;
typedef struct {VkBuffer buffer; VkDeviceSize offset,range;} VkDescriptorBufferInfo;
typedef struct {uint64_t addr,size;} p_bind_buf;
typedef struct {int reg;} dmsv_bind;
typedef struct {uint32_t ubo_extent[16];} vkshader_t;
typedef struct {uint64_t lo,hi,end; VkBuffer buf;} vkpool_t;
static struct {p_bind_buf ubo[6][16];} S;
static struct {struct {struct {uint64_t minUniformBufferOffsetAlignment;} limits;} props;} V;
static int g_var_now,g_ubo_bounds=1,g_spv_alt;
static uint64_t g_set_epoch=1,g_ubd_gen=1;
static vkpool_t pool;
static uint64_t shadow_request,copy_request;
static int shadow_present,cross_count,shadow_calls;
static uint8_t scratch[65536];
static vkpool_t *vk_pool_of(uint64_t a,uint64_t n) {
 return a>=pool.lo && a<pool.hi && n<=pool.hi-a ? &pool : NULL;
}
static VkBuffer vk_zero_buffer(void) {return 1;}
static uint8_t *vk_scratch(uint64_t n,uint64_t align,VkBuffer *b,VkDeviceSize *o) {
 (void)align;assert(n<=sizeof scratch);copy_request=n;*b=3;*o=0;return scratch;
}
static void vk_ubd_read(uint64_t a,uint64_t n,uint8_t *d) {(void)a;memset(d,17,n);}
static void vk_cross(int kind,uint64_t a,uint64_t n,const vkpool_t *p) {
 (void)a;(void)n;(void)p;assert(kind==0);cross_count++;
}
static void ubo_log(const vkshader_t *s,int st,int reg,uint64_t a,uint64_t n) {
 (void)s;(void)st;(void)reg;(void)a;(void)n;
}
static int vk_ubd_shadow(uint64_t a,uint64_t n,VkBuffer *b,VkDeviceSize *o) {
 (void)a;shadow_calls++;shadow_request=n;if(!shadow_present)return 0;*b=2;*o=0;return 1;
}
#define VKRED_ONCE(...) ((void)0)
""" + body + r"""
static void reset(vkshader_t *s) {
 memset(&S,0,sizeof S);memset(s,0,sizeof *s);
 pool=(vkpool_t){0x100000,0x120000,0x120000,5};
 for(int st=0;st<6;st++)S.ubo[st][5]=(p_bind_buf){pool.lo+256,64960};
 s->ubo_extent[5]=2192;V.props.limits.minUniformBufferOffsetAlignment=256;
 g_var_now=g_spv_alt=0;g_ubo_bounds=1;shadow_present=1;
 shadow_request=copy_request=0;cross_count=shadow_calls=0;g_set_epoch++;g_ubd_gen++;
}
int main(void) {
 vkshader_t s;dmsv_bind b={5};VkDescriptorBufferInfo d;
 for(int st=0;st<3;st++) {
  reset(&s);cbuf_desc(st,&s,&b,&d);
  assert(shadow_request==2192 && d.buffer==2 && d.range==2192);
 }
 reset(&s);g_ubo_bounds=0;cbuf_desc(1,&s,&b,&d);
 assert(shadow_request==64960 && d.range==64960);
 reset(&s);g_spv_alt=1;cbuf_desc(1,&s,&b,&d);
 assert(shadow_request==64960 && d.range==64960);
 reset(&s);cbuf_desc(5,&s,&b,&d);
 assert(shadow_request==64960 && d.range==64960);
 reset(&s);s.ubo_extent[5]=0;cbuf_desc(1,&s,&b,&d);
 assert(shadow_request==64960 && d.range==64960);
 reset(&s);S.ubo[1][5].size=1024;cbuf_desc(1,&s,&b,&d);
 assert(shadow_request==1024 && d.range==1024);
 reset(&s);shadow_present=0;cbuf_desc(1,&s,&b,&d);
 assert(shadow_request==2192 && d.buffer==5 && d.range==64960 && d.offset==256);
 reset(&s);S.ubo[1][5].addr=0;cbuf_desc(1,&s,&b,&d);
 assert(shadow_calls==0 && d.buffer==1 && d.range==65536);
 reset(&s);pool.hi=pool.lo+4096;S.ubo[1][5]=(p_bind_buf){pool.hi-128,1024};
 s.ubo_extent[5]=64;cbuf_desc(1,&s,&b,&d);
 assert(shadow_calls==0 && copy_request==1024 && cross_count==1 && d.buffer==3 && d.range==1024);
 reset(&s);pool.hi=pool.end=pool.lo+4096;S.ubo[1][5]=(p_bind_buf){pool.hi-128,1024};
 s.ubo_extent[5]=64;cbuf_desc(1,&s,&b,&d);
 assert(shadow_request==64 && d.range==64 && copy_request==0);
 return 0;
}
"""
        target = ROOT / 'target/tests'; target.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='nx2a-ubo-bounds-', dir=target) as folder:
            src = Path(folder) / 'test.c'
            exe = Path(folder) / 'test'
            src.write_text(code)
            subprocess.run(['cc', '-std=c11', '-Wall', '-Wextra', '-Werror',
                            '-fsanitize=address,undefined', str(src), '-o', str(exe)],
                           check=True, capture_output=True, text=True)
            subprocess.run([str(exe)], check=True, capture_output=True, text=True)


if __name__ == '__main__':
    unittest.main()
