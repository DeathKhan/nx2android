"""Opt-in original-index replay requires the actual draw-input signature."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
import sys; sys.path.insert(0, str(Path(__file__).resolve().parent))
import host_cc   # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


class OriginalIndex(unittest.TestCase):
    @unittest.skipUnless(shutil.which('cc'), 'a host C compiler is required')
    @unittest.skipUnless(host_cc.posix() and host_cc.sanitizers(), 'a POSIX host C compiler with ASan and UBSan is required (MinGW has no setenv, libasan or libubsan)')
    def test_capture_match_fallback_and_submission_lifetime(self):
        dm = ROOT / 'nxnative/dm'
        execute = (dm / 'android/gfx_vk_exec.c').read_text(encoding='utf-8')
        header = (dm / 'android/gfx_vk.h').read_text(encoding='utf-8')
        obj = (dm / 'gfx/nvn/nvn_obj.h').read_text(encoding='utf-8')
        decl = []
        for first in ('RK_SUPPORT', 'RD_UBO', 'RS_NONE'):
            at = header.index('enum { ' + first)
            decl.append(header[at:header.index('};', at) + 2])
        decl.append(next(x for x in header.splitlines() if x.startswith('typedef struct ') and x.endswith(' dmsv_bind;')))
        for name in ('va_st', 'vs_st', 'poly_st', 'p_scissor', 'p_bind_buf', 'p_draw_idx'):
            line = next(x for x in obj.splitlines() if x.startswith('typedef struct ') and ('} ' + name + ';') in x)
            decl.append(line[:line.index('} ' + name + ';') + len('} ' + name + ';')])
        start = execute.index('enum { OI_MAX')
        helper = execute[start:execute.index('static void draw_idx(', start)]
        a = helper.index('// BEGIN original-index GPU snapshot diagnostic')
        z = helper.index('// END original-index GPU snapshot diagnostic', a)
        helper = helper[:a] + 'static int snapshots; static uint64_t vk_fence_done(void){return 0;} static void original_index_dump_poll(uint64_t d){(void)d;} static void original_index_snapshot(const original_index_entry *r,const p_draw_idx *d){(void)r;(void)d;snapshots++;}\n' + helper[z:]
        code = r'''
#define _POSIX_C_SOURCE 200809L
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
DECL
typedef struct {uint64_t address,size;} NVNbufferRange;
typedef struct {uint64_t serial,off;void *pool;} tex_t;
typedef struct {uint32_t first,count;} view_t;
typedef struct {uint64_t key;int xfb,nbind;uint32_t local[3],c1_len;const uint8_t *c1;const dmsv_bind *b;} vkshader_t;
static struct {vkshader_t *sh[6];int rt_has_dep,rast_discard,rt_has_dview,sc_set,prim_restart,depth_clamp;
 uint32_t vattr[16];NVNbufferRange vbuf[16];vs_st vstream[16];p_bind_buf ubo[6][16],ssbo[6][16];
 uint64_t tex[6][256],tex_sseq[6][256];tex_t rt_dep;view_t rt_dview;float vp[16][4],dnear,dfar;p_scissor sc;poly_st poly;uint8_t push[2][256];
 struct{int dt,dw,df;}ds;} S;
static int g_tf_on,g_lower_left,g_flipy;static uint64_t g_tex_frame;
static uint8_t memory[4096];static tex_t texture={.serial=7};static int log_count;
static void mock_log(const char *f,...){(void)f;log_count++;}
#define VKLOG(...) mock_log(__VA_ARGS__)
static void *vk_pool_peek(uint64_t a,uint64_t n){return a>=0x1000&&a<=0x2000&&n<=0x2000-a?memory:NULL;}
static void vk_ubd_read(uint64_t a,uint64_t n,uint8_t *p){assert(vk_pool_peek(a,n));memcpy(p,memory+a-0x1000,n);}
static int vk_tex_by_id(int id,const view_t **v,const tex_t **t){*v=NULL;*t=&texture;return id==42;}
HELPER
static const dmsv_bind vb[]={{.kind=RK_SUPPORT},{.kind=RK_CBUF,.reg=0,.count=1},{.kind=RK_TEX,.desc=RD_SAMPLED,.reg=0,.count=1,.samp=RS_TEX,.dim=1}};
static const dmsv_bind gb[]={{.kind=RK_SBUF,.desc=RD_SSBO,.reg=0,.count=1},{.kind=RK_SBUF,.desc=RD_SSBO,.reg=1,.count=1}};
static vkshader_t producer={.key=0x33,.nbind=3,.local={1,0,0},.b=vb},consumer={.key=0x22,.nbind=3,.local={1,0,0},.b=vb},gs={.key=0x11,.nbind=2,.b=gb};
static void producer_state(void){S.sh[0]=&producer;S.sh[2]=&gs;S.ds.dt=1;S.ds.dw=1;S.ds.df=2;}
static void consumer_state(void){S.sh[0]=&consumer;S.sh[2]=NULL;S.ds.dt=1;S.ds.dw=0;S.ds.df=3;}
int main(void){
 p_draw_idx direct={.prim=4,.itype=1,.count=3,.instances=1,.ibuf=0x1080};
 p_draw_idx indirect={.prim=4,.itype=1,.instances=1,.ibuf=0x1200,.ind=0x1100},out;
 S.rt_has_dep=1;S.rt_dep=(tex_t){.serial=10,.off=16,.pool=(void*)1};S.vbuf[0]=(NVNbufferRange){0x1000,64};S.vstream[0].stride=16;
 va_st attr={.fmt=1};memcpy(&S.vattr[0],&attr,sizeof attr);S.ubo[0][0]=(p_bind_buf){.addr=0x1040,.size=32};
 S.ssbo[2][0]=(p_bind_buf){.addr=0x1100,.size=20};S.ssbo[2][1]=(p_bind_buf){.addr=0x1200,.size=64};S.tex[0][0]=42;
 original_index_read();producer_state();original_index_capture(&direct);assert(!g_oi_n);
 setenv("DM_VK_ORIGINAL_INDEX","11:22",1);original_index_read();original_index_capture(&direct);assert(g_oi_n==1);
 consumer_state();assert(original_index_replay(&indirect,&out)&&!memcmp(&out,&direct,sizeof direct));
 indirect.ind++;assert(!original_index_replay(&indirect,&out));indirect.ind--;
 indirect.ibuf++;assert(!original_index_replay(&indirect,&out));indirect.ibuf--;
 S.vbuf[0].address+=16;assert(!original_index_replay(&indirect,&out));S.vbuf[0].address-=16;
 memory[0]++;assert(!original_index_replay(&indirect,&out));memory[0]--;
 memory[64]++;assert(!original_index_replay(&indirect,&out));memory[64]--;
 memory[128]++;assert(!original_index_replay(&indirect,&out));memory[128]--;
 texture.serial++;assert(!original_index_replay(&indirect,&out));texture.serial--;
 S.tex_sseq[0][0]++;assert(!original_index_replay(&indirect,&out));S.tex_sseq[0][0]--;
 S.vp[0][0]=1;assert(!original_index_replay(&indirect,&out));S.vp[0][0]=0;
 S.push[0][0]=1;assert(!original_index_replay(&indirect,&out));S.push[0][0]=0;
 S.rt_dep.serial++;assert(!original_index_replay(&indirect,&out));S.rt_dep.serial--;
 S.ds.df=4;assert(!original_index_replay(&indirect,&out));S.ds.df=3;
 assert(original_index_replay(&indirect,&out));
 setenv("DM_VK_ORIGINAL_INDEX_DUMP","11:22",1);original_index_read();producer_state();original_index_capture(&direct);consumer_state();
 assert(!original_index_replay(&indirect,&out)&&snapshots==1); // Snapshot mode validates the source but never substitutes draw data.
 setenv("DM_VK_ORIGINAL_INDEX_DUMP","0",1);original_index_read();producer_state();original_index_capture(&direct);consumer_state();
 assert(original_index_replay(&indirect,&out));
 producer_state();p_draw_idx unsupported=direct;unsupported.instances=2;original_index_capture(&unsupported);
 consumer_state();assert(!original_index_replay(&indirect,&out)); // The newer unsupported writer invalidates an earlier source.
 producer_state();original_index_capture(&direct);consumer_state();assert(original_index_replay(&indirect,&out));
 original_index_read();assert(!original_index_replay(&indirect,&out)); // Never cross submissions.
 setenv("DM_VK_ORIGINAL_INDEX","11:22junk",1);original_index_read();assert(!g_oi_gs&&!g_oi_vs);
 setenv("DM_VK_ORIGINAL_INDEX","0",1);original_index_read();assert(!g_oi_gs&&!g_oi_vs);
 assert(log_count>0);return 0;
}
'''
        code = code.replace('DECL', '\n'.join(decl)).replace('HELPER', helper)
        target = ROOT / 'target/tests'; target.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='original-index-', dir=target) as folder:
            src, exe = Path(folder) / 'check.c', Path(folder) / 'check'
            src.write_text(code, encoding='utf-8')
            subprocess.run(['cc', '-std=gnu11', '-Wall', '-Wextra', '-Werror', '-fsanitize=address,undefined',
                            str(src), '-o', str(exe)], check=True)
            subprocess.run([str(exe)], check=True, timeout=15)


if __name__ == '__main__':
    unittest.main()
