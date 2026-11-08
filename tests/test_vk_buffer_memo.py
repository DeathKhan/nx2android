"""Refresh mutable buffer dependencies using the actual descriptor/memo helpers."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


def function(source, declaration):
    start = source.index(declaration)
    pos = source.index('{', start) + 1
    depth = 1
    while depth:
        depth += (source[pos] == '{') - (source[pos] == '}')
        pos += 1
    return source[start:pos]


class BufferMemo(unittest.TestCase):
    @unittest.skipUnless(shutil.which('cc'), 'a host C compiler is required')
    def test_actual_mutable_descriptor_dependencies(self):
        dm = ROOT / 'nxnative/dm'
        source = (dm / 'android/gfx_vk_exec.c').read_text()
        header = (dm / 'android/gfx_vk.h').read_text()
        decl = []
        for first in ('RK_SUPPORT', 'RD_UBO'):
            begin = header.index('enum { ' + first)
            decl.append(header[begin:header.index('};', begin) + 2])
        decl.append(next(x for x in header.splitlines() if x.startswith('typedef struct ') and x.endswith(' dmsv_bind;')))
        memo_type = next(x for x in source.splitlines() if x.startswith('typedef struct ') and x.endswith(' set_memo_t;'))
        ssbo_case = source[source.index('        case OP_BIND_SSBO:'):source.index('        case OP_BIND_TEX:', source.index('        case OP_BIND_SSBO:'))]
        code = r"""
#define VK_NO_PROTOTYPES
#include <vulkan/vulkan_core.h>
#include <stdint.h>
#include <string.h>
#include <assert.h>
DECL
enum { MEMO_CB = 48, OP_BIND_SSBO = 123, SB_ALIGN = 256 };
typedef struct { int stage,index;uint64_t addr,size; } p_bind_buf;
typedef struct { int nbind,ndyn,setmemo;const dmsv_bind *b;uint8_t dyn[96],tbl[96]; } vkshader_t;
typedef struct { uint64_t lo,hi,end;VkBuffer buf; } vkpool_t;
static struct { p_bind_buf ssbo[6][16],ubo[6][16]; } S;
static struct { struct { VkPhysicalDeviceLimits limits; } props; } V;
MEMO_TYPE
static set_memo_t g_smemo[3];
static uint64_t g_set_epoch=1,g_bind_gen[6],g_ssbo_gen[6],g_ubo_gen[6],g_ubd_gen,g_tex_gen,g_samp_gen,g_num_um,g_num_im;
static uint32_t g_draw_binst,pool_gen;
static int g_buffer_memo=1,cb_reads,driver_reads;
static VkDescriptorBufferInfo g_draw_claim,cb_value,driver_value;
static vkpool_t pool={0x100000,0x200000,0x200000,(VkBuffer)(uintptr_t)3};
static uint32_t vk_pool_gen(void){return pool_gen;}
static VkBuffer vk_zero_buffer(void){return (VkBuffer)(uintptr_t)99;}
static vkpool_t *vk_pool_of(uint64_t a,int late){(void)late;return a>=pool.lo&&a<pool.hi?&pool:NULL;}
static void vk_cross(int k,uint64_t a,uint64_t n,const vkpool_t *p){(void)k;(void)a;(void)n;(void)p;}
static uint32_t ubo_word(int st,int reg,int off){(void)st;(void)reg;(void)off;assert(0&&"fixture has no uniform-held storage address");return 0;}
void vk_gpu_wrote(uint64_t a,uint64_t n){(void)a;(void)n;}
#define VKRED_ONCE(...) ((void)0)
static void cbuf_desc(int st,const vkshader_t *s,const dmsv_bind *b,VkDescriptorBufferInfo *d){(void)st;(void)s;(void)b;cb_reads++;*d=cb_value;}
static void drv_shared(int st,int bank,VkBuffer *buf,VkDeviceSize *off,int dyn){assert(st==2&&bank==0&&!dyn);driver_reads++;*buf=driver_value.buffer;*off=driver_value.offset;}
SBUF
OFFSET
BUFFER
OK
GET
static void bind_storage(p_bind_buf value) {
    int op=OP_BIND_SSBO;const void *a=&value;
    switch(op) {
SSBO_CASE
    default:assert(0);
    }
}
static const dmsv_bind bindings[]={
 {.binding=0,.kind=RK_SUPPORT,.desc=RD_UBO},
 {.binding=1,.kind=RK_DRV,.reg=0,.desc=RD_UBO},
 {.binding=4,.kind=RK_CBUF,.reg=3,.desc=RD_UBO},
 {.binding=2,.kind=RK_SBUF,.reg=0,.desc=RD_SSBO},
 {.binding=3,.kind=RK_SBUF,.reg=1,.desc=RD_SSBO},
 {.binding=95,.kind=RK_GS_CLAIM,.desc=RD_SSBO}
};
static vkshader_t shader={.nbind=6,.ndyn=5,.b=bindings,.dyn={0,1,1,2,2,2}};
static void baseline(void) {
 memset(g_smemo,0,sizeof g_smemo);memset(g_bind_gen,0,sizeof g_bind_gen);
 memset(g_ssbo_gen,0,sizeof g_ssbo_gen);memset(g_ubo_gen,0,sizeof g_ubo_gen);
 g_tex_gen=g_samp_gen=g_ubd_gen=0;pool_gen=g_draw_binst=0;g_set_epoch=1;g_buffer_memo=1;
 memset(&S,0,sizeof S);pool.buf=(VkBuffer)(uintptr_t)3;
 V.props.limits.minStorageBufferOffsetAlignment=64;
 S.ssbo[2][0]=(p_bind_buf){2,0,pool.lo+256,192};S.ssbo[2][1]=(p_bind_buf){2,1,pool.lo+8192,15128};
 cb_value=(VkDescriptorBufferInfo){(VkBuffer)(uintptr_t)2,512,2432};
 driver_value=(VkDescriptorBufferInfo){(VkBuffer)(uintptr_t)1,256,4096};
 g_draw_claim=(VkDescriptorBufferInfo){(VkBuffer)(uintptr_t)4,4096,2048};
 shader.setmemo=0;assert(set_memo_ok(&shader)&&shader.setmemo==3);
 set_memo_t *m=&g_smemo[2];m->s=&shader;m->epoch=1;m->ncb=5;m->set=(VkDescriptorSet)(uintptr_t)8;
 uint8_t ci[]={1,2,3,4,5},pos[]={0,3,1,2,4};memcpy(m->ci,ci,5);memcpy(m->cpos,pos,5);
 m->cd[0]=(VkDescriptorBufferInfo){driver_value.buffer,0,4096};
 m->cd[1]=(VkDescriptorBufferInfo){cb_value.buffer,0,2432};
 m->cd[2]=(VkDescriptorBufferInfo){pool.buf,0,192};
 m->cd[3]=(VkDescriptorBufferInfo){pool.buf,0,15128};
 m->cd[4]=(VkDescriptorBufferInfo){g_draw_claim.buffer,0,2048};
 uint32_t offsets[]={256,256,8192,512,4096};memcpy(m->dofs,offsets,sizeof offsets);
 cb_reads=driver_reads=0;
}
int main(void) {
 uint32_t dofs[96];baseline();set_memo_t *m=&g_smemo[2];
 // Every draw refreshes the disjoint claim range even without a guest generation change.
 g_draw_claim.offset=8192;assert(set_memo_get(2,&shader,dofs)&&dofs[4]==8192&&!cb_reads&&!driver_reads);
 // Actual SSBO record handler changes only the storage generation; equal rebinds do nothing.
 p_bind_buf update=S.ssbo[2][0];update.addr+=256;bind_storage(update);
 assert(g_ssbo_gen[2]==1&&!g_bind_gen[2]);bind_storage(update);assert(g_ssbo_gen[2]==1);
 driver_value.offset=4352;assert(set_memo_get(2,&shader,dofs)&&dofs[0]==4352&&dofs[1]==512&&dofs[2]==8192&&dofs[3]==512&&dofs[4]==8192);
 assert(driver_reads==1&&!cb_reads&&m->ssbo==1);
 // c0 includes every storage slot, including slots absent from direct SSBO reflection.
 update=(p_bind_buf){2,15,pool.lo+16384,777};bind_storage(update);driver_value.offset+=4096;
 assert(set_memo_get(2,&shader,dofs)&&driver_reads==2&&dofs[0]==8448);
 // Guest constants refresh independently while storage/driver offsets remain valid.
 cb_value.offset=1024;g_ubo_gen[2]++;assert(set_memo_get(2,&shader,dofs)&&cb_reads==1&&driver_reads==2&&dofs[3]==1024);
 cb_value.buffer=(VkBuffer)(uintptr_t)42;g_ubd_gen++;assert(!set_memo_get(2,&shader,dofs));
 baseline();update=S.ssbo[2][0];update.size=64;bind_storage(update);assert(!set_memo_get(2,&shader,dofs)); // range exact
 baseline();g_ssbo_gen[2]++;pool.buf=(VkBuffer)(uintptr_t)44;assert(!set_memo_get(2,&shader,dofs)); // buffer exact
 baseline();g_draw_claim.range*=2;assert(!set_memo_get(2,&shader,dofs));
 baseline();g_draw_claim.buffer=(VkBuffer)(uintptr_t)45;assert(!set_memo_get(2,&shader,dofs));
 baseline();g_draw_claim=(VkDescriptorBufferInfo){0};assert(!set_memo_get(2,&shader,dofs)); // disabled4B
 baseline();g_draw_claim.offset=65;assert(!set_memo_get(2,&shader,dofs)); // unaligned dynamic fallback retains static offset
 baseline();g_draw_claim.offset=UINT64_C(1)<<32;assert(!set_memo_get(2,&shader,dofs));
 m->cd[4].offset=g_draw_claim.offset;assert(set_memo_get(2,&shader,dofs)&&dofs[4]==0);
 baseline();g_draw_claim.range=VK_WHOLE_SIZE;assert(!set_memo_get(2,&shader,dofs));
 // Static direct SSBO offsets cannot be silently moved into a dynamic bind.
 baseline();shader.dyn[3]=0;m->cpos[2]=0xff;m->cd[2].offset=256;
 update=S.ssbo[2][0];update.addr+=256;bind_storage(update);assert(!set_memo_get(2,&shader,dofs));shader.dyn[3]=2;
 // Every original guard remains conservative.
 for(int which=0;which<8;which++){baseline();switch(which){case0:break;default:break;}}
 baseline();g_bind_gen[2]++;assert(!set_memo_get(2,&shader,dofs));
 baseline();g_tex_gen++;assert(!set_memo_get(2,&shader,dofs));
 baseline();g_samp_gen++;assert(!set_memo_get(2,&shader,dofs));
 baseline();pool_gen++;assert(!set_memo_get(2,&shader,dofs));
 baseline();g_draw_binst++;assert(!set_memo_get(2,&shader,dofs));
 baseline();g_set_epoch++;assert(!set_memo_get(2,&shader,dofs));
 baseline();g_buffer_memo=0;assert(!set_memo_ok(&shader));g_ssbo_gen[2]++;assert(!set_memo_get(2,&shader,dofs)&&!driver_reads);
 // Unsupported resource dependencies are never admitted, including UBO-derived handles/addresses.
 int bad[]={RK_TEXD,RK_SAMPD,RK_SBUF_UBO,RK_UBO_PTR,RK_TEX_UBO,RK_IMG_UBO,RK_TBUF_UBO};
 g_buffer_memo=1;
 for(unsigned i=0;i<sizeof bad/sizeof *bad;i++){dmsv_bind b={.kind=bad[i]};vkshader_t sh={.nbind=1,.b=&b};assert(!set_memo_ok(&sh));}
 dmsv_bind sampler={.kind=RK_TEX,.samp=RS_UBO};vkshader_t sh={.nbind=1,.b=&sampler};assert(!set_memo_ok(&sh));
 dmsv_bind many[49];for(int i=0;i<49;i++)many[i]=(dmsv_bind){.kind=RK_CBUF};sh=(vkshader_t){.nbind=49,.b=many};assert(!set_memo_ok(&sh));
 return 0;
}
"""
        code = code.replace(' for(int which=0;which<8;which++){baseline();switch(which){case0:break;default:break;}}','')
        for name, value in [('DECL','\n'.join(decl)),('MEMO_TYPE',memo_type),('SSBO_CASE',ssbo_case),
                            ('SBUF',function(source,'static void sbuf_desc(')),
                            ('OFFSET',function(source,'static uint32_t storage_dynamic_offset(')),
                            ('BUFFER',function(source,'static int memo_buffer(')),
                            ('OK',function(source,'static int set_memo_ok(')),
                            ('GET',function(source,'static int set_memo_get('))]:
            code = code.replace('\n'+name+'\n','\n'+value+'\n')
        # RS_UBO is a separate reflection enum.
        code = code.replace('enum { MEMO_CB', 'enum { RS_UBO = 4 };\nenum { MEMO_CB')
        target = ROOT / 'target/tests';target.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='vk-buffer-memo-',dir=target) as folder:
            src,exe=Path(folder)/'check.c',Path(folder)/'check';src.write_text(code)
            subprocess.run(['cc','-std=c11','-O1','-g','-Wall','-Wextra','-Werror','-fsanitize=address,undefined',
                            '-I'+str(dm/'third_party/vulkan-headers/include'),str(src),'-o',str(exe)],check=True)
            subprocess.run([str(exe)],check=True,timeout=15)


if __name__=='__main__':unittest.main()
