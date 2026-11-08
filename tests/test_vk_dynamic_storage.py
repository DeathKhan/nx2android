"""Dynamic storage eligibility, layout types and offset ordering use runtime code."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class DynamicStorage(unittest.TestCase):
    @unittest.skipUnless(shutil.which('cc'), 'a host C compiler is required')
    def test_limits_layouts_and_effective_addresses(self):
        dm = ROOT / 'nxnative/dm/android'
        shader = (dm / 'gfx_vk_shader.c').read_text()
        execute = (dm / 'gfx_vk_exec.c').read_text()
        header = (dm / 'gfx_vk.h').read_text()
        declarations = []
        for first in ('RK_SUPPORT', 'RD_UBO'):
            start = header.index('enum { ' + first)
            declarations.append(header[start:header.index('};', start) + 2])
        declarations.append(next(x for x in header.splitlines() if x.startswith('typedef struct ') and x.endswith(' dmsv_bind;')))
        start = shader.index('static uint32_t storage_dynamic_limit(')
        helpers = shader[start:shader.index('static vkshader_t *vk_shader_q(', start)]
        start = execute.index('static uint32_t storage_dynamic_offset(')
        offset = execute[start:execute.index('\n}\n', start) + 3]
        start = shader.index('    uint32_t sdmax = storage_dynamic_limit(')
        layout = shader[start:shader.index('    s->dsl = shader_layout(', start)]
        start = execute.index('        wr->pBufferInfo = &bi[nw]; wr->descriptorType = b->desc == RD_SSBO')
        bind = execute[start:execute.index('        nw++;', start) + len('        nw++;')]
        start = execute.index('static int set_memo_get(')
        memo_get = execute[start:execute.index('\n}\n', start) + 3]
        memo_type = next(x for x in execute.splitlines() if x.startswith('typedef struct ') and x.endswith(' set_memo_t;'))
        memo_pos = next(x for x in execute.splitlines() if x.strip().startswith('for (int c = 0; c < ncb; c++) { m->cpos[c]')).rstrip()[:-2]
        code = r'''
#include <vulkan/vulkan_core.h>
#include <stdint.h>
#include <assert.h>
#include <string.h>
DECLARATIONS
typedef struct { int nbind,ndyn;const dmsv_bind *b;uint8_t dyn[96]; } vkshader_t;
static struct { int dyn_ssbo;struct {VkPhysicalDeviceLimits limits;} props; } V;
static VkDescriptorType desc_type(int d) {assert(d==RD_UBO || d==RD_SSBO);return d==RD_SSBO?VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;}
static VkShaderStageFlags stage_flag(int stage) {(void)stage;return VK_SHADER_STAGE_VERTEX_BIT;}
#define VKRED_ONCE(...) ((void)0)
#define VKLOG(...) ((void)0)
HELPERS
OFFSET
enum { MEMO_CB = 16 };
MEMO_TYPE
static set_memo_t g_smemo[3];
static uint64_t g_set_epoch,g_bind_gen[6],g_ubo_gen[6],g_tex_gen,g_samp_gen,g_ubd_gen,g_num_um,g_num_im;
static uint32_t g_draw_binst;
static VkDescriptorBufferInfo cbuf_values[16];
static uint32_t vk_pool_gen(void){return 0;}
static void cbuf_desc(int st,vkshader_t *s,const dmsv_bind *b,VkDescriptorBufferInfo *d){(void)st;(void)s;*d=cbuf_values[b->reg];}
MEMO_GET
static uint32_t select_layout(vkshader_t *s,int stage,uint32_t dynmax,VkDescriptorSetLayoutBinding *result) {
LAYOUT
 memcpy(result,lb,nl*sizeof *lb);return nl;
}
static int bind_offsets(vkshader_t *s,VkDescriptorBufferInfo *bi,VkWriteDescriptorSet *w,uint32_t *dofs,uint32_t *bindings,uint8_t *positions) {
 struct {uint32_t b,o;int i;} dp[96];int nd=0;uint32_t nw=0;
 for(int i=0;i<s->nbind;i++) {const dmsv_bind *b=&s->b[i];VkWriteDescriptorSet *wr=&w[nw];
BIND
 }
 for(int j=0;j<nd;j++){dofs[j]=dp[j].o;bindings[j]=dp[j].b;}
 set_memo_t mm={0},*m=&mm;int ncb=0;uint8_t mci[16];
 for(int i=0;i<s->nbind;i++)if(s->b[i].kind==RK_CBUF)mci[ncb++]=(uint8_t)i;
MEMO_POS
 memcpy(positions,m->cpos,(size_t)ncb);
 return nd;
}
int main(void) {
 VkPhysicalDeviceLimits lim={.minStorageBufferOffsetAlignment=64,.maxDescriptorSetStorageBuffersDynamic=8};
 assert(!storage_dynamic_limit(0,0,&lim));
 for(int st=-1;st<=5;st++)assert(storage_dynamic_limit(st,1,&lim)==(st>=0&&st<3?2u:0u));
 for(uint32_t n=0;n<12;n++){lim.maxDescriptorSetStorageBuffersDynamic=n;assert(storage_dynamic_limit(2,1,&lim)==n/3);}
 lim.maxDescriptorSetStorageBuffersDynamic=8;
 VkDeviceSize bad_align[]={0,3,512};
 for(unsigned i=0;i<sizeof bad_align/sizeof *bad_align;i++){lim.minStorageBufferOffsetAlignment=bad_align[i];assert(!storage_dynamic_limit(0,1,&lim));}
 lim.minStorageBufferOffsetAlignment=256;assert(storage_dynamic_limit(0,1,&lim)==2);lim.minStorageBufferOffsetAlignment=64;
 dmsv_bind good={.kind=RK_SBUF,.desc=RD_SSBO,.count=1,.reg=0};
 assert(storage_dynamic_binding(&good,0,2)&&!storage_dynamic_binding(&good,2,2)&&!storage_dynamic_binding(&good,0,0));
 dmsv_bind b=good;b.reg=15;assert(storage_dynamic_binding(&b,0,2));
 b.reg=-1;assert(!storage_dynamic_binding(&b,0,2));b.reg=16;assert(!storage_dynamic_binding(&b,0,2));
 b=good;b.count=2;assert(!storage_dynamic_binding(&b,0,2));b.count=0;assert(!storage_dynamic_binding(&b,0,2));
 b=good;b.desc=RD_UBO;assert(!storage_dynamic_binding(&b,0,2));
 int excluded[]={RK_SBUF_UBO,RK_UBO_PTR,RK_GMEM_MAP,RK_IMG,RK_CBUF};
 for(unsigned i=0;i<sizeof excluded/sizeof *excluded;i++){b=good;b.kind=excluded[i];assert(!storage_dynamic_binding(&b,0,2));}
 VkDescriptorBufferInfo bi={(VkBuffer)(uintptr_t)7,8192,15064};
 assert(storage_dynamic_offset(&bi,64)==8192&&bi.offset==0&&bi.range==15064&&bi.buffer==(VkBuffer)(uintptr_t)7);
 VkDescriptorBufferInfo cases[]={ {(VkBuffer)(uintptr_t)7,UINT64_C(1)<<32,192}, {(VkBuffer)(uintptr_t)7,65,192}, {(VkBuffer)(uintptr_t)7,256,VK_WHOLE_SIZE} };
 for(unsigned i=0;i<sizeof cases/sizeof *cases;i++){bi=cases[i];assert(storage_dynamic_offset(&bi,64)==0&&!memcmp(&bi,&cases[i],sizeof bi));}
 bi=cases[0];bi.offset=256;VkDescriptorBufferInfo unchanged=bi;assert(!storage_dynamic_offset(&bi,0)&&!memcmp(&bi,&unchanged,sizeof bi));
 const dmsv_bind mix[]={
  {.binding=9,.kind=RK_SBUF,.desc=RD_SSBO,.count=1,.reg=0},
  {.binding=2,.kind=RK_CBUF,.desc=RD_UBO,.count=1,.reg=0},
  {.binding=3,.kind=RK_SBUF,.desc=RD_SSBO,.count=1,.reg=1},
  {.binding=7,.kind=RK_CBUF,.desc=RD_UBO,.count=1,.reg=1},
  {.binding=1,.kind=RK_SBUF,.desc=RD_SSBO,.count=1,.reg=2}
 };
 vkshader_t s={.nbind=5,.b=mix};VkDescriptorSetLayoutBinding lb[96];V.props.limits=lim;
 V.dyn_ssbo=0;assert(select_layout(&s,2,2,lb)==5&&s.ndyn==2&&s.dyn[0]==0&&lb[0].descriptorType==VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
 memset(s.dyn,0,sizeof s.dyn);s.ndyn=0;V.dyn_ssbo=1;
 assert(select_layout(&s,2,2,lb)==5&&s.ndyn==4);
 assert(s.dyn[0]==2&&s.dyn[1]==1&&s.dyn[2]==2&&s.dyn[3]==1&&!s.dyn[4]);
 VkDescriptorBufferInfo info[5];VkWriteDescriptorSet writes[5]={0};uint32_t dofs[5],bindings[5];uint8_t positions[2];
 uint64_t offsets[]={4096,256,8192,512,16384};
 for(int i=0;i<5;i++)info[i]=(VkDescriptorBufferInfo){(VkBuffer)(uintptr_t)(20+i),offsets[i],192+i};
 assert(bind_offsets(&s,info,writes,dofs,bindings,positions)==4);
 uint32_t want_bind[]={2,3,7,9},want_offsets[]={256,8192,512,4096};
 assert(!memcmp(bindings,want_bind,sizeof want_bind)&&!memcmp(dofs,want_offsets,sizeof want_offsets));
 for(int i=0;i<5;i++){
  assert(writes[i].descriptorType==lb[i].descriptorType);
  assert(info[i].range==(uint64_t)(192+i)&&info[i].buffer==(VkBuffer)(uintptr_t)(20+i));
  assert(info[i].offset==(i==4?offsets[i]:0));
 }
 assert(positions[0]==0&&positions[1]==2); // SSBO binding 3 precedes the second CBUF.
 set_memo_t *memo=&g_smemo[0];memo->s=&s;memo->ncb=2;memo->ci[0]=1;memo->ci[1]=3;
 memcpy(memo->cpos,positions,2);memcpy(memo->dofs,dofs,sizeof(uint32_t)*4);
 memo->cd[0]=info[1];memo->cd[1]=info[3];cbuf_values[0]=info[1];cbuf_values[1]=info[3];
 cbuf_values[0].offset=1280;cbuf_values[1].offset=2816;g_ubo_gen[0]++;
 assert(set_memo_get(0,&s,dofs)&&dofs[0]==1280&&dofs[1]==8192&&dofs[2]==2816&&dofs[3]==4096);
 g_bind_gen[0]++; // A changed storage binding must not reuse cached dynamic offsets.
 assert(!set_memo_get(0,&s,dofs));
 // A dynamic storage layout can keep a large base offset and bind dynamic zero.
 info[0].offset=UINT64_C(1)<<32;assert(bind_offsets(&s,info,writes,dofs,bindings,positions)==4);
 assert(info[0].offset==(UINT64_C(1)<<32)&&dofs[3]==0&&writes[0].descriptorType==VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC);
 return 0;
}
'''
        for name, value in [('DECLARATIONS', '\n'.join(declarations)), ('HELPERS', helpers), ('OFFSET', offset), ('LAYOUT', layout), ('BIND', bind), ('MEMO_TYPE', memo_type), ('MEMO_GET', memo_get), ('MEMO_POS', memo_pos)]:
            code = code.replace(name, value)
        target = ROOT / 'target/tests'; target.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='dynamic-storage-', dir=target) as folder:
            src, exe = Path(folder) / 'check.c', Path(folder) / 'check'
            src.write_text(code)
            subprocess.run(['cc', '-std=c11', '-Wall', '-Wextra', '-Werror', '-fsanitize=address,undefined',
                            '-I' + str(ROOT / 'nxnative/dm/third_party/vulkan-headers/include'), str(src), '-o', str(exe)], check=True)
            subprocess.run([str(exe)], check=True, timeout=15)


if __name__ == '__main__':
    unittest.main()
