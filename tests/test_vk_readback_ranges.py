"""Conservative eligibility and transfer-range coverage for opt-in readback."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


class ReadbackRanges(unittest.TestCase):
    @unittest.skipUnless(shutil.which('cc'), 'a host C compiler is required')
    def test_eligibility_and_range_capacity(self):
        dm = Path(__file__).resolve().parents[1] / 'nxnative/dm/android'
        core = (dm/'gfx_vk_core.c').read_text(encoding='utf-8')
        exec_src = (dm/'gfx_vk_exec.c').read_text(encoding='utf-8')
        header = (dm/'gfx_vk.h').read_text(encoding='utf-8')
        def enum_declaration(first_member):
            start = header.index('enum { ' + first_member)
            return header[start:header.index('};', start) + 2]
        # Reflection kinds span multiple lines; include their complete declaration.
        enums = '\n'.join(enum_declaration(name) for name in ('RK_SUPPORT', 'RD_UBO'))
        controls = '\n'.join(line for line in core.splitlines() if line.startswith(('int vk_readback_ranges(', 'void vk_readback_unknown(')))
        start = core.index('static void sub_gpu_wrote(uint64_t a, uint64_t n) {')
        ranges = core[start:core.index('\n}\n', start)+3]
        start = exec_src.index('static VkDescriptorSet stage_set(')
        gate = exec_src[exec_src.index('\n', start):exec_src.index('    int memo =', start)]
        # The gate must execute before descriptor memo/cache returns.
        self.assertIn('vk_readback_ranges()', gate)
        for marker in ['vk_gpu_wrote((uint64_t)(uintptr_t)t->mem, len); vk_readback_unknown();',
                       'vk_gpu_wrote(c->src, p ? p->hi - c->src : 1); vk_readback_unknown();',
                       'vk_readback_unknown();   // capture extents']:
            self.assertIn(marker, exec_src)
        harness = r'''
#include <assert.h>
#include <stdint.h>
#include <string.h>
ENUMS
enum { GW_SUB=64 };
typedef struct { struct { uint64_t lo,hi; } gw[64]; int ngw,readback_ranges; } vksub_t;
static vksub_t g_slot[2]; static int g_cur=-1;
typedef struct { int kind,desc,flags; } binding;
typedef struct { int nbind; binding b[2]; } vkshader_t;
typedef struct { int used; } set_profile_t;
static set_profile_t *set_profile_row(int st, const vkshader_t *s) { (void)st; (void)s; return NULL; }
CONTROLS
RANGES
static void observe(vkshader_t *s) {
    int st = 0;
GATE
    (void)sp;
}
int main(void) {
    assert(!vk_readback_ranges()); vk_readback_unknown();
    g_cur=0; g_slot[0].readback_ranges=1;
    int readonly[]={RD_UBO,RD_SAMPLED,RD_TBUF,RD_IMAGE,RD_SAMPLER};
    for (unsigned i=0;i<sizeof readonly/sizeof *readonly;i++) {
        vkshader_t s={.nbind=1,.b={{.kind=RK_CBUF,.desc=readonly[i]}}}; observe(&s); assert(vk_readback_ranges());
    }
    int writes[]={RD_SSBO,RD_STORAGE,RD_STORAGE_TBUF,255};
    for (unsigned i=0;i<sizeof writes/sizeof *writes;i++) {
        g_slot[0].readback_ranges=1; vkshader_t s={.nbind=1,.b={{.kind=RK_UBO_PTR,.desc=writes[i]}}};
        observe(&s); assert(!vk_readback_ranges()); observe(&s); assert(!vk_readback_ranges());
    }
    int kinds[]={RK_SBUF,RK_SBUF_UBO,RK_IMG,RK_IMG_UBO,RK_IMG_TBUF};
    for (unsigned i=0;i<sizeof kinds/sizeof *kinds;i++) {
        g_slot[0].readback_ranges=1; vkshader_t s={.nbind=1,.b={{.kind=kinds[i],.desc=RD_UBO}}};
        observe(&s); assert(!vk_readback_ranges());
    }
    // The raw-memory write bit must invalidate range tracking even if its
    // descriptor classification alone would otherwise allow a memo hit.
    for (int flags=0;flags<=4;flags+=4) {
        g_slot[0].readback_ranges=1;
        vkshader_t s={.nbind=1,.b={{.kind=RK_GMEM_MAP,.desc=RD_UBO,.flags=flags}}};
        observe(&s); assert(vk_readback_ranges()==!flags);
    }
    g_slot[0].readback_ranges=1;
    vkshader_t mixed={.nbind=2,.b={{.kind=RK_CBUF,.desc=RD_UBO},
                                 {.kind=RK_GMEM_MAP,.desc=RD_SSBO,.flags=4}}};
    observe(&mixed); assert(!vk_readback_ranges());
    memset(g_slot,0,sizeof g_slot); g_slot[0].readback_ranges=1;
    sub_gpu_wrote(0,0); assert(!g_slot[0].ngw);
    for (uint64_t i=0;i<100;i++) sub_gpu_wrote(1000+i*1000000,37);
    assert(g_slot[0].ngw<=GW_SUB);
    for (uint64_t i=0;i<100;i++) {
        uint64_t a=1000+i*1000000; int found=0;
        for (int j=0;j<g_slot[0].ngw;j++) found|=g_slot[0].gw[j].lo<=a && g_slot[0].gw[j].hi>=a+37;
        assert(found);
    }
    vk_readback_unknown(); assert(!vk_readback_ranges());
}
'''.replace('ENUMS', enums).replace('CONTROLS', controls).replace('RANGES', ranges).replace('GATE', gate)
        target = dm.parents[2] / 'target/tests'; target.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='nx2a-readback-ranges-', dir=target) as tmp:
            c, exe = Path(tmp)/'check.c', Path(tmp)/'check'
            c.write_text(harness, encoding='utf-8')
            subprocess.run(['cc', '-std=gnu11', '-Wall', '-Wextra', '-Werror', str(c), '-o', str(exe)], check=True)
            subprocess.run([str(exe)], check=True, timeout=15)
