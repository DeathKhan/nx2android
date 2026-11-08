"""Async compilation must preserve draws that can produce later draws' inputs."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class AsyncSideEffects(unittest.TestCase):
    @unittest.skipUnless(shutil.which('cc'), 'a host C compiler is required')
    def test_shader_side_effects_prevent_draw_skips(self):
        dm = ROOT / 'nxnative/dm/android'
        header = (dm / 'gfx_vk.h').read_text()
        source = (dm / 'gfx_vk_exec.c').read_text()
        declarations = []
        for first in ('RK_SUPPORT', 'RD_UBO'):
            start = header.index('enum { ' + first)
            declarations.append(header[start:header.index('};', start) + 2])
        declarations.append(next(line for line in header.splitlines()
                                 if line.startswith('typedef struct ') and line.endswith(' dmsv_bind;')))
        start = source.index('static int async_shader_writes(')
        helper = source[start:source.index('static int async_pipe(', start)]
        start = source.index('static int async_pipe(')
        # Run the real early eligibility guards. Attachment age and queue state
        # are deliberately treated as eligible, so they cannot hide a lost write.
        gate = source[source.index('\n', start):source.index('    for (int i = 0; i < S.ncol', start)]
        code = r'''
#include <assert.h>
#include <stdint.h>
#include <stddef.h>
DECLARATIONS
typedef struct { int xfb, nbind; const dmsv_bind *b; } vkshader_t;
typedef struct { int raster_discard; } pkey;
static int g_async_pipe = 1, counting;
static int vk_occ_counting(void) { return counting; }
static struct { vkshader_t *sh[5]; } S;
HELPER
static int otherwise_eligible(const pkey *k) {
GATE
    return 1;
}
int main(void) {
    pkey k = {0};
    vkshader_t empty = {0};
    assert(!async_shader_writes(NULL) && !async_shader_writes(&empty));
    assert(otherwise_eligible(&k));
    counting=1; assert(!otherwise_eligible(&k)); counting=0;
    g_async_pipe = 0; assert(!otherwise_eligible(&k)); g_async_pipe = 1;
    k.raster_discard = 1; assert(!otherwise_eligible(&k)); k.raster_discard = 0;

    // Reflection values from shipped Bananza terrain GS 5d537f7630ec3671:
    // both storage bindings have flags=0 despite OpAtomicIAdd and
    // OpAtomicCompareExchange. No game binary is needed for this regression.
    const dmsv_bind terrain[] = {
        {.kind=RK_SUPPORT,.desc=RD_UBO,.binding=0,.count=1,.reg=-1},
        {.kind=RK_DRV,.desc=RD_UBO,.binding=1,.count=1,.reg=0},
        {.kind=RK_CBUF,.desc=RD_UBO,.binding=4,.count=1,.reg=0},
        {.kind=RK_SBUF,.desc=RD_SSBO,.binding=2,.count=1,.reg=0},
        {.kind=RK_SBUF,.desc=RD_SSBO,.binding=3,.count=1,.reg=1}
    };
    vkshader_t terrain_shader = {.nbind=5,.b=terrain};
    assert(async_shader_writes(&terrain_shader));
    // A writer in any graphics stage must prevent asynchronous draw skipping.
    for (unsigned st=0; st<5; st++) {
        S.sh[st] = &terrain_shader;
        assert(!otherwise_eligible(&k));
        S.sh[st] = NULL;
    }

    const dmsv_bind writes[] = {
        {.kind=RK_SBUF,.desc=RD_SSBO},
        {.kind=RK_SBUF_UBO,.desc=RD_SSBO},
        {.kind=RK_GMEM_MAP,.desc=RD_SSBO},
        {.kind=RK_GMEM_MAP,.desc=RD_SSBO,.flags=4},
        {.kind=RK_IMG,.desc=RD_STORAGE},
        {.kind=RK_IMG_UBO,.desc=RD_STORAGE},
        {.kind=RK_IMG_TBUF,.desc=RD_STORAGE_TBUF},
        {.kind=RK_UNKNOWN,.desc=RD_SSBO},
        {.kind=RK_UNKNOWN,.desc=RD_UBO,.flags=4}
    };
    for (unsigned i=0; i<sizeof writes/sizeof *writes; i++) {
        vkshader_t shader = {.nbind=1,.b=&writes[i]};
        assert(async_shader_writes(&shader));
    }
    vkshader_t xfb = {.xfb=1};
    assert(async_shader_writes(&xfb));
    S.sh[0] = &xfb; assert(!otherwise_eligible(&k)); S.sh[0] = NULL;

    // Ordinary uniforms, sampled textures and samplers retain eligibility;
    // array/multisample flags must not be confused with the write flag.
    const dmsv_bind reads[] = {
        {.kind=RK_CBUF,.desc=RD_UBO},
        {.kind=RK_TEX,.desc=RD_SAMPLED,.flags=3},
        {.kind=RK_TBUF,.desc=RD_TBUF},
        {.kind=RK_TEXD,.desc=RD_IMAGE},
        {.kind=RK_SAMPD,.desc=RD_SAMPLER}
    };
    vkshader_t readonly = {.nbind=sizeof reads/sizeof *reads,.b=reads};
    assert(!async_shader_writes(&readonly));
    for (unsigned st=0; st<5; st++) S.sh[st] = &readonly;
    assert(otherwise_eligible(&k));
    return 0;
}
'''.replace('DECLARATIONS', '\n'.join(declarations)).replace('HELPER', helper).replace('GATE', gate)
        target = ROOT / 'target/tests'; target.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='vk-async-side-effects-', dir=target) as folder:
            cfile, exe = Path(folder) / 'check.c', Path(folder) / 'check'
            cfile.write_text(code)
            compile_result = subprocess.run(
                ['cc', '-std=c11', '-Wall', '-Wextra', '-Werror', str(cfile), '-o', str(exe)],
                capture_output=True, text=True)
            self.assertEqual(compile_result.returncode, 0, compile_result.stderr)
            result = subprocess.run([str(exe)], capture_output=True, text=True, timeout=15)
            self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':
    unittest.main()
