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
        header = (dm / 'gfx_vk.h').read_text(encoding='utf-8')
        source = (dm / 'gfx_vk_exec.c').read_text(encoding='utf-8')
        declarations = []
        for first in ('RK_SUPPORT', 'RD_UBO'):
            start = header.index('enum { ' + first)
            declarations.append(header[start:header.index('};', start) + 2])
        declarations.append(next(line for line in header.splitlines()
                                 if line.startswith('typedef struct ') and line.endswith(' dmsv_bind;')))
        start = source.index('static int async_shader_writes(')
        helper = source[start:source.index('\n', start) + 1]   # (one line: transform feedback, or shader_wmem's SPIR-V answer)
        start = source.index('static int pipe_must_draw(')
        # Run the real early eligibility guards. Attachment age and queue state
        # are deliberately treated as eligible, so they cannot hide a lost write.
        gate = source[source.index('\n', start):source.index('    for (int i = 0; i < S.ncol', start)]
        code = r'''
#include <assert.h>
#include <stdint.h>
#include <stddef.h>
DECLARATIONS
typedef struct { int xfb, wmem, nbind; const dmsv_bind *b; } vkshader_t;
typedef struct { int alt, raster_discard; } pkey;
static int g_async_pipe = 1, counting;
static int vk_occ_counting(void) { return counting; }
static int occ_reset_next(void) { return 0; }   // no reset is proven ahead: a counted draw is kept
static int shader_wmem(vkshader_t *s) { return s && s->wmem == 1; }   // gfx_vk_exec.c: spv_wmem's cached answer (1 writes, 2 does not)
static struct { vkshader_t *sh[5]; } S;
HELPER
static int must_draw(const pkey *k) {
GATE
    return 0;
}
static int otherwise_eligible(const pkey *k) { return !must_draw(k); }
int main(void) {
    pkey k = {0};
    vkshader_t empty = {0};
    assert(!async_shader_writes(NULL) && !async_shader_writes(&empty));
    assert(otherwise_eligible(&k));
    counting=1; assert(!otherwise_eligible(&k)); counting=0;
    g_async_pipe = 0; assert(!otherwise_eligible(&k)); g_async_pipe = 1;
    k.raster_discard = 1; assert(!otherwise_eligible(&k)); k.raster_discard = 0;
    // The validated observer pair (pkey.alt 2) is built at once; a generic alternate-code key (alt 1) is not held back.
    k.alt = 2; assert(!otherwise_eligible(&k)); k.alt = 1; assert(otherwise_eligible(&k)); k.alt = 0;

    // A shader that writes memory (shader_wmem: stores and atomics found in its SPIR-V, as Bananza's terrain GS 5d537f7630ec3671 has
    // behind bindings whose reflection flags say nothing) in any graphics stage must prevent asynchronous draw skipping.
    vkshader_t writer = {.wmem=1};
    assert(async_shader_writes(&writer));
    for (unsigned st=0; st<5; st++) {
        S.sh[st] = &writer;
        assert(!otherwise_eligible(&k));
        S.sh[st] = NULL;
    }
    vkshader_t xfb = {.xfb=1,.wmem=2};
    assert(async_shader_writes(&xfb));
    S.sh[0] = &xfb; assert(!otherwise_eligible(&k)); S.sh[0] = NULL;

    // A shader that writes nothing keeps eligibility, whatever its bindings are.
    const dmsv_bind reads[] = {
        {.kind=RK_CBUF,.desc=RD_UBO},
        {.kind=RK_TEX,.desc=RD_SAMPLED,.flags=3},
        {.kind=RK_SBUF,.desc=RD_SSBO}
    };
    vkshader_t readonly = {.wmem=2,.nbind=sizeof reads/sizeof *reads,.b=reads};
    assert(!async_shader_writes(&readonly));
    for (unsigned st=0; st<5; st++) S.sh[st] = &readonly;
    assert(otherwise_eligible(&k));
    return 0;
}
'''.replace('DECLARATIONS', '\n'.join(declarations)).replace('HELPER', helper).replace('GATE', gate)
        target = ROOT / 'target/tests'; target.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='vk-async-side-effects-', dir=target) as folder:
            cfile, exe = Path(folder) / 'check.c', Path(folder) / 'check'
            cfile.write_text(code, encoding='utf-8')
            compile_result = subprocess.run(
                ['cc', '-std=c11', '-Wall', '-Wextra', '-Werror', str(cfile), '-o', str(exe)],
                capture_output=True, text=True)
            self.assertEqual(compile_result.returncode, 0, compile_result.stderr)
            result = subprocess.run([str(exe)], capture_output=True, text=True, timeout=15)
            self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':
    unittest.main()
