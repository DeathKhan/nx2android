"""DM_VK_LOAD_SKIP (gfx_vk_exec.c): which fragment shaders can leave a pixel unwritten, the channels a format needs written, and how a first draw's count proves or refuses
its pass's signature, cut from the source."""
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from porter.toolchain import find_ndk
sys.path.insert(0, str(Path(__file__).resolve().parent))
import host_cc   # noqa: E402

EXEC = Path(__file__).resolve().parents[1] / 'nxnative/dm/android/gfx_vk_exec.c'
HARNESS = r'''
#define VK_NO_PROTOTYPES
#include <vulkan/vulkan_core.h>
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
CUTS
static uint32_t words[64]; static uint32_t nw;
static void spv(void) { nw = 0; uint32_t h[5] = {0x07230203, 0x10000, 0, 16, 0}; memcpy(words, h, sizeof h); nw = 5; }
static void op(uint32_t o, uint32_t n, const uint32_t *a) { words[nw++] = (n + 1) << 16 | o; for (uint32_t i = 0; i < n; i++) words[nw++] = a[i]; }
int main(void) {
    assert(spv_kills(NULL, 0) && spv_kills(words, 3));   // no SPIR-V: assumed to
    spv(); op(17, 1, (uint32_t[]){1}); op(253, 0, NULL); assert(!spv_kills(words, nw));   // OpCapability Shader, OpReturn
    spv(); op(252, 0, NULL); assert(spv_kills(words, nw));   // OpKill
    spv(); op(4416, 0, NULL); assert(spv_kills(words, nw));   // OpTerminateInvocation
    spv(); op(5380, 0, NULL); assert(spv_kills(words, nw));   // OpDemoteToHelperInvocation
    spv(); op(71, 3, (uint32_t[]){7, 11, 20}); assert(spv_kills(words, nw));   // OpDecorate %7 BuiltIn SampleMask
    spv(); op(71, 3, (uint32_t[]){7, 11, 15}); assert(!spv_kills(words, nw));   // OpDecorate %7 BuiltIn FragCoord
    spv(); words[nw++] = 0; assert(spv_kills(words, nw));   // a broken instruction: assumed to
    assert(fmt_comps(VK_FORMAT_R8_UNORM) == 1 && fmt_comps(VK_FORMAT_R32_SFLOAT) == 1 && fmt_comps(VK_FORMAT_R16G16_SFLOAT) == 3);
    assert(fmt_comps(VK_FORMAT_B10G11R11_UFLOAT_PACK32) == 7 && fmt_comps(VK_FORMAT_R5G6B5_UNORM_PACK16) == 7);
    assert(fmt_comps(VK_FORMAT_R8G8B8A8_UNORM) == 15 && fmt_comps(VK_FORMAT_R16G16B16A16_SFLOAT) == 15 && fmt_comps(VK_FORMAT_A2B10G10R10_UNORM_PACK32) == 15);
    // a 64x32 pass's signature in slot 5, generation 3: its probes' counts
    ls_ent *e = &g_ls_t[5]; e->gen = 3; e->area = 64 * 32; uint32_t tag = 3u << 10 | 6;
    for (int i = 0; i < 4; i++) vk_ls_probe_done(tag, 2048, 1);
    assert(e->good == 4 && !e->bad);
    vk_ls_probe_done(tag, 100, 0); assert(!e->bad);   // no result: no evidence either way
    vk_ls_probe_done(2u << 10 | 6, 100, 1); assert(!e->bad);   // an earlier signature's probe in that slot
    vk_ls_probe_done(tag, 4096, 1); assert(e->bad && e->bad_n == 4096 && g_vk_ls_refused == 1);   // overdraw counted twice: refused
    vk_ls_probe_done(tag, 2047, 1); assert(e->bad_n == 4096 && g_vk_ls_refused == 1);   // refused once
    vk_ls_probe_done(0, 1, 1); assert(g_vk_ls_refused == 1);   // (no such slot)
    puts("load skip ok"); return 0;
}
'''


@unittest.skipUnless(shutil.which('cc') and host_cc.posix(), 'a POSIX host C compiler is required')
class LoadSkip(unittest.TestCase):
    def test_shaders_formats_and_probes(self):
        ndk, _ = find_ndk()
        headers = next(ndk.glob('toolchains/llvm/prebuilt/*/sysroot/usr/include'), None) if ndk else None
        if not headers:
            self.skipTest('NDK Vulkan headers are required')
        src = EXEC.read_text(encoding='utf-8')
        def cut(start, end_marker):
            a = src.index(start); return src[a:src.index(end_marker, a)]
        cuts = (cut('enum { LS_N = ', 'static int g_ls, ') + cut('uint64_t g_vk_ls_colour', 'static void ls_decide(void);')
                + cut('static int spv_kills(', 'static int shader_kills(') + cut('static uint32_t fmt_comps(', '// the pass being begun for a draw')
                + cut('static void vk_ls_probe_done(uint32_t tag, uint64_t samples, int ok) {', 'static int prepare_draw('))
        with tempfile.TemporaryDirectory(prefix='nx2a-lskip-') as tmp:
            c, exe = Path(tmp) / 'check.c', Path(tmp) / 'check'
            c.write_text(HARNESS.replace('CUTS', cuts), encoding='utf-8')
            subprocess.run(['cc', '-std=gnu11', '-Wall', '-Wextra', '-Werror', '-Wno-unused-function', '-Wno-missing-field-initializers',
                            '-idirafter', str(headers), str(c), '-o', str(exe)], check=True)
            out = subprocess.run([str(exe)], capture_output=True, text=True, timeout=60)
            self.assertEqual(out.returncode, 0, out.stderr)
            self.assertIn('load skip ok', out.stdout)


if __name__ == '__main__':
    unittest.main()
