"""Command and control memory the host lends can be named by a control entry (below 2^47, untagged).

A control entry (nvn_obj.h ctl_e) keeps an address in 44 bits (address / 8). Android's Scudo tags
heap pointers in the top byte (0xb4), which ctl_e shifted into the entry's length: Pokémon Legends:
Arceus records into host-lent segments, and its pieces came out 0xb4000 bytes too long, so the
replay read the unwritten tail of each segment as records of size 0.
"""
from pathlib import Path
import re, shutil, subprocess, tempfile, unittest
import sys; sys.path.insert(0, str(Path(__file__).resolve().parent))
import host_cc   # noqa: E402
ROOT = Path(__file__).resolve().parents[1]
NVN = ROOT / 'nxnative/dm/gfx/nvn'


def function(s, signature):
    start = s.index(signature); p = s.index('{', start); depth = 1; i = p + 1
    while depth:
        depth += (s[i] == '{') - (s[i] == '}'); i += 1
    return s[start:i]


class HostLentBlocks(unittest.TestCase):
    def setUp(self):
        self.src = (NVN / 'hle_nvn.c').read_text(encoding='utf-8')

    def test_lent_blocks_come_from_host_block(self):
        hostseg = function(self.src, 'static uint8_t *hostseg_take(')
        ctlseg = function(self.src, 'static uint8_t *ctlseg_take(')
        for name, body in (('hostseg_take', hostseg), ('ctlseg_take', ctlseg)):
            self.assertIn('host_block(', body, f'{name} lends memory from host_block')
            self.assertNotRegex(body, r'\bm\s*=\s*malloc\(|\*m\s*=\s*malloc\(', f'{name} must not lend malloc memory (Android tags its pointers)')

    def test_segments_and_control_memory_refuse_tagged_addresses(self):
        self.assertIn('>> 47', function(self.src, 'static void cmd_new_segment('))
        self.assertIn('>> 47', function(self.src, 'static void ctl_set('))

    @unittest.skipUnless(shutil.which('cc'), 'C compiler required')
    @unittest.skipUnless(host_cc.posix(), 'a POSIX host C compiler is required (MinGW has no <sys/mman.h>)')
    def test_entries_name_lent_blocks_exactly(self):
        obj = (NVN / 'nvn_obj.h').read_text(encoding='utf-8')
        helpers = '\n'.join(l for l in obj.splitlines()
                            if l.startswith(('enum { CTL_CHUNK', '#define CTL_', 'static inline uint64_t ctl_e(', 'static inline unsigned ctl_kind(',
                                             'static inline const uint8_t *ctl_addr(', 'static inline uint64_t ctl_len(')))
        self.assertEqual(helpers.count('static inline'), 4, 'nvn_obj.h control entry helpers')
        block = function(self.src, 'static uint8_t *host_block(')
        code = r'''
#define _GNU_SOURCE
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
static void die(const char *f, ...) { fprintf(stderr, "die: %s\n", f); abort(); }
HELPERS
BLOCK
int main(void) {
    static const size_t sizes[] = { 4u << 10, 64u << 10, 256u << 10, 300u << 10 };
    for (int r = 0; r < 64; r++) {
        size_t n = sizes[r % 4]; uint8_t *m = host_block(n, "test"); assert(m);
        assert(!((uintptr_t)m >> 47) && !((uintptr_t)m & 7));
        size_t used = 1920 + 8 * (size_t)r; memset(m, 0x5a, used);   // records written up to used, the rest of the segment untouched
        uint64_t e = ctl_e(CTL_CHUNK, m, used) | CTL_LAST;
        assert(ctl_kind(e) == CTL_CHUNK && ctl_addr(e) == m && ctl_len(e) == used && (e & CTL_LAST));
        uint64_t l = ctl_e(CTL_LINK, m + 8, 0);
        assert(ctl_kind(l) == CTL_LINK && ctl_addr(l) == m + 8 && !ctl_len(l));
    }
    return 0;
}
'''.replace('HELPERS', helpers).replace('BLOCK', block)
        with tempfile.TemporaryDirectory() as d:
            c = Path(d) / 't.c'; exe = Path(d) / 't'
            c.write_text(code, encoding='utf-8')
            subprocess.run(['cc', '-std=gnu11', '-O1', '-g', str(c), '-o', str(exe)], check=True, capture_output=True, timeout=60)
            subprocess.run([str(exe)], check=True, timeout=15)


if __name__ == '__main__':
    unittest.main()
