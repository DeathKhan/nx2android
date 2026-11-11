"""Command memory a game adds over the recording in progress is not recorded into.

Nintendo Switch 2 Welcome Tour records each material into a block of command memory sized for the
driver's commands (0x180 bytes). The host's records outgrow it, and the game's out-of-memory callback
adds the same block again. Recording on into it wrote over the recording's own first records: the
control list then named a piece whose records read as op 0x70000 / size 1440, every submission copy
stopped there ("submission copy: a record of size 1440"), and the Joy-Con 2 (L) Area washed out white.
AddCommandMemory now asks rec_overlaps first and goes on in a host-lent segment instead.
"""
from pathlib import Path
import shutil, subprocess, tempfile, unittest
import sys; sys.path.insert(0, str(Path(__file__).resolve().parent))
import host_cc   # noqa: E402
ROOT = Path(__file__).resolve().parents[1]
NVN = ROOT / 'nxnative/dm/gfx/nvn'


def function(s, signature):
    start = s.index(signature); p = s.index('{', start); depth = 1; i = p + 1
    while depth:
        depth += (s[i] == '{') - (s[i] == '}'); i += 1
    return s[start:i]


class RecordingOverlap(unittest.TestCase):
    def setUp(self):
        self.src = (NVN / 'hle_nvn.c').read_text(encoding='utf-8')

    def test_add_command_memory_checks_before_releasing(self):
        body = function(self.src, 'NVN(nvnCommandBufferAddCommandMemory,')
        self.assertIn('rec_overlaps(', body)
        self.assertLess(body.index('rec_overlaps('), body.index('hostseg_release('),
                        'the overlap is checked before the block releases host segments (the recording may be in one hanging off it)')
        self.assertLess(body.index('rec_overlaps('), body.index('cmd_new_segment(CB(c), pl->mem + off, n)'))

    @unittest.skipUnless(shutil.which('cc'), 'C compiler required')
    @unittest.skipUnless(host_cc.posix(), 'a POSIX host C compiler is required')
    def test_overlap_cases(self):
        fn = function(self.src, 'static int rec_overlaps(')
        code = r'''
#include <assert.h>
#include <pthread.h>
#include <stdint.h>
#include <stddef.h>
typedef struct { uint32_t recording; uint8_t *cur, *end, *chunk; } cmd_t;
typedef struct { uint8_t *mem; size_t n; uint64_t frame; int busy; uintptr_t org_lo, org_hi; } hostseg_t;
static hostseg_t g_hostseg[2]; static int g_nhostseg; static pthread_mutex_t g_hostseg_mu = PTHREAD_MUTEX_INITIALIZER;
FN
static uint8_t block[0x180], other[0x180], seg[4096];
#define U(p) ((uintptr_t)(p))
int main(void) {
    cmd_t o = { 1, block + 0x170, block + 0x180, block };
    assert(rec_overlaps(&o, U(block), U(block + 0x180)));            // the same block again: the open piece is in it
    assert(!rec_overlaps(&o, U(other), U(other + 0x180)));           // fresh memory
    assert(rec_overlaps(&o, U(block + 0x100), U(block + 0x200)));    // a part of it
    o.recording = 0; assert(!rec_overlaps(&o, U(block), U(block + 0x180)));   // no recording: the game may reuse its memory
    o.recording = 1; o.chunk = NULL; o.cur = block;                   // nothing written in this segment yet
    assert(!rec_overlaps(&o, U(block), U(block + 0x180)));
    // the recording went on in a host-lent segment hanging off its guest bytes [block, block + 0x170)
    g_hostseg[0] = (hostseg_t){ seg, sizeof seg, 0, 1, U(block), U(block + 0x170) }; g_nhostseg = 1;
    o.cur = seg + 64; o.end = seg + sizeof seg; o.chunk = seg;
    assert(rec_overlaps(&o, U(block), U(block + 0x180)));            // adding the block a second time still finds the recording in it
    assert(!rec_overlaps(&o, U(other), U(other + 0x180)));
    return 0;
}
'''.replace('FN', fn)
        with tempfile.TemporaryDirectory() as d:
            c = Path(d) / 't.c'; exe = Path(d) / 't'
            c.write_text(code, encoding='utf-8')
            subprocess.run(['cc', '-std=gnu11', '-O1', '-g', str(c), '-o', str(exe), '-lpthread'], check=True, capture_output=True, timeout=60)
            subprocess.run([str(exe)], check=True, timeout=15)


if __name__ == '__main__':
    unittest.main()
