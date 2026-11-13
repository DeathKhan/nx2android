"""dm/host/crash_text.h, the text a register points at in a crash dump (crash.c's `[dm] *xN text` lines): UTF-16LE or UTF-8 up to 512 bytes, ended by a NUL or the
first character that is not printable, read in pieces that stay within a page. The `*xN:` line's six words cut Unreal Engine 4's PhysX error message after 24 characters
(droidplay-core#389). Built with the system's C compiler and called through ctypes; the reads go through process_vm_readv (Linux) or vm_read_overwrite (macOS), as the
hosts' do. Skipped without a C compiler."""
import ctypes
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest

REPO = Path(__file__).resolve().parents[3]
CC = (['xcrun', 'clang'] if sys.platform == 'darwin' and shutil.which('xcrun') else
      [c] if (c := shutil.which('clang') or shutil.which('cc') or shutil.which('gcc')) else None)
SHIM = r'''
#define _GNU_SOURCE
#include "crash_text.h"
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>
#ifdef __APPLE__
#include <mach/mach.h>
static int rd(uint64_t a, void *to, size_t n) { vm_size_t got = 0; return vm_read_overwrite(mach_task_self(), (vm_address_t)a, n, (vm_address_t)to, &got) == KERN_SUCCESS && got == n; }
#else
#include <sys/uio.h>
static int rd(uint64_t a, void *to, size_t n) { struct iovec l = {to, n}, r = {(void *)(uintptr_t)a, n}; return process_vm_readv(getpid(), &l, 1, &r, 1, 0) == (ssize_t)n; }
#endif
#define API __attribute__((visibility("default")))
API int fmt(const unsigned char *b, size_t n, int aligned2, char *out, size_t cap, int *wide) { return crash_text_format(b, n, aligned2, out, cap, wide); }
API int out_size(void) { return CRASH_TEXT_OUT; }
API int max_size(void) { return CRASH_TEXT_MAX; }
// s (len bytes) placed at `at` bytes before the end of a page whose next page is either readable too or not mapped readable, then read and formatted as crash.c does:
// returns the characters, or -1 when the pages could not be set up; *got the bytes read
API int placed(const unsigned char *s, size_t len, size_t at, int next_readable, char *out, size_t cap, int *wide, size_t *got) {
    size_t pg = (size_t)sysconf(_SC_PAGESIZE);
    unsigned char *m = mmap(NULL, 2 * pg, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
    if (m == MAP_FAILED) return -1;
    unsigned char *p = m + pg - at;
    memcpy(p, s, len < at || next_readable ? len : at);
    if (!next_readable && mprotect(m + pg, pg, PROT_NONE)) { munmap(m, 2 * pg); return -1; }
    unsigned char b[CRASH_TEXT_MAX];
    *got = crash_text_read((uint64_t)(uintptr_t)p, b, rd);
    int r = crash_text_format(b, *got, !((uintptr_t)p & 1), out, cap, wide);
    munmap(m, 2 * pg);
    return r;
}
'''
PHYSX = ('PHYSX: (D:/Build/++Fortnite/Sync/Engine/Source/ThirdParty/PhysX3/PhysX_3.4/Source/PhysX/src/NpScene.cpp 3191) eINVALID_OPERATION : '
         'PxScene::lockWrite() called without a matching unlock')   # the form of FPhysXErrorCallback::reportError's message


def lit(s):   # s as the header writes it: a C string literal
    return '"' + s.replace('\\', '\\\\').replace('"', '\\"').replace('\t', '\\t').replace('\n', '\\n').replace('\r', '\\r') + '"'


@unittest.skipIf(CC is None or os.name == 'nt', 'no C compiler')
class CrashText(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        assert CC is not None   # (skipped without one; pyright doesn't follow skipIf)
        cls.tmp = Path(tempfile.mkdtemp())
        src = cls.tmp / 'a.c'; src.write_text(SHIM)
        subprocess.run([*CC, '-shared', '-fPIC', '-O1', '-std=gnu11', '-Wall', '-Wextra', '-Werror', '-I', str(REPO / 'dm/host'), str(src), '-o', str(cls.tmp / 'a.so')], check=True)
        cls.lib = ctypes.CDLL(str(cls.tmp / 'a.so'))
        cls.lib.fmt.argtypes = [ctypes.c_char_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_char_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_int)]
        cls.lib.placed.argtypes = [ctypes.c_char_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_int, ctypes.c_char_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_size_t)]
        cls.out, cls.limit = cls.lib.out_size(), cls.lib.max_size()
        assert cls.limit == 512

    @classmethod
    def tearDownClass(cls): shutil.rmtree(cls.tmp, ignore_errors=True)

    def fmt(self, b, aligned2=1, cap=None):   # → (characters, 'utf-16' or 'utf-8', the quoted text) as crash.c prints them
        cap = cap or self.out
        out = ctypes.create_string_buffer(b'\xaa' * (cap + 16), cap + 16); wide = ctypes.c_int(-1)   # 16 bytes past cap that must stay as they are
        b = b[:self.limit]; n = self.lib.fmt(b, len(b), aligned2, out, cap, ctypes.byref(wide))
        self.assertTrue(all(c == 0xaa for c in out.raw[cap:]), 'wrote past cap')
        return n, ('utf-16' if wide.value else 'utf-8'), out.value.decode('utf-8')

    def test_unreals_utf16_message_whole(self):
        b = PHYSX.encode('utf-16-le') + b'\0\0' + os.urandom(64)
        self.assertGreater(len(PHYSX), 128)   # past 256 bytes, let alone the 24 characters the *x1 line kept
        self.assertEqual(self.fmt(b), (len(PHYSX), 'utf-16', lit(PHYSX)))

    def test_utf8_text_ends_at_its_nul(self):
        self.assertEqual(self.fmt(b'[RED] texture pool 3 overflowed\0garbage after it'), (31, 'utf-8', lit('[RED] texture pool 3 overflowed')))
        self.assertEqual(self.fmt(b'abc\0' * 8), (0, 'utf-8', ''))   # too short for UTF-8, and half ASCII read as UTF-16 ("扡c扡c…")

    def test_a_non_printable_character_ends_it(self):
        self.assertEqual(self.fmt(b'Pause menu opened\x01\x02more'), (17, 'utf-8', lit('Pause menu opened')))
        self.assertEqual(self.fmt('Loading level 2'.encode('utf-16-le') + b'\x07\x00'), (15, 'utf-16', lit('Loading level 2')))

    def test_text_past_the_read_is_marked(self):
        s = 'x' * 600
        self.assertEqual(self.fmt(s.encode()), (512, 'utf-8', lit('x' * 512) + ' ...'))
        self.assertEqual(self.fmt(s.encode('utf-16-le')), (256, 'utf-16', lit('x' * 256) + ' ...'))

    def test_escapes(self):
        s = 'D:\\Build\\"quoted"\ttab\r\nend'
        self.assertEqual(self.fmt(s.encode() + b'\0'), (len(s), 'utf-8', lit(s)))
        self.assertEqual(self.fmt(s.encode('utf-16-le') + b'\0\0'), (len(s), 'utf-16', lit(s)))

    def test_the_escaped_worst_case_fits(self):   # every byte an escape: CRASH_TEXT_OUT holds the whole read
        self.assertEqual(self.fmt(b'"' * 512), (512, 'utf-8', lit('"' * 512) + ' ...'))
        self.assertEqual(self.fmt(b'\\\0' * 256), (256, 'utf-16', lit('\\' * 256) + ' ...'))

    def test_a_small_buffer_cuts_the_text_and_says_so(self):
        self.assertEqual(self.fmt(b'A long line of text, longer than the buffer\0', cap=24), (14, 'utf-8', '"A long line of" ...'))
        self.assertEqual(self.fmt(b'A long line\0', cap=7), (0, 'utf-8', ''))

    def test_utf8_beyond_ascii(self):
        s = 'Zelda — ゼルダの伝説 🗡'
        self.assertEqual(self.fmt(s.encode() + b'\0'), (len(s), 'utf-8', lit(s)))

    def test_utf16_beyond_ascii(self):
        s = 'セーブ: save 3 🗡 ok'   # more ASCII than not, with a surrogate pair
        self.assertEqual(self.fmt(s.encode('utf-16-le') + b'\0\0'), (len(s), 'utf-16', lit(s)))

    def test_ascii_is_not_taken_for_utf16(self):   # read as UTF-16 it is CJK, with no ASCII character among them
        s = 'PHYSX: (D:/Build/++Fortnite'
        self.assertEqual(self.fmt(s.encode() + b'\0'), (len(s), 'utf-8', lit(s)))
        self.assertEqual(self.fmt(b'abcdefg\0ABCDEF'), (7, 'utf-8', lit('abcdefg')))   # the NUL at an odd offset makes one ASCII unit of seven

    def test_utf16_only_at_an_even_address(self):
        self.assertEqual(self.fmt('Loading level 2'.encode('utf-16-le'), aligned2=0), (0, 'utf-8', ''))

    def test_pointers_and_numbers_are_not_text(self):
        self.assertEqual(self.fmt(struct.pack('<QQ', 0x7161626364, 0xb400007a12345678)), (0, 'utf-8', ''))   # "dcbaq": five characters, one short
        self.assertEqual(self.fmt(struct.pack('<QQ', 0x7f41424344, 0)), (0, 'utf-8', ''))   # "DCBA", then DEL
        self.assertEqual(self.fmt(struct.pack('<ff', 1.0, 2.0)), (0, 'utf-8', ''))

    def test_malformed_utf8_ends_it(self):
        for bad in (b'\xc0\x80', b'\xe0\x80\x80', b'\xed\xa0\x80', b'\x80', b'\xf5\x80\x80\x80', b'\xc2\x85', b'\xe3\x81'):   # overlong twice, a surrogate, a continuation alone, past U+10FFFF, a C1 control, a byte short
            self.assertEqual(self.fmt(b'printable ' + bad + b' rest'), (10, 'utf-8', lit('printable ')), bad)
        self.assertEqual(self.fmt(b'printable \xe3\x81'), (10, 'utf-8', lit('printable ')))   # cut off by the end of the read, which then is not text to the end

    def test_lone_surrogates_end_utf16(self):
        for bad in (b'\x00\xd8A\x00', b'\x00\xdcA\x00', b'\xff\xff'):
            self.assertEqual(self.fmt('save data'.encode('utf-16-le') + bad + 'rest'.encode('utf-16-le')), (9, 'utf-16', lit('save data')), bad)

    def placed(self, s, at, next_readable):   # crash_text_read and format on s placed `at` bytes before a page's end
        out = ctypes.create_string_buffer(self.out); wide = ctypes.c_int(-1); got = ctypes.c_size_t(0)
        n = self.lib.placed(s, len(s), at, next_readable, out, self.out, ctypes.byref(wide), ctypes.byref(got))
        if n < 0: self.skipTest('mmap or mprotect refused')
        return n, ('utf-16' if wide.value else 'utf-8'), out.value.decode('utf-8'), got.value

    def test_read_across_a_page_boundary(self):
        r = self.placed(PHYSX.encode('utf-16-le') + b'\0\0', 10, 1)
        if r[3] == 0: self.skipTest('the read was refused (process_vm_readv blocked)')
        self.assertEqual(r, (len(PHYSX), 'utf-16', lit(PHYSX), 512))

    def test_read_stops_at_an_unreadable_page_and_keeps_what_came_before(self):
        r = self.placed(b'Unreal ran into an unmapped page', 100, 0)
        if r[3] == 0: self.skipTest('the read was refused (process_vm_readv blocked)')
        self.assertEqual(r[3], 100)
        self.assertEqual(r[:3], (32, 'utf-8', lit('Unreal ran into an unmapped page')))
        r = self.placed(b'y' * 100, 100, 0)   # text up to the unreadable page: no end seen
        self.assertEqual(r, (100, 'utf-8', lit('y' * 100) + ' ...', 100))
        self.assertEqual(self.placed(b'', 0, 0), (0, 'utf-8', '', 0))   # the page itself unreadable: nothing read


if __name__ == '__main__':
    unittest.main()
