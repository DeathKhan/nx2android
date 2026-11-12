"""rpf6.py: Rockstar RPF6 archives (Red Dead Redemption). No game files: a synthetic archive (header, entry table, name table) laid out as the module's comment describes,
stored in the clear, and one whose header says its tables are encrypted, which is refused (no decryption code)."""
import os
import struct
import sys
import tempfile
import unittest
import zlib
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / 'dm/gfx/tools'))
import rpf6


def archive(path, encrypted=False):
    """root/ {a.txt (plain), dir/ {b.bin (zlib, compressed bit), r.wtx (an RSC5 resource)}}: entries 0 root, 1 a.txt, 2 dir, 3 b.bin, 4 r.wtx"""
    plain, packed = b'plain text', zlib.compress(b'x' * 300)
    res_flags = 0x80000000 | 0x40000000 | 0x123
    rsc = rpf6.RSC5 + struct.pack('<II', 0x1e, res_flags) + zlib.compress(b'resource data')
    data_at = 0x800
    a_off, b_off, r_off = data_at, data_at + 0x40, data_at + 0x800   # multiples of 8; a resource's of 0x800, as its offset word's low byte is its type
    entries = [
        (0, 0, 0x80000001, 2, 0),                                                        # root: children 1-2
        (rpf6.joaat('a.txt'), len(plain), a_off // 8, len(plain), 0),                    # a plain file
        (rpf6.joaat('dir'), 0, 0x80000003, 2, 0),                                        # a directory: children 3-4
        (rpf6.joaat('b.bin'), len(packed), b_off // 8, 0x40000000 | 300, 0),             # compressed, 300 bytes decoded
        (rpf6.joaat('r.wtx'), len(rsc), (r_off // 8) | 0x1e, res_flags, 0),              # a resource: type in the offset's low byte
    ]
    table = b''.join(struct.pack('>5I', *e) for e in entries)
    table += bytes(-len(table) % 16)
    names = b'\0\0\0\0a.txt\0dir\0b.bin\0r.wtx\0'
    names += bytes(-len(names) % 16)
    names_at = 0x10 + len(table)
    if encrypted: table, names = os.urandom(len(table)), os.urandom(len(names))   # nothing may read them
    d = bytearray(data_at + 0x1000)
    d[:16] = rpf6.MAGIC + struct.pack('>3I', len(entries), names_at // 8, rpf6.ENCRYPTED if encrypted else 0)
    d[0x10:0x10 + len(table)] = table
    d[names_at:names_at + len(names)] = names
    d[a_off:a_off + len(plain)] = plain
    d[b_off:b_off + len(packed)] = packed
    d[r_off:r_off + len(rsc)] = rsc
    Path(path).write_bytes(bytes(d))


class Zlib:
    """Stands in for the zstd module: the synthetic archive has zlib streams only"""
    @staticmethod
    def decompress(d): raise AssertionError('no zstd frame in the synthetic archive')


class Archive(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, 'test.rpf')

    def tearDown(self):
        self.tmp.cleanup()

    def test_joaat(self):
        self.assertEqual(rpf6.joaat(''), 0)
        self.assertEqual(rpf6.joaat('a'), 0xca2e9442)   # Jenkins one-at-a-time of "a"
        self.assertEqual(rpf6.joaat('Shaders.RPF'), rpf6.joaat('shaders.rpf'))   # lower-cased first

    def test_header(self):
        archive(self.path)
        self.assertEqual(rpf6.header(self.path), (5, 0x10 + 0x70, False))
        archive(self.path, encrypted=True)
        self.assertEqual(rpf6.header(self.path), (5, 0x10 + 0x70, True))
        Path(self.path).write_bytes(b'RPF7' + bytes(12))
        self.assertIsNone(rpf6.header(self.path))
        Path(self.path).write_bytes(b'RPF6')
        self.assertIsNone(rpf6.header(self.path))   # shorter than a header

    def test_entry_table_and_names(self):
        archive(self.path)
        r = rpf6.Rpf(self.path)
        files = {e[0]: e for e in r.files}
        self.assertEqual(sorted(files), ['a.txt', 'dir/b.bin', 'dir/r.wtx'])
        self.assertEqual(files['a.txt'][1:5], (0x800, 10, 10, False))
        self.assertEqual(files['dir/b.bin'][3:5], (300, True))
        self.assertEqual(files['dir/r.wtx'][1], 0x1000)   # the resource type byte is not part of the offset
        self.assertIsNone(files['dir/r.wtx'][3])
        with open(self.path, 'rb') as f:
            self.assertEqual(r.read(f, files['a.txt'], Zlib), b'plain text')
            self.assertEqual(r.read(f, files['dir/b.bin'], Zlib), b'x' * 300)
            self.assertEqual(r.read(f, files['dir/r.wtx'], Zlib), b'resource data')

    def test_encrypted_tables_are_refused(self):
        archive(self.path, encrypted=True)
        with self.assertRaisesRegex(ValueError, 'tables are encrypted, and nup2android has no decryption code'):
            rpf6.Rpf(self.path)

    def test_directory_past_the_table_is_an_error(self):
        archive(self.path)
        d = bytearray(Path(self.path).read_bytes())
        struct.pack_into('>I', d, 0x10 + 2 * 20 + 12, 9)   # dir claims 9 children
        Path(self.path).write_bytes(bytes(d))
        with self.assertRaisesRegex(ValueError, 'runs past the table'):
            rpf6.Rpf(self.path)


if __name__ == '__main__': unittest.main()
