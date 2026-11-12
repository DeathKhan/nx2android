#!/usr/bin/env python3
"""dm/gfx/tools/cra.py on synthetic data: NARC members and the packing kinds (the scrambled archive itself is not read: no de-obfuscation code)."""
import lz4.block, os, struct, sys, unittest, zlib
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'tools'))
import cra


def narc(members):
    fat = b'BTAF' + struct.pack('<II', 12 + 8 * len(members), len(members)); data = b''
    for m in members:
        fat += struct.pack('<II', len(data), len(data) + len(m)); data += m + bytes(-len(m) & 15)
    fnt = b'BTNF' + struct.pack('<I', 8)
    img = b'GMIF' + struct.pack('<I', 8 + len(data)) + data
    return b'NARC\xfe\xff\0\0' + struct.pack('<IHH', 16 + len(fat) + len(fnt) + len(img), 16, 3) + fat + fnt + img


class Members(unittest.TestCase):
    def test_narc(self):
        self.assertEqual(cra.narc_members(narc([b'abc', b'', b'0123456789abcdefg'])), [b'abc', b'', b'0123456789abcdefg'])
        with self.assertRaises(cra.Error): cra.narc_members(b'SARC' + bytes(32))
        with self.assertRaises(cra.Error): cra.narc_members(narc([b'abc'])[:40])

    def test_kinds(self):
        text = b'BNSH\0\0\0\0 shaders shaders shaders shaders ' * 40
        lz4d = b'\x94\0\0\0' + struct.pack('<I', len(text)) + lz4.block.compress(text, store_size=False)
        self.assertEqual(cra.undo(lz4d), text)
        z = cra.zstd_mod().compress(text)
        self.assertEqual(cra.undo(b'Z\0\0\0' + struct.pack('<I', len(text)) + z), text)
        c = zlib.compressobj(wbits=-15); raw = c.compress(text) + c.flush()
        self.assertEqual(cra.undo(b'\x30' + struct.pack('>I', len(text)) + bytes(11) + raw), text)
        self.assertEqual(cra.undo(b'\x70' + struct.pack('>I', 3) + bytes(11) + b'abc'), b'abc')
        chunks = [lz4d, b'Z\0\0\0' + struct.pack('<I', len(text)) + z]
        body = b''.join(x + bytes(-len(x) & 15) for x in chunks)
        head = b'MMMM' + struct.pack('<II', 2 * len(text), 2) + b''.join(struct.pack('<I', len(x)) for x in chunks)
        self.assertEqual(cra.undo(head + bytes(-len(head) & 15) + body), text + text)
        self.assertEqual(cra.undo(b'FRES plain'), b'FRES plain')          # anything else is the member itself
        self.assertEqual(cra.undo(b'\x10not zlib, sixteen bytes or more'), b'\x10not zlib, sixteen bytes or more')

    def test_lz(self):
        # A flag byte holds 8 items from its high bit after negation (1 a match); a distance 0 ends. 'abcd', then 3 more at distance 4 (length in the low nibble), then the end
        flags = -0b00001100 & 0xff
        s = bytes([flags]) + b'abcd' + bytes([0x43, 0x00]) + bytes([0x00, 0x00])
        self.assertEqual(cra.lz_undo(b'\x50\0\0\0' + struct.pack('>I', 7) + s, 8), b'abcdabc')
        self.assertEqual(cra.undo(b'\x50\0\0\0' + struct.pack('>I', 7) + s), b'abcdabc')
        with self.assertRaises(cra.Error): cra.lz_undo(bytes([0x80]) + bytes([0x93, 0x00]), 0)   # before the start of the output
        # Length forms: low nibble 0 is the next byte + 16, low nibble 1 two bytes + 0x110
        s = bytes([flags]) + b'abcd' + bytes([0x40, 0x00, 0x05]) + bytes([0, 0])
        self.assertEqual(cra.lz_undo(s, 0), (b'abcd' * 7)[:4 + 21])
        s = bytes([flags]) + b'abcd' + bytes([0x41, 0x00, 0x05, 0x01]) + bytes([0, 0])
        self.assertEqual(len(cra.lz_undo(s, 0)), 4 + 5 + 0x11 + 0x100 + 0xff)
        # A flag byte 0 is 8 literals
        self.assertEqual(cra.lz_undo(bytes([0]) + b'12345678' + bytes([0x80]) + bytes([0, 0]), 0), b'12345678')


if __name__ == '__main__': unittest.main()
