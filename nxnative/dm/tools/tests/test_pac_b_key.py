"""patch_nso.py's A-key to B-key pointer authentication rewrite for Android (pac_b_key_sites), on words assembled by clang (-march=armv8.3-a), without game files."""
import ast
from pathlib import Path
import re
import struct
from typing import Any
import unittest

tree = ast.parse((Path(__file__).resolve().parents[1] / 'patch_nso.py').read_text())   # the program loads and rewrites an image: compile only the table and the search
keep = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ('pac_b_key_twin', 'pac_b_key_sites')
        or isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'PAC_HINT_B' for t in n.targets)]
ns: dict[str, Any] = {'struct': struct, 're': re}
exec(compile(ast.Module(keep, []), 'patch_nso.py', 'exec'), ns)
twin, find = ns['pac_b_key_twin'], ns['pac_b_key_sites']

def img(*w): return b''.join(struct.pack('<I', x) for x in w)
NOP = 0xD503201F
# clang -target aarch64-linux-android30 -march=armv8.3-a, llvm-objdump: each A-key form and its B-key twin
A_TO_B = {'pacia x16,x9': (0xDAC10130, 0xDAC10530), 'pacia x17,x9': (0xDAC10131, 0xDAC10531), 'autia x3,x4': (0xDAC11083, 0xDAC11483),
          'paciza x5': (0xDAC123E5, 0xDAC127E5), 'autiza x6': (0xDAC133E6, 0xDAC137E6),
          'pacia1716': (0xD503211F, 0xD503215F), 'autia1716': (0xD503219F, 0xD50321DF), 'paciaz': (0xD503231F, 0xD503235F),
          'paciasp': (0xD503233F, 0xD503237F), 'autiaz': (0xD503239F, 0xD50323DF), 'autiasp': (0xD50323BF, 0xD50323FF), 'retaa': (0xD65F0BFF, 0xD65F0FFF),
          'braa x1,x2': (0xD71F0822, 0xD71F0C22), 'braaz x3': (0xD61F087F, 0xD61F0C7F), 'blraa x4,x5': (0xD73F0885, 0xD73F0C85), 'blraaz x6': (0xD63F08DF, 0xD63F0CDF)}
# left alone: the B-key forms themselves, the data and generic keys, strips, the exception return, and plain branches
OTHERS = {'pacib x16,x9': 0xDAC10530, 'pacibsp': 0xD503237F, 'autibsp': 0xD50323FF, 'retab': 0xD65F0FFF, 'blrab x4,x5': 0xD73F0C85, 'pacda x1,x2': 0xDAC10841,
          'autda x1,x2': 0xDAC11841, 'pacga x1,x2,x3': 0x9AC33041, 'ldraa x1,[x2]': 0xF8200441, 'xpaci x1': 0xDAC143E1, 'eretaa': 0xD69F0BFF,
          'br x1': 0xD61F0020, 'blr x1': 0xD63F0020, 'ret': 0xD65F03C0, 'nop': NOP, 'rev x1,x2': 0xDAC00C41}


class PacBKey(unittest.TestCase):
    def test_every_a_key_form_gets_its_twin(self):
        for name, (a, b) in A_TO_B.items():
            self.assertEqual(twin(a), b, name)
            code = img(NOP, a, NOP)
            self.assertEqual(find(code, [(0, len(code))]), [(4, b)], name)

    def test_other_words_left_alone(self):
        for name, w in OTHERS.items():
            self.assertIsNone(twin(w), name)
            code = img(NOP, w, NOP)
            self.assertEqual(find(code, [(0, len(code))]), [], name)

    def test_ocarina_of_time_context_start(self):   # v131072 sdk +0x16d0e8 / +0x16d0f8 / +0x16d40c (sdk_patches.py's cut): pacia x16,x9; pacia x17,x9; autiasp
        code = bytes.fromhex('3001c1da') + img(NOP, NOP, NOP) + bytes.fromhex('3101c1da') + img(NOP) + bytes.fromhex('bf2303d5')
        self.assertEqual(find(code, [(0, len(code))]), [(0, 0xDAC10530), (16, 0xDAC10531), (24, 0xD50323FF)])

    def test_only_aligned_words(self):   # a form's bytes across two words are not an instruction
        a = img(A_TO_B['pacia x16,x9'][0]); r = img(A_TO_B['retaa'][0])
        for shifted in (b'\x00\x00' + a + b'\x00\x00', b'\x00' + r + b'\x00\x00\x00', b'\x00\x00\x00' + img(A_TO_B['braa x1,x2'][0]) + b'\x00'):
            self.assertEqual(find(shifted, [(0, len(shifted))]), [])
        # an unaligned match right before an aligned one does not hide it
        code = bytes([0x00, 0xC1, 0xDA, 0x00]) + a
        self.assertEqual(find(code, [(0, len(code))]), [(4, 0xDAC10530)])

    def test_ranges_are_respected(self):   # data blocks between code ranges
        a, b = A_TO_B['paciasp']
        code = img(a, NOP, a, NOP, a)
        self.assertEqual(find(code, [(0, 4), (8, 12)]), [(0, b), (8, b)])
        self.assertEqual(find(code, [(4, 8), (12, 16)]), [])

    def test_android_only(self):   # Apple targets keep their code: the rewrite is under DM_TARGET=android
        src = (Path(__file__).resolve().parents[1] / 'patch_nso.py').read_text()
        self.assertIn("PAC_B = pac_b_key_sites(IMG, CODE_RANGES) if TARGET == 'android' else []", src)


if __name__ == '__main__':
    unittest.main()
