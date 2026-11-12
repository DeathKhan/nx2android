#!/usr/bin/env python3
"""dm/gfx/tools/ext.py: how optional add-ons are found (a stand-in add-on written by the test; nup2android ships none)."""
import os, sys, tempfile, unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'tools'))
import ext


class Addons(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(); self.addCleanup(tmp.cleanup); self.tmp = Path(tmp.name)
        path = list(sys.path); self.addCleanup(lambda: sys.path.__setitem__(slice(None), path))

    def load(self, value):
        """ext with NUP2ANDROID_ADDONS set to value (None: unset), as a fresh process would see it"""
        env = {k: v for k, v in os.environ.items() if k != 'NUP2ANDROID_ADDONS'}
        if value is not None: env['NUP2ANDROID_ADDONS'] = value
        for p in (patch.dict(os.environ, env, clear=True), patch.object(ext, '_mods', None)): p.start(); self.addCleanup(p.stop)

    def test_without_the_variable_there_is_no_hook(self):
        self.load(None)
        self.assertEqual(ext.modules(), [])
        self.assertIsNone(ext.hook('cpk_utf'))

    def test_a_file_and_a_folder_with_an_addon_py_are_loaded_in_order(self):
        a = self.tmp / 'one.py'; a.write_text("def cpk_utf(t): return b'one' + t\n")
        (self.tmp / 'two').mkdir(); (self.tmp / 'two' / 'beside.py').write_text('WORD = b"two"\n')
        (self.tmp / 'two' / 'addon.py').write_text("import beside\ndef cpk_utf(t): return beside.WORD\ndef rpf_read(k, d): return beside.WORD + d\nnot_a_function = 1\ndef _private(): pass\n")
        self.load(os.pathsep.join([str(a), str(self.tmp / 'two')]))
        self.assertEqual(ext.hook('cpk_utf')(b'!'), b'one!')      # the first add-on that has the name
        self.assertEqual(ext.hook('rpf_read')(None, b'!'), b'two!')   # and it may import the files beside it
        for name in ('pak_keys', 'not_a_function', '_private'): self.assertIsNone(ext.hook(name))

    def test_a_name_that_is_no_addon_is_an_error(self):
        self.load(str(self.tmp / 'missing'))
        with self.assertRaisesRegex(FileNotFoundError, 'NUP2ANDROID_ADDONS: .*missing: no Python file'): ext.hook('cpk_utf')


if __name__ == '__main__': unittest.main()
