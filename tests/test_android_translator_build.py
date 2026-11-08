"""The actual Android compiler build/cache path, without running a real SDK."""
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock
from porter import pipeline

ROOT=Path(__file__).resolve().parents[1]
MODULES=('dmsh_xlate.dll','Ryujinx.Graphics.Shader.dll','Ryujinx.Common.dll','Spv.Generator.dll')

class TranslatorBuild(unittest.TestCase):
    def setUp(self):
        parent=ROOT/'target/tests';parent.mkdir(parents=True,exist_ok=True)
        self.tmp=tempfile.TemporaryDirectory(prefix='translator-build-',dir=parent);self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);dm=self.root/'dm';self.source=dm/'gfx/tools/dmsh_xlate/Program.cs'
        self.source.parent.mkdir(parents=True);self.source.write_text('first source')
        ryu=dm/'third_party/ryujinx';(ryu/'src').mkdir(parents=True);(ryu/'Directory.Build.props').write_text('net10')
        self.patch=mock.patch.object(pipeline,'DM',dm);self.patch.start();self.addCleanup(self.patch.stop)
        self.out=self.root/'dist/dmsh_xlate';self.log=self.root/'dist/compiler.log';self.out.parent.mkdir()
        self.calls=[];self.fail=False
        def run(cmd,log,env):
            self.calls.append(cmd)
            dest=Path(cmd[cmd.index('-o')+1]);artifacts=Path(cmd[cmd.index('--artifacts-path')+1])
            artifacts.mkdir(parents=True);(artifacts/'intermediate').write_text('object')
            dest.mkdir(parents=True,exist_ok=True)
            if self.fail:raise pipeline.BuildError('compiler failed')
            for name in MODULES:(dest/name).write_bytes((name+self.source.read_text()).encode())
        patch=mock.patch.object(pipeline.nsp,'run',side_effect=run);patch.start();self.addCleanup(patch.stop)
    def build(self):
        return pipeline.build_translator(sys.executable,self.out,self.log,dict(os.environ))
    def test_build_reuse_source_and_binary_keys(self):
        dll,key=self.build();self.assertEqual(len(self.calls),1)
        self.assertEqual(dll,self.out/'dmsh_xlate.dll')
        self.assertFalse(self.out.with_name('dmsh_xlate.artifacts').exists())
        manifest=json.loads((self.out/'build-manifest.json').read_text())
        self.assertEqual(manifest['sha256']['dmsh_xlate.dll'],hashlib.sha256(dll.read_bytes()).hexdigest())
        self.assertEqual(self.build()[1],key);self.assertEqual(len(self.calls),1)
        # Even with unchanged source, an externally replaced compiler changes the shader cache key.
        dll.write_bytes(dll.read_bytes()+b'changed compiler')
        changed=self.build()[1];self.assertNotEqual(changed,key);self.assertEqual(len(self.calls),1)
        self.source.write_text('second source')
        self.assertNotEqual(self.build()[1],changed);self.assertEqual(len(self.calls),2)
    def test_missing_dependency_rebuilds(self):
        self.build();(self.out/'Spv.Generator.dll').unlink()
        self.build();self.assertEqual(len(self.calls),2)
    def test_failure_keeps_previous_compiler_and_cleans_intermediates(self):
        dll,_=self.build();prior=dll.read_bytes()
        self.source.write_text('new source');self.fail=True
        with self.assertRaises(pipeline.BuildError):self.build()
        self.assertEqual(dll.read_bytes(),prior)
        self.assertFalse(self.out.with_name('dmsh_xlate.artifacts').exists())
        self.assertFalse(self.out.with_name('dmsh_xlate.key').exists())

if __name__=='__main__':unittest.main()

