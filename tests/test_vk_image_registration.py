"""Verify the Android RegisterImage hook retains the descriptor and image intent."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]

@unittest.skipUnless(shutil.which("cc"), "host C compiler required")
class ImageRegistration(unittest.TestCase):
    def test_storage_intent_and_original_snapshot(self):
        source = (ROOT / "nxnative/dm/android/gfx_common.c").read_text()
        start = source.index("void mtl_reg_image(")
        pos = source.index("{", start) + 1
        depth = 1
        while depth:
            depth += (source[pos] == "{") - (source[pos] == "}")
            pos += 1
        helper = source[start:pos]
        code = r"""
#include <assert.h>
#include <stdint.h>
#include <string.h>
#include "nvn_obj.h"
static int calls, got_id, got_null;
static tex_t got;
static const view_t *got_view;
void mtl_reg_tex(int id, const tex_t *t, const view_t *v) {
 calls++; got_id=id; got_null=!t; got_view=v;
 if(t) got=*t;
}
HELPER
int main(void) {
 tex_t original={0}; original.serial=27; original.d.format=0x25;
 original.d.target=1; original.d.width=1440; original.d.height=810;
 original.d.depth=1; original.d.levels=1;
 original.mem=(uint8_t *)(uintptr_t)0x12340000;
 original.L.size=1440*810*4;
 view_t view={0};view.set_format=1;view.format=0x0b;
 view.set_levels=1;view.base_level=2;view.nlevels=1;
 view.set_layers=1;view.min_layer=3;view.nlayers=1;
 tex_t before=original,expected=original;expected.d.flags|=NVN_TEXTURE_FLAGS_IMAGE;
 mtl_reg_image(1357,&original,&view);
 assert(calls==1&&got_id==1357&&!got_null&&got_view==&view);
 assert(!memcmp(&original,&before,sizeof original));
 assert(!memcmp(&got,&expected,sizeof got));
 original.d.flags|=NVN_TEXTURE_FLAGS_IMAGE;expected=original;
 mtl_reg_image(1358,&original,NULL);
 assert(calls==2&&got_id==1358&&!got_null&&!got_view);
 assert(!memcmp(&got,&expected,sizeof got));
 mtl_reg_image(1357,NULL,&view);
 assert(calls==3&&got_id==1357&&got_null&&got_view==&view);
}
"""
        code=code.replace("HELPER",helper)
        target=ROOT/"target/tests";target.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="image-registration-",dir=target) as tmp:
            src=Path(tmp)/"test.c";exe=Path(tmp)/"test";src.write_text(code)
            subprocess.run(["cc","-std=c11","-O1","-g","-Wall","-Wextra","-Werror",
                "-fsanitize=address,undefined",
                "-I"+str(ROOT/"nxnative/dm/gfx/nvn"),
                "-I"+str(ROOT/"nxnative/dm/gfx/nvn/include"),
                str(src),"-o",str(exe)],check=True)
            subprocess.run([str(exe)],check=True)

if __name__=="__main__":
    unittest.main()
