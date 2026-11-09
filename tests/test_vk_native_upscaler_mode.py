"""The real native-upscaler selector: quiet indefinite mode and traced countdown."""
import re
from pathlib import Path
import os
import subprocess
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
class NativeMode(unittest.TestCase):
    def test_actual_selector(self):
        text=(ROOT/'nxnative/dm/android/gfx_vk_exec.c').read_text()
        start=text.index('static int upscaler_native_dispatch(');brace=text.index('{',start);end=brace+1;depth=1
        while depth:
            depth+=(text[end]=='{')-(text[end]=='}');end+=1
        helper=text[start:end]
        code=r'''
#define _GNU_SOURCE
#include <assert.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
static int logs;
#define VKLOG(...) (++logs)
'''+helper+r'''
int main(void) {
    int trace=7;unsetenv("DM_VK_UPSCALER_NATIVE");
    assert(!upscaler_native_dispatch(1,&trace) && !trace);
    setenv("DM_VK_UPSCALER_NATIVE","-1",1);
    for(int i=0;i<10000;i++) { trace=7;assert(upscaler_native_dispatch(1,&trace) && !trace); }
    assert(!logs && !strcmp(getenv("DM_VK_UPSCALER_NATIVE"),"-1"));
    setenv("DM_VK_UPSCALER_NATIVE","3",1);
    for(int i=2;i>=0;i--) { assert(upscaler_native_dispatch(1,&trace) && trace);assert(atoi(getenv("DM_VK_UPSCALER_NATIVE"))==i); }
    assert(logs==3);assert(!upscaler_native_dispatch(1,&trace) && !trace);assert(logs==3);
    setenv("DM_VK_UPSCALER_NATIVE","-2",1);assert(!upscaler_native_dispatch(1,&trace) && !trace);
    setenv("DM_VK_UPSCALER_NATIVE","invalid",1);assert(!upscaler_native_dispatch(1,&trace) && !trace);
    puts("PASS native mode selector: quiet indefinite and finite traced countdown");
}
'''
        parent=ROOT/'target/tests';parent.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='native-mode-',dir=parent) as d:
            p=Path(d);(p/'test.c').write_text(code)
            subprocess.run([os.environ.get('CC','cc'),'-std=c11','-O1','-g','-fsanitize=address,undefined',str(p/'test.c'),'-o',str(p/'test')],check=True)
            r=subprocess.run([str(p/'test')],capture_output=True,text=True)
            self.assertEqual(r.returncode,0,r.stdout+r.stderr);print(r.stdout.strip())
        self.assertIn('if (!upscaler_native_dispatch(cs->key, &native_trace) && upscaler_replace(cs, d)) {',text)

if __name__=='__main__':unittest.main()

