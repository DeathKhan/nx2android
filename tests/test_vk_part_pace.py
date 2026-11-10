"""Part pacing (gfx_vk_exec.c part_pace): a long submission's parts are spread over its estimated draws, never cut below the fixed thresholds."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]

def function(source,declaration):
    start=source.index(declaration);pos=source.index('{',start)+1;depth=1
    while depth:
        depth+=(source[pos]=='{')-(source[pos]=='}');pos+=1
    return source[start:pos]

class PartPace(unittest.TestCase):
    @unittest.skipUnless(shutil.which('cc'),'a host C compiler is required')
    def test_parts_last_to_the_end_of_the_submission(self):
        exe=(ROOT/'nxnative/dm/android/gfx_vk_exec.c').read_text(encoding='utf-8')
        self.assertIn('static int g_pace; static uint64_t g_sub_d0, g_sub_hist[8]; static unsigned g_sub_hn;',exe)
        pace=function(exe,'static uint64_t part_pace(uint64_t floor)')
        code=r"""
#include <assert.h>
#include <stdint.h>
static struct { uint64_t draws; } g_vkst;
static int parts_left;
static int vk_cmd_parts_left(void){return parts_left;}
static int g_pace; static uint64_t g_sub_d0, g_sub_hist[8]; static unsigned g_sub_hn;
"""+pace+r"""
int main(void){
 (void)g_sub_hn;
 g_pace=0;parts_left=31;g_sub_hist[0]=12000;assert(part_pace(128)==128);           /* off: the fixed threshold */
 g_pace=1;assert(part_pace(128)==12000/32);                                        /* 12,000 draws over 31 parts and the commit's */
 g_sub_hist[3]=4000;assert(part_pace(128)==12000/32);                              /* the largest of the last eight */
 g_vkst.draws=6000;parts_left=9;assert(part_pace(128)==600);                        /* what is left over what is left */
 assert(part_pace(1024)==1024);                                                    /* never below the threshold */
 g_vkst.draws=13000;assert(part_pace(128)==128);                                   /* past the estimate: the threshold */
 g_vkst.draws=0;parts_left=0;assert(part_pace(128)==128);                          /* no parts left: the caller refuses the cut */
 for(int i=0;i<8;i++){g_sub_hist[i]=0;}parts_left=31;assert(part_pace(512)==512);    /* no estimate yet */
 g_sub_d0=100;g_vkst.draws=100;g_sub_hist[0]=3200;assert(part_pace(64)==100);       /* draws counted from this submission's start */
 return 0;}
"""
        target=ROOT/'target/tests';target.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='part-pace-',dir=target) as t:
            src=Path(t)/'test.c';out=Path(t)/'test';src.write_text(code, encoding='utf-8')
            subprocess.run(['cc','-std=c11','-g','-O1','-Wall','-Wextra','-Werror','-fsanitize=address,undefined','-fno-omit-frame-pointer',str(src),'-o',str(out)],check=True)
            subprocess.run([str(out)],check=True,timeout=15)

if __name__=='__main__':
    unittest.main()
