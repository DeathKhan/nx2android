"""Regression-only reproducer for the known partial-UBO GPU ordering gap.

No runtime change accompanies this fixture. The expected-failure case exercises
the actual Android CPU-read/shadow helpers and records the currently wrong
result when an earlier GPU write has not executed yet. It is not scene evidence.
"""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


def function(source, declaration):
    start = source.index(declaration)
    pos = source.index("{", start) + 1
    depth = 1
    while depth:
        depth += (source[pos] == "{") - (source[pos] == "}")
        pos += 1
    return source[start:pos]


@unittest.skipUnless(shutil.which("cc"), "host C compiler required")
class UniformShadowOrdering(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = (ROOT / "nxnative/dm/android/gfx_vk_core.c").read_text()
        helpers = "\n".join(function(source, name) for name in [
            "int vk_ubd_overlaps(",
            "int vk_gpu_pending(",
            "void vk_ubd_read(",
            "int vk_ubd_shadow(",
        ])
        prefix = r"""
#include <assert.h>
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
typedef uintptr_t VkBuffer;
typedef uint64_t VkDeviceSize;
typedef struct {uint64_t lo,hi;VkBuffer sb;VkDeviceSize so;const uint8_t *p;} ubd_t;
typedef struct {uint64_t lo,hi,fence;const uint8_t *data;} ubd_infl_t;
typedef struct {uint64_t lo,hi,fence;} gpuw_t;
static ubd_t g_ubd[8];static int g_nubd;
static ubd_infl_t g_infl[8];static int g_ninfl;
static gpuw_t g_gw[8];static int g_ngw;
static uint64_t g_ubd_gen=1,g_ubd_racy,g_ubd_bytes,g_ubd_shadows,done;
int g_x_noshc;
static struct {struct {struct {uint64_t minUniformBufferOffsetAlignment;} limits;} props;} V;
static unsigned char scratch[256];
static uint64_t vk_fence_done(void) {return done;}
static void *vk_scratch(uint64_t n,uint64_t align,VkBuffer *buf,VkDeviceSize *off) {
 assert(n<=sizeof scratch);assert(align==16);*buf=2;*off=0;return scratch;
}
#define VKLOG(...) ((void)snprintf(NULL,0,__VA_ARGS__))
"""
        suffix = r"""
int main(int argc,char **argv) {
 assert(argc==2);
 uint32_t guest[2]={1,2},update[2]={7,9};
 uint64_t address=(uintptr_t)guest;
 V.props.limits.minUniformBufferOffsetAlignment=16;
 /* Earlier command: GPU will write guest word0=7. CPU still sees1. */
 g_gw[0]=(gpuw_t){address,address+4,1};g_ngw=1;
 int full=atoi(argv[1]);
 g_ubd[0]=(ubd_t){address+(full?0:4),address+8,3,0,
                  (const uint8_t *)(update+(full?0:1))};
 g_nubd=1;
 VkBuffer b=0;VkDeviceSize off=99;
 int has=vk_ubd_shadow(address,8,&b,&off);
 assert(has && b==2 && off==0 && g_ubd_shadows==1 && g_ubd_bytes==8);
 /* Execute queued GPU commands now. The already-built shadow is immutable. */
 guest[0]=7;guest[1]=9;done=1;
 uint32_t observed[2];memcpy(observed,scratch,8);
 printf("%u %u %llu\n",observed[0],observed[1],
        (unsigned long long)g_ubd_racy);
 return 0;
}
"""
        base = ROOT / "target/tests"
        base.mkdir(parents=True, exist_ok=True)
        cls.folder = tempfile.TemporaryDirectory(prefix="ubo-order-", dir=base)
        path = Path(cls.folder.name)
        cls.exe = path / "probe"
        src = path / "probe.c"
        src.write_text(prefix + helpers + suffix)
        subprocess.run(["cc", "-std=c11", "-Wall", "-Wextra", "-Werror",
                        "-fsanitize=address,undefined", str(src), "-o", str(cls.exe)],
                       check=True, capture_output=True, text=True)

    @classmethod
    def tearDownClass(cls):
        cls.folder.cleanup()

    def run_case(self, full):
        result = subprocess.run([str(self.exe), str(int(full))],
                                check=True, capture_output=True, text=True)
        return tuple(map(int, result.stdout.split()))

    def test_fully_covered_update_preserves_command_order(self):
        # Every byte comes from the later inline update; old CPU memory is irrelevant.
        self.assertEqual(self.run_case(True), (7, 9, 1))

    @unittest.expectedFailure
    def test_partial_update_must_preserve_prior_gpu_written_word(self):
        # Known current result(1,9,1): the warning does not prevent a stale shadow.
        # Correct implementation must return(7,9), or decline the CPU-shadow route.
        word0, word1, warned = self.run_case(False)
        self.assertEqual(warned, 1)
        self.assertEqual((word0, word1), (7, 9))


if __name__ == "__main__":
    unittest.main()
