"""DM_VK_PASS_MERGE (gfx_vk_core.c vk_body_begin / vk_body_end / vk_body_out): which command buffer vk_cmd (and so the slot's main) gives while a pass is recorded as a secondary, cut from the source."""
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from porter.toolchain import find_ndk
sys.path.insert(0, str(Path(__file__).resolve().parent))
import host_cc   # noqa: E402

CORE = Path(__file__).resolve().parents[1] / 'nxnative/dm/android/gfx_vk_core.c'
HARNESS = r'''
#define VK_NO_PROTOTYPES
#include <vulkan/vulkan_core.h>
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
typedef struct vksub { VkCommandPool pool; VkCommandBuffer main; int cb_err; VkCommandBuffer body, prim, *bodies; int nbody_alloc, nbody_used; } vksub_t;
static vksub_t slot; static vksub_t *cur(void) { return &slot; }
static struct { VkDevice dev; } V;
static int nalloc, nbegin, nend; static VkCommandBufferUsageFlags last_flags; static const VkCommandBufferInheritanceInfo *last_ii; static uintptr_t next_cb = 0x1000;
static VkResult alloc_cbs(VkDevice d, const VkCommandBufferAllocateInfo *ai, VkCommandBuffer *out) {
    (void)d; assert(ai->level == VK_COMMAND_BUFFER_LEVEL_SECONDARY && ai->commandPool == slot.pool);
    for (uint32_t i = 0; i < ai->commandBufferCount; i++) { out[i] = (VkCommandBuffer)(next_cb++); }
    nalloc++; return VK_SUCCESS; }
static VkResult begin_cb(VkCommandBuffer cb, const VkCommandBufferBeginInfo *bi) { (void)cb; nbegin++; last_flags = bi->flags; last_ii = bi->pInheritanceInfo; return VK_SUCCESS; }
static VkResult end_cb(VkCommandBuffer cb) { (void)cb; nend++; return VK_SUCCESS; }
static PFN_vkAllocateCommandBuffers vkAllocateCommandBuffers = alloc_cbs;
static PFN_vkBeginCommandBuffer vkBeginCommandBuffer = begin_cb;
static PFN_vkEndCommandBuffer vkEndCommandBuffer = end_cb;
CUTS
int main(void) {
    VkCommandBuffer main_cb = (VkCommandBuffer)(uintptr_t)0x10; slot.main = main_cb; slot.pool = (VkCommandPool)(uintptr_t)0x20;
    assert(vk_cmd() == main_cb && vk_body_end() == VK_NULL_HANDLE);   // no body: the main buffer, as without the switch
    VkCommandBufferInheritanceInfo ii = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_INHERITANCE_INFO};
    VkCommandBuffer b = vk_body_begin(&ii);
    assert(b && vk_cmd() == b && last_ii == &ii && (last_flags & VK_COMMAND_BUFFER_USAGE_RENDER_PASS_CONTINUE_BIT) && nalloc == 1);
    assert(vk_body_begin(&ii) == VK_NULL_HANDLE && vk_cmd() == b && slot.prim == main_cb);   // one body at a time
    vk_body_out(1); assert(vk_cmd() == main_cb);   // a mesh draw's kernels: the main buffer, ahead of the pass
    vk_body_out(0); assert(vk_cmd() == b);
    vk_body_out(1); assert(vk_body_end() == b && nend == 1 && vk_cmd() == main_cb);   // ending clears the hold too
    vk_body_out(0); assert(vk_cmd() == main_cb);
    VkCommandBuffer b2 = vk_body_begin(&ii); assert(b2 && b2 != b && vk_body_end() == b2);   // the next body: the slot's next secondary
    for (int i = 0; i < 20; i++) { assert(vk_body_begin(&ii)); assert(vk_body_end()); }
    assert(nalloc == 2 && slot.nbody_used == 22 && slot.nbody_alloc == 32 && vk_cmd() == main_cb);   // allocated 16 at a time
    slot.nbody_used = 0; assert(vk_body_begin(&ii) == b && vk_body_end() == b && nalloc == 2);   // a new submission in the slot begins its secondaries again
    puts("pass merge ok"); return 0;
}
'''


@unittest.skipUnless(shutil.which('cc') and host_cc.posix(), 'a POSIX host C compiler is required')
class PassMergeBodies(unittest.TestCase):
    def test_vk_cmd_while_a_pass_is_a_secondary(self):
        ndk, _ = find_ndk()
        headers = next(ndk.glob('toolchains/llvm/prebuilt/*/sysroot/usr/include'), None) if ndk else None
        if not headers:
            self.skipTest('NDK Vulkan headers are required')
        src = CORE.read_text(encoding='utf-8')
        a = src.index('VkCommandBuffer vk_cmd(void) {'); cuts = src[a:src.index('\n', a) + 1]
        a = src.index('VkCommandBuffer vk_body_begin('); b = src.index('void vk_body_out(int on) {', a); cuts += src[a:src.index('\n', b) + 1]
        with tempfile.TemporaryDirectory(prefix='nx2a-pmerge-') as tmp:
            c, exe = Path(tmp) / 'check.c', Path(tmp) / 'check'
            c.write_text(HARNESS.replace('CUTS', cuts), encoding='utf-8')
            subprocess.run(['cc', '-std=gnu11', '-Wall', '-Wextra', '-Werror', '-Wno-unused-function', '-Wno-missing-field-initializers',
                            '-idirafter', str(headers), str(c), '-o', str(exe)], check=True)
            out = subprocess.run([str(exe)], capture_output=True, text=True, timeout=60)
            self.assertEqual(out.returncode, 0, out.stderr)
            self.assertIn('pass merge ok', out.stdout)


if __name__ == '__main__':
    unittest.main()
