"""Actual Vulkan host-read publication for copied and zero-copy pool submissions."""
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


class HostReadback(unittest.TestCase):
    @unittest.skipUnless(shutil.which("cc"), "a host C compiler is required")
    def test_actual_barrier_covers_zero_copy_and_shadow_pools(self):
        source = (ROOT / "nxnative/dm/android/gfx_vk_core.c").read_text()
        helper = function(source, "static void pools_readback_barrier(VkCommandBuffer cb) {")
        code = r"""
#define VK_NO_PROTOTYPES
#include <vulkan/vulkan_core.h>
#include <assert.h>
#include <stdint.h>
static int g_shadow_pools, calls;
static VkCommandBuffer expected;
static void vkCmdPipelineBarrier(VkCommandBuffer cb, VkPipelineStageFlags src,
 VkPipelineStageFlags dst, VkDependencyFlags flags, uint32_t nm,
 const VkMemoryBarrier *m, uint32_t nb, const VkBufferMemoryBarrier *b,
 uint32_t ni, const VkImageMemoryBarrier *i) {
 assert(cb == expected);
 assert(src == VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
 assert(dst == VK_PIPELINE_STAGE_HOST_BIT);
 assert(!flags && nm == 1 && !nb && !b && !ni && !i);
 assert(m->sType == VK_STRUCTURE_TYPE_MEMORY_BARRIER && !m->pNext);
 assert(m->srcAccessMask == VK_ACCESS_MEMORY_WRITE_BIT);
 assert(m->dstAccessMask == VK_ACCESS_HOST_READ_BIT);
 calls++;
}
HELPER
int main(void) {
 // All-zero-copy is the active OoT case; it must not return before recording.
 g_shadow_pools = 0;
 expected = (VkCommandBuffer)(uintptr_t)1;
 pools_readback_barrier(expected);
 assert(calls == 1);
 // Copied and mixed pools retain the same publication before CPU merge.
 g_shadow_pools = 1;
 expected = (VkCommandBuffer)(uintptr_t)2;
 pools_readback_barrier(expected);
 assert(calls == 2);
 // Every early native part publishes even if the final part is later dropped.
 g_shadow_pools = 0;
 for (uintptr_t part = 3; part < 35; part++) {
  expected = (VkCommandBuffer)part;
  pools_readback_barrier(expected);
 }
 assert(calls == 34);
 return 0;
}
""".replace("HELPER", helper)
        target = ROOT / "target/tests"
        target.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=target, prefix="host-readback-") as temp:
            c = Path(temp) / "test.c"
            exe = Path(temp) / "test"
            c.write_text(code)
            subprocess.run([
                "cc", "-std=c11", "-O1", "-g", "-fsanitize=address,undefined",
                "-fno-omit-frame-pointer", "-I",
                str(ROOT / "nxnative/dm/third_party/vulkan-headers/include"),
                str(c), "-o", str(exe)
            ], check=True)
            subprocess.run([str(exe)], check=True)

    def test_native_flush_points_precede_submission_and_cpu_release(self):
        source = (ROOT / "nxnative/dm/android/gfx_vk_core.c").read_text()
        split = function(source, "static int vk_cmd_split_(void) {")
        commit = function(source, "uint64_t mtl_sub_commit(void) {")
        completion = function(source, "static void *completion_main(void *a) {")
        for body in (split, commit):
            # Keep publication at native part boundaries, including parts sent early.
            self.assertLess(body.index("pools_readback_barrier(s->main)"),
                            body.index("vkEndCommandBuffer(s->main)"))
            self.assertLess(body.index("pools_readback_barrier(s->main)"),
                            body.index("vkQueueSubmit("))
        self.assertLess(completion.index("wait_fence(c->fence)"),
                        completion.index("pools_copy_back("))
        self.assertLess(completion.index("pools_copy_back("),
                        completion.index("if (o->kind == 1)"))
        self.assertLess(completion.index("if (o->kind == 1)"),
                        completion.index("__atomic_store_n(&g_done"))


if __name__ == "__main__":
    unittest.main()
