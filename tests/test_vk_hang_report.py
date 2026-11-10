"""GPU hang diagnostics: gfx_vk_exec.c's crumbs (what each pass recorded), gfx_vk_core.c's hang check, lost-device report and VK_EXT_device_fault log, cut from the sources."""
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from porter.toolchain import find_ndk
sys.path.insert(0, str(Path(__file__).resolve().parent))
import host_cc   # noqa: E402

ANDROID = Path(__file__).resolve().parents[1] / 'nxnative/dm/android'
HARNESS = r'''
#define VK_NO_PROTOTYPES
#include <vulkan/vulkan_core.h>
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static char logbuf[256][600]; static int nl;
#define VKLOG(...) do { assert(nl < 256); snprintf(logbuf[nl++], sizeof logbuf[0], __VA_ARGS__); } while (0)
#define VKCHK(x) assert((x) == VK_SUCCESS)
static struct { VkDevice dev; VkPhysicalDeviceProperties props; } V;
static uint64_t g_last_fence, g_done_fence; static int g_sub_pass, fills;
static uint64_t mtl_last_fence(void) { return g_last_fence; }
static uint64_t vk_fence_done(void) { return g_done_fence; }
static VkCommandBuffer vk_cmd(void) { return (VkCommandBuffer)1; }
static void vk_img_barrier_all(VkCommandBuffer cb) { (void)cb; }
static uint32_t vk_mem_type(uint32_t b, VkMemoryPropertyFlags w, VkMemoryPropertyFlags a) { (void)b; (void)w; (void)a; return 0; }
static void fill(VkCommandBuffer cb, VkBuffer b, VkDeviceSize o, VkDeviceSize n, uint32_t v) { (void)cb; (void)b; (void)o; (void)n; (void)v; fills++; }
static PFN_vkCmdFillBuffer vkCmdFillBuffer = fill;
static PFN_vkCreateBuffer vkCreateBuffer; static PFN_vkGetBufferMemoryRequirements vkGetBufferMemoryRequirements; static PFN_vkAllocateMemory vkAllocateMemory;
static PFN_vkBindBufferMemory vkBindBufferMemory; static PFN_vkMapMemory vkMapMemory;
typedef struct { uint64_t fence; int slot; } cpend;
static VkDeviceFaultAddressInfoEXT fake_ad[20]; static int fault_calls;
static VkResult fake_fault(VkDevice d, VkDeviceFaultCountsEXT *n, VkDeviceFaultInfoEXT *fi) {
    (void)d; fault_calls++;
    if (!fi) { n->addressInfoCount = 20; n->vendorInfoCount = 1; n->vendorBinarySize = 4096; return VK_SUCCESS; }
    assert(n->addressInfoCount == 16 && n->vendorInfoCount == 1 && !n->vendorBinarySize && !fi->pVendorBinaryData);
    snprintf(fi->description, sizeof fi->description, "GPU hang");
    for (uint32_t i = 0; i < n->addressInfoCount; i++) fi->pAddressInfos[i] = fake_ad[i];
    snprintf(fi->pVendorInfos[0].description, sizeof fi->pVendorInfos[0].description, "kgsl"); fi->pVendorInfos[0].vendorFaultCode = 7; fi->pVendorInfos[0].vendorFaultData = 9;
    return VK_INCOMPLETE;
}
static PFN_vkGetDeviceFaultInfoEXT g_fault_info;
EXEC
CORE
static int has(const char *s) { for (int i = 0; i < nl; i++) if (strstr(logbuf[i], s)) return i + 1; return 0; }
static void pass(int p) { g_sub_pass = p + 1; }
int main(void) {
    // submission 5: pass 0 with 10 draws over 10 pipelines (the last 8 kept), pass 1 a dispatch, passes 2-5 the same one pipeline, pass 6 another
    g_last_fence = 4;
    pass(0); for (uint64_t i = 1; i <= 10; i++) { crumb_draw(i, i + 100, 0); crumb_draw(i, i + 100, 0); }
    pass(1); crumb_dispatch(0xc5);
    for (int p = 2; p <= 5; p++) { pass(p); crumb_draw(0xa, 0xb, 0); crumb_draw(0xa, 0xb, 0); }
    pass(6); crumb_draw(0xc, 0xd, 0xe);
    g_last_fence = 5; pass(0); crumb_draw(0x77, 0x78, 0);   // submission 6
    assert(!fills);   // no GPU marks without DM_VK_CRUMBS=1
    vk_crumbs_log(5, "test");
    assert(has("[hang] submission 5 (test), what it recorded:"));
    int l = has("pass 0: 20 draws, 0 dispatches"); assert(l);
    assert(strstr(logbuf[l - 1], " 0000000000000003/0000000000000067/0000000000000000") && !strstr(logbuf[l - 1], "0000000000000002/") && strstr(logbuf[l - 1], "000000000000000a/000000000000006e"));
    assert(strstr(logbuf[l - 1], "0000000000000003/") < strstr(logbuf[l - 1], "000000000000000a/"));   // oldest first
    assert(has("pass 1: 0 draws, 1 dispatches; last shaders (vertex/fragment/geometry or cs): cs 00000000000000c5"));
    assert(has("pass 2: 2 draws") && !has("pass 3:") && has("and 3 more passes with the same shaders (6 draws and dispatches)"));
    assert(has("pass 6: 1 draws, 0 dispatches; last shaders (vertex/fragment/geometry or cs): 000000000000000c/000000000000000d/000000000000000e"));
    assert(!has("0000000000000077"));
    nl = 0; vk_crumbs_log(9, "nothing"); assert(has("nothing kept"));
    // the hang check: an end never written, a slow one, a fast one, off for an untimed slot, three at most
    V.props.limits.timestampPeriod = 1.0f; uint64_t ts[2] = {1000, 1000 + 250000000ull};
    nl = 0; cpend c = {5, 0};
    hang_check(&c, VK_NOT_READY, ts); assert(has("submission 5: its end timestamp never written") && has("[hang] submission 5"));
    nl = 0; hang_check(&c, VK_SUCCESS, ts); assert(has("submission 5: 250 ms on the GPU, over DM_VK_HANG_MS=200"));
    nl = 0; ts[1] = 1000 + 150000000ull; hang_check(&c, VK_SUCCESS, ts); assert(!nl);
    c.slot = -1; hang_check(&c, VK_NOT_READY, ts); assert(!nl);
    c.slot = 0; hang_check(&c, VK_NOT_READY, ts); assert(nl); nl = 0; hang_check(&c, VK_NOT_READY, ts); assert(!nl);
    // a lost device: once, the first submission not completed, then the driver's fault report
    nl = 0; g_done_fence = 4; vk_device_lost(VK_ERROR_DEVICE_LOST, "test");
    assert(has("the GPU was lost (-4, test): submissions completed up to 4 of 5") && has("[hang] submission 5 (the first one not completed"));
    assert(has("no VK_EXT_device_fault"));
    nl = 0; vk_device_lost(VK_ERROR_DEVICE_LOST, "again"); assert(!nl);
    g_fault_info = fake_fault; fake_ad[0] = (VkDeviceFaultAddressInfoEXT){VK_DEVICE_FAULT_ADDRESS_TYPE_WRITE_INVALID_EXT, 0x1234000, 0xfff};
    fault_log(); assert(fault_calls == 2 && has("(5): \"GPU hang\", 20 fault addresses, 1 vendor records") && has("fault: write of an invalid address at 0x1234000 (precision 0xfff)"));
    assert(has("vendor record: \"kgsl\", code 0x7, data 0x9"));
    puts("hang report ok"); return 0;
}
'''


@unittest.skipUnless(shutil.which('cc') and host_cc.posix(), 'a POSIX host C compiler is required')
class HangReport(unittest.TestCase):
    def test_crumbs_hang_check_and_fault_log(self):
        ndk, _ = find_ndk()
        headers = next(ndk.glob('toolchains/llvm/prebuilt/*/sysroot/usr/include'), None) if ndk else None
        if not headers:
            self.skipTest('NDK Vulkan headers are required')
        ex = (ANDROID / 'gfx_vk_exec.c').read_text()
        exec_cut = ex[ex.index('// Crumbs, for a GPU hang or a lost device'):ex.index('void vk_prof_end(void) {')]
        core = (ANDROID / 'gfx_vk_core.c').read_text()
        core_cut = core[core.index('// ---------------- a lost device, a hung submission'):core.index('static void *completion_main(void *a) {')]
        with tempfile.TemporaryDirectory(prefix='nx2a-hang-') as tmp:
            c, exe = Path(tmp) / 'check.c', Path(tmp) / 'check'
            c.write_text(HARNESS.replace('EXEC', exec_cut).replace('CORE', core_cut))
            subprocess.run(['cc', '-std=gnu11', '-Wall', '-Wextra', '-Werror', '-Wno-unused-function', '-Wno-missing-field-initializers', '-Wno-format-truncation',
                            '-idirafter', str(headers), str(c), '-o', str(exe)], check=True)
            out = subprocess.run([str(exe)], capture_output=True, text=True, timeout=60)
            self.assertEqual(out.returncode, 0, out.stderr)
            self.assertIn('hang report ok', out.stdout)


if __name__ == '__main__':
    unittest.main()
