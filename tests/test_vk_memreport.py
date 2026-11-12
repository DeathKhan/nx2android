"""DM_VK_MEMREPORT (gfx_vk_memreport.inc): the driver's memory reports and our own object counts, with a fake driver."""
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from porter.toolchain import find_ndk
sys.path.insert(0, str(Path(__file__).resolve().parent))
import host_cc   # noqa: E402

DM = Path(__file__).resolve().parents[1] / 'nxnative/dm'

HARNESS = r'''
#define _GNU_SOURCE
#define VK_NO_PROTOTYPES
#include <vulkan/vulkan_core.h>
#include <assert.h>
#include <errno.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "nvn_hmap.h"
static struct { VkPhysicalDevice pd; } V;
static char logged[4][2048]; static int nlogged, tables;
#define VKLOG(...) do { snprintf(logged[nlogged & 3], sizeof logged[0], __VA_ARGS__); nlogged++; } while (0)
static int has_ext(const VkExtensionProperties *e, uint32_t n, const char *name) { for (uint32_t i = 0; i < n; i++) if (!strcmp(e[i].extensionName, name)) return 1; return 0; }
static uint64_t kgsl_mapped(void) { return 900ull << 20; }
static uint64_t g_macct_total = 600ull << 20;
static uint64_t vk_img_vkbytes(void) { return 300ull << 20; }
static void vk_mem_acct_report(const char *why) { (void)why; tables++; }
static int feature;
static void features2(VkPhysicalDevice d, VkPhysicalDeviceFeatures2 *f) { (void)d; ((VkPhysicalDeviceDeviceMemoryReportFeaturesEXT *)f->pNext)->deviceMemoryReport = feature; }
static PFN_vkGetPhysicalDeviceFeatures2 vkGetPhysicalDeviceFeatures2 = features2;
static uintptr_t next_handle = 0x1000;
static VkResult fake_image(VkDevice d, const VkImageCreateInfo *ci, const VkAllocationCallbacks *a, VkImage *o) { (void)d; (void)ci; (void)a; *o = (VkImage)(next_handle += 16); return VK_SUCCESS; }
static void fake_destroy_image(VkDevice d, VkImage i, const VkAllocationCallbacks *a) { (void)d; (void)i; (void)a; }
static int asked_requirements;
static void fake_image_req(VkDevice d, VkImage i, VkMemoryRequirements *r) { (void)d; (void)i; asked_requirements++; r->size = 4u << 20; }
static VkResult fake_dpool(VkDevice d, const VkDescriptorPoolCreateInfo *ci, const VkAllocationCallbacks *a, VkDescriptorPool *o) { (void)d; (void)ci; (void)a; *o = (VkDescriptorPool)(next_handle += 16); return VK_SUCCESS; }
static void fake_destroy_dpool(VkDevice d, VkDescriptorPool p, const VkAllocationCallbacks *a) { (void)d; (void)p; (void)a; }
static VkResult fake_gpipe(VkDevice d, VkPipelineCache c, uint32_t n, const VkGraphicsPipelineCreateInfo *ci, const VkAllocationCallbacks *a, VkPipeline *o) {
    (void)d; (void)c; (void)ci; (void)a; for (uint32_t i = 0; i < n; i++) o[i] = i == 1 ? VK_NULL_HANDLE : (VkPipeline)(next_handle += 16); return VK_PIPELINE_COMPILE_REQUIRED; }
static void fake_destroy_pipe(VkDevice d, VkPipeline p, const VkAllocationCallbacks *a) { (void)d; (void)p; (void)a; }
static PFN_vkCreateImage vkCreateImage = fake_image; static PFN_vkDestroyImage vkDestroyImage = fake_destroy_image;
static PFN_vkGetImageMemoryRequirements vkGetImageMemoryRequirements = fake_image_req; static PFN_vkGetBufferMemoryRequirements vkGetBufferMemoryRequirements;
static PFN_vkCreateBuffer vkCreateBuffer; static PFN_vkDestroyBuffer vkDestroyBuffer;
static PFN_vkCreateDescriptorPool vkCreateDescriptorPool = fake_dpool; static PFN_vkDestroyDescriptorPool vkDestroyDescriptorPool = fake_destroy_dpool;
static PFN_vkCreateGraphicsPipelines vkCreateGraphicsPipelines = fake_gpipe; static PFN_vkCreateComputePipelines vkCreateComputePipelines; static PFN_vkDestroyPipeline vkDestroyPipeline = fake_destroy_pipe;
static PFN_vkCreateQueryPool vkCreateQueryPool; static PFN_vkCreateCommandPool vkCreateCommandPool;
#include "gfx_vk_memreport.inc"
#define MB (1ull << 20)
int main(void) {
    // off: nothing chained, the DM_CTL line still says what is always counted
    VkExtensionProperties ext[2] = {{"VK_KHR_swapchain", 1}, {"VK_EXT_device_memory_report", 1}};
    const char *on[4]; uint32_t non = 0; int next;
    unsetenv("DM_VK_MEMREPORT"); assert(mr_chain(ext, 2, &next, on, &non) == &next && !non && !g_mr.period);
    vk_memreport("DM_CTL"); assert(nlogged == 1 && tables == 1);
    assert(strstr(logged[0], "GPU memory mapped 900 MB, through vkAllocateMemory 600 MB") && strstr(logged[0], "texture images 300 MB") && strstr(logged[0], "not counted"));
    // a driver without the extension: our own accounting, every 30 s
    setenv("DM_VK_MEMREPORT", "30", 1); assert(mr_chain(ext, 1, &next, on, &non) == &next && !non && g_mr.period == 30 && !g_mr.ext);
    // the extension without its feature, then with it (DM_VK_MEMREPORT=1: every 10 s)
    setenv("DM_VK_MEMREPORT", "1", 1); feature = 0; assert(mr_chain(ext, 2, &next, on, &non) == &next && !non && !g_mr.ext);
    feature = 1; VkBaseOutStructure *c = mr_chain(ext, 2, &next, on, &non);
    assert(c && c->sType == VK_STRUCTURE_TYPE_DEVICE_DEVICE_MEMORY_REPORT_CREATE_INFO_EXT && c->pNext->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DEVICE_MEMORY_REPORT_FEATURES_EXT);
    assert((void *)c->pNext->pNext == &next && non == 1 && !strcmp(on[0], "VK_EXT_device_memory_report") && g_mr.ext && g_mr.period == 10);
    assert(((VkDeviceDeviceMemoryReportCreateInfoEXT *)c)->pfnUserCallback == mr_driver_cb);
    // the driver's reports: two descriptor pools, a pipeline, an import, a free, a free of an id never reported, a failure, an id reported twice
    VkDeviceMemoryReportCallbackDataEXT d = {VK_STRUCTURE_TYPE_DEVICE_MEMORY_REPORT_CALLBACK_DATA_EXT};
    d.type = VK_DEVICE_MEMORY_REPORT_EVENT_TYPE_ALLOCATE_EXT; d.objectType = VK_OBJECT_TYPE_DESCRIPTOR_POOL;
    d.memoryObjectId = 1; d.size = 12 * MB; mr_driver_cb(&d, NULL); d.memoryObjectId = 2; mr_driver_cb(&d, NULL);
    d.memoryObjectId = 3; d.size = MB; d.objectType = VK_OBJECT_TYPE_PIPELINE; mr_driver_cb(&d, NULL);
    d.memoryObjectId = 4; d.size = 3 * MB; d.objectType = VK_OBJECT_TYPE_ACCELERATION_STRUCTURE_KHR; mr_driver_cb(&d, NULL);
    d.type = VK_DEVICE_MEMORY_REPORT_EVENT_TYPE_IMPORT_EXT; d.memoryObjectId = 5; d.size = 256 * MB; d.objectType = VK_OBJECT_TYPE_DEVICE_MEMORY; mr_driver_cb(&d, NULL);
    d.type = VK_DEVICE_MEMORY_REPORT_EVENT_TYPE_FREE_EXT; d.memoryObjectId = 2; mr_driver_cb(&d, NULL); d.memoryObjectId = 99; mr_driver_cb(&d, NULL);
    d.type = VK_DEVICE_MEMORY_REPORT_EVENT_TYPE_ALLOCATION_FAILED_EXT; mr_driver_cb(&d, NULL);
    d.type = VK_DEVICE_MEMORY_REPORT_EVENT_TYPE_ALLOCATE_EXT; d.memoryObjectId = 3; d.size = 2 * MB; d.objectType = VK_OBJECT_TYPE_PIPELINE; mr_driver_cb(&d, NULL);
    assert(g_mr.drv[VK_OBJECT_TYPE_DESCRIPTOR_POOL] == 12 * MB && g_mr.drv_n[VK_OBJECT_TYPE_DESCRIPTOR_POOL] == 1);
    assert(g_mr.drv[VK_OBJECT_TYPE_PIPELINE] == 2 * MB && g_mr.drv_n[VK_OBJECT_TYPE_PIPELINE] == 1 && g_mr.drv[MR_TYPES - 1] == 3 * MB);
    assert(g_mr.imported == 256 * MB && g_mr.imported_n == 1 && g_mr.failed == 1 && g_mr.unseen == 1);
    d.type = VK_DEVICE_MEMORY_REPORT_EVENT_TYPE_UNIMPORT_EXT; d.memoryObjectId = 5; mr_driver_cb(&d, NULL); assert(!g_mr.imported && !g_mr.imported_n);
    // ours: through the wrappers, as mr_install puts them in the table
    mr_vkCreateImage = vkCreateImage; mr_vkDestroyImage = vkDestroyImage; mr_vkCreateDescriptorPool = vkCreateDescriptorPool; mr_vkDestroyDescriptorPool = vkDestroyDescriptorPool;
    mr_vkCreateGraphicsPipelines = vkCreateGraphicsPipelines; mr_vkDestroyPipeline = vkDestroyPipeline;
    VkImageCreateInfo ici = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO}; VkImage im[3];
    for (int i = 0; i < 3; i++) assert(mr_create_image(NULL, &ici, NULL, &im[i]) == VK_SUCCESS);
    VkExternalMemoryImageCreateInfo xi = {VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO}; ici.pNext = &xi; VkImage ahb;
    assert(mr_create_image(NULL, &ici, NULL, &ahb) == VK_SUCCESS && asked_requirements == 3);   // an AHardwareBuffer's image: counted, not sized before binding
    mr_destroy_image(NULL, im[1], NULL); mr_destroy_image(NULL, VK_NULL_HANDLE, NULL);
    assert(g_mr.n[MR_IMAGE] == 3 && g_mr.bytes[MR_IMAGE] == 8 * MB);
    VkDescriptorPool dp; assert(!mr_create_dpool(NULL, NULL, NULL, &dp)); mr_destroy_dpool(NULL, dp, NULL); assert(!mr_create_dpool(NULL, NULL, NULL, &dp));
    VkPipeline p[3]; VkGraphicsPipelineCreateInfo gci[3] = {{0}}; assert(mr_create_gpipe(NULL, VK_NULL_HANDLE, 3, gci, NULL, p) == VK_PIPELINE_COMPILE_REQUIRED);
    mr_destroy_pipe(NULL, p[0], NULL);
    assert(g_mr.n[MR_DPOOL] == 1 && g_mr.n[MR_PIPE] == 1);
    int base = nlogged; vk_memreport("timer");
    assert(nlogged == base + 2 && tables == 2);
    const char *own = logged[base & 3], *drv = logged[(base + 1) & 3];
    assert(strstr(own, "images 3 (8 MB), buffers 0 (0 MB), descriptor pools 1, pipelines 1, query pools 0, command pools 0"));
    assert(strstr(drv, "the driver's: 17 MB live, imports 0 MB (0);") && strstr(drv, " descriptor pool 12.0 MB (1),") && strstr(drv, " pipeline 2.0 MB (1),"));
    assert(strstr(drv, " later types 3.0 MB (1),") && strstr(drv, "1 failed allocations, 1 frees of objects not reported") && !strstr(drv, "device memory"));
    puts("memreport ok");
    return 0;
}
'''


@unittest.skipUnless(shutil.which('cc') and host_cc.posix() and sys.platform.startswith('linux'), 'a Linux host C compiler is required (the two-argument pthread_setname_np)')
class MemReport(unittest.TestCase):
    def test_driver_reports_and_own_counts(self):
        ndk, _ = find_ndk()
        headers = next(ndk.glob('toolchains/llvm/prebuilt/*/sysroot/usr/include'), None) if ndk else None
        if not headers:
            self.skipTest('NDK Vulkan headers are required')
        with tempfile.TemporaryDirectory(prefix='nx2a-memreport-') as tmp:
            c, exe = Path(tmp) / 'check.c', Path(tmp) / 'check'
            c.write_text(HARNESS, encoding='utf-8')
            subprocess.run(['cc', '-std=gnu11', '-Wall', '-Wextra', '-Werror', '-Wno-unused-function', '-Wno-missing-field-initializers', '-pthread', '-I', str(DM / 'android'), '-I', str(DM / 'gfx/nvn'),
                            '-idirafter', str(headers), str(c), '-o', str(exe)], check=True)
            out = subprocess.run([str(exe)], capture_output=True, text=True)
            self.assertEqual(out.returncode, 0, out.stderr)
            self.assertIn('memreport ok', out.stdout)


if __name__ == '__main__':
    unittest.main()
