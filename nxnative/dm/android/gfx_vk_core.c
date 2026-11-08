// Android port, Vulkan backend: device, memory pools, submissions, completion (see gfx_vk.h).
//   Memory pools: NVN gives the GPU the game's own memory (GPU address = CPU address). Each pool's pages are replaced by an AHardwareBuffer mapped at the same
//   address (its dma-buf mmapped MAP_FIXED, contents copied over), and the same AHardwareBuffer is imported as a VkBuffer: the game's pointer is GPU memory,
//   CPU-cached and coherent on Adreno (measured: probes/vkzc.c). A pool the allocator refuses keeps its anonymous pages and gets a GPU copy refreshed before
//   every submission (vk_pools_sync), correct but slow, reported once.
//   Submissions: a ring of slots, each with a command pool (upload + main command buffers), descriptor pools, scratch memory and objects freed once the GPU has
//   passed it. A submission signals a timeline semaphore with its fence; a completion thread waits for it and then applies what the CPU may see of its work
//   (event writes, counter reports), as nvn_mtl.m does in its completion handler.
#define _GNU_SOURCE
#include "gfx_vk.h"
#include "nvn_queue.h"
#include "nvn_hmap.h"
#include "nvn_dirty.h"
#include <fcntl.h>
#include <unistd.h>
#include <android/hardware_buffer.h>
#include <dlfcn.h>
#include <errno.h>
#include <math.h>
#include <stdatomic.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>

#define VK_DEF(f) PFN_##f f;
VK_FNS(VK_DEF) VK_DEV_FNS(VK_DEF)
static PFN_vkGetInstanceProcAddr p_vkGetInstanceProcAddr;
vkdev_t V;
pthread_mutex_t g_vk_queue_mu = PTHREAD_MUTEX_INITIALIZER;   // VkQueue access is externally synchronized: submissions and presents

// DM_VK_MEM_DIAG=1 at startup: observe copied-pool costs without changing memory protection or rendering.
static int mem_diag_on(void) {
    static _Thread_local int on = -1;
    if (on < 0) { const char *e = getenv("DM_VK_MEM_DIAG"); on = e && atoi(e); }
    return on;
}
static uint64_t mem_diag_ns(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return (uint64_t)t.tv_sec * 1000000000ull + (uint64_t)t.tv_nsec;
}
static void mem_diag_zc(unsigned stage, const char *name, uint64_t size, const char *kind, int error) {
    if (!mem_diag_on()) return;
    static _Atomic unsigned seen;
    if (atomic_fetch_or(&seen, 1u << stage) & (1u << stage)) return;
    VKLOG("[memdiag] zero-copy failed at %s: %s=%d, size=%llu, page_size=%d", name, kind, error, (unsigned long long)size, getpagesize());
}
static void mem_diag_note(unsigned phase, uint64_t start, uint64_t bytes) {
    static const char *const names[] = {"completion-wait", "cpu-to-gpu", "gpu-to-cpu", "pool-flush", "pool-invalidate", "merge-scan", "sync-scan", "sync-protect", "sync-upload", "sync-baseline"};
    static _Thread_local struct { uint64_t calls, ns, bytes, last; } stats[10];
    uint64_t end = mem_diag_ns(), elapsed = end - start;
    stats[phase].calls++; stats[phase].ns += elapsed; stats[phase].bytes += bytes;
    if (stats[phase].calls > 3 && end - stats[phase].last < 1000000000ull) return;
    uint64_t faults, preps; nvn_dirty_stats(&faults, &preps);
    // These page counts are process-wide, may overlap other threads, and are not SIGSEGV event counts.
    VKLOG("[memdiag] %s: last %.2f ms / %.2f MiB, mean %.2f ms / %.2f MiB (%llu calls); process fault-released-pages=%llu prep-pages=%llu",
          names[phase], elapsed / 1e6, bytes / 1048576.0, stats[phase].ns / 1e6 / stats[phase].calls,
          stats[phase].bytes / 1048576.0 / stats[phase].calls, (unsigned long long)stats[phase].calls,
          (unsigned long long)faults, (unsigned long long)preps);
    stats[phase].last = end;
}

static void mem_diag_copyback(uint64_t total, uint64_t invalidate, uint64_t calls, uint64_t bytes) {
    static _Thread_local uint64_t n, last, total_ns, invalidate_ns, total_bytes;
    n++; total_ns += total; invalidate_ns += invalidate; total_bytes += bytes;
    uint64_t now = mem_diag_ns(); if (n > 3 && now - last < 1000000000ull) return;
    last = now;
    // Invalidation includes its existing wrapper diagnostics; bytes are logical ranges, not allocation padding.
    VKLOG("[memdiag] copyback-breakdown: last total %.2f ms, invalidate %.2f ms (%.1f%%, %llu calls / %.2f MiB), other %.2f ms; mean total %.2f ms, invalidate %.2f ms (%.1f%%, %.2f MiB), %llu submissions",
        total / 1e6, invalidate / 1e6, total ? 100.0 * invalidate / total : 0.0,
        (unsigned long long)calls, bytes / 1048576.0, (total - invalidate) / 1e6,
        total_ns / 1e6 / n, invalidate_ns / 1e6 / n, total_ns ? 100.0 * invalidate_ns / total_ns : 0.0,
        total_bytes / 1048576.0 / n, (unsigned long long)n);
}

static void mem_diag_baselines(const uint64_t count[4], const uint64_t bytes[4]) {
    static _Thread_local uint64_t calls, last, total_count[4], total_bytes[4];
    for (int i = 0; i < 4; i++) { total_count[i] += count[i]; total_bytes[i] += bytes[i]; }
    uint64_t now = mem_diag_ns(); if (++calls > 3 && now - last < 1000000000ull) return;
    last = now;
    static const char *const names[] = {"resync", "untracked", "incomplete-uncached", "incomplete-other"};
    for (int i = 0; i < 4; i++) VKLOG("[memdiag] baseline-reason %s: last %llu copies / %.2f MiB; total %llu copies / %.2f MiB (%llu syncs)",
        names[i], (unsigned long long)count[i], bytes[i] / 1048576.0, (unsigned long long)total_count[i], total_bytes[i] / 1048576.0, (unsigned long long)calls);
}

uint32_t vk_mem_type(uint32_t bits, VkMemoryPropertyFlags want, VkMemoryPropertyFlags avoid) {
    for (uint32_t i = 0; i < V.mem.memoryTypeCount; i++)
        if ((bits >> i & 1) && (V.mem.memoryTypes[i].propertyFlags & want) == want && !(V.mem.memoryTypes[i].propertyFlags & avoid)) return i;
    for (uint32_t i = 0; i < V.mem.memoryTypeCount; i++) if ((bits >> i & 1) && (V.mem.memoryTypes[i].propertyFlags & want) == want) return i;
    return UINT32_MAX;
}
static void mem_diag_pool(uint64_t size, uint32_t bits, uint32_t selected) {
    if (!mem_diag_on()) return;
    VkMemoryPropertyFlags flags = V.mem.memoryTypes[selected].propertyFlags;
    uint32_t cached = vk_mem_type(bits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT, 0);
    VKLOG("[memdiag] copied buffer %.2f MiB: allowed-types %#x, selected type %u flags %#x (host-cached %s); cached candidate %d (candidate lookup only)",
        size / 1048576.0, bits, selected, flags, flags & VK_MEMORY_PROPERTY_HOST_CACHED_BIT ? "yes" : "no", cached == UINT32_MAX ? -1 : (int)cached);
}

static int has_ext(const VkExtensionProperties *e, uint32_t n, const char *name) { for (uint32_t i = 0; i < n; i++) if (!strcmp(e[i].extensionName, name)) return 1; return 0; }
static int is_on(const char *const *on, uint32_t n, const char *name) { for (uint32_t i = 0; i < n; i++) if (!strcmp(on[i], name)) return 1; return 0; }
// a feature struct as vk_init queries it (sType, pNext, then VkBool32s; the largest, descriptor indexing, has 20); featx: its VkBool32s, all off if it is not chained
typedef struct { VkBaseOutStructure h; VkBool32 b[24]; } featx_t;
static VkBool32 *featx(featx_t *x, uint32_t n, VkStructureType s) { static VkBool32 off[24]; for (uint32_t i = 0; i < n; i++) if (x[i].h.sType == s) return x[i].b; return off; }   // (off: only 0 is written to it)

// Report timestamps must share the guest CPU clock, not the completion thread's wake time.
// The optional KHR/EXT API pairs device ticks with CLOCK_MONOTONIC. If only RAW is
// supported, the API call's MONOTONIC bracket bounds the paired device sample instead.
static PFN_vkGetCalibratedTimestampsKHR g_gpu_calibrated;
static VkTimeDomainKHR g_gpu_host_domain;
static uint32_t g_gpu_timestamp_bits;
#define GPU_CLOCK_MAX_DEVIATION_NS 100000ull
#define GPU_CLOCK_MAX_AGE_NS 2000000000ull
typedef struct { uint64_t device, host, deviation; int valid; } gpu_clock_pair;
static void gpu_clock_init(const char *extension) {
    if (!extension || !g_gpu_timestamp_bits || g_gpu_timestamp_bits > 64) return;
    int khr = !strcmp(extension, "VK_KHR_calibrated_timestamps");
    PFN_vkGetPhysicalDeviceCalibrateableTimeDomainsKHR domains = (PFN_vkGetPhysicalDeviceCalibrateableTimeDomainsKHR)p_vkGetInstanceProcAddr(V.inst,
        khr ? "vkGetPhysicalDeviceCalibrateableTimeDomainsKHR" : "vkGetPhysicalDeviceCalibrateableTimeDomainsEXT");
    PFN_vkGetCalibratedTimestampsKHR sample = (PFN_vkGetCalibratedTimestampsKHR)vkGetDeviceProcAddr(V.dev,
        khr ? "vkGetCalibratedTimestampsKHR" : "vkGetCalibratedTimestampsEXT");
    uint32_t count = 0; VkTimeDomainKHR available[64];
    if (!domains || !sample || domains(V.pd, &count, NULL) != VK_SUCCESS || !count || count > 64) return;
    if (domains(V.pd, &count, available) != VK_SUCCESS || count > 64) return;
    int device = 0, mono = 0, raw = 0;
    for (uint32_t i = 0; i < count; i++) { device |= available[i] == VK_TIME_DOMAIN_DEVICE_KHR;
        mono |= available[i] == VK_TIME_DOMAIN_CLOCK_MONOTONIC_KHR; raw |= available[i] == VK_TIME_DOMAIN_CLOCK_MONOTONIC_RAW_KHR; }
    if (!device || (!mono && !raw)) return;
    g_gpu_host_domain = mono ? VK_TIME_DOMAIN_CLOCK_MONOTONIC_KHR : VK_TIME_DOMAIN_CLOCK_MONOTONIC_RAW_KHR;
    g_gpu_calibrated = sample;
    VKLOG("counter clock: %s, %s, %u valid device bits; calibration deviation limit %llu ns", extension,
        mono ? "paired CLOCK_MONOTONIC" : "RAW paired with bounded CLOCK_MONOTONIC call bracket", g_gpu_timestamp_bits,
        (unsigned long long)GPU_CLOCK_MAX_DEVIATION_NS);
}
static gpu_clock_pair gpu_clock_sample(void) {
    gpu_clock_pair p = {0};
    if (!g_gpu_calibrated) return p;
    VkCalibratedTimestampInfoKHR info[2] = {{VK_STRUCTURE_TYPE_CALIBRATED_TIMESTAMP_INFO_KHR, NULL, VK_TIME_DOMAIN_DEVICE_KHR},
                                         {VK_STRUCTURE_TYPE_CALIBRATED_TIMESTAMP_INFO_KHR, NULL, g_gpu_host_domain}};
    struct timespec a, b, ra = {0}, rb = {0}; uint64_t samples[2] = {0}, deviation = 0;
    int raw = g_gpu_host_domain == VK_TIME_DOMAIN_CLOCK_MONOTONIC_RAW_KHR;
    if (clock_gettime(CLOCK_MONOTONIC, &a)) return p;
    if (raw && clock_gettime(CLOCK_MONOTONIC_RAW, &ra)) return p;
    VkResult r = g_gpu_calibrated(V.dev, 2, info, samples, &deviation);
    if (raw && clock_gettime(CLOCK_MONOTONIC_RAW, &rb)) return p;
    if (clock_gettime(CLOCK_MONOTONIC, &b)) return p;
    uint64_t before = (uint64_t)a.tv_sec * 1000000000ull + (uint64_t)a.tv_nsec;
    uint64_t after = (uint64_t)b.tv_sec * 1000000000ull + (uint64_t)b.tv_nsec;
    if (r != VK_SUCCESS || !deviation || deviation > GPU_CLOCK_MAX_DEVIATION_NS || after < before) return p;
    uint64_t host = samples[1];
    if (raw) {
        uint64_t raw_before = (uint64_t)ra.tv_sec * 1000000000ull + (uint64_t)ra.tv_nsec;
        uint64_t raw_after = (uint64_t)rb.tv_sec * 1000000000ull + (uint64_t)rb.tv_nsec;
        if (raw_after < raw_before || (host < raw_before && raw_before - host > deviation) || (host > raw_after && host - raw_after > deviation)) return p;
        // No assumption that RAW and MONOTONIC have the same epoch or rate. The
        // sample happened inside this host interval; include its half-width in the error.
        uint64_t radius = (after - before + 1) / 2;
        if (radius > GPU_CLOCK_MAX_DEVIATION_NS - deviation) return p;
        deviation += radius; host = before + (after - before) / 2;
    } else if ((host < before && before - host > deviation) || (host > after && host - after > deviation)) return p;
    p = (gpu_clock_pair){samples[0], host, deviation, 1}; return p;
}
// Recover a nearby device time across timestampValidBits wrap. A sample can be
// slightly earlier than a completed query within its documented calibration error.
static int gpu_clock_host_ns(const gpu_clock_pair *p, uint64_t query, uint64_t *host) {
    double period = V.props.limits.timestampPeriod;
    uint32_t bits = g_gpu_timestamp_bits;
    if (!p->valid || !bits || bits > 64 || !p->deviation || p->deviation > GPU_CLOCK_MAX_DEVIATION_NS || !isfinite(period) || period <= 0) return 0;
    uint64_t mask = bits == 64 ? UINT64_MAX : (1ull << bits) - 1;
    // A narrow hardware counter cannot unambiguously represent the accepted age window.
    if (((long double)(mask >> 1) + 1) * period <= GPU_CLOCK_MAX_AGE_NS + p->deviation) return 0;
    uint64_t back = (p->device - query) & mask;
    if (back > (mask >> 1)) {
        uint64_t forward = (query - p->device) & mask;
        long double ns = (long double)forward * period;
        if (ns > p->deviation || ns > UINT64_MAX - p->host) return 0;
        *host = p->host + (uint64_t)ns;
    } else {
        long double ns = (long double)back * period;
        if (ns > GPU_CLOCK_MAX_AGE_NS || ns > p->host) return 0;
        *host = p->host - (uint64_t)ns;
    }
    return 1;
}

static void completion_start(void);
static void slots_init(void);
static VKAPI_ATTR VkBool32 VKAPI_CALL vk_validation_msg(VkDebugUtilsMessageSeverityFlagBitsEXT sev, VkDebugUtilsMessageTypeFlagsEXT type, const VkDebugUtilsMessengerCallbackDataEXT *d, void *u) {
    (void)type; (void)u;   // each distinct message ID logged its first 3 times
    enum { NID = 512 }; static int32_t ids[NID]; static uint8_t cnt[NID]; static int nid; static pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;
    pthread_mutex_lock(&mu); int k = 0; for (; k < nid; k++) if (ids[k] == d->messageIdNumber) break;
    if (k == nid && nid < NID) ids[nid++] = d->messageIdNumber;
    int show = k < NID && cnt[k] < 3; if (k < NID && cnt[k] < 255) cnt[k]++; pthread_mutex_unlock(&mu);
    if (show) VKLOG("[RED] validation %s: %s", sev & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT ? "error" : "warning", d->pMessage);
    return VK_FALSE;
}

// Whether a driver offers a Vulkan 1.3 GPU (DM_VK_DRIVER: a Turnip built for another Adreno generation loads and then has none)
// ---------------- device memory accounting ----------------
// Every allocation through the function table (all of gfx_vk_*.c and movie_zc_vk.c) is counted by a tag: the thread's vk_mem_tag where a site sets one
// (pools, scratch, staging, images, gmem table), else by what the memory is bound to (image / buffer, seen at vkBind*Memory) and the caller's offset in the
// library. A line at each 512 MB more of a tag, at each single allocation of 64 MB or more, and the whole table at each 512 MB more in all. Nintendo Switch 2
// Welcome Tour: 5 GB of device memory that was neither imported pools nor images appeared at its first scene load. DM_VK_MEM_ACCT=0 turns it off
__thread const char *vk_mem_tag;
static PFN_vkAllocateMemory real_vkAllocateMemory; static PFN_vkFreeMemory real_vkFreeMemory; static PFN_vkBindImageMemory real_vkBindImageMemory; static PFN_vkBindBufferMemory real_vkBindBufferMemory;
typedef struct { VkDeviceSize size; int tag; } macct_t;
enum { MACCT_TAGS = 32 };
static struct { const char *name; uint64_t live, peak, n, logged; } g_macct[MACCT_TAGS]; static int g_macct_n; static uint64_t g_macct_total, g_macct_total_logged;
static nvn_hmap g_macct_map = {.klen = 8}; static pthread_mutex_t g_macct_mu = PTHREAD_MUTEX_INITIALIZER; static void *g_lib_base;
static int macct_tag(const char *name) { for (int i = 0; i < g_macct_n; i++) if (!strcmp(g_macct[i].name, name)) return i; if (g_macct_n == MACCT_TAGS) return 0; g_macct[g_macct_n].name = strdup(name); return g_macct_n++; }
static void macct_table(const char *why) {
    char line[2048]; int n = snprintf(line, sizeof line, "device memory %s: %.0f MB live;", why, g_macct_total / 1048576.0);
    for (int i = 0; i < g_macct_n && n < (int)sizeof line - 80; i++) if (g_macct[i].live || g_macct[i].peak > (64u << 20)) n += snprintf(line + n, sizeof line - n, " %s %.0f MB (peak %.0f, %llu allocs);", g_macct[i].name, g_macct[i].live / 1048576.0, g_macct[i].peak / 1048576.0, (unsigned long long)g_macct[i].n);
    VKLOG("%s", line);
}
static VKAPI_ATTR VkResult VKAPI_CALL acct_alloc(VkDevice dev, const VkMemoryAllocateInfo *ai, const VkAllocationCallbacks *cb, VkDeviceMemory *out) {
    VkResult r = real_vkAllocateMemory(dev, ai, cb, out); if (r) return r;
    char nm[96]; const char *tag = vk_mem_tag;
    if (!tag) { for (const VkBaseInStructure *x = (const VkBaseInStructure *)ai->pNext; x; x = x->pNext) if (x->sType == VK_STRUCTURE_TYPE_IMPORT_ANDROID_HARDWARE_BUFFER_INFO_ANDROID) tag = "imported-ahb";
        if (!tag) { snprintf(nm, sizeof nm, "untagged@+%llx", (unsigned long long)((uintptr_t)__builtin_return_address(0) - (uintptr_t)g_lib_base)); tag = nm; } }
    pthread_mutex_lock(&g_macct_mu);
    macct_t *e = malloc(sizeof *e); e->size = ai->allocationSize; e->tag = macct_tag(tag); uint64_t key = (uint64_t)(uintptr_t)*out; nvn_hm_put(&g_macct_map, &key, e);
    g_macct[e->tag].live += e->size; g_macct[e->tag].n++; if (g_macct[e->tag].live > g_macct[e->tag].peak) g_macct[e->tag].peak = g_macct[e->tag].live; g_macct_total += e->size;
    int big = e->size >= (64u << 20), cross = g_macct[e->tag].live >= g_macct[e->tag].logged + (512u << 20), all = g_macct_total >= g_macct_total_logged + (512u << 20);
    if (cross) g_macct[e->tag].logged = g_macct[e->tag].live; if (all) g_macct_total_logged = g_macct_total;
    const char *name = g_macct[e->tag].name; uint64_t live = g_macct[e->tag].live, cnt = g_macct[e->tag].n;
    pthread_mutex_unlock(&g_macct_mu);
    if (big) { static int nb; if (nb++ < 40) VKLOG("device memory: %.1f MB allocated as %s (type %u; %s now %.0f MB in %llu allocations)", e->size / 1048576.0, name, ai->memoryTypeIndex, name, live / 1048576.0, (unsigned long long)cnt); }
    if (cross) VKLOG("device memory: %s at %.0f MB (%llu allocations live)", name, live / 1048576.0, (unsigned long long)cnt);
    if (all) macct_table("by tag");
    return r;
}
static void macct_rebind(VkDeviceMemory m, const char *kind) {   // an untagged allocation named by what it is bound to
    pthread_mutex_lock(&g_macct_mu); uint64_t key = (uint64_t)(uintptr_t)m; macct_t *e = nvn_hm_get(&g_macct_map, &key);
    if (e && !strncmp(g_macct[e->tag].name, "untagged@", 9)) { char nm[128]; snprintf(nm, sizeof nm, "%s-%s", kind, g_macct[e->tag].name + 9); int t = macct_tag(nm);
        g_macct[e->tag].live -= e->size; g_macct[e->tag].n--; e->tag = t; g_macct[t].live += e->size; g_macct[t].n++; if (g_macct[t].live > g_macct[t].peak) g_macct[t].peak = g_macct[t].live;
        if (g_macct[t].live >= g_macct[t].logged + (512u << 20)) { g_macct[t].logged = g_macct[t].live; VKLOG("device memory: %s at %.0f MB (%llu allocations live)", nm, g_macct[t].live / 1048576.0, (unsigned long long)g_macct[t].n); } }
    pthread_mutex_unlock(&g_macct_mu);
}
static VKAPI_ATTR VkResult VKAPI_CALL acct_bind_image(VkDevice dev, VkImage i, VkDeviceMemory m, VkDeviceSize off) { macct_rebind(m, "image"); return real_vkBindImageMemory(dev, i, m, off); }
static VKAPI_ATTR VkResult VKAPI_CALL acct_bind_buffer(VkDevice dev, VkBuffer b, VkDeviceMemory m, VkDeviceSize off) { macct_rebind(m, "buffer"); return real_vkBindBufferMemory(dev, b, m, off); }
// Pipelines: the driver allocates their shader and private (spill / local) memory itself; measured as the process's mapped GPU memory around the call
static PFN_vkCreateGraphicsPipelines real_vkCreateGraphicsPipelines; static PFN_vkCreateComputePipelines real_vkCreateComputePipelines;
static uint64_t g_pipe_kgsl, g_pipe_n, g_pipe_big;
static uint64_t kgsl_mapped(void);
static void pipe_note(const char *kind, uint32_t n, uint64_t before, const VkPipelineShaderStageCreateInfo *st, uint32_t nst) {
    uint64_t after = kgsl_mapped(), d = after > before ? after - before : 0; g_pipe_kgsl += d; g_pipe_n += n;
    if (d >= (16u << 20)) { g_pipe_big++; static int nl; if (nl++ < 24) { char mods[160]; int k = 0; for (uint32_t i = 0; i < nst && k < 120; i++) k += snprintf(mods + k, sizeof mods - k, " %s%llx", i ? "" : "", (unsigned long long)(uintptr_t)st[i].module);
        VKLOG("%s pipeline: %.1f MB of GPU memory mapped for it (stages %u: modules%s); pipelines %llu so far, %.0f MB in all, %llu of 16 MB or more", kind, d / 1048576.0, nst, mods, (unsigned long long)g_pipe_n, g_pipe_kgsl / 1048576.0, (unsigned long long)g_pipe_big); } }
    if (g_pipe_n && !(g_pipe_n & 63)) VKLOG("pipelines: %llu made, %.0f MB of GPU memory mapped around their creation (%llu of 16 MB or more)", (unsigned long long)g_pipe_n, g_pipe_kgsl / 1048576.0, (unsigned long long)g_pipe_big);
}
static VKAPI_ATTR VkResult VKAPI_CALL acct_gpipe(VkDevice dev, VkPipelineCache pc, uint32_t n, const VkGraphicsPipelineCreateInfo *ci, const VkAllocationCallbacks *cb, VkPipeline *out) {
    uint64_t b = kgsl_mapped(); VkResult r = real_vkCreateGraphicsPipelines(dev, pc, n, ci, cb, out); pipe_note("graphics", n, b, ci->pStages, ci->stageCount); return r; }
static VKAPI_ATTR VkResult VKAPI_CALL acct_cpipe(VkDevice dev, VkPipelineCache pc, uint32_t n, const VkComputePipelineCreateInfo *ci, const VkAllocationCallbacks *cb, VkPipeline *out) {
    uint64_t b = kgsl_mapped(); VkResult r = real_vkCreateComputePipelines(dev, pc, n, ci, cb, out); pipe_note("compute", n, b, &ci->stage, 1); return r; }
// ---------------- descriptor set layouts with large arrays: update-after-bind ----------------
// Adreno's vkCmdBindDescriptorSets copies a bound set's descriptors into the command stream: Nintendo Switch 2 Welcome Tour binds its texture / sampler tables
// (bindless arrays of tens of thousands of descriptors, RK_TEXD / RK_SAMPD) at most draws of its first scene, 12 MB of GPU memory a bind, 3.3 GB in one submission,
// and the low-memory killer ended it (measured with DM_VK_KGSL_TRACE=1). A binding flagged update-after-bind lives in the pool's memory and is bound by reference.
// Every layout with a binding of UAB_MIN descriptors or more gets the flag on those bindings (not on dynamic buffers, which the rule forbids), and every pool the
// update-after-bind bit, when the device has the Vulkan 1.2 features (g_uab: 1 sampled images and samplers, 2 also storage images, texel buffers and storage buffers).
// DM_VK_NO_UAB=1: off
static int g_uab; enum { UAB_MIN = 1024 };
static PFN_vkCreateDescriptorSetLayout real_vkCreateDescriptorSetLayout; static uint64_t g_uab_layouts;
static int uab_type_ok(VkDescriptorType t) {
    switch (t) { case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE: case VK_DESCRIPTOR_TYPE_SAMPLER: case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER: return g_uab >= 1;
        case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE: case VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER: case VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER: case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER: return g_uab >= 2; default: return 0; }
}
static VKAPI_ATTR VkResult VKAPI_CALL uab_dsl(VkDevice dev, const VkDescriptorSetLayoutCreateInfo *ci, const VkAllocationCallbacks *cb, VkDescriptorSetLayout *out) {
    int big = 0; for (uint32_t i = 0; i < ci->bindingCount; i++) if (ci->pBindings[i].descriptorCount >= UAB_MIN && uab_type_ok(ci->pBindings[i].descriptorType)) big = 1;
    if (!big || ci->pNext) return real_vkCreateDescriptorSetLayout(dev, ci, cb, out);   // (a chained info of the caller's: left as it is)
    VkDescriptorBindingFlags fl[128]; uint32_t n = ci->bindingCount < 128 ? ci->bindingCount : 128, maxc = 0;
    for (uint32_t i = 0; i < n; i++) { const VkDescriptorSetLayoutBinding *b = &ci->pBindings[i]; fl[i] = b->descriptorCount >= UAB_MIN && uab_type_ok(b->descriptorType) ? VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT | VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT : 0; if (b->descriptorCount > maxc) maxc = b->descriptorCount; }
    VkDescriptorSetLayoutBindingFlagsCreateInfo bf = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO, NULL, n, fl};
    VkDescriptorSetLayoutCreateInfo c2 = *ci; c2.pNext = &bf; c2.flags |= VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT;
    VkResult r = real_vkCreateDescriptorSetLayout(dev, &c2, cb, out);
    if (r) { VKRED_ONCE(160, "descriptor set layout with update-after-bind arrays refused (%d): made as before", r); return real_vkCreateDescriptorSetLayout(dev, ci, cb, out); }
    g_uab_layouts++; static int nl; if (nl++ < 4) VKLOG("descriptor set layout %llu with update-after-bind arrays (%u bindings, the largest %u descriptors)", (unsigned long long)g_uab_layouts, n, maxc);
    return r;
}
static void uab_install(void) {
    const char *e = getenv("DM_VK_UAB"); if (!(e && atoi(e)) || !g_uab) { g_uab = 0; return; }   // opt-in: on the Adreno 830 (512.800) the driver copied the set into the command stream all the same (12 MB a bind, measured with DM_VK_KGSL_TRACE=1)
    real_vkCreateDescriptorSetLayout = vkCreateDescriptorSetLayout; vkCreateDescriptorSetLayout = uab_dsl;
    VKLOG("descriptor set layouts with arrays of %d or more get update-after-bind (level %d; DM_VK_UAB=1)", UAB_MIN, g_uab);
}
// ---------------- GPU memory trace (DM_VK_KGSL_TRACE=1) ----------------
// The driver allocates GPU memory of its own as commands are recorded (command memory, visibility streams, per-pass and per-draw state): each wrapped entry point
// measures the process's mapped GPU memory around the call and charges the difference to the call's name. A call that maps 1 MB or more is logged (the first 80), and
// the totals by name go into the submission's line (sub_stats). Nintendo Switch 2 Welcome Tour: 3.3 GB mapped while recording 430 draws of its first scene
enum { KT_N = 36 };
static const char *g_kt_name[KT_N] = {"vkCmdDraw", "vkCmdDrawIndexed", "vkCmdDrawIndirect", "vkCmdDrawIndexedIndirect", "vkCmdDispatch", "vkCmdBeginRenderPass", "vkCmdBeginRendering", "vkCmdEndRenderPass", "vkCmdEndRendering", "vkCmdPipelineBarrier", "vkCmdCopyImage", "vkCmdCopyBufferToImage", "vkCmdCopyImageToBuffer", "vkCmdBlitImage", "vkCmdClearColorImage", "vkCmdClearDepthStencilImage", "vkCmdClearAttachments", "vkCmdResolveImage", "vkCmdFillBuffer", "vkCmdUpdateBuffer", "vkCmdCopyBuffer", "vkCmdBindDescriptorSets", "vkCmdBindPipeline", "vkCmdBindVertexBuffers", "vkCmdBindIndexBuffer", "vkCmdPushConstants", "vkCmdSetViewport", "vkCmdSetScissor", "vkCreateImageView", "vkCreateFramebuffer", "vkAllocateDescriptorSets", "vkUpdateDescriptorSets", "vkEndCommandBuffer", "vkBeginCommandBuffer", "vkQueueSubmit", "vkCreateImage"};
static uint64_t g_kt_bytes[KT_N], g_kt_calls[KT_N], g_kt_big[KT_N]; static int g_kt_on;
static void kt_note(int k, uint64_t before, const char *info) {
    uint64_t after = kgsl_mapped(); g_kt_calls[k]++; if (after <= before) return; uint64_t d = after - before; g_kt_bytes[k] += d;
    if (d >= (1u << 20)) { g_kt_big[k]++; static int nl; if (nl++ < 80) VKLOG("gpu memory +%.1f MB at %s (%s); %.0f MB mapped", d / 1048576.0, g_kt_name[k], info, after / 1048576.0); }
}
static void kt_table(void) {
    if (!g_kt_on) return; char line[2048]; int n = snprintf(line, sizeof line, "gpu memory by call since the start:");
    for (int k = 0; k < KT_N && n < (int)sizeof line - 96; k++) if (g_kt_bytes[k] >= (1u << 20)) n += snprintf(line + n, sizeof line - n, " %s %.0f MB (%llu calls, %llu of 1 MB+);", g_kt_name[k], g_kt_bytes[k] / 1048576.0, (unsigned long long)g_kt_calls[k], (unsigned long long)g_kt_big[k]);
    VKLOG("%s", line);
}
static PFN_vkCmdDraw real_vkCmdDraw;
static VKAPI_ATTR void VKAPI_CALL kt_vkCmdDraw(VkCommandBuffer cb, uint32_t vc, uint32_t ic, uint32_t fv, uint32_t fi) { uint64_t kt_b0 = kgsl_mapped(); real_vkCmdDraw(cb, vc, ic, fv, fi); char info[160]; snprintf(info, sizeof info, "%u vertices x %u", vc, ic); kt_note(0, kt_b0, info); }
static PFN_vkCmdDrawIndexed real_vkCmdDrawIndexed;
static VKAPI_ATTR void VKAPI_CALL kt_vkCmdDrawIndexed(VkCommandBuffer cb, uint32_t ic, uint32_t inst, uint32_t fi, int32_t vo, uint32_t fin) { uint64_t kt_b0 = kgsl_mapped(); real_vkCmdDrawIndexed(cb, ic, inst, fi, vo, fin); char info[160]; snprintf(info, sizeof info, "%u indices x %u", ic, inst); kt_note(1, kt_b0, info); }
static PFN_vkCmdDrawIndirect real_vkCmdDrawIndirect;
static VKAPI_ATTR void VKAPI_CALL kt_vkCmdDrawIndirect(VkCommandBuffer cb, VkBuffer b, VkDeviceSize o, uint32_t n, uint32_t st) { uint64_t kt_b0 = kgsl_mapped(); real_vkCmdDrawIndirect(cb, b, o, n, st); char info[160]; snprintf(info, sizeof info, "%u records", n); kt_note(2, kt_b0, info); }
static PFN_vkCmdDrawIndexedIndirect real_vkCmdDrawIndexedIndirect;
static VKAPI_ATTR void VKAPI_CALL kt_vkCmdDrawIndexedIndirect(VkCommandBuffer cb, VkBuffer b, VkDeviceSize o, uint32_t n, uint32_t st) { uint64_t kt_b0 = kgsl_mapped(); real_vkCmdDrawIndexedIndirect(cb, b, o, n, st); char info[160]; snprintf(info, sizeof info, "%u records", n); kt_note(3, kt_b0, info); }
static PFN_vkCmdDispatch real_vkCmdDispatch;
static VKAPI_ATTR void VKAPI_CALL kt_vkCmdDispatch(VkCommandBuffer cb, uint32_t x, uint32_t y, uint32_t z) { uint64_t kt_b0 = kgsl_mapped(); real_vkCmdDispatch(cb, x, y, z); char info[160]; snprintf(info, sizeof info, "%ux%ux%u", x, y, z); kt_note(4, kt_b0, info); }
static PFN_vkCmdBeginRenderPass real_vkCmdBeginRenderPass;
static VKAPI_ATTR void VKAPI_CALL kt_vkCmdBeginRenderPass(VkCommandBuffer cb, const VkRenderPassBeginInfo *bi, VkSubpassContents c) { uint64_t kt_b0 = kgsl_mapped(); real_vkCmdBeginRenderPass(cb, bi, c); char info[160]; snprintf(info, sizeof info, "%ux%u", bi->renderArea.extent.width, bi->renderArea.extent.height); kt_note(5, kt_b0, info); }
static PFN_vkCmdBeginRendering real_vkCmdBeginRendering;
static VKAPI_ATTR void VKAPI_CALL kt_vkCmdBeginRendering(VkCommandBuffer cb, const VkRenderingInfo *ri) { uint64_t kt_b0 = kgsl_mapped(); real_vkCmdBeginRendering(cb, ri); char info[160]; snprintf(info, sizeof info, "%ux%u, %u colour attachments, depth %d, layers %u", ri->renderArea.extent.width, ri->renderArea.extent.height, ri->colorAttachmentCount, ri->pDepthAttachment != NULL, ri->layerCount); kt_note(6, kt_b0, info); }
static PFN_vkCmdEndRenderPass real_vkCmdEndRenderPass;
static VKAPI_ATTR void VKAPI_CALL kt_vkCmdEndRenderPass(VkCommandBuffer cb) { uint64_t kt_b0 = kgsl_mapped(); real_vkCmdEndRenderPass(cb); char info[160]; snprintf(info, sizeof info, "%s", ""); kt_note(7, kt_b0, info); }
static PFN_vkCmdEndRendering real_vkCmdEndRendering;
static VKAPI_ATTR void VKAPI_CALL kt_vkCmdEndRendering(VkCommandBuffer cb) { uint64_t kt_b0 = kgsl_mapped(); real_vkCmdEndRendering(cb); char info[160]; snprintf(info, sizeof info, "%s", ""); kt_note(8, kt_b0, info); }
static PFN_vkCmdPipelineBarrier real_vkCmdPipelineBarrier;
static VKAPI_ATTR void VKAPI_CALL kt_vkCmdPipelineBarrier(VkCommandBuffer cb, VkPipelineStageFlags a, VkPipelineStageFlags b, VkDependencyFlags d, uint32_t nm, const VkMemoryBarrier *m, uint32_t nb, const VkBufferMemoryBarrier *bm, uint32_t ni, const VkImageMemoryBarrier *im) { uint64_t kt_b0 = kgsl_mapped(); real_vkCmdPipelineBarrier(cb, a, b, d, nm, m, nb, bm, ni, im); char info[160]; snprintf(info, sizeof info, "%u image barriers", ni); kt_note(9, kt_b0, info); }
static PFN_vkCmdCopyImage real_vkCmdCopyImage;
static VKAPI_ATTR void VKAPI_CALL kt_vkCmdCopyImage(VkCommandBuffer cb, VkImage si, VkImageLayout sl, VkImage di, VkImageLayout dl, uint32_t n, const VkImageCopy *r) { uint64_t kt_b0 = kgsl_mapped(); real_vkCmdCopyImage(cb, si, sl, di, dl, n, r); char info[160]; snprintf(info, sizeof info, "%ux%ux%u", r->extent.width, r->extent.height, r->extent.depth); kt_note(10, kt_b0, info); }
static PFN_vkCmdCopyBufferToImage real_vkCmdCopyBufferToImage;
static VKAPI_ATTR void VKAPI_CALL kt_vkCmdCopyBufferToImage(VkCommandBuffer cb, VkBuffer sb, VkImage di, VkImageLayout dl, uint32_t n, const VkBufferImageCopy *r) { uint64_t kt_b0 = kgsl_mapped(); real_vkCmdCopyBufferToImage(cb, sb, di, dl, n, r); char info[160]; snprintf(info, sizeof info, "%ux%ux%u", r->imageExtent.width, r->imageExtent.height, r->imageExtent.depth); kt_note(11, kt_b0, info); }
static PFN_vkCmdCopyImageToBuffer real_vkCmdCopyImageToBuffer;
static VKAPI_ATTR void VKAPI_CALL kt_vkCmdCopyImageToBuffer(VkCommandBuffer cb, VkImage si, VkImageLayout sl, VkBuffer db, uint32_t n, const VkBufferImageCopy *r) { uint64_t kt_b0 = kgsl_mapped(); real_vkCmdCopyImageToBuffer(cb, si, sl, db, n, r); char info[160]; snprintf(info, sizeof info, "%ux%ux%u", r->imageExtent.width, r->imageExtent.height, r->imageExtent.depth); kt_note(12, kt_b0, info); }
static PFN_vkCmdBlitImage real_vkCmdBlitImage;
static VKAPI_ATTR void VKAPI_CALL kt_vkCmdBlitImage(VkCommandBuffer cb, VkImage si, VkImageLayout sl, VkImage di, VkImageLayout dl, uint32_t n, const VkImageBlit *r, VkFilter f) { uint64_t kt_b0 = kgsl_mapped(); real_vkCmdBlitImage(cb, si, sl, di, dl, n, r, f); char info[160]; snprintf(info, sizeof info, "%dx%d", r->dstOffsets[1].x, r->dstOffsets[1].y); kt_note(13, kt_b0, info); }
static PFN_vkCmdClearColorImage real_vkCmdClearColorImage;
static VKAPI_ATTR void VKAPI_CALL kt_vkCmdClearColorImage(VkCommandBuffer cb, VkImage i, VkImageLayout l, const VkClearColorValue *v, uint32_t n, const VkImageSubresourceRange *r) { uint64_t kt_b0 = kgsl_mapped(); real_vkCmdClearColorImage(cb, i, l, v, n, r); char info[160]; snprintf(info, sizeof info, "%u ranges", n); kt_note(14, kt_b0, info); }
static PFN_vkCmdClearDepthStencilImage real_vkCmdClearDepthStencilImage;
static VKAPI_ATTR void VKAPI_CALL kt_vkCmdClearDepthStencilImage(VkCommandBuffer cb, VkImage i, VkImageLayout l, const VkClearDepthStencilValue *v, uint32_t n, const VkImageSubresourceRange *r) { uint64_t kt_b0 = kgsl_mapped(); real_vkCmdClearDepthStencilImage(cb, i, l, v, n, r); char info[160]; snprintf(info, sizeof info, "%u ranges", n); kt_note(15, kt_b0, info); }
static PFN_vkCmdClearAttachments real_vkCmdClearAttachments;
static VKAPI_ATTR void VKAPI_CALL kt_vkCmdClearAttachments(VkCommandBuffer cb, uint32_t na, const VkClearAttachment *a, uint32_t nr, const VkClearRect *r) { uint64_t kt_b0 = kgsl_mapped(); real_vkCmdClearAttachments(cb, na, a, nr, r); char info[160]; snprintf(info, sizeof info, "%u attachments", na); kt_note(16, kt_b0, info); }
static PFN_vkCmdResolveImage real_vkCmdResolveImage;
static VKAPI_ATTR void VKAPI_CALL kt_vkCmdResolveImage(VkCommandBuffer cb, VkImage si, VkImageLayout sl, VkImage di, VkImageLayout dl, uint32_t n, const VkImageResolve *r) { uint64_t kt_b0 = kgsl_mapped(); real_vkCmdResolveImage(cb, si, sl, di, dl, n, r); char info[160]; snprintf(info, sizeof info, "%ux%u", r->extent.width, r->extent.height); kt_note(17, kt_b0, info); }
static PFN_vkCmdFillBuffer real_vkCmdFillBuffer;
static VKAPI_ATTR void VKAPI_CALL kt_vkCmdFillBuffer(VkCommandBuffer cb, VkBuffer b, VkDeviceSize o, VkDeviceSize n, uint32_t v) { uint64_t kt_b0 = kgsl_mapped(); real_vkCmdFillBuffer(cb, b, o, n, v); char info[160]; snprintf(info, sizeof info, "%llu bytes", (unsigned long long)n); kt_note(18, kt_b0, info); }
static PFN_vkCmdUpdateBuffer real_vkCmdUpdateBuffer;
static VKAPI_ATTR void VKAPI_CALL kt_vkCmdUpdateBuffer(VkCommandBuffer cb, VkBuffer b, VkDeviceSize o, VkDeviceSize n, const void *d) { uint64_t kt_b0 = kgsl_mapped(); real_vkCmdUpdateBuffer(cb, b, o, n, d); char info[160]; snprintf(info, sizeof info, "%llu bytes", (unsigned long long)n); kt_note(19, kt_b0, info); }
static PFN_vkCmdCopyBuffer real_vkCmdCopyBuffer;
static VKAPI_ATTR void VKAPI_CALL kt_vkCmdCopyBuffer(VkCommandBuffer cb, VkBuffer sb, VkBuffer db, uint32_t n, const VkBufferCopy *r) { uint64_t kt_b0 = kgsl_mapped(); real_vkCmdCopyBuffer(cb, sb, db, n, r); char info[160]; snprintf(info, sizeof info, "%llu bytes", (unsigned long long)r->size); kt_note(20, kt_b0, info); }
static PFN_vkCmdBindDescriptorSets real_vkCmdBindDescriptorSets;
static VKAPI_ATTR void VKAPI_CALL kt_vkCmdBindDescriptorSets(VkCommandBuffer cb, VkPipelineBindPoint bp, VkPipelineLayout l, uint32_t first, uint32_t n, const VkDescriptorSet *sets, uint32_t nd, const uint32_t *d) { uint64_t kt_b0 = kgsl_mapped(); real_vkCmdBindDescriptorSets(cb, bp, l, first, n, sets, nd, d); char info[160]; snprintf(info, sizeof info, "%u sets", n); kt_note(21, kt_b0, info); }
static PFN_vkCmdBindPipeline real_vkCmdBindPipeline;
static VKAPI_ATTR void VKAPI_CALL kt_vkCmdBindPipeline(VkCommandBuffer cb, VkPipelineBindPoint bp, VkPipeline p) { uint64_t kt_b0 = kgsl_mapped(); real_vkCmdBindPipeline(cb, bp, p); char info[160]; snprintf(info, sizeof info, "%llx", (unsigned long long)(uintptr_t)p); kt_note(22, kt_b0, info); }
static PFN_vkCmdBindVertexBuffers real_vkCmdBindVertexBuffers;
static VKAPI_ATTR void VKAPI_CALL kt_vkCmdBindVertexBuffers(VkCommandBuffer cb, uint32_t first, uint32_t n, const VkBuffer *b, const VkDeviceSize *o) { uint64_t kt_b0 = kgsl_mapped(); real_vkCmdBindVertexBuffers(cb, first, n, b, o); char info[160]; snprintf(info, sizeof info, "%u buffers", n); kt_note(23, kt_b0, info); }
static PFN_vkCmdBindIndexBuffer real_vkCmdBindIndexBuffer;
static VKAPI_ATTR void VKAPI_CALL kt_vkCmdBindIndexBuffer(VkCommandBuffer cb, VkBuffer b, VkDeviceSize o, VkIndexType t) { uint64_t kt_b0 = kgsl_mapped(); real_vkCmdBindIndexBuffer(cb, b, o, t); char info[160]; snprintf(info, sizeof info, "%s", ""); kt_note(24, kt_b0, info); }
static PFN_vkCmdPushConstants real_vkCmdPushConstants;
static VKAPI_ATTR void VKAPI_CALL kt_vkCmdPushConstants(VkCommandBuffer cb, VkPipelineLayout l, VkShaderStageFlags st, uint32_t o, uint32_t n, const void *v) { uint64_t kt_b0 = kgsl_mapped(); real_vkCmdPushConstants(cb, l, st, o, n, v); char info[160]; snprintf(info, sizeof info, "%u bytes", n); kt_note(25, kt_b0, info); }
static PFN_vkCmdSetViewport real_vkCmdSetViewport;
static VKAPI_ATTR void VKAPI_CALL kt_vkCmdSetViewport(VkCommandBuffer cb, uint32_t f, uint32_t n, const VkViewport *v) { uint64_t kt_b0 = kgsl_mapped(); real_vkCmdSetViewport(cb, f, n, v); char info[160]; snprintf(info, sizeof info, "%u", n); kt_note(26, kt_b0, info); }
static PFN_vkCmdSetScissor real_vkCmdSetScissor;
static VKAPI_ATTR void VKAPI_CALL kt_vkCmdSetScissor(VkCommandBuffer cb, uint32_t f, uint32_t n, const VkRect2D *v) { uint64_t kt_b0 = kgsl_mapped(); real_vkCmdSetScissor(cb, f, n, v); char info[160]; snprintf(info, sizeof info, "%u", n); kt_note(27, kt_b0, info); }
static PFN_vkCreateImageView real_vkCreateImageView;
static VKAPI_ATTR VkResult VKAPI_CALL kt_vkCreateImageView(VkDevice dev, const VkImageViewCreateInfo *ci, const VkAllocationCallbacks *cb, VkImageView *v) { uint64_t kt_b0 = kgsl_mapped(); VkResult r = real_vkCreateImageView(dev, ci, cb, v); char info[160]; snprintf(info, sizeof info, "format %d", ci->format); kt_note(28, kt_b0, info); return r; }
static PFN_vkCreateFramebuffer real_vkCreateFramebuffer;
static VKAPI_ATTR VkResult VKAPI_CALL kt_vkCreateFramebuffer(VkDevice dev, const VkFramebufferCreateInfo *ci, const VkAllocationCallbacks *cb, VkFramebuffer *v) { uint64_t kt_b0 = kgsl_mapped(); VkResult r = real_vkCreateFramebuffer(dev, ci, cb, v); char info[160]; snprintf(info, sizeof info, "%ux%u x%u, %u attachments", ci->width, ci->height, ci->layers, ci->attachmentCount); kt_note(29, kt_b0, info); return r; }
static PFN_vkAllocateDescriptorSets real_vkAllocateDescriptorSets;
static VKAPI_ATTR VkResult VKAPI_CALL kt_vkAllocateDescriptorSets(VkDevice dev, const VkDescriptorSetAllocateInfo *ai, VkDescriptorSet *out) { uint64_t kt_b0 = kgsl_mapped(); VkResult r = real_vkAllocateDescriptorSets(dev, ai, out); char info[160]; snprintf(info, sizeof info, "%u", ai->descriptorSetCount); kt_note(30, kt_b0, info); return r; }
static PFN_vkUpdateDescriptorSets real_vkUpdateDescriptorSets;
static VKAPI_ATTR void VKAPI_CALL kt_vkUpdateDescriptorSets(VkDevice dev, uint32_t nw, const VkWriteDescriptorSet *w, uint32_t nc, const VkCopyDescriptorSet *c) { uint64_t kt_b0 = kgsl_mapped(); real_vkUpdateDescriptorSets(dev, nw, w, nc, c); char info[160]; snprintf(info, sizeof info, "%u writes", nw); kt_note(31, kt_b0, info); }
static PFN_vkEndCommandBuffer real_vkEndCommandBuffer;
static VKAPI_ATTR VkResult VKAPI_CALL kt_vkEndCommandBuffer(VkCommandBuffer cb) { uint64_t kt_b0 = kgsl_mapped(); VkResult r = real_vkEndCommandBuffer(cb); char info[160]; snprintf(info, sizeof info, "%s", ""); kt_note(32, kt_b0, info); return r; }
static PFN_vkBeginCommandBuffer real_vkBeginCommandBuffer;
static VKAPI_ATTR VkResult VKAPI_CALL kt_vkBeginCommandBuffer(VkCommandBuffer cb, const VkCommandBufferBeginInfo *bi) { uint64_t kt_b0 = kgsl_mapped(); VkResult r = real_vkBeginCommandBuffer(cb, bi); char info[160]; snprintf(info, sizeof info, "%s", ""); kt_note(33, kt_b0, info); return r; }
static PFN_vkQueueSubmit real_vkQueueSubmit;
static VKAPI_ATTR VkResult VKAPI_CALL kt_vkQueueSubmit(VkQueue q, uint32_t n, const VkSubmitInfo *si, VkFence f) { uint64_t kt_b0 = kgsl_mapped(); VkResult r = real_vkQueueSubmit(q, n, si, f); char info[160]; snprintf(info, sizeof info, "%u command buffers", si->commandBufferCount); kt_note(34, kt_b0, info); return r; }
static PFN_vkCreateImage real_vkCreateImage;
static VKAPI_ATTR VkResult VKAPI_CALL kt_vkCreateImage(VkDevice dev, const VkImageCreateInfo *ci, const VkAllocationCallbacks *cb, VkImage *out) { uint64_t kt_b0 = kgsl_mapped(); VkResult r = real_vkCreateImage(dev, ci, cb, out); char info[160]; snprintf(info, sizeof info, "%ux%ux%u format %d levels %u layers %u samples %u", ci->extent.width, ci->extent.height, ci->extent.depth, ci->format, ci->mipLevels, ci->arrayLayers, ci->samples); kt_note(35, kt_b0, info); return r; }
static void kt_install(void) {
    const char *e = getenv("DM_VK_KGSL_TRACE"); if (!(e && atoi(e)) || !kgsl_mapped()) return; g_kt_on = 1;
    real_vkCmdDraw = vkCmdDraw; vkCmdDraw = kt_vkCmdDraw;
    real_vkCmdDrawIndexed = vkCmdDrawIndexed; vkCmdDrawIndexed = kt_vkCmdDrawIndexed;
    real_vkCmdDrawIndirect = vkCmdDrawIndirect; vkCmdDrawIndirect = kt_vkCmdDrawIndirect;
    real_vkCmdDrawIndexedIndirect = vkCmdDrawIndexedIndirect; vkCmdDrawIndexedIndirect = kt_vkCmdDrawIndexedIndirect;
    real_vkCmdDispatch = vkCmdDispatch; vkCmdDispatch = kt_vkCmdDispatch;
    real_vkCmdBeginRenderPass = vkCmdBeginRenderPass; vkCmdBeginRenderPass = kt_vkCmdBeginRenderPass;
    real_vkCmdBeginRendering = vkCmdBeginRendering; vkCmdBeginRendering = kt_vkCmdBeginRendering;
    real_vkCmdEndRenderPass = vkCmdEndRenderPass; vkCmdEndRenderPass = kt_vkCmdEndRenderPass;
    real_vkCmdEndRendering = vkCmdEndRendering; vkCmdEndRendering = kt_vkCmdEndRendering;
    real_vkCmdPipelineBarrier = vkCmdPipelineBarrier; vkCmdPipelineBarrier = kt_vkCmdPipelineBarrier;
    real_vkCmdCopyImage = vkCmdCopyImage; vkCmdCopyImage = kt_vkCmdCopyImage;
    real_vkCmdCopyBufferToImage = vkCmdCopyBufferToImage; vkCmdCopyBufferToImage = kt_vkCmdCopyBufferToImage;
    real_vkCmdCopyImageToBuffer = vkCmdCopyImageToBuffer; vkCmdCopyImageToBuffer = kt_vkCmdCopyImageToBuffer;
    real_vkCmdBlitImage = vkCmdBlitImage; vkCmdBlitImage = kt_vkCmdBlitImage;
    real_vkCmdClearColorImage = vkCmdClearColorImage; vkCmdClearColorImage = kt_vkCmdClearColorImage;
    real_vkCmdClearDepthStencilImage = vkCmdClearDepthStencilImage; vkCmdClearDepthStencilImage = kt_vkCmdClearDepthStencilImage;
    real_vkCmdClearAttachments = vkCmdClearAttachments; vkCmdClearAttachments = kt_vkCmdClearAttachments;
    real_vkCmdResolveImage = vkCmdResolveImage; vkCmdResolveImage = kt_vkCmdResolveImage;
    real_vkCmdFillBuffer = vkCmdFillBuffer; vkCmdFillBuffer = kt_vkCmdFillBuffer;
    real_vkCmdUpdateBuffer = vkCmdUpdateBuffer; vkCmdUpdateBuffer = kt_vkCmdUpdateBuffer;
    real_vkCmdCopyBuffer = vkCmdCopyBuffer; vkCmdCopyBuffer = kt_vkCmdCopyBuffer;
    real_vkCmdBindDescriptorSets = vkCmdBindDescriptorSets; vkCmdBindDescriptorSets = kt_vkCmdBindDescriptorSets;
    real_vkCmdBindPipeline = vkCmdBindPipeline; vkCmdBindPipeline = kt_vkCmdBindPipeline;
    real_vkCmdBindVertexBuffers = vkCmdBindVertexBuffers; vkCmdBindVertexBuffers = kt_vkCmdBindVertexBuffers;
    real_vkCmdBindIndexBuffer = vkCmdBindIndexBuffer; vkCmdBindIndexBuffer = kt_vkCmdBindIndexBuffer;
    real_vkCmdPushConstants = vkCmdPushConstants; vkCmdPushConstants = kt_vkCmdPushConstants;
    real_vkCmdSetViewport = vkCmdSetViewport; vkCmdSetViewport = kt_vkCmdSetViewport;
    real_vkCmdSetScissor = vkCmdSetScissor; vkCmdSetScissor = kt_vkCmdSetScissor;
    real_vkCreateImageView = vkCreateImageView; vkCreateImageView = kt_vkCreateImageView;
    real_vkCreateFramebuffer = vkCreateFramebuffer; vkCreateFramebuffer = kt_vkCreateFramebuffer;
    real_vkAllocateDescriptorSets = vkAllocateDescriptorSets; vkAllocateDescriptorSets = kt_vkAllocateDescriptorSets;
    real_vkUpdateDescriptorSets = vkUpdateDescriptorSets; vkUpdateDescriptorSets = kt_vkUpdateDescriptorSets;
    real_vkEndCommandBuffer = vkEndCommandBuffer; vkEndCommandBuffer = kt_vkEndCommandBuffer;
    real_vkBeginCommandBuffer = vkBeginCommandBuffer; vkBeginCommandBuffer = kt_vkBeginCommandBuffer;
    real_vkQueueSubmit = vkQueueSubmit; vkQueueSubmit = kt_vkQueueSubmit;
    real_vkCreateImage = vkCreateImage; vkCreateImage = kt_vkCreateImage;
    VKLOG("GPU memory trace on (DM_VK_KGSL_TRACE): the driver's own allocations charged to the recorded commands");
}
static VKAPI_ATTR void VKAPI_CALL acct_free(VkDevice dev, VkDeviceMemory m, const VkAllocationCallbacks *cb) {
    if (m) { pthread_mutex_lock(&g_macct_mu); uint64_t key = (uint64_t)(uintptr_t)m; macct_t *e = nvn_hm_get(&g_macct_map, &key);
        if (e) { nvn_hm_del(&g_macct_map, &key); g_macct[e->tag].live -= e->size; g_macct[e->tag].n--; g_macct_total -= e->size; free(e); }
        pthread_mutex_unlock(&g_macct_mu); }
    real_vkFreeMemory(dev, m, cb);
}
void vk_mem_acct_report(const char *why) { pthread_mutex_lock(&g_macct_mu); macct_table(why); pthread_mutex_unlock(&g_macct_mu); }
static void mem_acct_install(void) {
    const char *e = getenv("DM_VK_MEM_ACCT"); if (e && !atoi(e)) return;
    Dl_info di; if (dladdr((void *)mem_acct_install, &di)) g_lib_base = di.dli_fbase;
    real_vkAllocateMemory = vkAllocateMemory; real_vkFreeMemory = vkFreeMemory; real_vkBindImageMemory = vkBindImageMemory; real_vkBindBufferMemory = vkBindBufferMemory;
    vkAllocateMemory = acct_alloc; vkFreeMemory = acct_free; vkBindImageMemory = acct_bind_image; vkBindBufferMemory = acct_bind_buffer;
    real_vkCreateGraphicsPipelines = vkCreateGraphicsPipelines; real_vkCreateComputePipelines = vkCreateComputePipelines; vkCreateGraphicsPipelines = acct_gpipe; vkCreateComputePipelines = acct_cpipe;
    VKLOG("device memory accounting on (DM_VK_MEM_ACCT=0: off): a line at each 512 MB more per tag, each allocation of 64 MB or more, the table at each 512 MB more in all");
}
static int drv_has_gpu(PFN_vkGetInstanceProcAddr g) {
    PFN_vkCreateInstance ci = (PFN_vkCreateInstance)g(NULL, "vkCreateInstance"); VkInstance in; int ok = 0;
    VkApplicationInfo app = {VK_STRUCTURE_TYPE_APPLICATION_INFO, NULL, "dmhost", 1, "nxnative", 1, VK_API_VERSION_1_3};
    VkInstanceCreateInfo ici = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, NULL, 0, &app, 0, NULL, 0, NULL};
    if (!ci || ci(&ici, NULL, &in)) return 0;
    PFN_vkEnumeratePhysicalDevices ep = (PFN_vkEnumeratePhysicalDevices)g(in, "vkEnumeratePhysicalDevices");
    PFN_vkGetPhysicalDeviceProperties gp = (PFN_vkGetPhysicalDeviceProperties)g(in, "vkGetPhysicalDeviceProperties");
    PFN_vkDestroyInstance di = (PFN_vkDestroyInstance)g(in, "vkDestroyInstance");
    VkPhysicalDevice pd[8]; uint32_t n = 8;
    if (ep && gp && ep(in, &n, pd) >= 0) for (uint32_t i = 0; i < n && !ok; i++) { VkPhysicalDeviceProperties p; gp(pd[i], &p); ok = p.apiVersion >= VK_API_VERSION_1_3;
        VKLOG("DM_VK_DRIVER: %s, Vulkan %u.%u, driver version 0x%x", p.deviceName, VK_API_VERSION_MAJOR(p.apiVersion), VK_API_VERSION_MINOR(p.apiVersion), p.driverVersion); }
    if (di) di(in, NULL);
    return ok;
}

void vk_init(void) {
    static int done; if (done) return; done = 1;
    { const char *e = getenv("DM_VK_DYN_SSBO"); V.dyn_ssbo = e && atoi(e); }
    // DM_VK_DRIVER=<library>: a Vulkan driver the APK carries (Mesa's Turnip for Adreno; porter --vulkan-driver), loaded through libadrenotools (BSD-2-Clause), which the
    // APK carries beside it with its hook libraries: a second copy of the system's loader in a linker namespace of its own, whose driver is this library. The loader keeps
    // making the surface and the swapchain, which Turnip on Android has none of (the platform loader makes them over VK_ANDROID_native_buffer). TOTKDroid, a port of Tears
    // of the Kingdom 1.4.3 that also replays NVN on Vulkan, measured its Great Sky Island forest at GPU 16-17 ms a frame on Turnip on an Adreno 830 (docs/FPS-2026-10-31.md),
    // against 22-27 ms here on Qualcomm's driver. A driver that does not load, or offers no Vulkan 1.3 GPU, leaves the system's
    const char *drv = getenv("DM_VK_DRIVER"); void *lib = NULL;
    if (drv && *drv) {
        Dl_info di; char dir[512] = "", dirs[520] = "";   // the APK's extracted libraries (mkapk.py: extractNativeLibs), where adrenotools finds its hooks and the driver
        if (dladdr((void *)vk_init, &di) && di.dli_fname && strrchr(di.dli_fname, '/')) {
            snprintf(dir, sizeof dir, "%.*s", (int)(strrchr(di.dli_fname, '/') - di.dli_fname), di.dli_fname); snprintf(dirs, sizeof dirs, "%s/", dir); }
        void *at = dlopen("libadrenotools.so", RTLD_NOW | RTLD_LOCAL);
        void *(*open_vk)(int, int, const char *, const char *, const char *, const char *, const char *, void **) =
            at ? (void *(*)(int, int, const char *, const char *, const char *, const char *, const char *, void **))dlsym(at, "adrenotools_open_libvulkan") : NULL;
        if (!open_vk || !*dir) VKLOG("[RED] DM_VK_DRIVER %s: no libadrenotools in the APK (%s); the system's driver instead", drv, at ? "no adrenotools_open_libvulkan" : dlerror());
        else if (!(lib = open_vk(RTLD_NOW | RTLD_LOCAL, 1 /* ADRENOTOOLS_DRIVER_CUSTOM */, NULL, dir, dirs, drv, NULL, NULL)))
            VKLOG("[RED] DM_VK_DRIVER %s: adrenotools could not load it from %s; the system's driver instead", drv, dir);
        else if (!(p_vkGetInstanceProcAddr = (PFN_vkGetInstanceProcAddr)dlsym(lib, "vkGetInstanceProcAddr")) || !drv_has_gpu(p_vkGetInstanceProcAddr)) {
            VKLOG("[RED] DM_VK_DRIVER %s: no Vulkan 1.3 GPU through it; the system's driver instead", drv); lib = NULL; }   // (left loaded: a driver does not always unload cleanly)
        else VKLOG("Vulkan driver %s through adrenotools, not the system's (%s)", drv, dir);
    }
    if (!lib) {
        if (!(lib = dlopen("libvulkan.so", RTLD_NOW | RTLD_LOCAL))) die("[vk] libvulkan.so: %s", dlerror());
        p_vkGetInstanceProcAddr = (PFN_vkGetInstanceProcAddr)dlsym(lib, "vkGetInstanceProcAddr");
    }
    vkCreateInstance = (PFN_vkCreateInstance)p_vkGetInstanceProcAddr(NULL, "vkCreateInstance");
    const char *iext[] = {"VK_KHR_surface", "VK_KHR_android_surface", "VK_EXT_debug_utils"};
    VkApplicationInfo app = {VK_STRUCTURE_TYPE_APPLICATION_INFO, NULL, "dmhost", 1, "nxnative", 1, VK_API_VERSION_1_3};
    // DM_VK_VALIDATE=1: Khronos's validation layer, which the APK carries only when built with it (mkapk.py --extra-so); its messages go to the log
    int validate = getenv("DM_VK_VALIDATE") && atoi(getenv("DM_VK_VALIDATE"));
    const char *layers[] = {"VK_LAYER_KHRONOS_validation"};
    VkInstanceCreateInfo ici = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, NULL, 0, &app, validate ? 1u : 0u, layers, validate ? 3u : 2u, iext};
    VkResult ir = vkCreateInstance(&ici, NULL, &V.inst);
    if (ir && validate) { VKLOG("[RED] validation layer not available (%d): running without it", ir); validate = 0; ici.enabledLayerCount = 0; ici.enabledExtensionCount = 2; ir = vkCreateInstance(&ici, NULL, &V.inst); }
    VKCHK(ir);
    if (validate) {
        PFN_vkCreateDebugUtilsMessengerEXT mk = (PFN_vkCreateDebugUtilsMessengerEXT)p_vkGetInstanceProcAddr(V.inst, "vkCreateDebugUtilsMessengerEXT");
        VkDebugUtilsMessengerCreateInfoEXT mi = {VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT, NULL, 0,
            VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT,
            VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT, vk_validation_msg, NULL};
        VkDebugUtilsMessengerEXT m; if (mk && !mk(V.inst, &mi, NULL, &m)) VKLOG("validation layer on");
    }
#define VK_LOADI(f) f = (PFN_##f)p_vkGetInstanceProcAddr(V.inst, #f);
    VK_FNS(VK_LOADI)
    uint32_t n = 1; VKCHK(vkEnumeratePhysicalDevices(V.inst, &n, &V.pd)); if (!n) die("[vk] no GPU");
    vkGetPhysicalDeviceProperties(V.pd, &V.props); vkGetPhysicalDeviceMemoryProperties(V.pd, &V.mem);
    // the device names itself before any check below can stop: a Moto g 5G 2024's log (Adreno 619) ended at the timeline check with no GPU or driver named.
    // driver: as Qualcomm encodes it (VK_MAKE_VERSION: 512.530.0), with the raw value for other vendors
    { uint32_t d = V.props.driverVersion;
      VKLOG("GPU %s (vendor %#x, device %#x), Vulkan %u.%u.%u, driver %u.%u.%u (%#x)", V.props.deviceName, V.props.vendorID, V.props.deviceID, VK_API_VERSION_MAJOR(V.props.apiVersion),
            VK_API_VERSION_MINOR(V.props.apiVersion), VK_API_VERSION_PATCH(V.props.apiVersion), d >> 22, d >> 12 & 0x3ff, d & 0xfff, d); }
    uint32_t qn = 0; vkGetPhysicalDeviceQueueFamilyProperties(V.pd, &qn, NULL);
    VkQueueFamilyProperties qp[16]; if (qn > 16) qn = 16; vkGetPhysicalDeviceQueueFamilyProperties(V.pd, &qn, qp);
    V.qf = 0; while (V.qf < qn && (qp[V.qf].queueFlags & (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT)) != (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT)) V.qf++;
    if (V.qf == qn) die("[vk] no graphics + compute queue");
    g_gpu_timestamp_bits = qp[V.qf].timestampValidBits;
    uint32_t ne = 0; vkEnumerateDeviceExtensionProperties(V.pd, NULL, &ne, NULL);
    VkExtensionProperties *ex = calloc(ne, sizeof *ex); vkEnumerateDeviceExtensionProperties(V.pd, NULL, &ne, ex);
    const char *want[] = {"VK_KHR_swapchain", "VK_ANDROID_external_memory_android_hardware_buffer", "VK_EXT_queue_family_foreign", "VK_EXT_custom_border_color",
                          "VK_EXT_robustness2", "VK_EXT_depth_clip_control", "VK_EXT_provoking_vertex", "VK_EXT_index_type_uint8", "VK_EXT_transform_feedback",
                          "VK_GOOGLE_display_timing", "VK_KHR_fragment_shader_barycentric"};   // barycentric: Ampere LDTRAM, a fragment input at each vertex (Welcome Tour's edge effects)
    const char *on[64]; uint32_t non = 0;
    for (unsigned i = 0; i < sizeof want / sizeof *want; i++) if (has_ext(ex, ne, want[i])) on[non++] = want[i]; else VKLOG("device has no %s", want[i]);
    const char *clock_extension = has_ext(ex, ne, "VK_KHR_calibrated_timestamps") ? "VK_KHR_calibrated_timestamps" :
        has_ext(ex, ne, "VK_EXT_calibrated_timestamps") ? "VK_EXT_calibrated_timestamps" : NULL;
    if (clock_extension) on[non++] = clock_extension;
    // Catmull-Rom is EXT cubic's default; QCOM selectable weights are not needed.
    V.has_filter_cubic = has_ext(ex, ne, "VK_EXT_filter_cubic");
    if (V.has_filter_cubic) on[non++] = "VK_EXT_filter_cubic";
    // A Vulkan 1.1 or 1.2 device (Legion Y700 2023, Adreno 730: its stock driver reports 1.1) has what the replay uses from 1.2 / 1.3 as extensions: they are enabled
    // when present, so the features chained below and the functions loaded by their KHR names are valid there. A 1.3 device needs none of them
    int v12 = V.props.apiVersion >= VK_API_VERSION_1_2, v13 = V.props.apiVersion >= VK_API_VERSION_1_3; uint32_t npromo = 0;
    static const char *promo12[] = {"VK_KHR_create_renderpass2", "VK_KHR_imageless_framebuffer", "VK_KHR_image_format_list", "VK_KHR_depth_stencil_resolve", "VK_KHR_timeline_semaphore",
        "VK_KHR_shader_float16_int8", "VK_KHR_8bit_storage", "VK_EXT_descriptor_indexing", "VK_EXT_scalar_block_layout", "VK_KHR_uniform_buffer_standard_layout",
        "VK_KHR_shader_float_controls", "VK_KHR_spirv_1_4", "VK_KHR_vulkan_memory_model", "VK_KHR_separate_depth_stencil_layouts", "VK_EXT_host_query_reset",
        "VK_KHR_sampler_mirror_clamp_to_edge", "VK_KHR_draw_indirect_count", "VK_KHR_buffer_device_address", "VK_EXT_shader_viewport_index_layer", "VK_EXT_sampler_filter_minmax",
        "VK_KHR_shader_subgroup_extended_types", "VK_KHR_shader_atomic_int64", "VK_KHR_driver_properties"};
    static const char *promo13[] = {"VK_EXT_image_robustness", "VK_KHR_dynamic_rendering", "VK_KHR_synchronization2", "VK_EXT_shader_demote_to_helper_invocation",
        "VK_EXT_subgroup_size_control", "VK_KHR_copy_commands2", "VK_KHR_maintenance4", "VK_KHR_format_feature_flags2", "VK_EXT_extended_dynamic_state",
        "VK_KHR_zero_initialize_workgroup_memory", "VK_KHR_shader_terminate_invocation", "VK_EXT_inline_uniform_block", "VK_EXT_pipeline_creation_cache_control",
        "VK_EXT_texel_buffer_alignment", "VK_EXT_4444_formats", "VK_KHR_shader_integer_dot_product", "VK_EXT_private_data"};
    for (unsigned i = 0; !v12 && i < sizeof promo12 / sizeof *promo12 && non < 64; i++) if (has_ext(ex, ne, promo12[i])) { on[non++] = promo12[i]; npromo++; }
    for (unsigned i = 0; !v13 && i < sizeof promo13 / sizeof *promo13 && non < 64; i++) if (has_ext(ex, ne, promo13[i])) { on[non++] = promo13[i]; npromo++; }
    // features: what the replay uses when the device has it. VkPhysicalDeviceVulkan11/12/13Features are valid only on a 1.2 / 1.3 device (a 1.1 driver ignores
    // them and nothing would be enabled); below that, the same features come through Vulkan 1.1's own structs and those of the extensions enabled above (xf)
    static const struct { const char *ext; VkStructureType s; } XF[] = {   // ext NULL: Vulkan 1.1 core, VkPhysicalDeviceVulkan11Features's members
        {NULL, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES}, {NULL, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MULTIVIEW_FEATURES},
        {NULL, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VARIABLE_POINTERS_FEATURES}, {NULL, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROTECTED_MEMORY_FEATURES},
        {NULL, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SAMPLER_YCBCR_CONVERSION_FEATURES}, {NULL, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_DRAW_PARAMETERS_FEATURES},
        {"VK_KHR_imageless_framebuffer", VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGELESS_FRAMEBUFFER_FEATURES}, {"VK_KHR_timeline_semaphore", VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES},
        {"VK_KHR_shader_float16_int8", VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES}, {"VK_KHR_8bit_storage", VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES},
        {"VK_EXT_descriptor_indexing", VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES}, {"VK_EXT_scalar_block_layout", VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SCALAR_BLOCK_LAYOUT_FEATURES},
        {"VK_KHR_uniform_buffer_standard_layout", VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_UNIFORM_BUFFER_STANDARD_LAYOUT_FEATURES}, {"VK_KHR_vulkan_memory_model", VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_MEMORY_MODEL_FEATURES},
        {"VK_KHR_separate_depth_stencil_layouts", VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SEPARATE_DEPTH_STENCIL_LAYOUTS_FEATURES}, {"VK_EXT_host_query_reset", VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_HOST_QUERY_RESET_FEATURES},
        {"VK_KHR_buffer_device_address", VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES}, {"VK_KHR_shader_subgroup_extended_types", VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_SUBGROUP_EXTENDED_TYPES_FEATURES},
        {"VK_KHR_shader_atomic_int64", VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_INT64_FEATURES},   // (mirror clamp, indirect count, minmax, viewport layer: the extension is the feature)
        {"VK_KHR_dynamic_rendering", VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES}, {"VK_KHR_synchronization2", VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES},
        {"VK_EXT_shader_demote_to_helper_invocation", VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_DEMOTE_TO_HELPER_INVOCATION_FEATURES}, {"VK_EXT_subgroup_size_control", VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES},
        {"VK_KHR_maintenance4", VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_4_FEATURES}, {"VK_KHR_zero_initialize_workgroup_memory", VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ZERO_INITIALIZE_WORKGROUP_MEMORY_FEATURES},
        {"VK_KHR_shader_terminate_invocation", VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_TERMINATE_INVOCATION_FEATURES}, {"VK_EXT_inline_uniform_block", VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_INLINE_UNIFORM_BLOCK_FEATURES},
        {"VK_EXT_pipeline_creation_cache_control", VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_CREATION_CACHE_CONTROL_FEATURES}, {"VK_KHR_shader_integer_dot_product", VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES},
        {"VK_EXT_private_data", VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRIVATE_DATA_FEATURES}};   // (VK_EXT_image_robustness: robustImageAccess stays off, as on 1.3)
    uint32_t nxf = sizeof XF / sizeof *XF; featx_t xf[sizeof XF / sizeof *XF]; memset(xf, 0, sizeof xf); VkBaseOutStructure *xs = NULL;
    for (int i = (int)nxf - 1; i >= 0; i--) if (XF[i].ext ? is_on(on, non, XF[i].ext) : !v12) { xf[i].h.sType = XF[i].s; xf[i].h.pNext = xs; xs = &xf[i].h; }
    VkPhysicalDeviceVulkan13Features f13 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES, xs};
    VkPhysicalDeviceVulkan12Features f12 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, v13 ? (void *)&f13 : xs};
    VkPhysicalDeviceVulkan11Features f11 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES, &f12};
    VkPhysicalDeviceFeatures2 ff = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, v12 ? (void *)&f11 : xs};
    vkGetPhysicalDeviceFeatures2(V.pd, &ff);
    if (v12) g_uab = f12.descriptorBindingSampledImageUpdateAfterBind && f12.descriptorBindingSampledImageUpdateAfterBind && f12.descriptorBindingPartiallyBound ? 1 + (f12.descriptorBindingStorageImageUpdateAfterBind && f12.descriptorBindingUniformTexelBufferUpdateAfterBind && f12.descriptorBindingStorageTexelBufferUpdateAfterBind && f12.descriptorBindingStorageBufferUpdateAfterBind) : 0;   // (enabled below with the rest of the queried chain)
    // without timeline semaphores (Vulkan 1.1 drivers: the Adreno 618 / 619 of 2021, 512.530) each submission signals a fence of a ring instead (slots_init).
    // DM_VK_X_FENCES=1 (start-up): the fences on a device that has them; with DM_VK_X_BC_CPU=1 below, a device of today stands in for those drivers
    { const char *e = getenv("DM_VK_X_FENCES");
      V.has_timeline = (v12 ? f12.timelineSemaphore : featx(xf, nxf, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES)[0]) && !(e && atoi(e)); }
    if (!V.has_timeline) VKLOG("no timeline semaphores (Vulkan 1.2 or VK_KHR_timeline_semaphore): submissions signal fences");
    // dynamic rendering is 1.3 or its extension with the feature on; a loaded pointer does not tell, since the loader hands out a trampoline for core functions the driver lacks
    V.has_dynrender = v13 ? f13.dynamicRendering : featx(xf, nxf, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES)[0];
    // imageless framebuffers need render pass 2 (gfx_vk_exec.c rp_get): core in 1.2, else its extension
    V.has_draw_count = v12 ? f12.drawIndirectCount : is_on(on, non, "VK_KHR_draw_indirect_count");
    V.has_imageless = v12 ? f12.imagelessFramebuffer : featx(xf, nxf, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGELESS_FRAMEBUFFER_FEATURES)[0] && is_on(on, non, "VK_KHR_create_renderpass2");
    if (!V.has_imageless && !V.has_dynrender) die("[vk] the device has neither dynamic rendering nor imageless framebuffers with render pass 2: passes cannot be begun");
    // buffer device addresses: the shaders' raw global memory (PhysicalStorageBuffer pointers through the pool table, vk_gmem_map) needs the pools' buffers to have one. DM_VK_NO_BDA=1 (start-up): off
    { const char *e = getenv("DM_VK_NO_BDA");
      V.has_bda = (v12 ? f12.bufferDeviceAddress : featx(xf, nxf, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES)[0]) && !(e && atoi(e));
      if (!V.has_bda) VKLOG("[RED] no buffer device addresses (Vulkan 1.2 bufferDeviceAddress or VK_KHR_buffer_device_address): shaders that access memory through loaded addresses reach nothing"); }
    if (!v13) VKLOG("Vulkan %u.%u device: %u extensions enabled for the features promoted to 1.2 / 1.3; dynamic rendering %s", VK_API_VERSION_MAJOR(V.props.apiVersion),
                    VK_API_VERSION_MINOR(V.props.apiVersion), npromo, V.has_dynrender ? "yes" : "no (render pass objects only)");
    VkPhysicalDeviceFragmentShaderBarycentricFeaturesKHR fbc = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADER_BARYCENTRIC_FEATURES_KHR};
    VkPhysicalDeviceCustomBorderColorFeaturesEXT fcb = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CUSTOM_BORDER_COLOR_FEATURES_EXT, &fbc};
    VkPhysicalDeviceRobustness2FeaturesEXT fr2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ROBUSTNESS_2_FEATURES_EXT, &fcb};
    VkPhysicalDeviceDepthClipControlFeaturesEXT fdc = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DEPTH_CLIP_CONTROL_FEATURES_EXT, &fr2};
    VkPhysicalDeviceIndexTypeUint8FeaturesEXT fu8 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_INDEX_TYPE_UINT8_FEATURES_EXT, &fdc};
    VkPhysicalDeviceTransformFeedbackFeaturesEXT fxf = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TRANSFORM_FEEDBACK_FEATURES_EXT, &fu8};
    VkPhysicalDeviceFeatures2 fx = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &fxf};
    vkGetPhysicalDeviceFeatures2(V.pd, &fx);
    // transform feedback (NVN BeginTransformFeedback; gfx_vk_exec.c xfb_*): Xenoblade Chronicles 3 captures particle state with it
    V.has_xfb = fxf.transformFeedback && has_ext(ex, ne, "VK_EXT_transform_feedback"); fxf.geometryStreams = 0;
    fr2.robustBufferAccess2 = 0; fr2.robustImageAccess2 = 0;   // nullDescriptor only: unbound slots read 0 without a dummy
    V.has_custom_border = fcb.customBorderColors && has_ext(ex, ne, "VK_EXT_custom_border_color");
    V.has_robust2 = fr2.nullDescriptor && has_ext(ex, ne, "VK_EXT_robustness2");
    V.has_depth_clip_ctl = fdc.depthClipControl && has_ext(ex, ne, "VK_EXT_depth_clip_control");
    // per-vertex fragment inputs (SPIR-V PerVertexKHR): the translator's LDTRAM (Ampere, NVN2) output needs them; the stock Adreno driver has the extension, Turnip has not
    V.has_barycentric = fbc.fragmentShaderBarycentric && has_ext(ex, ne, "VK_KHR_fragment_shader_barycentric");
    if (!V.has_barycentric) VKLOG("no VK_KHR_fragment_shader_barycentric: pipelines of shaders reading a fragment input per vertex (Ampere LDTRAM) will fail to build");
    V.has_dtiming = has_ext(ex, ne, "VK_GOOGLE_display_timing");   // frame pacing on the display (gfx_vk_present.c pace_*)
    V.has_bc = ff.features.textureCompressionBC; V.has_astc = ff.features.textureCompressionASTC_LDR; V.has_gs = ff.features.geometryShader;
    { const char *e = getenv("DM_VK_X_BC_CPU"); if (e && atoi(e)) V.has_bc = 0; }   // (start-up) BC textures decoded on the CPU as on a device without them (gfx_vk_tex.c bc_decode)
    V.has_occ_precise = ff.features.occlusionQueryPrecise;
    V.max_aniso = V.props.limits.maxSamplerAnisotropy;
    // subgroups: where the shaders' votes, ballots and shuffles may run; a module whose stage lacks them runs as subgroups of one invocation (gfx_vk_shader.c, spv_one_lane).
    // Turnip lists graphics stages and quad operations only on a GPU with getfiberid (tu_device.cc): the Adreno 7xx have it, the Adreno 650 (a6xx gen3) not (compute alone).
    // DM_VK_SUBGROUP_PIN=<lanes> (start-up): game pipelines' fragment and compute stages run at that subgroup size (VkPipelineShaderStageRequiredSubgroupSizeCreateInfo)
    // where requiredSubgroupSizeStages allows it; by default the driver picks per pipeline (Turnip on Adreno 6xx / 7xx: 64 or 128, only those two stages pinnable)
    { VkPhysicalDeviceSubgroupSizeControlProperties sc = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES};
      VkPhysicalDeviceSubgroupProperties sg = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES, v13 || has_ext(ex, ne, "VK_EXT_subgroup_size_control") ? (void *)&sc : NULL};
      VkPhysicalDeviceProperties2 p2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &sg}; vkGetPhysicalDeviceProperties2(V.pd, &p2);
      int ctl = v13 ? f13.subgroupSizeControl : featx(xf, nxf, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES)[0];   // (enabled below with the rest of the chain)
      V.sg_size = sg.subgroupSize; V.sg_stages = sg.supportedStages; V.sg_ops = sg.supportedOperations; V.sg_quad_all = sg.quadOperationsInAllStages;
      V.sg_min = sc.minSubgroupSize ? sc.minSubgroupSize : sg.subgroupSize; V.sg_max = sc.maxSubgroupSize ? sc.maxSubgroupSize : sg.subgroupSize; V.sg_maxwgs = sc.maxComputeWorkgroupSubgroups;
      uint32_t pinnable = ctl ? sc.requiredSubgroupSizeStages : 0, st = pinnable & (VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_COMPUTE_BIT);
      VKLOG("subgroups: size %u (%u-%u), stages %#x, ops %#x, quad all stages %s, pinnable %#x", V.sg_size, V.sg_min, V.sg_max, V.sg_stages, V.sg_ops, V.sg_quad_all ? "yes" : "no", pinnable);
      const char *e = getenv("DM_VK_SUBGROUP_PIN"); uint32_t pin = e ? (uint32_t)atoi(e) : 0;
      if (pin && (!st || (pin & (pin - 1)) || pin < V.sg_min || pin > V.sg_max)) VKLOG("[RED] DM_VK_SUBGROUP_PIN=%u: the device cannot pin fragment or compute shaders to it (sizes %u-%u, pinnable stages %#x): the driver picks", pin, V.sg_min, V.sg_max, pinnable);
      else if (pin) { V.sg_pin_stages = st; V.sg_pin = (VkPipelineShaderStageRequiredSubgroupSizeCreateInfo){VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO, NULL, pin};
                      VKLOG("subgroups: stages %#x pinned to %u lanes (DM_VK_SUBGROUP_PIN)", st, pin); } }
    // enable: everything the shaders may use that the device has
    VkPhysicalDeviceFeatures en = ff.features; en.robustBufferAccess = 0; en.shaderFloat64 = 0;
    // DM_VK_ROBUST=1 (start-up, A/B): robust buffer access on, where the device has it: a shader's access past its bound range does not fault (writes stay in the range or
    // are dropped, reads give values from the range or zeros). Off by default (it costs bounds checks on every access); for telling an out-of-range address from other
    // GPU faults (Welcome Tour's no-attachment draws, gfx_vk_exec.c noatt_on)
    { const char *e = getenv("DM_VK_ROBUST"); if (e && atoi(e)) { en.robustBufferAccess = ff.features.robustBufferAccess; VKLOG("%srobust buffer access %s (DM_VK_ROBUST)", en.robustBufferAccess ? "" : "[RED] ", en.robustBufferAccess ? "on" : "not offered by the device"); } }
    f12.bufferDeviceAddressCaptureReplay = 0; f12.bufferDeviceAddressMultiDevice = 0;
    featx(xf, nxf, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES)[1] = featx(xf, nxf, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES)[2] = 0;   // (its extension's: the same two)
    f13.robustImageAccess = 0;
    fbc.pNext = NULL; fcb.pNext = V.has_barycentric ? (void *)&fbc : NULL; fr2.pNext = V.has_custom_border ? (void *)&fcb : (V.has_barycentric ? (void *)&fbc : NULL); fdc.pNext = V.has_robust2 ? (void *)&fr2 : fr2.pNext;
    fu8.pNext = V.has_depth_clip_ctl ? (void *)&fdc : fdc.pNext;
    fxf.pNext = &fu8;
    VkBaseOutStructure *t = ff.pNext; while (t->pNext) t = t->pNext;   // the queried chain (core / promoted features), then the extensions'
    t->pNext = V.has_xfb ? (VkBaseOutStructure *)&fxf : (VkBaseOutStructure *)&fu8;
    VkPhysicalDeviceFeatures2 enf = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, ff.pNext, en};
    float prio = 1;
    VkDeviceQueueCreateInfo qci = {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, NULL, 0, V.qf, 1, &prio};
    VkDeviceCreateInfo dci = {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, &enf, 0, 1, &qci, 0, NULL, non, on, NULL};
    VKCHK(vkCreateDevice(V.pd, &dci, NULL, &V.dev));
    vkGetDeviceProcAddr = (PFN_vkGetDeviceProcAddr)p_vkGetInstanceProcAddr(V.inst, "vkGetDeviceProcAddr");
#define VK_LOADD(f) f = (PFN_##f)vkGetDeviceProcAddr(V.dev, #f); if (!f) f = (PFN_##f)vkGetDeviceProcAddr(V.dev, #f "KHR"); if (!f) f = (PFN_##f)p_vkGetInstanceProcAddr(V.inst, #f);   // a promoted function by its extension's name (Vulkan 1.1 / 1.2 devices)
    VK_DEV_FNS(VK_LOADD)
    mem_acct_install(); kt_install(); uab_install();   // device memory accounting (DM_VK_MEM_ACCT): every vkAllocateMemory / vkFreeMemory through the function table goes through it
    gpu_clock_init(clock_extension);
    // A driver can report the timeline-semaphore feature (and Vulkan 1.3) and still export no vkWaitSemaphores: the emulator's ranchu / gfxstream driver does
    // (both the device and the instance dispatcher return null for it and for vkGetSemaphoreCounterValue), and the first wait then called address 0
    // (SIGSEGV pc=0, lr=wait_fence+0x70, thread 'dm-vk-complete', South Park: The Stick of Truth). The feature query is not proof the entry points load:
    // without them the fence ring is used for both the submit and the wait, which V.has_timeline decides.
    if (V.has_timeline && (!vkWaitSemaphores || !vkGetSemaphoreCounterValue)) {
        VKLOG("[vk] timeline semaphores reported but vkWaitSemaphores / vkGetSemaphoreCounterValue did not load: submissions signal fences");
        V.has_timeline = 0;
    }
    { void vk_dq_init(void); vk_dq_init(); }   // DM_VK_MT_REPLAY: the vkCmd* calls through the recorder thread (gfx_vk_dq.c)
    vkGetDeviceQueue(V.dev, V.qf, 0, &V.q);
    VkPipelineCacheCreateInfo pci = {VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO};
    char path[1024]; const char *root = getenv("DM_ROOT"); void *data = NULL; size_t dn = 0;
    if (root && !getenv("DM_VK_NOPCACHE")) { snprintf(path, sizeof path, "%s/vk_pipeline_cache.bin", root); FILE *f = fopen(path, "rb");   // DM_VK_NOPCACHE=1: start empty
        if (f) { fseek(f, 0, SEEK_END); dn = (size_t)ftell(f); fseek(f, 0, SEEK_SET); data = malloc(dn); if (fread(data, 1, dn, f) != dn) dn = 0; fclose(f); } }
    pci.initialDataSize = dn; pci.pInitialData = data;
    if (vkCreatePipelineCache(V.dev, &pci, NULL, &V.pcache)) { pci.initialDataSize = 0; pci.pInitialData = NULL; VKCHK(vkCreatePipelineCache(V.dev, &pci, NULL, &V.pcache)); }
    free(data); free(ex);
    VKLOG("%s, Vulkan %u.%u, driver %#x; timeline semaphores %s, BC %s, ASTC %s, geometry shaders %s, nullDescriptor %s, custom border colours %s, depth clip control %s, transform feedback %s; pipeline cache %zu bytes",
          V.props.deviceName, VK_API_VERSION_MAJOR(V.props.apiVersion), VK_API_VERSION_MINOR(V.props.apiVersion), V.props.driverVersion, V.has_timeline ? "yes" : "no (fences)",
          V.has_bc ? "yes" : "no (decoded on the CPU)", V.has_astc ? "yes" : "no", V.has_gs ? "yes" : "no", V.has_robust2 ? "yes" : "no", V.has_custom_border ? "yes" : "no", V.has_depth_clip_ctl ? "yes" : "no", V.has_xfb ? "yes" : "no", dn);
    if (mem_diag_on()) VKLOG("[memdiag] enabled storage-image features: extended-formats %s, read-without-format %s, write-without-format %s",
        en.shaderStorageImageExtendedFormats ? "yes" : "no", en.shaderStorageImageReadWithoutFormat ? "yes" : "no", en.shaderStorageImageWriteWithoutFormat ? "yes" : "no");
    slots_init();
    completion_start();
    { void crash_handlers_check(const char *when); crash_handlers_check("after the Vulkan device was made"); }   // (again at present 600, gfx_vk_present.c: the shader compiler loads with the first pipelines)
}
const void *vk_sg_pin(VkShaderStageFlagBits stage, const uint32_t *local) {
    if (!(V.sg_pin_stages & stage)) return NULL;
    // a compute workgroup must fit in maxComputeWorkgroupSubgroups subgroups of the pinned size (VUID-VkPipelineShaderStageCreateInfo-pNext-02756): a larger one keeps the driver's size
    if (local && (uint64_t)local[0] * local[1] * local[2] > (uint64_t)V.sg_pin.requiredSubgroupSize * V.sg_maxwgs) return NULL;
    return &V.sg_pin;
}
void vk_pipeline_cache_save(void) {
    static pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER; static size_t last;
    const char *root = getenv("DM_ROOT"); if (!root || !V.pcache) return;
    pthread_mutex_lock(&mu);
    size_t n = 0; if (vkGetPipelineCacheData(V.dev, V.pcache, &n, NULL) || !n || n == last) { pthread_mutex_unlock(&mu); return; }   // unchanged since the last save
    last = n; pthread_mutex_unlock(&mu);
    void *d = malloc(n); if (vkGetPipelineCacheData(V.dev, V.pcache, &n, d)) { free(d); return; }
    char path[1024], tmp[1100]; snprintf(path, sizeof path, "%s/vk_pipeline_cache.bin", root); snprintf(tmp, sizeof tmp, "%s.tmp", path);
    FILE *f = fopen(tmp, "wb"); if (f) { fwrite(d, 1, n, f); fclose(f); rename(tmp, path); }
    free(d);
}

// ---------------- memory pools ----------------
// DARK SOULS: REMASTERED made more than 4,096 pools within 3 s of boot (FromSoftware's engine gives many resources a small pool of its own: 0x1000, 0x12000 bytes), which stopped it
// with "too many memory pools"; the index (g_pool_ord) is 16-bit, so up to 65,535
#define MAXPOOL 32768
static vkpool_t g_pool[MAXPOOL]; static int g_npool;
static pthread_rwlock_t g_pool_mu = PTHREAD_RWLOCK_INITIALIZER;
static uint32_t g_pool_gen;   // bumped under the write lock when a pool is added or removed: vk_pool_of's per-thread cache holds while it is unchanged
// g_pool's indices by lo, remade under the write lock whenever a pool is added or removed: vk_pool_of searches it. A linear scan of g_pool behind the per-thread
// cache was 18% of the profiler's samples of Super Mario Bros. Wonder's replay in its first level (a draw looks up more pools than the cache holds); with the
// search its world map's replay took 4.2 ms a frame instead of 4.6-4.9
static uint16_t g_pool_ord[MAXPOOL];
// The index kept in order one pool at a time (sorting it whole at every add is O(n log n) a pool, which thousands of pools made slow): pool k (just appended) goes in at its place
static void pool_ord_insert(int k) {
    int lo = 0, hi = k; uint64_t a = g_pool[k].lo;
    while (lo < hi) { int m = (lo + hi) / 2; if (g_pool[g_pool_ord[m]].lo <= a) lo = m + 1; else hi = m; }
    memmove(g_pool_ord + lo + 1, g_pool_ord + lo, (size_t)(k - lo) * sizeof *g_pool_ord); g_pool_ord[lo] = (uint16_t)k;
}
// pool i removed and the last pool (index g_npool, already decremented) moved into its place: i leaves the index and the moved pool's entry takes index i
static void pool_ord_remove(int i) {
    int n = g_npool + 1, w = 0;
    for (int r = 0; r < n; r++) { uint16_t v = g_pool_ord[r]; if (v == i) continue; g_pool_ord[w++] = v == g_npool ? (uint16_t)i : v; }
}
static int recv_fds(int sock, int *fds, int max) {
    char data[4096], ctl[CMSG_SPACE(sizeof(int) * 16)]; struct iovec iov = {data, sizeof data};
    struct msghdr m = {0}; m.msg_iov = &iov; m.msg_iovlen = 1; m.msg_control = ctl; m.msg_controllen = sizeof ctl;
    if (recvmsg(sock, &m, 0) < 0) return -1;
    int n = 0;
    for (struct cmsghdr *c = CMSG_FIRSTHDR(&m); c; c = CMSG_NXTHDR(&m, c))
        if (c->cmsg_level == SOL_SOCKET && c->cmsg_type == SCM_RIGHTS)
            for (size_t i = 0; i < (c->cmsg_len - CMSG_LEN(0)) / sizeof(int) && n < max; i++) fds[n++] = ((int *)CMSG_DATA(c))[i];
    return n;
}
static const VkBufferUsageFlags POOL_USAGE0 = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT |
    VK_BUFFER_USAGE_STORAGE_TEXEL_BUFFER_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT |
    VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT;
#define POOL_USAGE (POOL_USAGE0 | (V.has_xfb ? VK_BUFFER_USAGE_TRANSFORM_FEEDBACK_BUFFER_BIT_EXT : 0) | (V.has_bda ? VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT : 0))   // any pool range may be a transform feedback buffer, or reached through a raw global memory pointer (vk_gmem_map)
// A buffer made with the device-address usage must have its memory allocated with this flag (VUID-vkBindBufferMemory-bufferDeviceAddress-03339): every POOL_USAGE allocation's chain starts here
#define BDA_ALLOC_FLAGS(next) VkMemoryAllocateFlagsInfo bda_fl = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO, (next), VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT, 0}; void *bda_next = V.has_bda ? (void *)&bda_fl : (void *)(next)
static uint64_t buf_gpu_addr(VkBuffer b) {   // the buffer's device address, 0 without the feature
    if (!V.has_bda || !vkGetBufferDeviceAddress) return 0;
    VkBufferDeviceAddressInfo i = {VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO, NULL, b}; return vkGetBufferDeviceAddress(V.dev, &i);
}
static VkResult pool_copied_alloc(vkpool_t *p, uint64_t size, int cached) {
    VkBufferCreateInfo bi = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, NULL, 0, size, POOL_USAGE, VK_SHARING_MODE_EXCLUSIVE};
    VkBuffer b; VkResult r = vkCreateBuffer(V.dev, &bi, NULL, &b); if (r) return r;
    VkMemoryRequirements req; vkGetBufferMemoryRequirements(V.dev, b, &req);
    uint32_t type = vk_mem_type(req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
        (cached ? VK_MEMORY_PROPERTY_HOST_CACHED_BIT : VK_MEMORY_PROPERTY_HOST_COHERENT_BIT), 0);
    if (type == UINT32_MAX) { vkDestroyBuffer(V.dev, b, NULL); return VK_ERROR_FEATURE_NOT_PRESENT; }
    BDA_ALLOC_FLAGS(NULL);
    VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, bda_next, req.size, type};
    VkDeviceMemory m; vk_mem_tag = "pool-copied"; r = vkAllocateMemory(V.dev, &ai, NULL, &m); vk_mem_tag = NULL;
    if (r) { vkDestroyBuffer(V.dev, b, NULL); return r; }
    void *map; r = vkBindBufferMemory(V.dev, b, m, 0);
    if (!r) r = vkMapMemory(V.dev, m, 0, VK_WHOLE_SIZE, 0, &map);
    if (r) { vkDestroyBuffer(V.dev, b, NULL); vkFreeMemory(V.dev, m, NULL); return r; }
    p->buf = b; p->mem = m; p->shadow = map; p->gpu = buf_gpu_addr(b);
    p->noncoherent = !(V.mem.memoryTypes[type].propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    mem_diag_pool(size, req.memoryTypeBits, type); return VK_SUCCESS;
}
static void pool_cache_sync(vkpool_t *p, int invalidate, uint64_t lo, uint64_t hi) {
    if (!p->noncoherent) return;
    // Each pool owns/maps its allocation; round cache ranges out to whole atoms.
    VkMappedMemoryRange range = {VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE, NULL, p->mem, 0, VK_WHOLE_SIZE};
    uint64_t atom = V.props.limits.nonCoherentAtomSize, size = p->hi - p->lo;
    range.offset = lo - lo % atom;
    uint64_t pad = (atom - hi % atom) % atom;
    // Use the allocation's end at the pool tail; subtract before rounding to avoid overflow.
    if (hi < size && pad < size - hi) range.size = hi + pad - range.offset;
    int diag = mem_diag_on(); uint64_t t = diag ? mem_diag_ns() : 0;
    if (invalidate) { VKCHK(vkInvalidateMappedMemoryRanges(V.dev, 1, &range)); }
    else { VKCHK(vkFlushMappedMemoryRanges(V.dev, 1, &range)); }
    if (diag) mem_diag_note(invalidate ? 4 : 3, t, range.size == VK_WHOLE_SIZE ? p->hi - p->lo - range.offset : range.size);
}
// The pool's bytes into the hardware buffer's fresh (zero) pages: only the pages the game has touched (present or swapped, /proc/self/pagemap bits 63 and 62); an
// untouched anonymous page is zero on both sides. A pool backed lazily (below) may be mostly untouched, and the copy is what a game thread waits on (nvn_dirty_freeze)
static uint64_t g_zc_copied, g_zc_skipped;   // bytes copied and bytes skipped as untouched, for the lazy pools' line
static void copy_resident(uint8_t *dst, const uint8_t *src, uint64_t size) {
    static int fd = -2; if (fd == -2) fd = open("/proc/self/pagemap", O_RDONLY | O_CLOEXEC);
    uint64_t pg = (uint64_t)getpagesize();
    if (fd < 0 || ((uintptr_t)src & (pg - 1)) || (size & (pg - 1))) { memcpy(dst, src, size); g_zc_copied += size; return; }
    enum { N = 4096 }; uint64_t ent[N];
    for (uint64_t off = 0; off < size; ) {
        uint64_t first = ((uintptr_t)src + off) / pg, n = (size - off) / pg; if (n > N) n = N;
        ssize_t got = pread(fd, ent, n * sizeof *ent, (off_t)(first * sizeof *ent));
        if (got < (ssize_t)(n * sizeof *ent)) { memcpy(dst + off, src + off, size - off); g_zc_copied += size - off; return; }   // (the rest whole: the read came short)
        for (uint64_t i = 0; i < n; i++, off += pg) {
            if (ent[i] & (3ull << 62)) { memcpy(dst + off, src + off, pg); g_zc_copied += pg; } else g_zc_skipped += pg;
        }
    }
}
// Freed zero-copy pools' buffers, kept for the next pool of the same size (DM_VK_POOL_REUSE=0 turns it off): a pool made from one skips the hardware buffer's
// allocation (a binder call to the allocator service) and the driver's import, which took Pokémon Scarlet's three loading threads ~130 + ~100 ms each while a thrown
// ball's capture assets load (they drop old pools as they make new ones; the game waits on them, frames of 100-220 ms), and a pool dropped from a game thread no longer
// waits for the GPU (df_add outside a submission waits for every submission so far). An entry is taken only once the GPU has passed the submission open or next when
// it was kept; the oldest the GPU is done with make room
typedef struct { AHardwareBuffer *ahb; VkBuffer buf; VkDeviceMemory mem; uint64_t size, fence; } zcfree_t;
enum { ZCF_N = 128 };
#define ZCF_BYTES (384ull << 20)
static zcfree_t g_zcf[ZCF_N]; static int g_nzcf; static uint64_t g_zcf_bytes; static pthread_mutex_t g_zcf_mu = PTHREAD_MUTEX_INITIALIZER;
static uint64_t g_zcf_hit, g_zcf_miss, g_zcf_kept;   // under g_zcf_mu (the pool line)
uint64_t mtl_last_fence(void);
static int zcf_on(void) { static int v = -1; if (v < 0) { const char *e = getenv("DM_VK_POOL_REUSE"); v = !(e && !atoi(e)); } return v; }
static void zcf_free_now(const zcfree_t *e) { vkDestroyBuffer(V.dev, e->buf, NULL); vkFreeMemory(V.dev, e->mem, NULL); AHardwareBuffer_release(e->ahb); }
static int zcf_take(uint64_t size, zcfree_t *out) {
    if (!zcf_on()) return 0;
    uint64_t done = vk_fence_done(); int r = 0;
    pthread_mutex_lock(&g_zcf_mu);
    for (int i = 0; i < g_nzcf; i++) if (g_zcf[i].size == size && g_zcf[i].fence <= done) {
        *out = g_zcf[i]; memmove(g_zcf + i, g_zcf + i + 1, (size_t)(g_nzcf - i - 1) * sizeof *g_zcf); g_nzcf--; g_zcf_bytes -= size; r = 1; break; }
    if (r) g_zcf_hit++; else g_zcf_miss++;
    pthread_mutex_unlock(&g_zcf_mu);
    return r;
}
static int zcf_put(AHardwareBuffer *ahb, VkBuffer buf, VkDeviceMemory mem, uint64_t size) {   // 1: kept (the caller frees nothing)
    if (!zcf_on() || size > ZCF_BYTES / 4) return 0;
    zcfree_t e = {ahb, buf, mem, size, mtl_last_fence() + 1}, old[ZCF_N]; int nold = 0;
    uint64_t done = vk_fence_done();
    pthread_mutex_lock(&g_zcf_mu);
    while (g_nzcf == ZCF_N || g_zcf_bytes + size > ZCF_BYTES) {   // the oldest the GPU is done with go
        int i = 0; while (i < g_nzcf && g_zcf[i].fence > done) i++;
        if (i == g_nzcf) break;
        old[nold++] = g_zcf[i]; g_zcf_bytes -= g_zcf[i].size; memmove(g_zcf + i, g_zcf + i + 1, (size_t)(g_nzcf - i - 1) * sizeof *g_zcf); g_nzcf--;
    }
    int kept = g_nzcf < ZCF_N && g_zcf_bytes + size <= ZCF_BYTES;
    if (kept) { g_zcf[g_nzcf++] = e; g_zcf_bytes += size; g_zcf_kept++; }
    pthread_mutex_unlock(&g_zcf_mu);
    for (int i = 0; i < nold; i++) zcf_free_now(&old[i]);
    return kept;
}
// Small pools take a slot of a slab (DM_VK_POOL_SLABS=0 turns it off): one hardware buffer allocated and imported once, its slots mapped at the pools' addresses
// (its dma-buf mmapped at the slot's offset) and bound at the slot's offset (vkpool_t.boff). A hardware buffer of its own cost a 48 KB pool ~170 us to allocate (a binder
// call to the allocator service) and ~180 us to import on an AYN Thor, Turnip and Qualcomm's driver alike, ~0.9 ms a pool with the mappings, and Pokémon Scarlet makes
// ~1,000 pools of 4-256 KB a second while a thrown ball's capture assets load (1,500 a second while a save loads): its loading threads spent ~0.5 s a throw on them while
// the game waited, in frames of 100-190 ms. Neither driver offers VK_EXT_external_memory_host, which would take the game's own pages. Slots are 1, 2, 3, 4, 6, 8 ... 64 pages
// (at most 50% larger than the pool), 64 a slab, between 2 and 8 MB; slabs are kept for good, as the image blocks are, and a freed slot is taken again once the GPU has
// passed the submission open or next when it was freed
#define SLAB_POOL_MAX (256ull << 10)
static const uint32_t SLAB_PAGES[] = {1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64};
enum { SLAB_NCLS = sizeof SLAB_PAGES / sizeof *SLAB_PAGES, SLAB_SLOTS = 512 };
typedef struct slab_t { struct slab_t *next; AHardwareBuffer *ahb; int fd; uint8_t *map; VkBuffer buf; VkDeviceMemory mem; uint64_t slot_bytes; uint32_t nslot, nfree;
                        uint64_t bits[SLAB_SLOTS / 64], fence[SLAB_SLOTS]; } slab_t;   // bits: 1 = slot free; fence: the submission the GPU must have passed before it is taken
static slab_t *g_slab[SLAB_NCLS]; static pthread_mutex_t g_slab_mu = PTHREAD_MUTEX_INITIALIZER;
static uint64_t g_slab_used, g_slab_bytes;   // pools placed in slabs so far, slab memory (under g_slab_mu: the pool line)
static int slab_on(void) { static int v = -1; if (v < 0) { const char *e = getenv("DM_VK_POOL_SLABS"); v = !(e && !atoi(e)); } return v; }
static slab_t *slab_new(int c) {   // a slab of class c, allocated and imported (no lock held)
    uint64_t sb = (uint64_t)SLAB_PAGES[c] << 12; uint32_t n = 64;
    if (sb * n < (2u << 20)) n = (uint32_t)((2u << 20) / sb); if (sb * n > (8u << 20)) n = (uint32_t)((8u << 20) / sb); if (n > SLAB_SLOTS) n = SLAB_SLOTS;
    uint64_t size = sb * n; slab_t *s = calloc(1, sizeof *s); if (!s) return NULL;
    s->slot_bytes = sb; s->nslot = s->nfree = n; s->fd = -1;
    AHardwareBuffer_Desc d = {(uint32_t)size, 1, 1, AHARDWAREBUFFER_FORMAT_BLOB, AHARDWAREBUFFER_USAGE_CPU_READ_OFTEN | AHARDWAREBUFFER_USAGE_CPU_WRITE_OFTEN | AHARDWAREBUFFER_USAGE_GPU_DATA_BUFFER, 0, 0, 0};
    if (AHardwareBuffer_allocate(&d, &s->ahb)) { free(s); return NULL; }
    int sv[2], fds[16], nfd = -1;
    if (!socketpair(AF_UNIX, SOCK_SEQPACKET | SOCK_CLOEXEC, 0, sv)) {
        if (!AHardwareBuffer_sendHandleToUnixSocket(s->ahb, sv[0])) nfd = recv_fds(sv[1], fds, 16);
        close(sv[0]); close(sv[1]);
    }
    if (nfd >= 1) { s->fd = fds[0]; for (int i = 1; i < nfd; i++) close(fds[i]); }
    s->map = s->fd >= 0 ? mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, s->fd, 0) : MAP_FAILED;
    VkAndroidHardwareBufferPropertiesANDROID ap = {VK_STRUCTURE_TYPE_ANDROID_HARDWARE_BUFFER_PROPERTIES_ANDROID};
    VkExternalMemoryBufferCreateInfo eb = {VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO, NULL, VK_EXTERNAL_MEMORY_HANDLE_TYPE_ANDROID_HARDWARE_BUFFER_BIT_ANDROID};
    VkBufferCreateInfo bi = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, &eb, 0, size, POOL_USAGE, VK_SHARING_MODE_EXCLUSIVE};
    int ok = s->map != MAP_FAILED && !vkGetAndroidHardwareBufferPropertiesANDROID(V.dev, s->ahb, &ap) && !vkCreateBuffer(V.dev, &bi, NULL, &s->buf);
    if (ok) {
        VkMemoryDedicatedAllocateInfo ded = {VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO, NULL, VK_NULL_HANDLE, s->buf};
        VkImportAndroidHardwareBufferInfoANDROID imp = {VK_STRUCTURE_TYPE_IMPORT_ANDROID_HARDWARE_BUFFER_INFO_ANDROID, &ded, s->ahb};
        VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, &imp, ap.allocationSize, vk_mem_type(ap.memoryTypeBits, 0, 0)};
        ok = !vkAllocateMemory(V.dev, &ai, NULL, &s->mem);
        if (ok && vkBindBufferMemory(V.dev, s->buf, s->mem, 0)) { vkFreeMemory(V.dev, s->mem, NULL); ok = 0; }
        if (!ok) vkDestroyBuffer(V.dev, s->buf, NULL);
    }
    if (!ok) {
        static int said; if (!said++) VKLOG("[RED] memory pool slab of %llu KB could not be made: small pools get hardware buffers of their own", (unsigned long long)(size >> 10));
        if (s->map != MAP_FAILED) munmap(s->map, size); if (s->fd >= 0) close(s->fd); AHardwareBuffer_release(s->ahb); free(s); return NULL;
    }
    for (uint32_t i = 0; i < n; i++) s->bits[i >> 6] |= 1ull << (i & 63);
    return s;
}
static int slab_take(uint64_t size, slab_t **out, uint32_t *slot) {   // 1: a slot of at least size bytes
    if (!slab_on() || size > SLAB_POOL_MAX) return 0;
    uint64_t pages = (size + 4095) >> 12; int c = 0; while (c < SLAB_NCLS && SLAB_PAGES[c] < pages) c++;
    if (c == SLAB_NCLS) return 0;
    uint64_t done = vk_fence_done();
    for (int pass = 0; pass < 2; pass++) {
        pthread_mutex_lock(&g_slab_mu);
        for (slab_t *s = g_slab[c]; s; s = s->next) {
            if (!s->nfree) continue;
            for (uint32_t w = 0; w < (s->nslot + 63) / 64; w++) for (uint64_t b = s->bits[w]; b; b &= b - 1) {
                uint32_t i = w * 64 + (uint32_t)__builtin_ctzll(b); if (s->fence[i] > done) continue;
                s->bits[w] &= ~(1ull << (i & 63)); s->nfree--; g_slab_used++; pthread_mutex_unlock(&g_slab_mu); *out = s; *slot = i; return 1;
            }
        }
        pthread_mutex_unlock(&g_slab_mu);
        if (pass) break;
        slab_t *s = slab_new(c); if (!s) return 0;   // every slot taken or still the GPU's: a new slab, made outside the lock
        pthread_mutex_lock(&g_slab_mu); s->next = g_slab[c]; g_slab[c] = s; g_slab_bytes += s->slot_bytes * s->nslot; pthread_mutex_unlock(&g_slab_mu);
    }
    return 0;
}
static void slab_give(slab_t *s, uint32_t i, uint64_t fence) {
    pthread_mutex_lock(&g_slab_mu); s->fence[i] = fence; s->bits[i >> 6] |= 1ull << (i & 63); s->nfree++; pthread_mutex_unlock(&g_slab_mu);
}
// Zero copy: the pages of [mem, mem+size) become an AHardwareBuffer's, imported as a VkBuffer (a slab's slot for a small pool, or a kept buffer of a freed pool of that size). 0 on success
static int pool_zero_copy(vkpool_t *p, uint8_t *mem, uint64_t size) {
    slab_t *sl; uint32_t si;
    if (slab_take(size, &sl, &si)) {
        uint64_t off = (uint64_t)si * sl->slot_bytes;
        memcpy(sl->map + off, mem, size);
        if (mmap(mem, size, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_FIXED, sl->fd, (off_t)off) == mem) {
            p->buf = sl->buf; p->mem = VK_NULL_HANDLE; p->ahb = NULL; p->boff = off; p->slab = sl; p->slot = si; p->zero_copy = 1; p->gpu = buf_gpu_addr(sl->buf) + off; return 0; }
        mem_diag_zc(11, "mmap-fixed slab", size, "errno", errno); slab_give(sl, si, 0);   // (the anonymous pages are still there: a buffer of its own)
    }
    AHardwareBuffer_Desc d = {(uint32_t)size, 1, 1, AHARDWAREBUFFER_FORMAT_BLOB,
        AHARDWAREBUFFER_USAGE_CPU_READ_OFTEN | AHARDWAREBUFFER_USAGE_CPU_WRITE_OFTEN | AHARDWAREBUFFER_USAGE_GPU_DATA_BUFFER, 0, 0, 0};
    if (size > 0xffffffffull) { mem_diag_zc(0, "size", size, "limit", -1); return -1; }
    zcfree_t re; int reuse = zcf_take(size, &re);
    AHardwareBuffer *ahb = reuse ? re.ahb : NULL; int ar = reuse ? 0 : AHardwareBuffer_allocate(&d, &ahb);
    if (ar) { mem_diag_zc(1, "AHardwareBuffer_allocate", size, "return", ar); return -1; }
#define ZC_DROP() do { if (reuse) zcf_free_now(&re); else AHardwareBuffer_release(ahb); } while (0)
    int sv[2]; if (socketpair(AF_UNIX, SOCK_SEQPACKET | SOCK_CLOEXEC, 0, sv)) {
        mem_diag_zc(2, "socketpair", size, "errno", errno); ZC_DROP(); return -1; }
    int fds[16], nfd = -1;
    ar = AHardwareBuffer_sendHandleToUnixSocket(ahb, sv[0]);
    if (!ar) { nfd = recv_fds(sv[1], fds, 16); if (nfd < 1) mem_diag_zc(4, "recv_fds", size, "return", nfd); }
    else mem_diag_zc(3, "AHardwareBuffer_sendHandleToUnixSocket", size, "return", ar);
    close(sv[0]); close(sv[1]);
    if (nfd < 1) { ZC_DROP(); return -1; }
    // copy the pool's current contents into the buffer through a temporary mapping, then put the buffer's pages at the pool's address
    uint8_t *tmp = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, fds[0], 0);
    int ok = tmp != MAP_FAILED;
    if (ok) {
        if (reuse) { memcpy(tmp, mem, size); g_zc_copied += size; } else copy_resident(tmp, mem, size);   // (a kept buffer holds its last pool's bytes: every page, not only the touched ones)
        munmap(tmp, size);
        ok = mmap(mem, size, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_FIXED, fds[0], 0) == mem;
        if (!ok) mem_diag_zc(6, "mmap-fixed", size, "errno", errno);
    } else mem_diag_zc(5, "mmap-temporary", size, "errno", errno);
    for (int i = 0; i < nfd; i++) close(fds[i]);
    if (!ok) { ZC_DROP(); return -1; }
#undef ZC_DROP
    if (reuse) { p->buf = re.buf; p->mem = re.mem; p->ahb = ahb; p->zero_copy = 1; p->gpu = buf_gpu_addr(re.buf); return 0; }   // imported already
    VkAndroidHardwareBufferPropertiesANDROID ap = {VK_STRUCTURE_TYPE_ANDROID_HARDWARE_BUFFER_PROPERTIES_ANDROID};
    VkExternalMemoryBufferCreateInfo eb = {VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO, NULL, VK_EXTERNAL_MEMORY_HANDLE_TYPE_ANDROID_HARDWARE_BUFFER_BIT_ANDROID};
    VkBufferCreateInfo bi = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, &eb, 0, size, POOL_USAGE, VK_SHARING_MODE_EXCLUSIVE};
    VkResult vr = vkGetAndroidHardwareBufferPropertiesANDROID(V.dev, ahb, &ap);
    if (vr) { mem_diag_zc(7, "vkGetAndroidHardwareBufferPropertiesANDROID", size, "VkResult", vr); AHardwareBuffer_release(ahb); return -2; }
    vr = vkCreateBuffer(V.dev, &bi, NULL, &p->buf);
    if (vr) { mem_diag_zc(8, "vkCreateBuffer", size, "VkResult", vr); AHardwareBuffer_release(ahb); return -2; }
    VkMemoryDedicatedAllocateInfo ded = {VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO, NULL, VK_NULL_HANDLE, p->buf};   // the driver requires a dedicated allocation for an imported AHardwareBuffer
    BDA_ALLOC_FLAGS(&ded);
    VkImportAndroidHardwareBufferInfoANDROID imp = {VK_STRUCTURE_TYPE_IMPORT_ANDROID_HARDWARE_BUFFER_INFO_ANDROID, bda_next, ahb};
    VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, &imp, ap.allocationSize, vk_mem_type(ap.memoryTypeBits, 0, 0)};
    vk_mem_tag = "pool-zero-copy"; vr = vkAllocateMemory(V.dev, &ai, NULL, &p->mem); vk_mem_tag = NULL;
    if (!vr) { vr = vkBindBufferMemory(V.dev, p->buf, p->mem, 0); if (vr) mem_diag_zc(10, "vkBindBufferMemory", size, "VkResult", vr); }
    else mem_diag_zc(9, "vkAllocateMemory", size, "VkResult", vr);
    if (vr) { vkDestroyBuffer(V.dev, p->buf, NULL); AHardwareBuffer_release(ahb); return -2; }
    p->ahb = ahb; p->zero_copy = 1; p->gpu = buf_gpu_addr(p->buf);
    return 0;
}
// A zero-copy pool's dma-buf mapped again over [mem, mem+len) in one piece per chunk, read-only (nvn_dirty_defrag: the mappings its protection split never merge). 0 = done
int vk_pool_remap_ro(void *mem, size_t len) {
    int rc = 0; uint64_t done = 0;
    pthread_rwlock_rdlock(&g_pool_mu);
    for (int i = 0; i < g_npool && !rc; i++) {
        vkpool_t *p = &g_pool[i]; if (p->base != (uint64_t)(uintptr_t)mem || p->end - p->base != len) continue;
        if (!p->zero_copy) { if (!p->shadow) { rc = -1; break; } done += p->hi - p->lo; continue; }   // a copied pool: nothing to map again (its protection stays as it is), and done, so defrag stops taking its pages busy
        if (p->slab) { void *at = (void *)(uintptr_t)p->lo; rc = mmap(at, p->hi - p->lo, PROT_READ, MAP_SHARED | MAP_FIXED, ((slab_t *)p->slab)->fd, (off_t)p->boff) == at ? 0 : -1; if (!rc) done += p->hi - p->lo; continue; }   // a slab's slot
        int sv[2], fds[16], nfd = -1; if (socketpair(AF_UNIX, SOCK_SEQPACKET | SOCK_CLOEXEC, 0, sv)) { rc = -1; break; }
        if (!AHardwareBuffer_sendHandleToUnixSocket(p->ahb, sv[0])) nfd = recv_fds(sv[1], fds, 16);
        close(sv[0]); close(sv[1]);
        void *at = (void *)(uintptr_t)p->lo; rc = nfd >= 1 && mmap(at, p->hi - p->lo, PROT_READ, MAP_SHARED | MAP_FIXED, fds[0], 0) == at ? 0 : -1;
        for (int k = 0; k < nfd; k++) close(fds[k]);
        if (!rc) done += p->hi - p->lo;
    }
    pthread_rwlock_unlock(&g_pool_mu);
    return !rc && done == len ? 0 : -1;
}
static void pool_copy_back(vkpool_t *p) {   // the pages go back to anonymous memory with the buffer's contents (the game may reuse the memory)
    uint64_t size = p->hi - p->lo; uint8_t *mem = (uint8_t *)(uintptr_t)p->lo;
    uint8_t *save = malloc(size); if (save) memcpy(save, mem, size);
    mmap(mem, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED, -1, 0);
    if (save) { memcpy(mem, save, size); free(save); }
}
// Pools over ZC_CHUNK are made of chunks of at most that size, each a pool entry with an AHardwareBuffer of its own: the Adreno 830's gralloc allocator service
// (vendor.qti.hardware.display.allocator-service, snapalloc) aborts on a BLOB of 812 MB ("ubsan: mul-overflow" in SnapConstraintManager::AlignmentToAlignedConstraints),
// and the whole Android UI restarts with it; 400 MB ones work (13 Sentinels: Aegis Rim's 0x32c00000-byte pool). A range across two chunks has no buffer (vk_pool_of): Ocarina of
// Time's 288 MB pool (0x12000000) was two chunks of 256 MB and thousands of its ranges a scene crossed the seam, so chunks are 400 MB (the size known to work)
#define ZC_CHUNK 0x19000000ull
static void pool_add_one(const void *mem, uint64_t size, uint64_t base, uint64_t end);
// DM_VK_ZEROCOPY_MAX_MB=<n>: pools bigger than n MB are copied, not zero-copy, and kept whole (a copied pool takes no hardware buffer, so no chunks). The Legion Y700
// 2023's gralloc aborts the process (no error returned) allocating a hardware buffer for Unicorn Overlord's 860 MB pool, while its pools up to 250 MB are fine
static int zc_over(uint64_t size) { static uint64_t m = 1; if (m == 1) { const char *e = getenv("DM_VK_ZEROCOPY_MAX_MB"); m = e && atoi(e) > 0 ? (uint64_t)atoi(e) << 20 : 0; } return m && size > m; }
void mtl_pool_add(const void *mem, uint64_t size) {
    vk_init();
    if (!size) return;
    { void vk_cap_pool_note(const void *, uint64_t, int); vk_cap_pool_note(mem, size, 1); }   // DM_NVN_CAPTURE (gfx_vk_capture.inc): the pools as given
    uint64_t a = (uint64_t)(uintptr_t)mem, c = ((a | size) & 0xfff) || zc_over(size) ? size : ZC_CHUNK;
    for (uint64_t o = 0; o < size; o += c) pool_add_one((const uint8_t *)mem + o, size - o < c ? size - o : c, a, a + size);
    if (size > c) VKLOG("memory pool %p +%#llx: %llu chunks of at most %#llx", mem, (unsigned long long)size, (unsigned long long)((size + c - 1) / c), c);
}
// The pool line: pools made and dropped with their time on the calling threads and the reuse counts, at most once a second and only for a second in which
// pools took 20 ms or more (loads): "memory pools in the last 1.0 s: 212 made (148.0 MB, 640.0 ms; 150 from kept buffers), 180 dropped (...)"
static uint64_t pool_ns(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return (uint64_t)t.tv_sec * 1000000000ull + (uint64_t)t.tv_nsec; }
static pthread_mutex_t g_pn_mu = PTHREAD_MUTEX_INITIALIZER; static uint64_t g_pn_t0, g_pn_n[2], g_pn_b[2], g_pn_ns[2], g_pn_hit0, g_pn_slab0;
static void pool_note(int kind, uint64_t bytes, uint64_t ns) {   // kind 0: made, 1: dropped (lock order: g_pn_mu, then g_zcf_mu or g_slab_mu)
    uint64_t now = pool_ns(); char b[400] = "";
    pthread_mutex_lock(&g_pn_mu);
    if (!g_pn_t0) g_pn_t0 = now;
    g_pn_n[kind]++; g_pn_b[kind] += bytes; g_pn_ns[kind] += ns;
    if (now - g_pn_t0 >= 1000000000ull) {
        pthread_mutex_lock(&g_zcf_mu); uint64_t hit = g_zcf_hit, kn = (uint64_t)g_nzcf, kb = g_zcf_bytes; pthread_mutex_unlock(&g_zcf_mu);
        pthread_mutex_lock(&g_slab_mu); uint64_t su = g_slab_used, sbytes = g_slab_bytes; pthread_mutex_unlock(&g_slab_mu);
        if (g_pn_ns[0] + g_pn_ns[1] >= 20000000ull)
            snprintf(b, sizeof b, "memory pools in the last %.1f s: %llu made (%.1f MB, %.1f ms; %llu in slab slots, %llu from kept buffers), %llu dropped (%.1f MB, %.1f ms); slabs %.1f MB, %llu buffers kept (%.1f MB)",
                     (now - g_pn_t0) / 1e9, (unsigned long long)g_pn_n[0], g_pn_b[0] / 1048576.0, g_pn_ns[0] / 1e6, (unsigned long long)(su - g_pn_slab0), (unsigned long long)(hit - g_pn_hit0),
                     (unsigned long long)g_pn_n[1], g_pn_b[1] / 1048576.0, g_pn_ns[1] / 1e6, sbytes / 1048576.0, (unsigned long long)kn, kb / 1048576.0);
        g_pn_t0 = now; g_pn_hit0 = hit; g_pn_slab0 = su; memset(g_pn_n, 0, sizeof g_pn_n); memset(g_pn_b, 0, sizeof g_pn_b); memset(g_pn_ns, 0, sizeof g_pn_ns);
    }
    pthread_mutex_unlock(&g_pn_mu);
    if (*b) VKLOG("%s", b);
}
// Lazy backing (default; DM_VK_EAGER_POOLS=1 backs every pool at creation as before): a pool's chunk gets its hardware buffer (or GPU copy) only when the replay first
// resolves an address in it to a VkBuffer (vk_pool_of). Nintendo Switch 2 Welcome Tour makes over 3,800 pools, 5.7 GB, in its first seconds, and the hardware buffers
// committed every page of them (gralloc's system heap allocates eagerly, and the copy wrote every page) while the game itself had touched about 1.3 GB; with the first
// scene's images on top the low-memory killer ended it at 10 GB. Most of those pools hold packaged textures, which the replay reads on the CPU (nvn_deswizzle into
// scratch memory), so their buffers are never asked for. Until it is backed a chunk is the game's own anonymous pages: the dirty-page tracking works on them as on any
// (nvn_dirty_add_range), and the host copies / fills / reads them as before.
//   Backing later means the game may be writing the pool meanwhile: the chunk is frozen (nvn_dirty_freeze: a game write to it waits in the fault handler, a host write
// waits before it starts), its protected pages given up (nvn_dirty_hold, so no state says "read-only" of pages the new mapping makes writable), made read-only, copied
// (touched pages only), mapped over, then unfrozen. Raw global memory (shaders dereferencing addresses, vk_gmem_map) needs a device address for every pool such an
// address can lie in: once a shader uses it, pools the game took an address in (mtl_pool_addressed) are backed when the table is rebuilt, and new pools at creation.
static int g_pool_lazy = -1, g_gmem_used;
static uint64_t g_pool_chunks, g_pool_backed, g_pool_bytes, g_pool_backed_bytes, g_pool_lazy_backed;   // chunks and bytes made / backed, chunks backed after creation
static int pool_lazy_on(void) { if (g_pool_lazy < 0) { const char *e = getenv("DM_VK_EAGER_POOLS"); g_pool_lazy = !(e && atoi(e)); } return g_pool_lazy; }
void vk_pool_stats(uint64_t *pools, uint64_t *backed, uint64_t *bytes, uint64_t *backed_bytes) { *pools = g_pool_chunks; *backed = g_pool_backed; *bytes = g_pool_bytes; *backed_bytes = g_pool_backed_bytes; }
// The chunk's buffer: zero-copy where it can be, else a GPU copy refreshed before each submission. Under the write lock, or on a pool not published yet (pool_add_one).
// late: after creation (the game may be writing it)
static void pool_back_locked(vkpool_t *p, int late) {
    uint8_t *mem = (uint8_t *)(uintptr_t)p->lo; uint64_t size = p->hi - p->lo;
    int zc = ((uintptr_t)mem & 0xfff) || (size & 0xfff) ? -3 : getenv("DM_VK_NO_ZEROCOPY") ? -4 : zc_over(p->end - p->base) ? -5 : 0, frozen = 0, held = 0;
    if (!zc && late) {   // the game's writes to the chunk wait from here to the mapping; a write between the copy and the mapping would be lost
        nvn_dirty_hold(mem, size); nvn_dirty_freeze(mem, size); held = 1;
        if (!mprotect(mem, size, PROT_READ)) frozen = 1; else { static int said; if (!said++) VKLOG("[RED] memory pool %p +%#llx: read-only for its late backing refused (errno %d); backed without the wait", (void *)mem, (unsigned long long)size, errno); }
    }
    if (!zc) zc = pool_zero_copy(p, mem, size);
    if (frozen && zc) mprotect(mem, size, PROT_READ | PROT_WRITE);   // (zero-copy done: the new mapping is writable by itself)
    if (held) { nvn_dirty_unfreeze(); nvn_dirty_unhold(); }
    if (zc) {
        static int said; if (!said++) VKLOG("[RED] memory pool %p +%#llx could not be zero-copy (%d): a GPU copy is refreshed before each submission (slow)", (void *)mem, (unsigned long long)size, zc);
        const char *e = getenv("DM_VK_POOL_CACHED"); int cached = e && atoi(e);
        VkResult r = VK_ERROR_FEATURE_NOT_PRESENT;
        if (cached && vkFlushMappedMemoryRanges && vkInvalidateMappedMemoryRanges) r = pool_copied_alloc(p, size, 1);
        if (r == VK_ERROR_DEVICE_LOST) VKCHK(r);
        if (r) {
            if (cached && mem_diag_on()) VKLOG("[memdiag] cached pool unavailable (%d), using coherent memory", r);
            VKCHK(pool_copied_alloc(p, size, 0));
        }
    }
    p->backed = 1; g_pool_backed++; g_pool_backed_bytes += size; if (late) g_pool_lazy_backed++;
    static uint64_t zc_bytes, cp_bytes; if (zc) cp_bytes += size; else zc_bytes += size;
    static int nlog; if (nlog++ < 24) VKLOG("memory pool %p +%#llx: %s%s (zero-copy %.1f MB, copied %.1f MB)", (void *)mem, (unsigned long long)size, zc ? "copied" : "zero-copy", late ? ", backed at its first GPU use" : "", zc_bytes / 1048576.0, cp_bytes / 1048576.0);
}
// A pool backed at creation (DM_VK_EAGER_POOLS=1, or once shaders use raw global memory) gets its buffer before the write lock is taken, on a copy of its entry no lookup can
// see yet, and only the entry is put in under the lock (upstream nx2android 8a41d0413): the hardware buffer (a binder call to the allocator service), the copy into it and the
// driver's import take milliseconds each, and every lookup waits on the lock (vk_pool_of: each buffer binding of each draw on the replay thread, and the game threads'
// mtl_addr). Pokémon Scarlet makes and drops pools on three threads at once while a thrown ball flies (~180 MB of capture assets in 1.5 s): the replay thread stood almost
// still for a second with the GPU idle. A lazy chunk (the default) is backed at its first GPU use instead (vk_pool_of)
static void pool_add_one(const void *mem, uint64_t size, uint64_t base, uint64_t end) {
    uint64_t t0 = pool_ns();
    vkpool_t q; memset(&q, 0, sizeof q);
    q.lo = (uint64_t)(uintptr_t)mem; q.hi = q.lo + size; q.base = base; q.end = end;
    if (!pool_lazy_on() || __atomic_load_n(&g_gmem_used, __ATOMIC_ACQUIRE)) pool_back_locked(&q, 0);
    pthread_rwlock_wrlock(&g_pool_mu);
    if (g_npool == MAXPOOL) { pthread_rwlock_unlock(&g_pool_mu); die("[vk] too many memory pools"); }
    { static uint64_t ser; q.serial = ++ser; }
    g_pool_chunks++; g_pool_bytes += size;
    g_pool[g_npool] = q; pool_ord_insert(g_npool++); __atomic_fetch_add(&g_pool_gen, 1, __ATOMIC_RELEASE);
    int np = g_npool;
    pthread_rwlock_unlock(&g_pool_mu);
    if (np >= 1024 && !(np & (np - 1))) VKLOG("%d memory pools live; %llu chunks made so far (%.1f GB), %llu backed (%.1f GB; %llu of them at their first GPU use)", np, (unsigned long long)g_pool_chunks, g_pool_bytes / 1073741824.0, (unsigned long long)g_pool_backed, g_pool_backed_bytes / 1073741824.0, (unsigned long long)g_pool_lazy_backed);
    pool_note(0, size, pool_ns() - t0);
}
// Pools not yet backed whose [lo, hi) lies in or crosses [addr, addr+len): backed (the write lock taken; the caller holds no pool lock). From the replay (vk_pool_of, vk_gmem_map)
static vkpool_t *pool_find_locked(uint64_t addr, uint64_t len, int *across) {   // vk_pool_of's search, under either lock
    int lo = 0, hi = g_npool;   // the last pool by lo that starts at or below addr
    while (lo < hi) { int m = (lo + hi) / 2; if (g_pool[g_pool_ord[m]].lo <= addr) lo = m + 1; else hi = m; }
    if (lo > 0) { vkpool_t *p = &g_pool[g_pool_ord[lo - 1]]; if (addr + len <= p->hi) return p; }
    for (int i = 0; i < g_npool; i++) if (addr >= g_pool[i].lo && addr + len <= g_pool[i].hi) return &g_pool[i];   // (pools over the same memory, and the range across two chunks it reports)
                                       else if (across && addr >= g_pool[i].lo && addr < g_pool[i].hi && addr + len <= g_pool[i].end) *across = 1;
    return NULL;
}
static vkpool_t *pool_back_at(uint64_t addr, uint64_t len) {
    pthread_rwlock_wrlock(&g_pool_mu);
    vkpool_t *p = pool_find_locked(addr, len, NULL);   // found again under the write lock: the table may have changed since the read
    if (p && !p->backed) { pool_back_locked(p, 1); __atomic_fetch_add(&g_pool_gen, 1, __ATOMIC_RELEASE);
        static int said; if (!said++) VKLOG("memory pools are backed at their first GPU use (DM_VK_EAGER_POOLS=1: at creation); the first: %#llx +%#llx", (unsigned long long)p->lo, (unsigned long long)(p->hi - p->lo)); }
    pthread_rwlock_unlock(&g_pool_mu);
    return p;
}
// The game took a GPU address in [mem, mem+size) (hle_nvn.c pool_mark_addressed: a buffer or texture address, or the pool's): raw global memory may reach it, so once a
// shader uses raw global memory such a pool is backed when the pool table is rebuilt (a changed generation makes vk_gmem_map rebuild it). From the game's threads
void mtl_pool_addressed(const void *mem, uint64_t size) {
    uint64_t a = (uint64_t)(uintptr_t)mem, e = a + (size ? size : 1); int changed = 0;
    pthread_rwlock_rdlock(&g_pool_mu);
    for (int i = 0; i < g_npool; i++) { vkpool_t *p = &g_pool[i]; if (p->lo < e && a < p->hi && !p->addressed) { p->addressed = 1; changed |= !p->backed; } }
    pthread_rwlock_unlock(&g_pool_mu);
    if (changed && __atomic_load_n(&g_gmem_used, __ATOMIC_ACQUIRE)) __atomic_fetch_add(&g_pool_gen, 1, __ATOMIC_RELEASE);
}
static void pools_back_addressed(void) {   // raw global memory in use: every addressed pool gets its buffer (and new pools get theirs at creation)
    if (!__atomic_load_n(&g_gmem_used, __ATOMIC_ACQUIRE)) { __atomic_store_n(&g_gmem_used, 1, __ATOMIC_RELEASE); VKLOG("raw global memory in use: memory pools the game took addresses in are backed now, and new pools at creation"); }
    int n = 0; uint64_t bytes = 0;
    pthread_rwlock_wrlock(&g_pool_mu);
    for (int i = 0; i < g_npool; i++) { vkpool_t *p = &g_pool[i]; if (p->addressed && !p->backed) { pool_back_locked(p, 1); n++; bytes += p->hi - p->lo; } }
    if (n) __atomic_fetch_add(&g_pool_gen, 1, __ATOMIC_RELEASE);
    pthread_rwlock_unlock(&g_pool_mu);
    static int nl; if (n && nl++ < 8) VKLOG("raw global memory: %d addressed memory pool chunks backed (%.1f MB)", n, bytes / 1048576.0);
}
void mtl_pool_del(const void *mem) {
    { void vk_cap_pool_note(const void *, uint64_t, int); vk_cap_pool_note(mem, 0, 0); }
    for (;;) {   // every chunk of the pool
        pthread_rwlock_wrlock(&g_pool_mu);
        int i = 0; while (i < g_npool && g_pool[i].base != (uint64_t)(uintptr_t)mem) i++;
        if (i == g_npool) { pthread_rwlock_unlock(&g_pool_mu); return; }
        vkpool_t p = g_pool[i]; g_pool[i] = g_pool[--g_npool]; pool_ord_remove(i); __atomic_fetch_add(&g_pool_gen, 1, __ATOMIC_RELEASE);
        pthread_rwlock_unlock(&g_pool_mu);
        uint64_t t0 = pool_ns();
        { void vk_cpu_storage_gone(uint64_t lo, uint64_t hi); vk_cpu_storage_gone(p.lo, p.hi); }   // its GPU-written spans and taken textures (gfx_vk_tex.c)
        if (p.zero_copy) pool_copy_back(&p);
        free(p.baseline);
        if (p.slab) slab_give(p.slab, p.slot, mtl_last_fence() + 1);   // the slot goes back to its slab (the GPU may still read it until that submission is done)
        else if (p.buf && !(p.zero_copy && p.ahb && zcf_put(p.ahb, p.buf, p.mem, p.hi - p.lo))) {   // (kept for the next pool of its size; a lazy pool never backed has no buffer)
            vk_defer_free_buffer(p.buf, p.mem);   // the AHardwareBuffer goes with the imported memory's last reference
            if (p.ahb) AHardwareBuffer_release(p.ahb);
        }
        pool_note(1, p.hi - p.lo, pool_ns() - t0);
    }
}
__thread const char *vk_pool_tag;
vkpool_t *vk_pool_holding(uint64_t addr, uint64_t len) {
    vkpool_t *r = NULL;
    if (len > UINT64_MAX - addr) return NULL;
    pthread_rwlock_rdlock(&g_pool_mu);
    // CPU reads may span chunks of one pool: containment ends at pool.end, not chunk.hi.
    // Table fills ask this for every texture; scanning thousands of pools per entry
    // dominated OoT's replay. Use the same ordered index and lifetime lock as buffer lookups.
    int lo = 0, hi = g_npool;
    while (lo < hi) { int m = (lo + hi) / 2; if (g_pool[g_pool_ord[m]].lo <= addr) lo = m + 1; else hi = m; }
    if (lo > 0) { vkpool_t *p = &g_pool[g_pool_ord[lo - 1]]; if (addr < p->hi && addr + len <= p->end) r = p; }
    // An overlapping later-starting pool can be too short while an earlier one contains
    // the range. Preserve that case; no cached pointer survives a mutation without a lock.
    for (int i = 0; i < g_npool && !r; i++) if (addr >= g_pool[i].lo && addr < g_pool[i].hi && addr + len <= g_pool[i].end) r = &g_pool[i];
    pthread_rwlock_unlock(&g_pool_mu);
    return r;
}
static vkpool_t *vk_pool_of_(uint64_t addr, uint64_t len, int back) {
    // the last pools this thread found, while none was added or removed: the replay looks up every buffer binding of every draw, and the read lock alone was ~6% of it
    static __thread struct { uint32_t gen; vkpool_t *p[4]; } c;   // gen: g_pool_gen + 1 (0: empty)
    uint32_t g = __atomic_load_n(&g_pool_gen, __ATOMIC_ACQUIRE) + 1;
    if (c.gen == g) for (int i = 0; i < 4; i++) { vkpool_t *p = c.p[i]; if (p && addr >= p->lo && addr + len <= p->hi) return p; }   // (cached pools are backed: backing bumps the generation)
    vkpool_t *r = NULL; int across = 0;
    pthread_rwlock_rdlock(&g_pool_mu);
    r = pool_find_locked(addr, len, &across);
    int unbacked = r && !r->backed;
    pthread_rwlock_unlock(&g_pool_mu);
    if (unbacked && back) { r = pool_back_at(addr, len); g = __atomic_load_n(&g_pool_gen, __ATOMIC_ACQUIRE) + 1; }
    if (r && (r->backed || !back)) { if (c.gen != g) { memset(&c, 0, sizeof c); c.gen = g; } if (r->backed) { memmove(&c.p[1], &c.p[0], 3 * sizeof *c.p); c.p[0] = r; } }
    if (!r && across) { static uint64_t n; uint64_t k = ++n; if (!(k & (k - 1))) LOG("[RED] [vk] %llu ranges across two chunks of a memory pool (ZC_CHUNK; this one %#llx +%#llx, for %s): no buffer for them", (unsigned long long)k, (unsigned long long)addr, (unsigned long long)len, vk_pool_tag ? vk_pool_tag : "other"); }
    return r;
}
vkpool_t *vk_pool_of(uint64_t addr, uint64_t len) { return vk_pool_of_(addr, len, 1); }
vkpool_t *vk_pool_peek(uint64_t addr, uint64_t len) { return vk_pool_of_(addr, len, 0); }
uint32_t vk_pool_gen(void) { return __atomic_load_n(&g_pool_gen, __ATOMIC_ACQUIRE); }   // pools added or removed (gfx_vk_exec.c's descriptor set memo: vk_pool_of's results hold while it is unchanged)
int vk_buf_at(uint64_t addr, uint64_t len, VkBuffer *buf, VkDeviceSize *off) {
    vkpool_t *p = vk_pool_of(addr, len ? len : 1);
    if (!p) return 0;
    *buf = p->buf; *off = addr - p->lo + p->boff; return 1;
}
int mtl_addr(uint64_t addr, uint64_t len, void **buf, uint64_t *off) {   // (CPU-side: the pool's memory, not its buffer)
    vkpool_t *p = vk_pool_peek(addr, len ? len : 1);
    if (!p) return 0;
    if (buf) *buf = (void *)(uintptr_t)p->lo; if (off) *off = addr - p->lo; return 1;
}
void mtl_alias_drop_pool(const void *pool) { void vk_tex_drop_pool(const void *); vk_tex_drop_pool(pool); }
// The NVN flags of the pool at mem (hle_nvn.c, after mtl_pool_add): a copied pool the CPU maps uncached (CPU_UNCACHED, 0x2) is one the game writes and the GPU reads;
// the game reads GPU results from cached memory, so such a pool is not copied back after each submission
void vk_pool_set_flags(const void *mem, int flags) {
    pthread_rwlock_rdlock(&g_pool_mu);
    for (int i = 0; i < g_npool; i++) if (g_pool[i].base == (uint64_t)(uintptr_t)mem) g_pool[i].flags = flags;   // (every chunk of it)
    pthread_rwlock_unlock(&g_pool_mu);
}
static uint64_t g_sync_bytes, g_sync_full;   // copied pools: bytes copied by the incremental sync, full copies
// Copied pools: the guest's bytes into the GPU copy. The first sync copies the whole pool; later ones copy only the 64 KB blocks with pages the CPU wrote since the last
// sync (nvn_dirty's write-protect tracking: a block is protected again before it is copied, so a write during the copy is caught by the next sync). An 860 MB
// pool copied whole at every submission (Unicorn Overlord on a Legion Y700 2023) ran at a few frames a second
static void vk_pools_sync_(VkCommandBuffer cb);
void vk_pools_sync(VkCommandBuffer cb) { extern int g_prof_on; extern uint64_t g_prof_ns[8]; struct timespec a, b; if (g_prof_on) clock_gettime(CLOCK_MONOTONIC, &a);
    int diag = mem_diag_on(); uint64_t t = diag ? mem_diag_ns() : 0, bytes = diag ? g_sync_bytes : 0;
    vk_pools_sync_(cb); if (diag) mem_diag_note(1, t, g_sync_bytes - bytes);
    if (g_prof_on) { clock_gettime(CLOCK_MONOTONIC, &b); g_prof_ns[2] += (uint64_t)((b.tv_sec - a.tv_sec) * 1000000000ll + (b.tv_nsec - a.tv_nsec)); } }
static void vk_pools_sync_(VkCommandBuffer cb) {
    (void)cb;
    int diag = mem_diag_on(); uint64_t ns[4] = {0}, amount[4] = {0};
    uint64_t reason_count[4] = {0}, reason_bytes[4] = {0};
#define SYNC_MEASURE(k, n, op) do { if (diag) { uint64_t t_ = mem_diag_ns(); op; ns[k] += mem_diag_ns() - t_; amount[k] += (n); } else { op; } } while (0)
    enum { BLK = 64 << 10 };
    pthread_rwlock_rdlock(&g_pool_mu);
    for (int i = 0; i < g_npool; i++) {
        vkpool_t *p = &g_pool[i]; if (!p->shadow) continue;
        uint8_t *lo = (uint8_t *)(uintptr_t)p->lo; uint64_t size = p->hi - p->lo, st = nvn_dirty_now();
        if (!p->baseline) { p->baseline = malloc(size); if (!p->baseline) die("[RED] [vk] copied-pool baseline allocation failed (%llu bytes)", (unsigned long long)size);
            p->synced = 0; if (mem_diag_on()) VKLOG("[memdiag] copy-back baseline allocated: %.2f MiB", size / 1048576.0); }
        if (!p->span && (p->span = nvn_dirty_span(lo, size))) p->synced = 0;   // tracked only now (nvnMemoryPoolInitialize registers the range after the pool): copied whole once more, protected
        if (!p->synced || !p->span) {   // untracked: every sync copies all of it, as before
            if (diag) { int reason = !p->synced ? 0 : 1; reason_count[reason]++; reason_bytes[reason] += size; }
            if (p->span) SYNC_MEASURE(1, size, nvn_dirty_protect(lo, size));
            SYNC_MEASURE(2, size, memcpy(p->shadow, lo, size)); SYNC_MEASURE(3, size, memcpy(p->baseline, p->shadow, size)); pool_cache_sync(p, 0, 0, size); g_sync_bytes += size; g_sync_full++; p->synced = 1; p->baseline_complete = 0; p->stamp0 = p->stamp = st; continue; }
        // judged against the sync before the last: a write whose fault took its generation before the last sync's stamp but stored it after that sync looked
        // at its block would otherwise never count (its page is writable from then on, and clean by every later stamp). A dirty block is copied twice
        uint64_t first = size, last = 0;
        uint64_t scan_start = diag ? mem_diag_ns() : 0, work_start = ns[1] + ns[2] + ns[3];
        for (uint64_t off = 0; off < size; off += BLK) {
            uint64_t n = size - off < BLK ? size - off : BLK;
            if (!nvn_dirty_since_span(p->span, lo + off, n, p->stamp0)) continue;
            SYNC_MEASURE(1, n, nvn_dirty_protect(lo + off, n));
            SYNC_MEASURE(2, n, memcpy(p->shadow + off, lo + off, n)); g_sync_bytes += n;
            if (off < first) first = off; last = off + n;
            if (p->baseline_complete) SYNC_MEASURE(3, n, memcpy(p->baseline + off, p->shadow + off, n));
        }
        // A full merge already refreshed the baseline; only uploaded blocks changed. A pool synced again with no merge since (no completion) takes a full snapshot.
        if (diag) { ns[0] += mem_diag_ns() - scan_start - (ns[1] + ns[2] + ns[3] - work_start); amount[0] += size; }
        if (!p->baseline_complete) {
            if (diag) { int reason = p->flags & 0x2 ? 2 : 3; reason_count[reason]++; reason_bytes[reason] += size; }
            SYNC_MEASURE(3, size, memcpy(p->baseline, p->shadow, size));
        }
        if (last) pool_cache_sync(p, 0, first, last);
        p->baseline_complete = 0;
        p->stamp0 = p->stamp; p->stamp = st;
    }
    pthread_rwlock_unlock(&g_pool_mu);
    if (diag) for (unsigned k = 0; k < 4; k++) mem_diag_note(6 + k, mem_diag_ns() - ns[k], amount[k]);
    if (diag) mem_diag_baselines(reason_count, reason_bytes);
#undef SYNC_MEASURE
    static uint64_t n, last; if (++n % 600 == 0) { VKLOG("copied pools: %.1f MB synced in the last 600 submissions (%.2f MB each), %llu full copies so far", (g_sync_bytes - last) / 1048576.0, (g_sync_bytes - last) / 600.0 / 1048576.0, (unsigned long long)g_sync_full); last = g_sync_bytes; }
}
// Copied pools after a submission: merge GPU changes against their baseline, preserving independent CPU writes made after the sync. Never-synced pools are left alone.
// A CPU-uncached pool (vk_pool_set_flags) takes back only the ranges the submission's GPU writes covered (sub_gpu_wrote):
// copying 860 MB back after every submission held Unicorn Overlord to a few frames a second
struct vksub; static void pools_copy_back(const struct vksub *s);

// ---------------- submission slots ----------------
#define NSLOT 6
#define MAIN_CBS 32   // (allocated as used: DM_VK_PASS_SPLIT cuts inside passes as well)
// GPU time: a timestamp at the start and end of each submission's main command buffer (one queue: their sum is the time the GPU was busy)
static VkQueryPool g_tsq; uint64_t g_gpu_ns;
// Timestamp counters (ReportCounter TIMESTAMP): each writes a GPU timestamp where it is in the command stream (CQ per slot), and at completion it gets the guest
// clock mapped from a calibrated CPU/GPU pair (completion-relative time is the unsupported-device fallback). They used to all get the completion time of their submission: a game timing its GPU
// work between two counters read 0 when both were in one submission and the gap between completions (our CPU replay included) when not. Xenoblade Chronicles 3
// sizes its frame by that time (dynamic resolution): its main pass went between 960x544 and 776x440 every few frames while the camera turned; with GPU times
// the size holds steady
enum { CQ = 64 };
static VkQueryPool g_cq;
#define SCRATCH_BLOCK (8u << 20)
enum { SCRATCH_IDLE = 64 };
static VkDeviceSize scratch_cap(void);
enum { DPX_KEEP = 4, DPX_DESCRIPTORS = 1 << 20 };   // oversized sets retained per completed slot, with a descriptor-count budget
typedef struct { VkDescriptorPool pool; VkDescriptorSet set; VkDescriptorSetLayout layout; uint64_t descriptors; int used; uint64_t lease; unsigned cached, idle; } dpx_entry;
enum { DP_KEEP = 16 };   // descriptor pools kept per slot across recycles (vk_alloc_set, slot_recycle)
static uint64_t g_dp_made, g_dp_freed;   // descriptor pools made and destroyed
static uint64_t kgsl_mapped(void);   // recycles of its slot a scratch block may go unused before it is freed (slot_recycle)
static uint64_t g_scratch_freed;   // bytes of scratch blocks freed as idle
typedef struct { VkBuffer b; VkDeviceMemory m; uint8_t *p; VkDeviceSize size, used; uint32_t idle; } scratch_blk;
typedef struct { int kind; void *a; void *b; } dfree_t;   // 0 buffer+memory, 1 image+memory, 2 view
typedef struct { int kind; volatile uint32_t *val; uint32_t mode, v; uint64_t addr; uint32_t type; int q; } post_op;   // q: a timestamp counter's query in g_cq, or -1
enum { DP_TYPES = 10 };
uint64_t g_desc_alloc_ns, g_desc_alloc_calls, g_desc_alloc_sets;
int g_desc_timing;
struct vksub {
    VkCommandPool pool; VkCommandBuffer up, main; int up_used;
    VkCommandBuffer mains[MAIN_CBS]; int nmain_alloc, nmain_used, cb_err;
    VkCommandBuffer ups[MAIN_CBS]; int nup_alloc, nup_used; int nmain_sent;   // upload command buffers in order (one per part sent early), main buffers already submitted   // the submission's main command buffers in order (vk_cmd_split); main = the one being recorded
    uint64_t fence;   // the fence this slot last committed with (0 = never)
    uint64_t t_begin, part_ns[MAIN_CBS];   // DM_VK_PASS_TIMES chain line: host time of mtl_sub_begin and of each part's vkQueueSubmit
    int cut_diag; uint32_t cut_success, cut_budget_refused, cut_capacity_refused; uint64_t cut_cpu_ns;
    int ncq;          // timestamp counter queries used in this submission (g_cq)
    VkQueryPool *oq; int noq, capoq, nocc;   // occlusion blocks kept until this slot's completion; nocc queries used this submission
    struct { uint32_t first_q, last_q, begins, ends, first_block, last_block, resets; VkCommandBuffer reset_cb; } occ[MAIN_CBS]; // bounded failure diagnostics, one record per native part
    scratch_blk *blk; int nblk, capblk;
    VkDescriptorPool *dp; int ndp, capdp, dp_cur;
    uint32_t dp_used[DP_TYPES], dp_sets; int dp_counted; int no_ts;   // what pool dp_cur has given out (vk_alloc_set)
    dpx_entry *dpx; int ndpx, capdpx;   // one-set oversized pools; only completed-slot sets may be reused
    dfree_t *df; int ndf, capdf;
    post_op *post; int npost, cappost;
    struct { uint64_t lo, hi; } gw[64]; int ngw, readback_ranges;   // GPU writes into pool memory in this submission (sub_gpu_wrote), for pools_copy_back: nearby ones merged, and once full each joins the nearest
};
enum { GW_SUB = 64 };
static vksub_t g_slot[NSLOT]; static int g_cur = -1; uint64_t g_cmd_gen;   // g_cmd_gen: main command buffers begun (gfx_vk_exec.c bind cache)
static pthread_mutex_t g_df_mu = PTHREAD_MUTEX_INITIALIZER;   // the slots' deferred-free lists (df, ndf, capdf): df_add says why
static uint64_t pool_merge_back(vkpool_t *p, uint64_t lo, uint64_t hi) {
    int diag = mem_diag_on(); uint64_t t = diag ? mem_diag_ns() : 0;
    uint64_t bytes = 0;
    uint8_t after[4096];
    // A whole-pool copy overwrote new CPU data even outside GPU writes (reproduced with independent bytes in one cache line).
    for (uint64_t off = lo - p->lo, end = hi - p->lo; off < end; ) {
        size_t n = end - off < sizeof after ? (size_t)(end - off) : sizeof after;
        // Direct scans of mapped memory reached 1 s on Mali; stage once so byte comparisons use CPU memory.
        memcpy(after, p->shadow + off, n);
        const uint8_t *src = after; uint8_t *base = p->baseline + off, *dst = (uint8_t *)(uintptr_t)(p->lo + off);
        if (memcmp(base, src, n)) {
            dm_guest_write_prepare(dst, n);
            for (size_t i = 0; i < n; ) {
                if (base[i] == src[i]) { i++; continue; }
                size_t a = i++; while (i < n && base[i] != src[i]) i++;
                memcpy(dst + a, src + a, i - a); memcpy(base + a, src + a, i - a); bytes += i - a;
            }
        }
        off += n;
    }
    if (diag) mem_diag_note(5, t, hi - lo);
    return bytes;
}
static void pools_copy_back(const struct vksub *s) {
    int diag = mem_diag_on(); uint64_t t = diag ? mem_diag_ns() : 0, bytes = 0;
    uint64_t invalidate_ns = 0, invalidate_calls = 0, invalidate_bytes = 0;
    pthread_rwlock_rdlock(&g_pool_mu);
    for (int i = 0; i < g_npool; i++) { vkpool_t *p = &g_pool[i]; if (!p->shadow || !p->synced) continue;
        uint64_t first = 0, last = p->hi - p->lo;
        // Mali: full invalidation took 13 ms per submission; eligible ranges cover every GPU write.
        // Other shadow bytes stay unchanged; the next sync overwrites whole dirty blocks before snapshotting them.
        if (s->readback_ranges) {
            first = last; last = 0;
            for (int k = 0; k < s->ngw; k++) { uint64_t lo = s->gw[k].lo > p->lo ? s->gw[k].lo : p->lo, hi = s->gw[k].hi < p->hi ? s->gw[k].hi : p->hi;
                if (lo < hi) { if (lo - p->lo < first) first = lo - p->lo; if (hi - p->lo > last) last = hi - p->lo; } }
        }
        if (first < last) {
            uint64_t start = diag && p->noncoherent ? mem_diag_ns() : 0;
            pool_cache_sync(p, 1, first, last);
            if (diag && p->noncoherent) { invalidate_ns += mem_diag_ns() - start; invalidate_calls++; invalidate_bytes += last - first; }
        }
        if (!(p->flags & 0x2) && !s->readback_ranges) { bytes += pool_merge_back(p, p->lo, p->hi); p->baseline_complete = 1; continue; }
        // The baseline stays complete after a merge of the tracked ranges, for NVN CPU_UNCACHED pools too after a submission with an unknown writer: those pools
        // trust the tracked ranges, as before the baseline (an untracked GPU write stays in the GPU copy only and is merged with a later tracked range over it).
        // Taking a full snapshot at the next sync instead copied the whole pool before every submission: Unicorn Overlord's 860 MB CPU_UNCACHED pool, as before
        p->baseline_complete = 1;
        for (int k = 0; k < s->ngw; k++) { uint64_t lo = s->gw[k].lo > p->lo ? s->gw[k].lo : p->lo, hi = s->gw[k].hi < p->hi ? s->gw[k].hi : p->hi;
            if (lo < hi) bytes += pool_merge_back(p, lo, hi); } }
    pthread_rwlock_unlock(&g_pool_mu);
    if (diag) {
        uint64_t total = mem_diag_ns() - t;
        mem_diag_note(2, t, bytes);
        mem_diag_copyback(total, invalidate_ns, invalidate_calls, invalidate_bytes);
        static uint64_t seen, narrowed; if (s->readback_ranges) narrowed++;
        if (++seen <= 3 || seen % 600 == 0) VKLOG("[memdiag] readback coverage: %llu range submissions / %llu total (others retain full cached-pool merges)", (unsigned long long)narrowed, (unsigned long long)seen);
    }
}
int vk_readback_ranges(void) { return g_cur >= 0 && g_slot[g_cur].readback_ranges; }
void vk_readback_unknown(void) { if (g_cur >= 0) g_slot[g_cur].readback_ranges = 0; }
static VkCommandBuffer g_slot_main_cb(void) { return g_slot[g_cur].main; }
// DM_VK_PASS_TIMES=1: a timestamp before and after every render pass (the replay's per-frame line then lists the passes that cost the most GPU time)
enum { PASSQ = 512 };   // passes (and dispatches) timed per submission
static VkQueryPool g_pq; static int g_pq_on = -1;
typedef struct { uint32_t w, h, fmt, draws, fidx, part, flags; uint64_t cpu_ns; } pass_meta;   // part: the command buffer it was recorded in; cpu_ns: host time its end was recorded   // fidx: the pass's place in its frame (submissions are cut every DM_VK_CB_DRAWS draws)
enum { FPASSQ = 512 }; static int g_fpass;   // passes of the frame so far
void vk_pass_frame(void) { g_fpass = 0; }   // a present: the next pass is the frame's first
static pass_meta g_pmeta[NSLOT][PASSQ]; static int g_npass[NSLOT];
static double g_pacc[FPASSQ]; static uint32_t g_pcnt[FPASSQ]; static pass_meta g_plast[FPASSQ]; static int g_pmax;   // by the pass's place in its frame (the completion thread adds)
static int pass_times_on(void) { if (g_pq_on < 0) { const char *e = getenv("DM_VK_PASS_TIMES"); g_pq_on = e && atoi(e); } return g_pq_on && g_pq; }
uint32_t g_vk_pass_flags;   // the pass being ended (gfx_vk_exec.c vk_end_pass): colour attachments (bits 0-3), depth (4), depth loaded (5), colour 0 loaded (6), split continuation (7)
void vk_pass_ts(int end, uint32_t w, uint32_t h, uint32_t fmt, uint32_t draws) {
    if (!pass_times_on() || g_cur < 0) return; int *n = &g_npass[g_cur]; if (*n >= PASSQ) return;
    vkCmdWriteTimestamp(g_slot_main_cb(), end ? VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT : VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, g_pq, (uint32_t)(g_cur * PASSQ * 2 + *n * 2 + end));
    if (end) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); g_pmeta[g_cur][*n] = (pass_meta){w, h, fmt, draws, (uint32_t)(g_fpass < FPASSQ - 1 ? g_fpass : FPASSQ - 1), (uint32_t)(g_slot[g_cur].nmain_used - 1), g_vk_pass_flags, (uint64_t)t.tv_sec * 1000000000ull + (uint64_t)t.tv_nsec}; (*n)++; g_fpass++; }
}
void vk_pass_report(void) {   // the most expensive passes since the last report, then the counters start again
    if (!pass_times_on()) return;
    static int idx[FPASSQ]; int m = g_pmax; for (int i = 0; i < m; i++) idx[i] = i;
    for (int i = 0; i < m; i++) for (int j = i + 1; j < m; j++) if (g_pcnt[idx[j]] && (!g_pcnt[idx[i]] || g_pacc[idx[j]] / g_pcnt[idx[j]] > g_pacc[idx[i]] / g_pcnt[idx[i]])) { int t = idx[i]; idx[i] = idx[j]; idx[j] = t; }
    double tot = 0; for (int i = 0; i < m; i++) if (g_pcnt[i]) tot += g_pacc[i] / g_pcnt[i];
    char b[3000]; int o = snprintf(b, sizeof b, "passes by GPU time (%d passes, %.2f ms in passes per frame):", m, tot);
    for (int i = 0; i < m && i < 20 && o < (int)sizeof b - 80; i++) { int k = idx[i]; if (!g_pcnt[k]) continue;
        if (g_plast[k].fmt == PT_DISPATCH) o += snprintf(b + o, sizeof b - (size_t)o, " #%d %.2f ms (dispatch %ux%ux%u);", k, g_pacc[k] / g_pcnt[k], g_plast[k].w, g_plast[k].h, g_plast[k].draws);
        else o += snprintf(b + o, sizeof b - (size_t)o, " #%d %.2f ms (%ux%u fmt %u, %u draws over %u px wide);", k, g_pacc[k] / g_pcnt[k], g_plast[k].w, g_plast[k].h, g_plast[k].fmt, g_plast[k].draws & 0xffff, g_plast[k].draws >> 16); }
    { extern uint64_t g_nvn_discards[2]; static uint64_t pd0, pd1; o += snprintf(b + o, sizeof b - (size_t)o, " discards since: colour %llu, depth %llu", (unsigned long long)(g_nvn_discards[0] - pd0), (unsigned long long)(g_nvn_discards[1] - pd1)); pd0 = g_nvn_discards[0]; pd1 = g_nvn_discards[1]; }
    VKLOG("%s", b);
    memset(g_pacc, 0, sizeof g_pacc); memset(g_pcnt, 0, sizeof g_pcnt); g_pmax = 0;
}
// one submission at a time: the game's (hle_nvn.c holds its own lock around them) and the backend's own (texel readback, present)
static pthread_mutex_t g_sub_lock = PTHREAD_MUTEX_INITIALIZER;
static uint64_t g_fence;   // fence of the last commit
static VkSemaphore g_timeline;   // submission f signals the value f (V.has_timeline), else g_fring[f % NFRING]
#define NFRING 16   // more than the submissions in flight (NSLOT: mtl_sub_begin's slot_recycle waits for the slot's last one)
static VkFence g_fring[NFRING];
static pthread_mutex_t g_done_mu = PTHREAD_MUTEX_INITIALIZER; static pthread_cond_t g_done_cv = PTHREAD_COND_INITIALIZER;
static uint64_t g_done;
static void slots_init(void) {
    VkSemaphoreTypeCreateInfo ti = {VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO, NULL, VK_SEMAPHORE_TYPE_TIMELINE, 0};
    VkSemaphoreCreateInfo si = {VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, &ti};
    if (V.has_timeline) VKCHK(vkCreateSemaphore(V.dev, &si, NULL, &g_timeline));
    else for (int i = 0; i < NFRING; i++) { VkFenceCreateInfo fi = {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO}; VKCHK(vkCreateFence(V.dev, &fi, NULL, &g_fring[i])); }
    VkQueryPoolCreateInfo qi = {VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO, NULL, 0, VK_QUERY_TYPE_TIMESTAMP, 2 * NSLOT};
    if (V.props.limits.timestampComputeAndGraphics && vkCreateQueryPool(V.dev, &qi, NULL, &g_tsq)) g_tsq = VK_NULL_HANDLE;
    VkQueryPoolCreateInfo cqi = {VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO, NULL, 0, VK_QUERY_TYPE_TIMESTAMP, NSLOT * CQ};
    if (!g_tsq || vkCreateQueryPool(V.dev, &cqi, NULL, &g_cq)) g_cq = VK_NULL_HANDLE;
    { const char *e = getenv("DM_VK_PASS_TIMES"); if (e && atoi(e)) { VkQueryPoolCreateInfo pq = {VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO, NULL, 0, VK_QUERY_TYPE_TIMESTAMP, NSLOT * PASSQ * 2}; if (vkCreateQueryPool(V.dev, &pq, NULL, &g_pq)) g_pq = VK_NULL_HANDLE; } }
    for (int i = 0; i < NSLOT; i++) {
        // not TRANSIENT, and reset by command buffer (see slot_recycle)
        VkCommandPoolCreateInfo ci = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, NULL, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, V.qf};
        VKCHK(vkCreateCommandPool(V.dev, &ci, NULL, &g_slot[i].pool));
        VkCommandBuffer cb[2]; VkCommandBufferAllocateInfo ai = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, NULL, g_slot[i].pool, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 2};
        VKCHK(vkAllocateCommandBuffers(V.dev, &ai, cb)); g_slot[i].up = g_slot[i].ups[0] = cb[0]; g_slot[i].main = g_slot[i].mains[0] = cb[1]; g_slot[i].nmain_alloc = 1; g_slot[i].nup_alloc = 1;
    }
}
static uint64_t g_reports_done;
uint64_t vk_reports_fence_done(void) { return __atomic_load_n(&g_reports_done, __ATOMIC_ACQUIRE); }
uint64_t mtl_completed_fence(void) { return vk_fence_done(); }
uint64_t vk_fence_done(void) { return __atomic_load_n(&g_done, __ATOMIC_ACQUIRE); }   // written under g_done_mu (for the waiters) with a release store
int mtl_wait(uint64_t fence, uint64_t timeout_ns);
// Without timeline semaphores only the completion thread waits on the ring's fences, in submission order (a fence tells nothing of the submissions before
// it); everyone else waits for the completion thread to pass the fence (g_done), so a fence is never waited on while the commit resets it for reuse
static void wait_fence(uint64_t f) {
    if (!f) return;
    if (!V.has_timeline) { mtl_wait(f, ~0ull); return; }
    VkSemaphoreWaitInfo wi = {VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO, NULL, 0, 1, &g_timeline, &f};
    VkResult r = vkWaitSemaphores(V.dev, &wi, UINT64_MAX); if (r != VK_SUCCESS) die("[vk] GPU timeline wait did not complete: %d", r);
}
// The slot's command buffers are reset on the completion thread once its submission is done (completion_main), not when the replay begins them again: an
// implicit reset at vkBeginCommandBuffer took 7% of Xenoblade Chronicles 3's replay thread at a submission's start and more at every part (its battles
// replay ~3,000 draws a frame on a thread that is the frame's limit). The slot is then taken only after the completion thread has passed it (mtl_wait, g_done)
// rather than as soon as the GPU has. DM_VK_X_NOPRERESET=1 (start-up): reset as they are begun
static int g_prereset = -1;
// Live A/B: keep the already allocated one-set pool and set, not its contents. The caller rewrites
// the descriptors as usual. Only slot_recycle, after its completion wait, clears used; an open
// submission never receives the same set twice. A load's excess and a previously idle set go away.
// F1 content-leased sets alone may survive eight idle slot recycles, within the same count/descriptor
// caps: logical-queue continuation submissions otherwise expire tables before their next draw.
static uint64_t g_descriptor_lease;
static int descriptor_large_reuse(void) { const char *e = getenv("DM_VK_DPX_REUSE"); return e && atoi(e) != 0; }
static VkDescriptorSet descriptor_large_take(vksub_t *s, VkDescriptorSetLayout layout) {
    if (!descriptor_large_reuse()) return VK_NULL_HANDLE;
    for (int i = 0; i < s->ndpx && i < DPX_KEEP; i++) {
        dpx_entry *e = &s->dpx[i];
        if (!e->used && e->layout == layout) { e->used = 1; e->lease = ++g_descriptor_lease; e->cached = e->idle = 0;
            static uint64_t n; if (++n <= 4 || !(n & (n - 1))) VKLOG("descriptor oversized sets: %llu completed-slot reuses", (unsigned long long)n);
            return e->set; }
    }
    return VK_NULL_HANDLE;
}
static void descriptor_large_recycle(vksub_t *s) {
    int reuse = descriptor_large_reuse(), kept = 0, freed = 0; uint64_t descriptors = 0;
    const char *opt = getenv("DM_VK_POOL_PERSIST"); int persistent = opt && atoi(opt) != 0;
    for (int i = 0; i < s->ndpx; i++) { dpx_entry e = s->dpx[i];
        if (reuse && (e.used || (persistent && e.cached && e.idle < 8)) && kept < DPX_KEEP && e.descriptors <= DPX_DESCRIPTORS - descriptors) {
            descriptors += e.descriptors; e.idle = e.used ? 0 : e.idle + 1; e.used = 0; s->dpx[kept++] = e;
        } else { vkDestroyDescriptorPool(V.dev, e.pool, NULL); freed++; }
    }
    s->ndpx = kept;
    if (freed) { static int nl; if (nl++ < 4) VKLOG("descriptor pools: %d one-set pools of this slot destroyed after its submission (%d retained)", freed, kept); }
}
static void descriptor_pools_reset(vksub_t *s) {   // called only after the slot's completion wait
    for (int i = 0; i < s->ndp; i++) { if (i < DP_KEEP) vkResetDescriptorPool(V.dev, s->dp[i], 0); else { vkDestroyDescriptorPool(V.dev, s->dp[i], NULL); g_dp_freed++; } }
    descriptor_large_recycle(s);
    if (s->ndp > DP_KEEP) { static int nl; if (nl++ < 8) VKLOG("descriptor pools: %d of this slot's destroyed after its submission (%d kept; %.0f MB of GPU memory mapped now)", s->ndp - DP_KEEP, DP_KEEP, kgsl_mapped() / 1048576.0); s->ndp = DP_KEEP; }
    s->dp_cur = 0; s->dp_counted = -1;
}
static void slot_recycle(vksub_t *s) {   // the GPU has passed this slot's last submission: free what it held, rewind its allocators
    if (g_prereset < 0) { const char *e = getenv("DM_VK_X_NOPRERESET"); g_prereset = !(e && atoi(e)); }
    // Query results are consumed on the completion thread; a GPU fence alone must not allow their pools to be reset first.
    if (g_prereset || s->nocc) mtl_wait(s->fence, ~0ull); else wait_fence(s->fence);
    pthread_mutex_lock(&g_df_mu); dfree_t *df = s->df; int ndf = s->ndf; s->df = NULL; s->ndf = s->capdf = 0; pthread_mutex_unlock(&g_df_mu);   // the list taken whole: a game thread may add to it meanwhile
    for (int i = 0; i < ndf; i++) {
        dfree_t *d = &df[i];
        if (d->kind == 0) { vkDestroyBuffer(V.dev, (VkBuffer)d->a, NULL); if (d->b) vkFreeMemory(V.dev, (VkDeviceMemory)d->b, NULL); }
        else if (d->kind == 1) { vkDestroyImage(V.dev, (VkImage)d->a, NULL); if (d->b) vkFreeMemory(V.dev, (VkDeviceMemory)d->b, NULL); }
        else if (d->kind == 4) { void img_sub_free_(void *); vkDestroyImage(V.dev, (VkImage)d->a, NULL); img_sub_free_(d->b); }
        else if (d->kind == 2) vkDestroyImageView(V.dev, (VkImageView)d->a, NULL);
        else if (d->kind == 3) vkDestroyBufferView(V.dev, (VkBufferView)d->a, NULL);
    }
    free(df);
    // scratch blocks are kept for the slot's next submissions, but one unused for SCRATCH_IDLE recycles of the slot goes (a scene's texture uploads can take hundreds of MB of
    // scratch per slot, and the blocks were never freed; DM_VK_SCRATCH_KEEP=1 keeps them all). The GPU is past this slot's last submission here
    { static int keep = -1; if (keep < 0) { const char *e = getenv("DM_VK_SCRATCH_KEEP"); keep = e && atoi(e); }
      for (int i = s->nblk - 1; i >= 0; i--) { scratch_blk *b = &s->blk[i]; b->idle = b->used ? 0 : b->idle + 1;
          if (!keep && i > 0 && b->idle > SCRATCH_IDLE) { vkDestroyBuffer(V.dev, b->b, NULL); vkFreeMemory(V.dev, b->m, NULL); g_scratch_freed += b->size; memmove(b, b + 1, (size_t)(s->nblk - 1 - i) * sizeof *b); s->nblk--; } } }
    for (int i = 0; i < s->nblk; i++) s->blk[i].used = 0;
    descriptor_pools_reset(s);
    // The pool is not reset: each command buffer is reset as it is begun again (the pool allows it). A reset of the transient pool made Adreno's driver free the
    // command memory and map it again as the next submission grew (ioctl, mmap and munmap): in Xenoblade Chronicles 3's battles the submission's start took 21%
    // of the replay thread, now 9%, and frames went from about 42 to 38 ms
}
static vksub_t *cur(void) { if (g_cur < 0) die("[vk] no open submission"); return &g_slot[g_cur]; }
int vk_sub_open(void) { return g_cur >= 0; }
// A long submission is recorded into several command buffers, submitted together in order: Adreno's driver refuses a command buffer past some size at
// vkEndCommandBuffer (VK_ERROR_OUT_OF_HOST_MEMORY), which NEO's handheld frames reached at about the 712th draw. The caller has ended the render pass.
static void part_sent(vksub_t *s) { if (!pass_times_on()) return; struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); uint64_t ns = (uint64_t)t.tv_sec * 1000000000ull + (uint64_t)t.tv_nsec; for (int i = s->nmain_sent; i < s->nmain_used; i++) s->part_ns[i] = ns; }   // (the chain line's part times)
static int vk_cmd_split_(void);
static void pools_readback_barrier(VkCommandBuffer cb);
// DM_VK_CUT_STATS=N: live, bounded to64 submissions;0 disables and rearms.
// No split clocks or logging on the default path.
static void cut_stats_begin(vksub_t *s) {
    static int previous = -1, left;
    const char *e = getenv("DM_VK_CUT_STATS"); int n = e ? atoi(e) : 0;
    if (n < 0) n = 0; if (n > 64) n = 64;
    if (n != previous) { previous = n; left = n; }
    s->cut_diag = left > 0; if (left > 0) left--;
    s->cut_success = s->cut_budget_refused = s->cut_capacity_refused = 0; s->cut_cpu_ns = 0;
}
void vk_cmd_split_refused(int capacity) {
    vksub_t *s = cur(); if (!s->cut_diag) return;
    if (capacity) s->cut_capacity_refused++; else s->cut_budget_refused++;
}
static void cut_stats_report(const vksub_t *s, uint64_t frame, uint64_t fence) {
    if (s->cut_diag) VKLOG("cut stats frame %llu fence %llu: parts %d/%d, early %d, successful %u, refused budget %u/capacity %u, split CPU %.3f ms",
        (unsigned long long)frame, (unsigned long long)fence, s->nmain_used, MAIN_CBS, s->nmain_sent,
        s->cut_success, s->cut_budget_refused, s->cut_capacity_refused, s->cut_cpu_ns / 1e6);
}
int vk_cmd_split(void) {
    extern int g_prof_on; extern uint64_t g_prof_ns[8];
    vksub_t *s = cur(); int timed = g_prof_on || s->cut_diag; struct timespec a, b;
    if (timed) clock_gettime(CLOCK_MONOTONIC, &a);
    int ok = vk_cmd_split_();
    if (timed) {
        clock_gettime(CLOCK_MONOTONIC, &b);
        uint64_t ns = (uint64_t)((b.tv_sec - a.tv_sec) * 1000000000ll + (b.tv_nsec - a.tv_nsec));
        if (g_prof_on) g_prof_ns[6] += ns;
        if (s->cut_diag) { s->cut_cpu_ns += ns; if (ok) s->cut_success++; else s->cut_capacity_refused++; }
    }
    return ok;
}
extern struct { uint64_t draws, skip_prog, skip_rt, skip_pipe, passes, folded, dispatches, pipes_built, rt_kept, ns_exec, ns_desc, ns_pipe, max_pass_draws; } g_vkst;   // (gfx_vk_exec.c; as gfx_vk_present.c declares it)
extern uint64_t g_set_new, g_pkg_n, g_pkg_bytes;
static struct { uint64_t draws, passes, dispatches, sets, pkg_n, pkg_bytes, kgsl, pipes, pipe_kgsl; int parts; } g_sub0;   // at this submission's start (sub_stats)
static void sub_stats(const char *what, int part) {   // the submission so far: what a long one is made of (Nintendo Switch 2 Welcome Tour's scene load: GPU memory the driver allocates grows by gigabytes in one submission)
    static int nl, nend; if (part >= 0 ? nl++ >= 48 : nend++ >= 48) return;   // the first 48 of each kind: Bananza's outdoor frames are single submissions of 4000+ draws, logged every frame before
    VKLOG("%s: part %d; so far %llu draws, %llu passes, %llu dispatches, %llu pipelines built (%.0f MB of GPU memory around their creation), %llu new descriptor sets, %llu texture uploads (%.0f MB); %.0f MB of GPU memory mapped (%.0f at the start)", what, part,
          (unsigned long long)(g_vkst.draws - g_sub0.draws), (unsigned long long)(g_vkst.passes - g_sub0.passes), (unsigned long long)(g_vkst.dispatches - g_sub0.dispatches), (unsigned long long)(g_vkst.pipes_built - g_sub0.pipes), (g_pipe_kgsl - g_sub0.pipe_kgsl) / 1048576.0, (unsigned long long)(g_set_new - g_sub0.sets),
          (unsigned long long)(g_pkg_n - g_sub0.pkg_n), (g_pkg_bytes - g_sub0.pkg_bytes) / 1048576.0, kgsl_mapped() / 1048576.0, g_sub0.kgsl / 1048576.0);
}
static int vk_cmd_split_(void) {
    vksub_t *s = cur(); if (s->nmain_used >= MAIN_CBS || s->nup_used >= MAIN_CBS) { static int said; if (!said++) sub_stats("a long submission: no parts left, the rest goes in the last", s->nmain_used); return 0; }
    pools_readback_barrier(s->main);
    if (vkEndCommandBuffer(s->main) < 0) { s->cb_err = 1; static int n; if (n++ < 3) VKLOG("[RED] a part of a long submission was refused at its end (part %d)", s->nmain_used); }
    // what is recorded so far goes to the GPU now, as a submission of its own (with its uploads first): Adreno refuses a single submission past about
    // 700 draws (NEO's handheld frames: VK_ERROR_OUT_OF_HOST_MEMORY at vkEndCommandBuffer whatever the draws were, and in one command buffer or several)
    const char *se = getenv("DM_VK_SUBMIT_SPLIT"); int send = !(se && !atoi(se));   // on unless DM_VK_SUBMIT_SPLIT=0
    if (send && !s->cb_err) {
        // A live 0->1 SUBMIT_SPLIT change can leave older parts unsent. Submit the
        // entire unsent prefix before advancing its ownership watermark.
        VkCommandBuffer cbs[1 + MAIN_CBS]; uint32_t n = 0;
        if (s->up_used) { vk_img_barrier_all(s->up); if (vkEndCommandBuffer(s->up) < 0) s->cb_err = 1; cbs[n++] = s->up; }
        for (int i = s->nmain_sent; i < s->nmain_used; i++) cbs[n++] = s->mains[i];
        if (!s->cb_err) {
            VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO, NULL, 0, NULL, NULL, n, cbs, 0, NULL};
            vk_stage_flush();   // (staged uploads before the part that may read them)
            pthread_mutex_lock(&g_vk_queue_mu); VkResult r = vkQueueSubmit(V.q, 1, &si, VK_NULL_HANDLE); pthread_mutex_unlock(&g_vk_queue_mu);
            if (r) { void vk_crumbs_report(void); vk_crumbs_report(); die("[vk] vkQueueSubmit (part): %d (device lost?)", r); }
            sub_stats("a long submission: part sent early", s->nmain_used);
            part_sent(s);
            s->nmain_sent = s->nmain_used;
            if (s->up_used) {   // the next part records its uploads into a fresh upload buffer
                if (s->nup_used == s->nup_alloc) { VkCommandBufferAllocateInfo ai = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, NULL, s->pool, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1};
                    VKCHK(vkAllocateCommandBuffers(V.dev, &ai, &s->ups[s->nup_alloc])); s->nup_alloc++; }
                s->up = s->ups[s->nup_used++]; s->up_used = 0;
            }
        }
    }
    if (s->nmain_used == s->nmain_alloc) { VkCommandBufferAllocateInfo ai = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, NULL, s->pool, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1};
        VKCHK(vkAllocateCommandBuffers(V.dev, &ai, &s->mains[s->nmain_alloc])); s->nmain_alloc++; }
    s->main = s->mains[s->nmain_used++];
    VkCommandBufferBeginInfo bi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, NULL, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT}; VKCHK(vkBeginCommandBuffer(s->main, &bi)); g_cmd_gen++; return 1;
}   // outside a submission nothing may be recorded (present, a pool finalized between frames)
VkCommandBuffer vk_cmd(void) { return cur()->main; }
int vk_cmd_parts_left(void) { vksub_t *s = cur(); return MAIN_CBS - (s->nmain_used > s->nup_used ? s->nmain_used : s->nup_used); }   // parts vk_cmd_split can still begin (gfx_vk_exec.c pass splits)
uint64_t g_up_gen;   // upload command buffers begun (gfx_vk_tex.c lin_copy: once per upload buffer)
VkCommandBuffer vk_upload_cmd(void) {
    vksub_t *s = cur();
    if (!s->up_used) { VkCommandBufferBeginInfo bi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, NULL, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT}; VKCHK(vkBeginCommandBuffer(s->up, &bi)); s->up_used = 1; g_up_gen++; }
    return s->up;
}
void *vk_scratch(VkDeviceSize n, VkDeviceSize align, VkBuffer *buf, VkDeviceSize *off) {
    vksub_t *s = cur(); if (!align) align = 16;
    for (int i = 0; i < s->nblk; i++) {
        scratch_blk *b = &s->blk[i]; VkDeviceSize o = (b->used + align - 1) & ~(align - 1);
        if (o + n <= b->size) { b->used = o + n; *buf = b->b; *off = o; return b->p + o; }
    }
    if (s->nblk == s->capblk) { s->capblk = s->capblk ? 2 * s->capblk : 8; s->blk = realloc(s->blk, s->capblk * sizeof *s->blk); }
    scratch_blk *b = &s->blk[s->nblk++]; memset(b, 0, sizeof *b);
    static VkDeviceSize blk = 0; if (!blk) { const char *e = getenv("DM_VK_SCRATCH_MB"); blk = e && atoi(e) > 0 ? (VkDeviceSize)atoi(e) << 20 : SCRATCH_BLOCK; }
    b->size = n > blk ? (n + 0xffff) & ~0xffffull : blk;
    VkBufferCreateInfo bi = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, NULL, 0, b->size, POOL_USAGE, VK_SHARING_MODE_EXCLUSIVE};
    VKCHK(vkCreateBuffer(V.dev, &bi, NULL, &b->b));
    VkMemoryRequirements req; vkGetBufferMemoryRequirements(V.dev, b->b, &req);
    BDA_ALLOC_FLAGS(NULL);   // (POOL_USAGE carries the device-address usage)
    VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, bda_next, req.size, vk_mem_type(req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 0)};
    vk_mem_tag = "scratch"; VKCHK(vkAllocateMemory(V.dev, &ai, NULL, &b->m)); vk_mem_tag = NULL; VKCHK(vkBindBufferMemory(V.dev, b->b, b->m, 0));
    void *m; VKCHK(vkMapMemory(V.dev, b->m, 0, VK_WHOLE_SIZE, 0, &m)); b->p = m;
    b->used = n; *buf = b->b; *off = 0;
    { VkDeviceSize t = 0; for (int i = 0; i < s->nblk; i++) t += s->blk[i].size; static int said; if (!said && t >= (64u << 20)) { said = 1; VKLOG("scratch: a submission slot first passed 64 MB (%.0f MB; staging engages past %llu MB)", t / 1048576.0, (unsigned long long)(scratch_cap() >> 20)); } }
    { VkDeviceSize t = 0; for (int i = 0; i < s->nblk; i++) t += s->blk[i].size; static VkDeviceSize last; if (t >= last + (256u << 20)) { last = t; VKLOG("scratch: %.0f MB of blocks in this submission's slot (%d blocks; the latest %.1f MB)", t / 1048576.0, s->nblk, b->size / 1048576.0); } }
    return b->p;
}
// ---------------- bounded staging (vk_stage_*) ----------------
// A scene's first texture uploads can run to gigabytes in one submission (Nintendo Switch 2 Welcome Tour's first scene after the title), each a scratch allocation
// that lives until the submission completes: the staging alone was as much memory as the images. Past DM_VK_SCRATCH_MAX_MB (default 256; 0 = never) of scratch in
// the submission, a texture's first upload goes through this one buffer and command buffer instead, which are submitted and waited for whenever the buffer is full,
// and submitted before the submission itself (so its images are complete before the GPU reads them; a global barrier ends each flush)
enum { STAGE_SIZE = 128 << 20 };
static struct { VkBuffer b; VkDeviceMemory m; uint8_t *p; VkDeviceSize used; VkCommandPool pool; VkCommandBuffer cb; VkFence fence; int open, init; uint64_t flushes, bytes, wait_ns; } g_stage;
static VkDeviceSize scratch_cap(void) { static VkDeviceSize c = ~0ull; if (c == ~0ull) { const char *e = getenv("DM_VK_SCRATCH_MAX_MB"); c = (VkDeviceSize)(e ? atoi(e) : 256) << 20; } return c; }
int vk_stage_wanted(void) {
    if (!scratch_cap() || g_cur < 0) return 0;
    vksub_t *s = cur(); VkDeviceSize t = 0; for (int i = 0; i < s->nblk; i++) t += s->blk[i].size;
    return t > scratch_cap();
}
static void stage_init(void) {
    if (g_stage.init) return; g_stage.init = 1;
    VkBufferCreateInfo bi = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, NULL, 0, STAGE_SIZE, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_SHARING_MODE_EXCLUSIVE};
    VKCHK(vkCreateBuffer(V.dev, &bi, NULL, &g_stage.b));
    VkMemoryRequirements req; vkGetBufferMemoryRequirements(V.dev, g_stage.b, &req);
    VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL, req.size, vk_mem_type(req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 0)};
    vk_mem_tag = "staging"; VKCHK(vkAllocateMemory(V.dev, &ai, NULL, &g_stage.m)); vk_mem_tag = NULL; VKCHK(vkBindBufferMemory(V.dev, g_stage.b, g_stage.m, 0));
    void *m; VKCHK(vkMapMemory(V.dev, g_stage.m, 0, VK_WHOLE_SIZE, 0, &m)); g_stage.p = m;
    VkCommandPoolCreateInfo ci = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, NULL, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, V.qf};
    VKCHK(vkCreateCommandPool(V.dev, &ci, NULL, &g_stage.pool));
    VkCommandBufferAllocateInfo cai = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, NULL, g_stage.pool, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1};
    VKCHK(vkAllocateCommandBuffers(V.dev, &cai, &g_stage.cb));
    VkFenceCreateInfo fi = {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO}; VKCHK(vkCreateFence(V.dev, &fi, NULL, &g_stage.fence));
    VKLOG("texture uploads past %llu MB of scratch in a submission go through a %d MB staging buffer, submitted and waited for as it fills (DM_VK_SCRATCH_MAX_MB)", (unsigned long long)(scratch_cap() >> 20), STAGE_SIZE >> 20);
}
static void stage_begin(void) {
    if (g_stage.open) return;
    VkCommandBufferBeginInfo bi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, NULL, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT}; VKCHK(vkBeginCommandBuffer(g_stage.cb, &bi));
    g_stage.open = 1; g_stage.used = 0;
}
void vk_stage_flush(void) {
    if (!g_stage.open) return;
    vk_img_barrier_all(g_stage.cb); VKCHK(vkEndCommandBuffer(g_stage.cb));
    VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO, NULL, 0, NULL, NULL, 1, &g_stage.cb, 0, NULL};
    pthread_mutex_lock(&g_vk_queue_mu); VkResult r = vkQueueSubmit(V.q, 1, &si, g_stage.fence); pthread_mutex_unlock(&g_vk_queue_mu);
    if (r) die("[vk] staging vkQueueSubmit: %d (device lost?)", r);
    struct timespec t0, t1; clock_gettime(CLOCK_MONOTONIC, &t0);
    VKCHK(vkWaitForFences(V.dev, 1, &g_stage.fence, VK_TRUE, ~0ull)); VKCHK(vkResetFences(V.dev, 1, &g_stage.fence));
    clock_gettime(CLOCK_MONOTONIC, &t1); g_stage.wait_ns += (uint64_t)((t1.tv_sec - t0.tv_sec) * 1000000000ll + (t1.tv_nsec - t0.tv_nsec));
    g_stage.open = 0; g_stage.flushes++;
    static int nl; if (nl++ < 8 || g_stage.flushes % 64 == 0) VKLOG("staging: flush %llu (%.1f MB; %.0f MB uploaded through it so far, %.1f ms waited)", (unsigned long long)g_stage.flushes, g_stage.used / 1048576.0, g_stage.bytes / 1048576.0, g_stage.wait_ns / 1e6);
}
VkCommandBuffer vk_stage_cmd(void) { stage_init(); stage_begin(); return g_stage.cb; }
void *vk_stage_alloc(VkDeviceSize n, VkDeviceSize align, VkBuffer *buf, VkDeviceSize *off) {
    if (n > STAGE_SIZE) return vk_scratch(n, align, buf, off);   // (recorded into the staging command buffer all the same: the scratch lives until the submission's slot recycles)
    stage_init(); stage_begin(); if (!align) align = 16;
    VkDeviceSize o = (g_stage.used + align - 1) & ~(align - 1);
    if (o + n > STAGE_SIZE) { vk_stage_flush(); stage_begin(); o = 0; }
    g_stage.used = o + n; g_stage.bytes += n; *buf = g_stage.b; *off = o; return g_stage.p + o;
}
// Raw global memory (gfx_vk.h RK_GMEM_MAP): a shader that loads or stores at an address it read from memory, which the translator could tie to no storage buffer (Ryujinx
// GlobalToStorage.TryGenerateRawStorageOp), finds the pool holding the address in this table and goes through a PhysicalStorageBuffer pointer to the pool buffer's device address.
// Donkey Kong Bananza: gsys's vertex shader c9a56f796ea4a4cd writes {count 0, instances 1} into each terrain indirect draw record whose address its storage buffer lists, every frame
// before the culling geometry shader refills them; with the stores dropped the terrain's walls and floor were never drawn (instance count 0).
// Layout (dmsh_abi.h DMSH_GMEM_MAP_*, as nvn_mtl.m gmem_map builds it for Metal): DMSH_GMEM_MAP_ENTRIES entries of DMSH_GMEM_MAP_ENTRY bytes, {start, end (exclusive), delta} as
// 64-bit words then a 32-bit hint word and 4 bytes of padding, sorted by start; entry 0 is empty (0, 0), unused entries start at ~0. The pools (every chunk of a game pool is a buffer
// of its own, g_pool_ord lists them by lo) go in as disjoint ranges: a chunk overlapping an earlier one keeps only its part past that one's end. delta = the buffer's device address
// minus the chunk's guest address lo, so guest address + delta is the GPU address of the same bytes, in a zero-copy pool (the same memory) and in a copied one (its GPU copy, as the
// pre-submission sync left it). Each 2 MiB chunk (DMSH_GMEM_HINT_SHIFT) lying wholly inside an entry gets a hint in entry (chunk & (entries - 1)): the chunk in the high bits, the
// entry's index in the low DMSH_GMEM_MAP_LOG2 bits, 0 = none; the shader checks the tag and searches on a collision (GlobalToStorage.cs). Rebuilt when a pool comes or goes
// (vk_pool_gen); the replaced table is freed once the GPU has passed the current submission, which may still read it. Host-visible, written by the CPU before the submission.
#include "dmsh_abi.h"
#include "../gfx/nvn/nvn_timestamp.h"
#define GMEM_MAP_LOG2 13   // GlobalToStorage.GmemMapLog2: log2 of DMSH_GMEM_MAP_ENTRIES
_Static_assert(DMSH_GMEM_MAP_ENTRIES == 1 << GMEM_MAP_LOG2, "gmem table size");
static VkBuffer g_gmem_buf; static VkDeviceMemory g_gmem_mem; static uint32_t g_gmem_gen; static int g_gmem_n;
VkBuffer vk_gmem_map(VkDeviceSize *len) {
    *len = (VkDeviceSize)DMSH_GMEM_MAP_ENTRIES * DMSH_GMEM_MAP_ENTRY;
    if (g_gmem_buf && g_gmem_gen == vk_pool_gen()) return g_gmem_buf;
    pools_back_addressed();   // (lazy pools: a device address for every pool a raw pointer may lie in)
    VkBuffer b; VkDeviceMemory m; uint8_t *t;
    { VkBufferCreateInfo bi = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, NULL, 0, *len, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_SHARING_MODE_EXCLUSIVE};
      VKCHK(vkCreateBuffer(V.dev, &bi, NULL, &b)); VkMemoryRequirements req; vkGetBufferMemoryRequirements(V.dev, b, &req);
      VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL, req.size, vk_mem_type(req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 0)};
      vk_mem_tag = "gmem-table"; VKCHK(vkAllocateMemory(V.dev, &ai, NULL, &m)); vk_mem_tag = NULL; VKCHK(vkBindBufferMemory(V.dev, b, m, 0));
      void *mp; VKCHK(vkMapMemory(V.dev, m, 0, VK_WHOLE_SIZE, 0, &mp)); t = mp; }
    memset(t, 0, *len);
    for (int k = 1; k < DMSH_GMEM_MAP_ENTRIES; k++) memset(t + (size_t)k * DMSH_GMEM_MAP_ENTRY, 0xff, 8);
    int n = 1, over = 0, noaddr = 0, unbacked = 0; uint64_t end = 0; uint32_t gen;
    pthread_rwlock_rdlock(&g_pool_mu);
    gen = __atomic_load_n(&g_pool_gen, __ATOMIC_ACQUIRE);
    for (int k = 0; k < g_npool; k++) {
        const vkpool_t *p = &g_pool[g_pool_ord[k]];
        if (!p->gpu) { if (p->backed) noaddr++; else unbacked++; continue; }   // (a lazy pool the game took no address in: no raw pointer can lie in it)
        uint64_t lo = p->lo > end ? p->lo : end, hi = p->hi;
        if (lo >= hi) continue;
        if (n == DMSH_GMEM_MAP_ENTRIES) { over++; continue; }
        uint64_t e[3] = {lo, hi, p->gpu - p->lo}; memcpy(t + (size_t)n++ * DMSH_GMEM_MAP_ENTRY, e, sizeof e); end = hi;
    }
    pthread_rwlock_unlock(&g_pool_mu);
    // the hints: each 2 MiB chunk wholly inside an entry, with the chunk as a tag (a later entry's colliding chunk replaces it; the shader's tag check sends the other to the search).
    // Chunks at or above 1 << (32 - GMEM_MAP_LOG2) do not fit the tag, and the shader does not try the hint for an address that high (addrHigh >= 1 << (DMSH_GMEM_HINT_SHIFT - GMEM_MAP_LOG2))
    for (int k = 1; k < n; k++) {
        const uint64_t *e = (const uint64_t *)(t + (size_t)k * DMSH_GMEM_MAP_ENTRY);
        uint64_t size = 1ull << DMSH_GMEM_HINT_SHIFT, first = (e[0] + size - 1) >> DMSH_GMEM_HINT_SHIFT, last = e[1] >> DMSH_GMEM_HINT_SHIFT;
        for (uint64_t c = first; c < last && c < (1ull << (32 - GMEM_MAP_LOG2)); c++) {
            uint32_t hint = (uint32_t)(c << GMEM_MAP_LOG2 | (unsigned)k);
            memcpy(t + (size_t)(c % DMSH_GMEM_MAP_ENTRIES) * DMSH_GMEM_MAP_ENTRY + 24, &hint, sizeof hint);
        }
    }
    if (over) VKRED_ONCE(140, "raw global memory: %d more memory pool chunks than the %d entries of the pool table; addresses in them are not reached", over, DMSH_GMEM_MAP_ENTRIES - 1);
    if (noaddr) VKRED_ONCE(141, "raw global memory: %d memory pool chunks have no device address; addresses in them are not reached", noaddr);
    if (g_gmem_buf) vk_defer_free_buffer(g_gmem_buf, g_gmem_mem);
    g_gmem_buf = b; g_gmem_mem = m; g_gmem_gen = gen; g_gmem_n = n - 1;
    { static int said; if (said++ < 4) VKLOG("raw global memory: pool table of %d ranges (pool generation %u; %d chunks the game took no address in left out), rebuilt whenever a pool comes or goes", n - 1, gen, unbacked); }
    return b;
}
// The deferred frees have a lock of their own: the replay thread adds to its slot's list while it records (holding g_sub_lock for the whole submission), and a game
// thread that finalizes a pool (nvnMemoryPoolFinalize → mtl_pool_del) adds to the same list without g_sub_lock. Unlocked, two first adds to a recycled slot crashed
// Animal Crossing: New Horizons' release build (SIGSEGV in df_add: one thread had set capdf, the other wrote through the df still NULL). g_df_mu is above slot_recycle
static void df_add(int kind, void *a, void *b) {
    pthread_mutex_lock(&g_df_mu);
    if (g_cur >= 0) {
        vksub_t *s = &g_slot[g_cur]; if (s->ndf == s->capdf) { s->capdf = s->capdf ? 2 * s->capdf : 64; s->df = realloc(s->df, s->capdf * sizeof *s->df); }
        s->df[s->ndf++] = (dfree_t){kind, a, b};
        pthread_mutex_unlock(&g_df_mu); return;
    }
    pthread_mutex_unlock(&g_df_mu);
    {   // outside a submission (a pool finalized between frames): after everything submitted so far
        wait_fence(g_fence);
        if (kind == 0) { vkDestroyBuffer(V.dev, (VkBuffer)a, NULL); if (b) vkFreeMemory(V.dev, (VkDeviceMemory)b, NULL); }
        else if (kind == 1) { vkDestroyImage(V.dev, (VkImage)a, NULL); if (b) vkFreeMemory(V.dev, (VkDeviceMemory)b, NULL); }
        else if (kind == 4) { void img_sub_free_(void *); vkDestroyImage(V.dev, (VkImage)a, NULL); img_sub_free_(b); }
        else if (kind == 2) vkDestroyImageView(V.dev, (VkImageView)a, NULL);
        else vkDestroyBufferView(V.dev, (VkBufferView)a, NULL);
    }
}
// Image memory. Small images share blocks: each allocation the driver makes is a mapping of the process (Adreno maps them all), and a game with thousands of textures
// ran towards the kernel's per-process mapping limit (Xenoblade Chronicles 3: about 9,400 driver mappings in its first area, 2,216 of them 172 KB). Images up to
// IMG_SLOT_MAX take a slot of a block (64 slots, 2 MB to IMG_BLOCK bytes), in size classes 2^(k/2) x 4 KB apart (a slot is at most 41% larger than the image); larger ones, or
// ones that need more than 4 KB alignment, get memory of their own
enum { IMG_CLASSES = 18, IMG_BLOCK = 32 << 20 };
#define IMG_SLOT_MAX ((VkDeviceSize)2 << 20)
typedef struct imgblk { VkDeviceMemory m; uint32_t type, cls, nslot, nfree; uint64_t bits[128]; struct imgblk *next; } imgblk;   // bits: 1 = slot free
typedef struct { imgblk *b; uint32_t slot; } imgsub;
static imgblk *g_imgblk[IMG_CLASSES]; static pthread_mutex_t g_imgblk_mu = PTHREAD_MUTEX_INITIALIZER;
static VkDeviceSize img_class_size(int c) { return (c & 1 ? 6144ull : 4096ull) << (c >> 1); }   // 4K, 6K, 8K, 12K, 16K, 24K, ... 2M
VkDeviceMemory vk_image_memory(const VkMemoryRequirements *req, uint32_t type, VkDeviceSize *off, void **sub) {
    *off = 0; *sub = NULL;
    int c = 0; while (c < IMG_CLASSES && img_class_size(c) < req->size) c++;
    if (c == IMG_CLASSES || req->size > IMG_SLOT_MAX || req->alignment > 4096) {   // memory of its own
        VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL, req->size, type}; VkDeviceMemory m;
        vk_mem_tag = "image"; VkResult r = vkAllocateMemory(V.dev, &ai, NULL, &m); vk_mem_tag = NULL; return r ? VK_NULL_HANDLE : m;
    }
    pthread_mutex_lock(&g_imgblk_mu);
    imgblk *b = g_imgblk[c]; while (b && (b->type != type || !b->nfree)) b = b->next;
    if (!b) {
        b = calloc(1, sizeof *b); b->type = type; b->cls = (uint32_t)c; VkDeviceSize bs = img_class_size(c) * 64; if (bs < (2 << 20)) bs = 2 << 20; if (bs > IMG_BLOCK) bs = IMG_BLOCK; b->nslot = (uint32_t)(bs / img_class_size(c));   // 64 slots, 2 MB to 32 MB
        VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL, (VkDeviceSize)b->nslot * img_class_size(c), type};
        vk_mem_tag = "image-block"; VkResult r = vkAllocateMemory(V.dev, &ai, NULL, &b->m); vk_mem_tag = NULL; if (r) { pthread_mutex_unlock(&g_imgblk_mu); free(b); return VK_NULL_HANDLE; }
        for (uint32_t i = 0; i < b->nslot; i++) b->bits[i >> 6] |= 1ull << (i & 63);
        b->nfree = b->nslot; b->next = g_imgblk[c]; g_imgblk[c] = b;
    }
    uint32_t i = 0; while (!b->bits[i >> 6]) i += 64; i += (uint32_t)__builtin_ctzll(b->bits[i >> 6]);
    b->bits[i >> 6] &= ~(1ull << (i & 63)); b->nfree--;
    pthread_mutex_unlock(&g_imgblk_mu);
    imgsub *s = malloc(sizeof *s); s->b = b; s->slot = i; *sub = s; *off = (VkDeviceSize)i * img_class_size(c);
    return b->m;
}
static void img_sub_free(imgsub *s) {   // the slot is free again; blocks are kept (the next images of the class take them)
    pthread_mutex_lock(&g_imgblk_mu); s->b->bits[s->slot >> 6] |= 1ull << (s->slot & 63); s->b->nfree++; pthread_mutex_unlock(&g_imgblk_mu); free(s);
}
void vk_defer_free_buffer(VkBuffer b, VkDeviceMemory m) { df_add(0, (void *)b, (void *)m); }
void vk_defer_free_image_sub(VkImage i, void *sub) { df_add(4, (void *)i, sub); }
void img_sub_free_(void *s) { img_sub_free((imgsub *)s); }
void vk_defer_free_image(VkImage i, VkDeviceMemory m) { df_add(1, (void *)i, (void *)m); }
void vk_defer_free_view(VkImageView v) { df_add(2, (void *)v, NULL); }
void vk_defer_free_bview(VkBufferView v) { df_add(3, (void *)v, NULL); }
// The process's GPU memory as the kernel maps it (sysfs; 0 when unreadable): what the driver allocates on its own (descriptor pools, command memory) shows here and nowhere in vkAllocateMemory
static uint64_t kgsl_mapped(void) {
    static int fd = -2; if (fd == -2) { char path[96]; snprintf(path, sizeof path, "/sys/class/kgsl/kgsl/proc/%d/gpumem_mapped", (int)getpid()); fd = open(path, O_RDONLY | O_CLOEXEC); }
    if (fd < 0) return 0; char buf[32]; ssize_t n = pread(fd, buf, sizeof buf - 1, 0); if (n <= 0) return 0; buf[n] = 0; return strtoull(buf, NULL, 10);
}
uint64_t vk_kgsl_mapped(void) { return kgsl_mapped(); }
// Descriptor pools: the allocator counts what each set takes and moves to a new pool before any count runs out. Adreno's driver does not refuse an
// allocation past a pool's sizes: it succeeds and the submission then fails as a whole at vkEndCommandBuffer (VK_ERROR_OUT_OF_HOST_MEMORY; NEO's
// handheld frames with 32-slot texture arrays crossed the old 8192 combined-sampler limit at about the 712th draw)
// A pool's memory is the driver's, sized by its capacities whatever the sets use (Adreno 830: about 12 MB for the old capacities below, in kgsl as "VK/others(37)").
// Nintendo Switch 2 Welcome Tour's first scene load records one submission with hundreds of thousands of sets: 217 pools, 2.6 GB, and the low-memory killer.
// The allocator moves to a new pool when a set's needs would cross a capacity (vk_dsl_register), so the capacities can be modest: a set of a layout with
// more of a type than a pool holds just fills pools faster. DM_VK_DP_BIG=1: the old capacities. Pools beyond DP_KEEP per slot are destroyed when the slot
// recycles (slot_recycle), so a load's pile of them goes with the load
enum { DP_SETS = 4096 };
static const uint32_t DP_CAP_BIG[DP_TYPES] = {[VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC] = 16384, [VK_DESCRIPTOR_TYPE_SAMPLER] = 1024, [VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER] = 65536, [VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE] = 1024,
    [VK_DESCRIPTOR_TYPE_STORAGE_IMAGE] = 2048, [VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER] = 4096, [VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER] = 2048,
    [VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER] = 32768, [VK_DESCRIPTOR_TYPE_STORAGE_BUFFER] = 4096, [VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC] = 4096};
static const uint32_t DP_CAP_SMALL[DP_TYPES] = {[VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC] = 4096, [VK_DESCRIPTOR_TYPE_SAMPLER] = 512, [VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER] = 8192, [VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE] = 512,
    [VK_DESCRIPTOR_TYPE_STORAGE_IMAGE] = 512, [VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER] = 1024, [VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER] = 512,
    [VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER] = 8192, [VK_DESCRIPTOR_TYPE_STORAGE_BUFFER] = 2048, [VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC] = 2048};
static const uint32_t *DP_CAP = DP_CAP_BIG;   // (DM_VK_DP_SMALL=1: the small ones; a pool costs the same 11.8 MB either way on the Adreno 830, so the big ones fill slower)
static uint32_t dp_pool_sizes(VkDescriptorPoolSize *sz) {   // the shared pools' sizes: every capacity, dynamic storage only when its descriptors are in use (DM_VK_DYN_SSBO)
    uint32_t nz = 0; for (int t = 0; t < DP_TYPES; t++) if (DP_CAP[t] && (t != VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC || V.dyn_ssbo)) sz[nz++] = (VkDescriptorPoolSize){(VkDescriptorType)t, DP_CAP[t]};
    return nz;
}
static nvn_hmap g_dsl_need = {.klen = sizeof(VkDescriptorSetLayout)}; static pthread_mutex_t g_dsl_mu = PTHREAD_MUTEX_INITIALIZER;
void vk_dsl_register(VkDescriptorSetLayout l, const VkDescriptorSetLayoutBinding *b, uint32_t n) {   // what one set of this layout takes, by type
    uint32_t *need = calloc(DP_TYPES, sizeof *need);
    for (uint32_t i = 0; i < n; i++) if ((uint32_t)b[i].descriptorType < DP_TYPES) need[b[i].descriptorType] += b[i].descriptorCount;
    pthread_mutex_lock(&g_dsl_mu); nvn_hm_put(&g_dsl_need, &l, need); pthread_mutex_unlock(&g_dsl_mu);
}
// Allocate exactly one set; speculative batches cost more on Adreno 830 even with fewer driver calls.
// Registered need arrays are immutable heap allocations: hashmap growth only moves its pointer table.
static VkDescriptorSet descriptor_alloc_set(vksub_t *s, VkDescriptorSetLayout l) {
    pthread_mutex_lock(&g_dsl_mu); const uint32_t *need = nvn_hm_get(&g_dsl_need, &l); pthread_mutex_unlock(&g_dsl_mu);
    int huge = 0; if (need) for (int t = 0; t < DP_TYPES; t++) if (need[t] > DP_CAP[t]) huge = 1;
    if (huge) { VkDescriptorSet reused = descriptor_large_take(s, l); if (reused) return reused;
        // no shared pool ever holds a set of this layout: before this check the loop below made a new 11.8 MB pool per try, for ever (Nintendo Switch 2 Welcome
        // Tour's first scene: 700 pools and 8 GB of GPU memory in 5 s, then the low-memory killer; the game's bindless texture layouts ask for more combined samplers than a pool's capacity)
        VkDescriptorPoolSize sz[DP_TYPES]; uint32_t nz = 0; for (int t = 0; t < DP_TYPES; t++) if (need[t]) sz[nz++] = (VkDescriptorPoolSize){(VkDescriptorType)t, need[t]};
        VkDescriptorPoolCreateInfo pi = {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, NULL, g_uab ? VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT : 0, 1, nz, sz};
        if (s->ndpx == s->capdpx) { s->capdpx = s->capdpx ? 2 * s->capdpx : 16; s->dpx = realloc(s->dpx, s->capdpx * sizeof *s->dpx); }
        VkDescriptorPool dp; VKCHK(vkCreateDescriptorPool(V.dev, &pi, NULL, &dp));
        VkDescriptorSetAllocateInfo ai = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, NULL, dp, 1, &l}; VkDescriptorSet set; g_desc_alloc_calls++; VkResult r = vkAllocateDescriptorSets(V.dev, &ai, &set);
        if (r) die("[vk] vkAllocateDescriptorSets (a pool of its own): %d", r);
        uint64_t descriptors = 0; for (int t = 0; t < DP_TYPES; t++) descriptors += need[t];
        s->dpx[s->ndpx++] = (dpx_entry){dp, set, l, descriptors, 1, ++g_descriptor_lease, 0, 0};
        g_desc_alloc_sets++;
        static uint64_t n; if (++n <= 4 || !(n & (n - 1))) VKLOG("descriptor set %llu with a pool of its own: layout needs ubo %u / ubo-dyn %u / ssbo %u / ssbo-dyn %u / sampler %u / combined %u / sampled %u / storage-image %u / utb %u / stb %u (a shared pool holds %u combined; %d such pools in this slot)",
            (unsigned long long)n, need[VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER], need[VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC], need[VK_DESCRIPTOR_TYPE_STORAGE_BUFFER], need[VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC], need[VK_DESCRIPTOR_TYPE_SAMPLER], need[VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER], need[VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE], need[VK_DESCRIPTOR_TYPE_STORAGE_IMAGE], need[VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER], need[VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER], DP_CAP[VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER], s->ndpx);
        return set;
    }
    for (;;) {
        if (s->dp_cur == s->ndp) {
            VkDescriptorPoolSize sz[DP_TYPES]; uint32_t nz = dp_pool_sizes(sz);
            VkDescriptorPoolCreateInfo pi = {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, NULL, g_uab ? VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT : 0, DP_SETS, nz, sz};
            if (s->ndp == s->capdp) { s->capdp = s->capdp ? 2 * s->capdp : 4; s->dp = realloc(s->dp, s->capdp * sizeof *s->dp); }
            { static int chosen; if (!chosen) { chosen = 1; const char *e = getenv("DM_VK_DP_SMALL"); if (e && atoi(e)) { DP_CAP = DP_CAP_SMALL; pi.poolSizeCount = dp_pool_sizes(sz); } } }
            uint64_t k0 = kgsl_mapped();
            VKCHK(vkCreateDescriptorPool(V.dev, &pi, NULL, &s->dp[s->ndp++])); g_dp_made++;
            extern uint64_t g_set_new;
            if (g_dp_made <= 8 || g_dp_made % 32 == 0 || s->ndp % 16 == 0) VKLOG("descriptor pool %llu made (%d in this slot; %llu freed): %.1f MB of GPU memory for it, %.0f MB mapped in all; %llu new sets so far", (unsigned long long)g_dp_made, s->ndp, (unsigned long long)g_dp_freed, (kgsl_mapped() - k0) / 1048576.0, kgsl_mapped() / 1048576.0, (unsigned long long)g_set_new);
        }
        if (s->dp_cur != s->dp_counted) { memset(s->dp_used, 0, sizeof s->dp_used); s->dp_sets = 0; s->dp_counted = s->dp_cur; }
        int fits = s->dp_sets < DP_SETS;
        if (need) for (int t = 0; t < DP_TYPES; t++) if (need[t] > DP_CAP[t] - s->dp_used[t]) fits = 0;
        if (!fits) { s->dp_cur++; continue; }
        VkDescriptorSet set;
        VkDescriptorSetAllocateInfo ai = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, NULL, s->dp[s->dp_cur], 1, &l};
        g_desc_alloc_calls++; VkResult r = vkAllocateDescriptorSets(V.dev, &ai, &set);
        if (r == VK_SUCCESS) {
            s->dp_sets++; if (need) for (int t = 0; t < DP_TYPES; t++) s->dp_used[t] += need[t];
            g_desc_alloc_sets++; return set;
        }
        if (r != VK_ERROR_OUT_OF_POOL_MEMORY && r != VK_ERROR_FRAGMENTED_POOL) die("[vk] vkAllocateDescriptorSets: %d", r);
        s->dp_cur++;
    }
}
// n sets of layout l in a descriptor pool of their own, never reset or freed (gfx_vk_exec.c's ring of pool tables: they outlive submissions). 0 when the layout is unknown or the pool fails
int vk_alloc_sets_owned(VkDescriptorSetLayout l, int n, VkDescriptorSet *out) {
    pthread_mutex_lock(&g_dsl_mu); const uint32_t *need = nvn_hm_get(&g_dsl_need, &l); pthread_mutex_unlock(&g_dsl_mu);
    if (!need || n <= 0 || n > 16) return 0;
    VkDescriptorPoolSize sz[DP_TYPES]; uint32_t nz = 0; for (int t = 0; t < DP_TYPES; t++) if (need[t]) sz[nz++] = (VkDescriptorPoolSize){(VkDescriptorType)t, need[t] * (uint32_t)n};
    VkDescriptorPoolCreateInfo pi = {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, NULL, g_uab ? VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT : 0, (uint32_t)n, nz, sz};
    VkDescriptorPool dp; if (vkCreateDescriptorPool(V.dev, &pi, NULL, &dp)) return 0;
    VkDescriptorSetLayout ls[16]; for (int i = 0; i < n; i++) ls[i] = l;
    VkDescriptorSetAllocateInfo ai = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, NULL, dp, (uint32_t)n, ls};
    if (vkAllocateDescriptorSets(V.dev, &ai, out)) { vkDestroyDescriptorPool(V.dev, dp, NULL); return 0; }
    return 1;
}
VkDescriptorSet vk_alloc_set(VkDescriptorSetLayout l) {
    if (!g_desc_timing) return descriptor_alloc_set(cur(), l);
    uint64_t start = mem_diag_ns(); VkDescriptorSet set = descriptor_alloc_set(cur(), l);
    g_desc_alloc_ns += mem_diag_ns() - start;
    return set;
}

// Immutable-content leases are scoped to the current slot. Normal allocation always changes the
// lease before its caller rewrites descriptors. A cached handle alone is insufficient: the driver
// may recycle it after a pool is destroyed. Only completed-slot, unused sets can be claimed.
uint64_t vk_descriptor_lease(VkDescriptorSet set, int rewrite) {
    vksub_t *s = cur();
    for (int i = 0; i < s->ndpx && i < DPX_KEEP; i++) if (s->dpx[i].set == set && s->dpx[i].used) {
        if (rewrite) { s->dpx[i].lease = ++g_descriptor_lease; s->dpx[i].cached = 0; }
        else s->dpx[i].cached = 1;
        return s->dpx[i].lease;
    }
    return 0;
}
int vk_descriptor_claim(VkDescriptorSet set, uint64_t lease) {
    if (!lease || !descriptor_large_reuse()) return 0;
    vksub_t *s = cur();
    for (int i = 0; i < s->ndpx && i < DPX_KEEP; i++) {
        dpx_entry *e = &s->dpx[i];
        if (e->set == set && e->lease == lease && !e->used) { e->used = 1; return 1; }
    }
    return 0;
}

// ---------------- submissions: begin, CPU-visible effects, commit ----------------
static void post(post_op o) { vksub_t *s = cur(); if (s->npost == s->cappost) { s->cappost = s->cappost ? 2 * s->cappost : 64; s->post = realloc(s->post, s->cappost * sizeof *s->post); } s->post[s->npost++] = o; }
// Samples-passed on NVN is cumulative on the queue. Vulkan queries are per render pass, so completion replays ordered
// segment-add / reset / report events. A report closes the current segment without resetting the cumulative value.
// This matters for numeric consumers: Bananza divides five cube-face counts by five times the face area to derive a
// lighting visibility ratio. The old 1<<20 placeholder became 64 for 128x128 faces and multiplied the skin's red SSS term.
enum { OCC_BLOCK = 256 };
static int g_occ_counting, g_occ_active = -1;   // replay thread, under g_sub_lock; active belongs to the current submission
static uint64_t g_occ_completed;              // completion thread only, across submissions
static uintptr_t g_occ_first_site, g_occ_last_site;
static uint32_t g_occ_draw_calls;              // issued calls, not GPU indirect vertex/index counts
int vk_occ_counting(void) { return g_occ_counting; }   // query effects outlive each pass and report until the next reset
void vk_occ_draw(void) {
    if (!g_occ_counting) return;
    uintptr_t site = (uintptr_t)__builtin_extract_return_addr(__builtin_return_address(0));
    g_occ_last_site = site;
    if (g_occ_active >= 0) { if (g_occ_draw_calls != UINT32_MAX) g_occ_draw_calls++; return; }
    g_occ_first_site = site; g_occ_draw_calls = 1;
    vksub_t *s = cur(); int q = s->nocc, block = q / OCC_BLOCK;
    typeof(s->occ[0]) *trace = &s->occ[s->nmain_used - 1];
    if (!trace->begins++) trace->first_q = (uint32_t)q;
    trace->last_q = (uint32_t)q;
    if (!(q % OCC_BLOCK)) {
        if (block == s->noq) {
            if (s->noq == s->capoq) { s->capoq = s->capoq ? 2 * s->capoq : 4; s->oq = realloc(s->oq, (size_t)s->capoq * sizeof *s->oq); if (!s->oq) die("[vk] occlusion pool allocation"); }
            VkQueryPoolCreateInfo qi = {VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO, NULL, 0, VK_QUERY_TYPE_OCCLUSION, OCC_BLOCK};
            VKCHK(vkCreateQueryPool(V.dev, &qi, NULL, &s->oq[s->noq++]));
        }
        // Upload commands execute before this main command buffer. Reset outside the render pass, once per block,
        // including a block first allocated by a draw halfway through a pass or a long submission's later part.
        VkCommandBuffer up = vk_upload_cmd();
        if (!trace->resets++) trace->first_block = (uint32_t)block;
        trace->last_block = (uint32_t)block; trace->reset_cb = up;
        vkCmdResetQueryPool(up, s->oq[block], 0, OCC_BLOCK);
    }
    g_occ_active = s->nocc++;
    vkCmdBeginQuery(s->main, s->oq[block], (uint32_t)(q % OCC_BLOCK), V.has_occ_precise ? VK_QUERY_CONTROL_PRECISE_BIT : 0);
}
void vk_occ_end(void) {
    if (g_occ_active < 0) return;
    vksub_t *s = cur(); int q = g_occ_active;
    vkCmdEndQuery(s->main, s->oq[q / OCC_BLOCK], (uint32_t)(q % OCC_BLOCK));
    s->occ[s->nmain_used - 1].ends++;
    // The otherwise unused kind-3 fields retain failure-only draw metadata; no allocation or normal-path logging.
    post((post_op){.kind = 3, .val = (volatile uint32_t *)g_occ_last_site, .v = (uint32_t)(s->nmain_used - 1),
                   .addr = g_occ_first_site, .type = g_occ_draw_calls, .q = q});
    g_occ_active = -1;
}
void vk_occ_reset(void) {
    vk_occ_end();
    post((post_op){.kind = 4, .v = (uint32_t)(cur()->nmain_used - 1), .q = -1});
    g_occ_counting = 1;
    static int said; if (!said++) VKLOG("samples-passed counters: GPU occlusion queries, %s", V.has_occ_precise ? "precise counts" : "[RED] device lacks precise counts; nonzero results are approximate");
}
// Run only after the owning submission has completed. One result per segment; no wait or fabricated substitute.
// The submission slot is held until completion consumes every event. A dropped command-buffer part contributes none.
static void occ_failure(vksub_t *s, const post_op *o, int submitted_parts, uint64_t frame, uint64_t fence, VkResult result) {
    int begin_part = -1, reset_part = -1; uint32_t begins = 0, ends = 0, resets = 0;
    for (int i = 0; i < s->nmain_used && i < MAIN_CBS; i++) {
        const typeof(s->occ[0]) *t = &s->occ[i]; begins += t->begins; ends += t->ends; resets += t->resets;
        if (t->begins && (uint32_t)o->q >= t->first_q && (uint32_t)o->q <= t->last_q) begin_part = i;
        if (t->resets && (uint32_t)o->q / OCC_BLOCK >= t->first_block && (uint32_t)o->q / OCC_BLOCK <= t->last_block) reset_part = i;
    }
    VKLOG("[RED] samples query failure: result %d frame %llu fence %llu slot %d (slot fence %llu), query %d pool %#llx; parts submitted %d recorded %d sent %d, uploads %d; queries %d begins %u ends %u block resets %u; begin/reset/end parts %d/%d/%u",
          result, (unsigned long long)frame, (unsigned long long)fence, (int)(s - g_slot), (unsigned long long)s->fence,
          o->q, (unsigned long long)(uintptr_t)s->oq[o->q / OCC_BLOCK], submitted_parts, s->nmain_used, s->nmain_sent, s->nup_used,
          s->nocc, begins, ends, resets, begin_part, reset_part, o->v);
    VKLOG("  query draw calls %u, first/last call sites libdmhost.so+%#llx/+%#llx (base %#llx)",
          o->type, (unsigned long long)(o->addr - (uintptr_t)g_lib_base),
          (unsigned long long)((uintptr_t)o->val - (uintptr_t)g_lib_base), (unsigned long long)(uintptr_t)g_lib_base);
    // At most three part lines and five neighboring availability reads. These reads are diagnostic only:
    // even a now-available result does not replace the failed read or prevent the caller's fatal error.
    int first = o->q > 2 ? o->q - 2 : 0, last = o->q + 2;
    if (last >= s->nocc) last = s->nocc - 1;
    for (int q = first; q <= last; q++) {
        // A neighbor may belong to a dropped part and even a newly created, never-reset pool.
        // Vulkan forbids reading uninitialized queries, including for diagnostic availability.
        int used = 0, reset = 0;
        for (int i = 0; i < submitted_parts && i < s->nmain_used && i < MAIN_CBS; i++) {
            const typeof(s->occ[0]) *t = &s->occ[i];
            if (t->begins && (uint32_t)q >= t->first_q && (uint32_t)q <= t->last_q) used = 1;
            if (t->resets && (uint32_t)q / OCC_BLOCK >= t->first_block && (uint32_t)q / OCC_BLOCK <= t->last_block) reset = 1;
        }
        if (!used || !reset) continue;
        uint64_t value[2] = {UINT64_MAX, UINT64_MAX};
        VkResult r = vkGetQueryPoolResults(V.dev, s->oq[q / OCC_BLOCK], (uint32_t)(q % OCC_BLOCK), 1,
                                           sizeof value, value, sizeof value, VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT);
        VKLOG("  query availability %d: result %d samples %llu available %llu%s", q, r,
              (unsigned long long)value[0], (unsigned long long)value[1], q == o->q ? " (failed query)" : "");
    }
    for (int i = 0; i < s->nmain_used && i < MAIN_CBS; i++) if (i == begin_part || i == reset_part || (uint32_t)i == o->v) {
        const typeof(s->occ[0]) *t = &s->occ[i];
        VKLOG("  query part %d: main %#llx begins %u [%u..%u] ends %u; resets %u blocks [%u..%u] upload %#llx, %s",
              i, (unsigned long long)(uintptr_t)s->mains[i], t->begins, t->first_q, t->last_q, t->ends,
              t->resets, t->first_block, t->last_block, (unsigned long long)(uintptr_t)t->reset_cb, i < submitted_parts ? "submitted" : "dropped");
    }
}
static void occ_complete(vksub_t *s, const post_op *o, int submitted_parts, uint64_t frame, uint64_t fence) {
    if (o->v >= (uint32_t)submitted_parts) return;
    if (o->kind == 4) { g_occ_completed = 0; return; }
    uint64_t samples = 0; int q = o->q;
    VkResult r = vkGetQueryPoolResults(V.dev, s->oq[q / OCC_BLOCK], (uint32_t)(q % OCC_BLOCK), 1, sizeof samples, &samples, sizeof samples, VK_QUERY_RESULT_64_BIT);
    // A query whose result never became available (a few per submission part on the Odin 3, Ocarina of Time) counts as visible instead of ending the game: the condition
    // that reads it then draws, as a counter the host does not know does (render_cond). The first failures keep their full diagnostics
    if (r != VK_SUCCESS) { static unsigned failures; if (failures++ < 3) occ_failure(s, o, submitted_parts, frame, fence, r); else if (!(failures & (failures - 1))) VKLOG("[RED] samples query unavailable at completion: %u so far, counted as visible", failures); samples = 1; }
    g_occ_completed += samples;
}
void mtl_init(void) { vk_init(); }
static int g_shadow_pools;
static void pools_readback_barrier(VkCommandBuffer cb) {
    // Both copied pools and directly mapped AHardwareBuffer pools can contain GPU-written
    // results the guest reads after a fence/event. A fence wait alone does not make device
    // writes visible to host accesses, even when no shadow copy-back is needed.
    // The last part can be dropped after earlier ones were sent; publish every part.
    VkMemoryBarrier b = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, NULL, VK_ACCESS_MEMORY_WRITE_BIT, VK_ACCESS_HOST_READ_BIT};
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &b, 0, NULL, 0, NULL);
}
void mtl_sub_begin(void) {
    vk_init();
    pthread_mutex_lock(&g_sub_lock);
    // the next slot of the ring (g_cur is -1 between submissions, so it cannot say where the ring is: every submission took slot 0 and waited in slot_recycle
    // for the one before to finish on the GPU, which kept the replay of a frame from overlapping the GPU work of the last)
    static int next; g_cur = next; next = (next + 1) % NSLOT; vksub_t *s = &g_slot[g_cur];
    slot_recycle(s);
    cut_stats_begin(s);
    { const char *e = getenv("DM_VK_DESC_TIMING"); g_desc_timing = e && atoi(e); }
    s->npost = 0; s->up_used = 0; s->ngw = 0; s->nocc = 0; g_occ_active = -1; memset(s->occ, 0, sizeof s->occ);
    { const char *e = getenv("DM_VK_POOL_READBACK_RANGES"); s->readback_ranges = e && atoi(e); }
    VkCommandBufferBeginInfo bi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, NULL, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
    s->main = s->mains[0]; s->nmain_used = 1; s->cb_err = 0; s->up = s->ups[0]; s->nup_used = 1; s->nmain_sent = 0;
    { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); s->t_begin = (uint64_t)t.tv_sec * 1000000000ull + (uint64_t)t.tv_nsec; }
    g_sub0.draws = g_vkst.draws; g_sub0.passes = g_vkst.passes; g_sub0.dispatches = g_vkst.dispatches; g_sub0.sets = g_set_new; g_sub0.pkg_n = g_pkg_n; g_sub0.pkg_bytes = g_pkg_bytes; { static unsigned nb; if (!(nb++ & 63)) g_sub0.kgsl = kgsl_mapped(); }   /* a sysfs read: every 64th submission */ g_sub0.pipes = g_vkst.pipes_built; g_sub0.pipe_kgsl = g_pipe_kgsl;
    VKCHK(vkBeginCommandBuffer(s->main, &bi)); g_cmd_gen++;
    // copied pools: what the game wrote before submitting goes into the GPU copy before any part of the submission can reach the GPU (DM_VK_CB_DRAWS sends
    // parts early; synced at the commit, those parts read the copy as the last submission left it), and only once no earlier submission still reads it
    pthread_rwlock_rdlock(&g_pool_mu); int shadow = 0; for (int i = 0; i < g_npool; i++) shadow |= g_pool[i].shadow != NULL; pthread_rwlock_unlock(&g_pool_mu);
    g_shadow_pools = shadow;
    if (shadow) {
        int diag = mem_diag_on(); uint64_t t = diag ? mem_diag_ns() : 0;
        mtl_wait(g_fence, ~0ull); if (diag) mem_diag_note(0, t, 0);
        vk_pools_sync(NULL);
    }   // (after the completion thread's copy-back of the last submission, not only its GPU work)
    if (g_tsq) { vkCmdResetQueryPool(s->main, g_tsq, 2 * (uint32_t)g_cur, 2); vkCmdWriteTimestamp(s->main, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, g_tsq, 2 * (uint32_t)g_cur); }
    if (g_cq) vkCmdResetQueryPool(s->main, g_cq, (uint32_t)g_cur * CQ, CQ);
    s->ncq = 0;
    g_npass[g_cur] = 0; if (pass_times_on()) vkCmdResetQueryPool(s->main, g_pq, (uint32_t)(g_cur * PASSQ * 2), PASSQ * 2);
    vk_exec_reset();
}
void mtl_sub_event(volatile uint32_t *val, uint32_t mode, uint32_t v) { post((post_op){1, val, mode, v, 0, 0, -1}); }
void mtl_sub_counter(uint64_t addr, uint32_t type) {
    if (type == NVN_COUNTER_TYPE_SAMPLES_PASSED) vk_occ_end();
    vksub_t *s = cur(); int q = -1;
    if (g_cq && (type == NVN_COUNTER_TYPE_TIMESTAMP || type == NVN_COUNTER_TYPE_TIMESTAMP_TOP) && s->ncq < CQ) {
        q = g_cur * CQ + s->ncq++;
        // NVN TIMESTAMP_TOP (0x10) samples the front of the pipeline; the ordinary
        // timestamp samples completed work, matching Metal's start/end distinction.
        vkCmdWriteTimestamp(vk_cmd(), type == NVN_COUNTER_TYPE_TIMESTAMP_TOP ? VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT : VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, g_cq, (uint32_t)q);
    }
    post((post_op){2, NULL, 0, 0, addr, type, q});
}
static VkMemoryBarrier2 full_barrier(void) {
    return (VkMemoryBarrier2){VK_STRUCTURE_TYPE_MEMORY_BARRIER_2, NULL, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                              VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT};
}
uint64_t g_vk_nbar, g_vk_nlrz;   // full barriers recorded, LRZ texel touches (lrz_touch): counted for the per-frame pass-work line (gfx_vk_present.c)
void vk_img_barrier_all(VkCommandBuffer cb) {
    g_vk_nbar++;
    VkMemoryBarrier b = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, NULL, VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT, VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT};
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &b, 0, NULL, 0, NULL);
}
void vk_gpu_wrote(uint64_t a, uint64_t n);
void mtl_sub_copy(uint64_t src, uint64_t dst, uint64_t n) {
    VkBuffer sb, db; VkDeviceSize so, dof;
    if (!n) return;
    if (!vk_buf_at(src, n, &sb, &so) || !vk_buf_at(dst, n, &db, &dof)) { VKRED_ONCE(1, "buffer copy outside the memory pools (%#llx → %#llx, %llu bytes)", (unsigned long long)src, (unsigned long long)dst, (unsigned long long)n); return; }
    vk_end_pass_land();
    VkCommandBuffer cb = vk_cmd(); vk_img_barrier_all(cb);
    VkBufferCopy r = {so, dof, n}; vkCmdCopyBuffer(cb, sb, db, 1, &r);
    vk_img_barrier_all(cb); vk_gpu_wrote(dst, n);
}
void mtl_sub_fill(uint64_t dst, uint64_t n, uint32_t value) {
    VkBuffer db; VkDeviceSize dof;
    if (!n) return;
    if (!vk_buf_at(dst, n, &db, &dof)) { VKRED_ONCE(2, "buffer fill outside the memory pools"); return; }
    vk_end_pass_land();
    VkCommandBuffer cb = vk_cmd(); vk_img_barrier_all(cb);
    vkCmdFillBuffer(cb, db, dof, n & ~3ull, value);
    vk_img_barrier_all(cb); vk_gpu_wrote(dst, n);
}
// an inline uniform update (UpdateUniformBuffer): the bytes are taken at record time, copied in GPU order from scratch memory
// Inline uniform updates (UpdateUniformBuffer) are deferred instead of ending the render pass (as nvn_mtl.m's ubd_*): a copy cannot run inside a pass, and
// ending it stores every attachment and loads it again in the next pass, which on a tiler is a round trip of the whole tile memory per update (Unity updates
// before nearly every draw: NEO had about 145 passes for 148 draws a frame). NVN applies the update in command order, so a draw after it whose uniform
// binding overlaps it reads a shadow: memory with the deferred updates applied in order (vk_ubd_shadow). The real copies are made in order when a pass
// ends anyway, and at the end of the submission (vk_ubd_land), so later passes, copies, dispatches and the CPU see them as before.
typedef struct { uint64_t lo, hi; VkBuffer sb; VkDeviceSize so; const uint8_t *p; } ubd_t;
enum { UBD_MAX = 512 };
static ubd_t g_ubd[UBD_MAX]; static int g_nubd; uint64_t g_ubd_gen = 1, g_ubd_landed, g_ubd_shadows, g_ubd_bytes, g_ns_submit, g_ubd_superseded;
void vk_end_pass_land(void);
void vk_sub_update(uint64_t dst, const void *data, uint64_t n) {
    VkBuffer db, sb; VkDeviceSize dof, so;
    if (!n) return;
    if (!vk_buf_at(dst, n, &db, &dof)) { VKRED_ONCE(3, "uniform update outside the memory pools"); return; }
    // an update of exactly the range of a deferred one supersedes it: the older one goes (Unity rewrites the same few constant buffers before every draw,
    // so the list keeps about one entry per buffer region, which keeps the landing to a few copies and the shadows cheap to build)
    for (int i = g_nubd - 1; i >= 0; i--) if (g_ubd[i].lo == dst && g_ubd[i].hi == dst + n) { memmove(&g_ubd[i], &g_ubd[i + 1], sizeof *g_ubd * (size_t)(g_nubd - i - 1)); g_nubd--; g_ubd_superseded++; break; }
    if (g_nubd == UBD_MAX) vk_end_pass_land();   // too many distinct ranges: they land now
    uint8_t *p = vk_scratch(n, 4, &sb, &so); memcpy(p, data, n);
    g_ubd[g_nubd++] = (ubd_t){dst, dst + n, sb, so, p}; g_ubd_gen++;
}
// the draw bisection (gfx_vk_exec.c draw_allowed): the outcome of this submission
extern int g_bis_on, g_bis_lo, g_bis_hi, g_bis_trace, g_bis_var; void vk_bisect_reset_count(void);
static void vk_bisect_result(int ok) {
    static int done, draws_seen; extern uint32_t g_sub_ops[64]; int nd = (int)(g_sub_ops[25] + g_sub_ops[26]);
    if (done) {   // the found draw: one change per submission, then the draw is no longer traced
        if (g_bis_trace >= 0) {
            static const char *what[] = {"as recorded", "depth bias values 0", "no depth bias", "dummy textures", "vertex offset 0", "zero vertex buffer", "in a new pass", "skipped alone (the rest drawn)", "without their fragment shader", "with zero uniform buffers", "with their quad draws skipped", "with all but their quad draws skipped"};
            VKLOG("bisection: draws from %d on %s: %s", g_bis_trace, what[g_bis_var], ok ? "ACCEPTED" : "refused");
            if (++g_bis_var > 11) { g_bis_trace = -1; g_bis_var = 0; }
        }
        vk_bisect_reset_count(); return; }
    if (!g_bis_on) { if (!ok) { g_bis_on = 1; g_bis_lo = 0; g_bis_hi = nd + 1; draws_seen = nd; VKLOG("bisecting the refused submission's %d draws", nd); } vk_bisect_reset_count(); return; }
    int mid = (g_bis_lo + g_bis_hi) / 2;
    if (ok) g_bis_lo = mid; else g_bis_hi = mid;
    if (g_bis_hi - g_bis_lo <= 1) { done = 1; g_bis_on = 0; g_bis_trace = g_bis_lo; VKLOG("bisection: draw %d of %d is the first the driver refuses (traced in the next submission)", g_bis_lo, draws_seen); }
    vk_bisect_reset_count();
}
void vk_ubd_land_drop(void) { g_nubd = 0; g_ubd_gen++; }
// Landed updates still in flight: their copies are queued but the GPU may not have run them, so memory may not hold them yet; a shadow built on the
// CPU (vk_ubd_read) applies them over memory until their submission completes. Without this a shadow mixed the newest update with the frame before's
// memory (shadows detached from moving characters, far pedestrians flickering).
typedef struct { uint64_t lo, hi, fence; const uint8_t *data; } ubd_infl_t;   // data: the update's bytes in its submission's scratch memory, which is not reused before that submission completes
static ubd_infl_t *g_infl; static int g_ninfl, g_capinfl;
static void infl_prune(void) {
    uint64_t done = vk_fence_done(); int o = 0;
    for (int i = 0; i < g_ninfl; i++) if (g_infl[i].fence > done) g_infl[o++] = g_infl[i];
    g_ninfl = o;
}
static void sub_gpu_wrote(uint64_t a, uint64_t n);
void vk_ubd_land(VkCommandBuffer cb) {   // outside a render pass
    if (!g_nubd) return;
    infl_prune();
    for (int i = 0; i < g_nubd; i++) {
        if (g_ninfl == g_capinfl) { g_capinfl = g_capinfl ? 2 * g_capinfl : 256; g_infl = realloc(g_infl, sizeof *g_infl * (size_t)g_capinfl); }
        // (the bytes used to be copied into the heap here: a malloc, memcpy and free of up to 64 KB per update, for bytes the scratch memory keeps anyway)
        g_infl[g_ninfl++] = (ubd_infl_t){g_ubd[i].lo, g_ubd[i].hi, g_fence + 1, g_ubd[i].p};   // the fence of the submission being recorded
    }
    vk_img_barrier_all(cb);
    for (int i = 0; i < g_nubd; i++) { VkBuffer db; VkDeviceSize dof; if (vk_buf_at(g_ubd[i].lo, g_ubd[i].hi - g_ubd[i].lo, &db, &dof)) {
        VkBufferCopy r = {g_ubd[i].so, dof, g_ubd[i].hi - g_ubd[i].lo}; vkCmdCopyBuffer(cb, g_ubd[i].sb, db, 1, &r); sub_gpu_wrote(g_ubd[i].lo, g_ubd[i].hi - g_ubd[i].lo); } }
    vk_img_barrier_all(cb);
    g_ubd_landed += (uint64_t)g_nubd; g_nubd = 0; g_ubd_gen++;
}
int vk_ubd_overlaps(uint64_t a, uint64_t n) { for (int i = 0; i < g_nubd; i++) if (g_ubd[i].lo < a + n && a < g_ubd[i].hi) return 1; return 0; }
// GPU writes to pool memory (buffer copies and fills, texture → buffer copies, storage buffers of dispatches) whose submission may not have completed:
// until it has, a CPU read of that memory (vk_ubd_read: a shadow, a handle in a uniform buffer) may see the bytes from before them or after them
typedef struct { uint64_t lo, hi, fence; } gpuw_t; enum { GW_MAX = 1024 };
static gpuw_t g_gw[GW_MAX]; static int g_ngw; uint64_t g_ubd_racy;
static void sub_gpu_wrote(uint64_t a, uint64_t n) {   // into the open submission's list (pools_copy_back)
    if (g_cur < 0 || !n) return;
    vksub_t *s = &g_slot[g_cur]; uint64_t lo = a, hi = a + n, bd = ~0ull; int best = -1;
    for (int k = 0; k < s->ngw; k++) { uint64_t d = lo > s->gw[k].hi ? lo - s->gw[k].hi : s->gw[k].lo > hi ? s->gw[k].lo - hi : 0; if (d < bd) { bd = d; best = k; } }
    if (best >= 0 && (bd <= 65536 || s->ngw == GW_SUB)) { if (lo < s->gw[best].lo) s->gw[best].lo = lo; if (hi > s->gw[best].hi) s->gw[best].hi = hi; }
    else { s->gw[s->ngw].lo = lo; s->gw[s->ngw].hi = hi; s->ngw++; }
}
void vk_gpu_wrote(uint64_t a, uint64_t n) {
    uint64_t f = g_fence + 1;
    for (int i = g_ngw - 1; i >= 0 && g_gw[i].fence == f; i--) if (g_gw[i].lo <= a && a + n <= g_gw[i].hi) return;
    if (g_ngw == GW_MAX) { uint64_t done = vk_fence_done(); int o = 0; for (int i = 0; i < g_ngw; i++) if (g_gw[i].fence > done) g_gw[o++] = g_gw[i]; g_ngw = o; }
    if (g_ngw == GW_MAX) { memmove(g_gw, g_gw + 1, sizeof *g_gw * (GW_MAX - 1)); g_ngw--; }
    g_gw[g_ngw++] = (gpuw_t){a, a + n, f};
    sub_gpu_wrote(a, n);
}
int vk_gpu_pending(uint64_t a, uint64_t n) {
    if (!g_ngw) return 0;
    uint64_t done = vk_fence_done();
    for (int i = g_ngw - 1; i >= 0; i--) if (g_gw[i].fence > done && g_gw[i].lo < a + n && a < g_gw[i].hi) return 1;
    return 0;
}
// what a GPU read of [a, a + n) sees at this point of the command stream, into dst (memory, then the deferred updates in order)
void vk_ubd_read(uint64_t a, uint64_t n, uint8_t *dst) {
    if (vk_gpu_pending(a, n)) { static int k; g_ubd_racy++; if (k < 20) { k++; int cov = 0; for (int i = 0; i < g_nubd; i++) if (g_ubd[i].lo <= a && a + n <= g_ubd[i].hi) cov = 1;
        VKLOG("[RED] a CPU read of %#llx +%llu (%s) overlaps a GPU write not known to be complete", (unsigned long long)a, (unsigned long long)n, cov ? "covered by a deferred update" : "partly from memory"); } }
    memcpy(dst, (const void *)(uintptr_t)a, n);
    if (g_ninfl) { uint64_t done = vk_fence_done();
        for (int i = 0; i < g_ninfl; i++) { if (g_infl[i].fence <= done) continue; uint64_t lo = g_infl[i].lo > a ? g_infl[i].lo : a, hi = g_infl[i].hi < a + n ? g_infl[i].hi : a + n;
            if (lo < hi) memcpy(dst + (lo - a), g_infl[i].data + (lo - g_infl[i].lo), hi - lo); } }
    for (int i = 0; i < g_nubd; i++) { uint64_t lo = g_ubd[i].lo > a ? g_ubd[i].lo : a, hi = g_ubd[i].hi < a + n ? g_ubd[i].hi : a + n;
        if (lo < hi) memcpy(dst + (lo - a), g_ubd[i].p + (lo - g_ubd[i].lo), hi - lo); }
}
// a uniform binding [a, a + n) that deferred updates overlap: its shadow (0 when none overlaps). Shadows are kept per (address, size) until the next update.
int vk_ubd_shadow(uint64_t a, uint64_t n, VkBuffer *buf, VkDeviceSize *off) {
    if (!g_nubd || !vk_ubd_overlaps(a, n)) return 0;   // (a binding no deferred update touches reads memory on the GPU, after the in-flight copies: no shadow needed)
    enum { NC = 64 }; static struct { uint64_t a, n, gen; VkBuffer b; VkDeviceSize o; } c[NC];
    unsigned k = (unsigned)((a ^ n * 0x9E3779B97F4A7C15ull) >> 6) % NC;
    extern int g_x_noshc; if (c[k].a == a && c[k].n == n && c[k].gen == g_ubd_gen && !g_x_noshc) { *buf = c[k].b; *off = c[k].o; return 1; }
    uint8_t *d = vk_scratch(n, V.props.limits.minUniformBufferOffsetAlignment, buf, off); vk_ubd_read(a, n, d); g_ubd_bytes += n;
    c[k].a = a; c[k].n = n; c[k].gen = g_ubd_gen; c[k].b = *buf; c[k].o = *off; g_ubd_shadows++;
    return 1;
}
void mtl_sub_index(const uint64_t *h, int n) { (void)h; (void)n; }
void mtl_exec_rest(const uint64_t *h, int n) { (void)h; (void)n; }

// completion: a queue of committed submissions, each with its CPU-visible effects
typedef struct cpend { uint64_t fence, frame; post_op *ops; int n; int slot, rslot, occ_parts; int npass; pass_meta pmeta[PASSQ]; uint64_t t_begin, t_commit, part_ns[MAIN_CBS]; int nparts; struct cpend *next; } cpend;
static cpend *g_ch, *g_ct; static pthread_mutex_t g_cmu = PTHREAD_MUTEX_INITIALIZER; static pthread_cond_t g_ccv = PTHREAD_COND_INITIALIZER;
uint64_t guest_ns(void);
uint64_t guest_ns_at_host_time(uint64_t host_ns);
// Report timestamps are NVN's raw 614.4 MHz unit (nvn_timestamp.h), which nvnDeviceGetTimestampInNanoseconds converts back; they were written in 19.2 MHz CPU ticks, 32 times too
// small, so a game's frame-rate controller saw its GPU finish at once (Donkey Kong Bananza stayed at present interval 1 while its frames took 33 ms: half speed)
static uint64_t gpu_ticks(void) { uint64_t g = guest_ns(); if (!g) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); g = (uint64_t)t.tv_sec * 1000000000ull + (uint64_t)t.tv_nsec; } return nvn_timestamp_from_ns(g); }
// GPU time per NVN frame (the sum of its submissions' timestamp spans), for the timing line's GPU column: the frame a submission belongs to is the one being replayed
static uint64_t g_cur_frame, g_frame_gpu[64], g_frame_gpu_of[64];
void mtl_set_frame(uint64_t f) { __atomic_store_n(&g_cur_frame, f, __ATOMIC_RELAXED); }
// ADPF consumes completed, explicitly sealed frames. The GPU's calibrated host
// endpoints are independent of when this completion thread wakes or copies pools.
// One commit includes all early command-buffer parts; count it once, at its fence.
static int perf_frame_on(void) {
    static _Atomic int cached = -1;
    int on = atomic_load_explicit(&cached, memory_order_relaxed);
    if (on < 0) { const char *e = getenv("DM_ADPF_COMPLETE"); int value = e && atoi(e), expected = -1;
        atomic_compare_exchange_strong_explicit(&cached, &expected, !!value, memory_order_relaxed, memory_order_relaxed);
        on = atomic_load_explicit(&cached, memory_order_relaxed); }
    return on;
}
enum { PERF_FRAMES = 64 };
typedef struct {
    uint64_t frame, last_fence, seal_fence, begin, end, busy;
    uint32_t submitted, completed;
    int occupied, sealed, invalid;
} perf_frame_gpu;
static perf_frame_gpu g_perf_frames[PERF_FRAMES];
static pthread_mutex_t g_perf_frame_mu = PTHREAD_MUTEX_INITIALIZER;
static uint64_t g_perf_completed_fence;
static perf_frame_gpu *perf_frame_find(uint64_t frame, int create) {
    perf_frame_gpu *p = &g_perf_frames[frame % PERF_FRAMES];
    if (p->occupied && p->frame == frame) return p;
    // Delayed completion or polling must never overwrite a newer ring occupant.
    if (!create || (p->occupied && p->frame > frame)) return NULL;
    *p = (perf_frame_gpu){.frame = frame, .occupied = 1}; return p;
}
static void perf_frame_submit(uint64_t frame, uint64_t fence) {
    if (!perf_frame_on()) return;
    pthread_mutex_lock(&g_perf_frame_mu);
    perf_frame_gpu *p = perf_frame_find(frame, 1);
    if (p) {
        if (p->sealed || !fence || fence <= p->last_fence || p->submitted == UINT32_MAX) p->invalid = 1;
        else { p->submitted++; p->last_fence = fence; }
    }
    pthread_mutex_unlock(&g_perf_frame_mu);
}
static void perf_frame_complete(uint64_t frame, uint64_t fence, int valid, uint64_t begin, uint64_t end) {
    if (!perf_frame_on()) return;
    pthread_mutex_lock(&g_perf_frame_mu);
    perf_frame_gpu *p = perf_frame_find(frame, 0);
    if (p) {
        if (!p->submitted || p->completed >= p->submitted || fence > p->last_fence || fence <= g_perf_completed_fence) p->invalid = 1;
        else {
            p->completed++;
            if (!valid || !begin || end < begin || end - begin > UINT64_MAX - p->busy) p->invalid = 1;
            else {
                if (!p->begin || begin < p->begin) p->begin = begin;
                if (end > p->end) p->end = end;
                p->busy += end - begin;
            }
        }
    }
    if (fence > g_perf_completed_fence) g_perf_completed_fence = fence;
    pthread_mutex_unlock(&g_perf_frame_mu);
}
uint64_t vk_frame_current(void) { return __atomic_load_n(&g_cur_frame, __ATOMIC_RELAXED); }
void vk_perf_frame_seal(uint64_t frame) {
    if (!perf_frame_on()) return;
    pthread_mutex_lock(&g_perf_frame_mu);
    perf_frame_gpu *p = perf_frame_find(frame, 1);
    if (p && !p->sealed) {
        p->sealed = 1; p->seal_fence = p->last_fence;
        if (!p->submitted) p->invalid = 1; // no timestamps: never borrow another frame's fence or duration
    }
    pthread_mutex_unlock(&g_perf_frame_mu);
}
int vk_perf_frame_gpu(uint64_t frame, uint64_t *begin, uint64_t *end, uint64_t *busy) {
    if (!perf_frame_on()) return -1;
    int status = 0;
    pthread_mutex_lock(&g_perf_frame_mu);
    perf_frame_gpu *p = perf_frame_find(frame, 0);
    if (p && p->sealed) {
        if (p->invalid) status = -1;
        else if (p->completed == p->submitted && g_perf_completed_fence >= p->seal_fence) {
            if (!p->begin || p->end <= p->begin || !p->busy) status = -1;
            else { *begin = p->begin; *end = p->end; *busy = p->busy; status = 1; }
        }
    }
    pthread_mutex_unlock(&g_perf_frame_mu); return status;
}

double mtl_gpu_ms(uint64_t f) { return g_frame_gpu_of[f % 64] == f + 1 ? g_frame_gpu[f % 64] / 1e6 : -1; }
double vk_gpu_ms_recent(void) {   // the latest finished frame's GPU time among the last 8 (-1: none known)
    uint64_t f = __atomic_load_n(&g_cur_frame, __ATOMIC_RELAXED);
    for (uint64_t k = 2; k < 8 && k <= f; k++) { double g = mtl_gpu_ms(f - k); if (g >= 0) return g; }
    return -1;
}
// DM_VK_PASS_SPIKE=<ms> (read at start-up, with DM_VK_PASS_TIMES; default 25): a frame whose timed passes and dispatches, or whose submissions, take more than that lists its costliest
// ones, to find what a one-frame GPU spike drew (Tears of the Kingdom: 25-180 ms frames among 10-13 ms ones while fighting). Completion thread only
static uint64_t g_sf_frame = ~0ull; static struct { double ms; pass_meta m; } g_sf[PASSQ * 4]; static int g_nsf; static double g_sf_tot;
static void pass_spike_add(double ms, const pass_meta *m) { g_sf_tot += ms; if (g_nsf < PASSQ * 4) { g_sf[g_nsf].ms = ms; g_sf[g_nsf].m = *m; g_nsf++; } }
static void pass_spike_flush(void) {
    static double lim = -1; if (lim < 0) { const char *e = getenv("DM_VK_PASS_SPIKE"); lim = e && atof(e) > 0 ? atof(e) : 25; }
    double fg = g_sf_frame != ~0ull ? mtl_gpu_ms(g_sf_frame) : -1;   // the frame's submissions end to end: a spike outside the timed passes shows as the difference
    if (g_sf_frame != ~0ull && (g_sf_tot > lim || fg > lim)) {
        for (int i = 0; i < g_nsf && i < 6; i++) for (int j = i + 1; j < g_nsf; j++) if (g_sf[j].ms > g_sf[i].ms) { __typeof__(g_sf[0]) x = g_sf[i]; g_sf[i] = g_sf[j]; g_sf[j] = x; }
        char b[1200]; int o = snprintf(b, sizeof b, "GPU spike: frame %llu, %.2f ms in %d passes and dispatches (frame GPU %.2f ms, %.2f outside them):", (unsigned long long)g_sf_frame, g_sf_tot, g_nsf, fg, fg - g_sf_tot);
        for (int i = 0; i < g_nsf && i < 6 && o < (int)sizeof b - 100; i++) { const pass_meta *m = &g_sf[i].m;
            if (m->fmt == PT_DISPATCH) o += snprintf(b + o, sizeof b - (size_t)o, " #%u %.2f ms (dispatch %ux%ux%u);", m->fidx, g_sf[i].ms, m->w, m->h, m->draws);
            else o += snprintf(b + o, sizeof b - (size_t)o, " #%u %.2f ms (%ux%u fmt %u, %u draws);", m->fidx, g_sf[i].ms, m->w, m->h, m->fmt, m->draws & 0xffff); }
        VKLOG("%s", b);
    }
    g_nsf = 0; g_sf_tot = 0;
}
// DM_VK_PASS_TIMES: one long submission every 300 frames on a line of its own, on the host clock from the submission's begin (mtl_sub_begin): when each part
// was sent and when the GPU began and ended its passes, the commit, the GPU's end, and its idle time inside its span. What the game's frame-rate controller
// measures (its frame start to the later of its submission and the GPU's end) is this chain, not the GPU's busy time
static void chain_maybe(const cpend *c, const uint64_t *t, int np) {
    static uint64_t last_f; if (np < 40 || c->frame < last_f + 300) return;
    gpu_clock_pair clk = gpu_clock_sample(); if (!clk.valid) return; last_f = c->frame; const gpu_clock_pair *ck = &clk;
    double b0 = (double)c->t_begin; char b[3000]; int o = 0; uint64_t gs = 0, ge = 0, prev_end = 0; double idle = 0, busy = 0;
    double pst[MAIN_CBS], pen[MAIN_CBS]; int pn[MAIN_CBS]; for (int i = 0; i < MAIN_CBS; i++) { pst[i] = 1e18; pen[i] = 0; pn[i] = 0; }
    for (int i = 0; i < np; i++) { uint64_t hs, he; if (t[2 * i + 1] <= t[2 * i] || !gpu_clock_host_ns(ck, t[2 * i], &hs) || !gpu_clock_host_ns(ck, t[2 * i + 1], &he)) continue;
        if (!gs) gs = hs; if (prev_end && hs > prev_end) idle += (double)(hs - prev_end); if (he > ge) ge = he; prev_end = he; busy += (double)(he - hs);
        uint32_t p = c->pmeta[i].part < MAIN_CBS ? c->pmeta[i].part : MAIN_CBS - 1; if (hs < pst[p]) pst[p] = (double)hs; if (he > pen[p]) pen[p] = (double)he; pn[p]++; }
    o += snprintf(b + o, sizeof b - (size_t)o, "chain: frame %llu, %d passes timed: commit at %.2f ms, GPU %.2f → %.2f ms (passes %.2f ms, idle between them %.2f), tail after commit %.2f; parts (sent, GPU begin-end, passes):",
                  (unsigned long long)c->frame, np, (c->t_commit - b0) / 1e6, gs ? (gs - b0) / 1e6 : -1, ge ? (ge - b0) / 1e6 : -1, busy / 1e6, idle / 1e6, ge ? ((double)ge - (double)c->t_commit) / 1e6 : -1);
    for (int i = 0; i < c->nparts && i < MAIN_CBS && o < (int)sizeof b - 60; i++) if (pn[i]) o += snprintf(b + o, sizeof b - (size_t)o, " %d: %.2f, %.2f-%.2f, %d;", i, (c->part_ns[i] - b0) / 1e6, (pst[i] - b0) / 1e6, (pen[i] - b0) / 1e6, pn[i]);
    VKLOG("%s", b);
    { const char *e = getenv("DM_VK_PASS_TIMES"); if (!e || atoi(e) < 2) return; }   // =2: every pass of it too, in order (start on the same clock, duration, size, format, draws, attachments)
    o = 0; for (int i = 0; i < np; i++) { uint64_t hs, he; const pass_meta *m = &c->pmeta[i];
        if (t[2 * i + 1] <= t[2 * i] || !gpu_clock_host_ns(ck, t[2 * i], &hs) || !gpu_clock_host_ns(ck, t[2 * i + 1], &he)) continue;
        if (m->fmt == PT_DISPATCH) o += snprintf(b + o, sizeof b - (size_t)o, " %d@%.2f+%.3f D%ux%ux%u;", i, (hs - b0) / 1e6, (double)(he - hs) / 1e6, m->w, m->h, m->draws);
        else o += snprintf(b + o, sizeof b - (size_t)o, " %d@%.2f+%.3f %ux%u f%u d%u c%u%s%s%s%s;", i, (hs - b0) / 1e6, (double)(he - hs) / 1e6, m->w, m->h, m->fmt, m->draws & 0xffff, m->flags & 15,
                           m->flags & 16 ? (m->flags & 32 ? " DL" : " Dc") : "", m->flags & 64 ? " CL" : "", m->flags & 128 ? " S" : "", "");
        if (o > (int)sizeof b - 120) { VKLOG("chain passes:%s", b); o = 0; } }
    if (o) VKLOG("chain passes:%s", b);
}
// Reports/readbacks precede the release which can wake a CPU writer. Keep this
// frontier separate from g_done: slot reuse and SyncWait include event publication too.
static void completion_events_publish(const cpend *c) {
    __atomic_store_n(&g_reports_done, c->fence, __ATOMIC_RELEASE);
    for (int i = 0; i < c->n; i++) {
        const post_op *o = &c->ops[i];
        if (o->kind == 1) nvn_event_publish_gpu(o->val, o->mode, o->v, c->fence);
    }
}
static void *completion_main(void *a) {
    (void)a; pthread_setname_np(pthread_self(), "dm-vk-complete");
    for (;;) {
        pthread_mutex_lock(&g_cmu);
        while (!g_ch) pthread_cond_wait(&g_ccv, &g_cmu);
        cpend *c = g_ch; pthread_mutex_unlock(&g_cmu);
        if (V.has_timeline) wait_fence(c->fence); else { VkResult r = vkWaitForFences(V.dev, 1, &g_fring[c->fence % NFRING], VK_TRUE, UINT64_MAX); if (r != VK_SUCCESS) die("[vk] GPU fence wait did not complete: %d", r); }
        uint64_t submit_ts[2] = {0};
        int submit_timed = g_tsq && c->slot >= 0 && vkGetQueryPoolResults(V.dev, g_tsq, 2 * (uint32_t)c->slot, 2, sizeof submit_ts, submit_ts, 8, VK_QUERY_RESULT_64_BIT) == VK_SUCCESS;
        int clock_reports = 0; for (int i = 0; i < c->n; i++) clock_reports += c->ops[i].q >= 0 && c->ops[i].kind == 2;
        gpu_clock_pair clock = ((perf_frame_on() && submit_timed) || clock_reports) && c->slot >= 0 ? gpu_clock_sample() : (gpu_clock_pair){0};
        uint64_t frame_begin = 0, frame_end = 0;
        int frame_valid = perf_frame_on() && submit_timed && gpu_clock_host_ns(&clock, submit_ts[0], &frame_begin) && gpu_clock_host_ns(&clock, submit_ts[1], &frame_end);
        if (g_shadow_pools && c->rslot >= 0) pools_copy_back(&g_slot[c->rslot]);
        if (submit_timed && submit_ts[1] > submit_ts[0]) {
            uint64_t ns = (uint64_t)((double)(submit_ts[1] - submit_ts[0]) * V.props.limits.timestampPeriod); g_gpu_ns += ns; int k = (int)(c->frame % 64);
            if (g_frame_gpu_of[k] != c->frame + 1) { g_frame_gpu[k] = 0; g_frame_gpu_of[k] = c->frame + 1; } g_frame_gpu[k] += ns;
        }
        if (g_pq && pass_times_on() && c->slot >= 0) {   // this submission's passes
            int np = c->npass; uint64_t t[PASSQ * 2];
            if (c->frame != g_sf_frame) { pass_spike_flush(); g_sf_frame = c->frame; }
            if (np && vkGetQueryPoolResults(V.dev, g_pq, (uint32_t)(c->slot * PASSQ * 2), (uint32_t)np * 2, sizeof t, t, 8, VK_QUERY_RESULT_64_BIT) == VK_SUCCESS)
                chain_maybe(c, t, np);
                for (int i = 0; i < np; i++) if (t[2 * i + 1] > t[2 * i]) { pass_spike_add((double)(t[2 * i + 1] - t[2 * i]) * V.props.limits.timestampPeriod / 1e6, &c->pmeta[i]); int f = (int)c->pmeta[i].fidx; g_pacc[f] += (double)(t[2 * i + 1] - t[2 * i]) * V.props.limits.timestampPeriod / 1e6; g_pcnt[f]++; g_plast[f] = c->pmeta[i]; if (f + 1 > g_pmax) g_pmax = f + 1; }
        }
        uint64_t now = gpu_ticks(), tend = submit_timed ? submit_ts[1] : 0;
        // Live diagnostic budget: unset logs four startup samples; set DM_VK_TS_DIAG=0,
        // then N (at most64), to capture a later gameplay window without a restart.
        if (clock_reports) { static int previous = -1, left; const char *e = getenv("DM_VK_TS_DIAG"); int budget = e ? atoi(e) : 4;
            if (budget < 0) budget = 0; if (budget > 64) budget = 64; if (budget != previous) { previous = budget; left = budget; }
            if (left > 0) { left--; uint64_t end_host;
                if (gpu_clock_host_ns(&clock, tend, &end_host)) { uint64_t end = nvn_timestamp_from_ns(guest_ns_at_host_time(end_host));
                    double delay = now >= end ? (double)(now - end) * 625.0 / 384.0 : -(double)(end - now) * 625.0 / 384.0;
                    VKLOG("counter clock frame %llu: completion fallback minus calibrated GPU end %.3f ms, max deviation %.3f us, %d query reports",
                        (unsigned long long)c->frame, delay / 1e6, clock.deviation / 1e3, clock_reports);
                } else VKLOG("counter clock frame %llu: calibration unavailable/invalid, completion fallback kept (%d query reports)", (unsigned long long)c->frame, clock_reports);
            }
        }
        for (int i = 0; i < c->n; i++) {
            post_op *o = &c->ops[i];
            if (o->kind == 3 || o->kind == 4) { occ_complete(&g_slot[c->rslot], o, c->occ_parts, c->frame, c->fence); continue; }
            if (o->kind == 1) continue;   // publish after reports and readbacks below
            uint64_t ts = now, tq;
            if (o->q >= 0 && c->slot >= 0 && g_cq && vkGetQueryPoolResults(V.dev, g_cq, (uint32_t)o->q, 1, 8, &tq, 8, VK_QUERY_RESULT_64_BIT) == VK_SUCCESS) {
                uint64_t host;
                if (gpu_clock_host_ns(&clock, tq, &host)) ts = nvn_timestamp_from_ns(guest_ns_at_host_time(host));
                else if (tend && tq <= tend) { uint64_t back = nvn_timestamp_from_ns((uint64_t)((double)(tend - tq) * V.props.limits.timestampPeriod)); if (back < now) ts = now - back; }
            }   // unsupported/invalid calibration keeps the prior completion-relative fallback
            // render_cond: a timestamp report's counter field is 0, as on the GPU (gfx_vk_exec.c render_cond evaluates conditions on it)
            uint64_t rec[2] = {o->type == 0 || o->type == 0x10 ? (DM_GAME_RENDER_COND ? 0 : ts) : o->type == NVN_COUNTER_TYPE_SAMPLES_PASSED ? g_occ_completed : 0, ts}; memcpy((void *)(uintptr_t)o->addr, rec, sizeof rec);
        }
        completion_events_publish(c);
        pthread_mutex_lock(&g_cmu); g_ch = c->next; if (!g_ch) g_ct = NULL; pthread_mutex_unlock(&g_cmu);
        if (g_prereset > 0 && c->rslot >= 0) { vksub_t *s = &g_slot[c->rslot];   // its command buffers reset here (slot_recycle); the slot is not taken before g_done passes this fence
            if (s->fence == c->fence) { for (int i = 0; i < s->nmain_alloc; i++) vkResetCommandBuffer(s->mains[i], 0); for (int i = 0; i < s->nup_alloc; i++) vkResetCommandBuffer(s->ups[i], 0); } }
        perf_frame_complete(c->frame, c->fence, frame_valid, frame_begin, frame_end);
        pthread_mutex_lock(&g_done_mu); if (c->fence > g_done) __atomic_store_n(&g_done, c->fence, __ATOMIC_RELEASE); pthread_cond_broadcast(&g_done_cv); pthread_mutex_unlock(&g_done_mu);
        nvn_queue_gpu_progress();
        free(c->ops); free(c);
    }
    return NULL;
}
static void completion_start(void) { pthread_t t; if (pthread_create(&t, NULL, completion_main, NULL)) die("[vk] completion thread"); pthread_detach(t); }
uint64_t mtl_sub_commit(void) {
    { void vk_prof_end(void); vk_prof_end(); }
    vksub_t *s = cur();
    vk_end_pass_land();
    vk_queue_preserve();   // retain bindings before a logical job releases its descriptors
    pools_readback_barrier(s->main);
    if (g_tsq) vkCmdWriteTimestamp(s->main, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, g_tsq, 2 * (uint32_t)g_cur + 1);
    int occ_parts = s->nmain_used;
    VkResult er = vkEndCommandBuffer(s->main); if (s->cb_err && er >= 0) er = VK_ERROR_OUT_OF_HOST_MEMORY;
    if (er < 0) { occ_parts = s->nmain_sent;   // the driver refused something recorded in it: logged with what the submission held, then it is dropped and the game goes on
        extern uint32_t g_sub_ops[64]; char ops[600]; int o = 0;
        for (int i = 0; i < 64 && o < (int)sizeof ops - 16; i++) if (g_sub_ops[i]) o += snprintf(ops + o, sizeof ops - (size_t)o, " %d:%u", i, g_sub_ops[i]);
        static int n; if (n++ < 3) VKLOG("[RED] vkEndCommandBuffer failed (%d): submission dropped; record ops (op:count)%s", er, ops);
        vk_bisect_result(0);
        if (s->nmain_sent) {   // parts already sent may still run: nothing more is submitted (the submission below only signals its fence)
            s->nmain_sent = s->nmain_used; s->up_used = 0; s->cb_err = 0; s->no_ts = 1; vk_ubd_land_drop();
        } else {
            vkResetCommandPool(V.dev, s->pool, 0); VkCommandBufferBeginInfo bi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, NULL, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
            s->main = s->mains[0]; s->nmain_used = 1; s->cb_err = 0; s->no_ts = 1;   // no timestamps: the slot's queries still hold an older submission's
            if (s->nup_used > 1) { s->up = s->ups[0]; s->nup_used = 1; }
            VKCHK(vkBeginCommandBuffer(s->main, &bi)); g_cmd_gen++; s->up_used = 0; vk_ubd_land_drop();
            pools_readback_barrier(s->main);
            VKCHK(vkEndCommandBuffer(s->main));
        }
    }
    else vk_bisect_result(1);
    { extern uint32_t g_sub_ops[64]; memset(g_sub_ops, 0, sizeof g_sub_ops); }
    vk_stage_flush();   // staged uploads (vk_stage_alloc) complete before this submission reads them
    { static unsigned nsub; static int nlog;   // a long submission's summary: the GPU memory reading is a sysfs read, so every 64th submission only, and 48 lines at most
      if (nlog < 48 && (s->nmain_used > 2 || g_vkst.draws - g_sub0.draws > 4000 || (!(nsub++ & 63) && kgsl_mapped() > g_sub0.kgsl + (512u << 20)))) { nlog++; sub_stats("submission ends", -1); kt_table(); } }
    if (s->up_used) { vk_img_barrier_all(s->up); VKCHK(vkEndCommandBuffer(s->up)); }
    uint64_t f = ++g_fence;
    VkCommandBuffer cbs[1 + MAIN_CBS]; uint32_t ncb = 0; if (s->up_used) cbs[ncb++] = s->up; for (int i = s->nmain_sent; i < s->nmain_used; i++) cbs[ncb++] = s->mains[i];
    VkTimelineSemaphoreSubmitInfo ts = {VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO, NULL, 0, NULL, 1, &f};
    VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO, &ts, 0, NULL, NULL, ncb, cbs, 1, &g_timeline};
    VkFence fe = VK_NULL_HANDLE;
    if (!V.has_timeline) {   // the ring's fence for f: free once the completion thread has passed f - NFRING (already so: slot_recycle waited for f - NSLOT)
        if (f > NFRING) mtl_wait(f - NFRING, ~0ull);
        fe = g_fring[f % NFRING]; VKCHK(vkResetFences(V.dev, 1, &fe)); si.pNext = NULL; si.signalSemaphoreCount = 0;
    }
    struct timespec t0, t1; clock_gettime(CLOCK_MONOTONIC, &t0); pthread_mutex_lock(&g_vk_queue_mu); VkResult r = vkQueueSubmit(V.q, 1, &si, fe); pthread_mutex_unlock(&g_vk_queue_mu); clock_gettime(CLOCK_MONOTONIC, &t1); g_ns_submit += (uint64_t)((t1.tv_sec - t0.tv_sec) * 1000000000ll + (t1.tv_nsec - t0.tv_nsec));
    if (r) { void vk_crumbs_report(void); vk_crumbs_report(); die("[vk] vkQueueSubmit: %d (device lost?)", r); }
    s->fence = f;
    cpend *c = calloc(1, sizeof *c); c->fence = f; c->frame = __atomic_load_n(&g_cur_frame, __ATOMIC_RELAXED); c->n = s->npost; c->rslot = g_cur; c->occ_parts = occ_parts; c->slot = s->no_ts ? -1 : g_cur; s->no_ts = 0;
    c->npass = g_npass[g_cur]; memcpy(c->pmeta, g_pmeta[g_cur], sizeof(pass_meta) * (size_t)c->npass);
    c->t_begin = s->t_begin; c->t_commit = (uint64_t)t1.tv_sec * 1000000000ull + (uint64_t)t1.tv_nsec; c->nparts = s->nmain_used; memcpy(c->part_ns, s->part_ns, sizeof c->part_ns); for (int i = s->nmain_sent; i < s->nmain_used; i++) c->part_ns[i] = c->t_commit;
    cut_stats_report(s, c->frame, c->fence);
    if (s->npost) { c->ops = malloc(s->npost * sizeof *c->ops); memcpy(c->ops, s->post, s->npost * sizeof *c->ops); }
    perf_frame_submit(c->frame, c->fence);
    pthread_mutex_lock(&g_cmu); if (g_ct) g_ct->next = c; else g_ch = c; g_ct = c; pthread_cond_signal(&g_ccv); pthread_mutex_unlock(&g_cmu);
    // Command memory: Adreno keeps what a command buffer's recording mapped (a bound descriptor set's descriptors are copied into the stream: Nintendo Switch 2 Welcome
    // Tour's first scene load recorded 3.3 GB in one submission, and the slots kept it), and a pool reset is the only way to give it back. A submission that mapped
    // DM_VK_CMDMEM_RESET_MB (default 1024, a safety net since the texture tables left the per-draw sets (gfx_vk.h vk_tbl_dsl); 0 = never) or more is waited for here (after its completion record is queued: the completion thread moves the fence) and its slot's pool reset with its resources released: a stall, at a scene load only
    { static int64_t cap = -1; if (cap < 0) { const char *e = getenv("DM_VK_CMDMEM_RESET_MB"); cap = (int64_t)(e ? atoi(e) : 1024) << 20; }
      uint64_t now = kgsl_mapped(); if (cap && now > g_sub0.kgsl + (uint64_t)cap) {
          struct timespec t0, t1; clock_gettime(CLOCK_MONOTONIC, &t0); mtl_wait(f, ~0ull);
          VKCHK(vkResetCommandPool(V.dev, s->pool, VK_COMMAND_POOL_RESET_RELEASE_RESOURCES_BIT)); clock_gettime(CLOCK_MONOTONIC, &t1);
          static int nl; if (nl++ < 12) VKLOG("command memory: submission %llu mapped %.0f MB of GPU memory; waited %.1f ms for it and released its slot's command pool: %.0f MB mapped now (was %.0f)", (unsigned long long)f, (now - g_sub0.kgsl) / 1048576.0, (t1.tv_sec - t0.tv_sec) * 1e3 + (t1.tv_nsec - t0.tv_nsec) / 1e6, kgsl_mapped() / 1048576.0, now / 1048576.0); } }
    s->npost = 0; g_cur = -1;
    pthread_mutex_unlock(&g_sub_lock);
    return f;
}
uint64_t mtl_last_fence(void) { return g_fence; }
int mtl_wait(uint64_t fence, uint64_t timeout_ns) {
    struct timespec dl; clock_gettime(CLOCK_REALTIME, &dl);
    uint64_t ns = (uint64_t)dl.tv_nsec + (timeout_ns > 3600ull * 1000000000ull ? 3600ull * 1000000000ull : timeout_ns);
    dl.tv_sec += (time_t)(ns / 1000000000ull); dl.tv_nsec = (long)(ns % 1000000000ull);
    pthread_mutex_lock(&g_done_mu);
    while (g_done < fence) if (pthread_cond_timedwait(&g_done_cv, &g_done_mu, &dl)) break;
    int ok = g_done >= fence; pthread_mutex_unlock(&g_done_mu);
    return ok;
}
void mtl_finish(void) { mtl_wait(g_fence, ~0ull); }
void mtl_drain(void) { mtl_finish(); vk_pipeline_cache_save(); }
