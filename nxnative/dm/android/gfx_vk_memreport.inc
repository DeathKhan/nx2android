// ---------------- live device memory by object kind (DM_VK_MEMREPORT) ----------------
// DM_VK_MEMREPORT=<seconds> (start-up; 1: every 10 s): a line every that many seconds, and at a DM_CTL "memreport" line, of what the process holds now by kind of Vulkan
// object. Two parts. Ours: live images, buffers, descriptor pools, pipelines, query pools and command pools, counted at their creation and destruction through the function
// table, images and buffers with the bytes their memory requirements asked for; beside them the images' device memory (vk_img_vkbytes), the GPU memory the process has
// mapped (kgsl) and what of it went through vkAllocateMemory (mem_acct): the rest is the driver's own. The driver's: where it has VK_EXT_device_memory_report the device
// is made with it and the driver reports each memory object it allocates, imports or frees with the kind of object it is for (descriptor pools, command buffers, pipelines,
// the queue: allocations we never see), live bytes by kind and the imports (our zero-copy pools) apart. The device-memory table by tag (vk_mem_acct_report) follows.
// Why: Ocarina of Time's Kokiri Forest runs at the 3 GB image budget on the Odin 3, and nothing said how much of the graphics memory beside the images is ours and how
// much the driver's (another port of the same game found 1.75 GB in descriptor pools no set came from). Off: nothing hooked or chained, and the DM_CTL line logs only
// the accounting that is always there (kgsl, images, the tag table)
enum { MR_IMAGE, MR_BUFFER, MR_DPOOL, MR_PIPE, MR_QPOOL, MR_CPOOL, MR_KINDS };
static const char *const g_mr_kind[MR_KINDS] = {"images", "buffers", "descriptor pools", "pipelines", "query pools", "command pools"};
enum { MR_TYPES = 27 };   // the driver's reports by VkObjectType: Vulkan 1.0's 0-25, then every later type together
static const char *const g_mr_type[MR_TYPES] = {"unknown", "instance", "physical device", "device", "queue", "semaphore", "command buffer", "fence", "device memory",
    "buffer", "image", "event", "query pool", "buffer view", "image view", "shader module", "pipeline cache", "pipeline layout", "render pass", "pipeline", "set layout",
    "sampler", "descriptor pool", "descriptor set", "framebuffer", "command pool", "later types"};
typedef struct { uint64_t bytes; uint8_t kind; } mr_obj_t;   // one of ours (by handle) or one of the driver's memory objects (by id; kind: its type, MR_TYPES: an import)
static struct {
    _Atomic uint64_t n[MR_KINDS]; uint64_t bytes[MR_KINDS];   // ours: live objects (atomic: counted outside the lock) and the bytes of images and buffers (under g_mr_mu)
    uint64_t drv[MR_TYPES], drv_n[MR_TYPES], imported, imported_n, failed, unseen;   // the driver's: live bytes and objects by type, imports, failed allocations, frees of ids never seen
    int ext, period;   // ext: the device has the driver's reports; period: seconds between lines (0: off)
} g_mr;
static nvn_hmap g_mr_own = {.klen = 8}, g_mr_drv = {.klen = 8};
static pthread_mutex_t g_mr_mu = PTHREAD_MUTEX_INITIALIZER;

// the driver's report of one memory object (VK_EXT_device_memory_report's callback; any thread, the driver's own included)
static void mr_driver_event(VkDeviceMemoryReportEventTypeEXT ev, uint64_t id, uint64_t size, VkObjectType type) {
    pthread_mutex_lock(&g_mr_mu);
    if (ev == VK_DEVICE_MEMORY_REPORT_EVENT_TYPE_ALLOCATION_FAILED_EXT) g_mr.failed++;
    else if (ev == VK_DEVICE_MEMORY_REPORT_EVENT_TYPE_ALLOCATE_EXT || ev == VK_DEVICE_MEMORY_REPORT_EVENT_TYPE_IMPORT_EXT) {
        mr_obj_t *o = malloc(sizeof *o);
        if (o) {
            o->bytes = size; o->kind = ev == VK_DEVICE_MEMORY_REPORT_EVENT_TYPE_IMPORT_EXT ? MR_TYPES : (uint32_t)type < MR_TYPES - 1 ? (uint8_t)type : MR_TYPES - 1;
            if (o->kind == MR_TYPES) { g_mr.imported += size; g_mr.imported_n++; } else { g_mr.drv[o->kind] += size; g_mr.drv_n[o->kind]++; }
            mr_obj_t *old = nvn_hm_put(&g_mr_drv, &id, o);   // (an id reported twice: the later report stands)
            if (old) { if (old->kind == MR_TYPES) { g_mr.imported -= old->bytes; g_mr.imported_n--; } else { g_mr.drv[old->kind] -= old->bytes; g_mr.drv_n[old->kind]--; } free(old); }
        }
    } else {   // a free or an unimport
        mr_obj_t *o = nvn_hm_del(&g_mr_drv, &id);
        if (!o) g_mr.unseen++;
        else { if (o->kind == MR_TYPES) { g_mr.imported -= o->bytes; g_mr.imported_n--; } else { g_mr.drv[o->kind] -= o->bytes; g_mr.drv_n[o->kind]--; } free(o); }
    }
    pthread_mutex_unlock(&g_mr_mu);
}
static void VKAPI_PTR mr_driver_cb(const VkDeviceMemoryReportCallbackDataEXT *d, void *user) { (void)user; mr_driver_event(d->type, d->memoryObjectId, d->size, d->objectType); }

// external memory (an AHardwareBuffer's): an image's requirements are not asked before it is bound (VUID-vkGetImageMemoryRequirements-image-01897), so neither
// kind is sized; imports are the driver report's
static int mr_external(const void *pnext) {
    for (const VkBaseInStructure *x = pnext; x; x = x->pNext)
        if (x->sType == VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO || x->sType == VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO) return 1;
    return 0;
}
// one of ours made / destroyed (bytes: its memory requirements, images and buffers only)
static void mr_own_made(int kind, uint64_t handle, uint64_t bytes) {
    atomic_fetch_add(&g_mr.n[kind], 1);
    if (!bytes || !handle) return;
    mr_obj_t *o = malloc(sizeof *o); if (!o) return;
    o->bytes = bytes; o->kind = (uint8_t)kind;
    pthread_mutex_lock(&g_mr_mu);
    g_mr.bytes[kind] += bytes; mr_obj_t *old = nvn_hm_put(&g_mr_own, &handle, o); if (old) { g_mr.bytes[old->kind] -= old->bytes; free(old); }
    pthread_mutex_unlock(&g_mr_mu);
}
static void mr_own_gone(int kind, uint64_t handle) {
    if (!handle) return;
    atomic_fetch_sub(&g_mr.n[kind], 1);
    pthread_mutex_lock(&g_mr_mu);
    mr_obj_t *o = nvn_hm_del(&g_mr_own, &handle); if (o) { g_mr.bytes[o->kind] -= o->bytes; free(o); }
    pthread_mutex_unlock(&g_mr_mu);
}

// the report: one line of ours, one of the driver's (when it reports), then the tag table
static void mr_lines(const char *why, char *own, size_t no, char *drv, size_t nd) {
    pthread_mutex_lock(&g_mr_mu);
    int n = snprintf(own, no, "memory report (%s): GPU memory mapped %.0f MB, through vkAllocateMemory %.0f MB (the rest is the driver's own), texture images %.0f MB;", why,
                     kgsl_mapped() / 1048576.0, g_macct_total / 1048576.0, vk_img_vkbytes() / 1048576.0);
    if (g_mr.period) for (int k = 0; k < MR_KINDS && n > 0 && (size_t)n < no; k++) {
        uint64_t c = atomic_load(&g_mr.n[k]);
        n += k == MR_IMAGE || k == MR_BUFFER ? snprintf(own + n, no - (size_t)n, " %s %llu (%.0f MB)%s", g_mr_kind[k], (unsigned long long)c, g_mr.bytes[k] / 1048576.0, k + 1 < MR_KINDS ? "," : "")
                                             : snprintf(own + n, no - (size_t)n, " %s %llu%s", g_mr_kind[k], (unsigned long long)c, k + 1 < MR_KINDS ? "," : "");
    }
    else if (n > 0 && (size_t)n < no) snprintf(own + n, no - (size_t)n, " objects not counted (DM_VK_MEMREPORT off)");
    if (drv && nd) drv[0] = 0;
    if (g_mr.ext && drv && nd) {
        uint64_t all = 0; for (int t = 0; t < MR_TYPES; t++) all += g_mr.drv[t];
        int m = snprintf(drv, nd, "memory report (%s), the driver's: %.0f MB live, imports %.0f MB (%llu);", why, all / 1048576.0, g_mr.imported / 1048576.0, (unsigned long long)g_mr.imported_n);
        for (int t = 0; t < MR_TYPES && m > 0 && (size_t)m < nd; t++) if (g_mr.drv_n[t])
            m += snprintf(drv + m, nd - (size_t)m, " %s %.1f MB (%llu),", g_mr_type[t], g_mr.drv[t] / 1048576.0, (unsigned long long)g_mr.drv_n[t]);
        if (m > 0 && (size_t)m < nd) snprintf(drv + m, nd - (size_t)m, " %llu failed allocations, %llu frees of objects not reported", (unsigned long long)g_mr.failed, (unsigned long long)g_mr.unseen);
    }
    pthread_mutex_unlock(&g_mr_mu);
}
void vk_memreport(const char *why) {
    char own[1024], drv[1536]; mr_lines(why, own, sizeof own, drv, sizeof drv);
    VKLOG("%s", own); if (*drv) VKLOG("%s", drv);
    vk_mem_acct_report(why);
}

// the function table: our objects counted as they are made and destroyed (only with DM_VK_MEMREPORT; after mem_acct's and the KGSL trace's wrappers, which these call)
static PFN_vkCreateImage mr_vkCreateImage; static PFN_vkDestroyImage mr_vkDestroyImage; static PFN_vkCreateBuffer mr_vkCreateBuffer; static PFN_vkDestroyBuffer mr_vkDestroyBuffer;
static PFN_vkCreateDescriptorPool mr_vkCreateDescriptorPool; static PFN_vkDestroyDescriptorPool mr_vkDestroyDescriptorPool;
static PFN_vkCreateGraphicsPipelines mr_vkCreateGraphicsPipelines; static PFN_vkCreateComputePipelines mr_vkCreateComputePipelines; static PFN_vkDestroyPipeline mr_vkDestroyPipeline;
static PFN_vkCreateQueryPool mr_vkCreateQueryPool; static PFN_vkCreateCommandPool mr_vkCreateCommandPool;
static VKAPI_ATTR VkResult VKAPI_CALL mr_create_image(VkDevice d, const VkImageCreateInfo *ci, const VkAllocationCallbacks *a, VkImage *o) {
    VkResult r = mr_vkCreateImage(d, ci, a, o); if (r) return r;
    VkMemoryRequirements req = {0}; if (!mr_external(ci->pNext)) vkGetImageMemoryRequirements(d, *o, &req); mr_own_made(MR_IMAGE, (uint64_t)(uintptr_t)*o, req.size); return r; }
static VKAPI_ATTR void VKAPI_CALL mr_destroy_image(VkDevice d, VkImage i, const VkAllocationCallbacks *a) { mr_own_gone(MR_IMAGE, (uint64_t)(uintptr_t)i); mr_vkDestroyImage(d, i, a); }
static VKAPI_ATTR VkResult VKAPI_CALL mr_create_buffer(VkDevice d, const VkBufferCreateInfo *ci, const VkAllocationCallbacks *a, VkBuffer *o) {
    VkResult r = mr_vkCreateBuffer(d, ci, a, o); if (r) return r;
    VkMemoryRequirements req = {0}; if (!mr_external(ci->pNext)) vkGetBufferMemoryRequirements(d, *o, &req); mr_own_made(MR_BUFFER, (uint64_t)(uintptr_t)*o, req.size); return r; }
static VKAPI_ATTR void VKAPI_CALL mr_destroy_buffer(VkDevice d, VkBuffer b, const VkAllocationCallbacks *a) { mr_own_gone(MR_BUFFER, (uint64_t)(uintptr_t)b); mr_vkDestroyBuffer(d, b, a); }
static VKAPI_ATTR VkResult VKAPI_CALL mr_create_dpool(VkDevice d, const VkDescriptorPoolCreateInfo *ci, const VkAllocationCallbacks *a, VkDescriptorPool *o) {
    VkResult r = mr_vkCreateDescriptorPool(d, ci, a, o); if (!r) mr_own_made(MR_DPOOL, (uint64_t)(uintptr_t)*o, 0); return r; }
static VKAPI_ATTR void VKAPI_CALL mr_destroy_dpool(VkDevice d, VkDescriptorPool p, const VkAllocationCallbacks *a) { mr_own_gone(MR_DPOOL, (uint64_t)(uintptr_t)p); mr_vkDestroyDescriptorPool(d, p, a); }
static VKAPI_ATTR VkResult VKAPI_CALL mr_create_gpipe(VkDevice d, VkPipelineCache c, uint32_t n, const VkGraphicsPipelineCreateInfo *ci, const VkAllocationCallbacks *a, VkPipeline *o) {
    VkResult r = mr_vkCreateGraphicsPipelines(d, c, n, ci, a, o); for (uint32_t i = 0; r >= 0 && i < n; i++) if (o[i]) mr_own_made(MR_PIPE, (uint64_t)(uintptr_t)o[i], 0); return r; }
static VKAPI_ATTR VkResult VKAPI_CALL mr_create_cpipe(VkDevice d, VkPipelineCache c, uint32_t n, const VkComputePipelineCreateInfo *ci, const VkAllocationCallbacks *a, VkPipeline *o) {
    VkResult r = mr_vkCreateComputePipelines(d, c, n, ci, a, o); for (uint32_t i = 0; r >= 0 && i < n; i++) if (o[i]) mr_own_made(MR_PIPE, (uint64_t)(uintptr_t)o[i], 0); return r; }
static VKAPI_ATTR void VKAPI_CALL mr_destroy_pipe(VkDevice d, VkPipeline p, const VkAllocationCallbacks *a) { mr_own_gone(MR_PIPE, (uint64_t)(uintptr_t)p); mr_vkDestroyPipeline(d, p, a); }
static VKAPI_ATTR VkResult VKAPI_CALL mr_create_qpool(VkDevice d, const VkQueryPoolCreateInfo *ci, const VkAllocationCallbacks *a, VkQueryPool *o) {
    VkResult r = mr_vkCreateQueryPool(d, ci, a, o); if (!r) mr_own_made(MR_QPOOL, (uint64_t)(uintptr_t)*o, 0); return r; }   // (never destroyed: made once)
static VKAPI_ATTR VkResult VKAPI_CALL mr_create_cpool(VkDevice d, const VkCommandPoolCreateInfo *ci, const VkAllocationCallbacks *a, VkCommandPool *o) {
    VkResult r = mr_vkCreateCommandPool(d, ci, a, o); if (!r) mr_own_made(MR_CPOOL, (uint64_t)(uintptr_t)*o, 0); return r; }   // (never destroyed: made once)

static void *mr_main(void *arg) {
    (void)arg;
    for (;;) { struct timespec ts = {g_mr.period, 0}; while (nanosleep(&ts, &ts) && errno == EINTR) {} vk_memreport("timer"); }
    return NULL;
}
// vk_init, before the device is made: whether DM_VK_MEMREPORT is on, and whether the driver reports its memory (ext, the device's extensions). Returns the create
// infos to chain in front of next (the feature and the callback), or next as it was
static void *mr_chain(const VkExtensionProperties *ext, uint32_t n, void *next, const char **on, uint32_t *non) {
    static VkDeviceDeviceMemoryReportCreateInfoEXT ci = {VK_STRUCTURE_TYPE_DEVICE_DEVICE_MEMORY_REPORT_CREATE_INFO_EXT, NULL, 0, mr_driver_cb, NULL};
    static VkPhysicalDeviceDeviceMemoryReportFeaturesEXT f = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DEVICE_MEMORY_REPORT_FEATURES_EXT};
    const char *e = getenv("DM_VK_MEMREPORT"); int s = e ? atoi(e) : 0;
    if (s <= 0) return next;
    g_mr.period = s == 1 ? 10 : s;
    if (!has_ext(ext, n, "VK_EXT_device_memory_report")) { VKLOG("memory report (DM_VK_MEMREPORT): the driver has no VK_EXT_device_memory_report: our own accounting only"); return next; }
    VkPhysicalDeviceFeatures2 q = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &f}; vkGetPhysicalDeviceFeatures2(V.pd, &q);
    if (!f.deviceMemoryReport) { VKLOG("memory report (DM_VK_MEMREPORT): VK_EXT_device_memory_report without its feature: our own accounting only"); return next; }
    g_mr.ext = 1; on[(*non)++] = "VK_EXT_device_memory_report";
    f.pNext = next; ci.pNext = &f; return &ci;
}
// vk_init, after the function table is loaded and mem_acct / the KGSL trace have wrapped it
static void mr_install(void) {
    if (!g_mr.period) return;
    mr_vkCreateImage = vkCreateImage; vkCreateImage = mr_create_image; mr_vkDestroyImage = vkDestroyImage; vkDestroyImage = mr_destroy_image;
    mr_vkCreateBuffer = vkCreateBuffer; vkCreateBuffer = mr_create_buffer; mr_vkDestroyBuffer = vkDestroyBuffer; vkDestroyBuffer = mr_destroy_buffer;
    mr_vkCreateDescriptorPool = vkCreateDescriptorPool; vkCreateDescriptorPool = mr_create_dpool; mr_vkDestroyDescriptorPool = vkDestroyDescriptorPool; vkDestroyDescriptorPool = mr_destroy_dpool;
    mr_vkCreateGraphicsPipelines = vkCreateGraphicsPipelines; vkCreateGraphicsPipelines = mr_create_gpipe; mr_vkCreateComputePipelines = vkCreateComputePipelines; vkCreateComputePipelines = mr_create_cpipe;
    mr_vkDestroyPipeline = vkDestroyPipeline; vkDestroyPipeline = mr_destroy_pipe;
    mr_vkCreateQueryPool = vkCreateQueryPool; vkCreateQueryPool = mr_create_qpool; mr_vkCreateCommandPool = vkCreateCommandPool; vkCreateCommandPool = mr_create_cpool;
    pthread_t th; if (!pthread_create(&th, NULL, mr_main, NULL)) { pthread_setname_np(th, "dm-vk-memrep"); pthread_detach(th); }
    VKLOG("memory report on (DM_VK_MEMREPORT): every %d s and at a DM_CTL \"memreport\" line; the driver's own allocations %s", g_mr.period,
          g_mr.ext ? "by object type (VK_EXT_device_memory_report)" : "only as GPU memory mapped beyond vkAllocateMemory (no VK_EXT_device_memory_report)");
}
