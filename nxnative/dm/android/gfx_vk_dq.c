// Two-stage replay (DM_VK_MT_REPLAY=1, start-up; off by default): the replay prepares each draw as before (state, descriptor sets, pipelines), but its vkCmd* calls go
// into a queue and a recorder thread of its own makes them, in the same order, into the same command buffers. The driver's share of the replay (Tears of the Kingdom
// at the Great Sky Island pond, 1,900 draws: vkCmdBindDescriptorSets 11.5%, vkCmdDrawIndexed 11.7%, vkCmdBeginRenderPass 8.8%, vkEndCommandBuffer 5.5% of the replay
// thread's samples) then runs beside the rest. Every vkCmd* entry point (gfx_vk.h VK_DEV_FNS) is swapped for a wrapper at vk_init; the command buffers' begin, end and
// allocation, queue submission and presentation first wait until the recorder has made every queued call (vk_dq_drain): command pools are externally synchronized, and a
// submission must hold everything recorded before it. A call whose arguments the wrapper cannot copy (an unknown pNext chain) is made directly after such a wait.
// Producers: the replay thread and the present thread (its blits into its own pool) at once: each record is written under a spinlock
// DM_VK_MT_REPLAY=2 (start-up): the recorder also begins and ends the command buffers, submits them and writes the descriptor sets, so the replay waits for it only where
// it needs a result. Ocarina of Time in Kokiri Forest on the Odin 3 (simpleperf of the replay thread, an older build): besides the vkCmd* calls (about 20%), vkEndCommandBuffer
// and vkQueueSubmit took about 10% and vkUpdateDescriptorSets 5%, and with =1 each end and submission (several a frame) waited for the recorder to empty the queue.
//   - vkBeginCommandBuffer, vkEndCommandBuffer and vkQueueSubmit are records and return VK_SUCCESS at once. A command buffer whose begin or end the driver refuses on the
//     recorder is left out of the submission that names it (a [RED] line; mtl_sub_commit's recovery of a refused submission does not run then). A submission the driver
//     refuses is made again without command buffers, so that its fence and semaphores still signal; a lost device dies as before.
//   - A submission's fence or timeline value may be waited for before the recorder has made it (the completion thread, slot_recycle, mtl_wait): the wait lasts until it
//     has, which Vulkan allows; nothing the recorder makes waits for a producer.
//   - vkUpdateDescriptorSets: the writes and their arrays are copied into the queue ahead of the set's bind, from every thread, so two writes of one set keep their order. A
//     set's pool is reset or destroyed only after the GPU passed the submission that used it (slot_recycle), so after the recorder wrote it, and the views, samplers and
//     buffers a write names go through the deferred lists (vk_defer_free_*) to that point too. A write with a pNext chain, or copies between sets, is made directly after a
//     drain. Allocation stays on the replay thread: Vulkan synchronizes a pool's allocations and a set's writes separately.
//   - The VkQueue: the recorder's submissions and the direct queue calls (presents, waits for idle, a submission this cannot copy; each after a drain) hold g_qmu, so the
//     recorder never takes the callers' g_vk_queue_mu (a present holds it while it drains).
//   - vkAllocateCommandBuffers and vkResetCommandPool still drain (rare), and so does a present: its submission must be made before it.
// vk_dq_report: beside the replay's line every 600 presents, the replay thread's and the recorder's CPU time a frame, in any mode (=0 too)
#include "gfx_vk.h"
#include <sched.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define DQ_CMDS(X) \
    X(vkCmdPipelineBarrier) X(vkCmdCopyBuffer) X(vkCmdCopyBufferToImage) X(vkCmdCopyImageToBuffer) X(vkCmdCopyImage) X(vkCmdBlitImage) X(vkCmdFillBuffer) X(vkCmdUpdateBuffer) \
    X(vkCmdClearColorImage) X(vkCmdClearDepthStencilImage) X(vkCmdClearAttachments) X(vkCmdResolveImage) X(vkCmdBeginRenderPass) X(vkCmdEndRenderPass) X(vkCmdBeginRendering) \
    X(vkCmdEndRendering) X(vkCmdBindPipeline) X(vkCmdBindDescriptorSets) X(vkCmdBindVertexBuffers) X(vkCmdBindIndexBuffer) X(vkCmdSetViewport) X(vkCmdSetScissor) \
    X(vkCmdSetBlendConstants) X(vkCmdSetStencilReference) X(vkCmdSetStencilCompareMask) X(vkCmdSetStencilWriteMask) X(vkCmdSetDepthBias) X(vkCmdSetLineWidth) X(vkCmdDraw) \
    X(vkCmdPushConstants) X(vkCmdDrawIndexed) X(vkCmdDrawIndirect) X(vkCmdDrawIndexedIndirect) X(vkCmdDispatch) X(vkCmdDispatchIndirect) X(vkCmdResetQueryPool) X(vkCmdBeginQuery) X(vkCmdEndQuery) \
    X(vkCmdWriteTimestamp) X(vkCmdBindTransformFeedbackBuffersEXT) X(vkCmdBeginTransformFeedbackEXT) X(vkCmdEndTransformFeedbackEXT) X(vkCmdDrawIndirectByteCountEXT) X(vkCmdDrawIndirectCount) X(vkCmdDrawIndexedIndirectCount)
#define DQ_SYNC(X) X(vkBeginCommandBuffer) X(vkEndCommandBuffer) X(vkAllocateCommandBuffers) X(vkResetCommandPool) X(vkQueueSubmit) X(vkQueuePresentKHR) X(vkQueueWaitIdle) X(vkDeviceWaitIdle)
#define DQ_MODE2(X) X(vkUpdateDescriptorSets)   // swapped with DM_VK_MT_REPLAY=2 only
#define DQ_REAL(f) static PFN_##f r_##f;
DQ_CMDS(DQ_REAL) DQ_SYNC(DQ_REAL) DQ_MODE2(DQ_REAL)
#define DQ_OP(f) OP_##f,
enum { OP_WRAP, DQ_CMDS(DQ_OP) OP_UPDDS, OP_BEGINCB, OP_ENDCB, OP_SUBMIT, OP_N };

// ---- the queue: one producer, one consumer; positions count bytes since the start (the ring index is position % RING), records 16-byte aligned ----
enum { RING = 32 << 20 };
typedef struct { uint32_t op, size; VkCommandBuffer cb; } rec_hdr;   // size: the whole record, header included
static uint8_t *g_ring; static uint64_t g_w;   // the producer's write position
static _Atomic uint64_t g_pub, g_rd;           // published end, the recorder's read position
static _Atomic int g_sleeping, g_draining;
static pthread_spinlock_t g_pl;   // the producers' lock, from rec_begin to rec_end
static pthread_mutex_t g_mu = PTHREAD_MUTEX_INITIALIZER; static pthread_cond_t g_cv = PTHREAD_COND_INITIALIZER, g_dcv = PTHREAD_COND_INITIALIZER;
static int g_on, g_mode;   // g_mode: DM_VK_MT_REPLAY, 1 or 2
static pthread_mutex_t g_qmu = PTHREAD_MUTEX_INITIALIZER;   // mode 2: the VkQueue, between the recorder's submissions and the direct queue calls
uint64_t g_dq_recs, g_dq_drains, g_dq_direct;   // records queued, drains, calls made directly (unknown arguments)
static uint64_t g_dq_desc, g_dq_desc_direct, g_dq_subs, g_dq_refused;   // mode 2: descriptor set writes queued / made directly, submissions queued, command buffers the driver refused
static _Atomic uint64_t g_dq_wait_ns;   // the producers' time in drains that waited
static clockid_t g_rec_clk, g_rep_clk; static _Atomic int g_rec_clk_ok, g_rep_clk_ok; static pthread_t g_rep_thr;   // the recorder's and the replay thread's CPU-time clocks (vk_dq_report)
static uint64_t ns_now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return (uint64_t)t.tv_sec * 1000000000ull + (uint64_t)t.tv_nsec; }

static void wake(void) { if (atomic_load(&g_sleeping)) { pthread_mutex_lock(&g_mu); pthread_cond_signal(&g_cv); pthread_mutex_unlock(&g_mu); } }
void vk_dq_drain(void) {
    if (!g_on) return;
    pthread_spin_lock(&g_pl); uint64_t w = g_w; pthread_spin_unlock(&g_pl);   // the records queued so far (another producer may add more meanwhile)
    if (atomic_load_explicit(&g_rd, memory_order_acquire) >= w) return;
    g_dq_drains++; wake(); uint64_t t0 = ns_now();
    for (int i = 0; i < 2000 && atomic_load_explicit(&g_rd, memory_order_acquire) < w; i++) sched_yield();   // the recorder is usually a few calls behind
    if (atomic_load_explicit(&g_rd, memory_order_acquire) >= w) { atomic_fetch_add(&g_dq_wait_ns, ns_now() - t0); return; }
    pthread_mutex_lock(&g_mu); atomic_store(&g_draining, 1);
    while (atomic_load_explicit(&g_rd, memory_order_acquire) < w) { pthread_cond_signal(&g_cv);
        struct timespec dl; clock_gettime(CLOCK_REALTIME, &dl); dl.tv_sec += 1;
        if (pthread_cond_timedwait(&g_dcv, &g_mu, &dl)) { uint64_t rd = atomic_load(&g_rd); const rec_hdr *h = (const rec_hdr *)(g_ring + rd % RING); static int n;
            if (n++ < 20) VKLOG("[RED] two-stage replay: a drain waits over 1 s: written %llu, published %llu, read %llu; the record there: op %u size %u",
                (unsigned long long)w, (unsigned long long)atomic_load(&g_pub), (unsigned long long)rd, h->op, h->size); } }
    atomic_store(&g_draining, 0); pthread_mutex_unlock(&g_mu);
    atomic_fetch_add(&g_dq_wait_ns, ns_now() - t0);
}
static void *rec_begin(uint32_t op, VkCommandBuffer cb, size_t n) {   // room for a record with n bytes after its header
    size_t need = (sizeof(rec_hdr) + n + 15) & ~(size_t)15;   // 16-byte records: the ring's end always has room for a whole header
    if (need > RING / 4) return NULL;   // (no call copies that much: such a call is made directly)
    pthread_spin_lock(&g_pl);   // released by rec_end
    size_t off = (size_t)(g_w % RING);
    if (off + need > RING) {   // the rest of the ring is skipped
        while (g_w + (RING - off) - atomic_load_explicit(&g_rd, memory_order_acquire) > RING) { wake(); sched_yield(); }
        rec_hdr *h = (rec_hdr *)(g_ring + off); h->op = OP_WRAP; h->size = (uint32_t)(RING - off); g_w += RING - off; atomic_store_explicit(&g_pub, g_w, memory_order_release); off = 0;
    }
    while (g_w + need - atomic_load_explicit(&g_rd, memory_order_acquire) > RING) { wake(); sched_yield(); }
    rec_hdr *h = (rec_hdr *)(g_ring + off); h->op = op; h->size = (uint32_t)need; h->cb = cb;
    return h + 1;
}
static void rec_end(void *payload) {
    rec_hdr *h = (rec_hdr *)payload - 1; g_w += h->size; g_dq_recs++;
    atomic_store(&g_pub, g_w);   // (seq_cst, against the recorder's sleep flag: either it sees this record or this sees it asleep)
    pthread_spin_unlock(&g_pl);
    wake();
}
// copying helpers: a record is a fixed struct followed by its arrays
#define ARR(T, p, n) ((n) && (p) ? (size_t)(n) * sizeof(T) : 0)
static uint8_t *put(uint8_t **at, const void *src, size_t n) { uint8_t *d = *at; if (n) memcpy(d, src, n); *at += n; return d; }

// ---- mode 2: descriptor set writes, command buffer begins and ends, submissions ----
// A vkUpdateDescriptorSets call in a record: a_updds, then each write as a ds_write followed by its descriptors (image infos, buffer infos or buffer views, by its
// type). All are multiples of 8 bytes, so every structure stays aligned in the 16-byte records
typedef struct { VkDevice dev; uint32_t n, pad; } a_updds;
typedef struct { VkDescriptorSet set; uint32_t binding, elem, count; VkDescriptorType type; } ds_write;
enum { DS_MAX = 128 };   // writes in one record: a call with more is made directly
static int ds_kind(VkDescriptorType t) {   // 1: image infos, 2: buffer infos, 3: buffer views; 0: in a pNext chain (inline uniform blocks, acceleration structures) or unknown here
    switch (t) {
    case VK_DESCRIPTOR_TYPE_SAMPLER: case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER: case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE: case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE: case VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT: return 1;
    case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER: case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER: case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC: case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC: return 2;
    case VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER: case VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER: return 3;
    default: return 0;
    }
}
static const size_t ds_elem[4] = {0, sizeof(VkDescriptorImageInfo), sizeof(VkDescriptorBufferInfo), sizeof(VkBufferView)};
_Static_assert(sizeof(ds_write) % 8 == 0 && sizeof(VkDescriptorImageInfo) % 8 == 0 && sizeof(VkDescriptorBufferInfo) % 8 == 0 && sizeof(a_updds) % 8 == 0, "8-byte multiples");
static const void *ds_src(const VkWriteDescriptorSet *w, int k) { return k == 1 ? (const void *)w->pImageInfo : k == 2 ? (const void *)w->pBufferInfo : (const void *)w->pTexelBufferView; }
static size_t ds_size(uint32_t n, const VkWriteDescriptorSet *w) {   // the writes' bytes in a record; 0: they cannot be copied (none, too many, a pNext chain, an unknown type, a missing array)
    if (!n || n > DS_MAX) return 0;
    size_t b = 0;
    for (uint32_t i = 0; i < n; i++) { int k = ds_kind(w[i].descriptorType);
        if (!k || w[i].pNext || (w[i].descriptorCount && !ds_src(&w[i], k))) return 0;
        b += sizeof(ds_write) + ds_elem[k] * w[i].descriptorCount; }
    return b;
}
static void ds_pack(uint8_t *d, uint32_t n, const VkWriteDescriptorSet *w) {
    for (uint32_t i = 0; i < n; i++) { int k = ds_kind(w[i].descriptorType); ds_write h = {w[i].dstSet, w[i].dstBinding, w[i].dstArrayElement, w[i].descriptorCount, w[i].descriptorType};
        memcpy(d, &h, sizeof h); d += sizeof h; size_t b = ds_elem[k] * h.count; if (b) memcpy(d, ds_src(&w[i], k), b); d += b; }
}
static void ds_unpack(const uint8_t *s, uint32_t n, VkWriteDescriptorSet *out) {   // out's arrays point into the record
    for (uint32_t i = 0; i < n; i++) { const ds_write *h = (const void *)s; s += sizeof *h; int k = ds_kind(h->type);
        out[i] = (VkWriteDescriptorSet){VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, NULL, h->set, h->binding, h->elem, h->count, h->type, NULL, NULL, NULL};
        if (k == 1) out[i].pImageInfo = (const void *)s; else if (k == 2) out[i].pBufferInfo = (const void *)s; else out[i].pTexelBufferView = (const void *)s;
        s += ds_elem[k] * h->count; }
}
// A vkQueueSubmit call in a record: a_submit, then each submit info as an a_subinfo followed by its wait semaphores, command buffers, signal semaphores, timeline wait
// and signal values (8 bytes each), then its wait stages (4 bytes each, padded to 8). Only a VkTimelineSemaphoreSubmitInfo may be chained; anything else is made directly
typedef struct { VkCommandBufferUsageFlags flags; } a_begincb;
typedef struct { VkQueue q; VkFence f; uint32_t n, pad; } a_submit;
typedef struct { uint32_t nw, ncb, ns, ts, ntw, nts; } a_subinfo;   // ts: a VkTimelineSemaphoreSubmitInfo with ntw wait and nts signal values was chained
enum { SUB_MAX = 8, SUB_N = 64 };   // submit infos in a record; semaphores, command buffers or values of one kind in a submit info
_Static_assert(sizeof(a_submit) % 8 == 0 && sizeof(a_subinfo) % 8 == 0 && sizeof(VkSemaphore) == 8 && sizeof(VkCommandBuffer) == 8, "8-byte multiples");
static size_t sub_size(const VkSubmitInfo *x) {   // a submit info's bytes in a record; 0: it cannot be copied
    const VkTimelineSemaphoreSubmitInfo *t = x->pNext;
    if (t && (t->sType != VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO || t->pNext)) return 0;
    uint32_t ntw = t ? t->waitSemaphoreValueCount : 0, nts = t ? t->signalSemaphoreValueCount : 0;
    if (x->waitSemaphoreCount > SUB_N || x->commandBufferCount > SUB_N || x->signalSemaphoreCount > SUB_N || ntw > SUB_N || nts > SUB_N) return 0;
    if ((x->waitSemaphoreCount && (!x->pWaitSemaphores || !x->pWaitDstStageMask)) || (x->commandBufferCount && !x->pCommandBuffers) || (x->signalSemaphoreCount && !x->pSignalSemaphores) ||
        (ntw && !t->pWaitSemaphoreValues) || (nts && !t->pSignalSemaphoreValues)) return 0;
    return sizeof(a_subinfo) + 8u * (x->waitSemaphoreCount + x->commandBufferCount + x->signalSemaphoreCount + ntw + nts) + ((4u * x->waitSemaphoreCount + 7) & ~7u);
}
// Command buffers whose begin or end the driver refused on the recorder: left out of the submission that names them (under g_qmu)
static VkCommandBuffer g_failed[64]; static int g_nfailed;
static int failed_take(VkCommandBuffer cb) { for (int i = 0; i < g_nfailed; i++) if (g_failed[i] == cb) { g_failed[i] = g_failed[--g_nfailed]; return 1; } return 0; }
static void cb_refused(VkCommandBuffer cb, VkResult r, const char *what) {
    pthread_mutex_lock(&g_qmu); int had = 0; for (int i = 0; i < g_nfailed; i++) had |= g_failed[i] == cb;
    if (!had && g_nfailed < 64) g_failed[g_nfailed++] = cb;
    uint64_t n = ++g_dq_refused; pthread_mutex_unlock(&g_qmu);
    if (n <= 3 || !(n & (n - 1))) VKLOG("[RED] two-stage replay: %s refused on the recorder thread (%d): the command buffer is left out of its submission (%llu so far)", what, r, (unsigned long long)n);
}
// a submission made now, under g_qmu, without the command buffers the driver refused: the recorder's, or a producer's after a drain (mode 2)
static VkResult submit_now(VkQueue q, uint32_t n, const VkSubmitInfo *si, VkFence f) {
    pthread_mutex_lock(&g_qmu); VkResult r;
    if (!g_nfailed) r = r_vkQueueSubmit(q, n, si, f);
    else {   // (rare: a copy of the submit infos without them)
        uint32_t ncb = 0; for (uint32_t i = 0; i < n; i++) ncb += si[i].commandBufferCount;
        VkSubmitInfo *cs = malloc(n * sizeof *cs + ncb * sizeof(VkCommandBuffer) + 8); if (!cs) die("[vk] two-stage replay: no memory for a submission");
        VkCommandBuffer *cbs = (VkCommandBuffer *)(cs + n);
        for (uint32_t i = 0; i < n; i++) { cs[i] = si[i]; uint32_t k = 0;
            for (uint32_t j = 0; j < si[i].commandBufferCount; j++) if (!failed_take(si[i].pCommandBuffers[j])) cbs[k++] = si[i].pCommandBuffers[j];
            cs[i].pCommandBuffers = k ? cbs : NULL; cs[i].commandBufferCount = k; cbs += k; }
        r = r_vkQueueSubmit(q, n, cs, f); free(cs);
    }
    pthread_mutex_unlock(&g_qmu); return r;
}
static void submit_rec(const uint8_t *p) {   // the recorder: a queued submission
    const a_submit *a = (const void *)p; p += sizeof *a;
    VkSubmitInfo si[SUB_MAX]; VkTimelineSemaphoreSubmitInfo ts[SUB_MAX];
    for (uint32_t i = 0; i < a->n && i < SUB_MAX; i++) { const a_subinfo *b = (const void *)p; p += sizeof *b;
        const VkSemaphore *ws = (const void *)p; p += 8u * b->nw; const VkCommandBuffer *cbs = (const void *)p; p += 8u * b->ncb; const VkSemaphore *ss = (const void *)p; p += 8u * b->ns;
        const uint64_t *wv = (const void *)p; p += 8u * b->ntw; const uint64_t *sv = (const void *)p; p += 8u * b->nts; const VkPipelineStageFlags *st = (const void *)p; p += (4u * b->nw + 7) & ~7u;
        ts[i] = (VkTimelineSemaphoreSubmitInfo){VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO, NULL, b->ntw, b->ntw ? wv : NULL, b->nts, b->nts ? sv : NULL};
        si[i] = (VkSubmitInfo){VK_STRUCTURE_TYPE_SUBMIT_INFO, b->ts ? &ts[i] : NULL, b->nw, b->nw ? ws : NULL, b->nw ? st : NULL, b->ncb, b->ncb ? cbs : NULL, b->ns, b->ns ? ss : NULL};
    }
    VkResult r = submit_now(a->q, a->n, si, a->f);
    if (r < 0 && r != VK_ERROR_DEVICE_LOST) {   // its fence and semaphores must still signal: the same submission without its command buffers
        static int nl; if (nl++ < 8) VKLOG("[RED] two-stage replay: vkQueueSubmit refused on the recorder thread (%d): made again without its command buffers", r);
        for (uint32_t i = 0; i < a->n; i++) { si[i].commandBufferCount = 0; si[i].pCommandBuffers = NULL; }
        r = submit_now(a->q, a->n, si, a->f);
    }
    if (r < 0) { if (r == VK_ERROR_DEVICE_LOST) vk_device_lost(r, "vkQueueSubmit (the recorder thread)"); die("[vk] vkQueueSubmit on the recorder thread: %d (device lost?)", r); }
}
static void begin_rec(VkCommandBuffer cb, const a_begincb *a) {
    pthread_mutex_lock(&g_qmu); failed_take(cb); pthread_mutex_unlock(&g_qmu);   // (begun again: no longer refused)
    VkCommandBufferBeginInfo bi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, NULL, a->flags, NULL};
    VkResult r = r_vkBeginCommandBuffer(cb, &bi); if (r < 0) cb_refused(cb, r, "vkBeginCommandBuffer");
}

// ---- the recorder thread ----
typedef struct { VkPipelineStageFlags src, dst; VkDependencyFlags dep; uint32_t nm, nb, ni; } a_barrier;
typedef struct { VkBuffer s, d; uint32_t n; } a_copybuf;
typedef struct { VkBuffer b; VkImage i; VkImageLayout l; uint32_t n; } a_b2i;
typedef struct { VkImage i; VkImageLayout l; VkBuffer b; uint32_t n; } a_i2b;
typedef struct { VkImage si; VkImageLayout sl; VkImage di; VkImageLayout dl; uint32_t n; VkFilter f; } a_img2;
typedef struct { VkBuffer b; VkDeviceSize off, size; uint32_t data; } a_fill;
typedef struct { VkBuffer b; VkDeviceSize off, size; } a_upd;
typedef struct { VkImage i; VkImageLayout l; VkClearColorValue c; VkClearDepthStencilValue ds; uint32_t n; } a_clrimg;
typedef struct { uint32_t na, nr; } a_clratt;
typedef struct { VkRenderPassBeginInfo bi; VkRenderPassAttachmentBeginInfo ab; int has_ab; VkSubpassContents c; } a_rpbegin;
typedef struct { VkRenderingInfo ri; int has_d, has_s; VkRenderingAttachmentInfo d, s; } a_rbegin;
typedef struct { VkPipelineBindPoint bp; VkPipeline p; } a_bindpipe;
typedef struct { VkPipelineBindPoint bp; VkPipelineLayout l; uint32_t first, n, ndyn; } a_sets;
typedef struct { uint32_t first, n; } a_range;
typedef struct { VkBuffer b; VkDeviceSize off; VkIndexType t; } a_index;
typedef struct { float c[4]; } a_blend;
typedef struct { VkStencilFaceFlags f; uint32_t v; } a_stencil;
typedef struct { float c, clamp, slope; } a_bias;
typedef struct { uint32_t a, b, c, d; int32_t e; uint32_t f; } a_draw;
typedef struct { VkPipelineLayout l; VkShaderStageFlags st; uint32_t off, size; } a_push;
typedef struct { VkBuffer b; VkDeviceSize off; uint32_t n, stride; } a_indirect;
typedef struct { VkQueryPool p; uint32_t first, n; } a_qreset;
typedef struct { VkQueryPool p; uint32_t q; VkQueryControlFlags flags; } a_query;
typedef struct { VkPipelineStageFlagBits st; VkQueryPool p; uint32_t q; } a_ts;
typedef struct { uint32_t first, n; int has_sizes, has_bufs, has_offs; } a_xfb;
typedef struct { uint32_t inst, first_inst; VkBuffer b; VkDeviceSize off; uint32_t ctr_off, stride; } a_bytecount;
typedef struct { VkBuffer b; VkDeviceSize off; VkBuffer cb; VkDeviceSize coff; uint32_t max, stride; } a_indcount;

static void run(const rec_hdr *h) {
    const uint8_t *p = (const uint8_t *)(h + 1); VkCommandBuffer cb = h->cb;
    switch (h->op) {
    case OP_vkCmdPipelineBarrier: { const a_barrier *a = (const void *)p; const uint8_t *q = p + sizeof *a;
        const VkMemoryBarrier *m = (const void *)q; q += a->nm * sizeof *m; const VkBufferMemoryBarrier *b = (const void *)q; q += a->nb * sizeof *b; const VkImageMemoryBarrier *im = (const void *)q;
        r_vkCmdPipelineBarrier(cb, a->src, a->dst, a->dep, a->nm, a->nm ? m : NULL, a->nb, a->nb ? b : NULL, a->ni, a->ni ? im : NULL); break; }
    case OP_vkCmdCopyBuffer: { const a_copybuf *a = (const void *)p; r_vkCmdCopyBuffer(cb, a->s, a->d, a->n, (const VkBufferCopy *)(p + sizeof *a)); break; }
    case OP_vkCmdCopyBufferToImage: { const a_b2i *a = (const void *)p; r_vkCmdCopyBufferToImage(cb, a->b, a->i, a->l, a->n, (const VkBufferImageCopy *)(p + sizeof *a)); break; }
    case OP_vkCmdCopyImageToBuffer: { const a_i2b *a = (const void *)p; r_vkCmdCopyImageToBuffer(cb, a->i, a->l, a->b, a->n, (const VkBufferImageCopy *)(p + sizeof *a)); break; }
    case OP_vkCmdCopyImage: { const a_img2 *a = (const void *)p; r_vkCmdCopyImage(cb, a->si, a->sl, a->di, a->dl, a->n, (const VkImageCopy *)(p + sizeof *a)); break; }
    case OP_vkCmdBlitImage: { const a_img2 *a = (const void *)p; r_vkCmdBlitImage(cb, a->si, a->sl, a->di, a->dl, a->n, (const VkImageBlit *)(p + sizeof *a), a->f); break; }
    case OP_vkCmdResolveImage: { const a_img2 *a = (const void *)p; r_vkCmdResolveImage(cb, a->si, a->sl, a->di, a->dl, a->n, (const VkImageResolve *)(p + sizeof *a)); break; }
    case OP_vkCmdFillBuffer: { const a_fill *a = (const void *)p; r_vkCmdFillBuffer(cb, a->b, a->off, a->size, a->data); break; }
    case OP_vkCmdUpdateBuffer: { const a_upd *a = (const void *)p; r_vkCmdUpdateBuffer(cb, a->b, a->off, a->size, p + sizeof *a); break; }
    case OP_vkCmdClearColorImage: { const a_clrimg *a = (const void *)p; r_vkCmdClearColorImage(cb, a->i, a->l, &a->c, a->n, (const VkImageSubresourceRange *)(p + sizeof *a)); break; }
    case OP_vkCmdClearDepthStencilImage: { const a_clrimg *a = (const void *)p; r_vkCmdClearDepthStencilImage(cb, a->i, a->l, &a->ds, a->n, (const VkImageSubresourceRange *)(p + sizeof *a)); break; }
    case OP_vkCmdClearAttachments: { const a_clratt *a = (const void *)p; const VkClearAttachment *at = (const void *)(p + sizeof *a);
        r_vkCmdClearAttachments(cb, a->na, at, a->nr, (const VkClearRect *)(at + a->na)); break; }
    case OP_vkCmdBeginRenderPass: { a_rpbegin a; memcpy(&a, p, sizeof a); const uint8_t *q = p + sizeof a;
        a.bi.pClearValues = a.bi.clearValueCount ? (const VkClearValue *)q : NULL; q += a.bi.clearValueCount * sizeof(VkClearValue);
        if (a.has_ab) { a.ab.pAttachments = (const VkImageView *)q; a.bi.pNext = &a.ab; } else a.bi.pNext = NULL;
        r_vkCmdBeginRenderPass(cb, &a.bi, a.c); break; }
    case OP_vkCmdEndRenderPass: r_vkCmdEndRenderPass(cb); break;
    case OP_vkCmdBeginRendering: { a_rbegin a; memcpy(&a, p, sizeof a);
        a.ri.pColorAttachments = a.ri.colorAttachmentCount ? (const VkRenderingAttachmentInfo *)(p + sizeof a) : NULL;
        a.ri.pDepthAttachment = a.has_d ? &a.d : NULL; a.ri.pStencilAttachment = a.has_s ? &a.s : NULL; r_vkCmdBeginRendering(cb, &a.ri); break; }
    case OP_vkCmdEndRendering: r_vkCmdEndRendering(cb); break;
    case OP_vkCmdBindPipeline: { const a_bindpipe *a = (const void *)p; r_vkCmdBindPipeline(cb, a->bp, a->p); break; }
    case OP_vkCmdBindDescriptorSets: { const a_sets *a = (const void *)p; const VkDescriptorSet *s = (const void *)(p + sizeof *a);
        r_vkCmdBindDescriptorSets(cb, a->bp, a->l, a->first, a->n, s, a->ndyn, a->ndyn ? (const uint32_t *)(s + a->n) : NULL); break; }
    case OP_vkCmdBindVertexBuffers: { const a_range *a = (const void *)p; const VkBuffer *b = (const void *)(p + sizeof *a);
        r_vkCmdBindVertexBuffers(cb, a->first, a->n, b, (const VkDeviceSize *)(b + a->n)); break; }
    case OP_vkCmdBindIndexBuffer: { const a_index *a = (const void *)p; r_vkCmdBindIndexBuffer(cb, a->b, a->off, a->t); break; }
    case OP_vkCmdSetViewport: { const a_range *a = (const void *)p; r_vkCmdSetViewport(cb, a->first, a->n, (const VkViewport *)(p + sizeof *a)); break; }
    case OP_vkCmdSetScissor: { const a_range *a = (const void *)p; r_vkCmdSetScissor(cb, a->first, a->n, (const VkRect2D *)(p + sizeof *a)); break; }
    case OP_vkCmdSetBlendConstants: { const a_blend *a = (const void *)p; r_vkCmdSetBlendConstants(cb, a->c); break; }
    case OP_vkCmdSetStencilReference: { const a_stencil *a = (const void *)p; r_vkCmdSetStencilReference(cb, a->f, a->v); break; }
    case OP_vkCmdSetStencilCompareMask: { const a_stencil *a = (const void *)p; r_vkCmdSetStencilCompareMask(cb, a->f, a->v); break; }
    case OP_vkCmdSetStencilWriteMask: { const a_stencil *a = (const void *)p; r_vkCmdSetStencilWriteMask(cb, a->f, a->v); break; }
    case OP_vkCmdSetDepthBias: { const a_bias *a = (const void *)p; r_vkCmdSetDepthBias(cb, a->c, a->clamp, a->slope); break; }
    case OP_vkCmdSetLineWidth: { float w; memcpy(&w, p, sizeof w); r_vkCmdSetLineWidth(cb, w); break; }
    case OP_vkCmdDraw: { const a_draw *a = (const void *)p; r_vkCmdDraw(cb, a->a, a->b, a->c, a->d); break; }
    case OP_vkCmdDrawIndexed: { const a_draw *a = (const void *)p; r_vkCmdDrawIndexed(cb, a->a, a->b, a->c, a->e, a->f); break; }
    case OP_vkCmdPushConstants: { const a_push *a = (const void *)p; r_vkCmdPushConstants(cb, a->l, a->st, a->off, a->size, p + sizeof *a); break; }
    case OP_vkCmdDrawIndirect: { const a_indirect *a = (const void *)p; r_vkCmdDrawIndirect(cb, a->b, a->off, a->n, a->stride); break; }
    case OP_vkCmdDrawIndexedIndirect: { const a_indirect *a = (const void *)p; r_vkCmdDrawIndexedIndirect(cb, a->b, a->off, a->n, a->stride); break; }
    case OP_vkCmdDispatch: { const a_draw *a = (const void *)p; r_vkCmdDispatch(cb, a->a, a->b, a->c); break; }
    case OP_vkCmdDispatchIndirect: { const a_indirect *a = (const void *)p; r_vkCmdDispatchIndirect(cb, a->b, a->off); break; }
    case OP_vkCmdResetQueryPool: { const a_qreset *a = (const void *)p; r_vkCmdResetQueryPool(cb, a->p, a->first, a->n); break; }
    case OP_vkCmdBeginQuery: { const a_query *a = (const void *)p; r_vkCmdBeginQuery(cb, a->p, a->q, a->flags); break; }
    case OP_vkCmdEndQuery: { const a_query *a = (const void *)p; r_vkCmdEndQuery(cb, a->p, a->q); break; }
    case OP_vkCmdWriteTimestamp: { const a_ts *a = (const void *)p; r_vkCmdWriteTimestamp(cb, a->st, a->p, a->q); break; }
    case OP_vkCmdBindTransformFeedbackBuffersEXT: { const a_xfb *a = (const void *)p; const VkBuffer *b = (const void *)(p + sizeof *a); const VkDeviceSize *o = (const void *)(b + a->n);
        r_vkCmdBindTransformFeedbackBuffersEXT(cb, a->first, a->n, b, o, a->has_sizes ? o + a->n : NULL); break; }
    case OP_vkCmdBeginTransformFeedbackEXT: case OP_vkCmdEndTransformFeedbackEXT: { const a_xfb *a = (const void *)p; const VkBuffer *b = (const void *)(p + sizeof *a);
        const VkDeviceSize *o = (const void *)(b + (a->has_bufs ? a->n : 0));
        (h->op == OP_vkCmdBeginTransformFeedbackEXT ? r_vkCmdBeginTransformFeedbackEXT : r_vkCmdEndTransformFeedbackEXT)(cb, a->first, a->n, a->has_bufs ? b : NULL, a->has_offs ? o : NULL); break; }
    case OP_vkCmdDrawIndirectCount: { const a_indcount *a = (const void *)p; r_vkCmdDrawIndirectCount(cb, a->b, a->off, a->cb, a->coff, a->max, a->stride); break; }
    case OP_vkCmdDrawIndexedIndirectCount: { const a_indcount *a = (const void *)p; r_vkCmdDrawIndexedIndirectCount(cb, a->b, a->off, a->cb, a->coff, a->max, a->stride); break; }
    case OP_vkCmdDrawIndirectByteCountEXT: { const a_bytecount *a = (const void *)p; r_vkCmdDrawIndirectByteCountEXT(cb, a->inst, a->first_inst, a->b, a->off, a->ctr_off, a->stride); break; }
    case OP_UPDDS: { const a_updds *a = (const void *)p; VkWriteDescriptorSet w[DS_MAX]; ds_unpack(p + sizeof *a, a->n, w); r_vkUpdateDescriptorSets(a->dev, a->n, w, 0, NULL); break; }
    case OP_BEGINCB: begin_rec(cb, (const a_begincb *)(const void *)p); break;
    case OP_ENDCB: { VkResult r = r_vkEndCommandBuffer(cb); if (r < 0) cb_refused(cb, r, "vkEndCommandBuffer"); break; }
    case OP_SUBMIT: submit_rec(p); break;
    default: break;
    }
}
static void *recorder_main(void *arg) {
    (void)arg; pthread_setname_np(pthread_self(), "vk record"); uint64_t r = 0;
    if (!pthread_getcpuclockid(pthread_self(), &g_rec_clk)) atomic_store(&g_rec_clk_ok, 1);   // (vk_dq_report)
    for (;;) {
        uint64_t pub = atomic_load(&g_pub);
        if (r == pub) {
            for (int i = 0; i < 200 && (pub = atomic_load(&g_pub)) == r; i++) sched_yield();   // the producer usually queues the next call within microseconds
            if (pub == r) {
                pthread_mutex_lock(&g_mu);
                if (atomic_load(&g_draining)) pthread_cond_broadcast(&g_dcv);
                atomic_store(&g_sleeping, 1);
                while ((pub = atomic_load(&g_pub)) == r) pthread_cond_wait(&g_cv, &g_mu);
                atomic_store(&g_sleeping, 0); pthread_mutex_unlock(&g_mu);
            }
        }
        while (r != pub) { const rec_hdr *h = (const rec_hdr *)(g_ring + r % RING); run(h); r += h->size; atomic_store_explicit(&g_rd, r, memory_order_release); }
        if (atomic_load(&g_draining)) { pthread_mutex_lock(&g_mu); pthread_cond_broadcast(&g_dcv); pthread_mutex_unlock(&g_mu); }
    }
    return NULL;
}

// ---- the wrappers (the producer) ----
#define DIRECT(f, ...) do { vk_dq_drain(); g_dq_direct++; r_##f(__VA_ARGS__); } while (0)
#define BEGIN(op, T, extra) uint8_t *base_ = rec_begin(op, cb, sizeof(T) + (extra)); if (!base_) goto direct; T *a = (T *)base_; uint8_t *at_ = base_ + sizeof(T)
static VKAPI_ATTR void VKAPI_CALL w_vkCmdPipelineBarrier(VkCommandBuffer cb, VkPipelineStageFlags src, VkPipelineStageFlags dst, VkDependencyFlags dep, uint32_t nm, const VkMemoryBarrier *m,
                                                         uint32_t nb, const VkBufferMemoryBarrier *b, uint32_t ni, const VkImageMemoryBarrier *im) {
    for (uint32_t i = 0; i < nm; i++) if (m[i].pNext) goto direct;
    for (uint32_t i = 0; i < nb; i++) if (b[i].pNext) goto direct;
    for (uint32_t i = 0; i < ni; i++) if (im[i].pNext) goto direct;
    { BEGIN(OP_vkCmdPipelineBarrier, a_barrier, ARR(VkMemoryBarrier, m, nm) + ARR(VkBufferMemoryBarrier, b, nb) + ARR(VkImageMemoryBarrier, im, ni));
      *a = (a_barrier){src, dst, dep, m ? nm : 0, b ? nb : 0, im ? ni : 0}; put(&at_, m, ARR(VkMemoryBarrier, m, nm)); put(&at_, b, ARR(VkBufferMemoryBarrier, b, nb)); put(&at_, im, ARR(VkImageMemoryBarrier, im, ni));
      rec_end(base_); return; }
direct: DIRECT(vkCmdPipelineBarrier, cb, src, dst, dep, nm, m, nb, b, ni, im);
}
#define W_REGIONS(f, OP, T, R, sig, init, call) \
    static VKAPI_ATTR void VKAPI_CALL w_##f sig { { BEGIN(OP, T, ARR(R, rg, n)); *a = init; put(&at_, rg, ARR(R, rg, n)); rec_end(base_); return; } direct: DIRECT call; }
W_REGIONS(vkCmdCopyBuffer, OP_vkCmdCopyBuffer, a_copybuf, VkBufferCopy, (VkCommandBuffer cb, VkBuffer s, VkBuffer d, uint32_t n, const VkBufferCopy *rg), ((a_copybuf){s, d, n}), (vkCmdCopyBuffer, cb, s, d, n, rg))
W_REGIONS(vkCmdCopyBufferToImage, OP_vkCmdCopyBufferToImage, a_b2i, VkBufferImageCopy, (VkCommandBuffer cb, VkBuffer b, VkImage i, VkImageLayout l, uint32_t n, const VkBufferImageCopy *rg),
          ((a_b2i){b, i, l, n}), (vkCmdCopyBufferToImage, cb, b, i, l, n, rg))
W_REGIONS(vkCmdCopyImageToBuffer, OP_vkCmdCopyImageToBuffer, a_i2b, VkBufferImageCopy, (VkCommandBuffer cb, VkImage i, VkImageLayout l, VkBuffer b, uint32_t n, const VkBufferImageCopy *rg),
          ((a_i2b){i, l, b, n}), (vkCmdCopyImageToBuffer, cb, i, l, b, n, rg))
W_REGIONS(vkCmdCopyImage, OP_vkCmdCopyImage, a_img2, VkImageCopy, (VkCommandBuffer cb, VkImage si, VkImageLayout sl, VkImage di, VkImageLayout dl, uint32_t n, const VkImageCopy *rg),
          ((a_img2){si, sl, di, dl, n, VK_FILTER_NEAREST}), (vkCmdCopyImage, cb, si, sl, di, dl, n, rg))
W_REGIONS(vkCmdBlitImage, OP_vkCmdBlitImage, a_img2, VkImageBlit, (VkCommandBuffer cb, VkImage si, VkImageLayout sl, VkImage di, VkImageLayout dl, uint32_t n, const VkImageBlit *rg, VkFilter f),
          ((a_img2){si, sl, di, dl, n, f}), (vkCmdBlitImage, cb, si, sl, di, dl, n, rg, f))
W_REGIONS(vkCmdResolveImage, OP_vkCmdResolveImage, a_img2, VkImageResolve, (VkCommandBuffer cb, VkImage si, VkImageLayout sl, VkImage di, VkImageLayout dl, uint32_t n, const VkImageResolve *rg),
          ((a_img2){si, sl, di, dl, n, VK_FILTER_NEAREST}), (vkCmdResolveImage, cb, si, sl, di, dl, n, rg))
W_REGIONS(vkCmdSetViewport, OP_vkCmdSetViewport, a_range, VkViewport, (VkCommandBuffer cb, uint32_t first, uint32_t n, const VkViewport *rg), ((a_range){first, n}), (vkCmdSetViewport, cb, first, n, rg))
W_REGIONS(vkCmdSetScissor, OP_vkCmdSetScissor, a_range, VkRect2D, (VkCommandBuffer cb, uint32_t first, uint32_t n, const VkRect2D *rg), ((a_range){first, n}), (vkCmdSetScissor, cb, first, n, rg))
static VKAPI_ATTR void VKAPI_CALL w_vkCmdFillBuffer(VkCommandBuffer cb, VkBuffer b, VkDeviceSize off, VkDeviceSize size, uint32_t data) {
    { BEGIN(OP_vkCmdFillBuffer, a_fill, 0); *a = (a_fill){b, off, size, data}; (void)at_; rec_end(base_); return; } direct: DIRECT(vkCmdFillBuffer, cb, b, off, size, data); }
static VKAPI_ATTR void VKAPI_CALL w_vkCmdUpdateBuffer(VkCommandBuffer cb, VkBuffer b, VkDeviceSize off, VkDeviceSize size, const void *data) {
    { BEGIN(OP_vkCmdUpdateBuffer, a_upd, (size_t)size); *a = (a_upd){b, off, size}; put(&at_, data, (size_t)size); rec_end(base_); return; } direct: DIRECT(vkCmdUpdateBuffer, cb, b, off, size, data); }
static VKAPI_ATTR void VKAPI_CALL w_vkCmdClearColorImage(VkCommandBuffer cb, VkImage i, VkImageLayout l, const VkClearColorValue *c, uint32_t n, const VkImageSubresourceRange *rg) {
    { BEGIN(OP_vkCmdClearColorImage, a_clrimg, ARR(VkImageSubresourceRange, rg, n)); memset(a, 0, sizeof *a); a->i = i; a->l = l; a->c = *c; a->n = n; put(&at_, rg, ARR(VkImageSubresourceRange, rg, n)); rec_end(base_); return; }
    direct: DIRECT(vkCmdClearColorImage, cb, i, l, c, n, rg); }
static VKAPI_ATTR void VKAPI_CALL w_vkCmdClearDepthStencilImage(VkCommandBuffer cb, VkImage i, VkImageLayout l, const VkClearDepthStencilValue *ds, uint32_t n, const VkImageSubresourceRange *rg) {
    { BEGIN(OP_vkCmdClearDepthStencilImage, a_clrimg, ARR(VkImageSubresourceRange, rg, n)); memset(a, 0, sizeof *a); a->i = i; a->l = l; a->ds = *ds; a->n = n; put(&at_, rg, ARR(VkImageSubresourceRange, rg, n)); rec_end(base_); return; }
    direct: DIRECT(vkCmdClearDepthStencilImage, cb, i, l, ds, n, rg); }
static VKAPI_ATTR void VKAPI_CALL w_vkCmdClearAttachments(VkCommandBuffer cb, uint32_t na, const VkClearAttachment *at, uint32_t nr, const VkClearRect *rc) {
    { BEGIN(OP_vkCmdClearAttachments, a_clratt, ARR(VkClearAttachment, at, na) + ARR(VkClearRect, rc, nr)); *a = (a_clratt){na, nr};
      put(&at_, at, ARR(VkClearAttachment, at, na)); put(&at_, rc, ARR(VkClearRect, rc, nr)); rec_end(base_); return; }
    direct: DIRECT(vkCmdClearAttachments, cb, na, at, nr, rc); }
static VKAPI_ATTR void VKAPI_CALL w_vkCmdBeginRenderPass(VkCommandBuffer cb, const VkRenderPassBeginInfo *bi, VkSubpassContents c) {
    const VkRenderPassAttachmentBeginInfo *ab = NULL;
    if (bi->pNext) { const VkBaseInStructure *x = bi->pNext; if (x->sType != VK_STRUCTURE_TYPE_RENDER_PASS_ATTACHMENT_BEGIN_INFO || x->pNext) goto direct; ab = (const void *)x; }
    { size_t nc = ARR(VkClearValue, bi->pClearValues, bi->clearValueCount), nv = ab ? ARR(VkImageView, ab->pAttachments, ab->attachmentCount) : 0;
      BEGIN(OP_vkCmdBeginRenderPass, a_rpbegin, nc + nv); memset(a, 0, sizeof *a); a->bi = *bi; a->c = c;
      if (!bi->pClearValues) a->bi.clearValueCount = 0;
      if (ab) { a->ab = *ab; a->has_ab = 1; }
      put(&at_, bi->pClearValues, nc); put(&at_, ab ? ab->pAttachments : NULL, nv); rec_end(base_); return; }
direct: DIRECT(vkCmdBeginRenderPass, cb, bi, c);
}
static VKAPI_ATTR void VKAPI_CALL w_vkCmdBeginRendering(VkCommandBuffer cb, const VkRenderingInfo *ri) {
    if (ri->pNext || (ri->pDepthAttachment && ri->pDepthAttachment->pNext) || (ri->pStencilAttachment && ri->pStencilAttachment->pNext)) goto direct;
    for (uint32_t i = 0; i < ri->colorAttachmentCount; i++) if (ri->pColorAttachments[i].pNext) goto direct;
    { BEGIN(OP_vkCmdBeginRendering, a_rbegin, ARR(VkRenderingAttachmentInfo, ri->pColorAttachments, ri->colorAttachmentCount)); memset(a, 0, sizeof *a); a->ri = *ri;
      if (!ri->pColorAttachments) a->ri.colorAttachmentCount = 0;
      if (ri->pDepthAttachment) { a->d = *ri->pDepthAttachment; a->has_d = 1; }
      if (ri->pStencilAttachment) { a->s = *ri->pStencilAttachment; a->has_s = 1; }
      put(&at_, ri->pColorAttachments, ARR(VkRenderingAttachmentInfo, ri->pColorAttachments, ri->colorAttachmentCount)); rec_end(base_); return; }
direct: DIRECT(vkCmdBeginRendering, cb, ri);
}
#define W_NOARG(f) static VKAPI_ATTR void VKAPI_CALL w_##f(VkCommandBuffer cb) { void *a = rec_begin(OP_##f, cb, 0); if (!a) { DIRECT(f, cb); return; } rec_end(a); }
W_NOARG(vkCmdEndRenderPass) W_NOARG(vkCmdEndRendering)
static VKAPI_ATTR void VKAPI_CALL w_vkCmdBindPipeline(VkCommandBuffer cb, VkPipelineBindPoint bp, VkPipeline pl) {
    { BEGIN(OP_vkCmdBindPipeline, a_bindpipe, 0); *a = (a_bindpipe){bp, pl}; (void)at_; rec_end(base_); return; } direct: DIRECT(vkCmdBindPipeline, cb, bp, pl); }
static VKAPI_ATTR void VKAPI_CALL w_vkCmdBindDescriptorSets(VkCommandBuffer cb, VkPipelineBindPoint bp, VkPipelineLayout l, uint32_t first, uint32_t n, const VkDescriptorSet *s, uint32_t nd, const uint32_t *d) {
    { BEGIN(OP_vkCmdBindDescriptorSets, a_sets, ARR(VkDescriptorSet, s, n) + ARR(uint32_t, d, nd)); *a = (a_sets){bp, l, first, n, d ? nd : 0};
      put(&at_, s, ARR(VkDescriptorSet, s, n)); put(&at_, d, ARR(uint32_t, d, nd)); rec_end(base_); return; }
    direct: DIRECT(vkCmdBindDescriptorSets, cb, bp, l, first, n, s, nd, d); }
static VKAPI_ATTR void VKAPI_CALL w_vkCmdBindVertexBuffers(VkCommandBuffer cb, uint32_t first, uint32_t n, const VkBuffer *b, const VkDeviceSize *o) {
    { BEGIN(OP_vkCmdBindVertexBuffers, a_range, ARR(VkBuffer, b, n) + ARR(VkDeviceSize, o, n)); *a = (a_range){first, n}; put(&at_, b, ARR(VkBuffer, b, n)); put(&at_, o, ARR(VkDeviceSize, o, n)); rec_end(base_); return; }
    direct: DIRECT(vkCmdBindVertexBuffers, cb, first, n, b, o); }
static VKAPI_ATTR void VKAPI_CALL w_vkCmdBindIndexBuffer(VkCommandBuffer cb, VkBuffer b, VkDeviceSize off, VkIndexType t) {
    { BEGIN(OP_vkCmdBindIndexBuffer, a_index, 0); *a = (a_index){b, off, t}; (void)at_; rec_end(base_); return; } direct: DIRECT(vkCmdBindIndexBuffer, cb, b, off, t); }
static VKAPI_ATTR void VKAPI_CALL w_vkCmdSetBlendConstants(VkCommandBuffer cb, const float c[4]) {
    { BEGIN(OP_vkCmdSetBlendConstants, a_blend, 0); memcpy(a->c, c, sizeof a->c); (void)at_; rec_end(base_); return; } direct: DIRECT(vkCmdSetBlendConstants, cb, c); }
#define W_STENCIL(f) static VKAPI_ATTR void VKAPI_CALL w_##f(VkCommandBuffer cb, VkStencilFaceFlags fl, uint32_t v) { \
    { BEGIN(OP_##f, a_stencil, 0); *a = (a_stencil){fl, v}; (void)at_; rec_end(base_); return; } direct: DIRECT(f, cb, fl, v); }
W_STENCIL(vkCmdSetStencilReference) W_STENCIL(vkCmdSetStencilCompareMask) W_STENCIL(vkCmdSetStencilWriteMask)
static VKAPI_ATTR void VKAPI_CALL w_vkCmdSetDepthBias(VkCommandBuffer cb, float c, float clamp, float slope) {
    { BEGIN(OP_vkCmdSetDepthBias, a_bias, 0); *a = (a_bias){c, clamp, slope}; (void)at_; rec_end(base_); return; } direct: DIRECT(vkCmdSetDepthBias, cb, c, clamp, slope); }
static VKAPI_ATTR void VKAPI_CALL w_vkCmdSetLineWidth(VkCommandBuffer cb, float w) {
    { BEGIN(OP_vkCmdSetLineWidth, float, 0); *a = w; (void)at_; rec_end(base_); return; } direct: DIRECT(vkCmdSetLineWidth, cb, w); }
static VKAPI_ATTR void VKAPI_CALL w_vkCmdDraw(VkCommandBuffer cb, uint32_t vc, uint32_t ic, uint32_t fv, uint32_t fi) {
    { BEGIN(OP_vkCmdDraw, a_draw, 0); *a = (a_draw){vc, ic, fv, fi, 0, 0}; (void)at_; rec_end(base_); return; } direct: DIRECT(vkCmdDraw, cb, vc, ic, fv, fi); }
static VKAPI_ATTR void VKAPI_CALL w_vkCmdDrawIndexed(VkCommandBuffer cb, uint32_t ic, uint32_t n, uint32_t fi, int32_t vo, uint32_t fin) {
    { BEGIN(OP_vkCmdDrawIndexed, a_draw, 0); *a = (a_draw){ic, n, fi, 0, vo, fin}; (void)at_; rec_end(base_); return; } direct: DIRECT(vkCmdDrawIndexed, cb, ic, n, fi, vo, fin); }
static VKAPI_ATTR void VKAPI_CALL w_vkCmdPushConstants(VkCommandBuffer cb, VkPipelineLayout l, VkShaderStageFlags st, uint32_t off, uint32_t size, const void *v) {
    { BEGIN(OP_vkCmdPushConstants, a_push, size); *a = (a_push){l, st, off, size}; put(&at_, v, size); rec_end(base_); return; } direct: DIRECT(vkCmdPushConstants, cb, l, st, off, size, v); }
#define W_INDIRECT(f) static VKAPI_ATTR void VKAPI_CALL w_##f(VkCommandBuffer cb, VkBuffer b, VkDeviceSize off, uint32_t n, uint32_t stride) { \
    { BEGIN(OP_##f, a_indirect, 0); *a = (a_indirect){b, off, n, stride}; (void)at_; rec_end(base_); return; } direct: DIRECT(f, cb, b, off, n, stride); }
W_INDIRECT(vkCmdDrawIndirect) W_INDIRECT(vkCmdDrawIndexedIndirect)
#define W_INDCOUNT(f) static VKAPI_ATTR void VKAPI_CALL w_##f(VkCommandBuffer cb, VkBuffer b, VkDeviceSize off, VkBuffer c, VkDeviceSize coff, uint32_t max, uint32_t stride) { \
    { BEGIN(OP_##f, a_indcount, 0); *a = (a_indcount){b, off, c, coff, max, stride}; (void)at_; rec_end(base_); return; } direct: DIRECT(f, cb, b, off, c, coff, max, stride); }
W_INDCOUNT(vkCmdDrawIndirectCount) W_INDCOUNT(vkCmdDrawIndexedIndirectCount)
static VKAPI_ATTR void VKAPI_CALL w_vkCmdDispatch(VkCommandBuffer cb, uint32_t x, uint32_t y, uint32_t z) {
    { BEGIN(OP_vkCmdDispatch, a_draw, 0); *a = (a_draw){x, y, z, 0, 0, 0}; (void)at_; rec_end(base_); return; } direct: DIRECT(vkCmdDispatch, cb, x, y, z); }
static VKAPI_ATTR void VKAPI_CALL w_vkCmdDispatchIndirect(VkCommandBuffer cb, VkBuffer b, VkDeviceSize off) {
    { BEGIN(OP_vkCmdDispatchIndirect, a_indirect, 0); *a = (a_indirect){b, off, 0, 0}; (void)at_; rec_end(base_); return; } direct: DIRECT(vkCmdDispatchIndirect, cb, b, off); }
static VKAPI_ATTR void VKAPI_CALL w_vkCmdResetQueryPool(VkCommandBuffer cb, VkQueryPool p, uint32_t first, uint32_t n) {
    { BEGIN(OP_vkCmdResetQueryPool, a_qreset, 0); *a = (a_qreset){p, first, n}; (void)at_; rec_end(base_); return; } direct: DIRECT(vkCmdResetQueryPool, cb, p, first, n); }
static VKAPI_ATTR void VKAPI_CALL w_vkCmdBeginQuery(VkCommandBuffer cb, VkQueryPool p, uint32_t q, VkQueryControlFlags flags) {
    { BEGIN(OP_vkCmdBeginQuery, a_query, 0); *a = (a_query){p, q, flags}; (void)at_; rec_end(base_); return; } direct: DIRECT(vkCmdBeginQuery, cb, p, q, flags); }
static VKAPI_ATTR void VKAPI_CALL w_vkCmdEndQuery(VkCommandBuffer cb, VkQueryPool p, uint32_t q) {
    { BEGIN(OP_vkCmdEndQuery, a_query, 0); *a = (a_query){p, q, 0}; (void)at_; rec_end(base_); return; } direct: DIRECT(vkCmdEndQuery, cb, p, q); }
static VKAPI_ATTR void VKAPI_CALL w_vkCmdWriteTimestamp(VkCommandBuffer cb, VkPipelineStageFlagBits st, VkQueryPool p, uint32_t q) {
    { BEGIN(OP_vkCmdWriteTimestamp, a_ts, 0); *a = (a_ts){st, p, q}; (void)at_; rec_end(base_); return; } direct: DIRECT(vkCmdWriteTimestamp, cb, st, p, q); }
static VKAPI_ATTR void VKAPI_CALL w_vkCmdBindTransformFeedbackBuffersEXT(VkCommandBuffer cb, uint32_t first, uint32_t n, const VkBuffer *b, const VkDeviceSize *o, const VkDeviceSize *sz) {
    { BEGIN(OP_vkCmdBindTransformFeedbackBuffersEXT, a_xfb, ARR(VkBuffer, b, n) + ARR(VkDeviceSize, o, n) + ARR(VkDeviceSize, sz, n)); *a = (a_xfb){first, n, sz != NULL, 1, 1};
      put(&at_, b, ARR(VkBuffer, b, n)); put(&at_, o, ARR(VkDeviceSize, o, n)); put(&at_, sz, ARR(VkDeviceSize, sz, n)); rec_end(base_); return; }
    direct: DIRECT(vkCmdBindTransformFeedbackBuffersEXT, cb, first, n, b, o, sz); }
#define W_XFB(f) static VKAPI_ATTR void VKAPI_CALL w_##f(VkCommandBuffer cb, uint32_t first, uint32_t n, const VkBuffer *b, const VkDeviceSize *o) { \
    { BEGIN(OP_##f, a_xfb, ARR(VkBuffer, b, n) + ARR(VkDeviceSize, o, n)); *a = (a_xfb){first, n, 0, b != NULL, o != NULL}; \
      put(&at_, b, ARR(VkBuffer, b, n)); put(&at_, o, ARR(VkDeviceSize, o, n)); rec_end(base_); return; } direct: DIRECT(f, cb, first, n, b, o); }
W_XFB(vkCmdBeginTransformFeedbackEXT) W_XFB(vkCmdEndTransformFeedbackEXT)
static VKAPI_ATTR void VKAPI_CALL w_vkCmdDrawIndirectByteCountEXT(VkCommandBuffer cb, uint32_t inst, uint32_t first_inst, VkBuffer b, VkDeviceSize off, uint32_t ctr_off, uint32_t stride) {
    { BEGIN(OP_vkCmdDrawIndirectByteCountEXT, a_bytecount, 0); *a = (a_bytecount){inst, first_inst, b, off, ctr_off, stride}; (void)at_; rec_end(base_); return; }
    direct: DIRECT(vkCmdDrawIndirectByteCountEXT, cb, inst, first_inst, b, off, ctr_off, stride); }

// the calls that own command pools or the queue: after every queued call is made; with DM_VK_MT_REPLAY=2 a begin, an end and a submission are records themselves
static VKAPI_ATTR VkResult VKAPI_CALL w_vkBeginCommandBuffer(VkCommandBuffer cb, const VkCommandBufferBeginInfo *bi) {
    if (g_mode >= 2 && !bi->pNext && !bi->pInheritanceInfo) { a_begincb *a = rec_begin(OP_BEGINCB, cb, sizeof *a); if (a) { a->flags = bi->flags; rec_end(a); return VK_SUCCESS; } }
    vk_dq_drain();
    if (g_mode >= 2) { pthread_mutex_lock(&g_qmu); failed_take(cb); pthread_mutex_unlock(&g_qmu); }
    return r_vkBeginCommandBuffer(cb, bi);
}
static VKAPI_ATTR VkResult VKAPI_CALL w_vkEndCommandBuffer(VkCommandBuffer cb) {
    if (g_mode >= 2) { void *a = rec_begin(OP_ENDCB, cb, 0); if (a) { rec_end(a); return VK_SUCCESS; } }
    vk_dq_drain(); return r_vkEndCommandBuffer(cb);
}
static VKAPI_ATTR VkResult VKAPI_CALL w_vkAllocateCommandBuffers(VkDevice d, const VkCommandBufferAllocateInfo *ai, VkCommandBuffer *cb) { vk_dq_drain(); return r_vkAllocateCommandBuffers(d, ai, cb); }
static VKAPI_ATTR VkResult VKAPI_CALL w_vkResetCommandPool(VkDevice d, VkCommandPool p, VkCommandPoolResetFlags f) { vk_dq_drain(); return r_vkResetCommandPool(d, p, f); }
static VKAPI_ATTR VkResult VKAPI_CALL w_vkQueueSubmit(VkQueue q, uint32_t n, const VkSubmitInfo *si, VkFence f) {
    if (g_mode < 2) { vk_dq_drain(); return r_vkQueueSubmit(q, n, si, f); }
    size_t b = n && n <= SUB_MAX ? sizeof(a_submit) : 0;
    for (uint32_t i = 0; b && i < n; i++) { size_t s = sub_size(&si[i]); b = s ? b + s : 0; }
    a_submit *a = b ? rec_begin(OP_SUBMIT, VK_NULL_HANDLE, b) : NULL;
    if (!a) { vk_dq_drain(); g_dq_direct++; return submit_now(q, n, si, f); }
    *a = (a_submit){q, f, n, 0}; uint8_t *at = (uint8_t *)(a + 1);
    for (uint32_t i = 0; i < n; i++) { const VkSubmitInfo *x = &si[i]; const VkTimelineSemaphoreSubmitInfo *t = x->pNext;
        a_subinfo *h = (a_subinfo *)(void *)at; at += sizeof *h;
        *h = (a_subinfo){x->waitSemaphoreCount, x->commandBufferCount, x->signalSemaphoreCount, t != NULL, t ? t->waitSemaphoreValueCount : 0, t ? t->signalSemaphoreValueCount : 0};
        put(&at, x->pWaitSemaphores, 8u * h->nw); put(&at, x->pCommandBuffers, 8u * h->ncb); put(&at, x->pSignalSemaphores, 8u * h->ns);
        put(&at, t ? t->pWaitSemaphoreValues : NULL, 8u * h->ntw); put(&at, t ? t->pSignalSemaphoreValues : NULL, 8u * h->nts);
        put(&at, x->pWaitDstStageMask, 4u * h->nw); at += (8 - 4u * h->nw % 8) % 8; }
    g_dq_subs++; rec_end(a); return VK_SUCCESS;
}
static VKAPI_ATTR VkResult VKAPI_CALL w_vkQueuePresentKHR(VkQueue q, const VkPresentInfoKHR *pi) {
    vk_dq_drain(); if (g_mode < 2) return r_vkQueuePresentKHR(q, pi);
    pthread_mutex_lock(&g_qmu); VkResult r = r_vkQueuePresentKHR(q, pi); pthread_mutex_unlock(&g_qmu); return r;
}
static VKAPI_ATTR VkResult VKAPI_CALL w_vkQueueWaitIdle(VkQueue q) {
    vk_dq_drain(); if (g_mode < 2) return r_vkQueueWaitIdle(q);
    pthread_mutex_lock(&g_qmu); VkResult r = r_vkQueueWaitIdle(q); pthread_mutex_unlock(&g_qmu); return r;
}
static VKAPI_ATTR VkResult VKAPI_CALL w_vkDeviceWaitIdle(VkDevice d) {
    vk_dq_drain(); if (g_mode < 2) return r_vkDeviceWaitIdle(d);
    pthread_mutex_lock(&g_qmu); VkResult r = r_vkDeviceWaitIdle(d); pthread_mutex_unlock(&g_qmu); return r;
}
static VKAPI_ATTR void VKAPI_CALL w_vkUpdateDescriptorSets(VkDevice d, uint32_t nw, const VkWriteDescriptorSet *w, uint32_t nc, const VkCopyDescriptorSet *c) {
    size_t b = nc ? 0 : ds_size(nw, w);
    a_updds *a = b ? rec_begin(OP_UPDDS, VK_NULL_HANDLE, sizeof *a + b) : NULL;
    if (!a) { if (nw || nc) { vk_dq_drain(); g_dq_desc_direct++; r_vkUpdateDescriptorSets(d, nw, w, nc, c); } return; }
    *a = (a_updds){d, nw, 0}; ds_pack((uint8_t *)(a + 1), nw, w); g_dq_desc++; rec_end(a);
}

// The replay thread's and the recorder's CPU time a frame, beside the replay's line (gfx_vk_present.c, every 600 presents), and what went through the queue
static uint64_t clk_ns(clockid_t c) { struct timespec t; return clock_gettime(c, &t) ? 0 : (uint64_t)t.tv_sec * 1000000000ull + (uint64_t)t.tv_nsec; }
void vk_dq_replay_thread(void) {   // gfx_vk_exec.c mtl_exec: the thread replaying now
    pthread_t t = pthread_self(); if (atomic_load_explicit(&g_rep_clk_ok, memory_order_acquire) && pthread_equal(t, g_rep_thr)) return;
    clockid_t c; if (pthread_getcpuclockid(t, &c)) return;
    atomic_store(&g_rep_clk_ok, 0); g_rep_thr = t; g_rep_clk = c; atomic_store_explicit(&g_rep_clk_ok, 1, memory_order_release);
}
void vk_dq_report(unsigned frames) {
    static uint64_t prep, prec, precs, pdesc, pddir, psubs, pdrains, pwait, pdirect; static clockid_t pclk;
    if (!frames) return;
    double f = frames; int rok = atomic_load_explicit(&g_rep_clk_ok, memory_order_acquire); clockid_t rc = g_rep_clk;
    uint64_t rep = rok ? clk_ns(rc) : 0, rec = atomic_load(&g_rec_clk_ok) ? clk_ns(g_rec_clk) : 0, wait = atomic_load(&g_dq_wait_ns);
    double rep_ms = rok && rc == pclk && rep >= prep ? (rep - prep) / f / 1e6 : -1;   // (-1: the replay thread changed, or the first line)
    if (!g_on) VKLOG("replay thread per frame (last %u): %.2f ms of CPU; no recorder thread (DM_VK_MT_REPLAY=0)", frames, rep_ms);
    else VKLOG("two-stage replay per frame (last %u, DM_VK_MT_REPLAY=%d): replay thread %.2f ms of CPU, recorder thread %.2f; %.0f calls queued (%.0f descriptor set writes, %.1f submissions), "
               "%.1f drains (the producers waited %.2f ms in them), %.1f calls made directly (%.1f descriptor set writes); %llu command buffers refused on the recorder (cumulative)",
               frames, g_mode, rep_ms, prec ? (rec - prec) / f / 1e6 : -1, (g_dq_recs - precs) / f, (g_dq_desc - pdesc) / f, (g_dq_subs - psubs) / f,
               (g_dq_drains - pdrains) / f, (wait - pwait) / f / 1e6, (g_dq_direct + g_dq_desc_direct - pdirect - pddir) / f, (g_dq_desc_direct - pddir) / f, (unsigned long long)g_dq_refused);
    prep = rep; pclk = rc; prec = rec; precs = g_dq_recs; pdesc = g_dq_desc; pddir = g_dq_desc_direct; psubs = g_dq_subs; pdrains = g_dq_drains; pwait = wait; pdirect = g_dq_direct;
}

void vk_dq_init(void) {   // vk_init, after the device's entry points are loaded
    const char *e = getenv("DM_VK_MT_REPLAY"); int m = e ? atoi(e) : 0; if (m <= 0) return;
    pthread_spin_init(&g_pl, PTHREAD_PROCESS_PRIVATE);
    if (!(g_ring = malloc(RING))) { VKLOG("[RED] DM_VK_MT_REPLAY: no memory for the queue; recording on the replay thread"); return; }
    g_mode = m >= 2 ? 2 : 1;
#define DQ_SWAP(f) r_##f = f; if (r_##f) f = w_##f;
    DQ_CMDS(DQ_SWAP) DQ_SYNC(DQ_SWAP)
    if (g_mode >= 2) { DQ_MODE2(DQ_SWAP) }
    g_on = 1;
    pthread_t t; pthread_attr_t at; pthread_attr_init(&at); pthread_attr_setdetachstate(&at, PTHREAD_CREATE_DETACHED);
    if (pthread_create(&t, &at, recorder_main, NULL)) die("[vk] recorder thread");
    pthread_attr_destroy(&at);
    if (g_mode >= 2) VKLOG("two-stage replay: vkCmd* calls, command buffer begins and ends, submissions and descriptor set writes queued for the recorder thread (DM_VK_MT_REPLAY=2)");
    else VKLOG("two-stage replay: vkCmd* calls queued for the recorder thread (DM_VK_MT_REPLAY)");
}
