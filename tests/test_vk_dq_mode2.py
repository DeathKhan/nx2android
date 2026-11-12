"""Run the real two-stage replay queue (nxnative/dm/android/gfx_vk_dq.c) with mock Vulkan entry points.

DM_VK_MT_REPLAY=2: command buffer begins and ends, submissions and descriptor set writes are made on the recorder thread in the producers' order, from copies of
their arguments (the callers' arrays are overwritten right after each call); a refused end leaves the command buffer out of its submission, a refused submission
is made again without command buffers, a present drains first. Two producers at once keep each command buffer's order. DM_VK_MT_REPLAY=1 keeps its drains.
"""
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parent))
import host_cc   # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DQ = ROOT / 'nxnative/dm/android/gfx_vk_dq.c'

HEADER = r'''
#pragma once
#define VK_NO_PROTOTYPES
#include <vulkan/vulkan_core.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define VKLOG(...) (fprintf(stderr, "[vk] " __VA_ARGS__), fputc('\n', stderr))
void die(const char *fmt, ...);
void vk_device_lost(VkResult r, const char *where);
DECLS
'''

MAIN = r'''
#include "gfx_vk_dq.c"
#include <assert.h>
#include <stdarg.h>
DEFS
void die(const char *fmt, ...) { va_list ap; va_start(ap, fmt); vfprintf(stderr, fmt, ap); va_end(ap); fputc('\n', stderr); abort(); }
void vk_device_lost(VkResult r, const char *where) { fprintf(stderr, "lost %d %s\n", r, where); }
#define H(T, v) ((T)(uintptr_t)(v))
static pthread_mutex_t lmu = PTHREAD_MUTEX_INITIALIZER;
static char evs[4096]; static int nev, nmain;   // the events the mocks saw, and how many on the main thread
static pthread_t main_thr; static int stress;
static _Atomic int nsub;
static void ev(char c) { if (stress) return; pthread_mutex_lock(&lmu); assert(nev < 4000); evs[nev++] = c; nmain += pthread_equal(pthread_self(), main_thr) != 0; pthread_mutex_unlock(&lmu); }
static void wait_subs(int n) { for (long i = 0; i < 50000000 && atomic_load(&nsub) < n; i++) sched_yield(); assert(atomic_load(&nsub) >= n); }
static void reset(void) { vk_dq_drain(); pthread_mutex_lock(&lmu); nev = nmain = 0; memset(evs, 0, sizeof evs); pthread_mutex_unlock(&lmu); }

// ---- the stress test's state: two producers, 400 command buffers each ----
enum { NP = 2, NI = 400 };
static int st_state[NP][NI], st_written[NP][NI];   // 1 begun, 2 drawn, 3 ended, 4 submitted; the set's write seen
static int who(uintptr_t h, int *k) { *k = (int)(h >> 16) - 1; return (int)(h & 0xffff); }   // handle = (producer + 1) << 16 | i

static VkResult end_fail; static VkCommandBuffer fail_cb; static int submit_fail;   // a refused end (fail_cb) / the next submission refused
static VkCommandBuffer seen_cbs[8]; static uint32_t seen_ncb; static VkSemaphore seen_ws, seen_ss; static uint64_t seen_wv, seen_sv; static VkPipelineStageFlags seen_st; static VkFence seen_f; static int seen_ts;
static VkDescriptorImageInfo seen_img[2]; static VkDescriptorBufferInfo seen_buf; static VkBufferView seen_view; static VkDescriptorSet seen_set; static uint32_t seen_nw;

static VKAPI_ATTR VkResult VKAPI_CALL m_begin(VkCommandBuffer cb, const VkCommandBufferBeginInfo *bi) {
    if (stress) { int k, i = who((uintptr_t)cb, &k); assert(st_state[k][i] == 0); st_state[k][i] = 1; return VK_SUCCESS; }
    assert(bi->flags == VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT && !bi->pNext); ev('B'); return VK_SUCCESS;
}
static VKAPI_ATTR VkResult VKAPI_CALL m_end(VkCommandBuffer cb) {
    if (stress) { int k, i = who((uintptr_t)cb, &k); assert(st_state[k][i] == 2); st_state[k][i] = 3; return VK_SUCCESS; }
    ev('E'); return cb == fail_cb ? end_fail : VK_SUCCESS;
}
static VKAPI_ATTR void VKAPI_CALL m_draw(VkCommandBuffer cb, uint32_t a, uint32_t b, uint32_t c, uint32_t d) {
    (void)b; (void)c; (void)d;
    if (stress) { int k, i = who((uintptr_t)cb, &k); assert(st_state[k][i] == 1 && st_written[k][i] && a == (uint32_t)i); st_state[k][i] = 2; return; }
    ev('D');
}
static VKAPI_ATTR void VKAPI_CALL m_bind(VkCommandBuffer cb, VkPipelineBindPoint bp, VkPipelineLayout l, uint32_t first, uint32_t n, const VkDescriptorSet *s, uint32_t nd, const uint32_t *d) {
    (void)cb; (void)bp; (void)l; (void)first; (void)nd; (void)d;
    if (stress) { int k, i = who((uintptr_t)s[0], &k); assert(n == 1 && st_written[k][i]); return; }
    assert(n == 1 && s[0] == seen_set); ev('S');
}
static VKAPI_ATTR void VKAPI_CALL m_update(VkDevice dev, uint32_t nw, const VkWriteDescriptorSet *w, uint32_t nc, const VkCopyDescriptorSet *c) {
    (void)c;
    if (stress) { assert(nw == 1 && !nc && w[0].pBufferInfo); int k, i = who((uintptr_t)w[0].dstSet, &k); assert(!st_written[k][i] && w[0].pBufferInfo->offset == 256u * (uint32_t)i); st_written[k][i] = 1; return; }
    assert(dev == H(VkDevice, 0x77)); seen_nw = nw; ev(nc ? 'c' : 'U');
    for (uint32_t j = 0; j < nw; j++) {
        seen_set = w[j].dstSet;
        if (w[j].descriptorType == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER) { assert(w[j].descriptorCount == 2 && w[j].dstArrayElement == 3); memcpy(seen_img, w[j].pImageInfo, sizeof seen_img); }
        else if (w[j].descriptorType == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC) { assert(w[j].dstBinding == 5); seen_buf = w[j].pBufferInfo[0]; }
        else if (w[j].descriptorType == VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER) seen_view = w[j].pTexelBufferView[0];
    }
}
static VKAPI_ATTR VkResult VKAPI_CALL m_submit(VkQueue q, uint32_t n, const VkSubmitInfo *si, VkFence f) {
    assert(q == H(VkQueue, 0x99));
    if (stress) { for (uint32_t s = 0; s < n; s++) for (uint32_t j = 0; j < si[s].commandBufferCount; j++) { int k, i = who((uintptr_t)si[s].pCommandBuffers[j], &k); assert(st_state[k][i] == 3); st_state[k][i] = 4; }
                  atomic_fetch_add(&nsub, 1); return VK_SUCCESS; }
    assert(n == 1); ev('Q'); seen_f = f; seen_ncb = si->commandBufferCount; if (seen_ncb) memcpy(seen_cbs, si->pCommandBuffers, sizeof *seen_cbs * seen_ncb);
    if (si->waitSemaphoreCount) { seen_ws = si->pWaitSemaphores[0]; seen_st = si->pWaitDstStageMask[0]; }
    if (si->signalSemaphoreCount) seen_ss = si->pSignalSemaphores[0];
    const VkTimelineSemaphoreSubmitInfo *t = si->pNext; if (t && t->sType != VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO) t = NULL; seen_ts = t != NULL;
    if (t) { assert(t->sType == VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO && t->signalSemaphoreValueCount == 1); seen_sv = t->pSignalSemaphoreValues[0]; seen_wv = t->waitSemaphoreValueCount ? t->pWaitSemaphoreValues[0] : 0; }
    atomic_fetch_add(&nsub, 1);
    if (submit_fail && seen_ncb) { submit_fail = 0; return VK_ERROR_OUT_OF_DEVICE_MEMORY; }
    return VK_SUCCESS;
}
static VKAPI_ATTR VkResult VKAPI_CALL m_present(VkQueue q, const VkPresentInfoKHR *pi) { (void)q; (void)pi; ev('P'); return VK_SUCCESS; }

static void producer_cycle(int k, int i) {   // one command buffer of the stress test: begin, write a set, bind it, draw, end, submit
    VkCommandBuffer cb = H(VkCommandBuffer, (uintptr_t)(k + 1) << 16 | (uintptr_t)i); VkDescriptorSet set = H(VkDescriptorSet, (uintptr_t)(k + 1) << 16 | (uintptr_t)i);
    VkCommandBufferBeginInfo bi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, NULL, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, NULL};
    assert(vkBeginCommandBuffer(cb, &bi) == VK_SUCCESS);
    VkDescriptorBufferInfo b = {H(VkBuffer, 0x500), 256u * (uint32_t)i, 64};
    VkWriteDescriptorSet w = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, NULL, set, 0, 0, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, NULL, &b, NULL};
    vkUpdateDescriptorSets(H(VkDevice, 0x77), 1, &w, 0, NULL); memset(&b, 0xee, sizeof b);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, H(VkPipelineLayout, 1), 0, 1, &set, 0, NULL);
    vkCmdDraw(cb, (uint32_t)i, 1, 0, 0);
    assert(vkEndCommandBuffer(cb) == VK_SUCCESS);
    VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO, NULL, 0, NULL, NULL, 1, &cb, 0, NULL};
    assert(vkQueueSubmit(H(VkQueue, 0x99), 1, &si, VK_NULL_HANDLE) == VK_SUCCESS);
}
static void *producer(void *arg) { int k = (int)(intptr_t)arg; for (int i = 0; i < NI; i++) producer_cycle(k, i); return NULL; }

int main(int argc, char **argv) {
    (void)argc; int mode = atoi(argv[1]); main_thr = pthread_self();
    setenv("DM_VK_MT_REPLAY", argv[1], 1);
    vkBeginCommandBuffer = m_begin; vkEndCommandBuffer = m_end; vkCmdDraw = m_draw; vkCmdBindDescriptorSets = m_bind; vkUpdateDescriptorSets = m_update; vkQueueSubmit = m_submit; vkQueuePresentKHR = m_present;
    vk_dq_init(); vk_dq_replay_thread();
    assert(vkUpdateDescriptorSets == (mode >= 2 ? w_vkUpdateDescriptorSets : m_update));
    VkCommandBuffer A = H(VkCommandBuffer, 0xA0), Bc = H(VkCommandBuffer, 0xB0), C = H(VkCommandBuffer, 0xC0);
    VkCommandBufferBeginInfo bi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, NULL, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, NULL};

    // 1. begin, draw, a set's writes (their arrays overwritten after the call), its bind, end, a submission with semaphores and timeline values
    reset();
    vkBeginCommandBuffer(A, &bi); vkCmdDraw(A, 3, 1, 0, 0);
    VkDescriptorImageInfo ii[2] = {{H(VkSampler, 0x11), H(VkImageView, 0x12), VK_IMAGE_LAYOUT_GENERAL}, {H(VkSampler, 0x21), H(VkImageView, 0x22), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL}};
    VkDescriptorBufferInfo bu = {H(VkBuffer, 0x31), 0, 65536}; VkBufferView bv = H(VkBufferView, 0x41); VkDescriptorSet set = H(VkDescriptorSet, 0x51);
    VkWriteDescriptorSet w[3] = {{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, NULL, set, 1, 3, 2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, ii, NULL, NULL},
                                 {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, NULL, set, 5, 0, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, NULL, &bu, NULL},
                                 {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, NULL, set, 6, 0, 1, VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, NULL, NULL, &bv}};
    vkUpdateDescriptorSets(H(VkDevice, 0x77), 3, w, 0, NULL); memset(ii, 0xcd, sizeof ii); memset(&bu, 0xcd, sizeof bu); memset(&bv, 0xcd, sizeof bv); memset(w, 0xcd, sizeof w);
    vkCmdBindDescriptorSets(A, VK_PIPELINE_BIND_POINT_GRAPHICS, H(VkPipelineLayout, 1), 0, 1, &set, 0, NULL);
    assert(vkEndCommandBuffer(A) == VK_SUCCESS);
    VkSemaphore ws = H(VkSemaphore, 0x61), ss = H(VkSemaphore, 0x62); VkPipelineStageFlags st = VK_PIPELINE_STAGE_TRANSFER_BIT; uint64_t wv = 0, sv = 42;
    VkTimelineSemaphoreSubmitInfo ts = {VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO, NULL, 1, &wv, 1, &sv};
    VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO, &ts, 1, &ws, &st, 1, &A, 1, &ss};
    assert(vkQueueSubmit(H(VkQueue, 0x99), 1, &si, H(VkFence, 0x71)) == VK_SUCCESS);
    sv = 7; ws = ss = H(VkSemaphore, 0xdead); st = 0; A = H(VkCommandBuffer, 0xdead); memset(&ts, 0xcd, sizeof ts); memset(&si, 0xcd, sizeof si);
    wait_subs(1); vk_dq_drain();
    // (mode 1: the writes are made at once on the main thread, so the queued draw may come before or after them)
    if (mode >= 2) assert(!strcmp(evs, "BDUSEQ")); else assert(strlen(evs) == 6 && evs[0] == 'B' && !strcmp(evs + 3, "SEQ"));
    assert(seen_nw == 3 && seen_img[0].imageView == H(VkImageView, 0x12) && seen_img[1].sampler == H(VkSampler, 0x21) && seen_img[1].imageLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    assert(seen_buf.buffer == H(VkBuffer, 0x31) && seen_buf.range == 65536 && seen_view == H(VkBufferView, 0x41));
    assert(seen_ncb == 1 && seen_cbs[0] == H(VkCommandBuffer, 0xA0) && seen_ws == H(VkSemaphore, 0x61) && seen_ss == H(VkSemaphore, 0x62) && seen_st == VK_PIPELINE_STAGE_TRANSFER_BIT);
    assert(seen_ts && seen_sv == 42 && seen_wv == 0 && seen_f == H(VkFence, 0x71));
    // mode 2: nothing on the main thread; mode 1: the begin, the writes (not queued), the end and the submission there, after drains
    if (mode >= 2) assert(nmain == 0); else assert(nmain == 4);

    // 2. a present drains first: the submission before it is made before it
    reset();
    VkCommandBuffer one = H(VkCommandBuffer, 0xA1); VkSubmitInfo s2 = {VK_STRUCTURE_TYPE_SUBMIT_INFO, NULL, 0, NULL, NULL, 1, &one, 0, NULL};
    vkBeginCommandBuffer(one, &bi); vkCmdDraw(one, 1, 1, 0, 0); vkEndCommandBuffer(one); vkQueueSubmit(H(VkQueue, 0x99), 1, &s2, VK_NULL_HANDLE);
    VkPresentInfoKHR pi = {VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
    assert(vkQueuePresentKHR(H(VkQueue, 0x99), &pi) == VK_SUCCESS); assert(!strcmp(evs, "BDEQP"));
    assert(seen_ncb == 1 && !seen_ts);

    // 3. writes with copies between sets: made directly after a drain, in order
    reset();
    vkCmdDraw(one, 1, 1, 0, 0); VkCopyDescriptorSet cp = {VK_STRUCTURE_TYPE_COPY_DESCRIPTOR_SET}; vkUpdateDescriptorSets(H(VkDevice, 0x77), 0, NULL, 1, &cp);
    if (mode >= 2) assert(!strcmp(evs, "Dc")); else { vk_dq_drain(); assert(strlen(evs) == 2); }

    if (mode >= 2) {
        // 4. an end the driver refuses: that command buffer is left out of its submission; the other goes
        reset(); int base = atomic_load(&nsub);
        fail_cb = Bc; end_fail = VK_ERROR_OUT_OF_HOST_MEMORY;
        vkBeginCommandBuffer(Bc, &bi); vkCmdDraw(Bc, 1, 1, 0, 0); assert(vkEndCommandBuffer(Bc) == VK_SUCCESS);
        vkBeginCommandBuffer(C, &bi); vkCmdDraw(C, 1, 1, 0, 0); vkEndCommandBuffer(C);
        VkCommandBuffer two[2] = {Bc, C}; VkSubmitInfo s4 = {VK_STRUCTURE_TYPE_SUBMIT_INFO, NULL, 0, NULL, NULL, 2, two, 0, NULL};
        vkQueueSubmit(H(VkQueue, 0x99), 1, &s4, H(VkFence, 0x72)); wait_subs(base + 1); vk_dq_drain();
        assert(seen_ncb == 1 && seen_cbs[0] == C && seen_f == H(VkFence, 0x72) && g_dq_refused == 1 && g_nfailed == 0);
        // begun again, it is no longer refused
        fail_cb = VK_NULL_HANDLE; reset(); base = atomic_load(&nsub);
        vkBeginCommandBuffer(Bc, &bi); vkEndCommandBuffer(Bc); VkSubmitInfo s5 = {VK_STRUCTURE_TYPE_SUBMIT_INFO, NULL, 0, NULL, NULL, 1, &Bc, 0, NULL};
        vkQueueSubmit(H(VkQueue, 0x99), 1, &s5, VK_NULL_HANDLE); wait_subs(base + 1); vk_dq_drain(); assert(seen_ncb == 1 && seen_cbs[0] == Bc);
        // 5. a submission the driver refuses is made again without its command buffers (its fence still signals)
        reset(); base = atomic_load(&nsub); submit_fail = 1;
        vkBeginCommandBuffer(C, &bi); vkEndCommandBuffer(C); VkSubmitInfo s6 = {VK_STRUCTURE_TYPE_SUBMIT_INFO, NULL, 0, NULL, NULL, 1, &C, 0, NULL};
        vkQueueSubmit(H(VkQueue, 0x99), 1, &s6, H(VkFence, 0x73)); wait_subs(base + 2); vk_dq_drain();
        assert(!strcmp(evs, "BEQQ") && seen_ncb == 0 && seen_f == H(VkFence, 0x73) && nmain == 0);
        // 6. a chained structure the queue cannot copy: made directly, after a drain
        reset(); base = atomic_load(&nsub);
        VkBaseInStructure other = {VK_STRUCTURE_TYPE_PROTECTED_SUBMIT_INFO, NULL}; VkSubmitInfo s7 = {VK_STRUCTURE_TYPE_SUBMIT_INFO, &other, 0, NULL, NULL, 1, &C, 0, NULL};
        vkBeginCommandBuffer(C, &bi); vkEndCommandBuffer(C); seen_ts = 1;
        vkQueueSubmit(H(VkQueue, 0x99), 1, &s7, VK_NULL_HANDLE); assert(!strcmp(evs, "BEQ") && nmain == 1);
    }

    // 7. two producers at once, 400 command buffers each: each keeps its order (begin, its set written, bound, drawn, ended, submitted)
    vk_dq_drain(); stress = 1; int base = atomic_load(&nsub);
    pthread_t t[NP]; for (int k = 0; k < NP; k++) assert(!pthread_create(&t[k], NULL, producer, (void *)(intptr_t)k));
    for (int k = 0; k < NP; k++) pthread_join(t[k], NULL);
    wait_subs(base + NP * NI); vk_dq_drain();
    for (int k = 0; k < NP; k++) for (int i = 0; i < NI; i++) assert(st_state[k][i] == 4 && st_written[k][i]);
    vk_dq_report(1); vk_dq_report(1);
    printf("mode %d ok: %llu records, %llu drains, %llu direct, %llu descriptor writes queued\n", mode, (unsigned long long)g_dq_recs, (unsigned long long)g_dq_drains, (unsigned long long)g_dq_direct, (unsigned long long)g_dq_desc);
    return 0;
}
'''


class DeferredQueueMode2(unittest.TestCase):
    @unittest.skipUnless(shutil.which('cc') and sys.platform.startswith('linux'), 'a Linux host C compiler is required (pthread spinlocks, pthread_getcpuclockid)')
    @unittest.skipUnless(host_cc.sanitizers(), 'a host C compiler with ASan and UBSan is required')
    def test_recorder_begins_ends_submits_and_writes_sets(self):
        src = DQ.read_text(encoding='utf-8')
        lists = src[src.index('#define DQ_CMDS'):src.index('#define DQ_REAL')]
        names = sorted(set(re.findall(r'X\((vk\w+)\)', lists)))
        self.assertIn('vkUpdateDescriptorSets', names)
        header = HEADER.replace('DECLS', '\n'.join(f'extern PFN_{n} {n};' for n in names))
        main = MAIN.replace('DEFS', '\n'.join(f'PFN_{n} {n};' for n in names))
        target = ROOT / 'target/tests'
        target.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='vk-dq-mode2-', dir=target) as folder:
            d = Path(folder)
            (d / 'gfx_vk.h').write_text(header, encoding='utf-8')
            (d / 'gfx_vk_dq.c').write_text(src, encoding='utf-8')
            (d / 'check.c').write_text(main, encoding='utf-8')
            exe = d / 'check'
            subprocess.run(['cc', '-std=gnu11', '-D_GNU_SOURCE', '-Wall', '-Wextra', '-Wno-unused-function', '-Wno-missing-field-initializers', '-Werror', '-g',
                            '-fsanitize=address,undefined', '-fno-sanitize-recover=undefined', '-I' + str(ROOT / 'nxnative/dm/third_party/vulkan-headers/include'),
                            str(d / 'check.c'), '-o', str(exe), '-lpthread'], check=True)
            for mode in ('1', '2'):
                r = subprocess.run([str(exe), mode], capture_output=True, text=True, timeout=60)
                self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
                self.assertIn(f'mode {mode} ok', r.stdout)
                self.assertIn('two-stage replay per frame', r.stderr)


if __name__ == '__main__':
    unittest.main()
