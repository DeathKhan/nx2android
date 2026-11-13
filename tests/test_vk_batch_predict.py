"""Batch prediction (gfx_vk_exec.c predict_rest) reads ahead across render target changes.

Kirby and the Forgotten Land, Odin 3, 2026-11-12 21:53 CST: an area load's submission had 68 passes, many of one draw each, and 51 pipelines built on the
replay thread (1.2 s). The first walk stopped at the next OP_SET_RT and queued nothing. This replays that pattern: a miss in pass 0, then 67 one-draw passes
on targets of their own formats. The walk must queue the passes' keys with their own attachment formats up to its cap, walk again at a miss past where it
stopped, not walk again at a miss within its reach, and leave the bound state and attachments as they were."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


def function(source, declaration):
    start = source.index(declaration)
    brace = source.index('{', start)
    depth, pos = 1, brace + 1
    while depth:
        depth += (source[pos] == '{') - (source[pos] == '}')
        pos += 1
    return source[start:pos]


def between(text, start, end):
    i = text.index(start)
    return text[i:text.index(end, i) + len(end)]


@unittest.skipUnless(shutil.which('cc'), 'a host C compiler is required')
class BatchPredict(unittest.TestCase):
    def test_read_ahead_across_render_targets(self):
        src = (ROOT / 'nxnative/dm/android/gfx_vk_exec.c').read_text(encoding='utf-8')
        pkey = between(src, 'typedef struct {\n    uint64_t key[3]; uint32_t topo', '} pkey;')
        code = r'''
#define VK_NO_PROTOTYPES
#include <vulkan/vulkan_core.h>
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include "nxnative/dm/gfx/nvn/nvn_obj.h"
#include "nxnative/dm/gfx/nvn/nvn_hmap.h"
typedef void *VkPipelineX;
#define PIPE_PENDING ((VkPipeline)(uintptr_t)2)
enum { APQ = 256 };
typedef struct { uint64_t key; } vkshader_t;
typedef struct vkimg { int fdz_frame; } vkimg_t;
PKEY
static struct {
    vkshader_t *sh[6]; float vp[16][4]; blend_st blend[8]; color_st color; chmask_st chmask; ms_st ms; poly_st poly; ds_st ds;
    uint32_t vattr[16]; vs_st vstream[16]; NVNbufferRange vbuf[16]; int sref[2], smask[2], swmask[2]; p_poly_offset pofs; float blendc[4];
    int depth_clamp, rast_discard, prim_restart; float alpha_ref;
    tex_t rt_col[8], rt_dep; view_t rt_cview[8], rt_dview; uint8_t rt_has_col[8], rt_has_cview[8], rt_has_dep, rt_has_dview; int rt_n;
    VkFormat cfmt[8], dfmt; VkImageAspectFlags daspect; uint32_t samples; int ncol;
} S;
static struct { vkimg_t *im; VkImageView v; uint32_t fmt, layers, w, h; } P_att[9];
static vkimg_t g_s0_image;
static nvn_hmap g_pipes = {.klen = sizeof(pkey)}, g_pred = {.klen = sizeof(pkey)};
static pthread_mutex_t g_pipe_mu = PTHREAD_MUTEX_INITIALIZER;
static uint64_t g_pred_hit; static int g_async_pipe = 1, g_draw_i;
static const uint8_t *g_la_at, *g_la_ret[16]; static int g_la_depth, g_la_span; static uint32_t g_la_epoch;
const uint64_t *vk_rest_h; int vk_rest_n;
static uint64_t tick(void) { return 0; }
static void log_ignore(const char *f, ...) { (void)f; }
#define VKLOG(...) log_ignore(__VA_ARGS__)
static vkshader_t shaders[80];
static void bind_program(const p_bind_program *bp) { S.sh[0] = &shaders[(uintptr_t)bp->prog]; S.sh[1] = &shaders[(uintptr_t)bp->prog + 1]; }
// the key as prepare_draw would make it, reduced to what the walk changes here: the shaders and the attachments
static void pkey_fill(pkey *kp, int prim, int indexed, int var, int predict) {
    (void)var; assert(predict); memset(kp, 0, sizeof *kp); kp->key[0] = S.sh[0]->key; kp->key[1] = S.sh[1]->key; kp->topo = (uint32_t)prim; kp->restart = (uint32_t)indexed;
    kp->ncol = (uint32_t)S.ncol; for (int i = 0; i < S.ncol; i++) kp->cfmt[i] = S.cfmt[i]; kp->dfmt = S.dfmt; kp->samples = S.samples;
}
static vkimg_t depth_img;
int vk_rt_predict(const tex_t *t, const view_t *vw, VkFormat *fmt, uint32_t *samples, VkImageAspectFlags *aspect, vkimg_t **img_out) {
    (void)vw; if (!t || t->magic != MAGIC('T', 'E', 'X', 0)) return 0;
    *fmt = (VkFormat)t->d.format; *samples = 1; *aspect = t->d.format == VK_FORMAT_D32_SFLOAT ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT;
    *img_out = t->d.format == VK_FORMAT_D32_SFLOAT ? &depth_img : NULL; return 1;
}
static int slot0_fill_wanted(int ncol, int slot0_view, uint32_t samples, uint32_t layers) { (void)ncol; (void)slot0_view; (void)samples; (void)layers; return 0; }
static pkey queued[512]; static int nqueued;
static int apq_push(const pkey *k) { queued[nqueued++] = *k; pthread_mutex_lock(&g_pipe_mu); nvn_hm_put(&g_pipes, k, PIPE_PENDING); pthread_mutex_unlock(&g_pipe_mu); return 1; }
SETRT
PREDICT
// ---- the 21:53 area load: pass 0 (where the miss is) and 67 passes of one draw each, each on a target of its own format ----
enum { NPASS = 68 };
static uint8_t buf[1 << 16]; static size_t at;
static tex_t tex[NPASS], depth;
static void put(uint32_t op, const void *rec, uint32_t n) { op_hdr h = {op, 8 + ((n + 7) & ~7u)}; memcpy(buf + at, &h, 8); memcpy(buf + at + 8, rec, n); at += h.size; }
static const uint8_t *pass_draw[NPASS];
int main(void) {
    for (int i = 0; i < 80; i++) shaders[i].key = 0x1000 + (uint64_t)i;
    depth.magic = MAGIC('T', 'E', 'X', 0); depth.d.format = VK_FORMAT_D32_SFLOAT;
    for (int i = 0; i < NPASS; i++) {
        tex[i].magic = MAGIC('T', 'E', 'X', 0); tex[i].d.format = i % 3 == 0 ? VK_FORMAT_R16G16B16A16_SFLOAT : i % 3 == 1 ? VK_FORMAT_R8G8B8A8_UNORM : VK_FORMAT_B10G11R11_UFLOAT_PACK32;
        p_set_rt r; memset(&r, 0, sizeof r); r.n = 1; r.color[0] = &tex[i]; r.depth = i % 2 ? &depth : NULL; put(OP_SET_RT, &r, sizeof r);
        p_bind_program bp = {(const prog_t *)(uintptr_t)(i % 40), 0x1f}; put(OP_BIND_PROGRAM, &bp, sizeof bp);
        p_draw d; memset(&d, 0, sizeof d); d.prim = 4; d.count = 3; put(OP_DRAW, &d, sizeof d); pass_draw[i] = buf + at;
    }
    put(OP_END, NULL, 0);
    // the miss: pass 0's draw, replayed (its state bound, the walk starts after it)
    p_set_rt r0; memset(&r0, 0, sizeof r0); r0.n = 1; r0.color[0] = &tex[0]; pred_set_rt(&r0);
    p_bind_program bp0 = {(const prog_t *)0, 0x1f}; bind_program(&bp0);
    pkey cur; pkey_fill(&cur, 4, 0, 0, 1);
    __typeof__(S) before; memcpy(&before, &S, sizeof S);
    g_la_at = pass_draw[0]; g_la_epoch = 1; g_draw_i = 1;
    predict_rest(&cur);
    // the cap (64 by default): the next 64 passes' keys, each with its own target's format and depth
    assert(g_pred_walks == 1 && nqueued == 64 && g_pred_last_queued == 64);
    for (int i = 0; i < 64; i++) {
        int pass = i + 1; const pkey *k = &queued[i];
        assert(k->key[0] == 0x1000 + (uint64_t)(pass % 40) && k->key[1] == 0x1001 + (uint64_t)(pass % 40));
        assert(k->ncol == 1 && k->cfmt[0] == (uint32_t)tex[pass].d.format);
        assert(k->dfmt == (pass % 2 ? (uint32_t)VK_FORMAT_D32_SFLOAT : 0u));
        assert(nvn_hm_get(&g_pred, k));   // marked predicted: recorded only when a draw takes it
    }
    assert(!memcmp(&before, &S, sizeof S) && !P_att[0].im && !P_att[8].im);   // the bound state and the attachments as they were
    // a miss within the walk's reach (a misprediction): no walk again
    g_draw_i = 30; predict_rest(&cur); assert(g_pred_walks == 1 && nqueued == 64);
    // a miss past where the walk stopped (pass 65, at its cap): the walk goes on from there and queues the rest
    g_draw_i = 66; g_la_at = pass_draw[65];
    p_set_rt r65; memset(&r65, 0, sizeof r65); r65.n = 1; r65.color[0] = &tex[65]; r65.depth = &depth; pred_set_rt(&r65);
    p_bind_program bp65 = {(const prog_t *)(uintptr_t)(65 % 40), 0x1f}; bind_program(&bp65); pkey_fill(&cur, 4, 0, 0, 1);
    predict_rest(&cur);
    assert(g_pred_walks == 2 && nqueued == 66 && queued[64].cfmt[0] == (uint32_t)tex[66].d.format && queued[65].cfmt[0] == (uint32_t)tex[67].d.format);
    // to the submission's end: no walk again in this submission; the next submission walks
    g_draw_i = 68; predict_rest(&cur); assert(g_pred_walks == 2);
    g_la_epoch = 2; g_la_at = pass_draw[66]; predict_rest(&cur); assert(g_pred_walks == 3 && nqueued == 66);   // (pass 67's key is queued already)
    puts("ok"); return 0;
}
'''
        code = code.replace('PKEY', pkey).replace('SETRT', function(src, 'static void pred_set_rt(')) \
                   .replace('PREDICT', 'static int g_pred_walks, g_pred_last_queued;\n' + function(src, 'static void predict_rest(const pkey *cur) {'))
        target = ROOT / 'target/tests'; target.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='vk-batch-predict-', dir=target) as folder:
            c, exe = Path(folder) / 'check.c', Path(folder) / 'check'
            c.write_text(code, encoding='utf-8')
            subprocess.run(['cc', '-std=gnu11', '-Wall', '-Wextra', '-Werror', '-Wno-unused-function', '-Wno-misleading-indentation', '-pthread', '-fsanitize=address,undefined',
                            '-I', str(ROOT), '-I', str(ROOT / 'nxnative/dm/gfx/nvn'), '-I', str(ROOT / 'nxnative/dm/gfx/nvn/include'), '-I', str(ROOT / 'nxnative/dm/third_party/vulkan-headers/include'),
                            str(c), '-o', str(exe)], check=True)
            out = subprocess.run([str(exe)], check=True, timeout=20, capture_output=True, text=True).stdout
            self.assertIn('ok', out)


if __name__ == '__main__':
    unittest.main()
