// Fixture for test_vk_indirect_compute.py. Production functions replace markers.
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "runtime.h"
#include "nvn_hmap.h"
#include "gfx_vk_slot.h"
#define LOG(...) do { if (0) fprintf(stderr, __VA_ARGS__); } while (0)
#define die(...) abort()
#define DM_GAME_TEX_DYN_SLOTS 0
#define gettid() 0
#define NSLOT 2
#define MAIN_CBS 4
#define MAXPOOL 8
vkdev_t V;
#define DEFINE(f) PFN_##f f;
VK_FNS(DEFINE) VK_DEV_FNS(DEFINE)
/* STATE */
/* SLOTS */
static vkpool_t g_pool[MAXPOOL]; static int g_npool, g_pool_ord[MAXPOOL]; static uint32_t g_pool_gen;
static pthread_rwlock_t g_pool_mu = PTHREAD_RWLOCK_INITIALIZER;
static uint64_t g_set_epoch = 1, g_tex_gen, g_samp_gen, g_num_um, g_num_im, g_ubolog_key, g_tex_sseq;
static int g_x_nosetmemo, g_tracing, g_var_now, g_x_nosetc, g_x_nodrvc, g_x_setverify, g_x_nodispbar;
static int g_x_tracepass = -1, g_sub_pass, g_prof_on, g_prereset = 1;
static uint64_t g_prof_ns[8], g_fence;
static long long g_td_off = -1;
static struct { uint64_t dispatches; } g_vkst;
static nvn_hmap g_cpipes = {.klen = 24};
static pthread_mutex_t g_cpipe_mu = PTHREAD_MUTEX_INITIALIZER;
static uint64_t tick(void) { return 0; }
static VkPipeline compute_build(vkshader_t *s, uint64_t u, uint64_t i) { return (VkPipeline)2; }
VkPipelineLayout vk_pipeline_layout(VkDescriptorSetLayout a[5], int n) { return (VkPipelineLayout)2; }
static int mdi_log_now(void) { return 0; }
typedef struct { int n; uint64_t k[8][3]; } crumb_t;
static int crumbs_on(void) { return 0; }
static crumb_t *crumb_cur(void) { abort(); }
static void crumb_mark(int n) { abort(); }
static void crumb_dispatch(uint64_t key) { (void)key; }   /* gfx_vk_exec.c: a dispatch kept in the crumbs (host side only) */
void vk_device_lost(VkResult r, const char *where) { (void)r; (void)where; }   /* gfx_vk_core.c: a lost device logged (VKCHK) */
void vk_pass_ts(int a, uint32_t b, uint32_t c, uint32_t d, uint32_t e) {}
int vk_readback_ranges(void) { return 0; }
void vk_readback_unknown(void) { abort(); }
void vk_gpu_wrote(uint64_t a, uint64_t n) {}
static void sub_gpu_wrote(uint64_t a, uint64_t n) {}
uint64_t vk_fence_done(void) { return 0; }
static uint64_t mem_handle(uint64_t h) { abort(); }
static uint32_t ubo_word(int a, int b, int c) { abort(); }
static void cbuf_desc(int a, const vkshader_t *s, const dmsv_bind *b, VkDescriptorBufferInfo *d) { abort(); }
void vk_shader_c1(vkshader_t *s, VkBuffer *b, VkDeviceSize *o, VkDeviceSize *n) { abort(); }
VkBuffer vk_support_buffer(void) { abort(); }
VkBuffer vk_zero_buffer(void) { abort(); }
static void vk_cross(int k, uint64_t a, uint64_t n, const vkpool_t *p) { abort(); }
static int tex_desc_for(int st, const dmsv_bind *b, uint64_t h, VkDescriptorImageInfo *i, int storage) { abort(); }
VkSampler vk_samp_by_id(int id) { abort(); }
VkSampler vk_dummy_sampler(void) { abort(); }
vkimg_t *vk_tex_by_id(int id, const view_t **v, const tex_t **t) { abort(); }
static int view_num(const tex_t *t, const view_t *v) { abort(); }
VkBufferView vk_tbuf_view(const tex_t *t, int storage) { abort(); }
VkBufferView vk_dummy_tbuf(int storage) { abort(); }
VkImageView vk_dummy_view(int a, int b, int c, int d, int e) { abort(); }
void vk_end_pass(void) { S.pass_open = 0; }
int vk_sub_open(void) { return 1; }
VkCommandBuffer vk_cmd(void) { return (VkCommandBuffer)2; }
static void wait_fence(uint64_t fence);
void img_sub_free_(void *p) { abort(); }
// This fork's replay services around stage_set and dispatch, all off here as in a default run: the pool tables ring, dummy pools, MRT and
// light diagnostics, compute and fragment traces, texture dumps, the movie / upscaler kernels, dynamic sampler slots, descriptor pools
#include <math.h>
uint64_t g_vk_nbar, g_tex_frame, g_ubd_racy;
static int g_drv_compact = 1, g_buffer_memo = 1;   // production defaults (DM_VK_DRV_COMPACT / DM_VK_BUFFER_MEMO)
static int g_mrt_active, g_fs_buffer_collect, g_gmem_log, g_up_native_trace, g_x_nosetc_bits, g_pool_tex_base, g_pool_samp_base, g_pool_strict, g_up_diag_left;
static int g_cs_buffer_collect, g_cs_trace_collect, g_cs_trace_dump, g_cs_trace_buffers;
static unsigned g_cs_trace_seen, g_cs_images_n, g_cs_images_copies, g_cs_buffers_n, g_cs_buffers_reserved;
static uint64_t g_ring_full, g_ring_patch, g_ring_ids, g_ring_hits, g_ring_none, g_ring_ns;
static int g_pdum_lo = -1, g_pdum_hi, g_pdum_nf; static unsigned g_pdum_kinds; static uint32_t g_pdum_f[16];
static vkimg_t *g_desc_im;
static vkpool_t *pool_back_at(uint64_t addr, uint64_t len) { abort(); }   // the fixture's pools are made backed
static void sbuf_desc(int st, const dmsv_bind *b, VkDescriptorBufferInfo *bi) { abort(); }
static void light_diag_support(int st, const vkshader_t *s, VkDescriptorBufferInfo *bi) { abort(); }
static void mrt_support(int st, const vkshader_t *s, VkDescriptorBufferInfo *bi) { abort(); }
static void fs_trace_texture_note(int st, const dmsv_bind *b, uint64_t h, const VkDescriptorImageInfo *ii, vkimg_t *im) { abort(); }
static VkSampler samp_dynamic_slot(int st, int slot, int dynslots) { abort(); }
static void cs_trace_buffer_note(int st, uint64_t shader, const dmsv_bind *b, const VkDescriptorBufferInfo *source) { abort(); }
static int cs_trace_match(uint64_t key) { return 0; }
static void cs_trace_images_copy(VkCommandBuffer cb, uint64_t key, int after) { abort(); }
static void cs_trace_buffers_copy(VkCommandBuffer cb, uint64_t key, int after) { abort(); }
void vk_texdump_note(vkimg_t *im) {}
int vk_upscale_square_key(uint64_t key) { return 0; }
int vk_upscale_square(vkimg_t *src, VkImageView source_view, vkimg_t *dst, const VkImageBlit *region) { abort(); }
int vk_movie_fill_kernel(uint64_t key) { return 0; }
int vk_movie_fill(vkimg_t *out, uint32_t view_format, uint32_t level, uint32_t layer) { abort(); }
int vk_movie_fill_note(uint64_t key) { abort(); }
int vk_tex_storage_prepare(const tex_t *t, const view_t *v) { abort(); }
void mtl_tex_gpu_written(const tex_t *t) { abort(); }
static int vk_gpu_pending(uint64_t a, uint64_t n) { return 0; }
static void descriptor_pools_reset(vksub_t *s) {}
#define UNEXPECTED { fprintf(stderr, "unexpected call: %s\n", __func__); abort(); }
uint64_t g_tex_gen_why[4];
__thread const char *vk_mem_tag;
int dm_sdk_ounce(void) { return 0; }
uint64_t mtl_last_fence(void) { return g_fence; }
uint64_t nvn_replaying_job(void) { return 0; }
// no pool tables (vk_pool_n / vk_tbl_n 0), no registry snapshot: the set memo proves nothing from the registries and claims no leases
int vk_pool_n(void) { return 0; } int vk_pool_samp_n(void) { return 0; } unsigned vk_pool_kinds(void) { return 0; } int vk_tbl_n(void) { return 0; }
VkDescriptorSetLayout vk_tbl_dsl(void) UNEXPECTED
int vk_tex_id_max(void) { return 0; } int vk_samp_id_max(void) { return 0; }
void vk_tex_pool_fill(int on) UNEXPECTED
// the pool-texture prefetch (DM_VK_POOL_PREFETCH, off here) and the deferred free of an image's own memory
void vk_image_memory_soft(int on) UNEXPECTED
int vk_image_memory_soft_failed(void) { return 0; } int vk_image_memory_busy(void) { return 0; }
uint64_t vk_img_vkbytes(void) { return 0; }
void vk_img_budget_stats(uint64_t *budget, uint64_t *evicted, uint64_t *evicted_bytes, uint64_t *sweeps) { *budget = *evicted = *evicted_bytes = *sweeps = 0; }
void img_own_free_(VkDeviceMemory m) UNEXPECTED
int vk_tex_pool_ok(const tex_t *t) UNEXPECTED
int vk_registry_snapshot(uint64_t *textures, uint64_t *samplers) { return 0; }
uint64_t vk_reglog_visible_end(uint64_t before) { return 0; } uint64_t vk_reglog_head(void) { return 0; } int vk_reglog_pending(uint64_t p, uint64_t before) { return 0; }
uint64_t vk_reglog_take(uint64_t from, uint64_t before, int *tids, int max, int *n) UNEXPECTED
uint64_t vk_droplog_seq(void) { return 0; }
int vk_droplog_take(uint64_t from, const void **out, int max, uint64_t *to) { *to = 0; return 0; }
uint64_t vk_descriptor_lease(VkDescriptorSet set, int rewrite) { return 0; }
int vk_descriptor_claim(VkDescriptorSet set, uint64_t lease) { return 0; }
int vk_alloc_sets_owned(VkDescriptorSetLayout l, int n, VkDescriptorSet *out) UNEXPECTED
VkBuffer vk_gmem_map(VkDeviceSize *len) UNEXPECTED
VkBuffer vk_gmem_log_disabled(void) UNEXPECTED
const vkfmt_t *vk_fmt(int nvn_format) UNEXPECTED
vkimg_t *vk_tex(const tex_t *t) UNEXPECTED
VkImageView vk_tex_view(const tex_t *t, const view_t *v, int dim, int arrayed, int view_kind, vkimg_t **img_out) UNEXPECTED
void vk_tex_prepare(VkCommandBuffer cb, vkimg_t *im) UNEXPECTED

// Vulkan allocation and command sink: host and device argument bytes are separate.
typedef struct { uint8_t *bytes; size_t size; VkBufferUsageFlags usage; int write_stage, visible; } Buffer;
static Buffer *buffer(VkBuffer b) { assert(b); return (Buffer *)b; }
static VkBuffer make_buffer(size_t size, VkBufferUsageFlags usage) {
    Buffer *b = calloc(1, sizeof *b); assert(b); b->bytes = calloc(1, size); assert(b->bytes); b->size = size; b->usage = usage; return (VkBuffer)b;
}
static Buffer *buffers[64]; static int nbuffer;
static VkResult create_buffer(VkDevice d, const VkBufferCreateInfo *c, const VkAllocationCallbacks *a, VkBuffer *out) {
    *out = make_buffer(c->size, c->usage); buffers[nbuffer++] = buffer(*out); return VK_SUCCESS;
}
static void requirements(VkDevice d, VkBuffer b, VkMemoryRequirements *r) { *r = (VkMemoryRequirements){buffer(b)->size, 16, 1}; }
static VkResult allocate(VkDevice d, const VkMemoryAllocateInfo *i, const VkAllocationCallbacks *a, VkDeviceMemory *out) {
    *out = (VkDeviceMemory)buffers[nbuffer - 1]; return VK_SUCCESS;
}
static VkResult bind_memory(VkDevice d, VkBuffer b, VkDeviceMemory m, VkDeviceSize off) { assert((void *)b == (void *)m && off == 0); return VK_SUCCESS; }
static VkResult map_memory(VkDevice d, VkDeviceMemory m, VkDeviceSize o, VkDeviceSize n, VkMemoryMapFlags f, void **out) { *out = ((Buffer *)m)->bytes; return VK_SUCCESS; }
uint32_t vk_mem_type(uint32_t bits, VkMemoryPropertyFlags want, VkMemoryPropertyFlags avoid) {
    assert((want & (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) == (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)); return 0;
}
/* USAGE */
static struct { VkDescriptorBufferInfo bi[96]; VkDescriptorType type[96]; } descriptors[256];
static unsigned nset;
VkDescriptorSet vk_alloc_set(VkDescriptorSetLayout layout) { assert(++nset < 256); return (VkDescriptorSet)(uintptr_t)nset; }
static void update_sets(VkDevice d, uint32_t n, const VkWriteDescriptorSet *w, uint32_t nc, const VkCopyDescriptorSet *c) {
    for (unsigned i = 0; i < n; i++) { unsigned s = (unsigned)(uintptr_t)w[i].dstSet, b = w[i].dstBinding;
        assert(s && s <= nset && b < 96 && w[i].descriptorCount == 1 && w[i].pBufferInfo);
        descriptors[s].bi[b] = *w[i].pBufferInfo; descriptors[s].type[b] = w[i].descriptorType; }
}
enum { PRODUCE, COPY, BARRIER, DISPATCH };
enum { READ_TRANSFER = 1, READ_UNIFORM = 2, READ_INDIRECT = 4 };
typedef struct { int op; VkBuffer src, dst; VkBufferCopy copy; VkPipelineStageFlags from, to; VkMemoryBarrier barrier;
    uint32_t values[3], offsets[96]; unsigned set, ndyn; VkDescriptorBufferInfo c0[2], c2; int nc0, indirect; vkshader_t shader; } Event;
static Event events[256]; static int nevent, completed, ndispatch, ncopies;
static unsigned bound_set, bound_ndyn; static uint32_t bound_offsets[96];
static Event *event(int op) { assert(nevent < 256); Event *e = &events[nevent++]; memset(e, 0, sizeof *e); e->op = op; return e; }
static void pipeline_barrier(VkCommandBuffer cb, VkPipelineStageFlags from, VkPipelineStageFlags to, VkDependencyFlags f,
    uint32_t nm, const VkMemoryBarrier *m, uint32_t nb, const VkBufferMemoryBarrier *b, uint32_t ni, const VkImageMemoryBarrier *im) {
    assert(!S.pass_open && nm == 1 && !nb && !ni); Event *e = event(BARRIER); e->from = from; e->to = to; e->barrier = *m;
}
static void copy_buffer(VkCommandBuffer cb, VkBuffer src, VkBuffer dst, uint32_t n, const VkBufferCopy *r) {
    assert(!S.pass_open && n > 0);
    for (unsigned i = 0; i < n; i++) { assert(!(r[i].srcOffset % 4) && !(r[i].dstOffset % 4) && !(r[i].size % 4));
        assert(r[i].srcOffset + r[i].size <= buffer(src)->size && r[i].dstOffset + r[i].size <= buffer(dst)->size);
        assert(buffer(src)->usage & VK_BUFFER_USAGE_TRANSFER_SRC_BIT); assert(buffer(dst)->usage & VK_BUFFER_USAGE_TRANSFER_DST_BIT);
        Event *e = event(COPY); e->src = src; e->dst = dst; e->copy = r[i]; ncopies++; }
}
static void bind_pipeline(VkCommandBuffer cb, VkPipelineBindPoint bp, VkPipeline p) {}
static void bind_sets(VkCommandBuffer cb, VkPipelineBindPoint bp, VkPipelineLayout l, uint32_t first,
    uint32_t n, const VkDescriptorSet *s, uint32_t ndyn, const uint32_t *off) {
    assert(n == 1 && first == 0); bound_set = (unsigned)(uintptr_t)*s; bound_ndyn = ndyn; memcpy(bound_offsets, off, ndyn * sizeof *off);
}
static void dispatch_event(VkBuffer args, VkDeviceSize off, uint32_t x, uint32_t y, uint32_t z);
static void direct_dispatch(VkCommandBuffer cb, uint32_t x, uint32_t y, uint32_t z) { dispatch_event(0, 0, x, y, z); }
static void indirect_dispatch(VkCommandBuffer cb, VkBuffer b, VkDeviceSize off) {
    assert(off % 4 == 0 && off + 12 <= buffer(b)->size && (buffer(b)->usage & VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT)); dispatch_event(b, off, 0, 0, 0);
}
uint64_t g_vk_pass_key;   /* gfx_vk_core.c: the dispatch being timed (droidplay-core b631fa11b) */
/* CORE */
/* REPLAY */
/* LIFECYCLE */
static VkDescriptorSet cached_stage(int st, vkshader_t *s, uint32_t *offsets) { return /* STAGE_CALL */; }

static VkDescriptorBufferInfo descriptor(unsigned set, unsigned binding, const uint32_t *offsets) {
    VkDescriptorBufferInfo d = descriptors[set].bi[binding];
    if (descriptors[set].type[binding] == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC) {
        unsigned pos = 0; for (unsigned i = 0; i < binding; i++) pos += descriptors[set].type[i] == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
        d.offset += offsets[pos];
    }
    return d;
}
static void dispatch_event(VkBuffer args, VkDeviceSize off, uint32_t x, uint32_t y, uint32_t z) {
    Event *e = event(DISPATCH); e->src = args; e->copy.srcOffset = off; e->indirect = args != 0; e->values[0] = x; e->values[1] = y; e->values[2] = z;
    e->set = bound_set; e->ndyn = bound_ndyn; e->shader = *S.sh[5]; memcpy(e->offsets, bound_offsets, sizeof e->offsets);
    for (int i = 0; i < e->shader.nbind; i++) { const dmsv_bind *b = &e->shader.b[i];
        if (b->kind == RK_DRV && b->reg == 0) { assert(e->nc0 < 2); e->c0[e->nc0++] = descriptor(e->set, b->binding, e->offsets); }
        if (b->kind == RK_DRV && b->reg == 2) e->c2 = descriptor(e->set, b->binding, e->offsets);
    }
    if (e->nc0 == 2) assert(e->c0[0].buffer == e->c0[1].buffer && e->c0[0].offset == e->c0[1].offset);
    ndispatch++;
}
static void run_gpu(void) {
    for (; completed < nevent; completed++) { Event *e = &events[completed];
        if (e->op == PRODUCE) { memcpy(buffer(e->dst)->bytes + e->copy.dstOffset, e->values, e->copy.size); buffer(e->dst)->write_stage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT; buffer(e->dst)->visible = 0; }
        if (e->op == BARRIER) for (int i = 0; i < nbuffer; i++) { Buffer *b = buffers[i];
            if (!(e->from & (VK_PIPELINE_STAGE_ALL_COMMANDS_BIT | b->write_stage))) continue;
            if (!(e->barrier.srcAccessMask & (VK_ACCESS_MEMORY_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT))) continue;
            int all = e->barrier.dstAccessMask & VK_ACCESS_MEMORY_READ_BIT;
            if ((e->to & (VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_ALL_COMMANDS_BIT)) && (all || (e->barrier.dstAccessMask & VK_ACCESS_TRANSFER_READ_BIT))) b->visible |= READ_TRANSFER;
            if ((e->to & (VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_ALL_COMMANDS_BIT)) && (all || (e->barrier.dstAccessMask & VK_ACCESS_UNIFORM_READ_BIT))) b->visible |= READ_UNIFORM;
            if ((e->to & (VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_ALL_COMMANDS_BIT)) && (all || (e->barrier.dstAccessMask & VK_ACCESS_INDIRECT_COMMAND_READ_BIT))) b->visible |= READ_INDIRECT;
        }
        if (e->op == COPY) { Buffer *s = buffer(e->src), *d = buffer(e->dst);
            assert(!s->write_stage || (s->visible & READ_TRANSFER)); memcpy(d->bytes + e->copy.dstOffset, s->bytes + e->copy.srcOffset, e->copy.size);
            d->write_stage = VK_PIPELINE_STAGE_TRANSFER_BIT; d->visible = 0; }
        if (e->op == DISPATCH) { uint32_t groups[3]; memcpy(groups, e->values, 12);
            if (e->indirect) { Buffer *b = buffer(e->src); assert(!b->write_stage || (b->visible & READ_INDIRECT)); memcpy(groups, b->bytes + e->copy.srcOffset, 12); }
            memcpy(e->values, groups, 12);
            for (int c = 0; c < e->nc0; c++) { VkDescriptorBufferInfo d = e->c0[c]; Buffer *b = buffer(d.buffer);
                assert(!b->write_stage || (b->visible & READ_UNIFORM));
                if (memcmp(b->bytes + d.offset + 0x200, groups, 12)) {
                    const uint32_t *actual = (const void *)(b->bytes + d.offset + 0x200);
                    fprintf(stderr, "c0 counts %u,%u,%u differ from dispatch %u,%u,%u\n", actual[0],actual[1],actual[2],groups[0],groups[1],groups[2]); abort(); }
            }
        }
    }
}
static int waited;
static void wait_fence(uint64_t fence) { assert(g_slot[0].blk[0].used != 0); run_gpu(); waited++; }
int mtl_wait(uint64_t fence, uint64_t timeout) { wait_fence(fence); return 1; }
static void make_pool(uint64_t lo, uint64_t hi, uint64_t base, uint64_t end, uint64_t boff) {
    int i = g_npool++; assert(i < MAXPOOL); VkBuffer b = make_buffer(boff + hi - lo, POOL_USAGE); buffers[nbuffer++] = buffer(b);
    g_pool[i] = (vkpool_t){.lo=lo,.hi=hi,.base=base,.end=end,.boff=boff,.buf=b,.backed=1};   // (the fork backs lazy pools at their first GPU use) g_pool_ord[i]=i; g_pool_gen++;
}
static void produce(uint64_t addr, const uint32_t values[3]) {
    for (int i = 0; i < 3; i++) { VkBuffer b; VkDeviceSize o; assert(vk_buf_at(addr + 4*i, 4, &b, &o));
        Event *e = event(PRODUCE); e->dst=b; e->copy.dstOffset=o; e->copy.size=4; e->values[0]=values[i]; }
}
static dmsv_bind binds[] = {
    {.kind=RK_DRV,.desc=RD_UBO,.binding=7,.count=1,.reg=0},
    {.kind=RK_DRV,.desc=RD_UBO,.binding=3,.count=1,.reg=0},
    {.kind=RK_DRV,.desc=RD_UBO,.binding=9,.count=1,.reg=2}};
static vkshader_t shader = {.key=12,.dsl=(VkDescriptorSetLayout)1,.nbind=3,.b=binds,.stage=5};
static vkshader_t noc0 = {.key=13,.dsl=(VkDescriptorSetLayout)2,.nbind=1,.b=binds+2,.stage=5};
static uint32_t cpu_args[8];
static const uint32_t a[3]={7,3,2}, b[3]={11,5,1};
static void invoke(uint64_t addr) { p_dispatch d = {.ind=addr}; dispatch(&d); }
static void setup(int dynamic) {
    vkCreateBuffer=create_buffer; vkGetBufferMemoryRequirements=requirements; vkAllocateMemory=allocate; vkBindBufferMemory=bind_memory; vkMapMemory=map_memory;
    vkUpdateDescriptorSets=update_sets; vkCmdPipelineBarrier=pipeline_barrier; vkCmdCopyBuffer=copy_buffer;
    vkCmdBindPipeline=bind_pipeline; vkCmdBindDescriptorSets=bind_sets; vkCmdDispatch=direct_dispatch; vkCmdDispatchIndirect=indirect_dispatch;
    V.props.limits.minUniformBufferOffsetAlignment=256; V.props.limits.minStorageBufferOffsetAlignment=256;
    g_cur=0; S.sh[5]=&shader; S.tex[5][0]=0x5678; S.tex[0][0]=0x1234;
    if (dynamic) { memset(shader.dyn,1,3); shader.ndyn=3; noc0.dyn[0]=1; noc0.ndyn=1; }
}
int main(int argc, char **argv) {
    assert(argc==3); setup(atoi(argv[2])); uint64_t addr=(uintptr_t)cpu_args;
    if (!strcmp(argv[1],"counts") || !strcmp(argv[1],"lifetime")) {
        make_pool(addr,addr+sizeof cpu_args,addr,addr+sizeof cpu_args,64);
        uint32_t offsets[96]; VkDescriptorSet graphics=cached_stage(0,&shader,offsets);
        VkDescriptorBufferInfo gd=descriptor((unsigned)(uintptr_t)graphics,7,offsets);
        p_dispatch zero={0}, direct={.x=2,.y=4,.z=6};
        dispatch(&zero); Event zero_before=events[nevent-2];
        dispatch(&direct); Event before=events[nevent-2];
        produce(addr,a); invoke(addr); int first_i=nevent-2; Event first=events[first_i];
        produce(addr,b); invoke(addr); int second_i=nevent-2; Event second=events[second_i];
        dispatch(&direct); Event after=events[nevent-2];
        dispatch(&zero); Event zero_after=events[nevent-2];
        assert(zero_before.c0[0].buffer==zero_after.c0[0].buffer && zero_before.c0[0].offset==zero_after.c0[0].offset);
        assert(before.op==DISPATCH && first.op==DISPATCH && second.op==DISPATCH && after.op==DISPATCH);
        assert(first.c0[0].buffer!=second.c0[0].buffer || first.c0[0].offset!=second.c0[0].offset);
        assert(before.c0[0].buffer==after.c0[0].buffer && before.c0[0].offset==after.c0[0].offset);
        assert(before.c2.buffer==first.c2.buffer && before.c2.offset==first.c2.offset);
        assert(first.c2.buffer==second.c2.buffer && first.c2.offset==second.c2.offset);
        S.sh[5]=&noc0; produce(addr,a); invoke(addr); assert(events[nevent-2].nc0==0);
        assert(cpu_args[0]==0 && cpu_args[1]==0 && cpu_args[2]==0);
        if (!strcmp(argv[1],"lifetime")) { g_slot[0].fence=1; slot_recycle(&g_slot[0]); assert(waited==1 && g_slot[0].blk[0].used==0); }
        else run_gpu();
        assert(*(uint32_t *)(buffer(gd.buffer)->bytes+gd.offset+0x644)==0);
        assert(*(uint64_t *)(buffer(first.c2.buffer)->bytes+first.c2.offset+0x20)==0x5678);
        assert(!memcmp(events[first_i].values,a,12)); assert(!memcmp(events[second_i].values,b,12));
        assert(!memcmp(events[nevent-2].values,a,12));
        assert(!memcmp(buffer(zero_before.c0[0].buffer)->bytes+zero_before.c0[0].offset+0x200,(uint32_t[3]){0,0,0},12));
        assert(ndispatch==7 && ncopies==3);
    } else if (!strcmp(argv[1],"two") || !strcmp(argv[1],"three")) {
        int three=!strcmp(argv[1],"three");
        make_pool(addr,addr+4,addr,addr+12,64);
        make_pool(addr+4,addr+(three?8:12),addr,addr+12,128);
        if (three) { make_pool(addr+8,addr+12,addr,addr+12,256); S.sh[5]=&noc0; }
        produce(addr,a); invoke(addr); assert(ndispatch==1); run_gpu();
        Event *e=&events[nevent-2]; assert(!memcmp(e->values,a,12)); assert(ncopies==(three?3:2));
    } else if (!strcmp(argv[1],"invalid")) {
        make_pool(addr,addr+4,addr,addr+12,64); // remaining eight bytes unmapped
        invoke(addr); invoke(addr+1); invoke(UINT64_MAX-3); invoke(addr-4);
        assert(!ndispatch && !ncopies);
        make_pool(addr+4,addr+12,addr+4,addr+12,128); // adjacent but a different pool
        invoke(addr); assert(!ndispatch && !ncopies);
        g_pool[1].base=addr; g_pool[1].boff=127; g_pool_gen++; invoke(addr); assert(!ndispatch && !ncopies);
        g_pool[1].boff=UINT64_MAX-3; g_pool_gen++; invoke(addr); assert(!ndispatch && !ncopies);
        g_pool[1].boff=128; g_pool[1].end=addr+8; g_pool_gen++; invoke(addr); assert(!ndispatch && !ncopies);
    } else if (!strcmp(argv[1],"deferred")) {
        make_pool(addr,addr+sizeof cpu_args,addr,addr+sizeof cpu_args,64); S.pass_open=1;
        vk_sub_update(addr,a,12); assert(!ncopies); invoke(addr); assert(!S.pass_open && ndispatch==1); run_gpu();
        assert(!memcmp(events[nevent-2].values,a,12)); assert(cpu_args[0]==0);
    } else abort();
    return 0;
}
