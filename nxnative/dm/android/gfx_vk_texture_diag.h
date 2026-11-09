// Bounded live texture-generation diagnostics. No graphics or registry mutation.
// Included by gfx_vk_tex.c after its types and g_tex_gen declaration.
enum { TEX_DIAG_REGISTER, TEX_DIAG_FIRST_USE, TEX_DIAG_VIEW_DROP, TEX_DIAG_ORPHAN, TEX_DIAG_CAUSES };
uint64_t nvn_jobs_queued(void);
uint64_t nvn_replaying_job(void);
static pthread_mutex_t g_tex_diag_mu = PTHREAD_MUTEX_INITIALIZER;
static int g_tex_diag_active, g_tex_diag_frames, g_tex_diag_left;
static uint64_t g_tex_diag_causes[TEX_DIAG_CAUSES], g_tex_diag_start;
static int tex_diag_begin(unsigned cause) { // on success, caller owns the diagnostic leaf mutex
    if (!__atomic_load_n(&g_tex_diag_active, __ATOMIC_ACQUIRE)) return 0;
    pthread_mutex_lock(&g_tex_diag_mu);
    if (!g_tex_diag_active || cause >= TEX_DIAG_CAUSES) { pthread_mutex_unlock(&g_tex_diag_mu); return 0; }
    g_tex_diag_causes[cause]++;
    if (!g_tex_diag_left) { pthread_mutex_unlock(&g_tex_diag_mu); return 0; }
    g_tex_diag_left--; return 1;
}
static int tex_diag_diff(const void *a, const void *b, size_t n) {
    if (!a || !b) return -1;
    for (size_t i = 0; i < n; i++) if (((const uint8_t *)a)[i] != ((const uint8_t *)b)[i]) return (int)i;
    return -1;
}
static void vk_tex_diag_native(unsigned cause, const tex_t *t, const vkimg_t *im) {
    if (!tex_diag_begin(cause)) return;
    const char *name = cause == TEX_DIAG_FIRST_USE ? "first-use" : cause == TEX_DIAG_VIEW_DROP ? "view-drop" : "orphan";
    VKLOG("texture generation: %s gen %llu replay %llu queued %llu serial %llu image %p pool %p+%#llx fmt %#x %ux%u levels %u pins %u",
        name, (unsigned long long)g_tex_gen, (unsigned long long)nvn_replaying_job(), (unsigned long long)nvn_jobs_queued(),
        (unsigned long long)(t ? t->serial : 0), im ? (void *)im->img : NULL, im ? (void *)(uintptr_t)im->pool : NULL,
        im ? (unsigned long long)im->off : 0, im ? im->nvn_format : 0, im ? im->w : 0, im ? im->h : 0, im ? im->levels : 0, im ? im->pins : 0);
    pthread_mutex_unlock(&g_tex_diag_mu);
}
static void vk_tex_diag_registration(int id, const tex_t *old, const view_t *ov, const tex_t *next, const view_t *nv,
                                     uint64_t before, uint64_t after, uint64_t registration) {
    if (!tex_diag_begin(TEX_DIAG_REGISTER)) return;
    tex_t zero = {0}; view_t vz = {0}; const tex_t *a = old ? old : &zero, *b = next ? next : &zero;
    const view_t *av = ov ? ov : &vz, *bv = nv ? nv : &vz;
    uint64_t al = 0, bl = 0; memcpy(&al, a->layout, sizeof al); memcpy(&bl, b->layout, sizeof bl);
    VKLOG("texture generation: registration id %d gen %llu registration %llu replay %llu queued %llu visible-seq %llu->%llu present %d->%d view %d->%d first-byte texture %d view %d",
        id, (unsigned long long)g_tex_gen, (unsigned long long)registration, (unsigned long long)nvn_replaying_job(),
        (unsigned long long)nvn_jobs_queued(), (unsigned long long)before, (unsigned long long)after,
        old != NULL, next != NULL, ov != NULL, nv != NULL, tex_diag_diff(old, next, sizeof *a), tex_diag_diff(ov, nv, sizeof *av));
    VKLOG("texture generation: id %d storage serial %llu->%llu flags %#x->%#x fmt %#x->%#x target %d->%d size %ux%ux%u/%u->%ux%ux%u/%u mem %p->%p pool %p->%p offset %#llx->%#llx bytes %#llx->%#llx stride %u->%u layout %d/%#llx->%d/%#llx swizzle %d/%#x/%d->%d/%#x/%d",
        id, (unsigned long long)a->serial, (unsigned long long)b->serial, a->d.flags, b->d.flags, a->d.format, b->d.format, a->d.target, b->d.target,
        a->d.width, a->d.height, a->d.depth, a->d.levels, b->d.width, b->d.height, b->d.depth, b->d.levels,
        (const void *)a->mem, (const void *)b->mem, (const void *)a->pool, (const void *)b->pool, (unsigned long long)a->off, (unsigned long long)b->off,
        (unsigned long long)a->L.size, (unsigned long long)b->L.size, a->d.stride, b->d.stride, a->has_layout, (unsigned long long)al, b->has_layout, (unsigned long long)bl, a->set_swz, a->swz, a->stencil, b->set_swz, b->swz, b->stencil);
    VKLOG("texture generation: id %d view flags %#x->%#x format %#x->%#x target %u->%u mip %u+%u->%u+%u layers %u+%u->%u+%u swizzle %u/%u/%u/%u->%u/%u/%u/%u reserved %#x->%#x",
        id, *(const uint8_t *)av, *(const uint8_t *)bv, av->format, bv->format, av->target, bv->target,
        av->base_level, av->nlevels, bv->base_level, bv->nlevels, av->min_layer, av->nlayers, bv->min_layer, bv->nlayers,
        av->swz[0], av->swz[1], av->swz[2], av->swz[3], bv->swz[0], bv->swz[1], bv->swz[2], bv->swz[3], av->_rsv, bv->_rsv);
    pthread_mutex_unlock(&g_tex_diag_mu);
}
static void vk_tex_diag_poll(void) { // once per present, never under image/registry locks
    const char *e = getenv("DM_VK_TEX_DIAG"); char *end = NULL; long request = e ? strtol(e, &end, 10) : 0;
    if (!e || end == e || *end || request < 0) request = 0;
    if (!__atomic_load_n(&g_tex_diag_active, __ATOMIC_ACQUIRE) && !request) return;
    pthread_mutex_lock(&g_tex_diag_mu);
    if (g_tex_diag_active && !--g_tex_diag_frames) {
        __atomic_store_n(&g_tex_diag_active, 0, __ATOMIC_RELEASE);
        VKLOG("texture generation summary (64 presents): gen %llu->%llu, registration %llu, first-use %llu, view-drop %llu, orphan %llu; diagnostics off",
            (unsigned long long)g_tex_diag_start, (unsigned long long)g_tex_gen, (unsigned long long)g_tex_diag_causes[TEX_DIAG_REGISTER],
            (unsigned long long)g_tex_diag_causes[TEX_DIAG_FIRST_USE], (unsigned long long)g_tex_diag_causes[TEX_DIAG_VIEW_DROP], (unsigned long long)g_tex_diag_causes[TEX_DIAG_ORPHAN]);
    }
    if (request && !g_tex_diag_active) {
        setenv("DM_VK_TEX_DIAG", "0", 1); memset(g_tex_diag_causes, 0, sizeof g_tex_diag_causes);
        g_tex_diag_start = g_tex_gen; g_tex_diag_frames = 64; g_tex_diag_left = request > 64 ? 64 : (int)request;
        __atomic_store_n(&g_tex_diag_active, 1, __ATOMIC_RELEASE);
        VKLOG("texture generation diagnostics: next 64 presents, at most %d changed events", g_tex_diag_left);
    }
    pthread_mutex_unlock(&g_tex_diag_mu);
}
