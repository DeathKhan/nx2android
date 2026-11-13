// The translator's raw global memory helpers keep the range and delta of the entry they chose last beside its index, so a hit reads no table word. A helper
// (GlobalToStorage.TryGenerateRawStorageOp; docs/reference/shaders.md "Raw global memory lookups that start from their last entry") keeps the index of the pool table
// entry it chose last in a private word of the invocation, and on every call reads that entry's start and end (4 words of the table, a storage buffer) to see whether the
// address is still in it, then the entry's delta (2 more) to make the pointer: 6 dependent storage loads before the access itself, on every access, hit or miss.
// Ocarina of Time's culling and scene compute shaders make 700-odd such accesses a dispatch in 1-12 workgroups, where each dependent load is latency the GPU can't hide.
// spv_gmcache gives each helper six more private words, the start, end and delta of its last entry (zero at the start, as the index is: entry 0 is the empty one), reads
// those where the helper read the table, and fills them where the helper records a new entry (the block that stores the index after the hint or the search, which
// re-reads the entry's end anyway). The words always mirror the entry the index names, so the results are the same; a hit costs 0 table loads instead of 6. The host's
// table is the same, and nothing is re-translated: the rewrite runs when a module is made (gfx_vk_draw.inc lm_code). A module whose helpers are not in that form (another
// count of table reads in the entry block, no store of the index) is left as it is. Self-contained (no libc) for dm/tools/tests/test_spv_gmcache.py.
#pragma once
#include <stddef.h>
#include <stdint.h>

enum { SPV_GMCACHE_WORDS = 6 };   // private words a helper gets: start lo, start hi, end lo, end hi, delta lo, delta hi (the table entry's first six)
enum { SPV_GMCACHE_FORM = 1, SPV_GMCACHE_ROOM = 2, SPV_GMCACHE_MANY = 3 };   // why a module was left: not the translator's form; out too small; more helpers than fit
#define SPV_GMCACHE_MAX 16   // helpers rewritten at most in one module (Ocarina of Time's compute shaders have up to 5)
#define SPV_GMCACHE_EXTRA(nf) (16 + 6 * 4 + (size_t)(nf) * (6 * 5 + 6 * 15 + 6))   // words a rewrite adds at most for nf helpers: a pointer type, a null, six constants, the variables, the fills, the entry points' interfaces

// Whether the module goes through physical pointers (OpCapability PhysicalStorageBufferAddresses): the only modules with such helpers
static int spv_gmcache_wanted(const uint32_t *w, size_t n) {
    if (n < 5 || w[0] != 0x07230203) return 0;
    for (size_t i = 5; i < n; ) { uint32_t len = w[i] >> 16, op = w[i] & 0xffff; if (!len || i + len > n) return 0;
        if (op == 17 && len == 2 && w[i + 1] == 5347) return 1;
        if (op != 17 && op != 10 && op != 11) return 0;   // past the capabilities, extensions and imports
        i += len; }
    return 0;
}

typedef struct { size_t fn, end, ent[4], entac[4], del[2], delac[2], mul; uint32_t tab, tabc0, acty, wty, ity, found, pv[6]; int ne, nd; } spv_gmcache_fn;   // one helper's places

// An access chain's last index resolved as (base + c), c a small constant, through loads of locals stored once in the function, bitcasts and copies. def: each id's
// defining instruction (word index); stv: each local's one store's value (0 none, ~0 stored more than once); fl bit 1 with cv: a constant's value. → c, or -1
static int spv_gmcache_c(const uint32_t *w, const uint32_t *def, const uint32_t *stv, const uint32_t *cv, const uint8_t *fl, uint32_t bound, uint32_t v) {
    for (int k = 0; k < 10 && v < bound && def[v]; k++) { const uint32_t *x = w + def[v]; uint32_t op = x[0] & 0xffff, len = x[0] >> 16;
        if (op == 61 && len >= 4) { uint32_t var = x[3]; if (var >= bound || !stv[var] || stv[var] == ~0u) return -1; v = stv[var]; continue; }
        if ((op == 124 || op == 83) && len >= 4) { v = x[3]; continue; }
        if (op == 128 && len == 5) {
            if (x[4] < bound && (fl[x[4]] & 1) && cv[x[4]] < 8) return (int)cv[x[4]];
            if (x[3] < bound && (fl[x[3]] & 1) && cv[x[3]] < 8) return (int)cv[x[3]];
            return -1; }
        return -1; }
    return -1;
}

// The module rewritten into o (cap words) → its length; 0 when nothing is done (*why says why; 0: no helper). sc: 4 * bound words of scratch
static size_t spv_gmcache(const uint32_t *w, size_t n, uint32_t *sc, uint32_t *o, size_t cap, int *why) {
    *why = 0;
    if (n < 5 || w[0] != 0x07230203) { *why = SPV_GMCACHE_FORM; return 0; }
    uint32_t bound = w[3], ver = w[1];
    uint32_t *def = sc, *stv = sc + bound, *cv = sc + 2 * (size_t)bound; uint8_t *fl = (uint8_t *)(sc + 3 * (size_t)bound);   // fl: 1 a constant, 2 a Private variable with an initializer
    for (size_t i = 0; i < 3 * (size_t)bound; i++) sc[i] = 0;
    for (size_t i = 0; i < bound; i++) fl[i] = 0;
    size_t i_fn = 0;
    for (size_t i = 5; i < n; ) { uint32_t len = w[i] >> 16, op = w[i] & 0xffff; const uint32_t *x = w + i; if (!len || i + len > n) { *why = SPV_GMCACHE_FORM; return 0; }
        if (op == 54 && !i_fn) i_fn = i;
        if (op >= 19 && op <= 39 && len >= 2 && x[1] < bound) def[x[1]] = (uint32_t)i;   // types
        else if ((op == 43 || op == 46 || op == 59 || op == 61 || op == 65 || op == 83 || op == 124 || op == 128 || op == 132) && len >= 3 && x[2] < bound) def[x[2]] = (uint32_t)i;
        if (op == 43 && len == 4 && x[2] < bound) { fl[x[2]] |= 1; cv[x[2]] = x[3]; }
        if (op == 59 && len == 5 && x[3] == 6 && x[2] < bound) fl[x[2]] |= 2;
        i += len; }
    if (!i_fn) return 0;
    spv_gmcache_fn F[SPV_GMCACHE_MAX]; int nf = 0;
    for (size_t i = i_fn; i < n; ) {   // each function with an OpConvertUToPtr
        if ((w[i] & 0xffff) != 54) { *why = SPV_GMCACHE_FORM; return 0; }
        size_t s = i, e = i; int conv = 0;
        for (; e < n; e += w[e] >> 16) { uint32_t op = w[e] & 0xffff; if (op == 120) conv = 1; if (op == 56) { e += w[e] >> 16; break; } }
        i = e;
        if (!conv) continue;
        if (nf == SPV_GMCACHE_MAX) { *why = SPV_GMCACHE_MANY; return 0; }
        spv_gmcache_fn *f = &F[nf]; for (size_t q = 0; q < sizeof *f; q++) ((uint8_t *)f)[q] = 0; f->fn = s; f->end = e;
        for (size_t k = 0; k < bound; k++) stv[k] = 0;
        for (size_t j = s; j < e; j += w[j] >> 16) if ((w[j] & 0xffff) == 62 && (w[j] >> 16) >= 3 && w[j + 1] < bound) stv[w[j + 1]] = stv[w[j + 1]] ? ~0u : w[j + 2];
        uint32_t best = 0, bestn = 0;   // the table: the base of most of the function's access chains (the search alone has 26 on it)
        for (size_t j = s; j < e; j += w[j] >> 16) if ((w[j] & 0xffff) == 65 && (w[j] >> 16) == 6) { uint32_t b = w[j + 3], c = 0;
            for (size_t q = s; q < e; q += w[q] >> 16) if ((w[q] & 0xffff) == 65 && (w[q] >> 16) == 6 && w[q + 3] == b) c++;
            if (c > bestn) { bestn = c; best = b; } }
        if (bestn < 7) { *why = SPV_GMCACHE_FORM; return 0; }
        f->tab = best;
        size_t blk = 0, fill = 0; int nb = 0;
        for (size_t j = s; j < e; j += w[j] >> 16) { uint32_t op = w[j] & 0xffff, len = w[j] >> 16; const uint32_t *x = w + j;
            if (op == 248) { blk = j; nb++; continue; }
            if (op == 62 && len == 3 && x[1] < bound && (fl[x[1]] & 2)) { if (fill) { *why = SPV_GMCACHE_FORM; return 0; } fill = j; continue; }   // the index recorded
            if (op == 132 && len == 5 && fill && !f->mul && j > fill) {   // the first found = index * 8 after it
                if ((x[4] < bound && (fl[x[4]] & 1) && cv[x[4]] == 8) || (x[3] < bound && (fl[x[3]] & 1) && cv[x[3]] == 8)) { f->mul = j; f->found = x[2]; f->ity = x[1]; }
                continue; }
            if (op != 61 || len < 4 || x[3] >= bound || !def[x[3]]) continue;
            const uint32_t *ac = w + def[x[3]];   // a load through a chain of one index on the table, made in this block
            if ((ac[0] & 0xffff) != 65 || (ac[0] >> 16) != 6 || ac[3] != best || def[x[3]] < blk) continue;
            int c = spv_gmcache_c(w, def, stv, cv, fl, bound, ac[5]);
            if (c < 0) continue;
            if (!f->wty) { f->wty = x[1]; f->acty = ac[1]; f->tabc0 = ac[4]; }
            if (x[1] != f->wty || ac[1] != f->acty || ac[4] != f->tabc0) { *why = SPV_GMCACHE_FORM; return 0; }
            if (nb == 1 && c <= 3) { if (f->ent[c]) { *why = SPV_GMCACHE_FORM; return 0; } f->ent[c] = j; f->entac[c] = def[x[3]]; f->ne++; }
            else if (nb > 1 && (c == 4 || c == 5)) { if (f->del[c - 4]) { *why = SPV_GMCACHE_FORM; return 0; } f->del[c - 4] = j; f->delac[c - 4] = def[x[3]]; f->nd++; } }
        if (f->ne != 4 || f->nd != 2 || !fill || !f->mul) { *why = SPV_GMCACHE_FORM; return 0; }
        nf++;
    }
    if (!nf) return 0;
    uint32_t wty = F[0].wty, ity = F[0].ity;
    for (int k = 1; k < nf; k++) if (F[k].wty != wty || F[k].ity != ity) { *why = SPV_GMCACHE_FORM; return 0; }
    uint32_t pty = 0, nul = 0, cs[6] = {0};   // a Private pointer to the word type, a null of it, the constants 0..5 of the index type: the module's, else new
    for (size_t i = 5; i < i_fn; i += w[i] >> 16) { uint32_t op = w[i] & 0xffff, len = w[i] >> 16; const uint32_t *x = w + i;
        if (op == 32 && len == 4 && x[2] == 6 && x[3] == wty && !pty) pty = x[1];
        if (op == 46 && len == 3 && x[1] == wty && !nul) nul = x[2];
        if (op == 43 && len == 4 && x[1] == ity && x[3] < 6 && !cs[x[3]]) cs[x[3]] = x[2]; }
    uint32_t nb_ = bound;
    if (!pty) pty = nb_++;
    if (!nul) nul = nb_++;
    for (int c = 0; c < 6; c++) if (!cs[c]) cs[c] = nb_++;
    for (int k = 0; k < nf; k++) for (int c = 0; c < 6; c++) F[k].pv[c] = nb_++;
    if (cap < n + SPV_GMCACHE_EXTRA(nf)) { *why = SPV_GMCACHE_ROOM; return 0; }
    size_t k = 0;
    for (int q = 0; q < 5; q++) o[k++] = w[q];
    o[3] = nb_;
    for (size_t i = 5; i < n; ) { uint32_t len = w[i] >> 16, op = w[i] & 0xffff; const uint32_t *x = w + i;
        if (i == i_fn) {   // the globals, before the first function: after every type and constant they use
            if (pty >= bound) { o[k++] = 4u << 16 | 32; o[k++] = pty; o[k++] = 6; o[k++] = wty; }
            if (nul >= bound) { o[k++] = 3u << 16 | 46; o[k++] = wty; o[k++] = nul; }
            for (int c = 0; c < 6; c++) if (cs[c] >= bound) { o[k++] = 4u << 16 | 43; o[k++] = ity; o[k++] = cs[c]; o[k++] = (uint32_t)c; }
            for (int f = 0; f < nf; f++) for (int c = 0; c < 6; c++) { o[k++] = 5u << 16 | 59; o[k++] = pty; o[k++] = F[f].pv[c]; o[k++] = 6; o[k++] = nul; } }
        if (op == 15 && ver >= 0x10400) {   // SPIR-V 1.4: every global variable a function may use is in the interface
            o[k] = (len + 6u * (uint32_t)nf) << 16 | 15; for (uint32_t q = 1; q < len; q++) o[k + q] = x[q]; k += len;
            for (int f = 0; f < nf; f++) for (int c = 0; c < 6; c++) o[k++] = F[f].pv[c];
            i += len; continue; }
        int drop = 0, swap = -1, fill = 0; const spv_gmcache_fn *f = 0;
        for (int q = 0; q < nf && !f; q++) if (i >= F[q].fn && i < F[q].end) f = &F[q];
        if (f) { for (int c = 0; c < 4; c++) { if (i == f->entac[c]) drop = 1; if (i == f->ent[c]) swap = c; }
                 for (int c = 0; c < 2; c++) { if (i == f->delac[c]) drop = 1; if (i == f->del[c]) swap = 4 + c; }
                 if (i == f->mul) fill = 1; }
        if (drop) { i += len; continue; }
        if (swap >= 0) { o[k++] = 4u << 16 | 61; o[k++] = x[1]; o[k++] = x[2]; o[k++] = f->pv[swap]; i += len; continue; }
        for (uint32_t q = 0; q < len; q++) o[k + q] = x[q]; k += len; i += len;
        if (fill) for (int c = 0; c < 6; c++) {   // the found entry's six words into the private ones
            uint32_t a = nb_++, p = nb_++, v = nb_++;
            o[k++] = 5u << 16 | 128; o[k++] = f->ity; o[k++] = a; o[k++] = f->found; o[k++] = cs[c];
            o[k++] = 6u << 16 | 65; o[k++] = f->acty; o[k++] = p; o[k++] = f->tab; o[k++] = f->tabc0; o[k++] = a;
            o[k++] = 4u << 16 | 61; o[k++] = f->wty; o[k++] = v; o[k++] = p;
            o[k++] = 3u << 16 | 62; o[k++] = f->pv[c]; o[k++] = v; }
    }
    o[3] = nb_;
    return k;
}
