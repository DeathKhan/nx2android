// Memory pools backed by a file (DM_VK_POOL_FILEBACK=texture; off by default). The guest's memory is anonymous: what the game keeps and does not touch goes to compressed swap,
// where it still takes RAM. Ocarina of Time keeps its packaged textures in a few large pools (2.4 GB, 232, 202 and 88 MB), written at an area's load and afterwards read only when
// the replay uploads a texture into an image of its own, so the guest's copy is cold: in Kokiri Forest on the Odin 3, PSS 7.8 GB with 1.1 GB in swap, 3.9 GB of it the guest's
// anonymous memory. Backed by a file, a written page goes to flash once and is then a clean page the kernel drops first and reads back when wanted, outside swap (upstream
// droidplay-core a419c5914, after BananaNative's RAM saver; written for this backend's lazy zero-copy chunks).
// - Which pools (vk_fb_refuse), at their Initialize (gfx_vk_core.c vk_pool_fileback): at least DM_VK_POOL_FILEBACK_MIN_MB (default 64), and not shader code, compressible
//   (render targets), physical, virtual, CPU no-access or GPU no-access. Only chunks not backed yet: a backed chunk's pages are a hardware buffer's.
// - The swap (vk_fb_take): the range read-only while its writers wait (the caller's nvn_dirty_freeze), the pages present or in swap and not all zero (/proc/self/pagemap)
//   copied into the file through a second mapping, then the file mapped over the range. One unlinked file in DM_TEMP (the app's own storage); its blocks reserved first
//   (fallocate), so a full disk refuses the swap instead of a game write ending in SIGBUS later. DM_VK_POOL_FILEBACK_CAP_MB (default 4096) caps what the file holds.
// - A chunk backed later copies every page out of the file (vk_fb_has): copy_resident skips what pagemap says is neither present nor swapped, and a file page the kernel
//   dropped is exactly that, so it would have been backed as zeros. Its blocks are then given back (vk_fb_moved). A pool's Finalize gives its pieces back to anonymous
//   memory with their bytes (vk_fb_restore): the game reuses the memory, and the file would keep the blocks.
#include "gfx_vk_fileback.h"
#include <errno.h>
#include <fcntl.h>
#include <linux/falloc.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>
#ifdef FB_TEST   // dm/android/tests/t_pool_fileback.c, without the host
#include <stdio.h>
#define LOG(...) do { fprintf(stderr, __VA_ARGS__); fputc('\n', stderr); } while (0)
#else
#include "dm.h"   // LOG
#endif
#define FBLOG(...) LOG("[vk] pool file: " __VA_ARGS__)

typedef struct { uintptr_t lo, hi, owner; uint64_t off; } fb_piece;   // [lo, hi) of the pool at owner, at off in the file
static pthread_mutex_t g_fb_mu = PTHREAD_MUTEX_INITIALIZER;
static fb_piece *g_fb; static _Atomic uint32_t g_fb_n; static uint32_t g_fb_cap;   // the file's pieces (g_fb_n read without the lock only to see that there are none)
static int g_fb_fd = -1, g_fb_pm = -2; static uint64_t g_fb_next;   // the file, its next offset (never reused: given-back blocks are holes); /proc/self/pagemap (-1: not trusted)
static uint64_t g_fb_in, g_fb_pools, g_fb_copied, g_fb_moved, g_fb_back, g_fb_refused, g_fb_ns;   // bytes in the file, pools taken, copied in, moved to buffers, given back at Finalize; refusals; swap time
static struct { int init, on; uint64_t min, cap; char dir[512]; } C;
static void fb_config(void) {
    if (C.init) return;
    C.init = 1;
    const char *e = getenv("DM_VK_POOL_FILEBACK"); C.on = e && !strcmp(e, "texture");
    if (e && *e && *e != '0' && !C.on) LOG("[RED] [vk] DM_VK_POOL_FILEBACK=%s: unknown value (texture: texture pools backed by a file); off", e);
    const char *m = getenv("DM_VK_POOL_FILEBACK_MIN_MB"), *c = getenv("DM_VK_POOL_FILEBACK_CAP_MB");
    C.min = (uint64_t)(m && atol(m) > 0 ? atol(m) : 64) << 20; C.cap = (uint64_t)(c && atol(c) > 0 ? atol(c) : 4096) << 20;
    const char *d = getenv("DM_TEMP"); if (!d || !*d) d = getenv("TMPDIR"); snprintf(C.dir, sizeof C.dir, "%s", d ? d : "");
    if (C.on) FBLOG("DM_VK_POOL_FILEBACK=texture: texture pools of %llu MB or more backed by one unlinked file in %s, at most %llu MB (DM_VK_POOL_FILEBACK_MIN_MB, _CAP_MB)",
                    (unsigned long long)(C.min >> 20), *C.dir ? C.dir : "(no DM_TEMP)", (unsigned long long)(C.cap >> 20));
}
int vk_fb_on(void) { fb_config(); return C.on; }
#ifdef FB_TEST
void vk_fb_test_config(const char *dir, uint64_t min_bytes, uint64_t cap_bytes) { C.init = 1; C.on = 1; C.min = min_bytes; C.cap = cap_bytes; snprintf(C.dir, sizeof C.dir, "%s", dir); }
uint64_t vk_fb_test_in(void) { pthread_mutex_lock(&g_fb_mu); uint64_t v = g_fb_in; pthread_mutex_unlock(&g_fb_mu); return v; }
#endif
static uint64_t fb_ns(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return (uint64_t)t.tv_sec * 1000000000ull + (uint64_t)t.tv_nsec; }
// NVN's pool flags (nvn.h NVN_MEMORY_POOL_FLAGS_*): Ocarina of Time's texture pools are CPU_CACHED | GPU_CACHED (0x24), its render targets 0xa1, its shader code 0x64
const char *vk_fb_refuse(int flags, uint64_t size) {
    fb_config();
    const char *why = !C.on ? "off" : size < C.min ? "small" : flags & 0x40 ? "shader code" : flags & 0x80 ? "compressible" : flags & 0x300 ? "physical or virtual" :
                      flags & 0x1 ? "CPU no access" : flags & 0x8 ? "GPU no access" : NULL;
    static _Atomic int nl; if (C.on && size >= C.min && atomic_fetch_add(&nl, 1) < 32) FBLOG("pool of %.0f MB, flags %#x: %s", size / 1048576.0, flags, why ? why : "a texture pool, to the file");
    return why;
}
static int fb_open(void) {   // under g_fb_mu
    if (g_fb_fd >= 0) return 1;
    if (!*C.dir) { FBLOG("[RED] no DM_TEMP: pools stay anonymous"); return 0; }
    int fd = open(C.dir, O_TMPFILE | O_RDWR | O_CLOEXEC, 0600);
    if (fd < 0) { char p[600]; snprintf(p, sizeof p, "%s/poolfile_XXXXXX", C.dir); fd = mkostemp(p, O_CLOEXEC); if (fd >= 0) unlink(p); }
    if (fd < 0) { FBLOG("[RED] no file in %s (errno %d): pools stay anonymous", C.dir, errno); return 0; }
    g_fb_fd = fd;
    // pagemap's present / swapped bits tell the pages never touched (left as holes): trusted only if a page just written reads as present
    g_fb_pm = open("/proc/self/pagemap", O_RDONLY | O_CLOEXEC);
    if (g_fb_pm >= 0) {
        long pg = getpagesize(); uint8_t *t = mmap(NULL, (size_t)pg, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0); uint64_t ent = 0;
        if (t != MAP_FAILED) { *(volatile uint8_t *)t = 1; if (pread(g_fb_pm, &ent, 8, (off_t)((uintptr_t)t / (uintptr_t)pg * 8)) != 8) ent = 0; munmap(t, (size_t)pg); }
        if (!(ent >> 63)) { close(g_fb_pm); g_fb_pm = -1; }
    }
    FBLOG("one unlinked file in %s; %s", C.dir, g_fb_pm >= 0 ? "pages never touched are left as holes (/proc/self/pagemap)" : "/proc/self/pagemap unreadable or masked: every page not all zero is copied");
    return 1;
}
static int fb_zero(const uint8_t *p, size_t n) { const uint64_t *q = (const uint64_t *)p; for (size_t i = 0; i < n / 8; i++) if (q[i]) return 0; return 1; }
static void fb_punch(uint64_t off, uint64_t len) { if (len) fallocate(g_fb_fd, FALLOC_FL_PUNCH_HOLE | FALLOC_FL_KEEP_SIZE, (off_t)off, (off_t)len); }
static int fb_overlap(uintptr_t lo, uintptr_t hi) { for (uint32_t i = 0; i < g_fb_n; i++) if (g_fb[i].lo < hi && g_fb[i].hi > lo) return 1; return 0; }   // under g_fb_mu
static void fb_line(const char *what) {   // under g_fb_mu
    FBLOG("%s: %llu pools, %.0f MB in the file (at most %llu MB), %u pieces; %.0f MB copied in (%.0f ms in all), %.0f MB moved to pool buffers, %.0f MB given back at Finalize; %llu refused",
          what, (unsigned long long)g_fb_pools, g_fb_in / 1048576.0, (unsigned long long)(C.cap >> 20), (unsigned)g_fb_n, g_fb_copied / 1048576.0, g_fb_ns / 1e6,
          g_fb_moved / 1048576.0, g_fb_back / 1048576.0, (unsigned long long)g_fb_refused);
}
int vk_fb_take(uintptr_t lo, uintptr_t hi, uintptr_t owner, int flags) {
    size_t pg = (size_t)getpagesize(), size = hi - lo;
    if (!vk_fb_on() || hi <= lo) return -1;
    pthread_mutex_lock(&g_fb_mu);
    const char *why = NULL; uint64_t t0 = fb_ns(), copied = 0, off = g_fb_next; void *f = MAP_FAILED; int ro = 0, err = 0;
    if ((lo | size) & (pg - 1)) why = "not page-aligned";
    else if (fb_overlap(lo, hi)) why = "already in the file";
    else if (g_fb_in + size > C.cap) why = "over DM_VK_POOL_FILEBACK_CAP_MB";
    else if (g_fb_n == g_fb_cap) { fb_piece *p = realloc(g_fb, (g_fb_cap + 64) * sizeof *g_fb); if (p) { g_fb = p; g_fb_cap += 64; } else why = "no memory for the piece list"; }
    if (!why && !fb_open()) why = "no file";
    if (!why && fallocate(g_fb_fd, 0, (off_t)off, (off_t)size)) { why = "no room for it on the disk"; err = errno; }
    if (!why && (f = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, g_fb_fd, (off_t)off)) == MAP_FAILED) { why = "no mapping of the file"; err = errno; }
    if (!why && mprotect((void *)lo, size, PROT_READ)) { why = "the range could not be made read-only"; err = errno; } else if (!why) ro = 1;
    if (!why) {   // the game's pages into the file: present or in swap, and not all zero (a hole reads 0)
        enum { N = 512 }; uint64_t ent[N]; size_t pages = size / pg;
        for (size_t i = 0; i < pages; i += N) {
            size_t k = pages - i < N ? pages - i : N;
            int have = g_fb_pm >= 0 && pread(g_fb_pm, ent, k * 8, (off_t)((lo / pg + i) * 8)) == (ssize_t)(k * 8);
            for (size_t j = 0; j < k; j++) {
                const uint8_t *src = (const uint8_t *)(lo + (i + j) * pg);
                if ((have && !(ent[j] & (3ull << 62))) || fb_zero(src, pg)) continue;
                memcpy((uint8_t *)f + (i + j) * pg, src, pg); copied += pg;
            }
        }
        if (mmap((void *)lo, size, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_FIXED, g_fb_fd, (off_t)off) != (void *)lo) { why = "the file could not be mapped over the range"; err = errno; }
        else ro = 0;   // (the file's mapping is writable)
    }
    if (ro) mprotect((void *)lo, size, PROT_READ | PROT_WRITE);
    if (f != MAP_FAILED) munmap(f, size);
    if (why) {
        if (g_fb_fd >= 0 && off == g_fb_next) fb_punch(off, size);
        g_fb_refused++; static int nr; if (nr++ < 8) FBLOG("[RED] pool %p, %#llx +%#llx (flags %#x) stays anonymous: %s (errno %d)", (void *)owner, (unsigned long long)lo, (unsigned long long)size, flags, why, err);
        pthread_mutex_unlock(&g_fb_mu); return -1;
    }
    sync_file_range(g_fb_fd, (off_t)off, (off_t)size, SYNC_FILE_RANGE_WRITE);   // to flash now, so the pages are clean before memory runs low, not written out then
    g_fb[g_fb_n] = (fb_piece){lo, hi, owner, off}; atomic_store(&g_fb_n, g_fb_n + 1);
    g_fb_next += size; g_fb_in += size; g_fb_pools++; g_fb_copied += copied; uint64_t ns = fb_ns() - t0; g_fb_ns += ns;
    static int nl; if (nl++ < 16) FBLOG("pool %p, %#llx +%#llx (flags %#x): %.1f MB copied to the file, writers held %.1f ms", (void *)owner, (unsigned long long)lo, (unsigned long long)size, flags, copied / 1048576.0, ns / 1e6);
    fb_line("taken");
    pthread_mutex_unlock(&g_fb_mu);
    return 0;
}
int vk_fb_has(uintptr_t lo, uintptr_t hi) {
    if (!atomic_load(&g_fb_n)) return 0;
    pthread_mutex_lock(&g_fb_mu); int r = fb_overlap(lo, hi); pthread_mutex_unlock(&g_fb_mu);
    return r;
}
// [lo, hi) out of the pieces (of the pool at owner; 0: of any), its blocks punched: a piece cut in its middle becomes two. Under g_fb_mu; the bytes cut out
static uint64_t fb_cut(uintptr_t lo, uintptr_t hi, uintptr_t owner) {
    uint64_t cut = 0;
    for (uint32_t i = g_fb_n; i-- > 0;) {
        fb_piece *p = &g_fb[i]; uintptr_t a = lo > p->lo ? lo : p->lo, b = hi < p->hi ? hi : p->hi;
        if (a >= b || (owner && p->owner != owner)) continue;
        if (a > p->lo && b < p->hi) {   // its middle: the end becomes a piece of its own (or, with no room for it, stays in the file and goes with the piece's pool)
            if (g_fb_n == g_fb_cap) { fb_piece *q = realloc(g_fb, (g_fb_cap + 64) * sizeof *g_fb); if (!q) continue; g_fb = q; g_fb_cap += 64; p = &g_fb[i]; }
            g_fb[g_fb_n] = (fb_piece){b, p->hi, p->owner, p->off + (b - p->lo)}; atomic_store(&g_fb_n, g_fb_n + 1);
            fb_punch(p->off + (a - p->lo), b - a); p->hi = a;
        } else {
            fb_punch(p->off + (a - p->lo), b - a);
            if (a == p->lo && b == p->hi) { g_fb[i] = g_fb[g_fb_n - 1]; atomic_store(&g_fb_n, g_fb_n - 1); }
            else if (a == p->lo) { p->off += b - p->lo; p->lo = b; }
            else p->hi = a;
        }
        cut += b - a;
    }
    g_fb_in -= cut;
    return cut;
}
void vk_fb_moved(uintptr_t lo, uintptr_t hi) {
    if (!atomic_load(&g_fb_n)) return;
    pthread_mutex_lock(&g_fb_mu);
    uint64_t n = fb_cut(lo, hi, 0); g_fb_moved += n;
    static int nl; if (n && nl++ < 8) { FBLOG("%#llx +%#llx backed by a pool buffer: %.1f MB of the file given back", (unsigned long long)lo, (unsigned long long)(hi - lo), n / 1048576.0); fb_line("moved"); }
    pthread_mutex_unlock(&g_fb_mu);
}
uint64_t vk_fb_restore(uintptr_t lo, uintptr_t hi, uintptr_t owner) {
    if (!atomic_load(&g_fb_n)) return 0;
    size_t pg = (size_t)getpagesize(); uint64_t done = 0;
    pthread_mutex_lock(&g_fb_mu);
    for (int again = 1; again;) {   // a piece at a time: fb_cut reorders the list
        again = 0;
        for (uint32_t i = 0; i < g_fb_n; i++) {
            fb_piece p = g_fb[i]; uintptr_t a = lo > p.lo ? lo : p.lo, b = hi < p.hi ? hi : p.hi;
            if (a >= b || p.owner != owner) continue;
            size_t n = b - a; uint8_t *t = mmap(NULL, n, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
            if (t == MAP_FAILED) { static int nr; if (nr++ < 4) FBLOG("[RED] %#llx +%#llx: no anonymous memory to give it back to (errno %d): stays in the file", (unsigned long long)a, (unsigned long long)n, errno); continue; }
            int ro = !mprotect((void *)a, n, PROT_READ);   // (writers wait on the caller's freeze)
            // every page: one the kernel dropped reads back from the file. All-zero pages stay untouched in the new memory (a hole read costs a page of cache, not of the game's)
            for (size_t o = 0; o < n; o += pg) if (!fb_zero((const uint8_t *)a + o, pg)) memcpy(t + o, (const uint8_t *)a + o, pg);
            if ((void *)syscall(__NR_mremap, t, n, n, MREMAP_MAYMOVE | MREMAP_FIXED, (void *)a) != (void *)a) {
                static int nr; if (nr++ < 4) FBLOG("[RED] %#llx +%#llx could not be given back to anonymous memory (errno %d): stays in the file", (unsigned long long)a, (unsigned long long)n, errno);
                munmap(t, n); if (ro) mprotect((void *)a, n, PROT_READ | PROT_WRITE); continue;
            }
            uint64_t c = fb_cut(a, b, owner); g_fb_back += c; done += c; again = 1; break;
        }
    }
    static int nl; if (done && nl++ < 8) { FBLOG("pool %p gone: %.1f MB back to anonymous memory", (void *)owner, done / 1048576.0); fb_line("given back"); }
    pthread_mutex_unlock(&g_fb_mu);
    return done;
}
