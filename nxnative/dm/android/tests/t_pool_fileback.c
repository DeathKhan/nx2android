// dm/android/gfx_vk_fileback.c on the host (Linux): a pool's pages moved into the file and back, the whole-chunk rule for a later backing, the cuts, the refusals and which pools
#include "../gfx_vk_fileback.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#ifdef __linux__
#include <fcntl.h>
#include <stdint.h>
#include <sys/mman.h>
#include <sys/stat.h>
static unsigned checks, failed;
#define CHECK(label, c) do { checks++; if (!(c)) { failed++; fprintf(stderr, "FAIL: %s\n", label); } } while (0)
enum { NPG = 64 };
static size_t PG; static uint8_t *R;
static uint8_t pat(size_t pg, size_t i) { return (uint8_t)(pg * 31 + i * 7 + 1) | 1; }   // never 0
static void fill(size_t pg) { for (size_t i = 0; i < PG; i++) R[pg * PG + i] = pat(pg, i); }
static int same(const uint8_t *b, size_t pg) { for (size_t i = 0; i < PG; i++) if (b[i] != pat(pg, i)) return 0; return 1; }
static int zero(const uint8_t *b) { for (size_t i = 0; i < PG; i++) if (b[i]) return 0; return 1; }
// gfx_vk_core.c copy_resident: only the pages pagemap says are present or swapped
static void copy_resident(uint8_t *dst, const uint8_t *src, size_t n) {
    int fd = open("/proc/self/pagemap", O_RDONLY); uint64_t e;
    for (size_t o = 0; o < n; o += PG) if (fd < 0 || pread(fd, &e, 8, (off_t)(((uintptr_t)src + o) / PG * 8)) != 8 || (e & (3ull << 62))) memcpy(dst + o, src + o, PG);
    if (fd >= 0) close(fd);
}
static int anonymous(uintptr_t a) {   // /proc/self/maps: the mapping holding a has no file
    FILE *f = fopen("/proc/self/maps", "r"); char l[512]; int r = -1;
    while (f && fgets(l, sizeof l, f)) { unsigned long lo, hi, ino; char perm[8]; unsigned long long off; char dev[16];
        if (sscanf(l, "%lx-%lx %7s %llx %15s %lu", &lo, &hi, perm, &off, dev, &ino) == 6 && a >= lo && a < hi) { r = ino == 0; break; } }
    if (f) fclose(f);
    return r;
}
int main(void) {
    PG = (size_t)getpagesize();
    mkdir("build", 0700);
    vk_fb_test_config("build", 0, (uint64_t)NPG * PG);
    R = mmap(NULL, NPG * PG, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    uintptr_t lo = (uintptr_t)R, hi = lo + NPG * PG;
    for (size_t p = 0; p < 10; p++) fill(p);           // 0-9 written, 10-19 never touched
    memset(R + 20 * PG, 0, 10 * PG);                   // 20-29 touched, all zero
    for (size_t p = 30; p < NPG; p++) fill(p);         // 30-63 written

    CHECK("texture pool flags (CPU_CACHED | GPU_CACHED) are taken", !vk_fb_refuse(0x24, 1 << 20));
    CHECK("shader code is not", vk_fb_refuse(0x64, 1 << 20) != NULL);
    CHECK("render targets (compressible) are not", vk_fb_refuse(0xa1, 1 << 20) != NULL);
    CHECK("virtual pools are not", vk_fb_refuse(0x224, 1 << 20) != NULL);

    CHECK("the swap succeeds", vk_fb_take(lo, hi, lo, 0x24) == 0);
    CHECK("the range is the file's now", anonymous(lo) == 0);
    int ok = 1; for (size_t p = 0; p < NPG; p++) ok &= (p < 10 || p >= 30) ? same(R + p * PG, p) : zero(R + p * PG);
    CHECK("every byte is as it was after the swap", ok);
    CHECK("the range is in the file", vk_fb_has(lo, hi) && vk_fb_test_in() == (uint64_t)NPG * PG);
    R[11 * PG] = 5; CHECK("the file's mapping is writable", R[11 * PG] == 5); R[11 * PG] = 0;
    CHECK("a range in the file is refused a second time", vk_fb_take(lo, hi, lo, 0x24) == -1);

    // the kernel drops the file's pages (their bytes stay in the file): pagemap then says neither present nor swapped
    madvise(R, NPG * PG, MADV_DONTNEED);
    uint8_t *dst = calloc(NPG, PG);
    copy_resident(dst, R, NPG * PG);
    int lost = 0; for (size_t p = 0; p < 10; p++) lost += zero(dst + p * PG);
    printf("t_pool_fileback: copy_resident over the dropped file pages backs %d of 10 written pages as zeros (the rule vk_fb_has guards against)\n", lost);
    memset(dst, 0, NPG * PG);
    if (vk_fb_has(lo, hi)) memcpy(dst, R, NPG * PG); else copy_resident(dst, R, NPG * PG);   // pool_zero_copy's rule
    ok = 1; for (size_t p = 0; p < NPG; p++) ok &= (p < 10 || p >= 30) ? same(dst + p * PG, p) : zero(dst + p * PG);
    CHECK("a chunk in the file is copied whole: no page is lost", ok);

    // a zero-copy buffer mapped over pages 40-49 (an inner pool): cut out of the file, the piece split in two
    uint8_t *keep = malloc(10 * PG); memcpy(keep, R + 40 * PG, 10 * PG);
    uint8_t *m = mmap(R + 40 * PG, 10 * PG, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED, -1, 0);
    memcpy(m, keep, 10 * PG); free(keep);
    vk_fb_moved(lo + 40 * PG, lo + 50 * PG);
    CHECK("a range mapped over leaves the file", !vk_fb_has(lo + 40 * PG, lo + 50 * PG) && vk_fb_test_in() == (uint64_t)(NPG - 10) * PG);
    CHECK("the pieces on either side stay", vk_fb_has(lo, lo + 40 * PG) && vk_fb_has(lo + 50 * PG, hi));

    // the pool's Finalize: back to anonymous memory with its bytes
    CHECK("another pool's Finalize gives back nothing", vk_fb_restore(lo, hi, lo + PG) == 0);
    CHECK("the pool's Finalize gives back what the file held", vk_fb_restore(lo, hi, lo) == (uint64_t)(NPG - 10) * PG);
    CHECK("nothing is left in the file", vk_fb_test_in() == 0 && !vk_fb_has(lo, hi));
    CHECK("the range is anonymous memory again", anonymous(lo) == 1 && anonymous(lo + 60 * PG) == 1);
    ok = 1; for (size_t p = 0; p < NPG; p++) ok &= (p < 10 || p >= 30) ? same(R + p * PG, p) : zero(R + p * PG);
    CHECK("every byte is as it was after the give-back", ok);
    R[3 * PG] = 9; CHECK("and writable", R[3 * PG] == 9);

    vk_fb_test_config("build", 0, (uint64_t)4 * PG);
    fill(2); fill(3);
    CHECK("a pool over the cap stays anonymous", vk_fb_take(lo, hi, lo, 0x24) == -1 && anonymous(lo) == 1 && same(R + 2 * PG, 2));
    R[2 * PG] = 1; CHECK("and writable", R[2 * PG] == 1);
    CHECK("a pool under it is taken", vk_fb_take(lo + 8 * PG, lo + 12 * PG, lo + 8 * PG, 0x24) == 0 && vk_fb_restore(lo + 8 * PG, lo + 12 * PG, lo + 8 * PG) == 4 * PG);
    free(dst); munmap(R, NPG * PG);
    printf("t_pool_fileback: %u checks, %u failed\n", checks, failed);
    return failed != 0;
}
#else
int main(void) { printf("t_pool_fileback: Linux only (mremap, fallocate, /proc/self/pagemap): skipped\n"); return 0; }
#endif
