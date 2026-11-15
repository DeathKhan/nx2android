// KGSL command-stream capture (diagnostic; DM_KGSL_CAPTURE=<seconds>:<submissions>, start-up): what the Vulkan driver sends to the GPU, as a Mesa .rd file
// (files/kgsl-capture.rd; decode with Mesa's cffdump / rddecompiler). Made to see how Qualcomm's own driver programs the Adreno for the same frames Turnip
// runs (register state, shader binaries, cache and scratch setup), so Turnip can be brought to it. The driver's libraries' imports of ioctl, mmap and munmap
// are pointed here (their GOT entries) before the device is made: allocations (id → GPU address, size, flags) and CPU mappings are tracked from then on,
// and from <seconds> after start the next <submissions> IOCTL_KGSL_GPU_COMMAND submissions are written with every tracked buffer's address and the
// contents of the read-only ones (command streams, shaders, descriptors; skipped above 16 MB) and of buffers a submitted IB lies in.
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <link.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>
#include <elf.h>
#include <stdarg.h>

void dm_log(const char *fmt, ...) __attribute__((format(printf, 1, 2)));
#define KLOG(...) dm_log("[dm] [kgsl capture] " __VA_ARGS__)

struct kc_gpuobj_alloc { uint64_t size, flags, va_len, mmapsize; uint32_t id, metadata_len; uint64_t metadata; };
struct kc_gpuobj_info { uint64_t gpuaddr, flags, size, va_len, va_addr; uint32_t id; };
struct kc_gpumem_alloc_id { uint32_t id, flags; size_t size, mmapsize; unsigned long gpuaddr; unsigned long pad[2]; };
struct kc_command_object { uint64_t offset, gpuaddr, size; uint32_t flags, id; };
struct kc_gpu_command { uint64_t flags, cmdlist; uint32_t cmdsize, numcmds; uint64_t objlist; uint32_t objsize, numobjs; uint64_t synclist; uint32_t syncsize, numsyncs, context_id, timestamp; };
struct kc_gpuobj_free { uint64_t flags, priv; uint32_t id, type, len; };
struct kc_gpumem_free_id { uint32_t id, pad; };
#define KC_NR(req) (((req) >> 0) & 0xff)
#define KC_TYPE(req) (((req) >> 8) & 0xff)
enum { NR_GPUMEM_ALLOC_ID = 0x34, NR_GPUMEM_FREE_ID = 0x35, NR_GPUOBJ_ALLOC = 0x45, NR_GPUOBJ_FREE = 0x46, NR_GPUOBJ_INFO = 0x47, NR_GPU_COMMAND = 0x4A };
#define KC_GPUREADONLY (1ULL << 24)

typedef struct { uint32_t id; uint64_t gpuaddr, size, flags; void *cpu; int fd; } kc_buf;
static kc_buf *g_b; static int g_nb, g_cb; static pthread_mutex_t g_mu = PTHREAD_MUTEX_INITIALIZER;
static int (*real_ioctl)(int, int, ...); static void *(*real_mmap)(void *, size_t, int, int, int, off_t); static int (*real_munmap)(void *, size_t);
static int g_out = -1, g_left; static double g_t0, g_start_s; static uint64_t g_hist[256], g_subs, g_written;
static char g_path[1100];

static double now_s(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec / 1e9; }
static kc_buf *find_id(int fd, uint32_t id) { for (int i = 0; i < g_nb; i++) if (g_b[i].id == id && g_b[i].fd == fd) return &g_b[i]; return NULL; }
static void add_buf(int fd, uint32_t id, uint64_t gpuaddr, uint64_t size, uint64_t flags) {
    if (g_nb == g_cb) { g_cb = g_cb ? g_cb * 2 : 4096; g_b = realloc(g_b, (size_t)g_cb * sizeof *g_b); }
    g_b[g_nb++] = (kc_buf){id, gpuaddr, size, flags, NULL, fd};
}
static void del_id(int fd, uint32_t id) { for (int i = 0; i < g_nb; i++) if (g_b[i].id == id && g_b[i].fd == fd) { g_b[i] = g_b[--g_nb]; return; } }
static void sect(uint32_t type, const void *p, uint32_t n) { uint32_t h[2] = {type, n}; if (write(g_out, h, 8) != 8 || (n && write(g_out, p, n) != (ssize_t)n)) { close(g_out); g_out = -1; } else g_written += 8 + n; }

static void capture(int fd, const struct kc_gpu_command *c) {
    if (g_out < 0) { if (g_left <= 0 || now_s() - g_t0 < g_start_s) return;
        g_out = open(g_path, O_WRONLY | O_CREAT | O_TRUNC, 0644); if (g_out < 0) { KLOG("cannot write %s (errno %d)", g_path, errno); g_left = 0; return; }
        uint64_t chip = 0x44050001; sect(14 /* RD_CHIP_ID */, &chip, 8); sect(2 /* RD_CMD */, "kgsl-capture", 13);
        KLOG("capturing %d submissions into %s (%d buffers tracked)", g_left, g_path, g_nb); }
    const struct kc_command_object *cl = (const void *)(uintptr_t)c->cmdlist;
    for (int i = 0; i < g_nb; i++) {
        kc_buf *b = &g_b[i]; if (b->fd != fd) continue;
        uint32_t ga[3] = {(uint32_t)b->gpuaddr, (uint32_t)b->size, (uint32_t)(b->gpuaddr >> 32)}; sect(3 /* RD_GPUADDR */, ga, 12);
        int want = (b->flags & KC_GPUREADONLY) && b->size <= (16u << 20);
        for (uint32_t k = 0; !want && k < c->numcmds; k++) { const struct kc_command_object *o = (const void *)((const char *)cl + (size_t)k * c->cmdsize);
            want = o->gpuaddr >= b->gpuaddr && o->gpuaddr < b->gpuaddr + b->size; }
        if (want && b->cpu) sect(12 /* RD_BUFFER_CONTENTS */, b->cpu, (uint32_t)b->size);
    }
    for (uint32_t k = 0; k < c->numcmds; k++) { const struct kc_command_object *o = (const void *)((const char *)cl + (size_t)k * c->cmdsize);
        uint32_t ca[3] = {(uint32_t)o->gpuaddr, (uint32_t)(o->size / 4), (uint32_t)(o->gpuaddr >> 32)}; sect(6 /* RD_CMDSTREAM_ADDR */, ca, 12); }
    if (--g_left <= 0 && g_out >= 0) { close(g_out); g_out = -1; KLOG("capture done: %.1f MB written", g_written / 1048576.0); }
}

static int my_ioctl(int fd, int req, ...) {
    va_list ap; va_start(ap, req); void *arg = va_arg(ap, void *); va_end(ap);
    int r = real_ioctl(fd, req, arg);
    if (KC_TYPE((unsigned)req) != 0x09 || r < 0) return r;
    unsigned nr = KC_NR((unsigned)req);
    pthread_mutex_lock(&g_mu); g_hist[nr]++;
    if (nr == NR_GPUOBJ_ALLOC) { struct kc_gpuobj_alloc *a = arg; struct kc_gpuobj_info inf = {.id = a->id};
        if (real_ioctl(fd, (int)_IOWR(0x09, NR_GPUOBJ_INFO, struct kc_gpuobj_info), &inf) == 0) add_buf(fd, a->id, inf.gpuaddr, inf.size, a->flags); }
    else if (nr == NR_GPUMEM_ALLOC_ID) { struct kc_gpumem_alloc_id *a = arg; add_buf(fd, a->id, a->gpuaddr, a->size, a->flags); }
    else if (nr == NR_GPUOBJ_FREE) { struct kc_gpuobj_free *f = arg; del_id(fd, f->id); }
    else if (nr == NR_GPUMEM_FREE_ID) { struct kc_gpumem_free_id *f = arg; del_id(fd, f->id); }
    else if (nr == NR_GPU_COMMAND) { g_subs++; capture(fd, arg); }
    pthread_mutex_unlock(&g_mu);
    return r;
}
static void *my_mmap(void *addr, size_t len, int prot, int flags, int fd, off_t off) {
    void *p = real_mmap(addr, len, prot, flags, fd, off);
    if (p != MAP_FAILED && fd >= 0 && off) { pthread_mutex_lock(&g_mu); kc_buf *b = find_id(fd, (uint32_t)(off >> 12)); if (b) b->cpu = p; pthread_mutex_unlock(&g_mu); }
    return p;
}
static int my_munmap(void *addr, size_t len) {
    pthread_mutex_lock(&g_mu); for (int i = 0; i < g_nb; i++) if (g_b[i].cpu == addr) g_b[i].cpu = NULL; pthread_mutex_unlock(&g_mu);
    return real_munmap(addr, len);
}

// point a loaded library's imports of ioctl / mmap / munmap here
static int patch_lib(struct dl_phdr_info *info, size_t sz, void *data) {
    (void)sz; (void)data; const char *n = info->dlpi_name; if (!n || !(strstr(n, "gsl") || strstr(n, "adreno") || strstr(n, "vulkan."))) return 0;
    const ElfW(Dyn) *dyn = NULL; for (int i = 0; i < info->dlpi_phnum; i++) if (info->dlpi_phdr[i].p_type == PT_DYNAMIC) dyn = (const void *)(info->dlpi_addr + info->dlpi_phdr[i].p_vaddr);
    if (!dyn) return 0;
    const ElfW(Sym) *symtab = NULL; const char *strtab = NULL; const ElfW(Rela) *rel[2] = {0}; size_t relsz[2] = {0};
    for (const ElfW(Dyn) *d = dyn; d->d_tag != DT_NULL; d++) switch (d->d_tag) {
        case DT_SYMTAB: symtab = (const void *)d->d_un.d_ptr; break; case DT_STRTAB: strtab = (const void *)d->d_un.d_ptr; break;
        case DT_JMPREL: rel[0] = (const void *)d->d_un.d_ptr; break; case DT_PLTRELSZ: relsz[0] = d->d_un.d_val; break;
        case DT_RELA: rel[1] = (const void *)d->d_un.d_ptr; break; case DT_RELASZ: relsz[1] = d->d_un.d_val; break; }
    if (!symtab || !strtab) return 0;
    // bionic leaves the dynamic section's addresses as linked (offsets from the load address); glibc relocates them
#define KC_FIX(p) do { if ((uintptr_t)(p) && (uintptr_t)(p) < info->dlpi_addr) (p) = (void *)((uintptr_t)(p) + info->dlpi_addr); } while (0)
    KC_FIX(symtab); KC_FIX(strtab); KC_FIX(rel[0]); KC_FIX(rel[1]);
    int patched = 0;
    for (int t = 0; t < 2; t++) for (size_t i = 0; rel[t] && i < relsz[t] / sizeof(ElfW(Rela)); i++) {
        const ElfW(Rela) *r = &rel[t][i]; uint32_t type = ELF64_R_TYPE(r->r_info), si = ELF64_R_SYM(r->r_info);
        if (type != R_AARCH64_JUMP_SLOT && type != R_AARCH64_GLOB_DAT) continue;
        const char *s = strtab + symtab[si].st_name; void *to = !strcmp(s, "ioctl") ? (void *)my_ioctl : !strcmp(s, "mmap") || !strcmp(s, "mmap64") ? (void *)my_mmap : !strcmp(s, "munmap") ? (void *)my_munmap : NULL;
        if (!to) continue;
        void **slot = (void **)(info->dlpi_addr + r->r_offset); uintptr_t pg = (uintptr_t)slot & ~(uintptr_t)4095;
        if (mprotect((void *)pg, 4096, PROT_READ | PROT_WRITE)) continue;
        *slot = to; patched++;
    }
    if (patched) KLOG("%s: %d imports pointed at the capture", n, patched);
    return 0;
}
void kgsl_capture_install(const char *root) {   // before vkCreateDevice: the driver's libraries are loaded by then (instance and physical devices made)
    const char *e = getenv("DM_KGSL_CAPTURE"); if (!e || !*e || real_ioctl) return;
    g_start_s = atof(e); const char *c = strchr(e, ':'); g_left = c ? atoi(c + 1) : 30; g_t0 = now_s();
    real_ioctl = (int (*)(int, int, ...))dlsym(RTLD_DEFAULT, "ioctl"); real_mmap = (void *(*)(void *, size_t, int, int, int, off_t))dlsym(RTLD_DEFAULT, "mmap");
    real_munmap = (int (*)(void *, size_t))dlsym(RTLD_DEFAULT, "munmap");
    snprintf(g_path, sizeof g_path, "%s/kgsl-capture.rd", root ? root : ".");
    dl_iterate_phdr(patch_lib, NULL);
    KLOG("armed: %d submissions from %.0f s after start", g_left, g_start_s);
}
void kgsl_capture_report(void) {   // the ioctl histogram (which submission path the driver uses)
    if (!real_ioctl) return; char b[600]; int o = snprintf(b, sizeof b, "ioctls by number (0x09 type):");
    for (int i = 0; i < 256 && o < (int)sizeof b - 24; i++) if (g_hist[i]) o += snprintf(b + o, sizeof b - (size_t)o, " %#x=%llu", i, (unsigned long long)g_hist[i]);
    KLOG("%s; %d buffers tracked, %llu GPU_COMMAND submissions", b, g_nb, (unsigned long long)g_subs);
}
