// Explicit debug-only RenderDoc requests. Never load or inject a capture layer.
// DM_CTL "rdoccapture" requests one whole present-delimited frame; "rdocstatus" reports it.
#include "dm.h"
#include <stdbool.h>
#include "../third_party/renderdoc/renderdoc_app.h"
#include <dlfcn.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

static pthread_mutex_t g_rdoc_mu = PTHREAD_MUTEX_INITIALIZER;
static RENDERDOC_API_1_6_0 *g_rdoc;
static void *g_rdoc_handle; // Retain the NOLOAD reference while API pointers remain in use.
static _Atomic int g_rdoc_pending;
static uint32_t g_rdoc_before;

static int rdoc_api_locked(void) {
    if (g_rdoc) return 1;
    pRENDERDOC_GetAPI get = (pRENDERDOC_GetAPI)dlsym(RTLD_DEFAULT, "RENDERDOC_GetAPI");
    void *handle = NULL;
    if (get) { // Pin the owner even when its symbol was globally visible.
        Dl_info owner;
        if (dladdr((void *)get, &owner) && owner.dli_fname)
            handle = dlopen(owner.dli_fname, RTLD_NOW | RTLD_NOLOAD);
        if (!handle) get = NULL;
    }
    if (!get) {
        handle = dlopen("libVkLayer_GLES_RenderDoc.so", RTLD_NOW | RTLD_NOLOAD);
        if (handle) get = (pRENDERDOC_GetAPI)dlsym(handle, "RENDERDOC_GetAPI");
    }
    RENDERDOC_API_1_6_0 *api = NULL;
    if (!get || !get(eRENDERDOC_API_Version_1_6_0, (void **)&api) || !api ||
        !api->SetCaptureFilePathTemplate || !api->GetNumCaptures || !api->GetCapture || !api->TriggerCapture) {
        if (handle) dlclose(handle);
        LOG("[renderdoc] capture layer/API absent; no library loaded or capture requested");
        return 0;
    }
    g_rdoc = api; g_rdoc_handle = handle;
    int major = 0, minor = 0, patch = 0;
    if (api->GetAPIVersion) api->GetAPIVersion(&major, &minor, &patch);
    LOG("[renderdoc] using already-loaded API %d.%d.%d", major, minor, patch);
    return 1;
}

// Called with the module mutex held, only after a request. Captures have immutable paths;
// query the required length before allocating, and never infer success merely from a present.
static void rdoc_poll_locked(void) {
    if (!atomic_load_explicit(&g_rdoc_pending, memory_order_relaxed)) return;
    uint32_t count = g_rdoc->GetNumCaptures();
    if (count <= g_rdoc_before) return;
    uint32_t n = 0; uint64_t timestamp = 0;
    if (!g_rdoc->GetCapture(g_rdoc_before, NULL, &n, &timestamp)) return;
    if (!n || n > 65536) {
        LOG("[renderdoc] new capture metadata has invalid path length %u", n);
        atomic_store_explicit(&g_rdoc_pending, 0, memory_order_release);
        return;
    }
    char *path = calloc((size_t)n + 1, 1);
    if (!path) return;
    uint32_t capacity = n;
    if (!g_rdoc->GetCapture(g_rdoc_before, path, &capacity, &timestamp)) { free(path); return; }
    path[n] = 0;
    struct stat st;
    if (!stat(path, &st) && S_ISREG(st.st_mode))
        LOG("[renderdoc] capture completed index %u time %llu bytes %llu: %s", g_rdoc_before,
            (unsigned long long)timestamp, (unsigned long long)st.st_size, path);
    else LOG("[renderdoc] capture metadata available index %u time %llu; file not accessible: %s",
             g_rdoc_before, (unsigned long long)timestamp, path);
    free(path);
    atomic_store_explicit(&g_rdoc_pending, 0, memory_order_release);
}

void dm_renderdoc_capture(void) {
#if defined(DM_RELEASE)
    return;
#else
    pthread_mutex_lock(&g_rdoc_mu);
    if (g_rdoc) rdoc_poll_locked();
    if (atomic_load_explicit(&g_rdoc_pending, memory_order_relaxed)) {
        LOG("[renderdoc] one capture is already pending; request ignored");
        pthread_mutex_unlock(&g_rdoc_mu); return;
    }
    const char *root = getenv("DM_ROOT"); char path[1024];
    int n = root && root[0] == '/' ? snprintf(path, sizeof path, "%s/renderdoc/capture", root) : -1;
    if (n < 0 || (size_t)n >= sizeof path) {
        LOG("[renderdoc] capture requires a valid absolute app files DM_ROOT");
        pthread_mutex_unlock(&g_rdoc_mu); return;
    }
    if (!rdoc_api_locked()) { pthread_mutex_unlock(&g_rdoc_mu); return; }
    g_rdoc->SetCaptureFilePathTemplate(path);
    g_rdoc_before = g_rdoc->GetNumCaptures();
    atomic_store_explicit(&g_rdoc_pending, 1, memory_order_release);
    g_rdoc->TriggerCapture();
    LOG("[renderdoc] requested one next present-delimited frame, baseline captures %u, path %s", g_rdoc_before, path);
    pthread_mutex_unlock(&g_rdoc_mu);
#endif
}

// Fast no-op until an explicit request. Called after vkQueuePresentKHR and queue unlock.
void dm_renderdoc_poll(void) {
    if (!atomic_load_explicit(&g_rdoc_pending, memory_order_acquire)) return;
    pthread_mutex_lock(&g_rdoc_mu);
    rdoc_poll_locked();
    pthread_mutex_unlock(&g_rdoc_mu);
}
void dm_renderdoc_status(void) {
    pthread_mutex_lock(&g_rdoc_mu);
    if (g_rdoc) rdoc_poll_locked();
    LOG("[renderdoc] API %s, request %s", g_rdoc ? "loaded" : "not requested/absent",
        atomic_load_explicit(&g_rdoc_pending, memory_order_relaxed) ? "pending" : "idle");
    pthread_mutex_unlock(&g_rdoc_mu);
}
