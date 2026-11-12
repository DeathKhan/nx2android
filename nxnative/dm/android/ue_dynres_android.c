// Unreal's dynamic-resolution frame-time budget (DM_UE_DYNRES_BUDGET), the Android switch (from droidplay-core e4d92e5c8): a game that targets 30 fps on the Switch scales its internal size to fit 33.3 ms,
// so a host that could present 60 holds every second refresh. The switch writes r.DynamicRes.FrameTimeBudget (a float console variable the rewriter names in main's fixups,
// dm/tools/ue_cvar.py) to the number of milliseconds it gives, on every present; 0 or unset leaves the game's own budget. Android's own file, not the shared host's
// (CONTRIBUTING: Android behaviour sits behind a hook; hle_nvn.c keeps the weak dm_ue_dynres_tick and calls it beside dm_ue_cvar_tick).
#include "hle.h"
#include <mach/mach.h>
#include <TargetConditionals.h>

extern uint64_t dm_ue_cvar(const char *name);   // hle_uecvar.c: main's offset of a console variable, 0 if the rewriter found none
static int rd(uint64_t a, void *out, size_t n) { vm_size_t got = 0; return a >= 0x1000 && vm_read_overwrite(mach_task_self(), (vm_address_t)a, n, (vm_address_t)out, &got) == KERN_SUCCESS && got == n; }

// DM_UE_DYNRES_BUDGET=<ms>: the frame time Unreal's dynamic resolution aims for (r.DynamicRes.FrameTimeBudget, a float; the engine's default is 33.3 ms, the
// Switch's 30 fps). A game whose frame misses that budget lowers the rendering resolution to meet it, so at 16.7 the controller scales toward 60 fps instead of
// presenting every second refresh at the full size. The host writes both thread copies whenever the game changes them; 0 or unset leaves the game's own budget.
void dm_ue_dynres_tick(void) {
    static int state = -1; static float target; static uint64_t off, base;
    if (state < 0) {
        state = 0;
        const char *e = getenv("DM_UE_DYNRES_BUDGET"); if (!e || !*e) return;
        target = (float)atof(e); if (!(target > 0)) { LOG("[RED] DM_UE_DYNRES_BUDGET=%s: not a positive number of milliseconds; the game's budget stays", e); return; }
        for (int i = 0; i < g_nmod; i++) if (!strcmp(g_mod[i].name, "main")) base = (uint64_t)(uintptr_t)g_mod[i].base;
        off = dm_ue_cvar("r.DynamicRes.FrameTimeBudget");
        if (!base || !off) { LOG("[RED] main: Unreal r.DynamicRes.FrameTimeBudget not found in main by the rewriter (ue_cvar.py; DM_UE_DYNRES_BUDGET=%s not applied)", e); return; }
        state = 1;
        LOG("main: Unreal r.DynamicRes.FrameTimeBudget %g ms (DM_UE_DYNRES_BUDGET; the resolution controller scales toward it)", target);
    }
    if (state != 1) return;
    uint64_t obj = 0;
    if (!rd(base + off + 0x10, &obj, sizeof obj) || !obj) return;
    float cur[2] = {0, 0};
    if (!rd(obj, cur, sizeof cur)) return;
    if (cur[0] == target && cur[1] == target) return;
    for (int c = 0; c < 2; c++) { float *p = (float *)(uintptr_t)(obj + 4 * (uint64_t)c); dm_guest_write_prepare(p, sizeof *p); *p = target; }
}
