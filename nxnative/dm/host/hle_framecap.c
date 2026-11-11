// The pause menu's frame cap (front_android.c "Frame cap": Off, 60, 30 fps; settings.txt fpscap=). The cap must not change the game's speed. A game that steps its time by the
// present interval it chose (Donkey Kong Bananza: its frame-rate controller picks interval 1, 2 or 3 from the measured frame time and the game steps by it) ran at two thirds of
// its speed when the host presented at 40 fps behind its back (perf_android.c's adaptive rate, now off in its profile): a cap that only delays presents slows such a game.
// So the cap goes through the game's own choice where the profile shows how (DM_FRAME_CAP_GAME): an OBS_FRAME_CAP gate sits where the game's controller reads its frame time
// (in ticks of the guest's system counter, the register in the gate number's upper bits), and the cap lengthens that reading to the capped period. The frame really is on screen
// that long; the controller then asks for that interval itself, with its own hysteresis, and steps the game by it. A frame longer than the cap still reads as itself (the game
// may go lower, as it does at 20 fps in its cutscenes). Off reads the measure untouched.
// A game whose time follows the clock can be paced by the host instead (profile env DM_FRAME_CAP_MODE=pace: the present period is held at the cap's, hle_nvn.c present_period).
// Every port at 60 fps by an imported FPSLocker config is paced so without a profile row: the config holds the game's values at 30 and at 60 (hle_fpslock.c imp_tick writes the
// 30 fps ones under the cap).
// A game whose main has frame step stubs (patch_nso.py FRAME_STEP) steps by the vsyncs the host holds each frame: the cap sets the shortest hold (hle_nvn.c fps_step_cap),
// 30 = two vsyncs, and game time stays real time (DM_FRAME_CAP_STEP).
// None of them: the cap is not offered (the menu says so) - nothing is changed.
#include "dm.h"
#include "hle.h"
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>

static _Atomic int g_cap;   // 0 off, else fps (60, 30)
static int g_pace = -1;
__attribute__((weak)) int dm_fps_lock_imported(void) { return 0; }   // hle_fpslock.c
int dm_frame_cap_how(void) {
    if (gate_obs_present(OBS_FRAME_CAP)) return DM_FRAME_CAP_GAME;
    if (dm_fps_state) { float t; memcpy(&t, dm_fps_state, sizeof t); if (t > 0) return DM_FRAME_CAP_STEP; }   // (target 0: the stubs run the game's own code)
    if (g_pace < 0) { const char *e = getenv("DM_FRAME_CAP_MODE"); g_pace = e && !strcmp(e, "pace"); }
    return g_pace || dm_fps_lock_imported() ? DM_FRAME_CAP_PACE : DM_FRAME_CAP_NONE;
}
int dm_frame_cap(void) { return atomic_load(&g_cap); }
static _Atomic uint64_t g_seen, g_raised;   // the controller's readings, and those the cap lengthened (since the cap was last set)
void dm_frame_cap_set(int fps) {
    fps = fps == 60 || fps == 30 ? fps : 0;
    if (atomic_exchange(&g_cap, fps) == fps) return;
    atomic_store(&g_seen, 0); atomic_store(&g_raised, 0);
    if (!fps) LOG("[framecap] off: the game's own frame rate");
    else LOG("[framecap] %d fps", fps);   // (how: once the game's modules are loaded, the gate's first hit or the menu says)
}
// The controller's frame time, in ticks of the guest's system counter (guest_ticks: 31.25 MHz on the Switch 2, 19.2 MHz on the Switch)
static void res_tick(void);
// The game's time against the wall clock, counted where its controller runs once a frame: a game that steps by its present interval advances interval / 60 s a frame.
// Logged every 600 frames with the frames at each interval, so a cap that slowed the game (frames longer than the interval it steps by) shows as a ratio below 1
__attribute__((weak)) int nvn_game_interval(void) { return 1; }
static void game_time_tick(void) {
    static uint64_t t0, last; static double game; static unsigned n, at[4];
    uint64_t now = now_ns(); int itv = nvn_game_interval(); itv = itv < 1 ? 1 : itv > 3 ? 3 : itv;
    if (!t0 || now - last > 250000000ull) { t0 = now; game = 0; n = 0; memset(at, 0, sizeof at); last = now; return; }   // the first frame, or after a hold (menu, background, loading)
    game += itv / 60.0; at[itv]++; last = now;
    if (++n == 600) {
        double wall = (now - t0) / 1e9;
        LOG("[framecap] game time %.2f s in %.2f s (%.3f of real time, %.1f fps; frames at interval 1 / 2 / 3: %u / %u / %u; cap %d)", game, wall, game / wall, n / wall, at[1], at[2], at[3], atomic_load(&g_cap));
        t0 = now; game = 0; n = 0; memset(at, 0, sizeof at);
    }
}
static void cap_measure(uint64_t *r, int param) {
    res_tick();   // (once a frame: the resolution level's setting, held against the game's own writes)
    game_time_tick();
    int cap = atomic_load(&g_cap), reg = param > 0 && param < 29 ? param : 8; if (cap <= 0 || cap >= 60) return;
    int64_t v = (int64_t)r[reg]; if (v <= 0) return;   // none measured: the controller keeps its interval
    int64_t hz = dm_sdk_ounce() ? 31250000 : 19200000, floor = hz / cap;   // the capped period, whole vsyncs of 1/60 s
    static int told; if (told != cap) { told = cap; LOG("[framecap] %d fps: the game's frame-rate controller reads each frame as at least %d vsyncs long and picks that present interval itself", cap, 60 / cap); }
    uint64_t n = atomic_fetch_add(&g_seen, 1) + 1;
    if (v < floor) { r[reg] = (uint64_t)floor; atomic_fetch_add(&g_raised, 1); }
    if (n % 1800 == 0) LOG("[framecap] %d fps: %llu of the last %llu frame times the game's controller read were shorter than the cap and read as %.1f ms", cap, (unsigned long long)atomic_load(&g_raised), (unsigned long long)n, 1000.0 / cap);
    if (n % 1800 == 0) { atomic_store(&g_seen, 0); atomic_store(&g_raised, 0); }
}
uint64_t dm_frame_cap_period(uint64_t period) {
    int cap = atomic_load(&g_cap);
    if (!cap || !period || dm_frame_cap_how() != DM_FRAME_CAP_PACE) return period;
    uint64_t p = 1000000000ull / (uint64_t)cap;
    return period < p ? p : period;
}

// ---- Resolution: the operation mode and the game's own dynamic-resolution target. Donkey Kong Bananza's renderer settings (main+0x4c5228 builds them, sead-style parameter objects
// with the value at +0x64) include MaintainGpuPercentage (87.5 by default: the share of the frame's period the GPU may take before the game lowers its render size),
// FrameBufferMinScale (0.6) and NdsrUpscaleMinRate / MaxRate (1.375 / 1.9: the upscaler's range). OBS_RES_PARAM gates on the stores of those objects give the host their addresses; a
// level then sets the GPU percentage the game's own controller holds, and the game picks its render size within its own bounds. The frame time the frame-rate controller reads is
// not touched (hle_framecap.c's frame cap above): a higher target can make the game miss its interval more often, and then it steps to the longer one itself, at the right speed.
// The game makes its renderer settings more than once (four sets in the first minute: a title, a level), loads their values from its files right after, and drops the
// old sets. Each set is kept with the object's vtable; a value the host did not write is the game's own (its default, then its file's), and the level's value is written over
// it at once, so the renderer finds it when it sets up a scene: the upscale range is read then (a change reaches the picture at the next area or load), the GPU target every
// frame. A set whose vtable changed (freed) is never written, nor a value outside the setting's range
typedef struct { uint64_t *obj; uint64_t vt; float orig, wrote; } res_obj;
#define RES_SETS 8
static res_obj g_res_obj[DM_RES_P_N][RES_SETS]; static int g_res_n[DM_RES_P_N];
static _Atomic int g_res = -1;           // the level, -1 = never set (the game's own)
static const char *const RES_NAMES[DM_RES_N] = {"performance", "balanced", "quality", "max"};
// What each level asks of the game's own settings (Bananza handheld: the 1920x1080 picture from a render size of 960x540 to 1397x786, the upscale rate 1.375 to 2, the GPU
// target 90 % of the frame). Performance: the rate from 1.667 (render size at most 1152x648) and 70 % of the GPU target. Balanced: the game's own. Quality: the rate from 1
// (up to the screen's 1920x1080 where the GPU target allows). Max: docked, the game's own docked settings (a larger picture the host scales to the screen)
static const float RES_GPU_SCALE[DM_RES_N] = {0.70f, 1.0f, 1.0f, 1.0f};
static const float RES_NDSR_MIN[DM_RES_N] = {1.6667f, 0, 1.0f, 0};   // 0: the game's own
int dm_res_levels(void) { return gate_obs_present(OBS_RES_PARAM) ? DM_RES_N : 2; }
void dm_res_set(int level) {
    if (level < 0 || level >= DM_RES_N || atomic_exchange(&g_res, level) == level) return;
    LOG("[res] level %s: %s%s", RES_NAMES[level], level == DM_RES_MAX ? "docked" : "handheld", level == DM_RES_PERF ? ", a lower GPU target where the game has one (render size settles lower)" :
        level == DM_RES_QUALITY ? ", up to the screen's size where the game has an upscale floor" : "");
}
static void res_param_made(uint64_t *r, int param) {
    int which = param & 15, reg = param >> 4 & 31; if (which <= 0 || which >= DM_RES_P_N || reg > 28 || !r[reg]) return;
    res_obj *o = &g_res_obj[which][g_res_n[which]++ % RES_SETS];
    *o = (res_obj){(uint64_t *)(uintptr_t)r[reg], *(uint64_t *)(uintptr_t)r[reg], 0, -1};
    static const char *const nm[DM_RES_P_N] = {"", "MaintainGpuPercentage", "FrameBufferMinScale", "NdsrUpscaleMinRate", "NdsrUpscaleMaxRate"};
    LOG("[res] the game made its %s setting at %#llx", nm[which], (unsigned long long)r[reg]);
}
static void res_tick(void) {
    int lv = atomic_load(&g_res);
    static const int P[2] = {DM_RES_P_GPU_PCT, DM_RES_P_NDSR_MIN};
    static const float LO[2] = {10, 0.5f}, HI[2] = {200, 4};
    for (int p = 0; p < 2; p++) for (int i = 0; i < RES_SETS && i < g_res_n[P[p]]; i++) {
        res_obj *o = &g_res_obj[P[p]][i]; if (!o->obj || *o->obj != o->vt) continue;
        float *v = (float *)((uint8_t *)o->obj + 0x64), now = *v;
        if (now != o->wrote) {   // the game's own value (its default, its file's)
            if (!(now >= LO[p] && now <= HI[p])) continue;
            if (now != o->orig) LOG("[res] the game's %s (set %d): %.3f", p ? "lowest upscale rate" : "GPU target, % of the frame", i, now);
            o->orig = now;
        }
        if (lv < 0) continue;
        float want = !p ? o->orig * RES_GPU_SCALE[lv] : RES_NDSR_MIN[lv] > 0 ? RES_NDSR_MIN[lv] : o->orig;
        if (now == want) { o->wrote = want; continue; }
        LOG("[res] %s: the game's %s %.3f → %.3f (set %d)", RES_NAMES[lv], p ? "lowest upscale rate" : "GPU target", now, want, i);
        *v = want; o->wrote = want;
    }
}
__attribute__((constructor)) static void frame_cap_attach(void) { gate_obs_set(OBS_FRAME_CAP, cap_measure); gate_obs_set(OBS_RES_PARAM, res_param_made); }
