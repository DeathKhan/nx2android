"""The pause menu's frame cap and Resolution level (nxnative/dm/host/hle_framecap.c), with its actual C code over a stand-in for dm.h."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

SOURCE = Path(__file__).resolve().parents[1] / 'nxnative/dm/host/hle_framecap.c'

HLE = r'''
#include <stdint.h>
extern uint8_t *dm_fps_state;   // hle.h: the frame step state page (+0 float target fps); NULL when the build has none
'''

STUB = r'''
#include <stdint.h>
#include <stdio.h>
#define LOG(...) do { if (0) printf(__VA_ARGS__); } while (0)
enum { OBS_FRAME_CAP = 28, OBS_RES_PARAM = 29 };
enum { DM_FRAME_CAP_NONE, DM_FRAME_CAP_GAME, DM_FRAME_CAP_PACE, DM_FRAME_CAP_STEP };
enum { DM_RES_PERF, DM_RES_BALANCED, DM_RES_QUALITY, DM_RES_MAX, DM_RES_N };
enum { DM_RES_P_GPU_PCT = 1, DM_RES_P_MIN_SCALE, DM_RES_P_NDSR_MIN, DM_RES_P_NDSR_MAX, DM_RES_P_N };
extern int t_gate, t_ms;
static inline int gate_obs_present(int id) { (void)id; return t_gate; }
void gate_obs_set(int id, void (*fn)(uint64_t *, int));
static inline int dm_sdk_ounce(void) { return 1; }
static inline uint64_t now_ns(void) { return (uint64_t)t_ms * 1000000ull; }
int dm_frame_cap_how(void); int dm_frame_cap(void); void dm_frame_cap_set(int fps); uint64_t dm_frame_cap_period(uint64_t period);
int dm_res_levels(void); void dm_res_set(int level);
'''

HARNESS = r'''
#include <assert.h>
#include <math.h>
int t_gate = 1, t_ms;
uint8_t *dm_fps_state;
static void (*g_fn[32])(uint64_t *, int);
void gate_obs_set(int id, void (*fn)(uint64_t *, int)) { g_fn[id] = fn; }
int nvn_game_interval(void) { return 2; }
static int controller(int64_t ticks) {   // Donkey Kong Bananza's choice (main+0x204a18): floor(ticks / 31.25 MHz * 60 - 0.2) + 1, clamped to 1..3
    float v = (float)ticks / 31250000.0f * 60.0f - 0.2f; int k = (int)v + 1; return k < 1 ? 1 : k > 3 ? 3 : k;
}
static int64_t measure(int64_t ticks) { uint64_t r[31] = {0}; r[8] = (uint64_t)ticks; t_ms += 33; g_fn[OBS_FRAME_CAP](r, 8); return (int64_t)r[8]; }
int main(void) {
    assert(g_fn[OBS_FRAME_CAP] && g_fn[OBS_RES_PARAM]);
    assert(dm_frame_cap_how() == DM_FRAME_CAP_GAME && dm_res_levels() == DM_RES_N);
    int64_t light = 300000;   // 9.6 ms of work: the game picks interval 1
    assert(controller(measure(light)) == 1);
    dm_frame_cap_set(30); assert(dm_frame_cap() == 30);
    assert(controller(measure(light)) == 2);                 // capped: the game picks interval 2 itself
    assert(controller(measure(1200000)) == 3);               // a frame longer than the cap reads as itself (38 ms: the game's own interval 3)
    assert(measure(0) == 0 && measure(-5) == -5);            // nothing measured: untouched
    dm_frame_cap_set(60); assert(controller(measure(light)) == 1);   // 60: the game's own choice, as off
    dm_frame_cap_set(0); assert(measure(light) == light);
    dm_frame_cap_set(45); assert(dm_frame_cap() == 0);      // not a cap the game can name
    assert(dm_frame_cap_period(16666667) == 16666667);       // the game's own controller: the host does not pace
    // Resolution: the game's setting objects (vtable, value at +0x64), written over the game's own values, never once freed
    static uint64_t obj[16]; obj[0] = 0xabcdef; float *v = (float *)((uint8_t *)obj + 0x64); *v = 87.5f;
    uint64_t r[31] = {0}; r[20] = (uint64_t)(uintptr_t)obj; g_fn[OBS_RES_PARAM](r, 1 | 20 << 4);
    static uint64_t nd[16]; nd[0] = 0xabcdef; float *m = (float *)((uint8_t *)nd + 0x64); *m = 1.375f;
    r[20] = (uint64_t)(uintptr_t)nd; g_fn[OBS_RES_PARAM](r, 3 | 20 << 4);
    measure(light); assert(*v == 87.5f && *m == 1.375f);   // no level yet: the game's own
    dm_res_set(DM_RES_PERF); measure(light); assert(fabsf(*v - 61.25f) < 1e-4f && fabsf(*m - 1.6667f) < 1e-4f);
    *v = 90.0f; *m = 1.375f;   // the game loads its settings file over them
    measure(light); assert(fabsf(*v - 63.0f) < 1e-4f && fabsf(*m - 1.6667f) < 1e-4f);   // 70 % of the file's 90
    dm_res_set(DM_RES_BALANCED); measure(light); assert(*v == 90.0f && *m == 1.375f);
    dm_res_set(DM_RES_QUALITY); measure(light); assert(*v == 90.0f && *m == 1.0f);
    dm_res_set(DM_RES_MAX); measure(light); assert(*v == 90.0f && *m == 1.375f);   // Max: docked, the game's own settings
    obj[0] = 0x1234; dm_res_set(DM_RES_PERF); measure(light); assert(*v == 90.0f);   // freed (another vtable): never written
    obj[0] = 0xabcdef; *v = 1e9f; measure(light); assert(*v == 1e9f);   // a value outside the setting's range: left alone
    puts("ok"); return 0;
}
'''

# A game paced by the host (profile env DM_FRAME_CAP_MODE=pace, no gate): Nintendo Switch 2 Welcome Tour, whose Bezel engine runs as many fixed 1/60 s updates a frame as the
# elapsed time asks, so the cap is only the present period, live from Off to 30 and back
HARNESS_PACE = r'''
#include <assert.h>
#include <stdlib.h>
#include <string.h>
int t_gate = 0, t_ms;
uint8_t *dm_fps_state;
void gate_obs_set(int id, void (*fn)(uint64_t *, int)) { (void)id; (void)fn; }
int main(void) {
    setenv("DM_FRAME_CAP_MODE", "pace", 1);
    assert(dm_frame_cap_how() == DM_FRAME_CAP_PACE);
    assert(dm_frame_cap_period(16666667) == 16666667);       // off: the game's own period
    dm_frame_cap_set(30); assert(dm_frame_cap_period(16666667) == 33333333);
    assert(dm_frame_cap_period(50000000) == 50000000);       // a longer period the game asks for stays
    assert(dm_frame_cap_period(0) == 0);                     // no period (unpaced) stays unpaced
    dm_frame_cap_set(60); assert(dm_frame_cap_period(16666667) == 16666667);
    dm_frame_cap_set(0); assert(dm_frame_cap_period(16666667) == 16666667);
    // a main with frame step stubs (DM_FRAME_CAP_STEP): the host's step hold carries the cap (hle_nvn.c fps_step_cap), the present period is not held
    static uint8_t page[32]; dm_fps_state = page; assert(dm_frame_cap_how() == DM_FRAME_CAP_PACE);   // target 0: the stubs run the game's own code
    float target = 60.0f; memcpy(page, &target, sizeof target); assert(dm_frame_cap_how() == DM_FRAME_CAP_STEP);
    dm_frame_cap_set(30); assert(dm_frame_cap() == 30 && dm_frame_cap_period(16666667) == 16666667);
    dm_frame_cap_set(0);
    puts("ok"); return 0;
}
'''


class FrameCap(unittest.TestCase):
    def test_cap_and_resolution(self):
        self.run_harness(HARNESS)

    def test_cap_paced_by_host(self):
        self.run_harness(HARNESS_PACE)

    def run_harness(self, harness):
        cc = shutil.which('cc') or shutil.which('clang')
        if not cc:
            self.skipTest('a C compiler is required')
        with tempfile.TemporaryDirectory() as d:
            Path(d, 'dm.h').write_text(STUB, encoding='utf-8')
            Path(d, 'hle.h').write_text(HLE, encoding='utf-8')
            Path(d, 't.c').write_text('#include "dm.h"\n' + harness, encoding='utf-8')
            Path(d, 'hle_framecap.c').write_text(SOURCE.read_text(encoding='utf-8'), encoding='utf-8')   # beside the stand-in dm.h
            exe = Path(d, 't')
            subprocess.run([cc, '-std=gnu11', '-I', d, str(Path(d, 'hle_framecap.c')), str(Path(d, 't.c')), '-o', str(exe), '-lm'], check=True)
            self.assertEqual(subprocess.run([str(exe)], capture_output=True, text=True, check=True).stdout.strip(), 'ok')


if __name__ == '__main__':
    unittest.main()
