// Frame step in whole vsyncs (pure function, shared by Mac / iOS; unit test tests/t_fstep.c). The rewriter's frame-step stubs (patch_nso.py FRAME_STEP: Tears of the Kingdom 1.4.3,
// Breath of the Wild 1.9.0) step the game by the vsyncs (1/60 s) the host hands them each frame, and the host keeps each frame on screen for k vsyncs (hle_nvn.c present_itv).
// At a steady rate the step is the time on screen: even motion, and only steps the games' own interval logic uses (60 / 30 / 20 fps). Real time the game falls behind (a late
// frame, a hitch) is a debt paid back one extra vsync per frame, so over time game time = real time. k follows the frames' cost: three late frames in a row step up to the cost
// (30 fps when 60 does not hold: no 40 fps steps on 120 Hz screens); 30 frames in a row whose cost would fit k − 1 try it, and a trial that fails waits twice as long before
// the next (1 s → 8 s), so a device on the edge does not flap.
// A probe is a trial back to kmin that the cost does not justify: while the caller reports the CPU slowed for reasons other than heat (an iPhone's performance controller
// parks a game that waits out a long hold on the efficiency cores at half the clock, where its cost never fits k − 1 again), the hold-off alone starts one. Its first
// FSTEP_GRACE frames may be late without failing it, the time the controller needs to raise the clock for the shorter deadline; after them it is judged like a trial.
// A probe that holds leaves the hold-off as it is, and the run at kmin it starts must last FSTEP_HOLD_MAX (8 s) before the hold-off returns to 1 s: a shorter one counts as
// a failed probe, so a CPU that really cannot hold kmin is not probed every second.
// A step up after a run at kmin of FSTEP_HOLD_MAX keeps the cost that held there, and a cost at the longer hold of at most FSTEP_SAME times it starts a probe too: the scene
// costs what it did, so the step up came from a hitch (Breath of the Wild on an iPhone: pipeline compiles after a cutscene). The fit trial alone does not bring such a game
// back: an iPhone lowers its GPU clock until the work fills most of the frame, so a GPU-bound game's cost sits near FSTEP_FIT at either hold (BotW: GPU 12.8–15.2 ms at hold
// 2, p50, where hold 1 had held for minutes; the same 4,900 draws took 5.8 ms after a trip to the background and 13.5 ms half a minute later).
// A cap that falls and rises again (a movie, hle_movie.c dm_movie_playing; the iOS thermal cap in hle_nvn.c) keeps what was learned at each kmin: the cost that held it
// (cref) and the GPU reference (nvn_fgpu below); and the frames after the cap rises are a probe with its grace, as the clocks are still set for the longer hold. Before, the
// rise set k to kmin with no grace, so three frames late while the clock rose stepped up again, with cref dropped at the cap change and a GPU reference taken at the 30 fps
// cap: neither the fit trial (a GPU-bound cost near FSTEP_FIT) nor a probe brought kmin back (t_fstep: the movie and thermal cap cases). A cref kept from before the cap
// starts a probe only when the cost at the longer hold is within FSTEP_SAME of it, and a failed probe doubles the hold-off as any other does.
// A hold above kmin that the frames held, on time, for FSTEP_RUN frames keeps their cost (kheld): from a longer hold, a trial back to it starts after the hold-off when that
// cost fits it, whatever the cost at the longer hold is, with a probe's grace for the clock to rise. A GPU's clock falls until the work fills the longer hold: Ocarina of Time on
// the Odin 3 took 20-22 ms a frame at hold 2, then 26-31 ms at hold 3 after one heavy moment stepped it there, a cost at FSTEP_FIT of hold 2 that never made the fit trial's run of
// 30, so it stayed at 20 fps (the probe goes to kmin, 60 fps, which does not hold there). Three such trials failing in a row forget the cost until the frames hold that hold again
#pragma once
#include <math.h>
#include <stdint.h>

#define FSTEP_MAX 3           // the largest step: both games' own interval logic uses 1–3 (BotW clamps to [2, 3], TOTK picks 2 or 3), so their integer readers never see 4
#define FSTEP_LATE 0.1        // a frame is late when it or its GPU work took this many vsyncs (1.7 ms) more than k: 53 fps is not 60 (start-to-start jitter on time is well below)
#define FSTEP_FIT 0.85        // a frame would fit k − 1 when its cost is at most this share of it
#define FSTEP_RUN 30          // frames in a row that fit k − 1 before a trial, and frames a trial must hold
#define FSTEP_MISS 3          // frames of such a run that fit k but not k − 1 (not late): more end the run. A scene's odd heavier frame (OoT on the Odin 3 at hold 3: 1.86 vsyncs
                              // every dozen frames of 1.2) ended every run, and a trial back to hold 2 waited over 700 frames
#define FSTEP_SETTLE 4        // a trial's first frames, whose lateness neither fails it nor counts: presents queued at the longer hold still come at its pace (the Odin 3: the first
                              // three frames of every trial from hold 3 to 2 showed 3 vsyncs at a GPU time of 1.2, and failed it)
#define FSTEP_HITCH_MAX 4     // trials back to kmin in a row that may fail on hitches alone (late frames whose cost fits kmin) without doubling the hold-off: Pokémon Legends:
                              // Arceus's title stalls ended its first trials to 60 fps (cost 0.13-0.34 vsyncs) and the hold-off went 1, 2, 4 s, so 60 came 5-8 s after start-up
#define FSTEP_DEBT_MAX 15.0   // debt kept at most (0.25 s): the rest of a long stall (a load) is dropped, not replayed
#define FSTEP_HOLD0 60.0      // vsyncs since the last step up before a trial: 1 s, doubled after each failed trial up to 8 s
#define FSTEP_HOLD_MAX 480.0
#define FSTEP_GRACE 30        // a probe's first frames, whose lateness neither fails it nor counts toward a step up (the controller lowered the clock within 10 frames of a step to 30)
#define FSTEP_SAME 1.1        // the cost at the longer hold against the cost that held kmin: at most this much is the same scene (BotW at hold 2: 1.0–1.07 times its hold-1 GPU time)
// The grace is renewed (nvn_fstep.ramp) while the probe's frames keep getting cheaper: a controller that raises the clock slowly is still raising it. macOS raises the GPU clock
// of a game that misses its deadline at 85–100% GPU occupancy by about 100 MHz a second (Breath of the Wild on an M5 Pro: 338 → 432 MHz over a 33-frame probe, GPU 24.8 → 18.5 ms,
// then the probe failed at 22 ms frames; hold 1 needs about 560 MHz there), where the iPhone's controller settles within the 30 frames. A window of FSTEP_GRACE frames whose
// second half cost at most 1 − FSTEP_RAMP of its first half, and had a late frame (a window on time is judged as a trial), starts another, up to FSTEP_PROBE_MAX frames in all
#define FSTEP_RAMP 0.05       // the fall in cost from one half of a grace window to the next that shows the clock still rising (the per-frame noise of a 25 ms frame is 4%, of 15 frames 1%)
#define FSTEP_PROBE_MAX 180   // grace frames a probe may take in all (3 s at 60 fps): a clock that keeps rising without reaching the hold is not waited for longer
// At kmin, late frames step up only when the overrun lasts or is large (lateness by shown and cost; a screen verdict, nvn_mtl.m snapshots, steps up after three as before).
// Pokémon Legends: Arceus on the Odin 3 at 60 fps: turning the camera costs the CPU 1.18-1.29 vsyncs for a few frames at a time (GPU 0.91-0.93), and three such frames in
// a row stepped to 30, which the probes took 0.5-3 s to undo: up to 180 frames of every 1,800 at hold 2 while panning, though only one frame in twenty was late. A late frame
// at kmin is one 33 ms frame on screen, paid back by the step; a step up is 30 fps for a second or more. So at kmin three late frames in a row step up at once only when each
// took FSTEP_HEAVY vsyncs or more beyond the hold (a scene or a load that does not fit: Ocarina of Time's 1.3-2.1-vsync views, a stall); otherwise FSTEP_WIN_LATE late
// frames among the last FSTEP_WIN at kmin do, while the frames' time there (a running mean over about FSTEP_WIN_N frames) is FSTEP_OVER or more beyond the hold: under
// 52 fps for a third of a second. A scripted pan (right stick left and right for 40 s, 2026-11-11) took 1.06-1.15 vsyncs a frame through each turn with a few of 1.27-1.42:
// about 55 fps shown with a step of 2 every few frames, where the old rule spent 456 frames of 1,800 at 30 fps. Steady 18.7 ms frames (1.12) now stay at hold 1 for the same
// reason: 53 fps with a payback every eighth frame rather than 30. A burst that ends a trial, a probe or a probe's run at kmin fails it without doubling the hold-off, as a
// hitch does (FSTEP_HITCH_MAX in a row), unless its late frames came one after another, so the probe after it comes 1 s later
#define FSTEP_HEAVY 0.3       // vsyncs beyond the hold each of three late frames in a row must take to step up at once at kmin
#define FSTEP_WIN 60          // frames at kmin whose lateness is counted (1 s at 60 fps; at most 64)
#define FSTEP_WIN_LATE 20     // late frames among them that step up, with:
#define FSTEP_OVER 0.15       // the frames' running mean time at kmin at least this many vsyncs beyond the hold (1.15 vsyncs: 19.2 ms, 52 fps)
#define FSTEP_WIN_N 16        // the frames the running mean weighs (1 / FSTEP_WIN_N each)
// A frame late at a hold its GPU fits with room (gpu at most FSTEP_FIT × k) by at most FSTEP_LATE_SOFT vsyncs is the replay pipeline's lateness, not the scene's: the chunks'
// turnover and the game's own waits add a few ms over the GPU's time, and a longer hold would only pace the GPU slower (the clock governors follow the hold). Such frames
// are late for the debt (paid back) but do not step up. And from a longer hold, when the GPU's time has fitted k − 1 for FSTEP_RUN frames, k − 1 is tried with a probe's
// grace whatever the cost at k says (the cost at k holds the longer hold's own waits): Ocarina of Time on a OnePlus Pad 3 at hold 3 measured 2.15 vsyncs of cost with
// 1.3 of GPU, and at hold 2 its frames were 2.0–2.1 (27.5 fps, 32 ms of GPU-bound pipeline); the fit trial (cost ≤ 1.7) and the probe (to hold 1) never tried 2. Three such
// trials failing in a row stop them until the hold changes
#define FSTEP_LATE_SOFT 0.5   // 8 ms at 60 Hz: a frame later than that at a hold its GPU fits steps up as before (the debt pays the rest back). Ocarina of Time's Kokiri frames at hold 2 on the Pad 3 are 2.0–2.4 vsyncs with a p95 of 2.65; at 0.3 three of 2.32–2.39 in a row stepped up after 22 s
// A cost at the longer hold that fits kmin as it is (FSTEP_ASIS × kmin, the frames on time) starts a probe too, whatever held kmin before: cref is the cost of the last
// scene that held kmin, and a heavier scene that still fits is not "the same" (Pokémon Legends: Arceus on the Odin 3: a light field held hold 1 at 0.6-0.86, a hitch on the
// way into Jubilife Village stepped to hold 2, and the village's 0.91 vsyncs there (CPU and GPU 15 ms; 12.7 ms of GPU at hold 1) fitted neither the fit trial's 0.85 nor
// 1.1 × cref: 30 fps for minutes with no probe at all, where it holds 60). Such probes that fail in a row wait 2, 4, then 8 times the hold-off (up to 64 s), as the game
// may do more at the shorter hold than the cost at the longer one shows
#define FSTEP_ASIS 1.0
#define FSTEP_GFIT_MISS 10    // late frames a GPU-fit run of FSTEP_RUN may hold: at hold 3 one frame in ten is late there (the hold's own pacing), and FSTEP_MISS (3) ended every run

typedef struct {
    int k, step;              // vsyncs each frame stays on screen; the step handed to the game for the next frame (k, or k + 1 while paying back)
    int kmin;                 // the last call's kmin (a lower one takes effect at once)
    double debt;              // real time the game is behind, in vsyncs
    int late_run, fit_run, trial;   // late frames in a row; frames in a row whose cost fits k − 1; frames left of a trial at k − 1 (0 = none)
    int probe, grace;         // 1: the trial in progress is a probe; 2: a probe held and its run at kmin has not ended yet; frames left of the probe's grace
    int ramp;                 // set by the caller (kept): 1 = a probe's grace is renewed while its frames keep getting cheaper (FSTEP_RAMP; the host's DM_FSTEP_PROBE_RAMP)
    int probe_near;           // set by the caller (kept): 1 = a probe from a hold two or more above kmin goes to the next shorter hold, not to kmin. A phone's GPU governor sizes the clock
                              // to the busy share, so a probe only has to make the frames late at the next hold for the clock to rise; at kmin (60 fps) a 30 fps game's frames are
                              // late whatever the clock, the probe fails, and the step up lands on the cost's hold, where the clock falls again (Ocarina of Time on a Galaxy S25 Edge)
    double pa, pb; int plate, ptotal;   // the current grace window: the cost of its first and second halves (sums), its late frames; grace frames this probe has taken in all
    double hold, since;       // vsyncs a trial waits for after a step up; vsyncs since the last step up
    int per_vsync;            // set by the caller: the ticks a 1/60 s vsync is (0 = 1). At 120 fps the unit is 1/120 s (hle_nvn.c fps_high), and the waits between trials stay the same real time
    double shown_slack;       // vsyncs a frame on time may show longer than k (set by the caller): one refresh on screens whose rate is not a multiple of 60; the cost keeps FSTEP_LATE
    int screen;               // set by the caller before each call, then cleared: 0 = judge lateness by shown and cost; 1 = the screen showed a recent frame on time, 2 = late. With present
                              // snapshots (nvn_mtl.m) the game runs ahead of the screen and a frame that runs long is absorbed, so only what the screen shows counts
    double run, crun, cref;   // vsyncs at kmin in a row; the cost of its frames on time (running mean; not during a probe's trial, whose frames run while the clock rises);
                              // crun when the last run of FSTEP_HOLD_MAX or more ended in a step up or a cap change (0 = none yet)
    double ccur; int ck, cn;  // the cost of the frames on time at the longer hold ck (running mean) and their count
    int ckmin;                // the kmin run, crun and cref belong to
    double cref_at[FSTEP_MAX + 1];   // cref of each kmin, kept while another kmin is in force
    double kheld[FSTEP_MAX + 1];     // the cost of the frames on time at each hold above kmin, once FSTEP_RUN of them held it (0 = none or forgotten)
    int mtrial, mfail;        // the trial in progress is one by kheld; such trials failed in a row
    double gpu;               // set by the caller before each call: the frame's GPU time in vsyncs (0 = not known), for the soft-late rule and the GPU-fit trial
    int gfit_run, gfit_miss, gtrial, gfail;   // frames whose GPU time fits k − 1 (a run; late frames in it, at most FSTEP_GFIT_MISS); the trial in progress is one by the GPU fit; such trials failed in a row
    int atrial, afail;        // the probe in progress came from the cost fitting kmin as it is (FSTEP_ASIS); such probes failed in a row
    int fit_miss, settle;     // frames of the current fit run that fit k but not k − 1; frames left of a trial's settling (FSTEP_SETTLE)
    double late_cmax; int hitch_fails;   // the highest cost of the current late run; trials back to kmin failed on hitches alone in a row (FSTEP_HITCH_MAX)
    double late_tmin; uint64_t lwin;     // the shortest max(shown, cost) of the current late run; the late frames among the last FSTEP_WIN at kmin (bit 0 the latest)
    double cwin;              // the running mean of max(shown, cost) of the frames counted in lwin (0 = none yet)
    int up_by;                // why the last step up came (for the log): 1 three late frames, 2 three heavy late frames at kmin, 3 late frames within FSTEP_WIN at kmin
    uint64_t tolerated;       // late frames at kmin that did not step up (statistics)
    uint64_t late, paid, ups, downs, failed, probes; double dropped;   // statistics for the log
} nvn_fstep;

// Whether the cost at the longer hold is the cost that held kmin (a probe without a slowed report)
// Whether the cost at the longer hold fits kmin as it is, and the wait after failed probes of this kind is over (FSTEP_ASIS)
static inline int nvn_fstep_asis(const nvn_fstep *s, int kmin) { return s->cn >= FSTEP_RUN && !s->fit_run && s->ccur <= FSTEP_ASIS * kmin && s->since >= s->hold * (1 << (s->afail < 3 ? s->afail : 3)); }
static inline int nvn_fstep_same(const nvn_fstep *s) { return s->cref > 0 && s->cn >= FSTEP_RUN && s->ccur <= FSTEP_SAME * s->cref; }

// Once per frame. shown: real time since the previous frame's start, in vsyncs; cost: the frame's work, max(CPU, GPU), in vsyncs (a GPU that cannot keep up makes frames late too); kmin: 1 at a 60 fps cap, 2 at 30;
// kmax: the longest hold (FSTEP_MAX; 2 at a 30 fps cap, where the display link runs at 30 Hz and holds only even counts); fixed: a step the game has set itself (BotW's
// VFR override; 0 = none), which the host then holds as it is; slow: the CPU runs below its speed for reasons other than heat, so a cost above k − 1 is no evidence
// against kmin (probes). Returns the step for the next frame; s->k is how long that frame stays on screen
static inline int nvn_fstep_next(nvn_fstep *s, double shown, double cost, int kmin, int kmax, int fixed, int slow) {
    const double hold0 = FSTEP_HOLD0 * (s->per_vsync ? s->per_vsync : 1), hold_max = FSTEP_HOLD_MAX * (s->per_vsync ? s->per_vsync : 1);
    if (kmin < 1) kmin = 1;
    if (kmax > FSTEP_MAX) kmax = FSTEP_MAX;
    if (kmax < kmin) kmax = kmin;
    if (!s->hold) s->hold = hold0;
    if (fixed > 0) {   // the game's own step: speed is right by construction, nothing to pay back
        s->k = s->step = fixed < FSTEP_MAX ? fixed : FSTEP_MAX;
        s->debt = 0; s->late_run = s->fit_run = s->trial = s->probe = s->grace = 0; s->run = 0;
        s->kmin = FSTEP_MAX + 1;   // when it ends, k starts again from kmin
        return s->step;
    }
    if (!s->k) { s->k = s->step = s->kmin = kmin; return s->step; }   // first frame: nothing measured yet
    int held = s->k;   // the frame that just ended stayed on screen this long
    s->debt += shown - s->step;
    if (s->debt > FSTEP_DEBT_MAX) { s->dropped += s->debt - FSTEP_DEBT_MAX; s->debt = FSTEP_DEBT_MAX; }
    if (s->debt < -1) s->debt = -1;   // ahead (a frame shorter than its step): nothing slows the game down, so at most one vsync is remembered
    s->since += shown;
    int cap_probe = kmin < s->kmin && s->kmin <= FSTEP_MAX && s->k > kmin;
    if (cap_probe) {   // the cap rose: a probe at once, whose grace covers the clocks still set for the longer hold
        s->k = kmin; s->probes++; s->trial = FSTEP_RUN; s->probe = 1; s->grace = FSTEP_GRACE; s->fit_run = s->late_run = 0; s->pa = s->pb = 0; s->plate = s->ptotal = 0;
    } else if (kmin < s->kmin || s->k < kmin) s->k = kmin;   // the game's own step ended, or the cap fell
    if (s->k > kmax) s->k = kmax;
    s->kmin = kmin;
    int scr = s->screen; s->screen = 0;
    int late = scr ? scr == 2 : shown > held + FSTEP_LATE + s->shown_slack || cost > held + FSTEP_LATE, grace = s->grace > 0;
    int soft = late && !scr && s->gpu > 0 && s->gpu <= FSTEP_FIT * held && cost <= held + FSTEP_LATE_SOFT && shown <= held + FSTEP_LATE_SOFT + s->shown_slack;   // the pipeline's lateness at a hold the GPU fits: no step up
    if (grace && !cap_probe) {   // the completed frame at the old cap does not consume the new hold's grace
        if (s->grace > FSTEP_GRACE / 2) s->pa += cost; else s->pb += cost;
        s->plate += late; s->ptotal++;
        if (!--s->grace && s->ramp && s->plate && s->ptotal + FSTEP_GRACE <= FSTEP_PROBE_MAX && s->pb <= (1 - FSTEP_RAMP) * s->pa) { s->grace = FSTEP_GRACE; s->pa = s->pb = 0; s->plate = 0; }   // still getting cheaper: another window
    }
    int settling = s->settle > 0; if (settling) s->settle--;
    if (late) { s->late++; s->late_run = grace || settling || soft ? 0 : s->late_run + 1; s->fit_run = s->fit_miss = 0; s->late_cmax = s->late_run <= 1 ? (s->late_run ? cost : 0) : fmax(s->late_cmax, cost);
                s->late_tmin = s->late_run <= 1 ? (s->late_run ? fmax(shown, cost) : 0) : fmin(s->late_tmin, fmax(shown, cost)); }
    else { s->late_run = 0; s->late_cmax = s->late_tmin = 0; }
    if (!(s->k > kmin && s->gpu > 0 && s->gpu <= FSTEP_FIT * (s->k - 1))) s->gfit_run = s->gfit_miss = 0;   // the GPU does not fit the shorter hold (or none known): no run
    else if (!late || soft) s->gfit_run++;
    else if (++s->gfit_miss > FSTEP_GFIT_MISS) s->gfit_run = s->gfit_miss = 0;   // a late frame: up to FSTEP_GFIT_MISS of them in a run (Ocarina of Time's Kokiri Forest at hold 3: one in ten)
    else s->gfit_run++;   // (a late frame does not end the run: the GPU fitted it)
    // At kmin, judged by shown and cost: the late frames among the last FSTEP_WIN (a probe's grace and a trial's settling do not count, as for late_run)
    int tolerant = held == kmin && !scr;
    if (tolerant && !grace && !settling) { double t = fmax(shown, cost); s->lwin = (s->lwin << 1 | (uint64_t)late) & ((1ull << FSTEP_WIN) - 1); s->cwin = s->cwin > 0 ? s->cwin + (t - s->cwin) / FSTEP_WIN_N : t; }
    else s->lwin = 0, s->cwin = 0;
    int heavy = s->late_run >= 3 && s->late_tmin >= held + FSTEP_HEAVY, burst = late && __builtin_popcountll(s->lwin) >= FSTEP_WIN_LATE && s->cwin >= held + FSTEP_OVER;
    // The cost of the frames on time (a hitch is not what the scene costs): at kmin, and at the longer hold
    if (kmin != s->ckmin) {   // another cap: its run starts over, and what held it before comes back
        if (s->ckmin >= 1 && s->ckmin <= FSTEP_MAX) {
            // A cap can end a sustained run before any hitch saved its cost. Short runs keep the earlier proven reference.
            if (s->run >= hold_max && s->crun > 0) s->cref = s->crun;
            s->cref_at[s->ckmin] = s->cref;
        }
        s->ckmin = kmin; s->run = s->crun = 0; s->cref = kmin <= FSTEP_MAX ? s->cref_at[kmin] : 0;
    }
    // A frame the screen showed on time while its cost did not fit (absorbed by present snapshots) is not what the hold costs: the probes would take it for the scene that held kmin
    int fits = !scr || cost <= held;
    if (held <= kmin) { s->run += shown; s->cn = 0; if (!late && fits && s->probe != 1) s->crun = s->crun > 0 ? s->crun + (cost - s->crun) / 32 : cost; }
    else {
        s->run = 0;
        if (held != s->ck) { s->ck = held; s->cn = 0; }
        if (!late && fits) { s->ccur = s->cn ? s->ccur + (cost - s->ccur) / 8 : cost; s->cn++; if (s->cn >= FSTEP_RUN && !grace && held <= FSTEP_MAX) s->kheld[held] = s->ccur; }
    }
    if (tolerant && s->late_run && !heavy && !burst && s->k < kmax) s->tolerated++;
    if (s->k < kmax && (tolerant ? heavy || burst : s->late_run >= 3)) {   // step up to the cost, at least one vsync
        s->up_by = !tolerant ? 1 : heavy ? 2 : 3;
        int want = (int)ceil(cost - 0.05);
        if (want < s->k + 1) want = s->k + 1;
        if (want > kmax) want = kmax;
        // A trial back to kmin whose late frames all cost what fits kmin failed on hitches (a load, a stall), not on the scene's cost: the hold-off stays as it was, up to
        // FSTEP_HITCH_MAX such failures in a row. Trials to a longer hold, probes and trials by kheld (Ocarina of Time's 20 -> 30 fps) double it as before
        // A burst at kmin (FSTEP_WIN_LATE late frames, not three heavy ones in a row, nor all in a row) that ends a trial, a probe or a probe's run at kmin counts as one too
        int hitch = held == kmin && s->hitch_fails < FSTEP_HITCH_MAX && ((s->trial && !s->probe && !s->mtrial && !s->gtrial && s->late_cmax <= FSTEP_FIT * held) || (s->up_by == 3 && s->late_run < FSTEP_WIN_LATE && (s->trial || (s->probe == 2 && s->since < hold_max))));
        if (s->atrial && (s->trial || (s->probe == 2 && s->since < hold_max))) s->afail++;
        if (hitch) { s->failed++; s->trial = 0; s->hitch_fails++; }
        else if (s->trial || (s->probe == 2 && s->since < hold_max)) { s->failed++; s->trial = 0; s->hold = s->hold * 2 < hold_max ? s->hold * 2 : hold_max; s->hitch_fails = 0;
            if (s->mtrial && ++s->mfail >= 3 && held <= FSTEP_MAX) s->kheld[held] = 0;   // (three failed trials by kheld: that hold's cost is no longer the scene's)
            if (s->gtrial) s->gfail++; }
        else if (s->probe == 2) { s->hold = hold0; s->afail = 0; }   // the probe's run at kmin lasted
        if (held == kmin && s->run >= hold_max) s->cref = s->crun;   // kmin held 8 s at this cost
        s->k = want; s->ups++; s->late_run = s->fit_run = s->gfit_run = s->gfit_miss = 0; s->since = 0; s->probe = 0; s->mtrial = s->gtrial = s->atrial = 0; s->lwin = 0; s->cwin = 0;
    } else if (s->trial && !grace && !--s->trial) { if (s->probe) { s->probe = 2; s->since = 0; } else s->hold = hold0; s->mtrial = s->mfail = s->gtrial = s->gfail = 0; s->hitch_fails = 0; }   // the trial held (a probe's run at kmin is judged at the next step up)
    if (!late) {
        if (s->k <= kmin) s->fit_run = s->fit_miss = 0;
        else if (cost <= FSTEP_FIT * (s->k - 1)) s->fit_run++;
        else if (++s->fit_miss > FSTEP_MISS) s->fit_run = s->fit_miss = 0;
    }
    if (s->fit_run >= FSTEP_RUN && s->since >= s->hold && !s->trial) { s->k--; s->downs++; s->trial = FSTEP_RUN; s->fit_run = s->fit_miss = s->gfit_run = s->gfit_miss = 0; s->settle = FSTEP_SETTLE; }
    else if (s->k - 1 > kmin && s->k - 1 <= FSTEP_MAX && s->kheld[s->k - 1] > 0 && s->kheld[s->k - 1] <= FSTEP_FIT * (s->k - 1) && s->since >= s->hold && !s->trial && !s->probe) {
        s->k--; s->downs++; s->trial = FSTEP_RUN; s->mtrial = 1; s->grace = FSTEP_GRACE; s->fit_run = s->fit_miss = s->late_run = 0; s->pa = s->pb = 0; s->plate = s->ptotal = 0;   // back to a hold that held
    } else if (s->gfit_run >= FSTEP_RUN && s->k > kmin && s->gfail < 3 && s->since >= s->hold && !s->trial && !s->probe) {   // the GPU has fitted the shorter hold: try it, with a probe's grace
        s->k--; s->downs++; s->trial = FSTEP_RUN; s->gtrial = 1; s->grace = FSTEP_GRACE; s->fit_run = s->fit_miss = s->late_run = s->gfit_run = s->gfit_miss = 0; s->pa = s->pb = 0; s->plate = s->ptotal = 0;
    } else if ((slow || nvn_fstep_same(s) || nvn_fstep_asis(s, kmin)) && s->k > kmin && s->since >= s->hold && !s->trial
               && !(s->probe_near && s->gpu > 0 && s->gpu * s->gpu / s->k > FSTEP_FIT * (s->k - 1 > kmin ? s->k - 1 : kmin))) {   // probe: straight to kmin, so the controller sees the whole deadline
        // probe_near's headroom rule: a governor that sizes the clock to the busy share can raise it at most in proportion to the idle share, so the GPU's time at hold k, gpu, can
        // fall to about gpu * (gpu / k) at best; a probe whose target that does not fit is not made (Ocarina of Time on a Galaxy Z Fold 8: at hold 2 with a GPU of 1.6 vsyncs a
        // probe to 1 failed every time and doubled the hold-off, which then kept hold 3 longer)
        s->atrial = !slow && !nvn_fstep_same(s);
        s->k = s->probe_near && s->k - 1 > kmin ? s->k - 1 : kmin; s->downs++; s->probes++; s->trial = FSTEP_RUN; s->probe = 1; s->grace = FSTEP_GRACE; s->fit_run = s->late_run = 0; s->pa = s->pb = 0; s->plate = s->ptotal = 0;
    }
    s->step = s->k + (s->debt >= 1 && s->k < FSTEP_MAX);
    s->paid += s->step > s->k;
    return s->step;
}

// The GPU clock follows the hold too: Breath of the Wild on an iPhone went from 14.4 to about 25 ms of GPU time a frame at hold 2 with the same 4,900 draws, the CPU still on its
// performance cores, so its cost never fitted hold 1 again. The GPU counts as slowed (another reason for nvn_fstep_next's slow) when its time at a hold above kmin is at least
// FSTEP_GPU_SLOW times its time at kmin and it alone does not fit k − 1 (a GPU at 3 ms of 17, 6 times its time on a loading screen, is not what holds a frame back).
// The reference is not updated during a probe's grace (its frames run while the clock is still rising). The probe's frames after the grace ran at kmin with the clock up:
// they move the reference half-way each, so a scene that is heavy at every hold stops calling the GPU slowed after a failed probe. Tears of the Kingdom on the Y700 kept a
// reference from its title screen (0.1 vsyncs), so its open world (GPU 1.4-1.9 vsyncs at every hold) was "15 times slower" for good: 27 probes in 66 s, each a burst of
// late frames that sent the hold to 3 (20 fps), and every trial back from 3 to 2 that held set the hold-off back to 1 s. It is kept for each kmin: one
// learned at a 30 fps cap is the GPU time at the clock that cap left (lowered for 30 fps, and under the thermal cap for the heat), and taken as hold 1's after the cap rose it
// made the same work at hold 2 look not slowed.
// A reference that does not fit kmin itself (a mean above kmin: the GPU's time at kmin, at the clock kmin raised, is more than the hold, which no pacing sustains) rules the probe out:
// whatever the longer hold added, kmin was not held without it. Ocarina of Time on the Odin 3 (GPU clock at its maximum at every hold) costs 1.2-2.1 vsyncs of GPU time
// in Kokiri Forest by where the camera looks; a failed probe left a reference of 1.3, the heavy views were 1.5 times it, and each turn of the camera started another
// probe: 60-90 late frames at hold 1, then hold 3 (4 probes in 17 s, 358 late frames of 900; t_fstep: the view case)
#define FSTEP_GPU_SLOW 1.3    // GPU time at the longer hold against its time at kmin
#define FSTEP_GPU_N 30        // frames at the longer hold before the comparison counts
typedef struct { double ref, cur; int n, k, kmin; double ref_at[FSTEP_MAX + 1]; } nvn_fgpu;   // GPU time at kmin (running mean; of every kmin in ref_at); at the current hold k and its frame count; the kmin ref belongs to
static inline int nvn_fgpu_slow(const nvn_fgpu *g) { return g->n >= FSTEP_GPU_N && g->ref > 0 && g->ref <= g->kmin && g->cur >= FSTEP_GPU_SLOW * g->ref && g->cur > FSTEP_FIT * (g->k - 1); }
// What nvn_fstep_gpu_slowed's probe takes: 0 no probe's trial, 1 a probe's trial in its grace, 2 a probe's trial after it
static inline int nvn_fstep_probe_phase(const nvn_fstep *s) { return s->probe != 1 ? 0 : s->grace > 0 ? 1 : 2; }
// Once per frame, with the frame's GPU time (vsyncs; 0 = none finished yet), the hold it ran under and the probe phase (nvn_fstep_probe_phase)
static inline int nvn_fstep_gpu_slowed(nvn_fgpu *g, double gpu, int k, int kmin, int probe) {
    if (kmin < 1 || kmin > FSTEP_MAX) return 0;   // no reference slot for it (hle_nvn.c passes 1 or 2)
    g->ref = g->ref_at[kmin]; g->kmin = kmin;
    if (gpu <= 0) return nvn_fgpu_slow(g);
    if (k <= kmin) { if (probe != 1) g->ref = g->ref_at[kmin] = g->ref > 0 ? g->ref + (gpu - g->ref) / (probe == 2 ? 2 : 32) : gpu; g->n = 0; g->k = k; return 0; }
    if (k != g->k) { g->k = k; g->n = 0; }
    g->cur = g->n ? g->cur + (gpu - g->cur) / 8 : gpu; g->n++;
    return nvn_fgpu_slow(g);
}
