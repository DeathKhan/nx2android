# Paper Mario: The Origami King (0100A3900C3E2000): the rewriter's patches of this game's own builds. dm/tools/patch_nso.py loads every profile's patches.py and applies those of a module's NSO build ID;
# CUTS, OBS and FRAME_STEP as described there ("Registered patches"). Patches of Nintendo SDK module builds are in dm/tools/sdk_patches.py
from nso_asm import ADRP, B
CUTS = {
  # Paper Mario: The Origami King 1.0.1 (v65536) main: 60 fps (30 on the Switch). Each frame the renderer hands the window its present interval from the graphics object's field +0x71dc
  # (2) through main+0x1babb0, which calls nvnWindowSetPresentInterval when it changed
  'e74395f066fd8ccb51ec17b39b3da2c8': [
    (0x5beb1c, '626a68b8', '22008052', 'present interval ldr w2,[x19,x8] (field +0x71dc) → mov w2,#1'),
    # Every frame main+0x43ea30 measures the time since the last one (oe::GetProgramTotalActiveTime), rounds it to 60 Hz vsyncs (µs / 16,683.35 + 0.1) and clamps it to [2, cap] (a step
    # above cap + 10 counts as 2); the base clock (time object +0x20, main+0x43e8d0) advances by that step, and the three clocks after it (+0x58, +0x90, +0xc8) follow. The same clock
    # as The Thousand-Year Door's (its patches.py). With the floor at 2 a frame shown every vsync still stepped two vsyncs: Mario walked twice as far in a second at 60 fps as at 30
    (0x43ea9c, 'e9031f32', '29008052', 'frame step floor in 60 Hz vsyncs: mov w9,#2 → mov w9,#1 (the game steps by the vsyncs that passed, down to one)'),
  ],
}

# Paper Mario: The Origami King 1.0.1: main+0x43ea30 stores the clamped step as the base clock's step (time object +0x44, main+0x43eb04). Rounding a frame time between one and two
# vsyncs down to 1 runs the game slow whenever 60 does not hold. The stub stores the host's step instead, the vsyncs the frame stays on screen, so game time = real time at any rate,
# and counts the vsyncs stepped (as The Thousand-Year Door's). Before the host has decided a step, and at target 0, the game's own. x16 and x17 are dead at the site (the bl to the
# clock update follows)
def frame_step(ctx):
    here, emit, fs_site, fs_put, FS_STATE_OFF = ctx.here, ctx.emit, ctx.site, ctx.put, ctx.state
    fs_site(0x43eb04, '604600bd')
    start = here(); emit(ADRP(16, here(), FS_STATE_OFF))
    for w in (0xb9400211,   # ldr w17,[x16]          target fps (float bits; 0 = off)
              0x34000111,   # cbz w17,orig (+8)
              0xb9402211,   # ldr w17,[x16,#0x20]    step in vsyncs
              0x340000d1,   # cbz w17,orig (+6)      not decided yet
              0x1e230220,   # ucvtf s0,w17           the base clock's step
              0xf9401a10):  # ldr x16,[x16,#0x30]
        emit(w)
    emit(0x8b110210)                                                 # add x16,x16,x17
    emit(ADRP(17, here(), FS_STATE_OFF)); emit(0xf9001a30)           # str x16,[x17,#0x30]    vsyncs stepped
    emit(0xbd004660); emit(B(here(), 0x43eb08))                      # orig: str s0,[x19,#0x44]; back
    fs_put(0x43eb04, [B(0x43eb04, start)], 'frame step: the base clock steps by the host\'s vsyncs (state page target fps; 0 = the game\'s measured step)')

FRAME_STEP = {'e74395f066fd8ccb51ec17b39b3da2c8': frame_step}

# How general each patch is and could be (dm/tools/docs_patches.py "Binding", listed in docs/agents/game-patches.md): what, its binding now, the loosest it could have (goal),
# purpose, why; blocked or kept: why the gap stays open (cannot close yet, or left open on purpose); checked: what was checked on game files (else a judgement)
BINDING = [
    {'what': 'present interval field read → 1 (60 fps)', 'now': 'build', 'goal': 'game', 'purpose': 'frame rate',
     'why': "the game's own renderer hands the window its interval from a field of its graphics object; the function could be matched, as The Thousand-Year Door's clock is"},
    {'what': 'frame step floor 2 → 1 vsync', 'now': 'build', 'goal': 'game', 'purpose': 'frame rate',
     'why': "the game's own clock update clamps its step to two vsyncs; the same function as the frame step stub's"},
    {'what': "frame step: the base clock steps by the host's vsyncs (FRAME_STEP)", 'now': 'build', 'goal': 'game', 'purpose': 'frame rate',
     'why': "the game's own clock update stores the measured step; the host's step keeps game time equal to real time at any rate"},
]
