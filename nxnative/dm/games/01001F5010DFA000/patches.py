# Pokémon Legends: Arceus (01001F5010DFA000): the rewriter's patches of this game's own builds. dm/tools/patch_nso.py loads every profile's patches.py; MATCH_CUTS as described there
# ("Registered patches"). Patches of Nintendo SDK module builds are in dm/tools/sdk_patches.py
import struct
from nso_asm import ADRP, B

# 60 fps (30 on the Switch) on 1.1.1 (v262144) main, with game time at real time. Two things hold the game at 30, and the stubs move both to the host's frame step:
# - The present interval. The start-up code builds the display's settings on the stack (main+0x277d7xx) and hands them to the display setup (main+0x29d6a64 → main+0x2aa24c4),
#   which sets the window's present interval to that word + 1 (main+0x2aa2788). The word is 1: interval 2. The first stub makes it 0 (interval 1) when the state page's target is
#   not 0, read once at start-up (the loader writes the target before the game runs); at target 0 it stays 1, the Switch's 30
# - The frame period. The game steps its time by a stored period, not by the clock: the frame-time object's constructor (main+0x277bda4) stores 33,333,334 ns in four {0, ns} pairs
#   (object +0x78 to +0xb0), and its interface at +0x70 (vtable main+0x4171538) reads and sets them: [0] main+0x278303c returns the address of pair idx, [1] main+0x2783048 sets
#   a pair from the table {16,666,667, 33,333,334} at main+0x397e8d0, [2] main+0x277bef4 sets a pair's ns. With interval 1 alone a frame shown every vsync still stepped 1/30 s and
#   the game ran at double speed (Odin 3, Adreno 830, 2026-11-10; with the period at 16,666,667 ns, 60 fps ran at normal speed and 30 fps at half). The stub in the getter stores
#   the host's step (the vsyncs the frame stays on screen, state page +0x20) times 16,666,667 ns in the pair asked for, before the caller reads it: 60 fps steps 1/60 s, 30 fps
#   (target 30, heat, a movie, late frames) 1/30 s. The getter is in this vtable only (no other reference in main). Before the host has decided a step, and at target 0, the
#   stored period stays
# The third stub counts the vsyncs stepped where the scene update reads the period (main+0x279a6f4), for the host's game time log. x16 and x17: unused from main+0x277d838 to the
# bl at main+0x277d84c; dead in a function entered through blr (the getter); x8 at main+0x279a6f4 is set by the next instruction
def frame_step(ctx):
    here, emit, fs_site, fs_put, FS_STATE_OFF = ctx.here, ctx.emit, ctx.site, ctx.put, ctx.state
    fs_site(0x277d838, '280080d2')
    start = here(); emit(ADRP(16, here(), FS_STATE_OFF))
    for w in (0xb9400211,   # ldr w17,[x16]          target fps (float bits; 0 = off)
              0xd2800028,   # mov x8,#1              (original) interval 2
              0x34000051,   # cbz w17,back (+2)
              0xd2800008):  # mov x8,#0              interval 1
        emit(w)
    emit(B(here(), 0x277d83c))                                       # back: movk x8,#2,lsl #48
    fs_put(0x277d838, [B(0x277d838, start)], 'frame step: display settings, present interval 1 with a target fps (state page; 0 = the game\'s interval 2)')
    fs_site(0x278303c, '08d0218b00210091c0035fd6')
    start = here(); emit(0x8b21d008)                                 # add x8,x0,w1,sxtw #4   (original)
    emit(ADRP(16, here(), FS_STATE_OFF))
    for w in (0xb9400211,   # ldr w17,[x16]          target fps
              0x340000f1,   # cbz w17,back (+7)
              0xb9402211,   # ldr w17,[x16,#0x20]    step in vsyncs
              0x340000b1,   # cbz w17,back (+5)      not decided yet
              0x528a0570,   # mov w16,#0x502b
              0x72a01fd0,   # movk w16,#0xfe,lsl #16 16,666,667 ns
              0x9bb07e31,   # umull x17,w17,w16
              0xf9000911):  # str x17,[x8,#0x10]     the pair's ns
        emit(w)
    emit(B(here(), 0x2783040))                                       # back: add x0,x8,#8; ret
    fs_put(0x278303c, [B(0x278303c, start)], 'frame step: the frame period read is the host\'s vsyncs × 16,666,667 ns (state page target fps; 0 = the stored period)')
    fs_site(0x279a6f4, '000440fd')
    start = here(); emit(ADRP(16, here(), FS_STATE_OFF))
    for w in (0xb9400211,   # ldr w17,[x16]
              0x340000d1,   # cbz w17,orig (+6)
              0xb9402211,   # ldr w17,[x16,#0x20]
              0x34000091,   # cbz w17,orig (+4)
              0xf9401a08,   # ldr x8,[x16,#0x30]
              0x8b110108,   # add x8,x8,x17
              0xf9001a08):  # str x8,[x16,#0x30]     vsyncs stepped
        emit(w)
    emit(0xfd400400); emit(B(here(), 0x279a6f8))                     # orig: ldr d0,[x0,#8]; back
    fs_put(0x279a6f4, [B(0x279a6f4, start)], 'frame step: counts the vsyncs the scene update steps')
    # Code that does not read the frame period: it builds 1/30 s in a register (mov wN,#0x8889; movk wN,#0x3d08,lsl #16) and steps by it once a frame. At 60 fps that code ran at
    # double speed while the rest of the game ran at real time: the Poké Ball's throw and the capture sequence played too fast (Odin 3, 2026-11-13). Each site listed loads the host's
    # step in seconds instead (state page +0x2c, hle.h), and the game's 1/30 while that is 0. The stub uses the site's own register only (no flags: main+0x2795230 sits between a cmp
    # and its fcsel). Sites: timers that add 1/30 a frame; velocity × 1/30 and the step handed to update functions in the thrown-object code (main+0x198xxxx to main+0x1a1xxxx,
    # main+0x1b4xxxx, main+0x257xxxx) and its helpers (main+0x17177b4, main+0x1728054); the function that returns 1/30 (main+0x1983760); the update that takes 1/30 unless its
    # object's flag at +0x150 is set (main+0x2795230). Left alone: frames ↔ seconds conversions (main+0x8eedb4, main+0x191d2b8) and the defaults stored into objects
    # (main+0x2f3e8c4, main+0x3002a04, main+0x301439c, main+0x3014d24), which are not known to be a frame's step. The 1/60 constants (4 sites) are not a frame's step on the Switch
    for off in STEP_30:
        d = 22 if off == 0x1a0f498 else 8; mov, movk = 0x52911120 | d, 0x72a7a100 | d   # mov wD,#0x8889; movk wD,#0x3d08,lsl #16
        fs_site(off, struct.pack('<II', mov, movk).hex())
        start = here(); emit(ADRP(d, here(), FS_STATE_OFF))
        for w in (0xb9402c00 | d << 5 | d,   # ldr wD,[xD,#0x2c]    the host's step in seconds (float bits)
                  0x35000060 | d,            # cbnz wD,back (+3)
                  mov, movk):                # none: the game's 1/30
            emit(w)
        emit(B(here(), off + 8))
        fs_put(off, [B(off, start)], 'frame step: a constant 1/30 s step is the host\'s step in seconds (state page +0x2c; 0 = the constant)')
    # The same step with a sign or a factor (found 2026-11-13 after the throw and the evolution still ran at double speed with the sites above): −1/30 (movk #0xbd08: gravity × −1/30
    # added to a thrown object's vertical speed at main+0x1984a70, main+0x198da8c, main+0x1a158c8, main+0xafbbc4 and main+0x25ac3d8; count-down timers at the others), ±1/15
    # (#0x3d88, #0xbd88: a value moved towards its goal by k/15 a frame, main+0x1e5c5xx) and 1/60 (#0x3c88: a timer that adds 1/60 a frame at main+0x1d49838). The host's step's
    # float bits get the sign bit (eor) or one exponent step (add / sub 0x00800000: × 2, × 0.5); no flags
    for off, d, hi in STEP_30K:
        mov, movk = 0x52911120 | d, 0x72a00000 | hi << 5 | d
        ops = ([0x11600000 | d << 5 | d] if hi & 0x7fff == 0x3d88 else [0x51600000 | d << 5 | d] if hi & 0x7fff == 0x3c88 else []) + ([0x52010000 | d << 5 | d] if hi & 0x8000 else [])
        fs_site(off, struct.pack('<II', mov, movk).hex())
        start = here(); emit(ADRP(d, here(), FS_STATE_OFF))
        emit(0xb9402c00 | d << 5 | d)                # ldr wD,[xD,#0x2c]
        emit(0x34000000 | (len(ops) + 2) << 5 | d)   # cbz wD,const
        for w in ops: emit(w)                        # add / sub wD,wD,#0x800,lsl #12; eor wD,wD,#0x80000000
        emit(B(here(), off + 8))
        emit(mov); emit(movk); emit(B(here(), off + 8))   # const: the game's value
        fs_put(off, [B(off, start)], 'frame step: a constant %s1/%d s step is the host\'s step in seconds with that sign and factor (state page +0x2c; 0 = the constant)'
               % ('−' if hi & 0x8000 else '', {0x3d08: 30, 0x3d88: 15, 0x3c88: 60}[hi & 0x7fff]))
    # Sequences that count frames. The game counts 30 fps frames in thousands of places (4,881 integer counters and 415 float ones in 1.1.1's main), and two players of authored
    # sequences step one frame an update whatever the period: the battle effect sequence player (vtable main+0x40d3340; its advance at main+0x9dc3e0 adds the step at +0xf0, a
    # constant 1.0, to the frame at +0xe8, and its commands compare whole frames) and the demo system (evolution, capture and event demos; its timeline is not found). Both ran at
    # double speed at 60 fps. While one plays, its stub writes 1 to the state page +0x38 every update, and the host holds 30 fps while marks keep arriving (hle_nvn.c
    # guest_hold30), the rate the sequences were authored for. The demo mark is in the demo manager's "finished?" poll (main+0x148ff9c, called every update by the five tasks
    # that wait for a demo: main+0x115e138, 0x1290820, 0x13cf5c4, 0x13e0ec8, 0x148d854) while the manager's state at +0x6d8 is 5 or more (started).
    # x8 at main+0x9dc3e0 is dead (the function returns without reading it); x9 at main+0x148ffb8 is set before its next read; the flags there are set again by the cmp at main+0x148ffc0
    fs_site(0x9dc3e0, '62f240bd')
    start = here(); emit(0xbd40f262)                                 # ldr s2,[x19,#0xf0]     (original)
    emit(ADRP(16, here(), FS_STATE_OFF))
    emit(0x52800028); emit(0xb9003a08)                               # mov w8,#1; str w8,[x16,#0x38]
    emit(B(here(), 0x9dc3e4))
    fs_put(0x9dc3e0, [B(0x9dc3e0, start)], 'frame step: a battle effect sequence asks the host for 30 fps while it plays (state page +0x38)')
    fs_site(0x148ffb8, '08d846b9')
    start = here(); emit(0xb946d808)                                 # ldr w8,[x0,#0x6d8]     (original)
    emit(0x7100151f)                                                 # cmp w8,#5
    emit(0x5400008b)                                                 # b.lt back (+4)
    emit(ADRP(16, here(), FS_STATE_OFF))
    emit(0x52800029); emit(0xb9003a09)                               # mov w9,#1; str w9,[x16,#0x38]
    emit(B(here(), 0x148ffbc))
    fs_put(0x148ffb8, [B(0x148ffb8, start)], 'frame step: a demo (evolution, capture, event) asks the host for 30 fps while it plays (state page +0x38)')

STEP_30 = (0x13b339c, 0x167ff0c, 0x16832c0, 0x16832e4, 0x16836d0, 0x17177b4, 0x1728054, 0x1982930, 0x1983760, 0x198e020, 0x198ec6c, 0x1a0baf0, 0x1a0bee4, 0x1a0da20, 0x1a0eea0,
           0x1a0f304, 0x1a0f498, 0x1a0f55c, 0x1a17b10, 0x1a17f3c, 0x1a19104, 0x1a196c4, 0x1b48a48, 0x1b48a94, 0x1b49004, 0x1b493f4, 0x1d1b7cc, 0x1d5d684, 0x257ac30, 0x257c9f4,
           0x257ce38, 0x2795230, 0x29faf8c, 0x2af25f4)

STEP_30K = ((0xafbbc4, 8, 0xbd08), (0x1984a70, 8, 0xbd08), (0x198da8c, 8, 0xbd08), (0x1a0b3ac, 8, 0xbd08), (0x1a158c8, 8, 0xbd08), (0x1aab5a4, 8, 0xbd08), (0x1aab5d4, 8, 0xbd08),
            (0x1b47f5c, 8, 0xbd08), (0x1b47f94, 8, 0xbd08), (0x1b47fcc, 8, 0xbd08), (0x25ac3d8, 8, 0xbd08), (0x1e5c530, 9, 0xbd88), (0x1e5c588, 8, 0x3d88), (0x1e5c624, 9, 0x3d88),
            (0x1d49838, 8, 0x3c88))

FRAME_STEP = {'aee8f150dda1b5a838806e1a5ea6827a': frame_step}

# A workaround in the game's own code; the cause is open (the note's "Crash in the capture demo", GitHub issue #712). main+0x16e33ec reads a weak reference at this+0x88, locks it and
# then reads [result + 0x118] without checking the result. The capture demo task calls it once per ball (main+0x19d2368 from main+0x1ccf1a4). Null there was a data abort at
# main+0x16e34fc after 37 to 65 minutes of play. The reference is null from the object's constructor until the field setup's state 0x18 (main+0x1877a4c through main+0x16dde90), and the
# lock returns null for a dead object; which of the two the crashes are is unknown. The cuts return the function's empty result ({0, 0} through x20) on both, before it takes any reference.
# All 12 direct reads of the result (the ball's +0x810) null-check it.
#   main+0x16e33ec, 2672 bytes, in 1.1.1. The room is two of the function's five identical blocks (mov w23, wzr; b main+0x16e3a14) at main+0x16e3da4, whose two branches move to the
#   block at main+0x16e3d88. Taking the function's abort stub instead, as the issue's first form did, moves its callers to a stub in the next function, which a function match cannot hold.
capture = 'capture demo: the empty result when the weak reference at +0x88 is unset or dead'
MATCH_CUTS = {'main': [
    {'name': capture, 'size': 2672, 'fingerprints': ['123f93c4fa74d68f6100dab6ec8310a5047b2b3535b72361b6e658e3e48d792e'],
     'cuts': [
         # stur xzr, [x29, #-0x60]  →  b the empty result
         (48, 'bf031af8', '64020014', capture + ': no reference at +0x88 (as 1.1.1 main+0x16e341c)'),
         # b main+0x16e3420  →  b the null check
         (1084, 'fefeff17', '5f010014', capture + ': after the lock (as 1.1.1 main+0x16e3828)'),
         # b.ne to the freed blocks  →  b.ne to the identical block at main+0x16e3d88
         (1480, '811f0054', 'a11e0054', capture + ': frees a duplicate block (as 1.1.1 main+0x16e39b4)'),
         (1552, '811d0054', '611c0054', capture + ': frees a duplicate block (as 1.1.1 main+0x16e39fc)'),
         # the null check: ldur x8, [x29, #-0x60]; cbnz x8, main+0x16e3420. The empty result: stp xzr, xzr, [x20]; b the epilogue at main+0x16e3cd0
         (2488, 'f7031f2a1bffff17f7031f2a19ffff17', 'a8035af8c8b3ffb59f7e00a9c8ffff17', capture + ': the null check and the empty result (as 1.1.1 main+0x16e3da4)'),
     ]},
]}

# How general each patch is and could be (dm/tools/docs_patches.py "Binding", listed in docs/agents/game-patches.md): what, its binding now, the loosest it could have (goal),
# purpose, why; blocked or kept: why the gap stays open (cannot close yet, or left open on purpose); checked: what was checked on game files (else a judgement)
BINDING = [
    {'what': 'capture demo: the empty result when the weak reference at +0x88 is unset or dead', 'now': 'game', 'goal': 'game', 'purpose': 'runtime',
     'why': "the game's own function dereferences its own weak reference; no host service is on the path, so no host or engine rule can hold it. Whether another Game Freak game shares the function is unchecked",
     'checked': "function-matched on 1.1.1, the only build on hand: exactly one match, a boot to the field setup, and the patched function run on the module's own bytes in a small interpreter for both null paths. Issue #712's reporters ran the earlier cuts for about 7 hours with no crash; the path itself has not been reached with these cuts"},
    {'what': 'present interval 2 → 1 with a target fps, and the frame period from the host\'s frame step', 'now': 'build', 'goal': 'game', 'purpose': 'frame rate',
     'why': "the game's own display settings word and frame-time object; MATCH_FRAME_STEP could carry the three stubs (the start-up function, the period getter and the scene update), not tried: 1.1.1 is the only build on hand"},
]
