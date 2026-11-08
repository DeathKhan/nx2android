# Donkey Kong Bananza (040074A01BF12000): the rewriter's patches of this game's builds (dm/tools/patch_nso.py "Registered patches"), loaded by build ID
OBS = {
  # v458752 main: the frame-rate controller (main+0x204a18, called once a frame from main+0x203718 with x0 = its settings, x1 = the presenter) has just loaded the frame's
  # measured time, presenter +0x128 in ticks of the 31.25 MHz system counter (main+0x204a40 ldr x8, [x1, #0x128]; frame start to the later of submission and the GPU's end).
  # It then wants interval floor(time in vsyncs - 0.2) + 1, clamped to 1..3, steps up after 2 such frames and down after 6, and sets it through nvnWindowSetPresentInterval
  # (main+0x204c20); the game steps by the interval it chose. The gate replaces the next instruction (cmp w9, #0, position-independent) and lets the pause menu's frame cap
  # lengthen x8 to the capped period, so the controller picks the capped interval itself (hle_framecap.c). With the cap off it changes nothing
  'e25c90c1c136638885659fd55bf8a03b': [
    (0x204a44, 28 | 8 << 8, '7100013f', 'frame-rate controller: its frame time in x8 (ticks), lengthened to the frame cap (OBS_FRAME_CAP)'),
    # the renderer's settings (main+0x4c5228 makes each parameter object, its value a float at +0x64, and stores it in the settings with str x20, [x19, #off]): the gates on four of
    # those stores hand the object in x20 to the pause menu's Resolution level (hle_framecap.c; bits 8..11 which setting, bits 12..16 the register)
    (0x4ca58c, 29 | (1 | 20 << 4) << 8, 'f9029e74', 'renderer setting MaintainGpuPercentage (87.5): its object in x20 (OBS_RES_PARAM)'),
    (0x4ca610, 29 | (2 | 20 << 4) << 8, 'f902a274', 'renderer setting FrameBufferMinScale (0.6): its object in x20 (OBS_RES_PARAM)'),
    (0x4ca8b8, 29 | (3 | 20 << 4) << 8, 'f902e674', 'renderer setting NdsrUpscaleMinRate (1.375): its object in x20 (OBS_RES_PARAM)'),
    (0x4ca93c, 29 | (4 | 20 << 4) << 8, 'f902ea74', 'renderer setting NdsrUpscaleMaxRate (1.9): its object in x20 (OBS_RES_PARAM)'),
  ],
}

# How general each entry is (dm/tools/docs_patches.py "Binding")
BINDING = [
    {'what': 'OBS_FRAME_CAP gate at the frame-rate controller', 'now': 'build', 'goal': 'build', 'purpose': 'runtime',
     'why': 'the controller\'s address is this build\'s own; the gate changes the frame time it reads only while the pause menu\'s frame cap is on'},
    {'what': 'OBS_RES_PARAM gates at the renderer settings', 'now': 'build', 'goal': 'build', 'purpose': 'runtime',
     'why': 'the settings constructor is this build\'s own; the gates only report the objects; the pause menu\'s Resolution level writes the GPU target and the upscale floor'},
]
