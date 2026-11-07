# FM24 Touch (`01008B101B87A000`)

Update 1.0.1 (v65536), SDK 16. Sports Interactive's own engine on OpenGL ES 3.1 through NVIDIA's GL driver; modules `main` and `sdk` (`subsdk0`, NintendoSDK_OpenGL, is not loaded: the Android host hands the GL calls to the device's own EGL and GLES driver, `dm/android/gl_android.c`, with BC textures decoded on the CPU and frames shown through the Vulkan present path). The first game run by that OpenGL ES layer. Brought up on Android (Lenovo Legion Y700, 2026-11-06) from the experimental port's survey ([profile_guess.py](../profile_guess.py)), whose memory values held on the device; not run on the Mac or an iPhone. The Android keys are in [android.tsv](android.tsv).

## What it needs

| Setting | Where | Why |
|---|---|---|
| A 3328 MB heap region: `mem_reserve 0x100000000`, `mem_alias 0x10000000`, `mem_stack 0x10000000`, `app_pool 0xCD500000` | [`game.tsv`](game.tsv) | `main` sets a heap of at most the pool − 64 MB − 48 MB of bss (counted as used memory) |
| `sys_resource 0x0` | `game.tsv` | the NPDM declares 0 MB of system resource: `IsVirtualAddressMemoryEnabled` is false, as on Horizon |
| `mem_used 0x7f6c000` | `game.tsv` | the rewritten modules are larger than the default 64 MB of used memory |
| `setting docked=0`, `overlay=0`, `rumble=1` | `android.tsv` | played handheld (1280x720) by touch, with the on-screen gamepad hidden; a controller works too (ZR continues, the left stick moves the section focus) |
| The audio renderer's event replaced after the game let it go | core (`hle_audren.c` `ev_check`), found on this game | it aborted at its first audio render until the host replaced the renderer's event that the SDK keeps signalling |

No patches of the game's own builds.

## Rendering

1,224 shaders, GLSL ES text in the RomFS (`data/sigraphics/shaders/gles`) compiled at run time, so the split shader tables are empty and a GL game no longer logs the missing shader bundle as red.

## Frame rate

60 fps in menus and the 3D match view.

## Status

Playable on Android: the title, the controller tutorial and preferences, the main menu, a new career (the 3D head-coach editor, names through the overlay keyboard), the tactics tutorial and a match in the 3D match view (players, shadows, stadium, highlights). 0 audio underruns, no red log line; the user played it and found it playable. Not checked: a save, days past the first match, docked mode.
