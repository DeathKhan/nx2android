# Football Manager 26 Touch (`0100D4A023450000`)

Update 1.0.2 (v131072), SDK 20.5.14. Sports Interactive, Unity IL2CPP with FMOD on NVN; modules `main`, `subsdk0` (the SDK's movie library) and `sdk`, and `main` loads one NRO. Brought up on Android (Lenovo Legion Y700, 2026-11-06) from the experimental port's survey ([profile_guess.py](../profile_guess.py)), checked against it (the modules and build IDs, the 4 MB system resource of the NPDM, no heap set by `main`); not run on the Mac or an iPhone. The Android keys are in [android.tsv](android.tsv).

## What it needs

| Setting | Where | Why |
|---|---|---|
| The 5 GB block of a game that maps its own memory with no heap: `mem_reserve 0x140000000`, `mem_alias 0x118000000`, `mem_stack 0x10000000`, `app_pool 0xCD500000` | [`game.tsv`](game.tsv) | `main` never sets a heap; the game maps its memory itself in the 4.375 GB alias region |
| `sys_resource 0x400000` | `game.tsv` | the NPDM declares 4 MB of system resource (virtual address memory) |
| `setting docked=0`, `overlay=0`, `rumble=1` | `android.tsv` | played by touch in handheld mode, so the on-screen gamepad starts hidden (the pause menu shows it); a controller works too (the left stick moves the focus; the D-pad does not) |
| `env DM_TOUCH_DOCKED=pixels` | `android.tsv` | docked, its Unity player takes the touch panel's values as pixels of its 1920x1080 screen: a tap at 31% / 55% of the screen pressed the tile at 20% / 36% (reported by the user); given so, taps land where they are (`hle_touch.c`) |
| A Turnip Vulkan driver on Adreno devices (`porter port --vulkan-driver <libvulkan_freedreno.so>`, the GUI's "Vulkan driver" field) | the port, not the profile | see Rendering |

No patches of the game's own builds.

## Rendering

5,747 shaders (5,743 in the Unity asset bundles and the NRO), all translated. One red line at start, `ro::LookupModuleSymbol UnitySetGraphicsDevice`, a symbol the game looks up and does without.

On Qualcomm's own driver (Adreno 830, driver 0800.72), docked, in a friendly in the 3D match view the crowd's arms stretch along the seat rows, different people every frame, in every crowd shader (`33b0d5ca1675654b`, `67a22e46c025a212`, `56dce908e04beff6`: instanced, each person's matrices read from a uniform buffer by instance); with Mesa's Turnip 26.2.6 the same scenes are clean (8 of 8 half-time frames). Not the cause: the bone texture (its image matches the game's memory), uniform shadows (`DM_VK_X_NOSHC`), command-buffer cuts, set reuse, the bind cache, dynamic vec4 component indexing (shaders rewritten to load the whole vec4: still streaks). The root cause in Qualcomm's driver is open.

## Frame rate

30 fps, as the game asks (the window's present interval 2): frame work about 6 ms, GPU about 3 ms, 0 dropped frames per 300.

## Status

Playable on Android, handheld, by touch: the start movie and the welcome page, the main menu over the 3D locker room, a quick-start career (team selection, game setup, a new manager: 3D presets, names through the overlay keyboard, lists), the database load, the 3D press room, the portal and its messages, a save (save commits in the log), then a friendly in the 3D match view docked. 0 audio underruns. Not tested: days advancing past the first match.
