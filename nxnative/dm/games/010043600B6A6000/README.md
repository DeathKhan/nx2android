# South Park™: The Stick Of Truth™ (`010043600B6A6000`)

Base game 1.0.0 (v0, EUR), SDK 4.5.0, the oldest SDK the Android host runs. Obsidian's engine on NVN only (`main` imports neither Vulkan nor OpenGL); modules `main` and `sdk`. `main` carries the game's own GLSLC compiler, so its shaders are whole GLSLC outputs. The 183 Bink movies (3.68 GB of the 5.46 GB RomFS) are decoded by the game itself. No profile existed anywhere, so this one started as the survey's provisional profile ([profile_guess.py](../profile_guess.py)) and was checked by hand on 2026-11-05; brought up on Android (Xiaomi Pad 5), not on the Mac or an iPhone. The Android keys are in [android.tsv](android.tsv).

## What it needs

| Setting | Where | Why |
|---|---|---|
| 8 GB of reserve with a 4 GB alias region: `mem_reserve 0x200000000`, `mem_alias 0x100000000`, `mem_stack 0x10000000`, `app_pool 0xCD500000` | [`game.tsv`](game.tsv) | the game asks for the heap itself at `main+0x1b66160`: 0xC9400000 = its application pool − 64 MB of used memory − 1 MB, so the heap region must hold the whole pool. The guessed 5 GB reserve left 0x8000000 of heap region and the game aborted in `nninitStartup` (`SetHeapSize OOM`); this split gives it 0xE0000000 (3.5 GB). Written by hand from the request (the Mac host's heap check derives it) |
| `sys_resource 0x1000000` | `game.tsv` | the NPDM declares 16 MB of system resource |
| The named `nn::vi::OpenDisplay` overload hooked | core (`hle_vi`), found on this game | it opens the display as "Default" through `OpenDisplay(Display**, const char*)`, which no hook covered: the SDK's own version dereferenced a null vi service early in `nnMain` |
| A Turnip Vulkan driver on the test device (`--vulkan-driver vulkan.ad07xx.so`, Mesa 26.0.0) | the port, not the profile | the host must begin passes with dynamic rendering (Vulkan 1.3) or imageless framebuffers with render pass 2; Qualcomm's 512.502.0 driver on the Adreno 640 has neither (`vk_init` dies with "passes cannot be begun"). A device with a Vulkan 1.3 driver needs no packing |

No patches of the game's own builds.

## Rendering

351 shaders, whole GLSLC outputs in `main` (294 output magics), which `shader_split.py` takes as they are with no Mac-host capture step. 92 pipelines over 24,600 presents, about 200 draws a frame.

## Frame rate

30 fps solid, the game's own cap: frame work p50 33.4 ms of the 33.3 ms budget, GPU p50 18.8 ms, replay 6.0 ms.

## Status

Playable on Android (Xiaomi Pad 5, Snapdragon 860 / Adreno 640, Android 13, 2026-11-05), by hand and with adb taps: the "Press A to Begin" title, the opening Bink movie (audio through the SDK's audout, 0 underruns), the whole character creation (5 steps), both lessons of the tutorial in the New Kid's bedroom (walked with the on-screen gamepad's left stick), the game's own menu (character sheet, inventory) and the host's pause menu, and a save (the game mounted `SPSOT_SAVE`, the host committed it atomically, porter saves export brought both files off the device, 709 KB). Audio clean, no red log line. Not tested: past the opening tutorial (the game's first real scene), a controller.
