# UNDER NIGHT IN-BIRTH II Sys:Celes (`0100BD401D18A000`)

Update 1.42 (v1179648; the NACP names it "… Sys:Celes EU") with its six DLC (licences only, no content), SDK 15.3.2. French-Bread's own 2D fighting game engine over NVN; modules `main`, `subsdk0` (the SDK's movie library, for the opening movie) and `sdk`. Brought up on Android (Lenovo Legion Y700, 2026-10-31) from the survey's provisional profile ([profile_guess.py](../profile_guess.py)) after a player reported it working on a POCO F6 with update 1.0.1 and all DLC (GitHub issue #21), together with [UNDER NIGHT IN-BIRTH Exe:Late[cl-r]](../010038C00EC34000/README.md); not run on the Mac or an iPhone. The Android keys are in [android.tsv](android.tsv).

## What it needs

| Setting | Where | Why |
|---|---|---|
| A 2560 MB heap region: `mem_reserve 0xd0000000`, `mem_alias 0x10000000`, `mem_stack 0x10000000`, `app_pool 0xCD500000` | [`game.tsv`](game.tsv) | `main` sets a heap of at most the pool − 64 MB − 694 MB of bss (counted as used memory) |
| `sys_resource 0x0` | `game.tsv` | the NPDM declares 0 MB of system resource: `IsVirtualAddressMemoryEnabled` is false, as on Horizon (as [Big Brain Academy: Brain vs. Brain](../0100620012D6E000/README.md)) |

Every key is the provisional profile's guess, unchanged; nothing in the core changed for it, and there are no patches of its builds.

## Rendering

81 shaders, all translated. The opening movie is H.264 through MediaCodec.

## Frame rate

60 fps in battle: frame work p50 about 5 ms, GPU about 1.9 ms a frame; 35 of 36,553 frames dropped on screen in a 10-minute session.

## Status

Playable on Android, driven with DM_CTL: the new save and replay data prompts, the opening movie, the title, the main menu, Versus > Vs. CPU, the game settings, side select, character select (the full roster with the DLC characters), two matches against the CPU to the result screen and a rematch, the pause menu; after a restart the game opens without the new save prompt (its save is kept). No audio underrun, no red log line. Not tested: Single Player's story, Network (online play), handheld mode (played docked).
