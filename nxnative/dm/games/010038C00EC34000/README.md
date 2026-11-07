# UNDER NIGHT IN-BIRTH Exe:Late[cl-r] (`010038C00EC34000`)

Update 1.0.3 (v131072), the Aksys release, with its DLC (a licence only, no content), SDK 6.4.2. French-Bread's own 2D fighting game engine over NVN; modules `main`, `subsdk0` (the SDK's movie library, for the opening movie) and `sdk`. Brought up on Android (Lenovo Legion Y700, 2026-10-31) from the survey's provisional profile ([profile_guess.py](../profile_guess.py)) after a player reported another release of it working (GitHub issue #23), together with [UNDER NIGHT IN-BIRTH II Sys:Celes](../0100BD401D18A000/README.md); not run on the Mac or an iPhone. The Android keys are in [android.tsv](android.tsv).

## What it needs

| Setting | Where | Why |
|---|---|---|
| A 2816 MB heap region: `mem_reserve 0xe0000000`, `mem_alias 0x10000000`, `mem_stack 0x10000000`, `app_pool 0xCD500000` | [`game.tsv`](game.tsv) | `main` sets a heap of at most the pool − 64 MB − 589 MB of bss (counted as used memory) |
| `sys_resource 0x0` | `game.tsv` | the NPDM declares 0 MB of system resource: `IsVirtualAddressMemoryEnabled` is false, as on Horizon (as [Big Brain Academy: Brain vs. Brain](../0100620012D6E000/README.md)) |

Every key is the provisional profile's guess, unchanged; nothing in the core changed for it, and there are no patches of its builds.

## Rendering

36 shaders, all translated.

## Frame rate

60 fps in battle: frame work p50 about 2-4 ms, GPU under 1 ms a frame. Frames dropped on screen: about 275 in the first 5 seconds of each launch, while the game loads, then about 1 in 300 (42 of 24,554 in a 7-minute session).

## Status

Playable on Android, driven with DM_CTL: the new save and replay data prompts, the publisher logos, the title, the main menu, Arcade (side select, character select, the story scene, stage 1 against the CPU to its end and back to the main menu), the pause menu; after a restart the game opens without the new save prompt (its save is kept). No audio underrun, no red log line. The release in issue #23, `0100D3200C52A000` 1.0.3 (reported on a POCO F6, Snapdragon 8s Gen 3), was not tested and has no profile. Not tested: Chronicles, Network (online play), handheld mode (played docked).
