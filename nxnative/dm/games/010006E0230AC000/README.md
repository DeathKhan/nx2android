# Tales of Eternia Remastered (`010006E0230AC000`)

Update 2.0.0 (v65536), SDK 21.4.4. Unity 6000.0.66f2 IL2CPP with CRIWARE (ADX2 sound, Sofdec2 movies through the SDK's movie library) on NVN; modules `main`, `subsdk0` (the SDK's movie library) and `sdk`, the same `sdk` and `subsdk0` builds as [Viractal](../0100FD501F95A000/README.md) and [Graveyard Keeper 2](../0100005027A18000/README.md). Brought up on Android (Lenovo Legion Y700, 2026-10-27) from the NSP script's provisional profile ([profile_guess.py](../profile_guess.py)), checked by hand against the survey; not run on the Mac or an iPhone. The Android keys are in [android.tsv](android.tsv).

## What it needs

| Setting | Where | Why |
|---|---|---|
| The 5 GB block of a game that maps its own memory with no heap: `mem_reserve 0x140000000`, `mem_alias 0x118000000`, `mem_stack 0x10000000`, `app_pool 0xCD500000` | [`game.tsv`](game.tsv) | `main` never sets a heap; the game maps its memory itself in the 4.375 GB alias region |
| `sys_resource 0x10000000` | `game.tsv` | the NPDM declares 256 MB of system resource (virtual address memory) |
| Node area cut on its `sdk` build `14839413…` | `dm/tools/sdk_patches.py` | the SDK 21.4.4 build shared with Graveyard Keeper 2 and Viractal: without the cut Unity 6's regions start after the 1 GB node area (found on Graveyard Keeper 2) |
| AAudio callbacks of 5 ms | `dm/android/compat/audiotoolbox_aaudio.c` (core, not a profile key) | its CRI output keeps only about 2 blocks queued, so callbacks of up to 960 frames underran: 11% of the sound was zero padding and the opening movie, paced by the sound, ran at 25 ms a frame; with 5 ms callbacks 0 underruns and 16.7 ms |

No patches of the game's own builds.

## Rendering

15,983 shaders from its bundles, all translated (the tables were split on Windows and split again with `/` paths). The one red line at boot, `ro::LookupModuleSymbol UnitySetGraphicsDevice`, is Unity looking for an optional entry point in its native plugin NRO.

## Frame rate

60 fps held, as on the Switch: frame p50 16.6 ms, p95 about 18 ms, GPU about 6 ms a frame.

## Status

Playable on Android: boot, the opening movies, the title and play by hand by the user, who reported no issue. Footprint about 1.1 GB, no audio underruns over 100,000 device callbacks, no new red log line. Not checked: how far into the game it was played, handheld mode.
