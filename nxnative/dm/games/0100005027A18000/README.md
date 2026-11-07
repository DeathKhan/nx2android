# Graveyard Keeper 2 (`0100005027A18000`)

Update 1.0.4 (v262144, shown in game as 1.007.4), SDK 21.4.4. Unity 6000.3.9f1 IL2CPP with FMOD on NVN; `subsdk0` is the SDK's movie library (its `sdk` and `subsdk0` are Viractal's builds). Brought up on Android (Lenovo Legion Y700, 2026-10-26) from the NSP script's provisional profile ([profile_guess.py](../profile_guess.py)), checked by hand against the survey; not run on the Mac or an iPhone. The Android keys are in [android.tsv](android.tsv).

## What it needs

| Setting | Where | Why |
|---|---|---|
| The 5 GB block of a game that maps its own memory with no heap (as Viractal and Hollow Knight: Silksong), `sys_resource 0x1000000` | [`game.tsv`](game.tsv) | the NPDM declares 16 MB of system resource; `main` never sets a heap |
| Node area cut and size-class alignment on its `sdk` build `14839413…`, `mem_map_limit` | `dm/tools/sdk_patches.py`, `game.tsv` | 2026-10-28: the game aborted 24 minutes into play with Unity's allocator out of address space; with the cut, 34 minutes then 1 h 54 min at most 2.44 GB of the 3.21 GB pool mapped |
| `fps_target 60` with `present_interval display` | `game.tsv` | 60 fps (30 on the Switch): Unity 6's vsync count and frame time are written by `hle_fpslock.c`; frame p50 16.6 ms, p95 18.2 ms, no dropped frame after the load, the game's speed right at 60 |

## Rendering

13,986 shaders from its Addressables bundles, all translated (the tables were split on Windows and split again with `/` paths). Screen recordings found no flickering frame, standing or walking.

## Status

Playable on Android: boot, the title, a new game and the opening in the Inquisition Dungeon driven with DM_CTL, a 15-minute soak at 30 fps (frame p50 33.3 ms, p95 35 ms, no dropped frames, replay 3-6 ms, GPU 7-11 ms, footprint steady at 1.8 GB, no audio underruns, no red log line), then play by hand from a save at 60 fps (footprint 1.27 GB). Not checked: the areas past the dungeon, handheld mode (played docked).
