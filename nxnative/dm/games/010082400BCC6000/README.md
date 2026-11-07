# Untitled Goose Game (`010082400BCC6000`)

Update 1.1.2 (v458752), SDK 8.2.0. Unity 2018.4.1f1 IL2CPP with FMOD on NVN; modules `main`, `subsdk0` (the SDK's movie library) and `sdk`, the `sdk` build [Captain Toad: Treasure Tracker](../01009BF0072D4000/README.md) and [Link's Awakening](../01006BB00C6F0000/README.md) ship. Brought up on Android (a Galaxy S23 Ultra, 2026-10-30, then the Lenovo Legion Y700, 2026-11-06) from the NSP script's provisional profile ([profile_guess.py](../profile_guess.py)), checked by hand against the survey; not run on the Mac or an iPhone. The Android keys are in [android.tsv](android.tsv).

## What it needs

| Setting | Where | Why |
|---|---|---|
| The 5 GB block with a large alias region: `mem_reserve 0x140000000`, `mem_alias 0x118000000`, `mem_stack 0x10000000`, `app_pool 0xCD500000` | [`game.tsv`](game.tsv) | the NPDM declares 16 MB of system resource (virtual address memory) and `main` sets a heap; `nninitStartup` hands pool − used to StandardAllocator as one address region of 3.14 GB |
| `sys_resource 0x1000000` | `game.tsv` | the NPDM declares 16 MB of system resource |
| Node area cut on its `sdk` build `3ae34e68…` | `dm/tools/sdk_patches.py` | it boots only with it: `nninitStartup`'s `AllocateAddressRegion(+0xc9400000)` was refused (0x1803) with no region live, as the 1 GB-aligned 3.14 GB did not fit the alias region after the node area, and the game stopped at a Break before `nnMain` |
| Deferred sampler slot: an empty slot is skipped instead of read (8 instructions at `main+0x1b2a40c…0x1b2a428`) | [`patches.py`](patches.py), `main` build `dbc3424a…` | Unity 2018.4's NVN texture binding (`main+0x1b2a0c0`, on the render thread) fills a local array of 32 slots with the textures that are ready and never clears it; a binding that takes its sampler from another slot read a slot the first pass skipped, which held stack leftovers (null here) and crashed the render thread at the sampler ID (`main+0x1b2a424`) in about 1 launch in 3 to 5. Before: 2 crashes in 10 launches; after: 0 in 47 |

The patch is bound to this build; the function is the engine's and has not been looked for in another Unity 2018.4 build (`BINDING` in `patches.py`).

## Rendering

4,511 shaders, the Unity 2018.4 Shader objects in `Data/*.assets` and `Data/Resources` (read with the 2018.3 layout), all translated. About 625 draws per frame.

## Frame rate

60 fps held, as the game asks (present interval 1): in gameplay frame time p50 16.7 / p95 17.5 ms, 0 dropped frames, GPU busy about 10 ms on the S23 Ultra, about 7.3 ms and 0-1 dropped frames per 300 on the Y700.

## Status

Playable on Android: the save select, a new game into the garden, walking, honking, and a save made in another Switch emulator copied into the app's save folder, which loads into its later area with its language and settings; a play session by hand. Footprint about 770 MiB, 0 audio underruns, no red log line. On the Y700 two of the first 20 launches showed no frame line within 25 s (none of the 27 after). Not checked: the areas past the garden.
