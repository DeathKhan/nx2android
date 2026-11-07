# The World Ends with You -Final Remix- (`0100C1500B82E000`)

Base game 1.0.0 (v0; no update was extracted), SDK 4.4.1. Square Enix / h.a.n.d., Unity 2017.2 IL2CPP with FMOD on NVN; modules `main`, `subsdk0` (the SDK's movie library: H.264 + AAC MP4 movies in `StreamingAssets/Movie`, decoded by the host) and `sdk`. Brought up on Android (Lenovo Legion Y700, 2026-11-06) from the experimental port's survey ([profile_guess.py](../profile_guess.py)), checked against it (modules, build IDs, the heap the game sets); not run on the Mac or an iPhone. The Android keys are in [android.tsv](android.tsv). Its sequel is [NEO: The World Ends with You](../010043B013C5C000/README.md).

## What it needs

| Setting | Where | Why |
|---|---|---|
| A 3328 MB heap region: `mem_reserve 0x100000000`, `mem_alias 0x10000000`, `mem_stack 0x10000000`, `app_pool 0xCD500000` | [`game.tsv`](game.tsv) | `main` sets a heap of at most the pool − 64 MB |
| `sys_resource 0x0` | `game.tsv` | the NPDM declares 0 MB of system resource: `IsVirtualAddressMemoryEnabled` is false, as on Horizon |
| `op_mode handheld`, `pad_layout touch` | `game.tsv` | docked, the game is played with one Joy-Con as a gyro pointer ("Point the Joy-Con at the center of the screen"), which the host does not have; handheld, by the touch screen (every touch goes to the game) |
| `setting docked=0`, `overlay=0`, `rumble=1` | `android.tsv` | the pause menu starts in handheld mode with the on-screen gamepad hidden |
| A second end-of-stream packet gives no decoder output | core (`hle_movie.c` `h_movie_SendIn`), found on this game | the opening movie crashed the game 1 run in 2 at its end until then; 3 of 3 runs after |

No patches of the game's own builds.

## Rendering

458 Unity shader variants in `data.unity3d` and the asset bundles, all translated.

## Frame rate

60 fps throughout, as on the Switch: frame work p95 about 18 ms of a 16.7 ms period at the vsync, GPU about 2 ms, 0 dropped frames in each 300-frame window.

## Status

Playable on Android, handheld, by touch: the title, the tips, a new game's opening movie (sound and picture), the story scenes, Scramble Crossing (walking by touch, the scan with the player pin), the first battle (running from the frogs by dragging Neku). No red log line, 0 audio underruns. A drag that starts at the tablet's left edge is Android's back gesture (the pause menu). Not tested: a save point, the later battles (psych pins), the staff roll movie.
