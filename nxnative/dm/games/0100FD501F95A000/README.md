# Viractal (`0100FD501F95A000`)

Viractal: Will You Trust Your Party?, update 1.4.5 (v131072), SDK 21.4.4. Unity 6000.3 IL2CPP on NVN; modules `main`, `subsdk0` (the SDK's movie library) and `sdk`, the `sdk` and `subsdk0` builds [Graveyard Keeper 2](../0100005027A18000/README.md) and [Tales of Eternia Remastered](../010006E0230AC000/README.md) ship too. Brought up on Android (Lenovo Legion Y700, 2026-10-25, the second game after NEO) from the NSP script's provisional profile ([profile_guess.py](../profile_guess.py)), checked by hand against the survey (the modules, the 5 GB layout of a game without a heap, the system resource); not run on the Mac or an iPhone. The Android keys are in [android.tsv](android.tsv).

## What it needs

| Setting | Where | Why |
|---|---|---|
| The 5 GB block of a game that maps its own memory with no heap (as [Hollow Knight: Silksong](../010013C00E930000/README.md)): `mem_reserve 0x140000000`, `mem_alias 0x118000000`, `mem_stack 0x10000000`, `app_pool 0xCD500000` | [`game.tsv`](game.tsv) | `main` never sets a heap; the game maps its memory itself in the 4.375 GB alias region |
| `sys_resource 0x1000000` | `game.tsv` | the NPDM declares 16 MB of system resource (virtual address memory) |
| Node area cut on its `sdk` build `14839413…` | `dm/tools/sdk_patches.py` | the SDK 21.4.4 build shared with Graveyard Keeper 2 and Tales of Eternia: without the cut Unity 6's regions start after the 1 GB node area (found on Graveyard Keeper 2) |
| `nvnTextureViewCompare`; advanced blend mode 21 (multiply) bound as plain blend functions | core (`hle_nvn.c`), found on this game | it calls the compare over and over from its title screen on, where it was a red stub (40+ red lines by the title); the multiply mode was its last red line |
| The front end's text entry (the overlay keyboard) | core (`front_android.c` `front_text_input`), found on this game | the contract's Enter Name opened an `nn::swkbd` request that returned the defaults before; "Rina" typed by touch, an adb keyboard and controller keys signed the contract, cancel keeps the old name |

No patches of the game's own builds.

## Rendering

25,555 shaders, all translated: 1,894 failed in the translator until Ryujinx's AST optimizer stopped propagating a local a call takes as an argument (2026-10-25), and the 92 compute shaders of its `globalgamemanagers.assets` and default resources decode since `unityfs.py` has Unity 6000.3's ComputeShader layout (1 before). The tables were split on Windows and split again with `/` paths.

## Frame rate

30 fps, about 1.9 GB, nothing skipped over 12 minutes.

## Status

Playable on Android, driven with DM_CTL: the title, the Game Room tutorial, the story's opening (Baphoco's poster; its "Hey! Not that way!" is the game telling the player to pick it, not a soft-lock), the board game's setup (controller settings, contracts, jobs) and its first turns: dice, moves, tile events, the next player. No red log line. Not checked: the game past its first turns, handheld mode.
