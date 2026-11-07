# Unpacking (`0100701015846000`)

Update 2.01 (v327680), SDK 13.3.2. Unity 2019.4.34f1 IL2CPP with Wwise on NVN; modules `main`, `subsdk0` (the SDK's movie library, for the Humble Games logo intro, an H.264 MP4) and `sdk`, the same `sdk` and `subsdk0` builds as [Bayonetta 3](../01004A4010FEA000/README.md). Brought up on Android (Lenovo Legion Y700, 2026-10-26) from the NSP script's provisional profile ([profile_guess.py](../profile_guess.py)), checked by hand against the survey; not run on the Mac or an iPhone. The Android keys are in [android.tsv](android.tsv).

## What it needs

| Setting | Where | Why |
|---|---|---|
| The 5 GB block with a large alias region: `mem_reserve 0x140000000`, `mem_alias 0x118000000`, `mem_stack 0x10000000`, `app_pool 0xCD500000` | [`game.tsv`](game.tsv) | the NPDM declares 16 MB of system resource (virtual address memory) and `main` sets a heap; `nninitStartup` hands pool − used (3.14 GB) to StandardAllocator as one address region |
| `sys_resource 0x1000000` | `game.tsv` | the NPDM declares 16 MB of system resource |
| Node area cut and 1 GB alignment cut on its `sdk` build `357efc19…` | `dm/tools/sdk_patches.py` | without them the 1 GB-aligned region sat at +1.00 GB and Unity's own regions ran out of the alias region (0x1803 from 2.3 MB up): the camera target was never initialized and the title was a white screen behind the menu |
| `env DM_TOUCH_DOCKED=screen` | `android.tsv` | the game plays by touch as on the Switch in handheld mode; on Android the screen is a touch screen docked as well, where Unity 2019.4 takes the panel's values as screen pixels |
| `setting overlay=0` | `android.tsv` | the on-screen gamepad covers the room (the pause menu turns it on); a controller works as on the Switch |

No patches of the game's own builds. The base game 1.0 alone (v0, a derived profile, 2026-10-31, GitHub issue #26) ships SDK 11.4.3's `sdk` build `2d369d99`, which now has the same cuts: without them its StandardAllocator region ended at +4.14 GB of the alias region, with them +3.21 GB.

## Rendering

3,948 shaders, the Unity 2019.4 Shader and ComputeShader objects in `Data/*.assets` and `Data/Resources` (no type trees; `unityfs.py` has the 2019.4 layouts), all translated. Screen recordings found no flickering frame, standing or dragging items.

## Frame rate

60 fps held, as the game asks (present interval 1): frame work p50 6-7 ms in the room.

## Status

Playable on Android, by touch, docked (1920x1080) and handheld (1280x720): the Humble Games logo movie, the title, settings, credits, a new game, the name entry (the overlay keyboard) and the first room (1997); a 10-minute touch soak docked with no audio underrun and no red log line; a play session by hand with clean music and effects, the save kept across launches, and the notes screen (the album), which froze until the host had a standard shared font (Android's Roboto). The base game v0: the title, a new album, the first room and items out of a box at 60 fps. Not checked: the rooms past 1997.
