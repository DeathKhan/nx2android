# NEO: The World Ends with You (`010043B013C5C000`)

Update 1.03 (v196608), SDK 10. Unity 2019.4 IL2CPP with CRIWARE (`.cpk` archives, `.usm` movies) on NVN; modules `main`, `subsdk0` and `sdk`. The first game the Android port was brought up on (Lenovo Legion Y700, 2026-10-25): its profile is still the NSP script's provisional profile ([profile_guess.py](../profile_guess.py)), not checked by hand, and the port's host fixes from its bring-up are in the core. Not run on the Mac or an iPhone. The Android keys are in [android.tsv](android.tsv), the first one written (a profile without one is not ported to Android).

## What it needs

| Setting | Where | Why |
|---|---|---|
| The 5 GB block with a large alias region: `mem_reserve 0x140000000`, `mem_alias 0x118000000`, `mem_stack 0x10000000`, `app_pool 0xCD500000` | [`game.tsv`](game.tsv) | the NPDM declares 16 MB of system resource (virtual address memory) and `main` sets a heap; the heap check sizes the heap region from the heap the game asks for |
| `sys_resource 0x1000000` | `game.tsv` | the NPDM declares 16 MB of system resource |
| `requires bcn` | `android.tsv` | the textures are BC-compressed (Unity's Switch build); the installer checks the device for them |
| `setting docked=0`, `overlay=1`, `rumble=1` | `android.tsv` | handheld (1280x720), as it was played on the Y700, with the on-screen gamepad; docked (1920x1080) is a pause-menu switch away |

No patches of the game's own builds; every `game.tsv` key is the provisional profile's guess.

## Rendering

25,523 shaders, all translated since the translator stopped propagating call arguments (2026-10-25: 25,514 before, the earlier ones byte-identical). The split also gives the 20 compute shaders of Unity's default resources (Internal-Skinning, Internal-BlendShape) since `unityfs.py` reads Unity 2019.4 ComputeShaders (2026-10-27, found with [Unpacking](../0100701015846000/README.md)); the tables were split on Windows and split again with `/` paths.

## Frame rate

The bring-up installer check saw it boot to its file check at about 60 fps; no measurement past that is recorded in the profile.

## Status

Playable on the Legion Y700 (`status playable`). The profile records no test list: what was played, the frame times, the audio and the log lines are not written down, and `game.tsv` has not been checked against the survey.
