# Persona 3 Portable (`0100DCD01525A000`)

Update 1.0.2 (`v131072`), SDK **14.3.0** (`NintendoSdk_nnSdk-14_3_0`, `NintendoSDK_NVN-14_3_0`). Atlus's in-house engine with CRI middleware (CRIWARE), on NVN: the same engine as [Persona 4 Golden](../010062B01525C000/README.md), whose `subsdk0` and `sdk` it shares byte for byte. Modules `main`, `subsdk0` (`NintendoSDK_movie`) and `sdk`. The RomFS holds 11 CPK files in 7.09 GB.

- **On Android:** plays at 60 fps (as on the Switch) from a new game through the first school days to the night of April 9 and its first battle, on a Galaxy S25 Edge (Adreno) and a Galaxy Z Flip 7 (Xclipse), with saves and Continue working, in Android's default handheld mode as in console mode. See [Android](#android).
- **On the Mac:** not run yet.

## What it needs

| Setting | Where | Why |
|---|---|---|
| `modules main subsdk0 sdk` | [`game.tsv`](game.tsv) | every module `main` imports from |
| `sys_resource 0x0`, `mem_reserve 0x100000000`, `mem_alias 0x10000000`, `mem_stack 0x10000000`, `app_pool 0xCD500000` | [`game.tsv`](game.tsv) | the NPDM declares no system resource, so no virtual address memory. `main` sets a heap of at most the pool − 64 MB − its 148 MB of bss |
| `romfs_marker data/umd0.cpk`, `zip_store .cpk:.metallib` | [`game.tsv`](game.tsv) | the RomFS is CPK archives: `data/` (6.26 GB), one `data_XX/` per language and the small `sysdat.cpk` at the root |

## Shaders

- **Where:** 30 shaders (12 vertex, 17 pixel, 1 geometry), all BNSH entries under `sysdat.cpk/shaders` (`P4G/`, `system/` and `system/posteffect/hdr/`), found by `shader_split.py`. The survey's `shader_groups` names the `data*` folders, so the split puts them all in its `other` packages; [Persona 4 Golden](../010062B01525C000/README.md)'s `shader_groups /sysdat.cpk/=sysdat` would name them better, untested here.
- **Android, current:** all 30 translate for Vulkan, none fail (`main` at `30215723`).
- **Mac:** not built yet.

## Android

First run on Android, 2026-11-10: a Galaxy S25 Edge (SM-S937U1: Snapdragon 8 Elite, Adreno 830 with Qualcomm's driver, Vulkan 1.3.284, Android 16), with the on-screen controls. DroidPlay Core `main` at `30215723` (0.7.2), packed and installed with `build_android_from_nsp.py`. Five launches, about 95 minutes in all.

- **Plays from a new game through the first battle**, with nothing wrong seen in the picture, no crash and no hang:
  - the opening movie, the title, a new game (male protagonist, Beginner);
  - the prologue's arrival at the dorm and the name entry (the game's own keyboard, driven with the buttons; `DM_KBD` answers only the system keyboard);
  - the protagonist's room, the first point-and-click exploration (its cursor and hotspots), and bed;
  - April 7 at Gekkoukan High: the train, the main lobby and the Faculty Office hallway (the Y button's "Where do you want to go?" list), the Faculty Office, the opening ceremony and the classroom after school;
  - back at the dorm in the evening, and the lounge;
  - after Continue: bed, the calendar, April 8 (the class's question, after school, the Chairman's visit to the dorm and his questions) and that night's Dark Hour, the Velvet Room;
  - April 9 (the full moon): school, the evening at the dorm, bed from the room's Bed hotspot, then the Dark Hour attack, the escape to the dorm's rooftop and the summoning scene (Orpheus, then Thanatos);
  - **the first battle:** the command menu, Attack on a Cowardly Maya, the Slash Attack, the result screen (56 EXP) and Orpheus's level-up to 2, all with the buttons.
- **A save:** the lounge's Sign-In Sheet → Save, into slot 1, commits (`save commit #3`) and shows "Save complete." with 4/7 (Tu) Evening, Dorm, 0h 40m. After a restart the title loads it, and the game goes on in the dorm lounge, the cursor still on the Sign-In Sheet.
- **60 fps throughout, as on the Switch:** in all 894 of the logs' 300-frame windows over the three long launches (268,200 frames, 74 minutes), the frame time's median is 16.64–16.77 ms. p95 median 17.7 ms (worst window 23.0 ms); GPU p50 about 6 ms, 11.9 ms at most. The battle itself: 16.65–16.68 ms, GPU 4.9–7.6 ms. The game process holds about 0.94 GB resident (RSS) on April 8.
- **Sound:** CRI's sound plays on AAudio at 48 kHz stereo with no underrun. A `DM_AUDIO_DUMP` of the boot holds the opening movie's song.
- **One `[RED]` line,** once in each of two launches (frames 15,600 and 25,500; the first during the new game's opening): `[vk] storage buffer of shader 1f8fd67ad5e3e355 (c0 +0x120) is not a bound buffer: it reads 0`. The shader is `system/custom_Prim3D_Flat_vert.bnsh`. Nothing in the picture showed it.
- **Handheld mode works, as console mode does.** The play above ran in console mode (`DM_OPMODE=1`), because the first test took the title for menus that ignore the buttons in handheld mode, Android's default. A second test, on the same phone and `main` at `aade6451`, found nothing wrong in handheld mode (the settings screen's `DM_OPMODE=0`):
  - with the on-screen buttons: the title menu, the load screen (the D-pad moves between slots, B goes back), NEW GAME and the new game's gender choice, and in the dorm's lounge the cursor and Y's list of places;
  - with only a `DM_KEYS` script pressing A (no touch at all): the title, the load screen and "Load this file?" into the lounge;
  - no touch or gesture reached the game in either test (a temporary log of `dm_touch_set` and the gesture history stayed empty): the on-screen buttons are a window over the game's own and arrive only as the Handheld controller's buttons.

  What misled the first test is the title's timing. A press while the title fades in does nothing; the menu drops back to "PRESS ANY BUTTON" after a few seconds without input; and the prompt fades into the attract movie (the logo and the opening) after about 12 s. Presses a few seconds apart kept landing in a fade or on the prompt again, so A and B seemed not to confirm NEW GAME.
- **On Xclipse too:** a Galaxy Z Flip 7 (SM-F766U1: Exynos 2500, Xclipse 950 with Samsung's driver 0x6000215, Vulkan 1.3.279, Android 16), unfolded, 2026-11-10, `main` at `aade6451`, in handheld mode with the 2D on-screen layout (no D-pad: the left stick moves the cursor and the menus). Two launches, about 47 minutes:
  - a new game (Normal) through the opening, the name entry, the room's Bed, April 7 at school and the dorm; a save at the Sign-In Sheet (`save commit #3`, 4/7 Evening, 0h 20m); after a restart LOAD GAME and "Load complete." into the lounge; then April 8 and 9, the attack, the rooftop, the summoning and **the first battle**, won with Attack (48 EXP, level 3, 15 of 83 HP left on Normal);
  - 60 fps: in 541 of the 558 log windows (167,400 frames) the median frame time is 16.57–16.8 ms. Of the other 17, 16 come in a launch's first 90 s (the logos and the opening movie, up to 18.2 ms) and one, at 17.3 ms, about five minutes in. p95 median 17.0 ms, worst window 23.6 ms; GPU p50 about 3.8 ms, 8.2 ms at most (half the Adreno 830's); 1.42 GB PSS on April 9;
  - sound: the game opens 6 channels, played as stereo on AAudio; a `DM_AUDIO_DUMP` has sound on the front pair from 5 s in. One underrun in the first launch, none in the second;
  - the same single `[RED]` as on the Adreno, once per launch, and nothing wrong seen in the picture, except perhaps one frame of the opening movie covered in a fine dot grid (an 8-pixel period), which may be the movie's own look: the frames around it were clean, and that moment wasn't compared on the Adreno.
- **Driving it over adb:** the on-screen buttons need a held touch (`input swipe x y x y 150`). In the point-and-click areas, the D-pad moves the cursor (holding it moves it fast), a hotspot names itself at the bottom right when the cursor is on it, and Y lists the area's exits. At the title, press as soon as "PRESS ANY BUTTON" shows, then pick from the menu within a few seconds.
- **Not tried:** past the first battle (Tartarus and the following days), the female protagonist, Mali (the Pixel 9 Pro XL), the other devices and the Mac.
