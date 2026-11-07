# Tomodachi Life: Living the Dream (`010051F0207B2000`)

Update 1.0.4 (v262144), SDK 20.5.3. Nintendo's engine (sead / agl / gsys, NintendoWare G3d2 / Ui2d / Vfx2 / Font, Havok) on NVN, the Mii heads drawn by the SDK's Mii renderer; modules `main` and `sdk`. Brought up on Android (Lenovo Legion Y700, 2026-11-06) from the experimental port's survey ([profile_guess.py](../profile_guess.py)), checked against it (the modules and build IDs, the heap `main` sets, no system resource); not run on the Mac or an iPhone. The Android keys are in [android.tsv](android.tsv).

## What it needs

| Setting | Where | Why |
|---|---|---|
| A 3328 MB heap region: `mem_reserve 0x100000000`, `mem_alias 0x10000000`, `mem_stack 0x10000000`, `app_pool 0xCD500000` | [`game.tsv`](game.tsv) | `main` sets a heap of at most the pool − 64 MB |
| `sys_resource 0x0` | `game.tsv` | the NPDM declares 0 MB of system resource: `IsVirtualAddressMemoryEnabled` is false, as on Horizon |
| `render_cond on` | `game.tsv` | the Mii renderer draws a depth pre-pass, then clears depth under a render condition (`SetRenderEnableConditional` at `main+0x208f18`, the clear at `main+0x208f54`) and draws the colour pass with depth test EQUAL. Run unconditionally the clear wiped that depth and every head, face part and hair draw failed the test: the Mii heads and faces were empty |
| `mem_used 0x6600000` | `game.tsv` | the rewritten modules are larger than the default 64 MB of used memory |
| `env DM_STEADY_REALTIME=1` | `android.tsv` | the game compares its saved clock snapshot with a new one at each start (`main+0x871400`) and said "Your system time has changed" every time: the host's steady clock is monotonic time, which restarts at each device boot; with the switch it follows real time from a fixed epoch, as the Switch's RTC-backed one |
| `ClockSnapshot::CreateWithAdjustableSystemClock` answered; `CalculateSpanBetween` returns a bool | core (`hle_time.c`), found on this game | it aborted at boot until the first was answered; the host returned Result 0 from the second where the SDK returns true, which the game read as a clock change (checked 2026-11-06: two restarts after a save, no notice) |

No patches of the game's own builds. The game uses no MiiModel system data (the profile asked for it before the `render_cond` fix; porter's `--firmware` / `sysdata` path came from that).

## Rendering

14,400 shaders (14,388 in the RomFS's archives), all translated.

## Frame rate

60 fps on the Y700, frame work p50 about 5 ms, 0 dropped frames.

## Status

Playable on Android, docked: the date check, region and currency, the island (the 3D island, its sea and trees, the cursor and the camera), naming the island, the Mii creation menus, the guided Mii creation (face questions, the 3D Mii with its face, height and body type), then play by the user after the `render_cond` fix. 0 audio underruns, no red log line. Not checked: a save made in play (it saves only at its own moments), life on the island past Mii creation, handheld mode.
