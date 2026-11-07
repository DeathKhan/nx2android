# Lonely Mountains: Downhill (`0100A0C00E0DE000`)

Base game 1.0.5 (v0), SDK 9.3.1. Unity 2018.4.14f1 IL2CPP with FMOD on NVN; modules `main`, `subsdk0` and `sdk`. A generated profile ([profile_guess.py](../profile_guess.py)) contributed with its Android test (GitHub pull request #114, 2026-11-05, with nx2android v0.3.0), not checked by hand against the survey; not run on the Mac or an iPhone. The Android keys are in [android.tsv](android.tsv).

## What it needs

| Setting | Where | Why |
|---|---|---|
| The 5 GB block with a large alias region: `mem_reserve 0x140000000`, `mem_alias 0x118000000`, `mem_stack 0x10000000`, `app_pool 0xCD500000` | [`game.tsv`](game.tsv) | the NPDM declares 16 MB of system resource (virtual address memory) and `main` sets a heap; the heap check sizes the heap region from the heap the game asks for |
| `sys_resource 0x1000000` | `game.tsv` | the NPDM declares 16 MB of system resource |
| `env DM_VK_MT_REPLAY=1` | `android.tsv` | the tested conversion enabled multithread replay; the profile preserves that option |

No game-specific code patches. No custom driver is bundled or selected by the profile; the test device ran PurpleVK-public `git-62ac221a33` (Vulkan 1.4.359, driver 26.2.99), and stock-driver behaviour has not been compared.

## Rendering

Known issues from the test: a brief visual glitch between rider preparation and GO on the first race, and occasional one-frame black/white flashes. The runtime reports CPU reads overlapping GPU writes not known complete; their relation to the glitches is unproven, and the warnings also occur before the 120 Hz experiment. No shader count is recorded.

## Frame rate

Approximately 60 fps in downhill gameplay. A separate opt-in 120 Hz experiment (`DM_PRESENT_HZ=120`, `DM_ADAPTIVE_FPS=0`) was reported playable at normal speed with frame medians near 8.3 ms; it is not the default, as long-term thermals, precise simulation timing and all routes remain unverified.

## Status

Playable on a Retroid Pocket 6 (QCS8550 / Adreno 740, Android 13): launch, controller input and downhill gameplay, no crash in the short sessions. Not verified: save and reload, a full playthrough, the DLC courses.
