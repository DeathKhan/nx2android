# Pokémon: Let's Go, Pikachu! (`010003F003A34000`)

The base game 1.0.0 (v0, no update), SDK **5.4.150** (`nnSdk-5_4_150`, `NintendoSDK_NVN-5_4_150`). Game Freak's engine before Sword and Shield's, on NVN, with Wwise sound and NEX networking. Modules `main`, `subsdk0` (GLSLC), `subsdk1` (`NintendoSDK_movie`) and `sdk`. The RomFS holds 27,818 files in 4.1 GB.

- **On Android:** plays at 30 fps (as on the Switch) on a Galaxy Z Fold 8, a Pixel 9 Pro XL and a Galaxy Z Flip 7, from a new game through Pallet Town and Professor Oak's lab to Route 1, with Pikachu and the first catches made with buttons, then on the Pixel through Route 1 to Viridian City, Oak's Parcel and the rival's first battle. Saves and Continue work. Route 2 and Viridian Forest not reached yet. See [Android](#android).
- **On the Mac:** not run yet.

## What it needs

| Setting | Where | Why |
|---|---|---|
| `modules main subsdk0 subsdk1 sdk` | [`game.tsv`](game.tsv) | every module `main` imports from |
| `sys_resource 0x0`, `mem_reserve 0x100000000`, `mem_alias 0x10000000`, `mem_stack 0x10000000`, `app_pool 0xCD500000` | [`game.tsv`](game.tsv) | the NPDM declares no system resource, so no virtual address memory. `main` sets a heap of at most the pool − 64 MB |
| `shader_groups /bin/=bin /system_resource/=systemresource` | [`game.tsv`](game.tsv) | the shaders are under `bin/` and `system_resource/shader`. See [Shaders](#shaders) |
| `zip_store .arc:.bfotf:.gfpak:.ptcl:.webm:.metallib` | [`game.tsv`](game.tsv) | already-compressed archives, fonts, effects and movies |

The profile is the provisional one `build_android_from_nsp.py` made from the survey (`dm/games/profile_guess.py`), unchanged but for its comments. `pso_manifest.tsv` is still the empty provisional list.

## Shaders

- **Where:** 11,160 shaders (4,710 vertex, 6,411 pixel, 39 compute), found by `shader_split.py`:
  - 5,864 in the effect files (`.ptcl`);
  - 2,770 in BNSH data inside `main`;
  - 2,117 in GFPAK archives (`gfpak_members`);
  - 395 in loose `.bnsh_vsh`, `.bnsh_fsh` and `.bnsh` files;
  - 14 in SARC (`.arc`) archives.
- **Android, current:** all 11,160 translate for Vulkan, none fail (`main` at `61eb7c31`).
- **Mac:** not built yet.

## Android

First run on Android, 2026-11-10: a Galaxy Z Fold 8 (SM-F971U1: Snapdragon 8 Elite Gen 5, Adreno 840 with Qualcomm's driver 0x8034a013, Android 17), folded, on its cover screen, with the on-screen controls. DroidPlay Core `main` at `61eb7c31` (0.7.2), packed and installed with `build_android_from_nsp.py`. Seven launches, about 40 minutes in all, the longest 24 minutes.

- **Plays from a new game into Pallet Town**, with nothing wrong seen in the picture, no crash and no hang:
  - the controller screen, the opening movie, Professor Oak's introduction and choosing the player's look;
  - the names (`DM_KBD` answers both the player's and the rival's);
  - the bedroom, where the rival comes up the stairs;
  - downstairs past Mom, and out of the front door into Pallet Town.
- **On to Route 1, Pikachu and the first catches** (a 24-minute launch from a save outside the house):
  - Professor Oak's scene in Route 1's grass and the catch tutorial on Pikachu;
  - Oak's lab: the middle Poké Ball, Pikachu as the partner, the rival's choice and the Pokédex;
  - Mom's Town Map outside the lab;
  - two wild catches on Route 1, a Pidgey and a Rattata, each with its first throw.
- **Catching in handheld mode works with the buttons:** A readies a Poké Ball with the aim circle on the Pokémon, and A throws.
  - The game says to move the system to look around (motion aiming), but the circle starts on the Pokémon, so a throw at one in front needs no motion.
  - The host logs nothing about motion input. Whether DroidPlay hands the phone's gyroscope to the game wasn't checked.
- **No trainer battle on this stretch:** wild Pokémon in Let's Go are caught, not fought, and no trainer battled on the way to Route 1's first catches.
- **A save:** the X menu's Save, then "Save your progress", commits (`save commit #1`) and shows "You saved your progress!". After a restart the title offers "Continue your adventure" with the trainer card's name and play time, and the game goes on where it was saved: in the bedroom, outside the house, and at Route 1's edge after the catches (Pokédex 3 Pokémon, play time 0:34).
- **30 fps throughout, as on the Switch:**
  - the bedroom: frame p50 33.3 ms, p95 35.7 ms, GPU 5.7 ms;
  - the opening's 3D scenes: GPU up to 12.6 ms (15 s means);
  - walking in Pallet Town: frame p50 33.4 ms, p95 39.3 ms, p99 41.4 ms, max 43.1 ms, GPU 15.2–15.5 ms;
  - the two catches (the encounter and the throw): frame p50 33.3 ms, p95 35.4–35.6 ms, GPU p50 9.1–9.2 ms, p95 15.0–15.3 ms;
  - the whole 24-minute launch: frame p95 36.4 ms, 54 frames over 50 ms (the longest 131 ms) at scene changes and loads, GPU p50 10.7 ms, p95 17.5 ms.
- **Starting:** the first frame comes about 0.4 s after the host starts, and the controller screen is up within 10 s. With a save, the Game Freak and Pokémon logos and the title screen come next; without one, the game goes straight into the opening.
- **Sound:** Wwise opens its 10 banks (`Init.bnk`, `System_Common.bnk` and the others in `bin/sound/NX64/wwise/bank`) and streams its `.wem` music, on AAudio at 48 kHz stereo with no underrun.
  - It asks for an `English(US)` copy of each bank first, which the RomFS doesn't have: Wwise's localized lookup, not an error.
  - Confirmed by an audio dump (`DM_AUDIO_DUMP`, 156 s from the Route 1 save: walking and a catch). The front channels carry the mix at about −22 dBFS RMS, peaks to −6.4 dBFS; the centre and LFE channels are silent, the rear pair nearly so.
  - The music runs from the title on, steady at −24 dBFS on Route 1. The menu's confirm sound is the only signal before it, at each A press on the controller screen. The level changes at the wild encounter, and louder peaks come with the catch fanfare and Pikachu's level-up.
- **Two `[RED]` lines,** each kind at boot only:
  - `more than 16 font engines with a small work buffer: this one keeps the game's`, 5 times. The game makes 21 `nn::fontll` engines with small work buffers, and `hle_sdk16.c` gives host memory to 16. Every text seen drew.
  - `a second NVN queue submits: the Vulkan replay shares one state between queues`, as in Pokémon Shield. Nothing in the picture showed it.
- **Driving it over adb:** the on-screen buttons need a held touch (`input swipe x y x y 150`), since a plain `input tap` is shorter than one 30 fps frame and is sometimes missed. In the X menu, the ▶ cursor is moved onto SAVE with the stick; SAVE's highlight alone doesn't select it. A wild Pokémon that walks into the player starts an encounter, which B and then Yes leave.
- **Not tried:** a trainer battle (the first, the rival's in Oak's lab once Oak's Parcel is delivered, was reached on the Pixel 9 Pro XL below), aiming with motion, docked mode, the other devices and the Mac.

**On a Google Pixel 9 Pro XL**, 2026-11-10: Tensor G4, Mali-G715 with Arm's driver 0xd803000 and Vulkan 1.4.343, Android 17. The hosts from `main` at `898ac92f`, the game files from the same build folder; handheld, the on-screen controls, played by hand with adb input:

- **From a new game:** the controller screen, the opening, the language, Oak, the player and the rival (`DM_KBD`), the bedroom, the rival's visit, Mom downstairs, then Pallet Town. Nothing wrong was seen in the picture, and there was no crash and no hang.
- **A catch, from the Z Fold 8's Route 1 save:** a wild Pidgey (Lv 3) caught with the buttons on its first throw, then the Pokédex entry.
- **30 fps, as on the Switch:** 29.7–30.0 fps per 30 s over the new game; the bedroom GPU p50 11–12 ms; walking in Pallet Town 30.0 fps, frame p50 33.2 / p95 43.5 ms, GPU p50 16.6 / p95 23.5 ms; the catch 29.9–30.0 fps, GPU p50 15–18 ms.
- **Saves:** the X menu's Save committed (`save commit #1`), and after a restart "Continue your adventure" loaded it in Pallet Town.
- **Sound:** an audio dump from the Continue start (47 s): silent on the controller screen, as on the Z Fold 8, then the confirm sound and the music from about 20 s at −21 to −24 dBFS RMS on the front pair.
- **`[RED]`:** the same two kinds.
- **Not tried:** Route 1 from a save made on this phone, trainer battles, aiming with motion.

**Past Route 1, on the Pixel 9 Pro XL**, later on 2026-11-10: Let's Go's host from `main` at `5736949c`, the same game files; played by hand with adb input from the Z Fold 8's Route 1 save (Pokédex 3, play time 0:34), in one 35-minute launch:

- **The stretch:** north through Route 1's terraces and grass to Viridian City. The Poké Mart's clerk, outside his shop, hands over Oak's Parcel and offers to send the player straight back to the lab. Professor Oak takes the parcel, and the rival joins him.
- **The first trainer battle:** the rival stops the player at the lab's door ("You are challenged by Pokémon Trainer Red!", the name `DM_KBD` gave him on the Z Fold 8). His Eevee (Lv 6) against Pikachu (Lv 6), turn by turn with FIGHT and Thunder Shock; won with Pikachu at 15/23 HP, then the prize money (₽3,120 after it) and Exp. Points.
- **Catches on Route 1:** wild Oddish and Pidgey caught with the buttons (Pokédex 3 to 4), and wild Rattata and Pidgey that walked into the player run from. Route 1's wild Pokémon walk into a player standing at the grass's edge again and again, which made the walk back north slow by adb.
- **30 fps, as on the Switch:** every minute of the 35-minute launch at 29.6–30.0 fps; frame p50 33.3 / p95 41.5 ms over the launch, GPU p50 15.7 / p95 25.1 ms (per minute 13.9–16.9 ms). 290 frames over 50 ms, at scene changes, encounters and loads.
- **Saves:** the X menu's SAVE in the lab after the battle (`save commit #1`) and again on Route 1 (`save commit #2`, play time 1:08). After a cold start the title showed Route 1, Pokédex 4 and 1:08, and "Continue your adventure" loaded the player where the save was made.
- **Sound:** an audio dump of the cold start (113 s): the title music at −21 to −25 dBFS RMS on the front pair, silence on the Continue menu, then Route 1's music at −21 to −24 dBFS. No underrun in 1,108,000 callbacks over both launches.
- **`[RED]`:** the same two kinds, once each per boot; no crash.
- **Not tried:** the Pokémon Center and the Mart inside, Route 2, Viridian Forest and its trainers, Pewter City.

**On a Samsung Galaxy Z Flip 7**, 2026-11-10: Exynos 2500, Xclipse 950 with Samsung's driver 0x6000215 and Vulkan 1.3.279, Android 16, open, on its inner screen. The hosts from `main` at `898ac92f`, the game files from the same build folder; the on-screen controls, played by hand with adb input from a new game:

- **The stretch:** the controller screen, the opening movie, Oak's introduction, the default look, both names (`DM_KBD`), then walking in the bedroom.
- **30 fps, as on the Switch:** 29.8 fps over the 6.6-minute first launch, 29.9 walking; frame p50 33.4 ms, p95 37–40 ms; GPU p50 5.0–7.2 ms.
- **Saves:** the X menu's SAVE committed (`save commit #1`), and the title's "Continue your adventure" loaded it in the bedroom.
- **Sound:** an audio dump: the title music at −20 to −27 dBFS and the bedroom's at −23 to −26 dBFS on the front pair; no underrun in 256,000 callbacks.
- **`[RED]`:** the same two kinds.
- **Not tried:** downstairs and Pallet Town, catches, aiming with motion.

**Onto Route 2 and into Viridian Forest on the Z Flip 7**, 2026-11-10/11 (job q120): Let's Go's host from `main` at `90379f4e`, the game files from the same build folder (no shader translated again, none sent). Played by hand with adb input from the Pixel 9 Pro XL's save after the rival's battle (`saves/route1_after_rival`, a save made on an Android phone), about 40 minutes:

- **The stretch:**
  - Route 1's terraces north. Its ledges only drop, so the way up is at their ends: past the sign on the west side, then the gap between the ledge and the raised block in the middle.
  - On the way: two wild catches with the buttons (a Rattata, a Pidgey). The Poké Mart clerk hands over a Potion. Youngster Ronny's battle is won (his Rattata against Pikachu's Thunder Shock, ₽60).
  - Viridian City: the rival's advice about the Pokémon Center, then north through the town onto **Route 2**.
  - Route 2's way round its ledges to the forest's gatehouse, and through it into **Viridian Forest**, where a wild Weedle was caught (its second throw; Pokédex 4 to 5).
  - It stopped in the forest's first stretch of grass, before its Bug Catchers battled.
- **Wild Pokémon walk into the player** on Route 1, Route 2 and in the forest, an encounter every few steps. Leaving each one is down three times to Run away, then A twice. A script that does that after every move (an orange pixel of the encounter menu's "Get ready" bar) made the walk possible by adb.
- **Saves and Continue:** three commits, in Viridian City, on Route 2 and in Viridian Forest (play time 1:39). A cold start's title card showed Viridian Forest, and Continue loaded it there. Kept as `saves/viridian_forest_flip`.
- **30 fps, but for one place:** 242 windows of 300 frames over the first 40-minute launch, frame p50 median 33.3 ms. GPU p50 median 11.8 ms, the most 15.2.
- **Slower by the forest's first Bug Catcher, both times:** walking past him just inside the forest ran at 21–28 fps for about 90 s on the first visit and about 40 s after the Continue start, then held 30 again for the rest of the forest.
  - The frame there is the game's own work: frame work p50 47 ms, against the replay's submit and present at 8.6 ms and the GPU at 13.7 ms.
  - No pipelines were built and no textures uploaded in those frames, and the phone's thermal status was 0.
  - Not compared with the Mac or another phone (none has been in the forest yet). The Android host doesn't log per-thread CPU, so which game thread is busy isn't known.
- **Sound:** `DM_AUDIO_DUMP` over the Continue start (202 s): silent on the controller screen and the Continue menu, as on the other phones; the title music and then the forest's on the front pair. No underrun in 616 reports during play or 49 after the restart.
- **`[RED]`:** the same two kinds; no crash, no hang, and nothing wrong seen in the picture (Route 1, Viridian City, Route 2, the gatehouse, the forest's light).
- **Not played:** Viridian Forest's trainers and its exit, Pewter City, aiming with motion.

**Options → Open Pokémon GO Settings**, 2026-11-11 on the Z Flip 7 (enjoyerofslop/droidplay-core#220):
- **Before the fix:** from the title's "Change your options" or the X menu's Options, the entry ended the game at once with `Break#1 reason=0` at `sdk+0xd989c`. The cause was `nn::bluetooth::RegisterGattOperationNotification`, which reached btm:u with no session ([Bluetooth LE at boot](../../../docs/reference/services.md#bluetooth-le-at-boot)).
- **With the fix:** both routes show the Pokémon GO notice and then "Would you like to pair your game with a Pokémon GO account". Yes leads to "Searching for a Pokémon GO account…", which keeps searching, as a Switch with no phone nearby does, until B Quit returns to the options.
- **After the menu:** Continue loaded `viridian_forest_flip`. The player walked, met a wild Pidgey and ran from it, opened the menu again, and made a save (`save commit #1`). There was no new `[RED]` line.
