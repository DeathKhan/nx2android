# Can NX2Apple's approach bring Switch games to Android?

> **Note (2026-11-11):** these are the original research notes. This fork has since removed the `nx_extract` / LibHac step and every use of keys described below: it reads a game only as a NUP container, whose files are already plain.

*Research notes, 2026-10-24. Based on a read of [LNPANDY/NX2Apple](https://github.com/LNPANDY/NX2Apple) @ `6da7b16` (upstream [seedds/nxnative](https://github.com/seedds/nxnative) @ `ebe75c9`) plus web research. Nothing was built or run on a device.*

## TL;DR

**Yes, technically, and the CPU half carries over almost as-is.** The project is not an emulator. It statically rewrites the game's own ARM64 code so it runs natively, and replaces Horizon OS and the GPU API with a host layer. Android phones are ARM64 as well, so the core idea fits.

The cost is the **graphics backend**. About 1 MB of source reimplements Nintendo's NVN API on Apple Metal, and it would have to be rewritten for Vulkan. Android GPU fragmentation is the main technical risk: BCn texture support varies, host-memory import is unreliable, and drivers have bugs.

> **Update after on-device tests (§6):** on the Lenovo Legion Y700 (Snapdragon 8 Elite / Adreno 830, Android 16), every CPU assumption checked out. BCn textures are supported in hardware. Guest memory pools can be zero-copy after all: allocate an `AHardwareBuffer` and `mmap(MAP_FIXED)` its dma-buf at the game's address. That mapping is CPU-cached and GPU-coherent. The biggest design risk listed below is solved on this device.

The payoff is smaller than on iOS. Android already has Switch emulators that run game code natively (yuzu's NCE and its successors). The **legal exposure is the same** as for the emulators Nintendo targeted in February 2026, because the pipeline still decrypts games with `prod.keys`.

---

## 1. What NX2Apple actually is

- A fork of **nxnative** (MIT, ~2.6k stars, ~490 forks). NX2Apple adds a macOS GUI ("NS Port Builder"), a sharded RomFS archive (`ROMFSARC`) so huge games like Tears of the Kingdom fit through the iOS installer, resumable builds and a daily upstream sync.
- How the pipeline runs on a Mac:

```
your NSPs + prod.keys
  → nx_extract (C#/LibHac)          decrypt, layer update, dump ExeFS + RomFS
  → patch_nso.py                    static rewrite of each NSO (ARM64 → ARM64)
  → shader_split + dmsh_xlate       Maxwell shader binaries → SPIR-V (Ryujinx's translator)
  → spirv-cross → metal/metallib    SPIR-V → MSL → Metal libraries   (Apple-only step)
  → Xcode build of dm/host + dm/gfx + iOS shell
  → slim IPA + RomFS archive        sideload; copy RomFS to Documents/
```

- **Per-game profiles** (`dm/games/<titleID>/`) are mandatory. They hold build IDs, memory layout, shader tables and per-build code cuts. There are 48 profiles, and about 13–20 games are reported playable on iPhone. For a new game, the docs describe a survey → provisional profile → manual bring-up loop.
- Upstream requires an iPhone 15 Pro / M1 iPad or newer, because game textures are BC-compressed and older Apple GPUs can't sample BC. Big games need the increased-memory entitlement (~3.4 GB → ~6 GB).

### What the rewriter changes (`dm/tools/patch_nso.py`)

| Original instruction | Why it's rewritten | Needed on Android? |
|---|---|---|
| `x18` reads/writes | Apple's kernel clobbers x18 | **Probably keep.** Android reserves x18 (ShadowCallStack); the rewrite is harmless |
| `mrs/msr tpidr_el0`, `mrs tpidrro_el0` | The host libc owns the TLS register | **Yes.** Bionic uses `tpidr_el0` for TLS |
| `svc #n`, `mrs cntpct_el0` | Horizon syscalls / timer → host gate | **Yes.** `svc` would hit the Linux kernel |
| `sub sp, sp, xN` misalignment | Apple faults on a misaligned sp | **Yes.** Linux sets SA0 too |
| spin-wait loops | A busy poll is free on the Switch but burns battery on a phone | Yes (power) |

The output (`*.patched.bin`, `fixups.tsv`, `gates.tsv`) is platform-neutral. **The rewriter is reusable unchanged**, apart from small loader/key-page conventions.

## 2. Component-by-component portability

| Component | Size (source) | Apple coupling | Android work |
|---|---|---|---|
| `nx_extract` (.NET / LibHac) | small | none | **None.** Runs on Windows/Linux too |
| Rewriter `patch_nso.py` + per-build cuts | ~2.5k lines | none | **None / trivial** |
| Host HLE `dm/host` (svc, threads, fs, hid, audio, movies, …) | ~500 KB | Mach APIs (`vm_allocate`, thread ports, `thread_get_state` for dumps, time-constraint policy, `os_proc_available_memory`), Core Audio | **Medium.** Swap in `mmap`, signals / `tgkill`, ADPF hint sessions, AAudio/Oboe, `/proc` stats. Most service logic (fs, hid, account, time, movie decode via libvpx) is plain C and carries over |
| Assembly `entry.S`, `nvn_thunks.S` (generated) | small | Mach-O asm syntax (`_` prefixes, `@PAGE` relocations) | **Low.** Mechanical change to ELF syntax |
| NVN record layer `gfx/nvn/hle_nvn.c` | 229 KB | some Metal types / hooks | **Medium.** Mostly reusable |
| **NVN replay layer `gfx/nvn/nvn_mtl.m`** | **842 KB Obj-C** | **100% Metal** | **High. Full rewrite as a Vulkan backend.** This is the bulk of the project |
| Vulkan games (`gfx/vk/hle_vk.m`: Hades, Mario Galaxy) | 1.5k lines | Vulkan → Metal shim | **Easier than on iOS.** The game's Vulkan calls can mostly be forwarded to the phone's real Vulkan driver |
| GL games (`gfx/gl/hle_gl.m`: Digimon, Balatro) | 2k lines | desktop GL 4.5 → Metal | Medium: GL → Vulkan layer (or ANGLE / Zink-style) |
| Shader chain | 4.6k lines tools | Last two steps are SPIRV-Cross → MSL → `metallib` | **Medium, and simpler than today.** The chain already produces SPIR-V; drop the MSL/Metal steps. Rework bindings for mobile Vulkan (descriptor indexing for bindless) and regenerate the reflection data. **The build machine no longer needs a Mac** |
| Textures (`gfx/tex`) | small | Assumes hardware BCn | **Medium.** Mali lacks BCn and Adreno support varies by GPU/driver, so a compute-shader (or CPU) BCn decode is needed at upload, which costs memory |
| iOS shell `dm/ios` | ~800 lines | UIKit, GameController | **Low.** Write a GameActivity/NativeActivity app with Android input and Gradle packaging |
| Packaging / ROMFSARC | — | Works around iOS installer limits | Less critical: put the RomFS in `Android/data/<pkg>/files`. The archive still speeds up copying over MTP/adb |
| Mac-only build steps (heap check, GLSLC / Vulkan shader capture) | — | Boot the game in the macOS host | Need a Linux or Android host to run these captures |

### The one deep design assumption that breaks

The Metal backend creates every NVN memory pool as a **zero-copy `MTLBuffer` over the game's own memory** (`newBufferWithBytesNoCopy`), relying on Apple's unified memory. That is why CPU writes from the game are instantly visible to the GPU, with no tracking.

On Android the Vulkan equivalent is `VK_EXT_external_memory_host`, and it is **not reliably available on Qualcomm's Adreno driver**. The fallback is shadow GPU buffers with dirty-page tracking and uploads, which is the "buffer cache" machinery yuzu-family emulators have. It is slower and much more complex. Expect this to drive a large part of the redesign, and probe each target device for the extension first.

## 3. Android vs iOS for this approach

**Easier on Android**
- No JIT ban (not needed anyway), no code-signing or entitlement tricks.
- Memory: no hard per-app cap like iOS jetsam's ~3.4 GB; flagship phones have 12–24 GB.
- 16 KB pages are already handled (Apple uses them), so both 4 KB and 16 KB Android kernels are fine.
- Games that render with Vulkan can run on the real driver instead of a translation layer.
- The whole build pipeline could run on Windows/Linux.

**Harder on Android**
- **GPU fragmentation:** Adreno, Mali, Xclipse and PowerVR differ in features and driver bugs. Apple has one GPU family.
- No "GPU address = CPU address" shortcut (see above); BCn texture support is patchy.
- Thermals and sustained clocks on phones; many devices have only 6–8 GB RAM.
- **Sideloading is tightening.** Google's developer verification has been enforced since 30 Sep 2026 in Brazil, Indonesia, Singapore and Thailand (for installs via participating stores), with a global rollout planned for 2027. Unverified APKs need the new "advanced flow" (Developer Options, a 24 h wait, then allow unverified installs) or `adb install`, which remains exempt.

## 4. Landscape: what already exists on Android

- **yuzu added NCE (Native Code Execution) on Android in late 2023**, running Switch ARM64 code directly on the phone CPU, with reported 20–100% FPS gains. It needs games using the 39-bit address space; others fall back to the dynarmic JIT. Its descendants (Eden, Citron/Citron Neo, Kenji-NX, …) are the current Android options.
- So the main reason nxnative exists on iOS doesn't apply on Android: iOS bans JIT and blocks emulators from running guest code fast, while Android emulators can already run game code natively. What an nxnative-style Android port would still add:
  - shaders translated offline (no compile stutter);
  - per-game hand-tuned HLE (frame stepping, yield back-off, game-specific cuts);
  - no firmware or keys needed on the phone.
- **I found no Android port or fork of nxnative** (GitHub search, 2026-10-24).
- **Legal:** In February 2026 Nintendo sent DMCA notices against 13 Switch emulator projects (Eden, Citron, Kenji-NX, MeloNX, Ryujinx/Ryubing, Sudachi, Suyu, yuzu, …). The stated basis was decrypting games with `prod.keys` as circumvention of protection measures. nxnative/NX2Apple do exactly the same decryption at build time, so an Android port would carry the same exposure regardless of the "bring your own dump" framing.
- Licensing: nxnative is MIT; it vendors Ryujinx's shader translator (MIT). Borrowing yuzu/Eden code (GPLv3), such as their Vulkan BCn decoders, would make the port GPL.

## 5. Verdict and suggested path

**Feasible, but a multi-month project for someone fluent in Vulkan, ARM64 and Switch internals.** Most of the effort goes into the NVN → Vulkan replay backend. Suggested order, with the cheapest validation first:

1. **Linux/aarch64 host, headless.** Port `dm/host` off Mach APIs and switch the assembly to ELF syntax. Boot a rewritten game to `nnMain` under `adb shell` (or on an ARM64 Linux box). This proves the rewriter and HLE on Linux with no graphics work. *(~weeks)*
2. **First pixels via a Vulkan game.** Use Hades (profile `0100535012974000`), whose renderer is Vulkan, so `hle_vk` can mostly pass through to the phone's driver. Add a minimal GameActivity shell with AAudio and gamepad input.
3. **NVN → Vulkan backend.** Rewrite `nvn_mtl.m` against Vulkan 1.3 + descriptor indexing, with an explicit memory-pool strategy (host import where available, otherwise shadow + dirty tracking). Target one GPU family first: Adreno 7xx/8xx, ≥ 12 GB RAM. *(the bulk: months)*
4. **Shader chain → SPIR-V for Vulkan** (drop SPIRV-Cross/Metal), plus BCn decode, pipeline-cache prewarm and a Gradle/APK packager.
5. Widen GPU support (Mali, Xclipse) and add more profiles. The per-game profiles and rewriter cuts are reusable as-is, because they key on the game's build IDs, not on the host platform.

**Alternative if the goal is just "play my games on Android":** existing NCE emulators already cover that. The nxnative route makes sense mainly as a research or performance project: zero runtime shader translation and per-game tuning.

## 6. On-device results: Lenovo Legion Y700 (TB322FC), 2026-10-24

The device: **Snapdragon 8 Elite (SM8750P)**, 8 Oryon cores (6 × 3.53 GHz + 2 × 4.32 GHz), **Adreno 830** (Vulkan 1.3.284, Qualcomm driver of 02/2026), 12 GB RAM + 12 GB swap, **Android 16** (SDK 36), kernel 6.6 with 4 KB pages, SELinux enforcing.

The test programs are in `probes/`, built with NDK r27c. Raw output is in `build/*.txt`. They ran twice:

- from `adb shell` (the `shell` SELinux domain);
- **inside an ordinary app**: `build/apk/nx2a-probe.apk`, a Java-free NativeActivity (targetSdk 36, debuggable) built and v2-signed by `probes/app/mkapk.py` with no JDK or Android SDK. It ran in SELinux context `u:r:untrusted_app:s0:c60,c257,c512,c768`, the domain a real port would use. Results are in `build/app_results.txt`.

**Every result below was identical in both domains.**

### CPU / kernel (`probes/cpuprobe.c`)

| Check | Result | Consequence for the port |
|---|---|---|
| x18 kept across 200k syscalls, 1.8 s of preemption by 16 threads and SIGALRM every 0.5 ms | **Kept** (0 changes) | Unlike Apple, the kernel never clobbers x18. The rewriter's x18 rewrite could become optional, as long as host code at the gates doesn't depend on x18 (ShadowCallStack). Keep it at first; dropping it is an optimisation |
| `tpidr_el0` | Holds bionic's TLS pointer; a foreign value survives context switches | The rewrite is still needed (bionic and signal handlers need their TLS) |
| `tpidrro_el0` | Readable, value 0 | The rewrite to a thread slot is still needed (Horizon's TLR pointer) |
| `cntvct_el0` / `cntfrq_el0` | Readable; **19.2 MHz, the same as the Switch** | Guest `cntpct_el0` reads can be rewritten to a single `mrs cntvct_el0` instead of a host gate, which is cheaper than on Apple |
| `cntpct_el0` | **SIGILL** | The rewrite is required (nxnative already does it) |
| Misaligned `sp` access | **SIGBUS** (SA0 is set) | The rewriter's `sp` rounding is required (already there) |
| Address space | 39-bit user VA; the largest `PROT_NONE` reservation is **256 GB** | Enough for the 5 GB guest block (Tears of the Kingdom) and the 64 GB sparse pool (Monster Hunter Rise) |
| Executable memory | anon RW→RX, RWX, code inside a reserved block, a file-backed RX mapping **from the app's own `files/` dir**, and a memfd mapped RW and RX at once all **work** in `untrusted_app` | A host can `mmap` the rewritten `*.patched.bin` straight from app storage (Android's W^X rule for targetSdk ≥ 29 blocks `execve`, not `mmap(PROT_EXEC)`), or copy it into anonymous memory |

### GPU (`probes/vkprobe.c`)

- **BC1–BC7 sampled and filtered in hardware** (`textureCompressionBC` = yes), plus ASTC LDR/HDR and ETC2. No software texture decoding is needed on this GPU.
- Supported: geometry and tessellation shaders, `shaderInt64`, dual-source blend, logic op, depth clamp, wide lines, storage images without format, clip/cull distance, the full **descriptor indexing** set, **buffer device address** (with capture/replay), timeline semaphores, dynamic rendering, sync2, transform feedback, conditional rendering, custom border colours, provoking vertex, depth clip control, `index_type_uint8`, `robustness2`, `descriptor_buffer`, `push_descriptor` and `multi_draw`.
- Missing: `shaderFloat64`, `VK_EXT_external_memory_host`, `extended_dynamic_state3`, `graphics_pipeline_library` / `shader_object` (so expect pipeline compiles; nxnative's prewarm lists apply), `memory_budget`, mesh shaders, `map_memory_placed`.
- Limits: 2 GB max storage buffer range, 16.7M sampled images per stage (effectively bindless), 32 KB compute shared memory, subgroup size 64. The memory heap is 11 GB, device-local and host-visible, coherent and cached (unified memory).

### Zero-copy guest memory pools (`probes/vkzc.c`)

nxnative's Metal backend wraps the game's own memory as GPU buffers. Without host-pointer import, the test went the other way: allocate GPU-shareable memory and place it at a fixed guest address, inside an 8 GB `PROT_NONE` reservation, at 4 KB-aligned offsets.

| Method | Result |
|---|---|
| `vkMapMemory`, then `mremap(MREMAP_FIXED)` into the guest range | ✗ EINVAL (the KGSL mapping can't be moved) |
| Export as `OPAQUE_FD`, then `mmap(MAP_FIXED)` | ✗ `vkGetMemoryFdKHR` fails |
| **`AHardwareBuffer` (BLOB), dma-buf fd received via `AHardwareBuffer_sendHandleToUnixSocket`, `mmap(MAP_SHARED, MAP_FIXED)` at the guest address, AHB imported into Vulkan** | ✓ **Works.** CPU writes at the guest address → GPU copy → CPU reads at the guest address: 0 wrong words in 2 × 64 MB, with no flush or invalidate calls. A 64 MB GPU copy takes ~5.5 ms. CPU read/write speed through the mapping equals plain anonymous memory (16.5 vs 15.6 ms read, 2.4 vs 2.1 ms write per 64 MB), so it is **cached and coherent** |

How a backend would use it: at `nvnMemoryPoolInitialize(pool, ptr, size)`, allocate an AHB of `size`, copy the current contents of `[ptr, ptr+size)` into it, then `mmap(MAP_FIXED)` its dma-buf over that guest range. From then on the game's pointer *is* GPU memory. On finalize, map anonymous memory back with the contents. NVN pool sizes and addresses are already 4 KB-aligned, which matches the page size here.

GPU addresses still differ from CPU addresses, so the replay layer needs its pool-range → `VkDeviceAddress` lookup. nxnative already has one: `g_pools` in `nvn_mtl.m` keeps `buf.gpuAddress` per pool.

### What this device can't tell us

- A non-debuggable release build (the probe APK is `debuggable`; that normally doesn't change the SELinux domain, but it hasn't been checked).
- Memory limits under pressure: how much a foreground app can touch before the low-memory killer acts. Tears of the Kingdom needs > 3.4 GB resident.
- Other GPU families (Mali, Xclipse), which differ in BCn support and in how AHB memory is cached.
- Real game code. That needs your own dumps, run through `nx_extract` and `patch_nso.py`, which need .NET and Python on this PC.

## Sources

- [LNPANDY/NX2Apple](https://github.com/LNPANDY/NX2Apple) · [seedds/nxnative README](https://github.com/seedds/nxnative/blob/main/README.md)
- In-repo: `README.md`, `AGENTS.md`, `docs/development.md`, `docs/agents/03-bringup.md`, `dm/gfx/README.md`, `dm/tools/patch_nso.py`, `dm/host/*`, `dm/gfx/nvn/nvn_mtl.m`, `dm/gfx/vk/hle_vk.m`, `dm/gfx/tex/nvn_fmt_mtl.c`
- yuzu NCE on Android: [overclock3d](https://overclock3d.net/news/software/yuzu-switch-emulator-achieves-20-100-performance-boost-on-android/), [Emulation General Wiki – Switch emulators](https://emulation.gametechwiki.com/index.php/Nintendo_Switch_emulators)
- Feb 2026 DMCA wave: [Nintendo Life](https://www.nintendolife.com/news/2026/02/13-switch-emulators-hit-by-latest-wave-of-nintendo-takedowns), [VGC](https://www.videogameschronicle.com/news/nintendo-issues-dmca-takedown-on-more-switch-emulators-but-at-least-one-is-fighting-back/), [Android Authority](https://www.androidauthority.com/nintendo-switch-emulator-github-dmca-3640722/)
- Android developer verification: [Android Authority](https://www.androidauthority.com/android-developer-verification-rollout-sideloading-flow-3653395/), [9to5Google](https://9to5google.com/2026/08/18/google-gradually-rolling-out-androids-advanced-sideloading-ahead-of-developer-verification/), [Foresight Mobile](https://foresightmobile.com/blog/android-developer-verification-2026)
- BCn on mobile GPUs: [web3dsurvey (Android BC)](https://web3dsurvey.com/webgpu/features/texture-compression-bc/platform/Android), [Vortek internals (Mali)](https://kitemetric.com/blogs/vortek-internals-part-2-mastering-driver-workarounds-for-directx-gaming-on-mali)
- Host-memory import on Adreno: [Qualcomm developer forum](https://developer.qualcomm.com/forum/qdn-forums/software/adreno-gpu-sdk/68161)
