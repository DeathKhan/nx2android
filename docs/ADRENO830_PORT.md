# Switch 2 games on the Adreno 830: the port reworked for the chip

A Switch 2 game was written for an NVIDIA Ampere GPU (T239: 12 SMs, warp 32, 255 registers a thread, mesh shaders, immediate-mode rendering) behind NVN2. The Odin 3 has an Adreno 830 (wave 64, 192 registers a thread with occupancy falling as they fill, no mesh hardware, a tiler, 16-bit math at twice the rate). Today the port translates the game's Ampere code to SPIR-V once, generically, and the Vulkan driver and our host sort out the rest at run time. This document reworks the port so each layer does what suits this chip:
- **The translator** shapes shaders for the 830.
- **Our Turnip**, shipped in the APK, is a driver built for these games.
- **The host** stops paying for the stock driver's bugs.

It is grounded in Ocarina of Time measured on the Odin on 2026-11-14 (Kokiri Forest, warm caches, GPU ms per frame).

## Where we are

| | Stock driver | Our Turnip |
|---|---|---|
| Frame GPU | 35.8 | 47.7 |
| Mesh emulation (grass) | 12.8 | 12.3 |
| Rest of the frame | 23.0 | 35.4 |
| … compute | 10.3 | 16.4 |
| … render passes | 13.0 | 20.0 |
| Host CPU, replay + recorder threads | 25 + 18 | 27 + 16 |

The frame is limited by the CPU about as much as by the GPU: frame time is about 45–70 ms on both drivers.

OoT's 43,310 shaders (inventory: `inventory.py` over `shaders.dmsv`):
- **Size:** fragment median 4,766 SPIR-V instructions, compute up to 539,619.
- **Mesh and task:** all 481 mesh and 20 task shaders have 32-thread workgroups.
- **Precision:** no 16-bit math anywhere.
- **Addressing:** about 2,250 shaders compute with 64-bit memory addresses.
- **Warp operations:** 41,600 shuffles and ballots with 32-lane semantics.
- **Shared memory:** at most 25.7 KB.

## Principles
1. **One target, decided at port time.** A port is built for a device class (here the Odin 3: Adreno 830) and the driver it ships with (our Turnip build). Everything that can be decided before the game runs is decided in the porter.
2. **The driver is part of the port.** Our Turnip ships in the APK, so it may have private features that NVN games need and Vulkan lacks. The host uses them only when it is running on that driver (`V.turnip`).
3. **No work for another driver's bugs.** Workarounds for Qualcomm's driver run only on Qualcomm's driver. This is done (b44d70996).
4. **Every change is measured** with the same harness, against stock, in the same scene. A change that does not move the number is reverted or left off.

## The rework, by layer

### A. Translation (porter, `shader_vk.py` and a new SPIR-V pass after the translator)
- **A1. Two mesh workgroups per wave.** NVN2's 32-thread mesh and task workgroups run as 64-thread kernels, the upper half being the next workgroup:
  - workgroup id = 2·id + (local id ≥ 32)
  - shared memory doubled and indexed by half (the shaders use at most 404 bytes)
  - warp operations clustered to 32 lanes
  - the launch count halved
  
  This fills Adreno's 64-wide waves, which the emulation currently runs half-empty. It is the main lever for beating stock: 12.5 ms of grass, on both drivers.
- **A2. Relaxed precision.** Ryujinx's translator turns Ampere's half-precision instructions (HADD2/HFMA2/HMUL2) into 32-bit math around pack/unpack. The pass marks those values, and the colour maths that feeds them, as RelaxedPrecision. ir3 then computes them in 16-bit registers at twice the rate, which also lowers register pressure (see A4).
- **A3. 32-bit addressing.** Raw global memory goes through the pool table with 64-bit address arithmetic. Where an address provably stays inside one pool chunk, the pass rewrites it as base plus a 32-bit offset (about 2,250 shaders).
- **A4. A register budget at port time.** The porter compiles every shader with our Turnip's own compiler on Arch (Mesa's drm-shim for the A830) and gets each shader's registers, waves and spills. Spilling shaders get A2 and A3 applied first, then rematerialised address maths. The port reports what still spills.
- **A5. Pipelines compiled at port time.** The same offline compiler builds every shader's pipeline-library parts into the pipeline cache that ships in the data pack. Today the first launch spends 15–60 minutes in a warm-up pass and shaders compile in play until it finishes. With this, first launch has no compiling: no hitch, first frame or thousandth.

### B. Driver (our Turnip, `~/Projects/_worktrees/mesa/claude-trueup`)
Already done today:
- scratch memory sized for the A830 (fixes OoT's world-load GPU faults)
- A830 compute shared memory set to 32 KiB, as in Mesa main
- UBO range coalescing back to the A810 only, as its author scoped it
- in-pass dispatch, buffer update and copy (opt-in)

To do:
- **B1. Small passes that are slow on Turnip.** A 768×432 pass with 6 draws (6.3 ms), a 768×432 pass with no draws (3.6 ms) and a 384×216 pass (1.3 ms): about 11 ms. Find each one's shaders, formats and render mode, and fix the driver path. This is the biggest single item.
- **B2. Register allocation and spilling for big compute shaders.** Four shaders cost 5 ms more than on stock (7–10×). Record stock's compiled code for them by extending the KGSL recorder to dump what the compute shader state points to, then compare register counts and spill code with ir3. Improve ir3's register allocator and spill choices for these shapes.
- **B3. NVN texture and sampler pools as a descriptor heap.** Turnip has descriptor buffers. The pool tables become one GPU buffer the game's handles index directly, so the per-frame descriptor writes (2,100–3,600 a frame on Turnip, against 1,360 on stock) go away.
- **B4. Dynamic constant-buffer offsets with independent sets.** Turnip drew wrong with them, so the host turned them off under pipeline libraries. Fixed, the host goes back to dynamic offsets: fewer descriptor writes, and CPU time on every draw.
- **B5. NVN barriers.** Today every game barrier ends the pass and drains the GPU. A barrier that carries NVN's bits (order primitives, order indirect data, invalidate texture/shader/descriptor) lets the driver flush only what each bit names.

### C. Host runtime (`nxnative/dm/android`)
- **C1. CPU per frame.** The replay thread takes 25–27 ms and the recorder 16–18 ms. Targets:
  - descriptor writes, through B3 and B4
  - state already set (bind caches across passes on Turnip: done)
  - fewer submissions per frame (about 28 today)
  - NVN barrier handling, through B5
- **C2. Mesh emulation** to consume A1's packed kernels: halve the launch counts, and narrow barriers between its dispatches (done, DM_VK_MESH_BAR).
- **C3. Measurement.** The harness from today moves into `devtools/`:
  - world loads with the screensaver held off
  - stock against Turnip, with warm caches
  - per-shader GPU time (DM_VK_PASS_TIMES)
  - driver traces (MESA_GPU_TRACES)
  - compiler statistics (DM_VK_PIPE_STATS)
  - recordings of the stock driver (DM_KGSL_CAPTURE)

## Order of work and targets (OoT, Kokiri)
| Phase | Work | Target |
|---|---|---|
| 1 | B1, then B2 | Turnip's non-grass frame from 35 → 23 ms: parity with stock |
| 2 | A1 + C2 | Grass 12.5 → ~7 ms on both drivers: Turnip ahead of stock |
| 3 | B3, B4, B5, C1 | Host CPU down to the GPU's time: 30 fps held in Kokiri |
| 4 | A2, A3, A4 | Spill-free hot shaders; GPU headroom for 40 fps |
| 5 | A5 | No compiling at all on the device |
| 6 | Other Switch 2 games (Welcome Tour, Donkey Kong Bananza), then Switch 1 | |

## How each change is checked
- **Picture:** a screenshot in the same scene against the previous build, plus a world load with no GPU faults (`loop.sh`, four loads for anything touching memory).
- **Number:** GPU and frame-time medians from warm-cache runs (`one.sh`).
- **Driver changes:** also run the Vulkan CTS group they touch (compute, subgroups, pipeline libraries) on the Odin before they become the default.
