# Nintendo Switch 2 Welcome Tour (`040057C02159A000`)

Update 1.0.4 (v262144), SDK 20.5.29 built for the Switch 2 (`sdk` spec `Ounce`), NVN2 with Ampere shaders. Bezel Engine (as Super Mario Party Jamboree), modules `main`, `subsdk0`, `subsdk1`, `subsdk3`, `sdk` and 36 NROs. Brought up on Android (AYN Odin 3, 2026-11-07) from the NSP script's provisional profile, not on the Mac or an iPhone; the Android keys are in [android.tsv](android.tsv).

## What it needs

| Setting | Where | Why |
|---|---|---|
| `sys_resource 0x1000000`, `mem_alias 0x1100000000`, `mem_reserve 0x1360000000`, `mem_map_limit on` | [`game.tsv`](game.tsv) | the virtual-address-memory path, as on the console: Bezel probes a 63 GB address region before its 56 GB allocator region, so the alias region is 68 GB; the heap path (`sys_resource 0`) made the game reserve its 8.39 GB heap whole in arenas and its loader ran dry 16 frames in. `mem_map_limit` refuses mapping past the 9 GB pool as the hardware does |
| `app_pool 0x240000000` | `game.tsv` | the Switch 2's 9 GB |
| `nvnDeviceCalculateAccelerationStructureMemoryRequirements` answered | `dm/gfx/nvn/hle_nvn.c` | the red stub left the game's three result words uninitialised; their sum (0x100000001) became a 4 GB "acceleration structure pool" committed before the first frame |
| `vi::InitializeWithVrrLfcEnabled`, the refresh-rate queries (RefreshRate is one 32-bit Hz word), `SetDisplay120HzRefreshRateCapability` | `dm/gfx/hle_vi.c` | Switch 2 vi calls the host had not taken over (null service locks) |
| `hid::GetJcmsState`, `GetMousePointerSpeedScale` (a float), `GetGlGrButtonConfigState`, `SetSceneIdForPrivateApp`; `oe::QueryApplicationGpuTimeSpan`, `gs::sdk::CreateNativeMemory` | `dm/host/hle_hid.c`, `hle_oe.c` | the same for hid, oe and gs |
| OBS_REGS gates in [patches.py](patches.py) | `dm/tools/patch_nso.py` | bring-up instrumentation: registers at the Bezel GPU-block constructor (logged for the first hits and every null result) |

## Shaders

6,814 jobs from 6,766 shaders (pixel 5,630, vertex 1,031, compute 80, geometry 12, mesh 7, domain 4, hull 1, task 1); 6,812 translate as Ampere (the two left are bindless compute shaders). 27 pixel shaders read a fragment input per vertex (Ampere LDTRAM, the edge effects of the tech demos): on a device without `VK_KHR_fragment_shader_barycentric`, as the Odin's stock driver, their pipelines cannot build yet. The 8 task and mesh shaders translate as compute kernels and are not drawn yet (no `VK_EXT_mesh_shader`).

## Status

Boots to the title screen (the 3D intro at 27-40 fps, the title at 40 fps). Open: a low-memory kill about 44 s in (the device's MemAvailable falls by 6 GB while the process RSS stays at 1.4 GB: the GPU backing of the 6.5 GB the game maps in its first seconds), the 4K H.264 intro movie at 30 fps (three CPU copies a frame), the task / mesh and per-vertex shaders above, and the ray-tracing entry points still answered by the red stub.
