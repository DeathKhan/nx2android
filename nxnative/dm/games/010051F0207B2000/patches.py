# Tomodachi Life: Living the Dream (010051F0207B2000): the rewriter's patches of this game's own builds. dm/tools/patch_nso.py loads every profile's patches.py and applies those of
# a module's NSO build ID; CUTS as described there ("Registered patches"). Patches of Nintendo SDK module builds are in dm/tools/sdk_patches.py
CUTS = {
  # 1.0.0 (v0) main. Model initialisation runs out of two kinds of fixed-size child heaps natively. A LowPriorityJobExecuter thread lazily builds a model view's material
  # blocks (main+0x328f00: 12 blocks x (0x28 + 2 x 0x80) + 0x40 = 0xe20 bytes) from the model's frame heap; the heap is exactly full, the allocator (main+0x1e6f40) returns
  # null and main+0x328f5c stores through it: SIGSEGV at 0x10 on the loading screen (Odin 3, 2026-11-14: at 2 s, or after about 2 minutes with core preemption on).
  # MacNativeGames' Tomodachi port (games/tomodachi/runtime/game_patches.c) ships the same three words for the same crash on the same build. Doubling only the first heap
  # moved the crash to a 32 KiB cloth model heap 37 s later (0x7550 used, head = tail): the second pair of sites
  '56bf85bd535413464cb75bb6c2683b67': [
    (0x580434, 'e02f1b32', 'e0331b32', 'MiiBodyBaseModel child heap: mov w0,#0x1ffe0 -> #0x3ffe0 (128 -> 256 KiB)'),
    (0x57b9fc, '03009052', '2300a052', 'cloth model child heap (first class): mov w3,#0x8000 -> #0x10000 (32 -> 64 KiB)'),
    (0x57ba70, '03009052', '2300a052', 'cloth model child heap (second class): mov w3,#0x8000 -> #0x10000 (32 -> 64 KiB)'),
  ],
}

# How general each patch is and could be (dm/tools/docs_patches.py "Binding", listed in docs/agents/game-patches.md)
BINDING = [
    {'what': 'Mii body and cloth model child heaps doubled', 'now': 'build', 'goal': 'unreviewed', 'purpose': 'memory',
     'why': "the game's own fixed heap sizes; why a native run needs more than the Switch (a size or alignment the host answers larger) is not found yet",
     'checked': "original bytes asserted on 1.0.0 main, the only build on hand; the same three words as MacNativeGames' port of 1.0.0"},
]
