# Nintendo Switch 2 Welcome Tour (040057C02159A000): the rewriter's patches of this game's builds (dm/tools/patch_nso.py "Registered patches"). Bring-up observation gates only
# provisional (outside the repository, porter --experimental); this file holds only bring-up observation gates (OBS_REGS, dm.h) and is loaded by build ID
OBS = {
  # v262144 main: nn::bezel GPU memory block constructor (main+0x5e804c): after its CoreSystem::Allocate(size, align) call, x0 = the block (0 when the heap refused), x21 = the
  # size, x23 = the alignment. On the Odin it returned 0 for an 11 MB block a loader thread asked for 16 frames into the boot (abort at main+0x5e811c) at 792 MB RSS
  '6e3d11763a988c6b947510e0ae08ad00': [
    (0x5e80d8, 27 | 8 << 8, 'aa0003f6', 'GPU memory block: CoreSystem::Allocate result in x0, size x21, alignment x23 (OBS_REGS)'),
  ],
}

# How general each entry is (dm/tools/docs_patches.py "Binding"): these are observation gates, bound to the builds they read registers from
BINDING = [
    {'what': 'OBS_REGS gate at the Bezel GPU-block constructor', 'now': 'build', 'goal': 'build', 'purpose': 'runtime',
     'why': 'bring-up instrumentation (registers at a site, logged); the addresses are these builds\' own and the gates patch nothing'},
]

