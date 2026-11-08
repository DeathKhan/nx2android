# Nintendo Switch 2 Welcome Tour (040057C02159A000): the rewriter's patches of this game's builds (dm/tools/patch_nso.py "Registered patches"). Bring-up observation gates only
# provisional (outside the repository, porter --experimental); this file holds only bring-up observation gates (OBS_REGS, dm.h) and is loaded by build ID
OBS = {
  # v262144 main: nn::bezel GPU memory block constructor (main+0x5e804c): after its CoreSystem::Allocate(size, align) call, x0 = the block (0 when the heap refused), x21 = the
  # size, x23 = the alignment. On the Odin it returned 0 for an 11 MB block a loader thread asked for 16 frames into the boot (abort at main+0x5e811c) at 792 MB RSS
  '6e3d11763a988c6b947510e0ae08ad00': [
    (0x5e80d8, 27 | 8 << 8, 'aa0003f6', 'GPU memory block: CoreSystem::Allocate result in x0, size x21, alignment x23 (OBS_REGS)'),
  ],
  # the SDK 20 sdk: nn::mem::StandardAllocator's page allocator (nlib TlsHeapCentral::AllocPages_), Bezel's 8.39 GB heap on this game. Where its refusal comes from:
  # the free-span search (x0 = the span, 0 = none of x20 pages), the lazy page commit (AllocPhy_(x1 = address, x2 = bytes)) and the one failure tail
  'd743946430d5252672cb63013b73a0ea': [
    (0x138b1c, 27 | 8 << 8, 'aa0003f5', 'TlsHeapCentral::AllocPages_: SearchFreeSpan_ result in x0, pages wanted in x20 (OBS_REGS)'),
    (0x138b40, 27 | 4 << 8, 'aa1303e0', 'TlsHeapCentral::AllocPages_: before AllocPhy_ (x1 address, x2 bytes) (OBS_REGS)'),
    (0x138f4c, 27 | 8 << 8, 'aa1f03f5', 'TlsHeapCentral::AllocPages_: failure tail, x0 = AllocPhy_ result or the null span page (OBS_REGS)'),
  ],
}
