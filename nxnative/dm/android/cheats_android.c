// Android port: the cheats interface of dm/host/hle_cheats.c (dm.h) with no cheats. The Mac file writes its state files on a libdispatch serial queue with blocks and aliases
// main's pages with mach_vm_remap; neither exists on Android (the compat layer has semaphores only). Every entry point reports no cheat file.
// TODO(upsync): port hle_cheats.c (a pthread worker for the file writes; the compat mach_vm_remap for the alias) so the menu's cheats work on Android too.
#include "dm.h"
void dm_cheats_init(void) {}
int dm_cheats_found(void) { return 0; }
int dm_cheats_ncheats(void) { return 0; }
const char *dm_cheats_build(void) { return ""; }
const char *dm_cheats_dir_path(void) { return ""; }
int dm_cheats_master(void) { return 0; }
void dm_cheats_master_set(int on) { (void)on; }
void dm_cheats_exit(void) {}
void dm_cheats_present(void) {}
