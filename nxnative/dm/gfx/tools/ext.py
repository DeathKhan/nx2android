#!/usr/bin/env python3
# Optional external reader add-ons. nup2android holds no key handling, no decryption and no de-obfuscation code, and none ships with it: an archive stored encrypted, masked or
# scrambled is reported and left closed (uepak.py, rpf6.py, cpk.py, shader_split.py). Those places ask here first whether an add-on the user installed on their own reads the
# archive instead. With no add-on (the default) hook() returns None and every reader behaves as it does without this file.
# NUP2ANDROID_ADDONS: Python files, or folders holding an addon.py, separated like PATH. A module's public functions are its hooks, found by name; the first module that has one wins.
# The names asked for (each optional; "token" is whatever the add-on returned, passed back unread and kept picklable, as jobs carry it to worker processes):
#   cpk_utf(table) -> bytes                           a CPK @UTF table that does not start with '@UTF' (cpk.utf_at)
#   arh_tables(index, word, strings, nodes) -> (bytes, bytes)   an ARH1 index whose header word is not the format's constant; strings and nodes are (offset, length) (shader_split.arh_entries)
#   pak_keys(mains) -> object                         called with a pak path -> token, or None for a pak that needs none; has a list .found (uepak.keys)
#   pak_read(token, data) -> bytes                    data read from a pak or container marked encrypted, padded to 16 bytes (uepak.Pak, uepak.Toc)
#   rpf_key(main, archive) -> token or None           an RPF6 archive whose header marks its tables encrypted (shader_split.rpf_jobs)
#   rpf_read(token, data) -> bytes                    such an archive's tables, or a file of it that reads as neither a zstd frame nor a zlib stream (rpf6.Rpf)
#   cra_files(nros) -> [(…, name, length, …), …] or None   a CRA archive's files, from the NROs captured at a boot (shader_split.cra_jobs)
#   cra_head(file, entry) -> bytes, cra_read(file, entry) -> bytes   the start of one such file, and all of it (shader_split.cra_shaders)
#   game_container(folder, say) -> path or None       a folder that holds no NUP container: the .nup file to pack for it; say prints a line (build_ipa_from_nsp.nup_file, pipeline.scan)
import importlib.util, os, sys

_mods = None

def modules():
    """The add-on modules NUP2ANDROID_ADDONS names, loaded once per process; a name that is no file and no folder with an addon.py is an error, not skipped in silence"""
    global _mods
    if _mods is None:
        _mods = []
        for i, p in enumerate(x for x in os.environ.get('NUP2ANDROID_ADDONS', '').split(os.pathsep) if x):
            f = os.path.join(p, 'addon.py') if os.path.isdir(p) else p
            if not os.path.isfile(f): raise FileNotFoundError(f'NUP2ANDROID_ADDONS: {p}: no Python file and no folder with an addon.py')
            d = os.path.dirname(os.path.abspath(f))
            if d not in sys.path: sys.path.append(d)   # an add-on may import the files beside it
            spec = importlib.util.spec_from_file_location(f'nup2android_addon_{i}', f)
            m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); _mods.append(m)
    return _mods

def hook(name):
    """The first add-on's function of that name, or None (always None without NUP2ANDROID_ADDONS)"""
    if name.startswith('_'): return None
    for m in modules():
        f = getattr(m, name, None)
        if callable(f): return f
    return None
