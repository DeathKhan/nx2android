#!/usr/bin/env python3
"""Camelot's archive (Mario Tennis Fever): the whole game in one RomFS file, rom:/42, magic 'CRA.'. Its files are scrambled, and their index is built into the game's code in an
obfuscated form. Neither is read here: nup2android has no de-obfuscation code, so shader_split.py leaves the archive out with a message (cra_jobs).
What this module still reads is what such games keep in the clear: NARC containers (the DS's format: BTAF offsets, BTNF names, GMIF data; narc_members), each member packed by the kind in its
first byte (undo: LZ4, zstd, zlib, Camelot's LZ). Inside: FRES, FSHA, BNTX, VFXB, SARC."""
import struct, zlib
import lz4.block


class Error(ValueError): pass


def narc_members(d, where='NARC'):
    if d[:4] != b'NARC': raise Error(f'{where}: not a NARC')
    o, n = struct.unpack_from('<HH', d, 0xc); secs = {}
    for _ in range(n):
        if o + 8 > len(d): raise Error(f'{where}: a section past the end')
        s = struct.unpack_from('<I', d, o + 4)[0]; secs[d[o:o + 4]] = o; o += s
    if b'BTAF' not in secs or b'GMIF' not in secs: raise Error(f'{where}: no BTAF or GMIF')
    fat = secs[b'BTAF']; data = secs[b'GMIF'] + 8
    out = []
    for i in range(struct.unpack_from('<I', d, fat + 8)[0]):
        a, b = struct.unpack_from('<II', d, fat + 12 + 8 * i)
        if not data + a <= data + b <= len(d): raise Error(f'{where}: member {i} past the end')
        out.append(d[data + a:data + b])
    return out


def lz_undo(d, at):   # kinds 0x50 and 0x60 (nro0+0x1237f04): flag bytes, each negated and read from its high bit (1 a match, 0 a literal); a flag byte 0 is 8 literals; a distance 0 ends
    out = bytearray(); n = len(d)
    while at < n:
        f = d[at]; at += 1
        if not f:
            out += d[at:at + 8]; at += 8; continue
        f = -f & 0xff
        for bit in range(7, -1, -1):
            if not f >> bit & 1:
                out.append(d[at]); at += 1; continue
            lo, hi = d[at], d[at + 1]
            dist = lo >> 4 | hi << 4
            if not dist: return bytes(out)
            ln = lo & 0xf
            if ln > 1: at += 2
            elif ln == 1: ln = d[at + 2] + 0x11 + (d[at + 3] << 8) + 0xff; at += 4
            else: ln = d[at + 2] + 0x10; at += 3
            s = len(out) - dist
            if s < 0: raise Error('an LZ distance before the start')
            for k in range(ln): out.append(out[s + k])
    return bytes(out)


def zstd_mod():
    try: from compression import zstd
    except ImportError: from backports import zstd   # pyright: ignore[reportMissingImports]  # an optional package, only Python before 3.14 needs it
    return zstd


def undo(d):
    """A member, unpacked by the kind in its first byte, as the game does (nro0+0x1237c80). Anything else is the member itself"""
    if not d: return d
    k = d[0]
    if k == 0x4d and d[:4] == b'MMMM':   # chunks of any kind: u32 total, u32 count, u32 length of each, 16-aligned
        total, n = struct.unpack_from('<II', d, 4)
        at = (12 + 4 * n + 15) & ~15
        out = []
        for i in range(n):
            s = struct.unpack_from('<I', d, 12 + 4 * i)[0]
            out.append(undo(d[at:at + s])); at += (s + 15) & ~15
        return b''.join(out)
    if k == 0x94 and len(d) >= 8: return lz4.block.decompress(d[8:], uncompressed_size=struct.unpack_from('<I', d, 4)[0])
    if k == 0x5a and d[:4] == b'Z\0\0\0': return zstd_mod().decompress(d[8:])
    if k == 0x50 and len(d) >= 8: return lz_undo(d, 8)
    if k == 0x60 and len(d) >= 4: return lz_undo(d, 4)
    if k in (0x70, 0x80) and len(d) >= 0x10: return d[0x10:]
    if k in (0, 0x10, 0x20, 0x30) and len(d) >= 0x10:
        try: return zlib.decompress(d[0x10:], -15 if k == 0x30 else 47)
        except zlib.error: return d   # a member that only starts with such a byte
    return d
