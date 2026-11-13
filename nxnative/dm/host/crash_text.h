// The text a register points at, for the crash dump's `[dm] *xN text` lines (crash.c): UTF-16LE (Unreal Engine's TCHAR strings) or UTF-8, up to CRASH_TEXT_MAX
// bytes, ended by a NUL or by the first character that is not printable. The `*xN:` line's six words hold only the first 48 bytes: Unreal Engine 4's PhysX error
// callback had built its message in x1 before three games crashed at start-up (droidplay-core#389), and the dump kept its first 24 characters, a path's start.
// The whole message, `PHYSX: (<source path> <line>) <error code> : <text>`, runs to about 200 UTF-16 characters: 512 bytes hold it, 256 would cut its text.
// No I/O here and no faults: crash.c reads through vm_read_overwrite, and dm/tools/tests/test_crash_text.py builds this header alone.
#pragma once
#include <stddef.h>
#include <stdint.h>

#define CRASH_TEXT_MAX 512   // bytes read at a register's target
#define CRASH_TEXT_OUT (2 * CRASH_TEXT_MAX + 16)   // the quoted text: two output bytes at most for each byte read (an escape; UTF-16 makes three from two), the quotes, ` ...`, the NUL
// The shortest run taken for text. A pointer's own bytes can read as five printable characters (0x7161626364 reads "dcbaq" before its zero bytes), so a UTF-8 run
// needs six; UTF-16 needs a zero byte after each ASCII character, which no pointer has
#define CRASH_TEXT_MIN8 6
#define CRASH_TEXT_MIN16 4

// One UTF-8 character at b (n bytes left), when it is printable: its length, its code point in *cp. 0 for a NUL, a control character, a malformed or overlong
// sequence, a surrogate or one cut off by the end
static inline int crash_text_u8(const unsigned char *b, size_t n, uint32_t *cp) {
    uint32_t c = b[0];
    if (c < 0x80) { *cp = c; return (c >= 0x20 && c < 0x7f) || c == '\t' || c == '\n' || c == '\r'; }
    int len = c >= 0xf0 ? 4 : c >= 0xe0 ? 3 : c >= 0xc2 ? 2 : 0;
    if (!len || c > 0xf4 || (size_t)len > n) return 0;
    uint32_t v = c & (0x7fu >> len);
    for (int k = 1; k < len; k++) { if ((b[k] & 0xc0) != 0x80) return 0; v = v << 6 | (b[k] & 0x3f); }
    if (v < 0xa0 || (len == 3 && v < 0x800) || (len == 4 && (v < 0x10000 || v > 0x10ffff)) || (v >= 0xd800 && v < 0xe000)) return 0;   // a C1 control, overlong, a surrogate
    *cp = v; return len;
}

// One UTF-16LE character at b (n bytes left), when it is printable: the bytes it takes (2, or 4 for a surrogate pair), its code point in *cp; 0 otherwise
static inline int crash_text_u16(const unsigned char *b, size_t n, uint32_t *cp) {
    if (n < 2) return 0;
    uint32_t u = b[0] | (uint32_t)b[1] << 8;
    if (u < 0x80) { *cp = u; return (u >= 0x20 && u < 0x7f) || u == '\t' || u == '\n' || u == '\r' ? 2 : 0; }
    if (u < 0xa0 || u >= 0xfffe || (u >= 0xdc00 && u < 0xe000)) return 0;   // a C1 control, a noncharacter, a low surrogate alone
    if (u >= 0xd800 && u < 0xdc00) {   // a high surrogate: printable with a low one after it
        uint32_t lo = n >= 4 ? (b[2] | (uint32_t)b[3] << 8) : 0;
        if (lo < 0xdc00 || lo >= 0xe000) return 0;
        *cp = 0x10000 + ((u - 0xd800) << 10) + (lo - 0xdc00); return 4;
    }
    *cp = u; return 2;
}

// The run of printable characters at the start of b: its bytes; its characters in *chars, the ASCII ones among them in *ascii
static inline size_t crash_text_run(const unsigned char *b, size_t n, int wide, int *chars, int *ascii) {
    size_t at = 0; uint32_t cp = 0; int k;
    *chars = *ascii = 0;
    while (at < n && (k = wide ? crash_text_u16(b + at, n - at, &cp) : crash_text_u8(b + at, n - at, &cp))) { at += (size_t)k; ++*chars; *ascii += cp < 0x80; }
    return at;
}

// Reads up to CRASH_TEXT_MAX bytes at a into b, through rd (which fails instead of faulting), in pieces that each stay within one 4 KiB page: a string near the end
// of its mapping keeps its readable start, where one read of the whole range fails (the Mac's vm_read_overwrite) or comes back short (Android's process_vm_readv,
// dm_android.h). Returns the bytes read
static inline size_t crash_text_read(uint64_t a, unsigned char *b, int (*rd)(uint64_t, void *, size_t)) {
    size_t n = 0;
    while (n < CRASH_TEXT_MAX) {
        size_t k = 4096 - (size_t)((a + n) & 4095);
        if (k > CRASH_TEXT_MAX - n) k = CRASH_TEXT_MAX - n;
        if (!rd(a + n, b + n, k)) break;
        n += k;
    }
    return n;
}

// The text at the start of b (n bytes read), as a quoted C string literal in out (cap bytes; CRASH_TEXT_OUT holds any), with ` ...` after the quote when it runs to the
// end of what was read. UTF-16LE (*wide = 1) when b's address is 2-byte aligned (aligned2) and more than half the run's characters are ASCII: ASCII text read as
// UTF-16 makes CJK characters with no ASCII among them (half, from short strings four bytes apart: "abc\0abc\0"), and UTF-16 text read as UTF-8 ends at its first
// zero byte. Returns its characters; 0 when b starts with no text
static inline int crash_text_format(const unsigned char *b, size_t n, int aligned2, char *out, size_t cap, int *wide) {
    int chars = 0, ascii = 0; size_t len = 0, o = 0;
    *wide = 0;
    if (cap) out[0] = 0;
    if (cap < 8) return 0;
    if (aligned2) { len = crash_text_run(b, n, 1, &chars, &ascii); *wide = chars >= CRASH_TEXT_MIN16 && 2 * ascii > chars; }
    if (!*wide) { len = crash_text_run(b, n, 0, &chars, &ascii); if (chars < CRASH_TEXT_MIN8) return 0; }
    out[o++] = '"';
    int shown = 0;
    for (size_t at = 0; at < len && o + 4 + 6 <= cap; shown++) {   // 4: one character's UTF-8; 6: `" ...` and the NUL
        uint32_t cp = 0;
        at += (size_t)(*wide ? crash_text_u16(b + at, len - at, &cp) : crash_text_u8(b + at, len - at, &cp));
        if (cp == '"' || cp == '\\') { out[o++] = '\\'; out[o++] = (char)cp; }
        else if (cp == '\t' || cp == '\n' || cp == '\r') { out[o++] = '\\'; out[o++] = cp == '\t' ? 't' : cp == '\n' ? 'n' : 'r'; }
        else if (cp < 0x80) out[o++] = (char)cp;
        else if (cp < 0x800) { out[o++] = (char)(0xc0 | cp >> 6); out[o++] = (char)(0x80 | (cp & 0x3f)); }
        else if (cp < 0x10000) { out[o++] = (char)(0xe0 | cp >> 12); out[o++] = (char)(0x80 | (cp >> 6 & 0x3f)); out[o++] = (char)(0x80 | (cp & 0x3f)); }
        else { out[o++] = (char)(0xf0 | cp >> 18); out[o++] = (char)(0x80 | (cp >> 12 & 0x3f)); out[o++] = (char)(0x80 | (cp >> 6 & 0x3f)); out[o++] = (char)(0x80 | (cp & 0x3f)); }
    }
    out[o++] = '"';
    if (len == n || shown < chars) { out[o++] = ' '; out[o++] = '.'; out[o++] = '.'; out[o++] = '.'; }
    out[o] = 0;
    return shown;
}
