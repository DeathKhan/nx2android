// Android port: AV1 for the Switch 2 SDK's nn::codec decoders (hle_codec_video.c) on MediaCodec, VideoToolbox's and dav1d's counterpart (dm/host/hle_movie_av1.c).
// Ocarina of Time's title and menus sit over an AV1 movie (NintendoSDK_movie through nn::codec, 1920x1080, output format 3); Fire Emblem Fortune's Weave's 82 movies are AV1 too. Without a
// decoder every packet gave nothing and the movie stayed black ("end of stream after 0 pictures").
// The decoding is movie_avc.c's MediaCodec machinery in an AV1 mode (mavc_new_av1, mime video/av01): hle_codec_video.c drives it exactly as the H.264 decoder, so AV1 pictures get the same
// presentation-order queue, end-of-stream drain, timestamps and zero copy (the codec renders into an AImageReader, Vulkan samples the buffer as Y'CbCr into the game's native-image texture;
// a game that reads its pictures on the CPU gets compact NV12 copies, kept as I420 as the Mac's are). The decoder: the platform's first for video/av01, the hardware one
// (c2.qti.av1.decoder on the Odin 3), else c2.android.av1.decoder when the hardware one refuses its configuration, gives no picture for 64 packets or reports an error (from the next
// sequence header on), or with DM_AV1_SW=1; DM_AV1_DECODER=<name> names one. The codec is configured from the stream's sequence header (csd-0: its av1C record, ISO-BMFF AV1 binding 2.3.3;
// DM_AV1_CSD=0 leaves it out) and opened again when the header changes; packets before the first sequence header are skipped. Packets go in one temporal unit per input, in the
// low-overhead OBU format (each OBU with its size) that WebM / MP4 samples and the SDK's demuxer keep; an Annex B or IVF-framed packet is rewritten into it (mav1_tu), temporal delimiters
// and padding left out as the MP4 binding has samples.
// This file is the OBU side (no Android: dm/android/tests/t_movie_av1.c tests it on the host). hle_movie_av1.h's synchronous mav1_new / mav1_decode are not built here: MediaCodec gives
// pictures a few packets after their input, which hle_codec_video.c's H.264 path already handles. dav1d (dm/third_party/dav1d) is not built for Android: the platform's own software
// decoder is the fallback
#include "movie_av1.h"
#include "hle_movie_av1.h"
#include <string.h>

const char *mav1_backend(const mav1 *a) { (void)a; return "MediaCodec video/av01"; }

typedef struct { const uint8_t *p; size_t n, bit; int bad; } av1_bits;
static uint32_t rb(av1_bits *b, unsigned n) {
    uint32_t v = 0;
    if (b->bit + n > b->n * 8) { b->bad = 1; return 0; }
    while (n--) { v = v << 1 | ((b->p[b->bit / 8] >> (7 - b->bit % 8)) & 1); b->bit++; }
    return v;
}
static void uvlc(av1_bits *b) { unsigned z = 0; while (!b->bad && !rb(b, 1)) if (++z >= 32) return; if (z < 32) rb(b, z); }
static size_t leb128(const uint8_t *p, size_t n, uint64_t *v) {
    *v = 0;
    for (size_t i = 0; i < n && i < 8; i++) { *v |= (uint64_t)(p[i] & 0x7f) << (i * 7); if (!(p[i] & 0x80)) return i + 1; }
    return 0;
}
static size_t put_leb128(uint8_t *o, uint64_t v) { size_t i = 0; do { o[i] = (uint8_t)(v & 0x7f); v >>= 7; if (v) o[i] |= 0x80; i++; } while (v); return i; }

// The sequence header read as far as the colour config (AV1 5.5): hle_movie_av1.c's mav1_av1c, which cannot be built here (VideoToolbox), plus the colour description
int mav1_seq_parse(const uint8_t *obu, size_t n, uint8_t c[4], mav1_seq *s) {
    if (n < 2 || (obu[0] & 0x80) || (obu[0] >> 3 & 15) != 1 || !(obu[0] & 2)) return 0;   // a sequence header with a size field
    size_t at = obu[0] & 4 ? 2 : 1; uint64_t size;
    size_t l = at < n ? leb128(obu + at, n - at, &size) : 0;
    if (!l || size > n - at - l) return 0;
    av1_bits b = {obu + at + l, (size_t)size, 0, 0};
    unsigned profile = rb(&b, 3); rb(&b, 1); unsigned reduced = rb(&b, 1), level = 0, tier = 0;
    unsigned decoder_model = 0, delay_len = 0;
    if (reduced) level = rb(&b, 5);
    else {
        if (rb(&b, 1)) {   // timing_info
            rb(&b, 32); rb(&b, 32); if (rb(&b, 1)) uvlc(&b);
            if ((decoder_model = rb(&b, 1))) { delay_len = rb(&b, 5) + 1; rb(&b, 32); rb(&b, 5); rb(&b, 5); }
        }
        unsigned display_delay = rb(&b, 1), points = rb(&b, 5) + 1;
        for (unsigned i = 0; i < points && !b.bad; i++) {
            rb(&b, 12); unsigned lv = rb(&b, 5), t = lv > 7 ? rb(&b, 1) : 0;
            if (!i) { level = lv; tier = t; }
            if (decoder_model && rb(&b, 1)) { rb(&b, delay_len); rb(&b, delay_len); rb(&b, 1); }
            if (display_delay && rb(&b, 1)) rb(&b, 4);
        }
    }
    unsigned wb = rb(&b, 4) + 1, hb = rb(&b, 4) + 1;
    int w = (int)rb(&b, wb) + 1, h = (int)rb(&b, hb) + 1;
    if (!reduced && rb(&b, 1)) { rb(&b, 4); rb(&b, 3); }   // frame ids
    rb(&b, 3);   // 128x128 superblocks, filter intra, intra edge filter
    if (!reduced) {
        rb(&b, 4);   // inter-intra, masked compound, warped motion, dual filter
        unsigned order_hint = rb(&b, 1);
        if (order_hint) rb(&b, 2);   // jnt_comp, ref_frame_mvs
        unsigned tools = rb(&b, 1) ? 2 : rb(&b, 1);
        if (tools && !rb(&b, 1)) rb(&b, 1);   // force_integer_mv
        if (order_hint) rb(&b, 3);
    }
    rb(&b, 3);   // superres, cdef, restoration
    unsigned hbd = rb(&b, 1), twelve = profile == 2 && hbd ? rb(&b, 1) : 0, mono = profile == 1 ? 0 : rb(&b, 1), cp = 2, tc = 2, mc = 2, ssx = 1, ssy = 1, pos = 0, range = 0;
    if (rb(&b, 1)) { cp = rb(&b, 8); tc = rb(&b, 8); mc = rb(&b, 8); }
    if (mono) range = rb(&b, 1);
    else if (cp == 1 && tc == 13 && mc == 0) { ssx = ssy = 0; range = 1; }   // sRGB
    else {
        range = rb(&b, 1);
        if (profile == 1) ssx = ssy = 0;
        else if (profile == 2) { if (twelve) { ssx = rb(&b, 1); ssy = ssx ? rb(&b, 1) : 0; } else { ssx = 1; ssy = 0; } }
        if (ssx && ssy) pos = rb(&b, 2);
    }
    if (b.bad) return 0;
    if (c) {
        c[0] = 0x81;   // marker, version 1
        c[1] = (uint8_t)(profile << 5 | (level & 31));
        c[2] = (uint8_t)(tier << 7 | hbd << 6 | twelve << 5 | mono << 4 | ssx << 3 | ssy << 2 | pos);
        c[3] = 0;
    }
    if (s) {   // matrix coefficients (AV1 6.4.2): 1 BT.709, 5 / 6 BT.601, 9 / 10 BT.2020; unspecified (2) or anything else as the height says
        int m = mc == 1 ? 2 : mc == 5 || mc == 6 ? 1 : mc == 9 || mc == 10 ? 5 : h < 720 ? 1 : 2;
        *s = (mav1_seq){w, h, (int)profile, (int)level, (int)tier, twelve ? 12 : hbd ? 10 : 8, (int)mono, m, (int)range, (int)reduced};
    }
    return 1;
}
int mav1_av1c(const uint8_t *obu, size_t n, uint8_t c[4], int *w, int *h) {
    mav1_seq s; if (!mav1_seq_parse(obu, n, c, &s)) return 0;
    *w = s.w; *h = s.h; return 1;
}

// p holds OBUs in the low-overhead format: each with a size field, laid end to end over exactly n bytes
static int is_obu_type(unsigned t) { return (t >= 1 && t <= 8) || t == 15; }
static int low_overhead(const uint8_t *p, size_t n) {
    if (!n) return 0;
    for (size_t i = 0; i < n;) {
        if ((p[i] & 0x80) || !(p[i] & 2) || !is_obu_type(p[i] >> 3 & 15)) return 0;
        size_t hdr = p[i] & 4 ? 2 : 1; uint64_t size;
        size_t l = i + hdr < n ? leb128(p + i + hdr, n - i - hdr, &size) : 0;
        if (!l || size > n - i - hdr - l) return 0;
        i += hdr + l + (size_t)size;
    }
    return 1;
}
const uint8_t *mav1_find_seq(const uint8_t *p, size_t n, size_t *len) {
    if (!low_overhead(p, n)) return NULL;
    for (size_t i = 0; i < n;) {
        size_t hdr = p[i] & 4 ? 2 : 1; uint64_t size; size_t l = leb128(p + i + hdr, n - i - hdr, &size);
        if ((p[i] >> 3 & 15) == 1) { *len = hdr + l + (size_t)size; return p + i; }
        i += hdr + l + (size_t)size;
    }
    return NULL;
}
// The first frame header (3), redundant frame header (4) or frame (6) OBU's uncompressed_header (AV1 5.9.2): show_existing_frame f(1), frame_type f(2) (KEY_FRAME = 0); with
// reduced_still_picture_header neither is coded and the frame is a key frame
int mav1_tu_key(const uint8_t *p, size_t n, int reduced) {
    if (!low_overhead(p, n)) return -1;
    for (size_t i = 0; i < n;) {
        unsigned t = p[i] >> 3 & 15; size_t hdr = p[i] & 4 ? 2 : 1; uint64_t size; size_t l = leb128(p + i + hdr, n - i - hdr, &size);
        if (t == 3 || t == 4 || t == 6) {
            if (reduced) return 1;
            if (!size) return -1;
            uint8_t b = p[i + hdr + l];
            return !(b & 0x80) && !(b & 0x60);
        }
        i += hdr + l + (size_t)size;
    }
    return -1;
}
// One OBU (header, optional extension, payload of psize bytes) appended to out with its size field; temporal delimiters (2) and padding (15) are left out. The new end, or -1 when out is full
static ssize_t emit(uint8_t *out, size_t at, size_t cap, uint8_t h0, const uint8_t *ext, const uint8_t *payload, size_t psize) {
    unsigned t = h0 >> 3 & 15;
    if (t == 2 || t == 15) return (ssize_t)at;
    if (at + 2 + 8 + psize > cap) return -1;
    out[at++] = (uint8_t)(h0 | 2);
    if (ext) out[at++] = *ext;
    at += put_leb128(out + at, psize);
    memcpy(out + at, payload, psize);
    return (ssize_t)(at + psize);
}
static ssize_t from_low_overhead(const uint8_t *p, size_t n, uint8_t *out, size_t cap) {   // (p checked by low_overhead)
    ssize_t at = 0;
    for (size_t i = 0; i < n && at >= 0;) {
        size_t hdr = p[i] & 4 ? 2 : 1; uint64_t size; size_t l = leb128(p + i + hdr, n - i - hdr, &size);
        at = emit(out, (size_t)at, cap, p[i], hdr == 2 ? p + i + 1 : NULL, p + i + hdr + l, (size_t)size);
        i += hdr + l + (size_t)size;
    }
    return at;
}
// Annex B (AV1 B.2): temporal_unit_size, then frame_unit_size and obu_length pairs; an OBU there may lack its size field
static ssize_t from_annexb(const uint8_t *p, size_t n, uint8_t *out, size_t cap) {
    uint64_t tu; size_t l = leb128(p, n, &tu);
    if (!l || !tu || tu != n - l) return -1;
    ssize_t at = 0;
    for (size_t i = l; i < n;) {
        uint64_t fu; size_t fl = leb128(p + i, n - i, &fu);
        if (!fl || !fu || fu > n - i - fl) return -1;
        size_t fend = i + fl + (size_t)fu;
        for (i += fl; i < fend;) {
            uint64_t ol; size_t oll = leb128(p + i, fend - i, &ol);
            if (!oll || !ol || ol > fend - i - oll) return -1;
            const uint8_t *o = p + i + oll; size_t on = (size_t)ol, hdr = o[0] & 4 ? 2 : 1;
            if ((o[0] & 0x80) || !is_obu_type(o[0] >> 3 & 15) || on < hdr) return -1;
            const uint8_t *pay = o + hdr; size_t psize = on - hdr;
            if (o[0] & 2) { uint64_t s; size_t sl = leb128(pay, psize, &s); if (!sl || s > psize - sl) return -1; pay += sl; psize = (size_t)s; }
            if ((at = emit(out, (size_t)at, cap, (uint8_t)(o[0] & ~2), hdr == 2 ? o + 1 : NULL, pay, psize)) < 0) return -1;
            i += oll + on;
        }
    }
    return at;
}
// Low-overhead OBUs (as from_low_overhead writes them) with something to decode: a sequence header, a frame header, a tile group, a frame, a tile list
static int decodable(const uint8_t *p, size_t n) {
    for (size_t i = 0; i < n;) {
        unsigned t = p[i] >> 3 & 15; size_t hdr = p[i] & 4 ? 2 : 1; uint64_t size; size_t l = leb128(p + i + hdr, n - i - hdr, &size);
        if (t != 5 && t != 2 && t != 15) return 1;
        i += hdr + l + (size_t)size;
    }
    return 0;
}
const char *mav1_framing_name(int f) { return f == MAV1_LOW_OVERHEAD ? "low-overhead OBUs" : f == MAV1_ANNEXB ? "Annex B" : f == MAV1_IVF ? "IVF frames" : "unknown"; }
ssize_t mav1_tu(const uint8_t *in, size_t n, uint8_t *out, size_t cap, int *framing) {
    if (n >= 32 && !memcmp(in, "DKIF", 4)) { size_t hl = (size_t)in[6] | (size_t)in[7] << 8; if (hl < 32 || hl > n) return -1; in += hl; n -= hl; }   // an IVF file header before the first frame
    if (n > 12 && ((uint32_t)in[0] | (uint32_t)in[1] << 8 | (uint32_t)in[2] << 16 | (uint32_t)in[3] << 24) == n - 12 && low_overhead(in + 12, n - 12)) {
        if (framing) *framing = MAV1_IVF;
        return from_low_overhead(in + 12, n - 12, out, cap);
    }
    // An Annex B unit of one frame unit can read as a single metadata OBU too (its first byte as an OBU header with a size field, the frame unit's size as that size): low-overhead OBUs
    // with nothing but metadata are read as Annex B when they are that as well
    if (low_overhead(in, n)) {
        ssize_t r = from_low_overhead(in, n, out, cap);
        if (r <= 0 || decodable(out, (size_t)r) || from_annexb(in, n, out, cap) < 0) { if (framing) *framing = MAV1_LOW_OVERHEAD; return from_low_overhead(in, n, out, cap); }
    }
    ssize_t r = n ? from_annexb(in, n, out, cap) : -1;
    if (r >= 0 && framing) *framing = MAV1_ANNEXB;
    return r;
}
