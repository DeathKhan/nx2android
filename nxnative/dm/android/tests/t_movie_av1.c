// AV1 packets for MediaCodec without Android (dm/android/movie_av1.c): the sequence header (av1C record, size, colour description) and each packet framing turned into the temporal unit
// MediaCodec takes (low-overhead OBUs, no temporal delimiters). make -C dm/android/tests (clang or gcc on a Mac or Linux host; no game, no device)
#include "../movie_av1.h"
#include "hle_movie_av1.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// One temporal unit (SVT-AV1, 64x64 red key frame; dm/host/tests/t_codec_video.c's): temporal delimiter, sequence header (profile 0, 64x64), frame
static const uint8_t av1[] = {
    0x12,0x00,0x0a,0x0b,0x02,0x00,0x00,0x04,0x95,0x7f,0xfc,0x0a,0xf9,0x00,0x40,0x32,0x15,0x10,0x00,0xbc,0x00,0x49,0x22,0x08,0x08,0x21,0x10,0x00,0xad,0x3b,0x67,0x00,
    0x1e,0xb5,0xdb,0x79,0x52,0x90,
};
#define SEQ_AT 2
#define SEQ_N 13    // 0x0a, size 0x0b, 11 bytes
#define FRAME_AT 15 // 0x32, size 0x15, 21 bytes
static uint8_t out[512];

static void t_seq(void) {
    size_t n = 0; const uint8_t *s = mav1_find_seq(av1, sizeof av1, &n);
    assert(s == av1 + SEQ_AT && n == SEQ_N);
    uint8_t c[4]; mav1_seq si; assert(mav1_seq_parse(s, n, c, &si));
    assert(si.w == 64 && si.h == 64 && si.profile == 0 && si.bits == 8 && !si.mono && si.matrix == 1 && !si.full_range);   // no colour description: BT.601 below 720 lines
    assert(c[0] == 0x81 && (c[1] >> 5) == 0 && c[1] == si.level);
    int w = 0, h = 0; uint8_t c2[4]; assert(mav1_av1c(s, n, c2, &w, &h) && w == 64 && h == 64 && !memcmp(c, c2, 4));
    assert(!mav1_seq_parse(av1, 2, c, &si));                 // a temporal delimiter is no sequence header
    assert(!mav1_seq_parse(s, n - 4, c, &si));               // cut short
    // upstream's field-by-field header: 2560x1440, level 12, 8-bit 4:2:0 (t_codec_video.c), then the same with a BT.709 full-range colour description
    for (int desc = 0; desc < 2; desc++) {
        uint8_t obu[32] = {0x0a, 0}; size_t bit = 16;
        #define PUT(v, nb) do { for (int k_ = (nb) - 1; k_ >= 0; k_--, bit++) if (((uint32_t)(v) >> k_) & 1) obu[bit / 8] |= (uint8_t)(0x80 >> (bit % 8)); } while (0)
        PUT(0, 3); PUT(0, 1); PUT(0, 1); PUT(0, 1); PUT(0, 1); PUT(0, 5); PUT(0, 12); PUT(12, 5); PUT(0, 1);
        PUT(11, 4); PUT(10, 4); PUT(2559, 12); PUT(1439, 11);
        PUT(0, 1); PUT(0, 3); PUT(0, 4); PUT(1, 1); PUT(0, 2); PUT(1, 1); PUT(1, 1); PUT(6, 3); PUT(0, 3);
        PUT(0, 1); PUT(0, 1);                                // not high bit depth, not monochrome
        if (desc) { PUT(1, 1); PUT(1, 8); PUT(1, 8); PUT(1, 8); PUT(1, 1); }   // BT.709 primaries, transfer, matrix; full range
        else { PUT(0, 1); PUT(0, 1); }                       // no colour description, video range
        PUT(0, 2); PUT(0, 1);
        #undef PUT
        obu[1] = (uint8_t)((bit + 7) / 8 - 2);
        assert(mav1_seq_parse(obu, (bit + 7) / 8, c, &si) && si.w == 2560 && si.h == 1440 && si.level == 12);
        assert(c[0] == 0x81 && c[1] == 12 && c[2] == 0x0c && c[3] == 0);
        assert(si.matrix == 2 && si.full_range == desc);   // BT.709: declared, or unspecified at 1440 lines
    }
    puts("sequence header ok");
}
static void t_low_overhead(void) {
    int f = 0; ssize_t n = mav1_tu(av1, sizeof av1, out, sizeof out, &f);
    assert(f == MAV1_LOW_OVERHEAD && n == (ssize_t)sizeof av1 - 2 && !memcmp(out, av1 + 2, sizeof av1 - 2));   // the temporal delimiter left out, the rest as it came
    size_t sl; assert(mav1_find_seq(out, (size_t)n, &sl) == out && sl == SEQ_N);
    static const uint8_t td[] = {0x12, 0x00}, pad[] = {0x12, 0x00, 0x7a, 0x02, 0xaa, 0xbb};   // a delimiter alone; a delimiter and padding
    assert(mav1_tu(td, sizeof td, out, sizeof out, &f) == 0 && mav1_tu(pad, sizeof pad, out, sizeof out, &f) == 0);
    assert(mav1_tu(av1, sizeof av1, out, 20, &f) == -1);   // no room
    // an OBU with an extension header (temporal / spatial ids) keeps it
    static const uint8_t ext[] = {0x36, 0x28, 0x03, 0x01, 0x02, 0x03};   // frame OBU, extension 0x28, 3 bytes
    assert(mav1_tu(ext, sizeof ext, out, sizeof out, &f) == (ssize_t)sizeof ext && !memcmp(out, ext, sizeof ext));
    puts("low-overhead ok");
}
static void t_annexb(void) {
    // the same unit in Annex B: temporal_unit_size, frame_unit_size, then obu_length and each OBU without its size field
    uint8_t ab[64]; size_t k = 0;
    const uint8_t *seqp = av1 + SEQ_AT + 2, *frp = av1 + FRAME_AT + 2; size_t seqn = SEQ_N - 2, frn = 0x15;
    size_t obus = (1 + 1) + (1 + 1 + seqn) + (1 + 1 + frn);   // delimiter, sequence header, frame: obu_length and the OBU
    ab[k++] = (uint8_t)(1 + obus); ab[k++] = (uint8_t)obus;
    ab[k++] = 1; ab[k++] = 0x10;                                                  // temporal delimiter (no size field)
    ab[k++] = (uint8_t)(1 + seqn); ab[k++] = 0x08; memcpy(ab + k, seqp, seqn); k += seqn;
    ab[k++] = (uint8_t)(1 + frn); ab[k++] = 0x30; memcpy(ab + k, frp, frn); k += frn;
    int f = 0; ssize_t n = mav1_tu(ab, k, out, sizeof out, &f);
    assert(f == MAV1_ANNEXB && n == (ssize_t)sizeof av1 - 2 && !memcmp(out, av1 + 2, sizeof av1 - 2));
    // Annex B whose OBUs keep their size fields: the same
    k = 0; obus = (1 + 2) + (1 + SEQ_N) + (1 + 2 + frn);
    ab[k++] = (uint8_t)(1 + obus); ab[k++] = (uint8_t)obus;
    ab[k++] = 2; ab[k++] = 0x12; ab[k++] = 0;
    ab[k++] = SEQ_N; memcpy(ab + k, av1 + SEQ_AT, SEQ_N); k += SEQ_N;
    ab[k++] = (uint8_t)(2 + frn); memcpy(ab + k, av1 + FRAME_AT, 2 + frn); k += 2 + frn;
    n = mav1_tu(ab, k, out, sizeof out, &f);
    assert(f == MAV1_ANNEXB && n == (ssize_t)sizeof av1 - 2 && !memcmp(out, av1 + 2, sizeof av1 - 2));
    ab[1]++; assert(mav1_tu(ab, k, out, sizeof out, &f) == -1);   // a frame unit size that is not the packet's
    static const uint8_t junk[] = {0xff, 0xff, 0xff, 0xff, 0x00, 0x13}; assert(mav1_tu(junk, sizeof junk, out, sizeof out, &f) == -1);
    puts("Annex B ok");
}
static void t_ivf(void) {
    uint8_t iv[32 + 12 + sizeof av1] = {0}; size_t n = sizeof av1;
    memcpy(iv, "DKIF", 4); iv[6] = 32; memcpy(iv + 8, "AV01", 4);
    iv[32] = (uint8_t)n; memcpy(iv + 44, av1, n);
    int f = 0;
    assert(mav1_tu(iv + 32, 12 + n, out, sizeof out, &f) == (ssize_t)n - 2 && f == MAV1_IVF && !memcmp(out, av1 + 2, n - 2));   // a frame with its 12-byte header
    assert(mav1_tu(iv, sizeof iv, out, sizeof out, &f) == (ssize_t)n - 2 && f == MAV1_IVF && !memcmp(out, av1 + 2, n - 2));   // the first, after the file header
    puts("IVF ok");
}
int main(void) {
    t_seq(); t_low_overhead(); t_annexb(); t_ivf();
    puts("t_movie_av1: all ok");
    return 0;
}
