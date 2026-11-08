// Android port: AV1 on MediaCodec (movie_av1.c, movie_avc.c). The OBU side (no Android: unit-tested on the host, dm/android/tests/t_movie_av1.c) and the AV1 mode of movie_avc.c's
// MediaCodec decoder, which hle_codec_video.c drives exactly as its H.264 one (queue, end of stream, zero copy)
#pragma once
#include <stddef.h>
#include <stdint.h>
#include <sys/types.h>

// What a sequence header OBU says (AV1 5.5): the size, profile and level, and its colour description as mav1_picture gives it (matrix: BT.601 = 1, BT.709 = 2, BT.2020 = 5; unspecified:
// BT.601 below 720 lines, BT.709 from there, as the Mac's backends take it)
typedef struct { int w, h, profile, level, tier, bits, mono, matrix, full_range; } mav1_seq;
// A sequence header OBU (with its header and size field) read: 1, the av1C record's 4 bytes in c (ISO-BMFF AV1 binding 2.3.3; csd-0 is these and then the OBU) and *s; 0 when it cannot be read
int mav1_seq_parse(const uint8_t *obu, size_t n, uint8_t c[4], mav1_seq *s);
// The sequence header OBU of a temporal unit in the low-overhead format, or NULL
const uint8_t *mav1_find_seq(const uint8_t *p, size_t n, size_t *len);

// How the SDK's packets were framed (mav1_tu)
enum { MAV1_LOW_OVERHEAD = 1, MAV1_ANNEXB = 2, MAV1_IVF = 3 };
const char *mav1_framing_name(int framing);
// One packet as the temporal unit MediaCodec takes: OBUs in the low-overhead format (each with its size field), temporal delimiters and padding left out. The packet may be that already
// (WebM / MP4 samples, what the SDK's demuxer gives), Annex B (length-prefixed temporal_unit(), AV1 annex B) or an IVF frame (its 12-byte header, an IVF file header before the first).
// Returns the bytes written to out (cap: 2 n + 64 always suffices), 0 for a unit with nothing to decode, -1 when the packet is none of these
ssize_t mav1_tu(const uint8_t *in, size_t n, uint8_t *out, size_t cap, int *framing);

// movie_avc.c's MediaCodec decoder in its AV1 mode (mime video/av01): the hardware decoder first (c2.qti.av1.decoder on the Snapdragon 8 Elite), c2.android.av1.decoder when the hardware one
// refuses the stream, stalls or errors, or with DM_AV1_SW=1; DM_AV1_DECODER=<name> picks one by name; DM_AV1_CSD=0 leaves csd-0 (the av1C record) out of the configuration
typedef struct mavc mavc;
mavc *mavc_new_av1(void);
