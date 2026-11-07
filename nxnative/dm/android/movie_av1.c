// Android port: the AV1 decoder interface of dm/host/hle_movie_av1.c (VideoToolbox or dav1d) without a decoder yet. hle_codec_video.c opens one for an AV1 stream
// (the Switch 2 SDK's nn::codec, Fire Emblem Fortune's Weave) and gets -1 from every decode, so such a movie shows nothing instead of aborting the module.
// TODO(upsync): dav1d (dm/third_party/dav1d, already vendored) built by build.py as the Mac's software backend is, or MediaCodec's AV1 decoder as movie_avc.c uses it for H.264.
#include "hle_movie_av1.h"
#include <stdlib.h>
struct mav1 { int refused; };
mav1 *mav1_new(void) { return calloc(1, sizeof(mav1)); }
void mav1_free(mav1 *a) { free(a); }
void mav1_reset(mav1 *a) { if (a) a->refused = 0; }
int mav1_decode(mav1 *a, const uint8_t *data, size_t n, mav1_picture *out) { (void)data; (void)n; (void)out; if (a) a->refused++; return -1; }
const char *mav1_backend(const mav1 *a) { (void)a; return "none (Android: no AV1 decoder yet)"; }
int mav1_av1c(const uint8_t *obu, size_t n, uint8_t c[4], int *w, int *h) { (void)obu; (void)n; (void)c; (void)w; (void)h; return 0; }
