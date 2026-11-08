// Android port: H.264 for the SDK's movie library (dm/host/hle_movie_avc.h) on MediaCodec, VideoToolbox's counterpart (dm/host/hle_movie_avc.c).
// Unpacking's only movie is its 5 s Humble Games logo (StreamingAssets/HumbleGames_LogoIntro-2021_5sec_1080p30.mp4, H.264 Main 1080p, no audio), which Unity 2019.4 plays through
// the SDK's decoder in CPU mode (NV12 out); with no pictures the intro ended at once and the game crashed right after it (main+0x8a74d0).
// The decoder is configured from SPS / PPS (csd-0 / csd-1 with start codes, as Android's own movie players pass them) and fed one access unit per call, in Annex B.
// MediaCodec hands out pictures in presentation order, a few frames after their input (the hardware decoder's pipeline): mavc_take waits briefly for one while inputs are in flight,
// and at the end of the stream sends an end-of-stream input and collects every picture still in the codec. Pictures leave the codec as compact NV12 copies (the output buffer's stride,
// slice height and crop come from its format: Qualcomm's decoder pads rows to 128 bytes and planes to 32 rows), so a buffer goes back to the codec at once.
// The same machinery decodes AV1 (mavc_new_av1, movie_av1.c / movie_av1.h): mime video/av01, configured from the stream's sequence header instead of SPS / PPS, fed temporal units of
// low-overhead OBUs, with a software decoder to fall back on
#include "dm.h"
#include "hle_movie_avc.h"
#include <stdlib.h>
#include <string.h>
#include <media/NdkMediaCodec.h>
#include <media/NdkMediaFormat.h>
#include <unistd.h>
#include <android/native_window.h>
#include "movie_av1.h"  // the AV1 mode: sequence headers, packet framing
#include "movie_zc.h"   // the surface mode: pictures stay in the codec's AHardwareBuffers (mavc_set_zero_copy, mavc_take_frame)
#define COLOR_FORMAT_SURFACE 0x7F000789   // MediaCodecInfo.CodecCapabilities.COLOR_FormatSurface: the codec picks its buffers' layout for the surface

#define QMAX 16
#define AV1_SW "c2.android.av1.decoder"   // the platform's software AV1 decoder (libgav1 or dav1d inside), the fallback
#define AV1_STALL 64                    // packets the hardware AV1 decoder may take without a picture out before the software one takes over
#define COLOR_YUV420_PLANAR 19   // MediaCodecInfo.CodecCapabilities: I420 (the software decoder's); every other format is read as semi-planar (21, Qualcomm's 0x7fa30c04 venus NV12)
struct mavc {
    uint8_t sps[512], pps[512]; int nsps, npps, dirty;
    AMediaCodec *mc; int eos_sent;
    int fw, fh, fstride, fslice, fcolor, cl, ct, cr, cb;   // output format: size, stride, slice height, colour format, crop rectangle (inclusive)
    long nin, nout;                                        // inputs queued and pictures out since the codec started (or was flushed)
    struct { uint8_t *nv12; int w, h; int64_t pts; } q[QMAX]; int nq;
    uint8_t *au; size_t aucap;   // Annex B copy of the access unit being decoded
    int red, rgba;               // rgba: pictures as 4-byte RGBA (native texture mode) instead of NV12
    // The codec's timestamps: the game's while they increase, else the last one + 1 µs, each mapped back to the game's on output (pmap). 13 Sentinels: Aegis Rim sends every
    // packet of its 1024x576 movies with time 0, and c2.qti.avc.decoder gave no picture for 81 packets and then took no more input. With made-up timestamps the codec's
    // own output order stands (synth: the queue is not sorted)
    int64_t last_in; int synth; struct { int64_t c, g; } pmap[64]; int npmap;
    // Surface mode (zero copy, movie_zc.h): the codec renders each picture into the reader's buffers; rendered pictures wait in tsmap (codec time → game time) for their AImage,
    // which zq then holds in presentation order. The reader lives as long as the mode (the codec may be opened again for a new stream)
    int zc, zc_w, zc_h; zc_reader *rd; zc_tsmap tsmap; struct { zc_frame *f; int64_t pts; } zq[QMAX]; int nzq;
    // AV1 (mavc_new_av1): seq is the stream's sequence header OBU (csd-0 is made from it; a changed one opens the codec again), si what it says. sw: the software decoder; sw_next:
    // it takes over at the next sequence header (the hardware one stalled or failed); framing: how the packets came (movie_av1.h); pictures: out of the decoder since it was opened
    int av1, sw, sw_next, framing; uint8_t seq[512]; size_t nseq; mav1_seq si; long pictures;
};
#define KIND(a) ((a)->av1 ? "AV1" : "AVC")
static int64_t game_pts(mavc *a, int64_t cpts) {   // the codec's timestamp back to the game's (pmap)
    for (int m = 0; m < a->npmap; m++) if (a->pmap[m].c == cpts) { int64_t g = a->pmap[m].g; a->pmap[m] = a->pmap[--a->npmap]; return g; }
    return cpts;
}
static void zq_drop(mavc *a) { for (int i = 0; i < a->nzq; i++) zc_frame_unref(a->zq[i].f); a->nzq = 0; a->tsmap.n = 0; }
// the pictures the codec rendered into the reader's buffers, into zq in presentation order (discard: images from before a flush)
static int zc_pull(mavc *a, int discard) {
    int n = 0; zc_frame *f;
    while (a->rd && (f = zc_reader_acquire(a->rd))) {
        int64_t g; int known = zc_tsmap_take(&a->tsmap, zc_frame_ts(f) / 1000, &g);
        if (discard || !known) { zc_frame_unref(f); continue; }
        if (a->nzq == QMAX) { if (!a->red++) LOG("[RED] movie: %s picture queue full, dropping the oldest picture", KIND(a)); zc_frame_unref(a->zq[0].f); memmove(a->zq, a->zq + 1, --a->nzq * sizeof a->zq[0]); }
        int k = a->nzq; while (!a->synth && k > 0 && a->zq[k - 1].pts > g) { a->zq[k] = a->zq[k - 1]; k--; }
        a->zq[k].f = f; a->zq[k].pts = g; a->nzq++; n++;
    }
    return n;
}

mavc *mavc_new(void) { mavc *a = calloc(1, sizeof *a); if (!a) die("movie: AVC decoder allocation failed"); a->last_in = INT64_MIN; return a; }
static void close_codec(mavc *a) { if (a->mc) { AMediaCodec_stop(a->mc); AMediaCodec_delete(a->mc); a->mc = NULL; } a->nin = a->nout = 0; a->eos_sent = 0; a->last_in = INT64_MIN; a->npmap = 0; }
void mavc_drop(mavc *a) {
    for (int i = 0; i < a->nq; i++) free(a->q[i].nv12);
    a->nq = 0;
    if (a->zc) { zq_drop(a); zc_pull(a, 1); }
    if (a->mc) { AMediaCodec_flush(a->mc); a->nin = a->nout = 0; a->eos_sent = 0; a->last_in = INT64_MIN; a->npmap = 0; }   // a seek: pictures in the codec belong to the old position
}
void mavc_free(mavc *a) { if (!a) return; mavc_drop(a); close_codec(a); if (a->rd) zc_reader_free(a->rd); free(a->au); free(a); }
int mavc_queued(const mavc *a) { return a->nq + a->nzq; }
int mavc_held(const mavc *a) { return a->nq + a->nzq + a->tsmap.n + (a->mc && !a->eos_sent && a->nin > a->nout ? (int)(a->nin - a->nout) : 0); }
// Zero copy on: the codec is opened again (at the next packet) with the reader's surface as its output. DM_MOVIE_ZC_IMAGES (start-up): the reader's image count (16: hle_codec_video.c holds
// up to 12 pictures for the game, one is shown, a few wait for the GPU); the codec stalls, not the host, when every image is out
int mavc_set_zero_copy(mavc *a, int on, int w, int h) {
    if (on && !a->zc) {
        if (!zc_gpu_ok()) return 0;
        const char *e = getenv("DM_MOVIE_ZC_IMAGES"); int n = e ? atoi(e) : 16; if (n < 4) n = 4; if (n > 32) n = 32;
        if (!a->rd || a->zc_w != w || a->zc_h != h) { if (a->rd) zc_reader_free(a->rd); a->rd = zc_reader_new(w, h, n); a->zc_w = w; a->zc_h = h; }
        if (!a->rd) return 0;
        a->zc = 1; close_codec(a);
    } else if (!on && a->zc) {
        close_codec(a); zq_drop(a); zc_pull(a, 1); a->zc = 0;
        if (a->rd) { zc_reader_free(a->rd); a->rd = NULL; }
    }
    return a->zc;
}
void mavc_set_rgba(mavc *a) { a->rgba = 1; }

static void keep_ps(mavc *a, const uint8_t *nal, size_t n) {   // a parameter set NAL unit (without start code)
    int t = nal[0] & 31; uint8_t *dst = t == 7 ? a->sps : a->pps; int *dn = t == 7 ? &a->nsps : &a->npps;
    if (n > sizeof a->sps) { if (!a->red++) LOG("[RED] movie: AVC parameter set of %zu bytes, too long", n); return; }
    if ((size_t)*dn == n && !memcmp(dst, nal, n)) return;
    memcpy(dst, nal, n); *dn = (int)n; a->dirty = 1;
}
// walk the NAL units of p (Annex B start codes, or else 4-byte lengths); fn gets each unit without its prefix (as hle_movie_avc.c)
static size_t next_sc(const uint8_t *p, size_t n, size_t i) { for (; i + 3 <= n; i++) if (!p[i] && !p[i + 1] && p[i + 2] == 1) return i; return n; }
static void each_nal(const uint8_t *p, size_t n, void (*fn)(void *, const uint8_t *, size_t), void *ctx) {
    size_t s = next_sc(p, n, 0);
    if (s < n && s <= 1) {
        while (s < n) {
            size_t b = s + 3, e = next_sc(p, n, b), end = e;
            while (end > b && !p[end - 1]) end--;
            if (end > b) fn(ctx, p + b, end - b);
            s = e;
        }
    } else for (size_t i = 0; i + 4 <= n;) {
        size_t len = (size_t)p[i] << 24 | (size_t)p[i + 1] << 16 | (size_t)p[i + 2] << 8 | p[i + 3]; i += 4;
        if (!len || len > n - i) break;
        fn(ctx, p + i, len); i += len;
    }
}
static void ps_only(void *ctx, const uint8_t *nal, size_t n) { int t = nal[0] & 31; if (t == 7 || t == 8) keep_ps(ctx, nal, n); }
void mavc_param_sets(mavc *a, const uint8_t *p, size_t n) { each_nal(p, n, ps_only, a); }
void mavc_avcc(mavc *a, const uint8_t *p, size_t n) {   // AVCDecoderConfigurationRecord (the MP4 avcC box)
    if (n < 7 || p[0] != 1) { if (!a->red++) LOG("[RED] movie: unrecognized avcC record (%zu bytes, version %d)", n, n ? p[0] : -1); return; }
    size_t i = 5;
    for (int set = 0; set < 2 && i < n; set++) {
        int cnt = set ? p[i] : p[i] & 31; i++;
        for (int k = 0; k < cnt && i + 2 <= n; k++) { size_t len = (size_t)p[i] << 8 | p[i + 1]; i += 2; if (len > n - i) return; if (len) keep_ps(a, p + i, len); i += len; }
    }
}

static void set_csd(AMediaFormat *f, const char *key, const uint8_t *nal, int n) {
    uint8_t b[4 + 512] = {0, 0, 0, 1}; memcpy(b + 4, nal, (size_t)n); AMediaFormat_setBuffer(f, key, b, (size_t)n + 4);
}
// The decoder: H.264's is the platform's first; AV1's the platform's first too (the hardware one: c2.qti.av1.decoder on Snapdragon), the software one when asked (DM_AV1_SW=1) or fallen back
// to, or the one DM_AV1_DECODER names
static AMediaCodec *new_codec(mavc *a) {
    if (!a->av1) return AMediaCodec_createDecoderByType("video/avc");
    const char *e = getenv("DM_AV1_DECODER");
    if (e && *e && !a->sw) { AMediaCodec *c = AMediaCodec_createCodecByName(e); if (c) return c; LOG("[RED] movie: DM_AV1_DECODER=%s: no such decoder, the default one instead", e); }
    if (a->sw) { AMediaCodec *c = AMediaCodec_createCodecByName(AV1_SW); if (c) return c; LOG("[RED] movie: no %s, the default AV1 decoder instead", AV1_SW); }
    return AMediaCodec_createDecoderByType("video/av01");
}
static int open_codec(mavc *a) {
    close_codec(a);
    if (a->av1 ? !a->nseq : !a->nsps || !a->npps) { if (!a->red++) LOG("[RED] movie: %s has no %s, outputting black frames", KIND(a), a->av1 ? "sequence header" : "SPS / PPS"); return 0; }
    media_status_t st;
    for (;;) {
        a->mc = new_codec(a);
        if (!a->mc) { if (!a->red++) LOG("[RED] movie: MediaCodec has no %s decoder, outputting black frames", KIND(a)); return 0; }
        AMediaFormat *f = AMediaFormat_new();
        AMediaFormat_setString(f, AMEDIAFORMAT_KEY_MIME, a->av1 ? "video/av01" : "video/avc");
        int w = a->av1 ? a->si.w : 1920, h = a->av1 ? a->si.h : 1080;   // the decoder takes the real size from the SPS / sequence header (an output format change)
        AMediaFormat_setInt32(f, AMEDIAFORMAT_KEY_WIDTH, w); AMediaFormat_setInt32(f, AMEDIAFORMAT_KEY_HEIGHT, h);
        AMediaFormat_setInt32(f, AMEDIAFORMAT_KEY_MAX_INPUT_SIZE, 4 << 20);
        if (a->zc) { AMediaFormat_setInt32(f, AMEDIAFORMAT_KEY_COLOR_FORMAT, COLOR_FORMAT_SURFACE); if (a->zc_w && a->zc_h) { AMediaFormat_setInt32(f, AMEDIAFORMAT_KEY_WIDTH, a->zc_w); AMediaFormat_setInt32(f, AMEDIAFORMAT_KEY_HEIGHT, a->zc_h); } }
        if (a->av1) {
            // csd-0: the av1C record (4 bytes, then the sequence header OBU), as MediaExtractor gives it for MP4; the decoders also read the header in the stream (DM_AV1_CSD=0: left out)
            const char *e = getenv("DM_AV1_CSD"); uint8_t rec[4 + sizeof a->seq];
            if (!(e && *e == '0') && mav1_seq_parse(a->seq, a->nseq, rec, NULL)) { memcpy(rec + 4, a->seq, a->nseq); AMediaFormat_setBuffer(f, AMEDIAFORMAT_KEY_CSD_0, rec, 4 + a->nseq); }
            AMediaFormat_setInt32(f, AMEDIAFORMAT_KEY_LOW_LATENCY, 1);   // a temporal unit shows at most one frame, in order: each picture out as soon as it is decoded (ignored where unsupported)
        } else { set_csd(f, AMEDIAFORMAT_KEY_CSD_0, a->sps, a->nsps); set_csd(f, AMEDIAFORMAT_KEY_CSD_1, a->pps, a->npps); }
        st = AMediaCodec_configure(a->mc, f, a->zc ? (ANativeWindow *)zc_reader_window(a->rd) : NULL, NULL, 0);
        if (st == AMEDIA_OK) st = AMediaCodec_start(a->mc);
        AMediaFormat_delete(f);
        if (st == AMEDIA_OK) break;
        char *name = NULL; AMediaCodec_getName(a->mc, &name);
        LOG("[RED] movie: MediaCodec %s decoder %s refused its configuration (%d)%s", KIND(a), name ? name : "?", (int)st, a->av1 && !a->sw ? ": the software decoder " AV1_SW " instead" : ", outputting black frames");
        if (name) AMediaCodec_releaseName(a->mc, name);
        AMediaCodec_delete(a->mc); a->mc = NULL;
        if (!a->av1 || a->sw) return 0;
        a->sw = 1;
    }
    char *name = NULL; AMediaCodec_getName(a->mc, &name);
    if (a->av1) LOG("[movie] AV1: MediaCodec decoder %s, %dx%d profile %d level %d %d-bit%s%s", name ? name : "?", a->si.w, a->si.h, a->si.profile, a->si.level, a->si.bits,
                    a->sw ? " (software)" : "", a->zc ? ", pictures into the zero-copy reader's buffers" : ", NV12 copies");
    else LOG("[movie] AVC: MediaCodec decoder %s, profile %d%s", name ? name : "?", a->nsps > 1 ? a->sps[1] : 0, a->zc ? ", pictures into the zero-copy reader's buffers" : "");
    if (name) AMediaCodec_releaseName(a->mc, name);
    a->fw = a->fh = 0; a->dirty = 0; a->pictures = 0; return 1;
}
static void read_format(mavc *a) {
    AMediaFormat *f = AMediaCodec_getOutputFormat(a->mc); int32_t v;
    a->fw = AMediaFormat_getInt32(f, AMEDIAFORMAT_KEY_WIDTH, &v) ? v : 0; a->fh = AMediaFormat_getInt32(f, AMEDIAFORMAT_KEY_HEIGHT, &v) ? v : 0;
    a->fstride = AMediaFormat_getInt32(f, AMEDIAFORMAT_KEY_STRIDE, &v) && v >= a->fw ? v : a->fw;
    a->fslice = AMediaFormat_getInt32(f, AMEDIAFORMAT_KEY_SLICE_HEIGHT, &v) && v >= a->fh ? v : a->fh;
    a->fcolor = AMediaFormat_getInt32(f, AMEDIAFORMAT_KEY_COLOR_FORMAT, &v) ? v : 0;
    int32_t l, t, r, b;
    if (AMediaFormat_getRect(f, AMEDIAFORMAT_KEY_DISPLAY_CROP, &l, &t, &r, &b) && r >= l && b >= t && r < a->fw && b < a->fh) { a->cl = l; a->ct = t; a->cr = r; a->cb = b; }
    else { a->cl = a->ct = 0; a->cr = a->fw - 1; a->cb = a->fh - 1; }
    LOG("[movie] %s: output %dx%d stride %d slice height %d colour format %#x crop %d,%d-%d,%d", KIND(a), a->fw, a->fh, a->fstride, a->fslice, a->fcolor, a->cl, a->ct, a->cr, a->cb);
    AMediaFormat_delete(f);
}
// one output buffer → a compact NV12 picture of the crop rectangle, queued in presentation order
static void keep_picture(mavc *a, const uint8_t *src, size_t n, int64_t pts) {
    int w = (a->cr - a->cl + 1) & ~1, h = (a->cb - a->ct + 1) & ~1, s = a->fstride;
    size_t uvoff = (size_t)s * a->fslice;
    int planar = a->fcolor == COLOR_YUV420_PLANAR;
    size_t need = planar ? uvoff + uvoff / 2 : uvoff + (size_t)s * (a->fslice / 2);
    if (w <= 0 || h <= 0 || n < need - (size_t)(s - a->fw)) { if (!a->red++) LOG("[RED] movie: %s output buffer of %zu bytes is smaller than its %dx%d picture (stride %d, slice height %d)", KIND(a), n, a->fw, a->fh, s, a->fslice); return; }
    uint8_t *o = malloc((size_t)w * h * 3 / 2); if (!o) die("movie: AVC picture allocation failed");
    for (int r = 0; r < h; r++) memcpy(o + (size_t)r * w, src + (size_t)(a->ct + r) * s + a->cl, (size_t)w);
    uint8_t *ouv = o + (size_t)w * h;
    if (planar) {   // I420: U plane then V plane, half stride
        const uint8_t *u = src + uvoff, *v = u + (size_t)(s / 2) * (a->fslice / 2);
        for (int r = 0; r < h / 2; r++) {
            const uint8_t *ur = u + (size_t)(a->ct / 2 + r) * (s / 2) + a->cl / 2, *vr = v + (size_t)(a->ct / 2 + r) * (s / 2) + a->cl / 2; uint8_t *d = ouv + (size_t)r * w;
            for (int x = 0; x < w / 2; x++) { d[2 * x] = ur[x]; d[2 * x + 1] = vr[x]; }
        }
    } else for (int r = 0; r < h / 2; r++) memcpy(ouv + (size_t)r * w, src + uvoff + (size_t)(a->ct / 2 + r) * s + (a->cl & ~1), (size_t)w);
    if (a->nq == QMAX) { if (!a->red++) LOG("[RED] movie: %s picture queue full, dropping the oldest picture", KIND(a)); free(a->q[0].nv12); memmove(a->q, a->q + 1, --a->nq * sizeof a->q[0]); }
    pts = game_pts(a, pts);   // the game's own timestamp
    int k = a->nq; while (!a->synth && k > 0 && a->q[k - 1].pts > pts) { a->q[k] = a->q[k - 1]; k--; }
    a->q[k].nv12 = o; a->q[k].w = w; a->q[k].h = h; a->q[k].pts = pts; a->nq++;
}
// collect the pictures the codec has ready, waiting up to wait_us for the first; 1 = the end-of-stream output came
static int drain(mavc *a, int64_t wait_us) {
    for (;;) {
        AMediaCodecBufferInfo bi; ssize_t i = AMediaCodec_dequeueOutputBuffer(a->mc, &bi, wait_us);
        if (i == AMEDIACODEC_INFO_OUTPUT_FORMAT_CHANGED) { read_format(a); continue; }
        if (i == AMEDIACODEC_INFO_OUTPUT_BUFFERS_CHANGED) continue;
        if (i < AMEDIACODEC_INFO_OUTPUT_BUFFERS_CHANGED) {   // an error (a media_status_t): the codec failed; AV1 goes to the software decoder at the next sequence header
            static int n; if (n++ < 4) LOG("[RED] movie: MediaCodec %s decoder error %zd%s", KIND(a), i, a->av1 && !a->sw ? ": the software decoder takes over at the next key frame" : "");
            if (a->av1 && !a->sw) a->sw_next = 1;
            return 0;
        }
        if (i < 0) {   // AMEDIACODEC_INFO_TRY_AGAIN_LATER
            if (a->av1 && !a->sw && !a->sw_next && !a->pictures && a->nin >= AV1_STALL) { LOG("[RED] movie: MediaCodec AV1 decoder gave no picture for %ld packets: the software decoder takes over at the next key frame", a->nin); a->sw_next = 1; }
            return 0;
        }
        if (!a->fw) read_format(a);
        if (a->zc) {   // rendered into the reader's buffers: the picture comes back through the reader (zc_pull), found by its codec timestamp
            int render = bi.size > 0;
            if (render) { zc_tsmap_push(&a->tsmap, bi.presentationTimeUs, game_pts(a, bi.presentationTimeUs)); a->nout++; if (a->av1 && !a->pictures++) LOG("[movie] AV1: first picture out of the decoder"); }
            AMediaCodec_releaseOutputBuffer(a->mc, (size_t)i, render);
            zc_pull(a, 0);
            if (bi.flags & AMEDIACODEC_BUFFER_FLAG_END_OF_STREAM) return 1;
            wait_us = 0; continue;
        }
        size_t n = 0; uint8_t *b = AMediaCodec_getOutputBuffer(a->mc, (size_t)i, &n);
        if (b && bi.size > 0) { keep_picture(a, b + bi.offset, (size_t)bi.size, bi.presentationTimeUs); a->nout++; if (a->av1 && !a->pictures++) LOG("[movie] AV1: first picture out of the decoder"); }
        AMediaCodec_releaseOutputBuffer(a->mc, (size_t)i, false);
        if (bi.flags & AMEDIACODEC_BUFFER_FLAG_END_OF_STREAM) return 1;
        wait_us = 0;
    }
}
typedef struct { mavc *a; size_t len; } au_ctx;
static void au_add(void *ctx, const uint8_t *nal, size_t n) {
    au_ctx *c = ctx; mavc *a = c->a; int t = nal[0] & 31;
    if (t == 7 || t == 8) { keep_ps(a, nal, n); return; }
    if (t == 9) return;   // access unit delimiter
    if (c->len + 4 + n > a->aucap) { a->aucap = (c->len + 4 + n) * 2; a->au = realloc(a->au, a->aucap); if (!a->au) die("movie: AVC buffer allocation failed"); }
    uint8_t *o = a->au + c->len; o[0] = o[1] = o[2] = 0; o[3] = 1; memcpy(o + 4, nal, n); c->len += 4 + n;
}
static ssize_t input_slot(mavc *a) {   // a free input buffer; while none is free, the pictures holding the codec up are collected
    for (int tries = 0; tries < 50; tries++) {
        ssize_t i = AMediaCodec_dequeueInputBuffer(a->mc, 2000);
        if (i >= 0) return i;
        drain(a, 0);
    }
    return -1;
}
// AV1: the packet as a temporal unit of low-overhead OBUs in a->au (movie_av1.c mav1_tu), its sequence header kept. The unit's length; 0 for nothing to decode (before the stream's first
// sequence header, a unit of temporal delimiters); -1 for a packet that is no AV1
static ssize_t av1_unit(mavc *a, const uint8_t *p, size_t n) {
    if (2 * n + 64 > a->aucap) { a->aucap = 2 * n + 64; a->au = realloc(a->au, a->aucap); if (!a->au) die("movie: AV1 buffer allocation failed"); }
    int framing = 0; ssize_t len = mav1_tu(p, n, a->au, a->aucap, &framing);
    if (len < 0) { static int k; if (k++ < 4) LOG("[RED] movie: an AV1 packet of %zu bytes that is neither low-overhead OBUs, Annex B nor IVF (%02x %02x %02x %02x)", n, n > 0 ? p[0] : 0, n > 1 ? p[1] : 0, n > 2 ? p[2] : 0, n > 3 ? p[3] : 0); return -1; }
    if (len && framing != a->framing) { a->framing = framing; LOG("[movie] AV1: packets are %s", mav1_framing_name(framing)); }
    size_t sl = 0; const uint8_t *s = mav1_find_seq(a->au, (size_t)len, &sl);
    if (s) {
        mav1_seq si;
        if (sl > sizeof a->seq || !mav1_seq_parse(s, sl, NULL, &si)) { if (!a->red++) LOG("[RED] movie: an AV1 sequence header the host cannot read (%zu bytes)", sl); }
        else if (sl != a->nseq || memcmp(s, a->seq, sl)) {
            if (a->nseq) LOG("[movie] AV1: a new sequence header (%dx%d), the decoder opens again", si.w, si.h);
            memcpy(a->seq, s, sl); a->nseq = sl; a->si = si; a->dirty = 1;
        }
        if (a->sw_next) { a->sw_next = 0; a->sw = 1; a->dirty = 1; }   // the software decoder from this key frame on
    }
    return a->nseq ? len : 0;
}
int mavc_decode(mavc *a, const uint8_t *p, int n, int64_t pts_us) {
    size_t len;
    if (a->av1) { ssize_t l = av1_unit(a, p, (size_t)n); if (l < 0) return -1; len = (size_t)l; }
    else { au_ctx c = {a, 0}; each_nal(p, (size_t)n, au_add, &c); len = c.len; }
    if (!len) return 0;

    if ((a->dirty || !a->mc || a->eos_sent) && !open_codec(a)) return -1;
    ssize_t i = input_slot(a);
    if (i < 0) { static long nd; long k = ++nd; if (!(k & (k - 1))) LOG("[RED] movie: MediaCodec %s decoder has no free input buffer, a packet is dropped (%ld so far)", KIND(a), k); return -1; }
    size_t cap = 0; uint8_t *b = AMediaCodec_getInputBuffer(a->mc, (size_t)i, &cap);
    if (!b || cap < len) { if (!a->red++) LOG("[RED] movie: %s access unit of %zu bytes does not fit MediaCodec's %zu-byte input buffer", KIND(a), len, cap); AMediaCodec_queueInputBuffer(a->mc, (size_t)i, 0, 0, (uint64_t)pts_us, 0); return -1; }
    memcpy(b, a->au, len);
    int64_t cpts = pts_us > a->last_in ? pts_us : a->last_in + 1; if (cpts != pts_us) a->synth = 1; a->last_in = cpts;
    if (a->npmap == 64) a->pmap[0] = a->pmap[--a->npmap];   // (outputs the codec never gave)
    a->pmap[a->npmap].c = cpts; a->pmap[a->npmap].g = pts_us; a->npmap++;
    if (AMediaCodec_queueInputBuffer(a->mc, (size_t)i, 0, len, (uint64_t)cpts, 0) != AMEDIA_OK) { if (!a->red++) LOG("[RED] movie: MediaCodec refused an %s access unit (first case)", KIND(a)); return -1; }
    a->nin++;
    drain(a, 0);
    return 0;
}
// the end of the stream: an end-of-stream input, then every picture still in the codec; the codec is opened afresh for the next stream (a loop restarts with Flush and new packets)
static void finish(mavc *a) {
    if (!a->mc || a->eos_sent) return;
    ssize_t i = input_slot(a);
    if (i >= 0) AMediaCodec_queueInputBuffer(a->mc, (size_t)i, 0, 0, 0, AMEDIACODEC_BUFFER_FLAG_END_OF_STREAM);
    a->eos_sent = 1;
    for (int k = 0; i >= 0 && k < 100 && !drain(a, 10000); k++) {}
    close_codec(a); a->eos_sent = 1;
}
// NV12 (video range) → RGBA rows of w pixels; BT.709 from 720 lines up, else BT.601 (VideoToolbox converts with the stream's colour matrix)
static void nv12_rgba(const uint8_t *y, const uint8_t *uv, int pw, int ph, uint8_t *dst, int w, int h) {
    int hd = ph >= 720, cw = pw < w ? pw : w, chh = ph < h ? ph : h;
    int rv = hd ? 459 : 409, gu = hd ? 55 : 100, gv = hd ? 136 : 208, bu = hd ? 541 : 516;   // ×256, 8-bit video range (luma × 298)
    if (cw < w || chh < h) for (size_t i = 0; i < (size_t)w * h; i++) ((uint32_t *)dst)[i] = 0xff000000u;
    for (int r = 0; r < chh; r++) {
        const uint8_t *yr = y + (size_t)r * pw, *cr = uv + (size_t)(r / 2) * pw; uint8_t *o = dst + (size_t)r * w * 4;
        for (int x = 0; x < cw; x++) {
            int c = 298 * (yr[x] - 16), u = cr[x & ~1] - 128, v = cr[(x & ~1) + 1] - 128;
            int R = (c + rv * v + 128) >> 8, G = (c - gu * u - gv * v + 128) >> 8, B = (c + bu * u + 128) >> 8;
            o[4 * x] = (uint8_t)(R < 0 ? 0 : R > 255 ? 255 : R); o[4 * x + 1] = (uint8_t)(G < 0 ? 0 : G > 255 ? 255 : G); o[4 * x + 2] = (uint8_t)(B < 0 ? 0 : B > 255 ? 255 : B); o[4 * x + 3] = 255;
        }
    }
}
// the surface mode's mavc_take: a rendered picture's image follows its release by a little (the codec's and the reader's threads), waited for up to 8 ms
int mavc_take_frame(mavc *a, int flush, zc_frame **out, int64_t *pts_us) {
    if (!a->zc) return 0;
    if (flush) { finish(a); zc_pull(a, 0); }
    else if (!a->nzq && a->mc && a->nin > a->nout) { drain(a, 4000); zc_pull(a, 0); }   // a picture is in the codec's pipeline: give it a few milliseconds
    for (int k = 0; !a->nzq && a->tsmap.n && k < 32; k++) { usleep(250); zc_pull(a, 0); }
    if (!a->nzq) return 0;
    *out = a->zq[0].f; if (pts_us) *pts_us = a->zq[0].pts;
    memmove(a->zq, a->zq + 1, --a->nzq * sizeof a->zq[0]);
    return 1;
}
int mavc_take(mavc *a, int flush, uint8_t *dst, int w, int h, int64_t *pts_us) {
    if (a->zc) return 0;   // (surface mode: mavc_take_frame)
    if (flush) finish(a);
    else if (!a->nq && a->mc && a->nin > a->nout) drain(a, 4000);   // a picture is in the codec's pipeline: give it a few milliseconds
    if (!a->nq) return 0;
    uint8_t *pic = a->q[0].nv12; int pw = a->q[0].w, ph = a->q[0].h; if (pts_us) *pts_us = a->q[0].pts;
    memmove(a->q, a->q + 1, --a->nq * sizeof a->q[0]);
    const uint8_t *uv = pic + (size_t)pw * ph;
    if (a->rgba) nv12_rgba(pic, uv, pw, ph, dst, w, h);
    else {
        int cw = pw < w ? pw : w, chh = ph < h ? ph : h;
        if (cw < w || chh < h) { memset(dst, 16, (size_t)w * h); memset(dst + (size_t)w * h, 128, (size_t)w * h / 2); }
        for (int r = 0; r < chh; r++) memcpy(dst + (size_t)r * w, pic + (size_t)r * pw, (size_t)cw);
        for (int r = 0; r < chh / 2 && r < h / 2; r++) memcpy(dst + (size_t)w * h + (size_t)r * w, uv + (size_t)r * pw, (size_t)cw);
    }
    free(pic);
    return 1;
}

// The stream's colour matrix (hle_movie.c): BT.709 as its SPS declares; MediaCodec does not expose the SPS's VUI here, so as upstream's undeclared case: from 720 lines up.
// AV1: as its sequence header says (mav1_seq_parse: unspecified is the same height rule)
int mavc_bt709(const mavc *a, int h) { return a->av1 && a->nseq ? a->si.matrix == 2 : h >= 720; }
mavc *mavc_new_av1(void) {
    mavc *a = mavc_new(); a->av1 = 1;
    const char *e = getenv("DM_AV1_SW"); a->sw = e && *e && *e != '0';
    return a;
}
