// Android port: zero-copy movie pictures. MediaCodec decodes into an AImageReader's buffers (AHardwareBuffers), each picture is imported into Vulkan as a Y'CbCr image and
// converted on the GPU into the game's texture over the decoder pool's picture slot, with no CPU copy (movie_avc.c decodes, hle_codec_video.c shows, gfx_vk_tex.c samples).
// Three parts: movie_zc.c (no Android or Vulkan: picture slots, frame references, release once the GPU is done; unit-tested on the host, dm/android/tests/t_movie_zc.c),
// movie_zc_vk.c (the AImageReader, the import and the conversion dispatch), movie_avc.c (the codec's surface mode: mavc_set_zero_copy, mavc_take_frame).
// DM_MOVIE_ZEROCOPY=0 (start-up) keeps the CPU path (NV12 copies converted by the host, as before)
#pragma once
#include <stdint.h>
#include <stddef.h>

typedef struct zc_frame zc_frame;     // one decoded picture: an AImage and its AHardwareBuffer, reference counted (the codec gets the buffer back when the last reference goes and the GPU has read it)
typedef struct zc_reader zc_reader;   // the AImageReader the codec renders into (kept alive by its frames out)

// ---- frames (movie_zc.c) ----
void zc_frame_ref(zc_frame *f);
void zc_frame_unref(zc_frame *f);
int zc_frame_w(const zc_frame *f);     // the buffer's size
int zc_frame_h(const zc_frame *f);
int zc_frame_crop(const zc_frame *f, int *x, int *y, int *w, int *h);   // the picture inside the buffer (AImage_getCropRect; the whole buffer when none)
int64_t zc_frame_ts(const zc_frame *f);   // the buffer's timestamp (ns; MediaCodec's presentation time × 1000)
void zc_frame_used(zc_frame *f, uint64_t fence);   // the GPU reads it in the submission that commits with this fence (movie_zc_vk.c): it is not given back before that fence completes
void zc_reap(void);   // frames whose last reference went while the GPU still read them: given back once their fence completed (called at every slot change and lookup; cheap)
int zc_frames_live(void);   // frames not yet given back (diagnostics, tests)

// ---- picture slots (movie_zc.c; hle_codec_video.c fills, gfx_vk_tex.c looks up) ----
// A slot is the guest memory of one picture of a decoder image pool, which the game's native-image texture is made over. Showing a frame there takes a reference; the previous frame's goes
void zc_slot_set(const void *mem, uint64_t size, zc_frame *f);   // f NULL: the slot shows nothing (the texture keeps what it has)
void zc_slot_clear(const void *mem, uint64_t size);              // the pool is finalized: every slot inside [mem, mem+size) goes
int zc_slot_find(const void *mem, uint64_t len, zc_frame **f, uint64_t *gen);   // a texture over [mem, mem+len) lies in a slot: 1, with the frame shown (a reference is taken; NULL when none yet) and the slot's change count
int zc_slots_live(void);

// ---- timestamp map (movie_zc.c; movie_avc.c): the codec's timestamps of pictures rendered to the surface, waiting for their AImage ----
typedef struct { struct { int64_t c, g; } e[64]; int n; } zc_tsmap;
void zc_tsmap_push(zc_tsmap *m, int64_t codec_us, int64_t game_us);   // full: the oldest goes
int zc_tsmap_take(zc_tsmap *m, int64_t codec_us, int64_t *game_us);   // 1 and the game's time for a rendered picture's codec time, 0 when unknown (an image from before a flush)

// ---- Android / Vulkan side (movie_zc_vk.c) ----
int zc_gpu_ok(void);   // the device can take zero-copy pictures (the Vulkan device is up with Y'CbCr sampler conversion and the AHardwareBuffer import)
zc_reader *zc_reader_new(int w, int h, int max_images);   // NULL when it could not be made
void zc_reader_free(zc_reader *r);     // the decoder is done with it: it goes once its frames out are given back
void *zc_reader_window(zc_reader *r);  // its ANativeWindow for AMediaCodec_configure
zc_frame *zc_reader_acquire(zc_reader *r);   // the next picture the codec rendered (one reference), NULL when none is ready
struct vkimg;
int zc_tex_wants(const void *mem, uint64_t len);   // gfx_vk_tex.c img_new: a texture over a picture slot (its image needs storage usage)
int zc_tex_refresh(struct vkimg *im, const void *mem, uint64_t len);   // gfx_vk_tex.c vk_tex: 1 = the texture is a slot's (nothing is uploaded from its storage); the slot's current frame is converted into it when it changed
// movie_zc.c needs these two from movie_zc_vk.c (weak there: the host test gives its own)
uint64_t zc_fence_done(void);
void zc_frame_destroy(zc_frame *f);   // the last reference went and the GPU is done: AImage_delete, and the reader's reference goes

// ---- the H.264 decoder's surface mode (movie_avc.c) ----
typedef struct mavc mavc;
int mavc_set_zero_copy(mavc *a, int on, int w, int h);   // pictures come out as frames (mavc_take_frame) instead of NV12 copies (mavc_take); w, h: the stream's size as the game declared it. Returns the mode in effect
int mavc_take_frame(mavc *a, int flush, zc_frame **out, int64_t *pts_us);   // the next picture in presentation order: 1 and a frame (one reference), 0 none yet; flush as mavc_take

// movie_zc.c: the frame type as the Android side fills it (movie_zc_vk.c) and the tests build it
struct zc_frame {
    void *img, *ahb;            // AImage*, AHardwareBuffer*
    uint64_t id;                // AHardwareBuffer_getId: the Vulkan import cache's key (movie_zc_vk.c)
    int w, h, cx, cy, cw, ch;   // the buffer and the picture's crop rectangle in it
    int64_t ts_ns;
    int refs;                   // references out (slots, the decoder's queue, hle_codec_video's held ring)
    uint64_t fence;             // the last submission fence the GPU reads it in (0: never)
    zc_reader *r;
    zc_frame *next;             // zc_reap's pending list
};
