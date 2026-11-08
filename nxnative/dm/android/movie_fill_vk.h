// Movie fill (movie_fill_vk.c): a compute stage of a game's movie post-processing network whose shaders the device's driver cannot build is replaced by the decoded picture itself.
// The decoder's latest NV12 picture (hle_codec_video.h dm_codec_last_nv12: in the game's image pool memory, which is GPU memory) is converted by a small compute shader
// (dm/android/shaders/movie_fill.comp; build.py compiles it with glslc into gen/<game>/movie_fill_comp.spv.h) into an intermediate RGBA8 image of the picture's size, redone
// only when the picture changes, and that is blitted (scaled, converting formats) into the storage image the named stage would have written. gfx_vk_exec.c passthru_replace
// calls it for the shaders DM_VK_MOVIE_FILL_CS names (they must also be in DM_VK_PASSTHRU_CS, which routes a dispatch to passthru_replace)
#pragma once
#include <stdint.h>
struct vkimg;
// The fill over the whole of out's mip 0 layer 0, recorded in the current submission's main command buffer (gfx_vk.h vk_cmd); 1 recorded (out->gpu_w set);
// 0: no picture yet, the pool has no buffer, or the device refuses the blit: the caller goes on with its own replacement
int vk_movie_fill(struct vkimg *out);
// DM_VK_MOVIE_FILL_CS=<key>,<key>... (hex shader keys): this shader is named (parsed once; 0 when the variable is unset)
int vk_movie_fill_kernel(uint64_t key);
// The first call for a named key (the caller's one-time log line)
int vk_movie_fill_note(uint64_t key);
