// Audited encoded-RGB spatial reconstruction. Include after gfx_vk.h.
#pragma once
// The caller proves shader/binding/color metadata and supplies a single-level 2D sampled view.
// This helper accepts full selected subresources only, filters encoded RGB, then applies x*abs(x).
int vk_upscale_square(vkimg_t *src, VkImageView source_view, vkimg_t *dst, const VkImageBlit *region);
// Strict startup profile metadata; unknown or malformed lists never select this transfer.
int vk_upscale_square_key(uint64_t key);
