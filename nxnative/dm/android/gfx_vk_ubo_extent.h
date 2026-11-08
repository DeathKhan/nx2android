// Conservative read extents for the translator's uniform blocks; no Vulkan dependency, shader changes or SPIR-V rewrites.
#pragma once
#include <stddef.h>
#include <stdint.h>
// bytes[binding] is the proven 16-byte-rounded read extent, or zero (unknown: keep the guest's full bound range).
// Only Uniform Block { 32-bit vec4[constant] } at Offset 0 / ArrayStride 16 and direct constant AccessChain + Load are recognized.
// Dynamic indices, aliases/escapes, other layouts and unsupported uses keep that binding unknown. Malformed framing keeps every binding unknown.
void vk_spv_ubo_extents(const uint32_t *words, size_t nwords, uint32_t descriptor_set, uint32_t *bytes, size_t nbindings);
