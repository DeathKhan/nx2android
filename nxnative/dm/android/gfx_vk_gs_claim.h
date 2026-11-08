#ifndef DM_GFX_VK_GS_CLAIM_H
#define DM_GFX_VK_GS_CLAIM_H
#include <stdint.h>
#include <stddef.h>
/* Fresh per-draw storage block: uint enabled; uint claims[]. NULL leaves the module unchanged. */
uint32_t *spv_gs_claim(const uint32_t *, uint32_t, uint32_t set, uint32_t binding,
                       uint32_t *out_words, uint32_t *invocations, char *why, size_t why_size);
#endif
