// Android port: Apple's libcompression is not available. nvn_capture.c compresses chunks with LZFSE when it can: here encoding always fails (the chunk is stored raw)
// and an LZFSE chunk in a capture made on a Mac cannot be read (the reader reports it). TODO(upsync): zstd (dm/third_party/zstd) as the Android codec, with its own codec id.
#pragma once
#include <stddef.h>
#include <stdint.h>
typedef enum { COMPRESSION_LZ4 = 0x100, COMPRESSION_ZLIB = 0x205, COMPRESSION_LZMA = 0x306, COMPRESSION_LZFSE = 0x801 } compression_algorithm;
static inline size_t compression_encode_buffer(uint8_t *dst, size_t dst_size, const uint8_t *src, size_t src_size, void *scratch, compression_algorithm a) {
    (void)dst; (void)dst_size; (void)src; (void)src_size; (void)scratch; (void)a; return 0; }
static inline size_t compression_decode_buffer(uint8_t *dst, size_t dst_size, const uint8_t *src, size_t src_size, void *scratch, compression_algorithm a) {
    (void)dst; (void)dst_size; (void)src; (void)src_size; (void)scratch; (void)a; return 0; }
