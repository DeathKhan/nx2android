#include "../gfx_vk_ubo_extent.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint32_t words[512]; static size_t count; static unsigned checks;
#define EMIT(op, ...) do { uint32_t a[] = {__VA_ARGS__}; words[count++] = ((uint32_t)(sizeof a / sizeof *a + 1) << 16) | (op); memcpy(words + count, a, sizeof a); count += sizeof a / sizeof *a; } while (0)
static void fixture(void) {
    memset(words, 0, sizeof words); count = 5;
    words[0] = 0x07230203; words[1] = 0x00010300; words[3] = 128;
    EMIT(17, 1); EMIT(14, 0, 1);   // Shader; Logical GLSL450
    EMIT(15, 5, 16, 0x6e69616d, 0); EMIT(16, 16, 17, 1, 1, 1);   // compute main, LocalSize
    EMIT(71, 5, 6, 16); EMIT(71, 6, 2); EMIT(72, 6, 0, 35, 0);
    EMIT(71, 13, 34, 0); EMIT(71, 13, 33, 21);
    EMIT(22, 1, 32); EMIT(21, 2, 32, 1); EMIT(23, 3, 1, 4);
    EMIT(43, 2, 4, 4096); EMIT(28, 5, 3, 4); EMIT(30, 6, 5);
    EMIT(32, 7, 2, 6); EMIT(32, 8, 2, 3); EMIT(32, 9, 2, 1);
    EMIT(43, 2, 10, 0); EMIT(43, 2, 11, 136); EMIT(43, 2, 12, 3);
    EMIT(59, 7, 13, 2); EMIT(19, 14); EMIT(33, 15, 14);
    EMIT(54, 14, 16, 0, 15); EMIT(248, 17);
    EMIT(65, 9, 18, 13, 10, 11, 12); EMIT(61, 1, 19, 18);
}
static uint32_t *find(uint32_t op, unsigned operand, uint32_t value) {
    for (size_t i = 5; i < count; i += words[i] >> 16) if ((words[i] & 0xffff) == op && (words[i] >> 16) > operand && words[i + operand] == value) return words + i;
    assert(0); return NULL;
}
static void expect(const char *name, uint32_t extent) {
    uint32_t out[32]; for (unsigned i = 0; i < 32; i++) out[i] = UINT32_MAX;
    // The fixture has no pointer use after these instructions; an unused read still counts.
    vk_spv_ubo_extents(words, count, 0, out, 32);
    if (out[21] != extent) { fprintf(stderr, "%s: got %u, expected %u\n", name, out[21], extent); abort(); }
    checks++;
}
static void tests(void) {
    fixture(); expect("constant scalar read, rounded to vec4", 2192);
    fixture(); find(65, 2, 18)[0] = (7u << 16) | 66; expect("in-bounds chain", 2192);
    fixture(); uint32_t *ac = find(65, 2, 18); size_t pos = (size_t)(ac - words); ac[0] = (6u << 16) | 65; ac[1] = 8;
    memmove(words + pos + 6, words + pos + 7, (count - pos - 7) * 4); count--;
    find(61, 2, 19)[1] = 3; expect("whole vector read", 2192);
    fixture(); find(43, 2, 11)[3] = 0; expect("first element", 16);
    fixture(); find(43, 2, 11)[3] = 4094; expect("last useful extent", 65520);
    fixture(); find(43, 2, 11)[3] = 4095; expect("full binding", 0);
    fixture(); find(43, 2, 11)[3] = 4096; expect("array out of bounds", 0);
    fixture(); find(43, 2, 11)[3] = UINT32_MAX; expect("negative index", 0);
    fixture(); find(43, 2, 12)[3] = 4; expect("component out of bounds", 0);
    fixture(); find(65, 2, 18)[5] = 30; EMIT(45, 2, 30); expect("dynamic index", 0);
    fixture(); find(43, 2, 11)[0] = (4u << 16) | 50; expect("specialization constant", 0);
    fixture(); EMIT(83, 9, 20, 18); expect("pointer CopyObject escape", 0);
    fixture(); EMIT(57, 14, 20, 31, 18); expect("pointer function argument", 0);
    fixture(); EMIT(245, 9, 20, 18, 17, 18, 17); expect("pointer Phi", 0);
    fixture(); EMIT(169, 9, 20, 31, 18, 18); expect("pointer Select", 0);
    fixture(); EMIT(62, 31, 18); expect("pointer stored elsewhere", 0);
    fixture(); EMIT(65, 9, 20, 18, 10); expect("secondary access chain", 0);
    fixture(); EMIT(67, 9, 20, 13, 10, 11); expect("PtrAccessChain", 0);
    fixture(); EMIT(9999, 18); expect("unrecognized pointer use", 0);
    fixture(); EMIT(43, 2, 30, 13); expect("scalar literal equals root ID", 2192);
    fixture(); EMIT(12, 1, 20, 31, 13, 19); expect("extended instruction literal equals root ID", 2192);
    fixture(); EMIT(12, 1, 20, 31, 13, 18); expect("extended instruction pointer argument", 0);
    fixture(); find(61, 2, 19)[3] = 13; expect("whole-block read", 0);
    fixture(); find(61, 2, 19)[1] = 3; expect("load type mismatch", 0);
    fixture(); find(71, 2, 6)[3] = 32; expect("unsupported array stride", 0);
    fixture(); find(72, 3, 35)[4] = 16; expect("unsupported member offset", 0);
    fixture(); find(23, 1, 3)[3] = 3; expect("unsupported vec3 layout", 0);
    fixture(); find(22, 1, 1)[2] = 64; expect("unsupported scalar width", 0);
    fixture(); EMIT(71, 22, 34, 0); EMIT(71, 22, 33, 21); EMIT(59, 7, 22, 2); expect("second variable aliases descriptor", 0);
    fixture(); EMIT(63, 31, 18); expect("CopyMemory component source", 0);
    fixture(); EMIT(63, 31, 13); expect("CopyMemory whole block source", 0);
    fixture(); EMIT(64, 31, 18, 11); expect("CopyMemorySized", 0);
    fixture(); EMIT(117, 2, 20, 18); expect("pointer converted to integer", 0);
    fixture(); EMIT(124, 9, 20, 18); expect("pointer bitcast", 0);
    fixture(); EMIT(254, 18); expect("pointer return value", 0);
    fixture(); find(43, 2, 4)[0] = (4u << 16) | 50; expect("specialization array length", 0);
    fixture(); find(43, 2, 11)[0] = (4u << 16) | 52; expect("specialization expression index", 0);
    fixture(); EMIT(72, 6, 0, 4); expect("unsupported row-major layout", 0);
    fixture(); EMIT(71, 13, 33, 22); expect("conflicting descriptor binding", 0);
    fixture(); find(61, 2, 19)[0] = (5u << 16) | 61; words[count++] = 1; expect("optional volatile load operands", 0);
    fixture(); EMIT(73, 22); expect("decoration groups hide aliases", 0);
    fixture(); EMIT(71, 13, 33, 21); expect("duplicate decoration", 0);
    fixture(); EMIT(65, 9, 18, 13, 10, 11, 12); expect("duplicate pointer ID", 0);
    fixture(); words[count++] = 0; expect("zero instruction length", 0);
    fixture(); words[count++] = (100u << 16) | 65; expect("truncated instruction", 0);
    fixture(); words[3] = 1u << 30; expect("oversized ID bound", 0);
    fixture(); words[0] = 0; expect("invalid magic", 0);
    fixture(); uint32_t out[32]; vk_spv_ubo_extents(words, count, 1, out, 32); assert(out[21] == 0); checks++;
    fixture(); vk_spv_ubo_extents(words, count, 0, out, 21); for (int i = 0; i < 21; i++) assert(out[i] == 0); checks++;
    printf("ubo extent: %u layout/read/escape/alias/error checks passed\n", checks);
}
int main(int argc, char **argv) {
    tests();
    if (argc == 2) {
        FILE *f = fopen(argv[1], "rb"); assert(f); assert(!fseek(f, 0, SEEK_END)); long size = ftell(f); assert(size > 0 && !(size % 4)); rewind(f);
        uint32_t *w = malloc((size_t)size); assert(w); assert(fread(w, 1, (size_t)size, f) == (size_t)size); fclose(f);
        uint32_t out[96]; vk_spv_ubo_extents(w, (size_t)size / 4, 0, out, 96);
        for (int i = 0; i < 96; i++) if (out[i]) printf("binding %d: %u\n", i, out[i]);
        free(w);
    }
    return 0;
}
