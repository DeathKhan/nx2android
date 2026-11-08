// Android port: the pause menu's typeface (overlay_font.h). A TrueType font from the system (Roboto on Android: the same file the host's shared font falls
// back to, hle_sdk16.c), rasterized antialiased with stb_truetype (dm/third_party/stb_truetype) into a glyph cache. The menu's panel is rasterized again only
// when the menu changes, so a glyph is made once per size and quarter-pixel position and copied after that. No font: ovf_ready is 0 and the menu keeps the 5x7
// bitmap letters (gfx_vk_present.c). DM_MENU_FONT=<path> chooses another file.
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "overlay_font.h"
#define STBTT_STATIC
#define STB_TRUETYPE_IMPLEMENTATION
#include "../third_party/stb_truetype/stb_truetype.h"

static pthread_mutex_t g_mu = PTHREAD_MUTEX_INITIALIZER;   // the present thread and the external display's Java thread both draw menus
static int g_state;   // 0 not tried, 1 loaded, -1 none
static stbtt_fontinfo g_font; static unsigned char *g_data; static float g_cap;   // g_cap: the capital H's height, of the em
static char g_path[160];
// The cache: open addressing on (code point, size in quarter pixels, weight, quarter-pixel x offset). Full, it is emptied (the menu uses a few sizes)
enum { OVF_SLOTS = 2048 };
typedef struct { uint32_t cp; uint16_t q; uint8_t bold, sub, used; int w, h, x0, y0; unsigned char *bm; } ovf_glyph;
static ovf_glyph g_cache[OVF_SLOTS]; static int g_used;

static int load(void) {
    static const char *const paths[] = {"/system/fonts/RobotoStatic-Regular.ttf", "/system/fonts/Roboto-Regular.ttf", "/system/fonts/NotoSans-Regular.ttf", "/system/fonts/DroidSans.ttf"};
    const char *env = getenv("DM_MENU_FONT");
    for (int i = -1; i < (int)(sizeof paths / sizeof paths[0]); i++) {
        const char *p = i < 0 ? env : paths[i]; if (!p || !*p) continue;
        FILE *f = fopen(p, "rb"); if (!f) continue;
        fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
        unsigned char *d = n > 0 && n < (64 << 20) ? malloc((size_t)n) : NULL;
        int ok = d && fread(d, 1, (size_t)n, f) == (size_t)n; fclose(f);
        int off = ok ? stbtt_GetFontOffsetForIndex(d, 0) : -1;
        if (off < 0 || !stbtt_InitFont(&g_font, d, off) || !stbtt_FindGlyphIndex(&g_font, 'A')) { free(d); continue; }
        g_data = d; snprintf(g_path, sizeof g_path, "%s", p);
        int x0, y0, x1, y1; float em = stbtt_ScaleForMappingEmToPixels(&g_font, 1);
        g_cap = stbtt_GetCodepointBox(&g_font, 'H', &x0, &y0, &x1, &y1) ? y1 * em : 0.7f;
        fprintf(stderr, "[menu] font %s (%ld bytes)\n", p, n);
        return 1;
    }
    fprintf(stderr, "[menu] no TrueType font found: the menu uses the bitmap letters\n");
    return -1;
}
int ovf_ready(void) {
    pthread_mutex_lock(&g_mu); if (!g_state) g_state = load(); int r = g_state > 0; pthread_mutex_unlock(&g_mu); return r;
}
const char *ovf_path(void) { return g_state > 0 ? g_path : NULL; }

static uint32_t utf8_next(const char **s) {   // the next code point of a UTF-8 string (a broken sequence: U+FFFD, one byte on)
    const unsigned char *p = (const unsigned char *)*s; uint32_t c = *p;
    int n = c < 0x80 ? 0 : (c & 0xe0) == 0xc0 ? 1 : (c & 0xf0) == 0xe0 ? 2 : (c & 0xf8) == 0xf0 ? 3 : -1;
    if (n < 0) { *s += 1; return 0xfffd; }
    c &= n == 0 ? 0x7f : n == 1 ? 0x1f : n == 2 ? 0x0f : 0x07;
    for (int i = 1; i <= n; i++) { if ((p[i] & 0xc0) != 0x80) { *s += 1; return 0xfffd; } c = c << 6 | (p[i] & 0x3f); }
    *s += n + 1; return c;
}
static float bold_px(float px) { return px * 0.045f; }   // the synthetic bold's widening (no bold Roboto file on Android 12+: Roboto-Regular.ttf is the variable font)
static const ovf_glyph *glyph(uint32_t cp, float px, int bold, int sub) {
    uint16_t q = (uint16_t)lroundf(px * 4); uint32_t h = (cp * 2654435761u) ^ (q * 40503u) ^ (uint32_t)(bold << 7 | sub) * 97u;
    for (int k = 0; k < OVF_SLOTS; k++) {
        ovf_glyph *g = &g_cache[(h + k) % OVF_SLOTS];
        if (g->used && g->cp == cp && g->q == q && g->bold == bold && g->sub == sub) return g;
        if (g->used) continue;
        if (g_used > OVF_SLOTS * 3 / 4) {   // full: start again
            for (int i = 0; i < OVF_SLOTS; i++) free(g_cache[i].bm);
            memset(g_cache, 0, sizeof g_cache); g_used = 0; return glyph(cp, px, bold, sub);
        }
        float sc = stbtt_ScaleForMappingEmToPixels(&g_font, q / 4.0f); int w, hh, x0, y0;
        unsigned char *bm = stbtt_GetCodepointBitmapSubpixel(&g_font, sc, sc, sub / 4.0f, 0, (int)cp, &w, &hh, &x0, &y0);
        if (bm && bold) {   // widened to the right: each texel the most of itself and the ones to its left, the last one in part
            float e = bold_px(q / 4.0f); int ei = (int)ceilf(e), nw = w + ei; float part = e - (ei - 1);
            unsigned char *nb = calloc((size_t)nw * hh, 1);
            if (nb) {
                for (int y = 0; y < hh; y++) for (int x = 0; x < nw; x++) {
                    float m = 0;
                    for (int t = 0; t <= ei; t++) { int sx = x - t; if (sx < 0 || sx >= w) continue; float v = bm[y * w + sx] * (t == ei ? part : 1); if (v > m) m = v; }
                    nb[y * nw + x] = (unsigned char)(m + 0.5f);
                }
                stbtt_FreeBitmap(bm, NULL); bm = nb; w = nw;
            }
        }
        *g = (ovf_glyph){cp, q, (uint8_t)bold, (uint8_t)sub, 1, bm ? w : 0, bm ? hh : 0, x0, y0, bm}; g_used++;
        return g;
    }
    return NULL;
}
static float advance(uint32_t cp, uint32_t next, float px, int bold) {
    float sc = stbtt_ScaleForMappingEmToPixels(&g_font, px); int adv, lsb; stbtt_GetCodepointHMetrics(&g_font, (int)cp, &adv, &lsb);
    float a = adv * sc + (bold ? bold_px(px) : 0);
    if (next) a += stbtt_GetCodepointKernAdvance(&g_font, (int)cp, (int)next) * sc;
    return a;
}
float ovf_width(const char *s, float px, int bold) {
    if (!ovf_ready()) return 0;
    pthread_mutex_lock(&g_mu); float w = 0; const char *p = s;
    while (*p) { uint32_t c = utf8_next(&p); const char *q = p; uint32_t n = *q ? utf8_next(&q) : 0; w += advance(c, n, px, bold); }
    pthread_mutex_unlock(&g_mu); return w;
}
float ovf_cap(float px) { return g_cap * px; }
void ovf_draw(const char *s, float x, float cy, float px, int bold, ovf_put put, void *ctx) {
    if (!ovf_ready()) return;
    pthread_mutex_lock(&g_mu);
    float base = roundf(cy + g_cap * px / 2);   // the capitals centred on cy, on a whole pixel's baseline
    const char *p = s;
    while (*p) {
        uint32_t c = utf8_next(&p); const char *q = p; uint32_t n = *q ? utf8_next(&q) : 0;
        int ix = (int)floorf(x), sub = (int)((x - ix) * 4) & 3; const ovf_glyph *g = glyph(c, px, bold, sub);
        if (g && g->bm) for (int y = 0; y < g->h; y++) for (int xx = 0; xx < g->w; xx++) {
            unsigned char v = g->bm[y * g->w + xx]; if (v) put(ctx, ix + g->x0 + xx, (int)base + g->y0 + y, v / 255.0f);
        }
        x += advance(c, n, px, bold);
    }
    pthread_mutex_unlock(&g_mu);
}
