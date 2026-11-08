// Android port: the pause menu's typeface (overlay_font.c): a system TrueType font, antialiased, glyphs cached. Sizes are the em in pixels (a CSS font size).
#pragma once
int ovf_ready(void);          // 1 when a font is loaded (the first call looks for one), 0: draw the bitmap letters instead
const char *ovf_path(void);   // the font file in use, NULL none
float ovf_width(const char *s, float px, int bold);   // a UTF-8 string's advance width in pixels (bold: the synthetic bold)
float ovf_cap(float px);      // a capital letter's height in pixels
typedef void (*ovf_put)(void *ctx, int x, int y, float cov);   // one pixel's coverage, 0..1
// s with its left edge at x and its capitals centred on the line cy (each glyph at its quarter-pixel position)
void ovf_draw(const char *s, float x, float cy, float px, int bold, ovf_put put, void *ctx);
