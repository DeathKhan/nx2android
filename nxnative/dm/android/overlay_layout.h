// Android port: the on-screen gamepad's layout, shared by the touch handling (front_android.c) and its drawing (gfx_vk_present.c, with the app's theme: overlay_theme.c).
// Two layouts, the tablet's and the phone's (dm/android/overlay_default.tsv): a control's centre is x window heights from its anchor (the left edge, the middle
// or the right edge: a thumb reaches as far from the edge on any width) and y of the window's height down; its size is in window heights (round controls stay round)
#pragma once
#include <math.h>
#include <stddef.h>
#include "dm_input.h"
enum { OV_STICK_L, OV_STICK_R, OV_BUTTON, OV_DPAD, OV_MENU };   // OV_MENU: the menu button (always shown, 70%): pauses the game and opens the settings
typedef struct { int kind; float ax, x, y, w, h; uint64_t btn; const char *label, *name; } ov_ctl;   // ax: the anchor, as a fraction of the window's width; name: the table's
// The layouts: dm/android/overlay_default.tsv with the game profile's moves (android.tsv overlay rows), compiled by build.py into android_profile.h. Both list the
// same controls in the same order, so a control's index means the same in either (a press keeps it across a change of layout)
#include "android_profile.h"
static const ov_ctl OV_LAYOUTS[2][DM_AOV_N] = {{DM_AOV_TABLET}, {DM_AOV_PHONE}};
#define OV_N DM_AOV_N
// the layout in use: the phone's when the window is at least OV_PHONE_ASPECT wide for its height (a 20:9 phone is 2.2, a 16:10 tablet 1.6), else the tablet's
#define OV_PHONE_ASPECT 1.9f
static inline int ov_layout_for_size(int width, int height) { return height > 0 && (float)width / (float)height >= OV_PHONE_ASPECT; }
int dm_overlay_layout(void);   // front_android.c: 0 the tablet's, 1 the phone's
#define OV_CTLS (OV_LAYOUTS[dm_overlay_layout()])
#define OV_KNOB 0.42f        // a stick's knob radius, of the stick's (the theme-less drawing)
#define OV_MENU_ALPHA 0.70f
#define OV_FLASH_NS 180000000ll   // a pressed button's flash fades to its held glow in this time
// front_android.c → gfx_vk_present.c: what to draw (0 nothing: a controller is attached, and its keys open the menu; 1 the gamepad; 2 only the menu
// button: the player hid the gamepad, or keys came from a device not counted as a controller), where a stick's knob is (its offset from the stick's centre
// in stick radii, -1..1), and when each control was pressed (0 = not held; for the d-pad, its four directions by DM_BTN_* bit; a stick: while a finger holds it)
int dm_overlay_visible(void);
void dm_overlay_knob(int stick, float *dx, float *dy);
int64_t dm_overlay_pressed(int ctl, uint64_t bit);   // CLOCK_MONOTONIC ns of the press, 0 = not held
// the player's own layout (the layout editor; the menu button can also be dragged any time, where it hides nothing of the game's): a control's centre as
// fractions of the window's width and height and its size factor, per layout, kept in settings.txt. 1 and x, y when the player moved the control, 0 when it
// is where the layout puts it; *s is its size factor either way (1 = the layout's)
int dm_overlay_custom(int layout, int ctl, float *x, float *y, float *s);
int dm_overlay_custom_ver(void);   // counts the changes (the theme-less drawing is rasterized again)
float dm_overlay_opacity(void);    // the gamepad's opacity, 0.2..0.8 (the menu's Opacity slider)
static inline int ov_index(const ov_ctl *c, int *layout) { int k = (int)(c - &OV_LAYOUTS[0][0]); *layout = k / OV_N; return k % OV_N; }   // c: in OV_LAYOUTS
static inline float ov_cscale(const ov_ctl *c) { int l, i = ov_index(c, &l); float x, y, s = 1; dm_overlay_custom(l, i, &x, &y, &s); return s; }
#define OV_DRAG_SLOP 0.02f   // a press on the menu button that moves this far (of the window's height) drags it instead of opening the menu
// the controls' size factor: 1, or more on a small screen so a face button stays at least OV_MIN_MM wide (sizes follow the window's height, which on a
// 6.7" phone made them 7 mm against 12 on an 8.8" tablet). Positions stay where they are
#define OV_MIN_MM 10.0f
float dm_overlay_scale(void);
// where a control is now, in window pixels: its centre x, y and half its width and height
static inline void ov_box(const ov_ctl *c, float W, float H, float b[4]) {   // (c: in OV_LAYOUTS, whose index finds the player's own place and size)
    int l, i = ov_index(c, &l); float s = dm_overlay_scale(), mx, my, cs = 1;
    if (dm_overlay_custom(l, i, &mx, &my, &cs)) { b[0] = mx * W; b[1] = my * H; }
    else { b[0] = c->ax * W + c->x * H; b[1] = c->y * H; }
    b[2] = c->w * H * s * cs / 2; b[3] = c->h * H * s * cs / 2;
}
// a dragged control's centre on one axis (half: its half size, extent: the window's): kept on the screen, and with step > 0 on the nearest multiple
// of step that keeps it there (none may fit an oversized control)
static inline float ov_drag_pos(float p, float half, float extent, float step) {
    float lo = fminf(half, extent / 2), hi = extent - lo;
    if (step > 0) {
        float first = ceilf(lo / step), last = floorf(hi / step);
        if (first <= last) return fminf(fmaxf(roundf(p / step), first), last) * step;
    }
    return fminf(fmaxf(p, lo), hi);
}
// its corners' radius: a round control (as wide as high), the menu button (a pill) and the d-pad (its diagonals) are round; the shoulder buttons are rounded boxes
static inline float ov_corner(const ov_ctl *c, const float b[4]) { float m = fminf(b[2], b[3]); return c->w == c->h || c->kind == OV_MENU ? m : m * 0.35f; }
// the signed distance from (dx, dy), an offset from a rounded box's centre, to its edge (negative inside); hw, hh: half its width and height, rad: its corners
static inline float ov_box_sdf(float dx, float dy, float hw, float hh, float rad) {
    float qx = fabsf(dx) - hw + rad, qy = fabsf(dy) - hh + rad, px = qx > 0 ? qx : 0, py = qy > 0 ? qy : 0;
    return sqrtf(px * px + py * py) + fminf(fmaxf(qx, qy), 0) - rad;
}
// the same for a control (b: ov_box), (dx, dy) in pixels from its centre
static inline float ov_sdf(const ov_ctl *c, const float b[4], float dx, float dy) { return ov_box_sdf(dx, dy, b[2], b[3], ov_corner(c, b)); }
#define OV_REACH_MIN 0.045f   // a small button (-, +, the menu button) still takes touches this far from its centre (of the window's height)
// how far outside its edge a control still takes a touch, in pixels: a button a quarter of its size (at least to OV_REACH_MIN), a stick or the d-pad half
static inline float ov_reach(const ov_ctl *c, const float b[4], float H) {
    float m = fminf(b[2], b[3]);
    return c->kind == OV_BUTTON || c->kind == OV_MENU ? fmaxf(m * 0.25f, OV_REACH_MIN * H * dm_overlay_scale() - m) : m * 0.5f;
}
// A small area between adjacent face buttons lets one thumb press both without changing their individual hit areas.
static inline uint64_t ov_face_combo(const ov_ctl *layout, float x, float y, float W, float H) {
    const uint64_t face = DM_BTN_A | DM_BTN_B | DM_BTN_X | DM_BTN_Y;
    for (int i = 0; i < OV_N; i++) {
        const ov_ctl *a = &layout[i]; if (a->kind != OV_BUTTON || !(a->btn & face)) continue;
        for (int j = i + 1; j < OV_N; j++) {
            const ov_ctl *b = &layout[j]; uint64_t pair = a->btn | b->btn;
            if (b->kind != OV_BUTTON || !(b->btn & face) || a->btn == b->btn || pair == (DM_BTN_A | DM_BTN_Y) || pair == (DM_BTN_X | DM_BTN_B)) continue;
            float ab[4], bb[4]; ov_box(a, W, H, ab); ov_box(b, W, H, bb);
            float distance = hypotf(ab[0] - bb[0], ab[1] - bb[1]);
            float radius = fminf(fminf(ab[2], ab[3]), fminf(bb[2], bb[3]));
            if (distance < radius || distance > radius * 3.5f) continue;
            if (hypotf(x - (ab[0] + bb[0]) * 0.5f, y - (ab[1] + bb[1]) * 0.5f) <= radius * 0.35f) return pair;
        }
    }
    return 0;
}
static inline uint64_t ov_button_bits(const ov_ctl *layout, const ov_ctl *button, float x, float y, float W, float H) {
    uint64_t pair = (button->btn & (DM_BTN_A | DM_BTN_B | DM_BTN_X | DM_BTN_Y)) ? ov_face_combo(layout, x, y, W, H) : 0;
    return pair && (pair & button->btn) ? pair : button->btn;
}
// a d-pad direction's circle in the theme-less drawing: centre offset and radius, in the d-pad's half width (k: 0 right, 1 left, 2 down, 3 up)
static inline void ov_dpad_dir(int k, float *ox, float *oy, float *r) { float s = 0.42f; *ox = k == 0 ? s * 1.2f : k == 1 ? -s * 1.2f : 0; *oy = k == 2 ? s * 1.2f : k == 3 ? -s * 1.2f : 0; *r = s * 0.75f; }
static const uint64_t OV_DPAD_BITS[4] = {DM_BTN_RIGHT, DM_BTN_LEFT, DM_BTN_DOWN, DM_BTN_UP};

// ---- the app's theme (overlay_theme.c): the gamepad drawn as its design's sprites, from the APK's assets/overlay (porter packs the game's pack from
// dm/android/themes). A sprite belongs to a control (and, for the d-pad, a direction), is drawn in one of its states, and sits at an offset from the control's
// centre; offsets and sizes are the design's CSS pixels, which the window's height scales (the design's height = the window's) ----
enum { OVT_IDLE, OVT_HELD, OVT_KNOB, OVT_DECOR };   // a button or a stick's base at rest or held; a stick's knob; a decoration drawn under the controls
typedef struct { int ctl, dir, state; float u, v, w, h, ox, oy; } ov_tsprite;   // dir: OV_DPAD_BITS's index, -1 none; u, v, w, h: in the atlas, texels; ox, oy: top left, CSS px
typedef struct { int layout, w, h; uint8_t *px; float dh, scale, travel, alpha; int n; ov_tsprite *sp; } ov_theme;   // px: w x h RGBA; dh: the design's height;
                                                                                                                       // scale: texels per CSS px; travel: the knob's, CSS px
const ov_theme *dm_overlay_theme(int layout);   // the theme's sprites for a layout, NULL when the app has none (it keeps its pixels until dm_overlay_theme_done)
void dm_overlay_theme_done(const ov_theme *t);   // the pixels are on the GPU: their memory can go

// ---- the pause menu: a panel in the middle of the window: a header with the game's name, a row of tabs, the open tab's rows, Resume, and a line of the
// controller's keys. Everything is placed in a panel rectangle p (x, y, width, height: the window's for touches, the raster's for drawing), so the touches
// and the drawing agree. The rows are data (OVM_ROW): a row is a kind (OVR_*) with its label and, for a choice, the names of its values; a tab lists its
// rows (OVM_TAB_ITEMS). front_android.c keeps each row's value (dm_menu_value) and changes it (menu_set); gfx_vk_present.c draws a row by its kind.
// A new choice row (one that cycles through values with left and right, "< value >"): an OVM_* item, its OVM_ROW entry, its place in a tab, and its value
// in front_android.c dm_menu_value and menu_set ----
enum { OVM_OVERLAY, OVM_OPACITY, OVM_EDIT, OVM_RUMBLE, OVM_FPS, OVM_MODE, OVM_EXPORT, OVM_IMPORT, OVM_ACCENT, OVM_FRAMECAP, OVM_FPSPOS, OVM_PADLABEL, OVM_TEMPS, OVM_RESUME, OVM_TAB, OVM_N };   // OVM_TAB: the tabs' row;
            // OVM_EDIT: the layout editor; OVM_MODE: Resolution, the level (host/dm.h DM_RES_*: Max is docked, the others handheld; a game without its own
            // dynamic-resolution settings has two, Balanced = handheld and Max = docked); OVM_EXPORT, OVM_IMPORT: the save manager (saves_android.c);
            // OVM_ACCENT: the menu's highlight colour (OVM_ACCENTS); OVM_FRAMECAP: the frame cap (Off, 60, 30 fps: OVM_CAPS; host/hle_framecap.c);
            // OVM_FPSPOS: the frame rate counter's corner (OVM_FPSPOS_KEYS); OVM_PADLABEL: a controller's face buttons by position or by label (front_android.c pad_labels)
enum { OVR_TOGGLE, OVR_SLIDER, OVR_CHOICE, OVR_ACTION, OVR_TABS, OVR_RESUME };   // a switch; a slider (Opacity); values cycled with left and right; a button
                                                                                 // (its word: button); the tabs; Resume
typedef struct { int kind; const char *label, *sub; const char *const *choices; int n; const char *button; const unsigned *swatches; const char *na; } ov_mrow;
            // sub: a line under the label, or NULL; swatches: a choice of colours (0xRRGGBB, n of them), shown as a row of them, the chosen one's name under the label;
            // na: what a choice shows, greyed and without its arrows, when the game cannot take it (dm_menu_value OVM_NA)
#define OVM_NA (-2)   // dm_menu_value: the row does not apply to this game (its na text is shown; left, right and A do nothing)
static const char *const OVM_RES_NAMES[] = {"Performance", "Balanced", "Quality", "Max"};   // DM_RES_* (front_android.c dm_menu_choice_name: Handheld, Docked for two)
#define OVM_NCAPS 3
static const int OVM_CAPS[OVM_NCAPS] = {0, 60, 30};   // the frame cap's values, fps (0 = off: the game's own rate; settings.txt fpscap=)
static const char *const OVM_CAP_NAMES[OVM_NCAPS] = {"Off", "60 fps", "30 fps"};
// the frame rate counter's corner (settings.txt fpspos=; bit 0 right, bit 1 bottom): top left unless the player moves it (Welcome Tour draws its Area banner there)
static const char *const OVM_FPSPOS_NAMES[4] = {"Top left", "Top right", "Bottom left", "Bottom right"};
static const char *const OVM_FPSPOS_KEYS[4] = {"tl", "tr", "bl", "br"};
static const char *const OVM_PADLABEL_NAMES[2] = {"Position", "Label"};
int dm_overlay_fps_pos(void);   // front_android.c: the counter's corner, 0..3
// the highlight colours (settings.txt accent=RRGGBB; another value is kept and shown as Custom). OVM_ACCENT_DEFAULT (Orange) is the colour of a settings.txt
// without accent= (Neon pink, the first, was the menu's colour before there was a choice and stays one)
#define OVM_ACCENT_DEFAULT 0xFF7A1A
#define OVM_ACCENTS_N 9
static const unsigned OVM_ACCENTS[OVM_ACCENTS_N] = {0xFF3C8E, 0xE4002B, 0xFF7A1A, 0xF6C431, 0x2FC46B, 0x16B8AE, 0x3B82F6, 0x8B5CF6, 0xE9E9EF};
static const char *const OVM_ACCENT_NAMES[OVM_ACCENTS_N] = {"Neon pink", "Red", "Orange", "Yellow", "Green", "Teal", "Blue", "Purple", "White"};
static const ov_mrow OVM_ROW[OVM_N] = {
    [OVM_OVERLAY] = {OVR_TOGGLE, "Show gamepad", "Touch controls on the screen"},
    [OVM_OPACITY] = {OVR_SLIDER, "Opacity", "How solid the touch controls are"},
    [OVM_EDIT]    = {OVR_ACTION, "Edit layout", "Move and resize the touch controls", NULL, 0, "Edit"},
    [OVM_RUMBLE]  = {OVR_TOGGLE, "Rumble", "Vibration and tap feedback"},
    [OVM_FPS]     = {OVR_TOGGLE, "Show FPS", "Frame rate in the corner"},
    [OVM_TEMPS]   = {OVR_TOGGLE, "Show temperatures", "Chip and battery, beside the FPS"},
    [OVM_MODE]    = {OVR_CHOICE, "Resolution", "Render size; Max renders as docked", OVM_RES_NAMES, 4},
    [OVM_EXPORT]  = {OVR_ACTION, "Export save", "Copy the save to the files folder", NULL, 0, "Export"},
    [OVM_IMPORT]  = {OVR_ACTION, "Import save", "Replace the save with an exported one", NULL, 0, "Import"},
    [OVM_ACCENT]  = {OVR_CHOICE, "Highlight colour", NULL, OVM_ACCENT_NAMES, OVM_ACCENTS_N, NULL, OVM_ACCENTS},
    [OVM_FPSPOS]  = {OVR_CHOICE, "FPS position", "Corner of the FPS and temperatures", OVM_FPSPOS_NAMES, 4},
    [OVM_PADLABEL] = {OVR_CHOICE, "Face buttons", "A, B, X and Y by place or by their letters", OVM_PADLABEL_NAMES, 2},
    [OVM_FRAMECAP] = {OVR_CHOICE, "Frame cap", "Hold the game to a steady rate", OVM_CAP_NAMES, OVM_NCAPS, NULL, NULL, "Not for this game"},
    [OVM_RESUME]  = {OVR_RESUME, "Resume"},
    [OVM_TAB]     = {OVR_TABS, ""},
};
#define OVM_TABS 3
#define OVM_PER_TAB 5   // the most rows a tab has (every tab's panel is as tall); -1: no row there (up and down skip it)
static const int OVM_TAB_ITEMS[OVM_TABS][OVM_PER_TAB] = {{OVM_FPS, OVM_TEMPS, OVM_FRAMECAP, OVM_MODE, -1}, {OVM_OVERLAY, OVM_OPACITY, OVM_EDIT, OVM_RUMBLE, OVM_PADLABEL},
                                                         {OVM_ACCENT, OVM_FPSPOS, OVM_EXPORT, OVM_IMPORT, -1}};
static const char *const OVM_TAB_NAMES[OVM_TABS] = {"Game", "Controls", "System"};
#define OVM_ROWS (OVM_PER_TAB + 2)   // the tabs, a tab's rows, Resume
static inline int ov_menu_item(int tab, int row) {
    if (tab < 0 || tab >= OVM_TABS) tab = 0;
    return row == 0 ? OVM_TAB : row == OVM_ROWS - 1 ? OVM_RESUME : row > 0 && row < OVM_ROWS - 1 ? OVM_TAB_ITEMS[tab][row - 1] : -1;
}
static inline int ov_menu_kind(int item) { return item >= 0 && item < OVM_N ? OVM_ROW[item].kind : -1; }
int dm_menu_value(int item);   // front_android.c: a row's value: a switch 0 or 1, the slider's percent, a choice's index (-1: none of them, e.g. a custom colour;
                               // OVM_NA: the row does not apply to the game)
const char *dm_menu_choice_name(int item, int v);   // front_android.c: the name a choice row shows for value v (the table's, or the game's own: Resolution's two)
unsigned dm_menu_accent(void); // front_android.c: the highlight colour, 0xRRGGBB
int dm_game_title(char *out, int cap);   // front_android.c: the app's name (the menu's header), 0 when there is none
// The panel's parts, of the window's height (times ov_menu_scale): the padding, the header, the tabs, the gap around the rows, a row, the key hints' line
#define OV_MENU_W 1.12f      // panel width
#define OV_MENU_PAD 0.032f
#define OV_MENU_HEAD 0.088f
#define OV_MENU_TABH 0.078f
#define OV_MENU_GAP 0.018f
#define OV_MENU_ROW 0.094f
#define OV_MENU_FOOT 0.072f
#define OV_MENU_H (2 * OV_MENU_PAD + OV_MENU_HEAD + OV_MENU_TABH + 2 * OV_MENU_GAP + (OVM_ROWS - 1) * OV_MENU_ROW + OV_MENU_FOOT - 0.02f)   // panel height
#define OV_OPACITY_MIN 20    // the gamepad's opacity, percent (the Opacity slider, settings.txt opacity=)
#define OV_OPACITY_MAX 80
// what the menu shows (-1 = closed): bit 0 the gamepad is on, bit 1 rumble is on, bit 2 docked, bit 3 the frame rate shown, bits 4..7 the highlighted row + 1
// (0 = none), bits 8..9 the open tab, bit 10 a finger is down on the highlighted row, bits 20..23 the last export's outcome (OVX_*), bits 24..27 a count of
// exports (its name changes with each). The rows' values are dm_menu_value's. Bit 16: the text entry's panel instead (its edit count below); bit 17: the
// layout editor (OVE_ST_* below); bit 18: the save picker (OVS_ST)
int dm_menu_state(void);
#define OVM_PRESSED 0x400
enum { OVX_IDLE, OVX_BUSY, OVX_DONE, OVX_NOSAVE, OVX_FAILED };   // Export save: the button; copying; the export's name; the game has not saved yet; failed
int dm_save_note(char *out, int cap);   // front_android.c: the outcome (OVX_*) and the last export's name, or why it failed
static inline float ov_menu_scale(void) { float s = dm_overlay_scale(), fit = 0.86f / OV_MENU_H; return s < fit ? s : fit; }
static inline void ov_menu_rect(float W, float H, float r[4]) {   // x, y, width, height in window pixels
    float s = ov_menu_scale(), pw = OV_MENU_W * H * s, ph = OV_MENU_H * H * s;
    if (pw > W * 0.96f) pw = W * 0.96f;
    r[0] = (W - pw) / 2; r[1] = (H - ph) / 2; r[2] = pw; r[3] = ph;
}
static inline void ov_menu_rowrect(const float p[4], int row, float r[4]) {   // row's rectangle in the panel p (0 the tabs, OVM_ROWS - 1 Resume)
    float u = p[3] / OV_MENU_H, y = OV_MENU_PAD + OV_MENU_HEAD;
    if (row > 0) y += OV_MENU_TABH + OV_MENU_GAP + (row - 1) * OV_MENU_ROW;
    if (row >= OVM_ROWS - 1) y += OV_MENU_GAP;
    r[0] = p[0] + OV_MENU_PAD * u; r[1] = p[1] + y * u; r[2] = p[2] - 2 * OV_MENU_PAD * u; r[3] = (row == 0 ? OV_MENU_TABH : OV_MENU_ROW) * u;
}
static inline void ov_menu_headrect(const float p[4], float r[4]) { float u = p[3] / OV_MENU_H; r[0] = p[0] + OV_MENU_PAD * u; r[1] = p[1] + OV_MENU_PAD * u; r[2] = p[2] - 2 * OV_MENU_PAD * u; r[3] = OV_MENU_HEAD * u; }
static inline void ov_menu_footrect(const float p[4], float r[4]) {   // the key hints, under Resume
    float q[4]; ov_menu_rowrect(p, OVM_ROWS - 1, q); r[0] = q[0]; r[1] = q[1] + q[3]; r[2] = q[2]; r[3] = p[1] + p[3] - r[1];
}
static inline int ov_menu_row_at(float x, float y, float W, float H) {   // the row under (x, y): 0..OVM_ROWS-1, -1 the panel elsewhere, -2 outside it
    float r[4]; ov_menu_rect(W, H, r);
    if (x < r[0] || x >= r[0] + r[2] || y < r[1] || y >= r[1] + r[3]) return -2;
    for (int i = 0; i < OVM_ROWS; i++) { float q[4]; ov_menu_rowrect(r, i, q); if (y >= q[1] && y < q[1] + q[3]) return i; }
    return -1;
}
static inline int ov_menu_tab_at(float x, const float p[4]) { float r[4]; ov_menu_rowrect(p, 0, r); int t = (int)floorf((x - r[0]) / (r[2] / OVM_TABS)); return t < 0 ? 0 : t >= OVM_TABS ? OVM_TABS - 1 : t; }
// in a row: the label's left edge and the value's right edge (a row's insets: a third of its height)
static inline float ov_menu_inset(const float r[4]) { return r[3] * 0.30f; }
// the Opacity slider's track in its row (x, y of its left end's centre line, width): the right part of the row, room for the value after it
static inline void ov_menu_track(const float p[4], int row, float t[3]) {
    float r[4]; ov_menu_rowrect(p, row, r); float x1 = r[0] + r[2] - ov_menu_inset(r) - r[3] * 0.95f;
    t[0] = r[0] + r[2] * 0.52f; t[1] = r[1] + r[3] / 2; t[2] = x1 - t[0];
}
static inline int ov_menu_track_value(float x, const float p[4], int row) {   // the opacity a finger at x sets, whole 5 percents
    float t[3]; ov_menu_track(p, row, t); float f = (x - t[0]) / t[2]; f = f < 0 ? 0 : f > 1 ? 1 : f;
    return OV_OPACITY_MIN + 5 * (int)lroundf(f * (OV_OPACITY_MAX - OV_OPACITY_MIN) / 5);
}
// a choice row's value box ("< value >", right-aligned: its left x, centre y, width) and the step a tap at x makes: its left half back, elsewhere on
// the row forward
static inline void ov_menu_choice(const float p[4], int row, float c[3]) {
    float r[4]; ov_menu_rowrect(p, row, r); c[2] = r[2] * 0.40f; c[0] = r[0] + r[2] - ov_menu_inset(r) - c[2]; c[1] = r[1] + r[3] / 2;
}
static inline int ov_menu_choice_dir(float x, const float p[4], int row) { float c[3]; ov_menu_choice(p, row, c); return x >= c[0] && x < c[0] + c[2] / 2 ? -1 : 1; }
// a swatch row: swatch i's centre x (the box split in n even cells), and the swatch a tap at x picks (-1: none, outside the box)
static inline float ov_menu_swatch_x(const float p[4], int row, int i, int n) { float c[3]; ov_menu_choice(p, row, c); return c[0] + c[2] * (i + 0.5f) / n; }
static inline int ov_menu_swatch_at(float x, const float p[4], int row, int n) { float c[3]; ov_menu_choice(p, row, c); int i = (int)floorf((x - c[0]) / (c[2] / n)); return x >= c[0] && i >= 0 && i < n ? i : -1; }

// ---- the layout editor (front_android.c, from the menu's Edit layout): the game stays held; every control (the menu button too) can be dragged, and
// resized by a pinch or the toolbar's - and + (the tapped control, or all of them when none is). A toolbar under the top edge: its hint line and buttons;
// Cancel with changes asks first, in a prompt in the middle. Both are drawn into the menu panel's raster (gfx_vk_present.c ov_edit_panel) ----
enum { OVE_SMALLER, OVE_BIGGER, OVE_RESET, OVE_CANCEL, OVE_SAVE, OVE_GRID, OVE_N, OVE_DISCARD = OVE_N, OVE_KEEP, OVE_ALL };
// the editor's state in dm_menu_state: bit 17, bit 0 the prompt is open, bit 1 something changed, bit 2 grid snapping, bits 4..7 the button held + 1, bits 8..15 the chosen control + 1
#define OVE_ST 0x20000
#define OVE_GRID_ON 4
#define OVE_GRID_ROWS 20   // Snap to grid's rows; its cells are square
#define OVE_SCALE_MIN 0.5f   // a control's size factor
#define OVE_SCALE_MAX 2.0f
static inline void ov_ed_rect(float W, float H, float r[4]) {   // the toolbar, in window pixels: under the top edge, in the middle (below the menu button's place)
    float s = ov_menu_scale(); r[2] = fminf(0.85f * H * s, W * 0.96f); r[3] = 0.30f * H * s; r[0] = (W - r[2]) / 2; r[1] = 0.13f * H;
}
static inline void ov_ed_prompt_rect(float W, float H, float r[4]) {   // the discard prompt, in the middle
    float s = ov_menu_scale(); r[2] = fminf(0.75f * H * s, W * 0.9f); r[3] = 0.30f * H * s; r[0] = (W - r[2]) / 2; r[1] = (H - r[3]) / 2;
}
static inline void ov_ed_button(int i, const float p[4], float b[4]) {   // button i in its panel p (the toolbar's, or the prompt's for OVE_DISCARD, OVE_KEEP)
    static const float x0[OVE_ALL] = {0.15f, 1.30f, 2.80f, 5.25f, 7.70f, 0.15f, 0.50f, 5.20f}, w[OVE_ALL] = {1.0f, 1.0f, 2.30f, 2.30f, 2.15f, 9.70f, 4.30f, 4.30f};
    static const float y0[OVE_ALL] = {0.28f, 0.28f, 0.28f, 0.28f, 0.28f, 0.68f, 0.50f, 0.50f}, h[OVE_ALL] = {0.34f, 0.34f, 0.34f, 0.34f, 0.34f, 0.28f, 0.36f, 0.36f};
    float u = p[2] / 10; b[0] = p[0] + x0[i] * u; b[1] = p[1] + y0[i] * p[3]; b[2] = w[i] * u; b[3] = h[i] * p[3];
}
static inline int ov_ed_button_at(float x, float y, const float p[4], int first, int n) {   // the button of first..first+n-1 under (x, y), -1 none
    for (int i = first; i < first + n; i++) { float b[4]; ov_ed_button(i, p, b); if (x >= b[0] && x < b[0] + b[2] && y >= b[1] && y < b[1] + b[3]) return i; }
    return -1;
}

// ---- the save picker (front_android.c, from the menu's Import save): the pause menu's panel and rows, with what front_android.c puts in each row: the saves in
// saves/ newest first, OVM_ROWS - 2 to a page (left and right page), then the confirmation, then the note that the game closes. dm_menu_state: bit 18, bits 4..7
// the highlighted row + 1, bits 20..27 the picker's edit count (a page, a confirmation) ----
#define OVS_ST 0x40000
enum { OVS_EMPTY, OVS_TITLE, OVS_ITEM, OVS_NOTE, OVS_BUTTON, OVS_BACK };   // a row: nothing; the title (left) and the page (right); a save (its name, its size);
typedef struct { int style; char left[80], right[40]; } ov_srow;            // a line of text; a button (the import); Back or Cancel, centred as Resume is
int dm_save_rows(ov_srow rows[OVM_ROWS]);   // front_android.c: the picker's rows, 0 when it is not open

// ---- the text entry's keyboard (front_android.c front_text_input, drawn by gfx_vk_present.c ov_text_panel), in the pause menu's panel rectangle ----
// Four rows of ten character keys and a row of four wide ones; p is the panel's rectangle (x, y, width, height: the window's for touches, the raster's for drawing).
// The panel: the title (its top 13%), the text field (to 28%), the keys (to 90%), the key hints
#define OVK_N 44
static const char *const OVK_ROWS[4] = {"1234567890", "qwertyuiop", "asdfghjkl'", "zxcvbnm-._"};
enum { OVK_SHIFT = 40, OVK_SPACE, OVK_DEL, OVK_OK };
static inline void ov_kb_key(int i, const float p[4], float r[4]) {   // key i's rectangle, a small gap inside its cell
    float pad = p[2] * 0.03f, w = p[2] - 2 * pad, top = p[1] + p[3] * 0.30f, kh = p[3] * 0.60f / 5;
    if (i < 40) { float kw = w / 10; r[0] = p[0] + pad + (i % 10) * kw; r[1] = top + (i / 10) * kh; r[2] = kw; r[3] = kh; }
    else { static const float x0[4] = {0, 0.2f, 0.65f, 0.8f}, ww[4] = {0.2f, 0.45f, 0.15f, 0.2f}; r[0] = p[0] + pad + x0[i - 40] * w; r[1] = top + 4 * kh; r[2] = ww[i - 40] * w; r[3] = kh; }
    float g = kh * 0.06f; r[0] += g; r[1] += g; r[2] -= 2 * g; r[3] -= 2 * g;
}
static inline int ov_kb_at(float x, float y, const float p[4]) {   // the key under (x, y), -1 none
    for (int i = 0; i < OVK_N; i++) { float r[4]; ov_kb_key(i, p, r); if (x >= r[0] && x < r[0] + r[2] && y >= r[1] && y < r[1] + r[3]) return i; }
    return -1;
}
static inline int ov_kb_char(int i, int shift) {   // a character key's character (0 for the wide keys)
    if (i < 0 || i >= 40) return 0;
    int c = OVK_ROWS[i / 10][i % 10]; return shift && c >= 'a' && c <= 'z' ? c - 32 : c;
}
static inline int ov_kb_move(int i, int dx, int dy) {   // the d-pad's move from key i (rows wrap; the wide row maps to the columns above it)
    if (i < 0) return 0;
    static const int wide_col[4] = {0, 4, 7, 9};   // the column each wide key sits under
    int col = i < 40 ? i % 10 : wide_col[i - 40], row = i < 40 ? i / 10 : 4;
    if (dy) { row = (row + dy + 5) % 5; if (row == 4) return col < 2 ? OVK_SHIFT : col < 6 ? OVK_SPACE : col < 8 ? OVK_DEL : OVK_OK; return row * 10 + col; }
    if (row == 4) return 40 + (i - 40 + dx + 4) % 4;
    return row * 10 + (col + dx + 10) % 10;
}
