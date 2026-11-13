// Android port: the host front end (dm/host/front.h), in two forms.
//   dmhost executable (adb shell): headless, the guest runs on the calling thread as on the Mac without DM_WINDOW; no window, no input, no text entry.
//   libdmhost.so in the app (a NativeActivity, no Java): ANativeActivity_onCreate lays out the game's files in the app's external files folder
//   (/sdcard/Android/data/<package>/files: mods/ = the rewritten modules, RomFS/ (or env.txt's DM_ROMFS: romfs_readable), shaders.dmsv; env.txt = KEY=VALUE switches; log.txt = the log), keeps saves in
//   internal storage, and starts the host on a thread of its own. The window goes to the Vulkan backend's swapchain (dm_front_window); game controllers
//   (KEYCODE_BUTTON_*, sticks, triggers, hat) feed player 1's pad; touches go to the game's touch screen; shaking the device shakes player 1's controller and turning it turns
//   the controller's six-axis sensor (the gyroscope);
//   leaving and coming back pauses as on iOS.
#include "dm.h"
#include "front.h"
#include "dm_input.h"
#include "overlay_layout.h"
#include "pixel_font.h"
#include "saves_android.h"
#include <math.h>
#include <time.h>
#include <android/asset_manager.h>
#include <android/configuration.h>
#include <android/input.h>
#include "external_input_event.h"
#include <android/log.h>
#include <android/looper.h>
#include <android/native_activity.h>
#include <android/window.h>
#include <android/native_window.h>
#include <android/native_window_jni.h>
#include <android/sensor.h>
#include <errno.h>
#include <dlfcn.h>
#include <jni.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

void front_run(void (*guest)(void)) { guest(); }
void *front_metal_layer(void) { return NULL; }
int front_screen_max_fps(void) { return 0; }
void front_screen_match(int fps) { (void)fps; }
// front_text_input (the game's software keyboard, hle_swkbd.c) is further down, with the text entry

// ---------------- the app ----------------
static ANativeActivity *g_act;
AAssetManager *dm_front_assets(void) { return g_act ? g_act->assetManager : NULL; }   // the APK's assets (overlay_theme.c: the gamepad's theme)
static pthread_mutex_t g_win_mu = PTHREAD_MUTEX_INITIALIZER;
static ANativeWindow *g_win, *g_external_win;
static _Atomic int g_external_active;
static _Atomic unsigned g_external_epoch;
static _Atomic uint64_t g_ext_buttons, g_ext_presses;
static _Atomic int g_ext_lx, g_ext_ly, g_ext_rx, g_ext_ry, g_ext_menu;
ANativeWindow *dm_front_window(void) { pthread_mutex_lock(&g_win_mu); ANativeWindow *w = g_external_win ? g_external_win : g_win; pthread_mutex_unlock(&g_win_mu); return w; }
static ANativeWindow *phone_window(void) { pthread_mutex_lock(&g_win_mu); ANativeWindow *w = g_win; pthread_mutex_unlock(&g_win_mu); return w; }
int dm_external_active(void) { return atomic_load(&g_external_active); }
__attribute__((weak)) void vk_window_lost(void) {}   // the backend lets go of the window's swapchain before the window goes
static void on_window_created(ANativeActivity *a, ANativeWindow *w) { (void)a; pthread_mutex_lock(&g_win_mu); g_win = w; pthread_mutex_unlock(&g_win_mu); }
static void on_window_destroyed(ANativeActivity *a, ANativeWindow *w) { (void)a; (void)w; pthread_mutex_lock(&g_win_mu); g_win = NULL; pthread_mutex_unlock(&g_win_mu); vk_window_lost(); }
JNIEXPORT void JNICALL Java_org_nx2a_ExternalActivity_externalSurface(JNIEnv *env, jclass cls, jobject surface) {
    (void)cls;
    ANativeWindow *next = surface ? ANativeWindow_fromSurface(env, surface) : NULL;
    pthread_mutex_lock(&g_win_mu); ANativeWindow *old = g_external_win; g_external_win = next; atomic_fetch_add(&g_external_epoch, 1); atomic_store(&g_external_active, next != NULL); pthread_mutex_unlock(&g_win_mu);
    vk_window_lost();
    if (old) ANativeWindow_release(old);
    if (!next) { atomic_store(&g_ext_buttons, 0); atomic_store(&g_ext_presses, 0); atomic_store(&g_ext_menu, 0); dm_input_set(0, 0, 0, 0, 0); }
}

void dm_focus_set(int st); void dm_audio_suspend(int on); void guest_clock_freeze_for(int reason, int on); void dm_gfx_suspend(int on);
// ---- settings (the pause menu's), kept in the files folder as KEY=VALUE lines ----
static char g_root[512];
static char g_title[128];   // the app's name (on_create: app_label), the menu's header
static _Atomic int g_set_temps = -1;   // Show temperatures (settings.txt temps=): -1 until the player chooses, then they follow Show FPS, as before the switch
static _Atomic int g_set_overlay = 1, g_set_rumble = 1, g_set_fps;   // the gamepad shown; rumble and tap vibration; the frame rate counter (top left)
static _Atomic int g_set_opacity = 60;   // the gamepad's opacity, percent (OV_OPACITY_MIN..MAX: the menu's Opacity slider)
static _Atomic int g_set_grid;   // the layout editor's Snap to grid
static _Atomic unsigned g_set_accent = OVM_ACCENT_DEFAULT;   // the menu's highlight colour, 0xRRGGBB (overlay_layout.h OVM_ACCENTS; settings.txt accent=RRGGBB; none: orange)
static _Atomic int g_set_res = DM_RES_BALANCED;   // the Resolution level (DM_RES_*: docked at DM_RES_MAX, else handheld; settings.txt res=, docked= kept beside it)
static const char *const RES_KEYS[DM_RES_N] = {"performance", "balanced", "quality", "max"};
static _Atomic int g_set_cap;    // the frame cap's index in OVM_CAPS (0 off; settings.txt fpscap= in fps; hle_framecap.c holds the game to it)
static _Atomic int g_set_fpspos;   // the frame rate counter's corner (OVM_FPSPOS_KEYS: 0 top left for every game until the player picks another; settings.txt fpspos=)
static _Atomic int g_set_thermal = -1;    // the pause menu's Thermal throttle (settings.txt thermal=): -1 until the player chooses there (DM_THERMAL_CAP decides), then 0 off, 1 on
static _Atomic int g_set_floor = -1;      // the pause menu's GPU clock floor (settings.txt gpufloor=<percent>): -1 until the player chooses there (the profile's DM_GPU_MIN_FREQ, else 75%), then OVM_FLOORS' index
// The menu button hides while a controller plays that has a key for the menu (Back, Home or Menu: pad_watch asks InputDevice.hasKeys of every controller
// in use). It stays for a controller with none of them, which has no other way to the menu
static _Atomic int g_pad_menukey;
static _Atomic int g_set_mouse = -1;      // the pause menu's Joy-Con mouse (settings.txt mouse=): -1 until the player chooses there (DM_MOUSE decides), then 0 off, 1 auto
static _Atomic int g_set_padlabel = -1;   // the pause menu's Face buttons (settings.txt labels=): -1 until the player chooses there, then 0 by position, 1 by label
static int fpspos_index(const char *s) { for (int i = 0; i < 4; i++) if (!strncmp(s, OVM_FPSPOS_KEYS[i], 2)) return i; return -1; }
int dm_overlay_fps_pos(void) { return atomic_load(&g_set_fpspos); }
static int cap_index(int fps) { for (int i = 0; i < OVM_NCAPS; i++) if (OVM_CAPS[i] == fps) return i; return 0; }
// settings.txt lines this build does not know (a newer build's, or one a tool wrote): kept as they are and written again after the known ones
static char g_set_extra[2048];
// the player's own layout (overlay_layout.h dm_overlay_custom): [0] the saved one (settings.txt ui= rows), [1] the layout editor's while it is open; per layout
// and control: x + 1, y + 1 in 1/10000 of the window's width and height (0 = the layout's place), the size factor in 1/1000 (0 = 1), 1 when hidden
#define CUST_K 4
static _Atomic int g_cust[2][2][DM_AOV_N][CUST_K], g_cust_ver, g_edit;
static const char *const LAYOUT_NAMES[2] = {"tablet", "phone"};
static int ctl_named(const char *n) { for (int i = 0; i < OV_N; i++) if (!strcmp(OV_LAYOUTS[0][i].name, n)) return i; return -1; }
int dm_opmode(void); void dm_set_opmode(int m);   // hle_oe.c: 0 handheld, 1 docked; a change reaches the game as the Switch's operation mode notification
static void settings_load(void) {
    char p[600]; snprintf(p, sizeof p, "%s/settings.txt", g_root); FILE *f = fopen(p, "r"); char line[128];
    if (!f) return;
    g_set_extra[0] = 0;
    while (fgets(line, sizeof line, f)) { int v; if (sscanf(line, "labels=%d", &v) == 1) atomic_store(&g_set_padlabel, v != 0); else if (sscanf(line, "mouse=%d", &v) == 1) atomic_store(&g_set_mouse, v != 0); else if (sscanf(line, "thermal=%d", &v) == 1) atomic_store(&g_set_thermal, v != 0); else if (sscanf(line, "overlay=%d", &v) == 1) atomic_store(&g_set_overlay, v != 0); else if (sscanf(line, "rumble=%d", &v) == 1) atomic_store(&g_set_rumble, v != 0);
        else if (sscanf(line, "fps=%d", &v) == 1) atomic_store(&g_set_fps, v != 0);
        else if (sscanf(line, "temps=%d", &v) == 1) atomic_store(&g_set_temps, v != 0);
        else if (sscanf(line, "docked=%d", &v) == 1) { setenv("DM_OPMODE", v ? "1" : "0", 1); atomic_store(&g_set_res, v ? DM_RES_MAX : DM_RES_BALANCED); }   // over env.txt's: the host reads it once, at its first query
        else if (!strncmp(line, "res=", 4)) { for (int k = 0; k < DM_RES_N; k++) if (!strncmp(line + 4, RES_KEYS[k], strlen(RES_KEYS[k]))) { atomic_store(&g_set_res, k); setenv("DM_OPMODE", k == DM_RES_MAX ? "1" : "0", 1); } }   // (after docked=)
        else if (sscanf(line, "fpscap=%d", &v) == 1) atomic_store(&g_set_cap, cap_index(v));
        else if (sscanf(line, "gpufloor=%d", &v) == 1) { for (int k = 0; k < OVM_NFLOORS; k++) if (OVM_FLOORS[k] == v) atomic_store(&g_set_floor, k); }
        else if (!strncmp(line, "fpspos=", 7) && fpspos_index(line + 7) >= 0) atomic_store(&g_set_fpspos, fpspos_index(line + 7));
        else if (sscanf(line, "opacity=%d", &v) == 1) atomic_store(&g_set_opacity, v < OV_OPACITY_MIN ? OV_OPACITY_MIN : v > OV_OPACITY_MAX ? OV_OPACITY_MAX : v);
        else if (sscanf(line, "grid=%d", &v) == 1) atomic_store(&g_set_grid, v != 0);
        else if (!strncmp(line, "accent=", 7)) { unsigned c; if (sscanf(line + 7, "%6x", &c) == 1 && strspn(line + 7, "0123456789abcdefABCDEF") == 6) atomic_store(&g_set_accent, c); }
        else if (!strncmp(line, "ui=", 3)) {   // ui=<layout>,<control>,<x>,<y>,<size>[,<hidden>]: x, y in 1/10000 of the window (-1 the layout's place), size in 1/1000
            char lay[16], name[32]; int x, y, z, hid = 0, l, i;
            if (sscanf(line + 3, "%15[^,],%31[^,],%d,%d,%d,%d", lay, name, &x, &y, &z, &hid) < 5 || (i = ctl_named(name)) < 0) continue;
            for (l = 0; l < 2 && strcmp(lay, LAYOUT_NAMES[l]); l++) {}
            if (l == 2) continue;
            if (x >= 0 && x <= 10000 && y >= 0 && y <= 10000) { atomic_store(&g_cust[0][l][i][0], x + 1); atomic_store(&g_cust[0][l][i][1], y + 1); }
            if (z >= OVE_SCALE_MIN * 1000 && z <= OVE_SCALE_MAX * 1000) atomic_store(&g_cust[0][l][i][2], z);
            if (hid == 1 && OV_LAYOUTS[l][i].kind != OV_MENU) atomic_store(&g_cust[0][l][i][3], 1);
        } else if (sscanf(line, "menu_x=%d", &v) == 1 && v >= 0 && v <= 10000) { int m = ctl_named("menu"); for (int l = 0; l < 2 && m >= 0; l++) atomic_store(&g_cust[0][l][m][0], v + 1); }   // before
        else if (sscanf(line, "menu_y=%d", &v) == 1 && v >= 0 && v <= 10000) { int m = ctl_named("menu"); for (int l = 0; l < 2 && m >= 0; l++) atomic_store(&g_cust[0][l][m][1], v + 1); }   // the editor
        else if (strchr(line, '=') && strlen(g_set_extra) + strlen(line) + 2 < sizeof g_set_extra) {   // a key this build does not know: kept
            strcat(g_set_extra, line); if (line[strlen(line) - 1] != '\n') strcat(g_set_extra, "\n");
        }
    }
    fclose(f);
}
static void settings_save(void) {
    char p[600], tmp[610]; snprintf(p, sizeof p, "%s/settings.txt", g_root); snprintf(tmp, sizeof tmp, "%s.new", p);
    FILE *f = fopen(tmp, "w"); if (!f) { LOG("[RED] cannot write %s", tmp); return; }
    fprintf(f, "overlay=%d\nrumble=%d\ndocked=%d\nfps=%d\nopacity=%d\ngrid=%d\nfpscap=%d\nres=%s\naccent=%06X\n", atomic_load(&g_set_overlay), atomic_load(&g_set_rumble), dm_opmode() != 0,
            atomic_load(&g_set_fps), atomic_load(&g_set_opacity), atomic_load(&g_set_grid), OVM_CAPS[atomic_load(&g_set_cap)], RES_KEYS[atomic_load(&g_set_res)],
            atomic_load(&g_set_accent) & 0xffffff);
    if (atomic_load(&g_set_temps) >= 0) fprintf(f, "temps=%d\n", atomic_load(&g_set_temps));   // (only once chosen)
    fprintf(f, "fpspos=%s\n", OVM_FPSPOS_KEYS[atomic_load(&g_set_fpspos) & 3]);
    if (atomic_load(&g_set_padlabel) >= 0) fprintf(f, "labels=%d\n", atomic_load(&g_set_padlabel));
    if (atomic_load(&g_set_mouse) >= 0) fprintf(f, "mouse=%d\n", atomic_load(&g_set_mouse));
    if (atomic_load(&g_set_floor) >= 0) fprintf(f, "gpufloor=%d\n", OVM_FLOORS[atomic_load(&g_set_floor)]);   // (only once chosen: until then the profile, else the default)
    if (atomic_load(&g_set_thermal) >= 0) fprintf(f, "thermal=%d\n", atomic_load(&g_set_thermal));   // (only once chosen: until then env.txt decides)
    fputs(g_set_extra, f);   // the lines settings_load did not know
    for (int l = 0; l < 2; l++) for (int i = 0; i < OV_N; i++) {
        _Atomic int *c = g_cust[0][l][i]; int x = atomic_load(&c[0]), y = atomic_load(&c[1]), z = atomic_load(&c[2]), hid = atomic_load(&c[3]);
        if ((x && y) || z || hid) fprintf(f, "ui=%s,%s,%d,%d,%d,%d\n", LAYOUT_NAMES[l], OV_LAYOUTS[0][i].name, x && y ? x - 1 : -1, x && y ? y - 1 : -1, z ? z : 1000, hid);
    }
    fclose(f); rename(tmp, p);
}
void dm_lan_android_start(ANativeActivity *act);   // lan_android.c: the Wi-Fi multicast lock for local play over the LAN (DM_LAN=1)
void dm_haptics_android_start(ANativeActivity *act, int on); void dm_haptics_android_tap(void); void dm_haptics_android_device(int id); void dm_haptics_android_set(int on);
void dm_haptics_suspend(unsigned reason, int suspended);
__attribute__((weak)) void dm_overlay_redraw(void) {}   // the backend presents the last frame again (the pause menu changed while the game is held)

// ---- holding the game: in the background and while the pause menu is open. Its clock reads wait (reason 1) and its audio goes to the silent sink ----
static _Atomic int g_running, g_bg, g_menu;
static pthread_mutex_t g_hold_mu = PTHREAD_MUTEX_INITIALIZER; static int g_held;
static _Atomic int g_ime_on;
static _Atomic int g_tx_on;   // text entry open (front_text_input): the game is held as by the menu
static void hold_sync(void) {
    pthread_mutex_lock(&g_hold_mu);
    int want = atomic_load(&g_bg) || atomic_load(&g_menu) || atomic_load(&g_tx_on) || atomic_load(&g_ime_on);
    if (want != g_held) { g_held = want; if (want) { dm_movie_hold(1); dm_audio_suspend(1); guest_clock_freeze_for(1, 1); } else { guest_clock_freeze_for(1, 0); dm_audio_suspend(0); dm_movie_hold(0); } }
    pthread_mutex_unlock(&g_hold_mu);
}
static void shake_sensor(int on);   // the accelerometer, off in the background (input, below)
static void go_background(void) { if (!atomic_load(&g_running) || atomic_exchange(&g_bg, 1)) return; dm_focus_set(2); dm_focus_set(3);   // through OutOfFocus, as a game sees HOME and then the suspension (front_mac.m does the same)
    hold_sync(); dm_gfx_suspend(1); shake_sensor(0); }
static void go_foreground(void) { if (!atomic_load(&g_running) || !atomic_exchange(&g_bg, 0)) return; dm_gfx_suspend(0); hold_sync(); dm_focus_set(2); shake_sensor(1); }
static void gpu_floor_restore(void), gpu_floor_resume(void), gpu_floor_end(void);   // (GPU clock floor, below)
static void on_pause(ANativeActivity *a) { (void)a; go_background(); gpu_floor_restore(); }
// Back in front, the game is out of focus until the window has the input focus again (Android sends it after onResume): without this it stayed OutOfFocus for good,
// and Blasphemous 2's Unity, which asks for focus notifications, kept its title paused and ignored every press after a trip to another app. A window losing focus while in
// front (a system dialog, the notification shade) is OutOfFocus as on the Switch's HOME overlay
// The system bars off the screen: AWINDOW_FLAG_FULLSCREEN hides the status bar only, so on a device with three-button or gesture navigation the bar (Back, Home, Recents)
// sat over the bottom of the picture and the touch controls. Immersive-sticky full screen hides both: an edge swipe shows them until they slide away again. Asked again each
// time the window gets the focus back (a dialog, the notification shade or another app brings the bars back) and at each resume. Through JNI on the main thread (the
// app has no Java of its own): getWindow().getDecorView().setSystemUiVisibility(IMMERSIVE_STICKY | FULLSCREEN | HIDE_NAVIGATION | LAYOUT_STABLE | LAYOUT_FULLSCREEN |
// LAYOUT_HIDE_NAVIGATION), deprecated since Android 11 but still honoured. DM_IMMERSIVE=0 (env.txt) leaves the bars as before. The idea is droidplay-core b4557945f's
static void immersive(ANativeActivity *a) {
    static int on = -1; if (on < 0) { const char *e = getenv("DM_IMMERSIVE"); on = !(e && *e == '0'); }
    if (!on || !a || !a->env || !a->clazz) return;
    JNIEnv *E = a->env; jobject w = NULL, d = NULL; int ok = 0;
    jclass ca = (*E)->GetObjectClass(E, a->clazz); jmethodID gw = ca ? (*E)->GetMethodID(E, ca, "getWindow", "()Landroid/view/Window;") : NULL;
    if (gw && !(*E)->ExceptionCheck(E)) w = (*E)->CallObjectMethod(E, a->clazz, gw);
    if (w && !(*E)->ExceptionCheck(E)) { jclass cw = (*E)->GetObjectClass(E, w); jmethodID gd = cw ? (*E)->GetMethodID(E, cw, "getDecorView", "()Landroid/view/View;") : NULL;
        if (gd && !(*E)->ExceptionCheck(E)) d = (*E)->CallObjectMethod(E, w, gd); }
    if (d && !(*E)->ExceptionCheck(E)) { jclass cv = (*E)->GetObjectClass(E, d); jmethodID sv = cv ? (*E)->GetMethodID(E, cv, "setSystemUiVisibility", "(I)V") : NULL;
        if (sv && !(*E)->ExceptionCheck(E)) { (*E)->CallVoidMethod(E, d, sv, (jint)(0x1000 | 0x4 | 0x2 | 0x100 | 0x400 | 0x200)); ok = !(*E)->ExceptionCheck(E); } }
    if ((*E)->ExceptionCheck(E)) (*E)->ExceptionClear(E);
    static int said; if (!said++) { if (ok) LOG("[front] system bars hidden, immersive-sticky (DM_IMMERSIVE=0 leaves them)"); else LOG("[RED] [front] the system bars could not be hidden (setSystemUiVisibility failed)"); }
}
static void on_window_focus(ANativeActivity *a, int focused) { if (focused) immersive(a); if (atomic_load(&g_running) && !atomic_load(&g_bg)) dm_focus_set(focused ? 1 : 2); }
// A save manager's request, or a RomFS the app cannot read (romfs_readable), asked the app to end (ANativeActivity_onCreate): Activity.finish() when it is resumed, through JNI on the main thread.
// ANativeActivity_finish, from onCreate or from here, left the activity on the screen (Y700, Android 16), and porter's am start -W waited out its timeout
static int g_quit;
static void app_label(ANativeActivity *a) {   // the app's name (the launcher's label: the game's), for the menu's header (dm_game_title)
    JNIEnv *E = a->env; jobject act = a->clazz, o = NULL;
#define AL_CALL(obj, name, sig, ...) do { jclass c_ = (*E)->GetObjectClass(E, obj); jmethodID m_ = c_ ? (*E)->GetMethodID(E, c_, name, sig) : NULL; \
        o = m_ && !(*E)->ExceptionCheck(E) ? (*E)->CallObjectMethod(E, obj, m_, ##__VA_ARGS__) : NULL; if ((*E)->ExceptionCheck(E)) { (*E)->ExceptionClear(E); o = NULL; } } while (0)
    AL_CALL(act, "getPackageManager", "()Landroid/content/pm/PackageManager;"); jobject pm = o; if (!pm) return;
    AL_CALL(act, "getApplicationInfo", "()Landroid/content/pm/ApplicationInfo;"); jobject ai = o; if (!ai) return;
    AL_CALL(pm, "getApplicationLabel", "(Landroid/content/pm/ApplicationInfo;)Ljava/lang/CharSequence;", ai); if (!o) return;
    AL_CALL(o, "toString", "()Ljava/lang/String;"); if (!o) return;
#undef AL_CALL
    const char *u = (*E)->GetStringUTFChars(E, (jstring)o, NULL);
    if (u) { snprintf(g_title, sizeof g_title, "%s", u); (*E)->ReleaseStringUTFChars(E, (jstring)o, u); }
}
static void act_finish(ANativeActivity *a) {
    JNIEnv *E = a->env; jclass c = (*E)->GetObjectClass(E, a->clazz); jmethodID m = c ? (*E)->GetMethodID(E, c, "finish", "()V") : NULL;
    if (m) (*E)->CallVoidMethod(E, a->clazz, m);
    if (!m || (*E)->ExceptionCheck(E)) { (*E)->ExceptionClear(E); LOG("[RED] Activity.finish() failed: the app ends itself"); _exit(0); }
}
// A game left with HOME stays in memory: without this its process is cached, and Android ended it within minutes (the low memory killer takes a cached process of
// several GB first, and the cached-app freezer kills one whose binder buffer fills: "excessive binder traffic during cached" in the Odin 3's exit records), so
// coming back started the game from its beginning. org.nx2a.KeepAlive (java/, in the APK with mkapk.py --keep-alive) is a foreground service: with it the process is
// not cached. startService and not startForegroundService: the activity is in front here, and a service that cannot go foreground then ends without the system
// ending the app. An APK without the class (code-less, as before): startService finds nothing and returns null. DM_KEEP_ALIVE=0 (env.txt) leaves it off
static void keep_alive(ANativeActivity *a) {
    const char *e = getenv("DM_KEEP_ALIVE"); if ((e && *e == '0') || !a || !a->env || !a->clazz) return;
    JNIEnv *E = a->env; jobject it = NULL, r = NULL;
    jclass ci = (*E)->FindClass(E, "android/content/Intent");
    jmethodID init = ci ? (*E)->GetMethodID(E, ci, "<init>", "()V") : NULL;
    jmethodID scn = init ? (*E)->GetMethodID(E, ci, "setClassName", "(Landroid/content/Context;Ljava/lang/String;)Landroid/content/Intent;") : NULL;
    if (scn && !(*E)->ExceptionCheck(E)) it = (*E)->NewObject(E, ci, init);
    if (it && !(*E)->ExceptionCheck(E)) { jstring n = (*E)->NewStringUTF(E, "org.nx2a.KeepAlive"); if (n) (*E)->CallObjectMethod(E, it, scn, a->clazz, n); }
    if (it && !(*E)->ExceptionCheck(E)) { jclass ca = (*E)->GetObjectClass(E, a->clazz);
        jmethodID ss = ca ? (*E)->GetMethodID(E, ca, "startService", "(Landroid/content/Intent;)Landroid/content/ComponentName;") : NULL;
        if (ss && !(*E)->ExceptionCheck(E)) r = (*E)->CallObjectMethod(E, a->clazz, ss, it); }
    if ((*E)->ExceptionCheck(E)) { (*E)->ExceptionClear(E); r = NULL; }
    if (r) LOG("[front] the game stays in memory while in the background (keep-alive service; DM_KEEP_ALIVE=0 leaves it off)");
    else LOG("[front] no keep-alive service in this APK: Android may end the game while it is in the background");
}
static void on_resume(ANativeActivity *a) { if (g_quit) { act_finish(a); return; } immersive(a); gpu_floor_resume(); go_foreground(); }
static void on_destroy(ANativeActivity *a) { (void)a; gpu_floor_end(); _exit(0); }   // one game per process

// ---- input ----
static AInputQueue *g_queue; static ALooper *g_ilooper;
// Java's dialog has its own input channel; hand its events to the normal input thread.
typedef struct ext_touch { const AInputEvent *event; struct ext_touch *next; unsigned epoch; } ext_touch;
static pthread_mutex_t g_ext_touch_mu = PTHREAD_MUTEX_INITIALIZER;
static ext_touch *g_ext_touch_head, *g_ext_touch_tail;
static _Atomic int g_ext_touch_enabled;
typedef const AInputEvent *(*motion_from_java_fn)(JNIEnv *, jobject);
typedef const AInputEvent *(*key_from_java_fn)(JNIEnv *, jobject);
typedef void (*input_release_fn)(const AInputEvent *);
static motion_from_java_fn g_motion_from_java;
static key_from_java_fn g_key_from_java;
static input_release_fn g_input_release;
static pthread_once_t g_ext_touch_once = PTHREAD_ONCE_INIT;
static void ext_touch_init(void) {
    g_motion_from_java = (motion_from_java_fn)dlsym(RTLD_DEFAULT, "AMotionEvent_fromJava");
    g_key_from_java = (key_from_java_fn)dlsym(RTLD_DEFAULT, "AKeyEvent_fromJava");
    g_input_release = (input_release_fn)dlsym(RTLD_DEFAULT, "AInputEvent_release");
}
static jboolean external_event(JNIEnv *env, jobject java_event, int key, int touch) {
    pthread_once(&g_ext_touch_once, ext_touch_init);
    if (!java_event || !dm_external_active() || !g_input_release || (key ? !g_key_from_java : !g_motion_from_java)) return JNI_FALSE;
    unsigned epoch = atomic_load(&g_external_epoch);
    const AInputEvent *event = key ? g_key_from_java(env, java_event) : g_motion_from_java(env, java_event);
    ext_touch *item = event ? malloc(sizeof *item) : NULL;
    if (!item) { if (event) g_input_release(event); return JNI_FALSE; }
    *item = (ext_touch){event, NULL, epoch};
    pthread_mutex_lock(&g_ext_touch_mu);
    if (g_ext_touch_tail && g_ext_touch_tail->epoch == item->epoch && external_moves_match(g_ext_touch_tail->event, event)) {
        g_input_release(g_ext_touch_tail->event); g_ext_touch_tail->event = event; free(item);
    } else {
        if (g_ext_touch_tail) g_ext_touch_tail->next = item; else g_ext_touch_head = item;
        g_ext_touch_tail = item;
    }
    if (touch) atomic_store(&g_ext_touch_enabled, 1);
    pthread_mutex_unlock(&g_ext_touch_mu);
    if (g_ilooper) ALooper_wake(g_ilooper);
    return JNI_TRUE;
}
JNIEXPORT jboolean JNICALL Java_org_nx2a_ExternalActivity_externalTouch(JNIEnv *env, jclass cls, jobject event) { (void)cls; return external_event(env, event, 0, 1); }
JNIEXPORT jboolean JNICALL Java_org_nx2a_ExternalActivity_externalControllerKey(JNIEnv *env, jclass cls, jobject event) { (void)cls; return external_event(env, event, 1, 0); }
JNIEXPORT jboolean JNICALL Java_org_nx2a_ExternalActivity_externalControllerMotion(JNIEnv *env, jclass cls, jobject event) { (void)cls; return external_event(env, event, 0, 0); }
void dm_audio_route_device(int32_t id);
JNIEXPORT void JNICALL Java_org_nx2a_ExternalActivity_externalAudio(JNIEnv *env, jclass cls, jint id) { (void)env; (void)cls; dm_audio_route_device(id); }
JNIEXPORT void JNICALL Java_org_nx2a_ExternalActivity_externalHaptic(JNIEnv *env, jclass cls) { (void)env; (void)cls; dm_haptics_android_tap(); }
JNIEXPORT void JNICALL Java_org_nx2a_ExternalActivity_externalInput(JNIEnv *env, jclass cls, jlong buttons, jint lx, jint ly, jint rx, jint ry, jboolean menu) {
    (void)env; (void)cls;
    uint64_t old = atomic_exchange(&g_ext_buttons, (uint64_t)buttons);
    atomic_fetch_or(&g_ext_presses, (uint64_t)buttons & ~old);
    if (((uint64_t)buttons & ~old) || menu) dm_haptics_android_tap();
    atomic_store(&g_ext_lx, lx); atomic_store(&g_ext_ly, ly); atomic_store(&g_ext_rx, rx); atomic_store(&g_ext_ry, ry);
    if (menu) atomic_fetch_add(&g_ext_menu, 1);
    if (g_ilooper) ALooper_wake(g_ilooper);
}
static uint64_t g_btn; static float g_ax[8]; static int g_pad_used;   // lx ly rx ry lt rt hatx haty
// Face buttons. Android names a pad's buttons by the Xbox layout (BUTTON_A at the bottom, B right, X left, Y top), so by position the bottom button is Nintendo's B and the
// right one A (as the iOS front end maps). A device whose buttons carry Nintendo's labels in Nintendo's places and reports them by label (the AYN Odin 3: its right button, labelled A,
// sends BUTTON_A) wants them as labelled: env.txt DM_PAD_LABELS=1, which porter install writes for such devices
// upstream nx2android names the same switch DM_PAD_XBOX=1 (its Super Mario Odyssey and Super Mario Bros. Wonder profile rows): it turns this on too
static int pad_labels(void) {   // read at every key: a change in the menu applies at once, its own A and B too (upstream nx2android #185)
    int v = atomic_load(&g_set_padlabel); if (v >= 0) return v;
    static int env = -1; if (env < 0) { const char *n[3] = {"DM_PAD_LABELS", "DM_PAD_LABEL", "DM_PAD_XBOX"}; env = 0;   // ours (the porter's for the Odin), upstream's, its first name
                                         for (int i = 0; i < 3; i++) { const char *e = getenv(n[i]); if (e && atoi(e) > 0) env = 1; } }
    return env;
}
int dm_pad_labels(void) { return pad_labels(); }
// host/hle_hid.c: the pause menu's Joy-Con mouse, read at every mouse call (a change applies at once): 0 off, 1 auto, -1 not chosen (DM_MOUSE decides, on by default)
// perf_android.c dm_thermal_now: the pause menu's Thermal throttle, read at every call (a change applies at once): 0 off, 1 on, -1 not chosen (DM_THERMAL_CAP=1 decides)
int dm_front_thermal(void) { return atomic_load(&g_set_thermal); }
int dm_front_mouse(void) { int v = atomic_load(&g_set_mouse); if (v >= 0) return v; const char *e = getenv("DM_MOUSE"); return e && !atoi(e) ? 0 : e ? 1 : -1; }   // (gfx_vk_present.c: the menu's key hints name the buttons as labelled)
static uint64_t key_bit(int32_t k) {
    switch (k) {
    case AKEYCODE_BUTTON_A: return pad_labels() ? DM_BTN_A : DM_BTN_B; case AKEYCODE_BUTTON_B: return pad_labels() ? DM_BTN_B : DM_BTN_A;
    case AKEYCODE_BUTTON_X: return pad_labels() ? DM_BTN_X : DM_BTN_Y; case AKEYCODE_BUTTON_Y: return pad_labels() ? DM_BTN_Y : DM_BTN_X;
    case AKEYCODE_BUTTON_L1: return DM_BTN_L; case AKEYCODE_BUTTON_R1: return DM_BTN_R; case AKEYCODE_BUTTON_L2: return DM_BTN_ZL; case AKEYCODE_BUTTON_R2: return DM_BTN_ZR;
    case AKEYCODE_BUTTON_THUMBL: return DM_BTN_LS; case AKEYCODE_BUTTON_THUMBR: return DM_BTN_RS; case AKEYCODE_BUTTON_START: return DM_BTN_PLUS; case AKEYCODE_BUTTON_SELECT: return DM_BTN_MINUS;
    case AKEYCODE_DPAD_UP: return DM_BTN_UP; case AKEYCODE_DPAD_DOWN: return DM_BTN_DOWN; case AKEYCODE_DPAD_LEFT: return DM_BTN_LEFT; case AKEYCODE_DPAD_RIGHT: return DM_BTN_RIGHT;
    default: return 0;
    }
}
static int32_t axis(float v) { if (v > -0.12f && v < 0.12f) return 0; return (int32_t)(v * DM_STICK_MAX); }
static uint64_t g_shake_btn;   // DM_BTN_SHAKE while the device is being shaken (below); input thread only
static uint64_t pad_axis_buttons(const float axes[8]) {
    uint64_t b = 0;
    if (axes[4] > 0.5f) b |= DM_BTN_ZL;
    if (axes[5] > 0.5f) b |= DM_BTN_ZR;
    if (axes[6] < -0.5f) b |= DM_BTN_LEFT;
    if (axes[6] > 0.5f) b |= DM_BTN_RIGHT;
    if (axes[7] < -0.5f) b |= DM_BTN_UP;
    if (axes[7] > 0.5f) b |= DM_BTN_DOWN;
    return b;
}
static void pad_publish(void) {
    uint64_t b = g_btn | g_shake_btn | pad_axis_buttons(g_ax);
    dm_pad_set(b, axis(g_ax[0]), -axis(g_ax[1]), axis(g_ax[2]), -axis(g_ax[3]));   // Android's y grows downward, the Switch's upward
}
// ---- the on-screen gamepad (overlay_layout.h): a pointer that goes down on a control keeps it until it lifts; every other pointer is the game's touch screen ----
static _Atomic int g_overlay = 1;   // shown until a controller sends input, restored at the next touch if enabled
static _Atomic int g_pads;          // controllers attached and used (pad_watch), for the vibration handoff
static _Atomic int g_pad_in = -1;   // the device that last sent a controller's input (handle_event), which pad_watch then counts
static _Atomic int g_knob[2][2];    // the sticks' knob offsets, in 1/10000 of the stick radius
static _Atomic int64_t g_press_t[16], g_menu_t, g_stick_t[2];   // when each overlay button (by DM_BTN_* bit), the menu button and each stick went down, 0 = not held
static _Atomic uint64_t g_touch_btn;   // the overlay buttons held, cleared by the menu or keyboard
static int64_t mono_ns(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return (int64_t)ts.tv_sec * 1000000000 + ts.tv_nsec; }
// A system input device can report itself as an active controller (issues #22, #210). The saved choice must still work, and the menu must stay reachable.
int dm_overlay_visible(void) {
    if (atomic_load(&g_set_overlay) && atomic_load(&g_overlay)) return 1;
    if (g_pad_used && atomic_load(&g_pad_menukey) && !atomic_load(&g_menu)) return 0;   // the controller's own key opens the menu
    return 2;
}
int dm_overlay_layout(void) {   // the phone's layout in a window at least OV_PHONE_ASPECT wide for its height (overlay_layout.h)
    static int last_w = -1, last_h = -1, phone;
    ANativeWindow *w = phone_window(); int ww = w ? ANativeWindow_getWidth(w) : 0, wh = w ? ANativeWindow_getHeight(w) : 0;
    if (ww != last_w || wh != last_h) { last_w = ww; last_h = wh; phone = ov_layout_for_size(ww, wh);
        if (wh > 0) LOG("[overlay] window %dx%d: the %s layout", ww, wh, phone ? "phone" : "tablet"); }
    return phone;
}
int dm_overlay_custom(int layout, int ctl, float *x, float *y, float *s) {   // (the editor's own while it is open)
    if (layout < 0 || layout > 1 || ctl < 0 || ctl >= OV_N) return 0;
    _Atomic int *c = g_cust[atomic_load(&g_edit) ? 1 : 0][layout][ctl]; int cx = atomic_load(&c[0]), cy = atomic_load(&c[1]), z = atomic_load(&c[2]);
    *s = z ? z / 1000.0f : 1;
    if (!cx || !cy) return 0;
    *x = (cx - 1) / 10000.0f; *y = (cy - 1) / 10000.0f; return 1;
}
int dm_overlay_custom_ver(void) { return atomic_load(&g_cust_ver); }
int dm_overlay_hidden(int layout, int ctl) { return layout >= 0 && layout <= 1 && ctl >= 0 && ctl < OV_N && atomic_load(&g_cust[atomic_load(&g_edit) ? 1 : 0][layout][ctl][3]); }
int dm_overlay_off(int layout, int ctl) { return !atomic_load(&g_edit) && dm_overlay_hidden(layout, ctl); }
float dm_overlay_opacity(void) { return atomic_load(&g_set_opacity) / 100.0f; }
static void cust_place(int set, int l, int i, float x, float y) { atomic_store(&g_cust[set][l][i][0], 1 + (int)lroundf(x * 10000)); atomic_store(&g_cust[set][l][i][1], 1 + (int)lroundf(y * 10000)); }
int64_t dm_overlay_pressed(int ctl, uint64_t bit) {
    if (ctl < 0 || ctl >= OV_N) return 0;
    if (OV_CTLS[ctl].kind == OV_MENU) return atomic_load(&g_menu_t);
    if (OV_CTLS[ctl].kind == OV_STICK_L || OV_CTLS[ctl].kind == OV_STICK_R) return atomic_load(&g_stick_t[OV_CTLS[ctl].kind == OV_STICK_R]);
    if (OV_CTLS[ctl].kind == OV_BUTTON) bit = OV_CTLS[ctl].btn;
    return bit && bit < 1u << 16 ? atomic_load(&g_press_t[__builtin_ctzll(bit)]) : 0;
}
static int g_dpi;   // the display's density (AConfiguration), 0 = unknown
float dm_overlay_scale(void) {
    static int last_w = -1, last_h = -1; static float s = 1;
    ANativeWindow *w = phone_window(); int ww = w ? ANativeWindow_getWidth(w) : 0, h = w ? ANativeWindow_getHeight(w) : 0;
    if (h != last_h || ww != last_w) { last_w = ww; last_h = h; s = 1;   // (the width picks the layout, whose buttons are sized for its form factor)
        float face = 0.1f; const ov_ctl *L = OV_CTLS; for (int i = 0; i < OV_N; i++) if (L[i].kind == OV_BUTTON && L[i].btn == DM_BTN_A) face = L[i].w;   // A's width, in window heights
        if (h > 0 && g_dpi > 0 && g_dpi < 0xfffe) { float mm = face * (float)h / (float)g_dpi * 25.4f; s = OV_MIN_MM / mm; if (s < 1) s = 1; if (s > 1.45f) s = 1.45f; s = ov_fit_scale(OV_CTLS, (float)ww, (float)h, s); }   // (no further than the controls fit the window)
        LOG("[overlay] window %dx%d, density %d dpi: face buttons %.1f mm, controls x%.2f", ww, h, g_dpi, h > 0 && g_dpi > 0 && g_dpi < 0xfffe ? face * h / g_dpi * 25.4f : 0, s); }
    return s;
}
void dm_overlay_knob(int stick, float *dx, float *dy) { *dx = atomic_load(&g_knob[stick][0]) / 10000.0f; *dy = atomic_load(&g_knob[stick][1]) / 10000.0f; }
// Geometry and state come from the same tables and saved settings as native hit-testing.
JNIEXPORT jfloatArray JNICALL Java_org_nx2a_ExternalActivity_externalOverlay(JNIEnv *env, jclass cls, jint width, jint height) {
    (void)cls;
    if (width <= 0 || height <= 0) return NULL;
    int layout = ov_layout_for_size(width, height);
    jfloat values[5 + OV_N * 9];
    int state = dm_menu_state();
    values[0] = layout; values[1] = dm_overlay_scale(); values[2] = dm_overlay_opacity(); values[3] = state < 0 ? -1 : state & 0xffffff;
    for (int i = 0; i < OV_N; i++) {
        const ov_ctl *c = &OV_LAYOUTS[layout][i]; float b[4], kx = 0, ky = 0;
        ov_box(c, (float)width, (float)height, b);
        if (c->kind == OV_STICK_L || c->kind == OV_STICK_R) dm_overlay_knob(c->kind == OV_STICK_R, &kx, &ky);
        uint64_t held = 0;
        if (c->kind == OV_DPAD) { for (int d = 0; d < 4; d++) if (dm_overlay_pressed(i, OV_DPAD_BITS[d])) held |= 1u << d; }
        else if (dm_overlay_pressed(i, 0)) held = 1;
        jfloat *v = &values[4 + i * 9];
        // v[0] = -1: hidden by the player (ExternalActivity draws nothing of it)
        v[0] = ov_off(c) ? -1 : c->kind; v[1] = b[0]; v[2] = b[1]; v[3] = b[2]; v[4] = b[3]; v[5] = (jfloat)held; v[6] = kx; v[7] = ky; v[8] = ov_cscale(c);
    }
    values[4 + OV_N * 9] = dm_overlay_visible();
    jfloatArray out = (*env)->NewFloatArray(env, (jsize)(5 + OV_N * 9));
    if (out) (*env)->SetFloatArrayRegion(env, out, 0, (jsize)(5 + OV_N * 9), values);
    return out;
}
static int g_ptr_ctl[32];   // pointer id → control index + 1 (0 = the touch screen)
static int ctl_at(float x, float y, float W, float H) {   // the control under (x, y) in pixels, -1 if none (in the layout editor: every control, shown or not)
    int best = -1, vis = atomic_load(&g_edit) ? 1 : dm_overlay_visible(), shown = vis == 1; float bd = 1e9f;
    if (!vis) return -1;   // no overlay: every touch is the game's
    for (int i = 0; i < OV_N; i++) {
        const ov_ctl *c = &OV_CTLS[i]; if (!shown && c->kind != OV_MENU) continue;   // hidden: only the menu button answers
        if (ov_off(c)) continue;   // hidden by the player in the layout editor
        float b[4]; ov_box(c, W, H, b);
        float d = ov_sdf(c, b, x - b[0], y - b[1]), reach = ov_reach(c, b, H), m = fminf(b[2], b[3]);
        if (d < reach && (d + m) / (m + reach) < bd) { bd = (d + m) / (m + reach); best = i; }   // between two, the one the touch is deeper into, for its size
    }
    if (shown && !atomic_load(&g_edit) && (best < 0 || (OV_CTLS[best].kind == OV_BUTTON && (OV_CTLS[best].btn & (DM_BTN_A | DM_BTN_B | DM_BTN_X | DM_BTN_Y))))) {
        uint64_t pair = ov_face_combo(OV_CTLS, x, y, W, H);
        if (pair) for (int i = 0; i < OV_N; i++) if (OV_CTLS[i].kind == OV_BUTTON && (OV_CTLS[i].btn & pair)) return i;
    }
    return best;
}
// dragging the menu button: the pointer holding it, whether it has moved past OV_DRAG_SLOP, where it went down and the finger's offset from the button's centre
static struct { int id, moved; float x0, y0, gx, gy; } g_drag = {-1};
static void menu_grab(int id, const ov_ctl *m, float x, float y, float W, float H) {
    float b[4]; ov_box(m, W, H, b);
    g_drag.id = id; g_drag.moved = 0; g_drag.x0 = x; g_drag.y0 = y; g_drag.gx = b[0] - x; g_drag.gy = b[1] - y;
}
static void menu_drag(const ov_ctl *m, float x, float y, float W, float H) {   // the button follows the finger once it has moved, kept whole on the screen
    if (!g_drag.moved && hypotf(x - g_drag.x0, y - g_drag.y0) < OV_DRAG_SLOP * H) return;
    g_drag.moved = 1;
    float b[4]; ov_box(m, W, H, b);
    float px = ov_drag_pos(x + g_drag.gx, b[2], W, 0), py = ov_drag_pos(y + g_drag.gy, b[3], H, 0); int l, i = ov_index(m, &l);
    cust_place(0, l, i, px / W, py / H); atomic_fetch_add(&g_cust_ver, 1);
}

// ---- controllers attached: InputDevice's list, read once a second through JNI (a code-less NativeActivity has no InputManager listener). A device counts
// when it is a gamepad or a joystick and not virtual (the tablet's own keys, touch screen and pen are neither), and once it has sent input: Xiaomi phones
// keep a system device "uinput-xiaomi" attached for good that says it is a gamepad (issues #22, #210). The count controls vibration routing.
// When the last used controller goes, the overlay can return and vibration returns to the phone ----
static int jx(JNIEnv *E) { if (!(*E)->ExceptionCheck(E)) return 0; (*E)->ExceptionClear(E); return 1; }
static void *pad_watch(void *a) {
    JavaVM *vm = a; JNIEnv *E; pthread_setname_np(pthread_self(), "padwatch");
    if ((*vm)->AttachCurrentThread(vm, &E, NULL) != JNI_OK) { LOG("[RED] [input] cannot attach to the Java VM: attached controllers are not watched"); return NULL; }
    jclass cID = (*E)->FindClass(E, "android/view/InputDevice");
    jmethodID mIds = cID ? (*E)->GetStaticMethodID(E, cID, "getDeviceIds", "()[I") : NULL, mGet = cID ? (*E)->GetStaticMethodID(E, cID, "getDevice", "(I)Landroid/view/InputDevice;") : NULL;
    jmethodID mSrc = cID ? (*E)->GetMethodID(E, cID, "getSources", "()I") : NULL, mVirt = cID ? (*E)->GetMethodID(E, cID, "isVirtual", "()Z") : NULL;
    jmethodID mName = cID ? (*E)->GetMethodID(E, cID, "getName", "()Ljava/lang/String;") : NULL;
    jmethodID mVen = cID ? (*E)->GetMethodID(E, cID, "getVendorId", "()I") : NULL, mProd = cID ? (*E)->GetMethodID(E, cID, "getProductId", "()I") : NULL;
    jmethodID mKeys = cID ? (*E)->GetMethodID(E, cID, "hasKeys", "([I)[Z") : NULL;   // (optional: without it the menu button stays)
    static const jint MENU_KEYS[3] = {AKEYCODE_BACK, AKEYCODE_BUTTON_MODE, AKEYCODE_MENU};   // on_input's menu keys
    jintArray keys = mKeys ? (*E)->NewIntArray(E, 3) : NULL; if (keys) (*E)->SetIntArrayRegion(E, keys, 0, 3, MENU_KEYS);
    if (jx(E) || !mIds || !mGet || !mSrc || !mVirt || !mName || !mVen || !mProd) { LOG("[RED] [input] InputDevice methods missing: attached controllers are not watched"); return NULL; }
    int prev[16], used[16], np = 0, menukey = 0;
    for (;;) {
        int ids[16], on[16], n = 0, nu = 0, in = atomic_load(&g_pad_in), mk = 0;
        jintArray arr = (*E)->CallStaticObjectMethod(E, cID, mIds);
        if (!jx(E) && arr) {
            jsize len = (*E)->GetArrayLength(E, arr); jint *v = (*E)->GetIntArrayElements(E, arr, NULL);
            for (jsize i = 0; v && i < len && n < 16; i++) {
                jobject d = (*E)->CallStaticObjectMethod(E, cID, mGet, v[i]); if (jx(E) || !d) continue;
                jint src = (*E)->CallIntMethod(E, d, mSrc); jboolean virt = (*E)->CallBooleanMethod(E, d, mVirt);
                if (!jx(E) && !virt && ((src & AINPUT_SOURCE_GAMEPAD) == AINPUT_SOURCE_GAMEPAD || (src & AINPUT_SOURCE_JOYSTICK) == AINPUT_SOURCE_JOYSTICK)) {
                    int known = 0, was = 0; for (int k = 0; k < np; k++) if (prev[k] == v[i]) { known = 1; was = used[k]; }
                    ids[n] = v[i]; on[n] = was || v[i] == in; nu += on[n];
                    if (!known) {
                        jstring s = (*E)->CallObjectMethod(E, d, mName); const char *nm = !jx(E) && s ? (*E)->GetStringUTFChars(E, s, NULL) : NULL;
                        jint ven = (*E)->CallIntMethod(E, d, mVen), prod = (*E)->CallIntMethod(E, d, mProd); jx(E);
                        LOG("[input] controller attached: %s (device %d, sources 0x%x, %04x:%04x)", nm ? nm : "?", v[i], src, ven, prod);
                        if (nm) (*E)->ReleaseStringUTFChars(E, s, nm);
                        if (s) (*E)->DeleteLocalRef(E, s);
                    }
                    if (on[n] && !was) LOG("[input] controller in use (device %d)", v[i]);
                    if (on[n] && keys) {   // one of the menu's keys on a controller in use
                        jbooleanArray has = (*E)->CallObjectMethod(E, d, mKeys, keys); jboolean h[3] = {0, 0, 0};
                        if (!jx(E) && has) { if ((*E)->GetArrayLength(E, has) >= 3) (*E)->GetBooleanArrayRegion(E, has, 0, 3, h); (*E)->DeleteLocalRef(E, has); }
                        mk |= h[0] || h[1] || h[2];
                    }
                    n++;
                }
                (*E)->DeleteLocalRef(E, d);
            }
            if (v) (*E)->ReleaseIntArrayElements(E, arr, v, JNI_ABORT);
            (*E)->DeleteLocalRef(E, arr);
        }
        for (int k = 0; k < np; k++) { int still = 0; for (int i = 0; i < n; i++) still |= ids[i] == prev[k]; if (!still) LOG("[input] controller detached (device %d)", prev[k]); }
        if (atomic_exchange(&g_pads, nu) && !nu) {   // the last one went: the device's motor for rumble
            atomic_store(&g_overlay, 1); dm_haptics_android_device(-1);
            LOG("[input] no controller in use: vibration returns to the phone");
        }
        if (mk != menukey) { menukey = mk; atomic_store(&g_pad_menukey, mk); LOG("[overlay] the menu button %s", mk ? "hides while the controller plays: the controller has a key for the menu" : "shows: no controller in use has a key for the menu"); }
        memcpy(prev, ids, sizeof ids); memcpy(used, on, sizeof on); np = n;
        sleep(1);
    }
    return NULL;
}

// ---- temperatures for the frame rate counter (gfx_vk_present.c ov_fps): read every 2 s while the counter is shown ----
// Battery: the sticky ACTION_BATTERY_CHANGED intent's "temperature" (tenths of a degree), which any app may read. Chip: the hottest of the kernel's
// thermal zones whose type starts with "cpu" or "gpu" and whose temp file the app may read (SELinux keeps them from apps on some phones: then none).
static _Atomic int g_temp_soc = -1000, g_temp_bat = -1000;   // whole degrees C; -1000 = not known
static _Atomic int64_t g_temp_want;                           // when the counter last asked (CLOCK_MONOTONIC ns)
void dm_android_temps(int *soc, int *bat) { atomic_store(&g_temp_want, mono_ns()); *soc = atomic_load(&g_temp_soc); *bat = atomic_load(&g_temp_bat); }
static void *temp_watch(void *a) {
    JavaVM *vm = a; JNIEnv *E = NULL; pthread_setname_np(pthread_self(), "tempwatch");
    char zones[64][64]; int nz = 0;
    for (int i = 0; i < 200 && nz < 64; i++) {   // the chip's zones the app can read
        char p[96], t[48] = {0}; snprintf(p, sizeof p, "/sys/class/thermal/thermal_zone%d/type", i); FILE *f = fopen(p, "r"); if (!f) { if (errno == ENOENT) break; continue; }
        if (!fgets(t, sizeof t, f)) t[0] = 0; fclose(f);
        if (strncmp(t, "cpu", 3) && strncmp(t, "gpu", 3)) continue;
        if (strstr(t, "trip")) continue;   // a trip point's limit, not a sensor (Galaxy Z Fold: cpu-hw-trip-0 always 105000)
        snprintf(p, sizeof p, "/sys/class/thermal/thermal_zone%d/mode", i); f = fopen(p, "r");
        if (f) { char m[16] = {0}; if (!fgets(m, sizeof m, f)) m[0] = 0; fclose(f); if (!strncmp(m, "disabled", 8)) continue; }   // a zone the kernel does not poll
        snprintf(p, sizeof p, "/sys/class/thermal/thermal_zone%d/temp", i); f = fopen(p, "r"); if (!f) continue; long v; int ok = fscanf(f, "%ld", &v) == 1; fclose(f);
        if (ok) snprintf(zones[nz++], sizeof zones[0], "%s", p);
    }
    jobject filter = NULL; jmethodID mReg = NULL, mInt = NULL;
    if ((*vm)->AttachCurrentThread(vm, &E, NULL) == JNI_OK) {
        jclass cF = (*E)->FindClass(E, "android/content/IntentFilter"); jmethodID mF = cF ? (*E)->GetMethodID(E, cF, "<init>", "(Ljava/lang/String;)V") : NULL;
        jstring act = (*E)->NewStringUTF(E, "android.intent.action.BATTERY_CHANGED");
        jobject fl = mF && act ? (*E)->NewObject(E, cF, mF, act) : NULL; if (!jx(E) && fl) filter = (*E)->NewGlobalRef(E, fl);
        jclass cA = (*E)->GetObjectClass(E, g_act->clazz);
        mReg = cA ? (*E)->GetMethodID(E, cA, "registerReceiver", "(Landroid/content/BroadcastReceiver;Landroid/content/IntentFilter;)Landroid/content/Intent;") : NULL;
        jclass cI = (*E)->FindClass(E, "android/content/Intent"); mInt = cI ? (*E)->GetMethodID(E, cI, "getIntExtra", "(Ljava/lang/String;I)I") : NULL;
        if (jx(E)) mReg = NULL;
    }
    LOG("[perf] temperatures for the counter: %d chip zones readable, battery %s", nz, filter && mReg && mInt ? "from the battery intent" : "not readable");
    if (!nz && !(filter && mReg && mInt)) return NULL;
    for (;;) {
        struct timespec d = {2, 0}; nanosleep(&d, NULL);
        if (mono_ns() - atomic_load(&g_temp_want) > 5000000000ll) continue;   // the counter is off
        long hi = -1000000;
        for (int i = 0; i < nz; i++) { FILE *f = fopen(zones[i], "r"); long v; if (f) { if (fscanf(f, "%ld", &v) == 1) { if (v > 1000) v /= 1000; if (v > hi && v < 200) hi = v; } fclose(f); } }   // millidegrees on most kernels
        if (nz) atomic_store(&g_temp_soc, hi > -1000000 ? (int)hi : -1000);
        if (filter && mReg && mInt) {
            jobject in = (*E)->CallObjectMethod(E, g_act->clazz, mReg, NULL, filter); jstring k = (*E)->NewStringUTF(E, "temperature");
            jint t = in && k && !jx(E) ? (*E)->CallIntMethod(E, in, mInt, k, -10000) : -10000; if (jx(E)) t = -10000;
            atomic_store(&g_temp_bat, t > -10000 ? (t + 5) / 10 : -1000);
            if (in) (*E)->DeleteLocalRef(E, in); if (k) (*E)->DeleteLocalRef(E, k);
        }
    }
    return NULL;
}

// ---- the pause menu: the game is held while it is open; touches and a controller's keys go to it. Three tabs (overlay_layout.h OVM_TAB_ITEMS): the game's
// (the mode, the frame rate counter), the touch controls' (shown, opacity, the layout editor, rumble) and the app's (the highlight colour, exporting and
// importing the save); Resume under them. Each row's value is dm_menu_value's, changed by menu_set; what a press does follows the row's kind ----
static int g_menu_last = 1;   // the row last highlighted (a controller's, or the last row tapped): the highlight comes back there (while the app runs)
static _Atomic int g_menu_hi = -1, g_menu_tab, g_menu_down;   // the highlighted row (a controller's selection, or the row a finger is on), -1 none; the open tab;
                                                              // a finger is down on the highlighted row (the row is drawn pressed)
static int g_menu_ptr = -1, g_menu_row = -2;   // the pointer pressing in the menu, and the row it went down on (-3 the menu button)
static _Atomic int g_tx_ver;   // the text entry's edit count: the panel is drawn again when it changes
static _Atomic int g_ed_st;    // the layout editor's state for dm_menu_state (overlay_layout.h OVE_ST)
static _Atomic int g_sp_on, g_sp_hi = -1, g_sp_ver;   // the save picker (below): open, its highlighted row, its edit count
static _Atomic int g_sv_st, g_sv_n;                   // Export save: its outcome (overlay_layout.h OVX_*), the exports made
static int temps_on(void) { int v = atomic_load(&g_set_temps); return v < 0 ? atomic_load(&g_set_fps) : v; }   // Show temperatures (unset: as Show FPS)
int dm_menu_state(void) {
    if (atomic_load(&g_tx_on)) return 1 << 16 | (atomic_load(&g_tx_ver) & 0xffff);   // the text entry's panel (gfx_vk_present.c ov_text_panel)
    if (!atomic_load(&g_menu)) return -1;
    if (atomic_load(&g_edit)) return OVE_ST | atomic_load(&g_ed_st);
    if (atomic_load(&g_sp_on)) return OVS_ST | (atomic_load(&g_sp_hi) + 1) << 4 | (atomic_load(&g_sp_ver) & 255) << 20;
    return atomic_load(&g_set_overlay) | atomic_load(&g_set_rumble) << 1 | (dm_opmode() != 0) << 2 | atomic_load(&g_set_fps) << 3 | temps_on() << 11 | (atomic_load(&g_menu_hi) + 1) << 4 |
           atomic_load(&g_menu_tab) << 8 | (atomic_load(&g_menu_down) ? OVM_PRESSED : 0) | atomic_load(&g_sv_st) << 20 | (atomic_load(&g_sv_n) & 15) << 24;
}
static int accent_index(unsigned c) { for (int i = 0; i < OVM_ACCENTS_N; i++) if (OVM_ACCENTS[i] == c) return i; return -1; }
static int gpu_floor_choice(void); static void gpu_floor_pick(void);   // below, with the floor thread
int dm_menu_value(int item) {
    switch (item) {
    case OVM_OVERLAY: return atomic_load(&g_set_overlay);
    case OVM_RUMBLE: return atomic_load(&g_set_rumble);
    case OVM_FPS: return atomic_load(&g_set_fps);
    case OVM_TEMPS: return temps_on();
    case OVM_OPACITY: return atomic_load(&g_set_opacity);
    case OVM_MODE: return atomic_load(&g_set_res);
    case OVM_ACCENT: return accent_index(atomic_load(&g_set_accent));
    case OVM_FPSPOS: return dm_overlay_fps_pos();
    case OVM_PADLABEL: return pad_labels();
    case OVM_MOUSE: return dm_front_mouse() != 0;
    case OVM_THERMAL: return dm_front_thermal() > 0;
    case OVM_GPUFLOOR: return gpu_floor_choice();
    case OVM_FRAMECAP: return dm_frame_cap_how() == DM_FRAME_CAP_NONE ? OVM_NA : atomic_load(&g_set_cap);
    default: return 0;
    }
}
const char *dm_menu_choice_name(int item, int v) {
    if (item < 0 || item >= OVM_N) return "";
    if (v == OVM_NA) return OVM_ROW[item].na ? OVM_ROW[item].na : "";
    if (item == OVM_MODE && dm_res_levels() != DM_RES_N) return v == DM_RES_MAX ? "Docked" : "Handheld";   // the two modes only
    return v >= 0 && v < OVM_ROW[item].n ? OVM_ROW[item].choices[v] : "Custom";
}
static int menu_choice_ok(int item, int v) {   // a value the game can take: Resolution's two levels without the game's own settings (Balanced, Max)
    return item != OVM_MODE || dm_res_levels() == DM_RES_N || v == DM_RES_BALANCED || v == DM_RES_MAX;
}
unsigned dm_menu_accent(void) { return atomic_load(&g_set_accent) & 0xffffff; }
int dm_game_title(char *out, int cap) { if (cap > 0) snprintf(out, (size_t)cap, "%s", g_title); return g_title[0] != 0; }
static void menu_set(int item, int v) {   // a row's new value (a switch, a choice's index, the slider's percent), saved
    if (item == OVM_OVERLAY) { atomic_store(&g_set_overlay, v != 0); atomic_store(&g_overlay, 1); }
    else if (item == OVM_RUMBLE) { atomic_store(&g_set_rumble, v != 0); dm_haptics_android_set(v != 0); }
    else if (item == OVM_FPS) { if (atomic_load(&g_set_temps) < 0) atomic_store(&g_set_temps, atomic_load(&g_set_fps)); atomic_store(&g_set_fps, v != 0); }   // (the temperatures stay as they were shown)
    else if (item == OVM_TEMPS) atomic_store(&g_set_temps, v != 0);
    else if (item == OVM_MODE && v >= 0 && v < DM_RES_N) {   // Resolution: the mode reaches the game as the notification it takes when the menu closes, the GPU target at its next frame
        atomic_store(&g_set_res, v); dm_set_opmode(v == DM_RES_MAX); dm_res_set(v);
        LOG("[menu] resolution %s (%s; the game takes the mode as it goes on)", RES_KEYS[v], v == DM_RES_MAX ? "docked" : "handheld");
    } else if (item == OVM_FPSPOS && v >= 0 && v < 4) { atomic_store(&g_set_fpspos, v); LOG("[menu] FPS position %s", OVM_FPSPOS_KEYS[v]); }
    else if (item == OVM_FRAMECAP && v >= 0 && v < OVM_NCAPS) {   // applied at once (hle_framecap.c)
        if (dm_frame_cap_how() == DM_FRAME_CAP_NONE) return;
        atomic_store(&g_set_cap, v); dm_frame_cap_set(OVM_CAPS[v]);
        if (OVM_CAPS[v]) LOG("[menu] frame cap %d", OVM_CAPS[v]); else LOG("[menu] frame cap off");
    }
    else if (item == OVM_ACCENT && v >= 0 && v < OVM_ACCENTS_N) atomic_store(&g_set_accent, OVM_ACCENTS[v]);
    else if (item == OVM_GPUFLOOR && v >= 0 && v < OVM_NFLOORS) {   // applied at once, by the floor thread
        atomic_store(&g_set_floor, v); gpu_floor_pick();
        if (OVM_FLOORS[v]) LOG("[menu] GPU clock floor %d%% of the top clock", OVM_FLOORS[v]); else LOG("[menu] GPU clock floor off");
    }
    else if (item == OVM_THERMAL && v >= 0 && v < 2) { atomic_store(&g_set_thermal, v); LOG("[menu] thermal throttle %s", v ? "on" : "off"); }
    else if (item == OVM_MOUSE && v >= 0 && v < 2) { atomic_store(&g_set_mouse, v); LOG("[menu] Joy-Con mouse %s", v ? "auto" : "off"); }
    else if (item == OVM_PADLABEL && v >= 0 && v < 2) { atomic_store(&g_set_padlabel, v); LOG("[menu] face buttons by %s", v ? "label" : "position"); }
    else if (item == OVM_OPACITY) atomic_store(&g_set_opacity, v < OV_OPACITY_MIN ? OV_OPACITY_MIN : v > OV_OPACITY_MAX ? OV_OPACITY_MAX : v);
    else return;
    settings_save();
}
int dm_overlay_fps(void) { return atomic_load(&g_set_fps) && !atomic_load(&g_menu); }
int dm_overlay_temps(void) { return temps_on() && !atomic_load(&g_menu); }   // gfx_vk_present.c draws the temperatures in the counter's box (Show temperatures)   // gfx_vk_present.c draws the counter (not while the menu holds the game)
static void menu_open(int hi) {
    if (!atomic_load(&g_running) || atomic_exchange(&g_menu, 1)) return;
    if (hi >= 0) hi = g_menu_last >= 0 && g_menu_last < OVM_ROWS && ov_menu_item(atomic_load(&g_menu_tab), g_menu_last) >= 0 ? g_menu_last : 1;   // a controller opened it: the last row used
    atomic_store(&g_menu_hi, hi); atomic_store(&g_menu_down, 0); g_menu_ptr = -1;
    if (atomic_load(&g_sv_st) != OVX_BUSY) atomic_store(&g_sv_st, OVX_IDLE);   // the last export's note goes (one still copying stays)
    dm_input_set(0, 0, 0, 0, 0); dm_touch_set(NULL, 0); atomic_store(&g_touch_btn, 0);   // nothing stays pressed in the game
    for (int i = 0; i < 16; i++) atomic_store(&g_press_t[i], 0);
    for (int s = 0; s < 2; s++) { atomic_store(&g_stick_t[s], 0); for (int a = 0; a < 2; a++) atomic_store(&g_knob[s][a], 0); }
    if (g_pad_used) { g_btn = 0; memset(g_ax, 0, sizeof g_ax); pad_publish(); }
    for (int i = 0; i < 32; i++) g_ptr_ctl[i] = 0;
    dm_haptics_suspend(4, 1);   // (bits 1 and 2 are the iOS front end's reasons)
    hold_sync(); dm_overlay_redraw();
    LOG("[menu] opened: the game is paused");
}
static void ed_cancel(void); static void sp_back(void);
static void menu_close(void) {
    if (atomic_load(&g_edit)) { ed_cancel(); return; }   // the editor first: its changes are kept or thrown away before the game goes on
    if (atomic_load(&g_sp_on)) { sp_back(); return; }    // the save picker: back a step
    if (!atomic_exchange(&g_menu, 0)) return;
    g_menu_ptr = -1; atomic_store(&g_menu_t, 0); atomic_store(&g_menu_down, 0);
    hold_sync(); dm_haptics_suspend(4, 0); dm_overlay_redraw();
    LOG("[menu] closed: the game goes on (gamepad %s, opacity %d%%, rumble %s, frame rate %s, %s, resolution %s, frame cap %d)", atomic_load(&g_set_overlay) ? "on" : "off", atomic_load(&g_set_opacity),
        atomic_load(&g_set_rumble) ? "on" : "off", atomic_load(&g_set_fps) ? "shown" : "hidden", dm_opmode() ? "docked" : "handheld", RES_KEYS[atomic_load(&g_set_res)],
        OVM_CAPS[atomic_load(&g_set_cap)]);   // (the mode in force: a Resolution change reaches the game as it goes on)
}
static void menu_opacity(int v) {   // the slider: a new value is drawn at once and saved when the finger lifts (menu_touch) or with each controller step
    v = v < OV_OPACITY_MIN ? OV_OPACITY_MIN : v > OV_OPACITY_MAX ? OV_OPACITY_MAX : v;
    if (atomic_exchange(&g_set_opacity, v) != v) { dm_haptics_android_tap(); dm_overlay_redraw(); }
}
static void menu_tab(int t) { t = (t % OVM_TABS + OVM_TABS) % OVM_TABS; if (atomic_exchange(&g_menu_tab, t) != t) dm_haptics_android_tap(); dm_overlay_redraw(); }
static void menu_step(int item, int d) {   // left or right on a row: a choice's next or previous value (round), the slider by 5, the tabs
    int k = ov_menu_kind(item);
    if (k == OVR_SLIDER) { menu_opacity(atomic_load(&g_set_opacity) + 5 * d); settings_save(); }
    else if (k == OVR_TABS) menu_tab(atomic_load(&g_menu_tab) + d);
    else if (k == OVR_CHOICE) {
        int n = OVM_ROW[item].n, v = dm_menu_value(item);
        if (v == OVM_NA) { static int told; if (!told++) LOG("[menu] %s: not for this game", OVM_ROW[item].label); return; }   // (hle_framecap.c: a game that steps by its frames)
        v = v < 0 ? (d > 0 ? 0 : n - 1) : ((v + d) % n + n) % n;
        for (int k = 0; k < n && !menu_choice_ok(item, v); k++) v = ((v + d) % n + n) % n;
        menu_set(item, v); dm_haptics_android_tap(); dm_overlay_redraw();
    }
}
static void ed_open(void); static void sp_open(void); static void save_export(void);
static void menu_do(int item) {   // A on a row, or a tap on it: a switch flips, a choice steps on, a button does its thing
    int k = ov_menu_kind(item);
    if (k == OVR_TOGGLE) menu_set(item, !dm_menu_value(item));
    else if (k == OVR_CHOICE) { menu_step(item, 1); return; }
    else if (item == OVM_TAB) { menu_tab(atomic_load(&g_menu_tab) + 1); return; }
    else if (item == OVM_EDIT) { dm_haptics_android_tap(); ed_open(); return; }
    else if (item == OVM_OPACITY) return;   // (the slider is dragged, or stepped with left and right)
    else if (item == OVM_EXPORT) { if (atomic_load(&g_sv_st) == OVX_BUSY) return; save_export(); }
    else if (item == OVM_IMPORT) { dm_haptics_android_tap(); sp_open(); return; }
    if (item >= 0 && item < OVM_N) dm_haptics_android_tap();
    if (item == OVM_RESUME) menu_close(); else dm_overlay_redraw();
}
static void menu_touch(AInputEvent *e, float W, float H) {   // a press on a row does it when it lifts there; a tap outside the panel or on the menu button closes
    int act = AMotionEvent_getAction(e) & AMOTION_EVENT_ACTION_MASK, ai = AMotionEvent_getAction(e) >> AMOTION_EVENT_ACTION_POINTER_INDEX_SHIFT;
    int i = act == AMOTION_EVENT_ACTION_DOWN || act == AMOTION_EVENT_ACTION_UP ? 0 : ai, id = AMotionEvent_getPointerId(e, (size_t)i) & 31;
    float x = AMotionEvent_getX(e, (size_t)i), y = AMotionEvent_getY(e, (size_t)i), p[4]; ov_menu_rect(W, H, p);
    int c = ctl_at(x, y, W, H), row = c >= 0 && OV_CTLS[c].kind == OV_MENU ? -3 : ov_menu_row_at(x, y, W, H), tab = atomic_load(&g_menu_tab);
    int item = row >= 0 ? ov_menu_item(tab, row) : -1;
    if (act == AMOTION_EVENT_ACTION_DOWN || act == AMOTION_EVENT_ACTION_POINTER_DOWN) {
        if (g_menu_ptr >= 0) return;
        g_menu_ptr = id; g_menu_row = row;
        if (item == OVM_TAB) menu_tab(ov_menu_tab_at(x, p));   // a tab opens as the finger touches it
        else if (item == OVM_OPACITY) menu_opacity(ov_menu_track_value(x, p, row));
        if (row >= 0) { atomic_store(&g_menu_hi, item >= 0 ? row : -1); atomic_store(&g_menu_down, item >= 0 && item != OVM_TAB); dm_overlay_redraw(); }
    } else if (act == AMOTION_EVENT_ACTION_MOVE && g_menu_ptr >= 0 && g_menu_row >= 0) {   // the slider follows the finger; another row's highlight follows it off the row and back
        for (size_t k = 0; k < AMotionEvent_getPointerCount(e); k++) if ((AMotionEvent_getPointerId(e, k) & 31) == g_menu_ptr) {
            float kx = AMotionEvent_getX(e, k), ky = AMotionEvent_getY(e, k);
            if (ov_menu_item(tab, g_menu_row) == OVM_OPACITY) { menu_opacity(ov_menu_track_value(kx, p, g_menu_row)); continue; }
            int r = ov_menu_row_at(kx, ky, W, H), hi = r == g_menu_row && ov_menu_item(tab, r) >= 0 ? r : -1;
            if (atomic_exchange(&g_menu_hi, hi) != hi) { atomic_store(&g_menu_down, hi >= 0 && ov_menu_item(tab, hi) != OVM_TAB); dm_overlay_redraw(); }
        }
    } else if ((act == AMOTION_EVENT_ACTION_UP || act == AMOTION_EVENT_ACTION_POINTER_UP) && id == g_menu_ptr) {
        int down = g_menu_row; g_menu_ptr = -1; atomic_store(&g_menu_hi, -1); atomic_store(&g_menu_down, 0);
        if (row == down && row > 0) g_menu_last = row;
        if (down >= 0 && ov_menu_item(tab, down) == OVM_OPACITY) { settings_save(); dm_overlay_redraw(); }
        else if (row == down && ov_menu_kind(item) == OVR_CHOICE && OVM_ROW[item].swatches && ov_menu_swatch_at(x, p, row, OVM_ROW[item].n) >= 0) {   // a colour tapped
            menu_set(item, ov_menu_swatch_at(x, p, row, OVM_ROW[item].n)); dm_haptics_android_tap(); dm_overlay_redraw(); }
        else if (row == down && ov_menu_kind(item) == OVR_CHOICE) menu_step(item, ov_menu_choice_dir(x, p, row));   // the value's left arrow goes back
        else if (row == down && item >= 0 && item != OVM_TAB) menu_do(item);
        else if (row == down && (row == -2 || row == -3)) { dm_haptics_android_tap(); menu_close(); }
        else dm_overlay_redraw();
    } else if (act == AMOTION_EVENT_ACTION_CANCEL) { g_menu_ptr = -1; atomic_store(&g_menu_hi, -1); atomic_store(&g_menu_down, 0); dm_overlay_redraw(); }
}
static void ed_key(uint64_t bit); static void sp_key(uint64_t bit);
static int menu_first_row(int tab) { for (int r = 1; r < OVM_ROWS; r++) if (ov_menu_item(tab, r) >= 0) return r; return OVM_ROWS - 1; }
static void sp_close_all(void);
static void menu_key(uint64_t bit) {   // a controller key pressed while the menu is open (also each repeat of a held direction: menu_nav). Up and down go through
    // the rows (the tabs' row too; round), left and right change the value on a row that has one (a slider, a choice, a switch: left off, right on) and the
    // tab on the tabs' row, L and R (ZL, ZR) the tab anywhere, keeping the row; A does the row, B and + close. Without a highlight (after a touch) the first key
    // shows it again on the last row used
    if (atomic_load(&g_edit)) { ed_key(bit); return; }
    if (atomic_load(&g_sp_on)) { if (bit == DM_BTN_PLUS) sp_close_all(); else sp_key(bit); return; }
    int hi = atomic_load(&g_menu_hi), tab = atomic_load(&g_menu_tab);
    if (bit == DM_BTN_B || bit == DM_BTN_PLUS) { menu_close(); return; }
    if (bit == DM_BTN_L || bit == DM_BTN_ZL || bit == DM_BTN_R || bit == DM_BTN_ZR) {
        menu_tab(tab + (bit == DM_BTN_L || bit == DM_BTN_ZL ? -1 : 1));
        int t = atomic_load(&g_menu_tab), r = hi >= 0 ? hi : g_menu_last;
        if (r > 0 && r < OVM_ROWS - 1 && ov_menu_item(t, r) < 0) { while (r > 1 && ov_menu_item(t, r) < 0) r--; if (ov_menu_item(t, r) < 0) r = menu_first_row(t); }   // (the other tab has fewer rows): its last one
        if (hi >= 0) { atomic_store(&g_menu_hi, r); g_menu_last = r; dm_overlay_redraw(); }
        return;
    }
    if (hi < 0 || ov_menu_item(tab, hi) < 0) {   // no highlight: shown again where it was, the key does nothing else
        hi = g_menu_last >= 0 && g_menu_last < OVM_ROWS && ov_menu_item(tab, g_menu_last) >= 0 ? g_menu_last : menu_first_row(tab);
        atomic_store(&g_menu_hi, hi); g_menu_last = hi; dm_haptics_android_tap(); dm_overlay_redraw(); return;
    }
    int item = ov_menu_item(tab, hi), k = ov_menu_kind(item);
    if (bit == DM_BTN_UP || bit == DM_BTN_DOWN) {
        for (int n = 0; n < OVM_ROWS; n++) { hi = (hi + (bit == DM_BTN_UP ? OVM_ROWS - 1 : 1)) % OVM_ROWS; if (ov_menu_item(tab, hi) >= 0) break; }
        atomic_store(&g_menu_hi, hi); g_menu_last = hi; dm_haptics_android_tap(); dm_overlay_redraw();
    } else if (bit == DM_BTN_LEFT || bit == DM_BTN_RIGHT) {
        int d = bit == DM_BTN_LEFT ? -1 : 1;
        if (k == OVR_SLIDER || k == OVR_CHOICE || k == OVR_TABS) menu_step(item, d);
        else if (k == OVR_TOGGLE && dm_menu_value(item) != (d > 0)) { menu_set(item, d > 0); dm_haptics_android_tap(); dm_overlay_redraw(); }
    }
    else if (bit == DM_BTN_A) menu_do(item);
}

// ---- the save manager (saves_android.c): Export save copies the game's committed save to the files folder's saves/ on a thread of its own (the row shows it
// copying, then the export's name); Import save opens the picker, in the menu's panel (overlay_layout.h OVS_*): the saves there newest first, a page of
// OVM_ROWS - 2 (left and right, L and R page), Back. A save chosen asks first; the import then waits for the next start (the save is not touched while the
// game runs) and the app ends, as the panel says. A controller: up and down choose, A does, B goes back; a finger taps a row, or outside the panel to go back ----
#define SP_MAX 64
#define SP_PER (OVM_ROWS - 2)
static pthread_mutex_t g_sp_mu = PTHREAD_MUTEX_INITIALIZER;   // the picker's list and the export's note
static struct { int n, page, confirm, closing, ptr, row; dm_save_entry e[SP_MAX], pick; } g_sp = {.ptr = -1};
static char g_sv_note[96];
int dm_save_note(char *out, int cap) { pthread_mutex_lock(&g_sp_mu); snprintf(out, (size_t)cap, "%s", g_sv_note); pthread_mutex_unlock(&g_sp_mu); return atomic_load(&g_sv_st); }
static void *export_run(void *a) {
    (void)a; char name[64], err[600]; int r = dm_saves_export("", name, sizeof name, err, sizeof err);
    pthread_mutex_lock(&g_sp_mu); snprintf(g_sv_note, sizeof g_sv_note, "%s", r ? err : name); pthread_mutex_unlock(&g_sp_mu);
    atomic_store(&g_sv_st, !r ? OVX_DONE : r == -ENOENT ? OVX_NOSAVE : OVX_FAILED); atomic_fetch_add(&g_sv_n, 1); dm_overlay_redraw();
    return NULL;
}
static void save_export(void) {
    pthread_t t; atomic_store(&g_sv_st, OVX_BUSY); dm_overlay_redraw();
    if (pthread_create(&t, NULL, export_run, NULL)) { atomic_store(&g_sv_st, OVX_FAILED); LOG("[RED] [saves] cannot start the export"); } else pthread_detach(t);
}
static void sp_changed(void) { atomic_fetch_add(&g_sp_ver, 1); dm_overlay_redraw(); }
static int sp_on_page(void) { int k = g_sp.n - g_sp.page * SP_PER; return k < 0 ? 0 : k > SP_PER ? SP_PER : k; }   // the caller holds g_sp_mu
static int sp_selectable(int row) {   // the rows a controller and a finger can choose
    pthread_mutex_lock(&g_sp_mu);
    int ok = g_sp.closing ? 0 : g_sp.confirm ? row == OVM_ROWS - 2 || row == OVM_ROWS - 1 : (row >= 1 && row <= sp_on_page()) || row == OVM_ROWS - 1;
    pthread_mutex_unlock(&g_sp_mu); return ok;
}
static void sp_open(void) {
    pthread_mutex_lock(&g_sp_mu);
    int n = g_sp.n = dm_saves_list(g_sp.e, SP_MAX); g_sp.page = g_sp.confirm = g_sp.closing = 0; g_sp.ptr = -1;
    pthread_mutex_unlock(&g_sp_mu);
    atomic_store(&g_sp_hi, n ? 1 : OVM_ROWS - 1); atomic_store(&g_sp_on, 1); sp_changed();
    LOG("[saves] picker opened: %d saves in %s", n, dm_saves_dir());
}
static void sp_back(void) {   // the confirmation → the list → the menu (nothing once the import is under way)
    pthread_mutex_lock(&g_sp_mu); int closing = g_sp.closing, confirm = g_sp.confirm; if (!closing) g_sp.confirm = 0; g_sp.ptr = -1; pthread_mutex_unlock(&g_sp_mu);
    if (closing) return;
    dm_haptics_android_tap();
    if (confirm) { atomic_store(&g_sp_hi, 1); sp_changed(); return; }
    atomic_store(&g_sp_on, 0); atomic_store(&g_menu_hi, -1); dm_overlay_redraw();
}
static void sp_close_all(void) {   // + or the menu key in the picker: the picker and the menu close (not once the import is under way)
    pthread_mutex_lock(&g_sp_mu); int closing = g_sp.closing; if (!closing) g_sp.confirm = 0; g_sp.ptr = -1; pthread_mutex_unlock(&g_sp_mu);
    if (closing) return;
    atomic_store(&g_sp_on, 0); menu_close();
}
static void *sp_end(void *a) {   // the panel's note shows for a moment, then the app ends; the next start imports the save (saves_android.c dm_saves_start)
    (void)a; sleep(3); LOG("[saves] the app ends for the import"); _exit(0);
}
static void sp_do(int row) {
    if (!sp_selectable(row)) return;
    dm_haptics_android_tap();
    if (row == OVM_ROWS - 1) { sp_back(); return; }
    pthread_mutex_lock(&g_sp_mu);
    if (!g_sp.confirm) { g_sp.pick = g_sp.e[g_sp.page * SP_PER + row - 1]; g_sp.confirm = 1; pthread_mutex_unlock(&g_sp_mu); atomic_store(&g_sp_hi, OVM_ROWS - 2); sp_changed(); return; }
    char name[64]; snprintf(name, sizeof name, "%s", g_sp.pick.name);
    pthread_mutex_unlock(&g_sp_mu);
    if (dm_saves_request_import(name)) return;
    pthread_mutex_lock(&g_sp_mu); g_sp.closing = 1; pthread_mutex_unlock(&g_sp_mu);
    atomic_store(&g_sp_hi, -1); sp_changed();
    pthread_t t; if (pthread_create(&t, NULL, sp_end, NULL)) sp_end(NULL); else pthread_detach(t);
}
static void sp_page(int d) {
    pthread_mutex_lock(&g_sp_mu); int pages = (g_sp.n + SP_PER - 1) / SP_PER, p = g_sp.page + d, ok = !g_sp.confirm && !g_sp.closing && p >= 0 && p < pages;
    if (ok) g_sp.page = p;
    int k = sp_on_page(); pthread_mutex_unlock(&g_sp_mu);
    if (!ok) return;
    int hi = atomic_load(&g_sp_hi); if (hi >= 1 && hi < OVM_ROWS - 1 && hi > k) atomic_store(&g_sp_hi, k);
    dm_haptics_android_tap(); sp_changed();
}
static void sp_key(uint64_t bit) {
    int hi = atomic_load(&g_sp_hi);
    if (bit == DM_BTN_UP || bit == DM_BTN_DOWN) {
        for (int k = 0; k < OVM_ROWS; k++) { hi = hi < 0 ? (bit == DM_BTN_UP ? OVM_ROWS - 1 : 0) : (hi + (bit == DM_BTN_UP ? OVM_ROWS - 1 : 1)) % OVM_ROWS; if (sp_selectable(hi)) break; }
        if (sp_selectable(hi)) { atomic_store(&g_sp_hi, hi); sp_changed(); }
    } else if (bit == DM_BTN_LEFT || bit == DM_BTN_L || bit == DM_BTN_ZL) sp_page(-1);
    else if (bit == DM_BTN_RIGHT || bit == DM_BTN_R || bit == DM_BTN_ZR) sp_page(1);
    else if (bit == DM_BTN_A) sp_do(hi);
    else if (bit == DM_BTN_B) sp_back();
}
static void sp_touch(AInputEvent *e, float W, float H) {   // a row is done when the finger that went down on it lifts there; outside the panel goes back
    int act = AMotionEvent_getAction(e) & AMOTION_EVENT_ACTION_MASK, ai = AMotionEvent_getAction(e) >> AMOTION_EVENT_ACTION_POINTER_INDEX_SHIFT;
    int i = act == AMOTION_EVENT_ACTION_DOWN || act == AMOTION_EVENT_ACTION_UP ? 0 : ai, id = AMotionEvent_getPointerId(e, (size_t)i) & 31;
    int row = ov_menu_row_at(AMotionEvent_getX(e, (size_t)i), AMotionEvent_getY(e, (size_t)i), W, H);
    if (act == AMOTION_EVENT_ACTION_DOWN || act == AMOTION_EVENT_ACTION_POINTER_DOWN) {
        if (g_sp.ptr >= 0) return;
        g_sp.ptr = id; g_sp.row = row;
        if (sp_selectable(row)) { atomic_store(&g_sp_hi, row); sp_changed(); }
    } else if ((act == AMOTION_EVENT_ACTION_UP || act == AMOTION_EVENT_ACTION_POINTER_UP) && id == g_sp.ptr) {
        int down = g_sp.row; g_sp.ptr = -1;
        if (row == down && row == -2) sp_back(); else if (row == down) sp_do(row);
    } else if (act == AMOTION_EVENT_ACTION_CANCEL) g_sp.ptr = -1;
}
static void sp_size(char *out, int cap, const dm_save_entry *s) {
    if (s->bytes >= 1 << 20) snprintf(out, (size_t)cap, "%ld file%s, %.1f MB", s->files, s->files == 1 ? "" : "s", s->bytes / 1048576.0);
    else snprintf(out, (size_t)cap, "%ld file%s, %lld KB", s->files, s->files == 1 ? "" : "s", (s->bytes + 1023) / 1024);
}
#define SP_ROW(i, st, ...) do { rows[i].style = st; snprintf(rows[i].left, sizeof rows[i].left, __VA_ARGS__); } while (0)
int dm_save_rows(ov_srow rows[OVM_ROWS]) {   // (gfx_vk_present.c ov_save_panel)
    if (!atomic_load(&g_sp_on)) return 0;
    memset(rows, 0, sizeof(ov_srow) * OVM_ROWS);
    pthread_mutex_lock(&g_sp_mu);
    if (g_sp.closing) {
        SP_ROW(0, OVS_TITLE, "Importing the save"); SP_ROW(1, OVS_ITEM, "%s", g_sp.pick.name); sp_size(rows[1].right, sizeof rows[1].right, &g_sp.pick);
        SP_ROW(2, OVS_NOTE, "The game closes now. Open it again:"); SP_ROW(3, OVS_NOTE, "the save is imported as it starts.");
    } else if (g_sp.confirm) {
        SP_ROW(0, OVS_TITLE, "Import this save?"); SP_ROW(1, OVS_ITEM, "%s", g_sp.pick.name); sp_size(rows[1].right, sizeof rows[1].right, &g_sp.pick);
        SP_ROW(2, OVS_NOTE, "Your current save is exported first,"); SP_ROW(3, OVS_NOTE, "then the game closes: open it again.");
        SP_ROW(OVM_ROWS - 2, OVS_BUTTON, "Import and close"); SP_ROW(OVM_ROWS - 1, OVS_BACK, "Cancel");
    } else {
        int pages = (g_sp.n + SP_PER - 1) / SP_PER, k = sp_on_page();
        SP_ROW(0, OVS_TITLE, "Import a save"); if (pages > 1) snprintf(rows[0].right, sizeof rows[0].right, "page %d of %d", g_sp.page + 1, pages);
        for (int j = 0; j < k; j++) { const dm_save_entry *s = &g_sp.e[g_sp.page * SP_PER + j]; SP_ROW(1 + j, OVS_ITEM, "%s", s->name); sp_size(rows[1 + j].right, sizeof rows[1 + j].right, s); }
        if (!g_sp.n) {
            const char *d = strstr(dm_saves_dir(), "Android/data/");
            SP_ROW(1, OVS_NOTE, "No saves yet. Export one, or put a save folder"); SP_ROW(2, OVS_NOTE, "(a console's, as its save tools copy it) in");
            SP_ROW(3, OVS_NOTE, "%s", d ? d : dm_saves_dir());
        }
        SP_ROW(OVM_ROWS - 1, OVS_BACK, "Back");
    }
    pthread_mutex_unlock(&g_sp_mu);
    return 1;
}

// ---- the layout editor (overlay_layout.h OVE_*): opened from the menu, the game stays held. A finger on a control chooses it and drags it; a second finger
// pinches it bigger or smaller; - and + resize the chosen control, or every control when none is chosen; Reset puts the layout's places and sizes back;
// Save keeps the editor's layout (settings.txt, the next start too) and Cancel throws it away, asking first when something changed. Either goes back to the
// menu. A controller: + saves, B cancels; in the prompt A throws away and B goes back to editing. It edits the layout in use (tablet or phone) ----
static struct { int ptr, ptr2, sel, held, prompt, dirty, bdown, bar; float gx, gy, d0, s0; } g_ed = {-1, -1, -1, -1};   // bar: ptr went down on the toolbar or prompt
static void ed_publish(void) { int hid = g_ed.sel >= 0 && atomic_load(&g_cust[1][dm_overlay_layout()][g_ed.sel][3]);
    atomic_store(&g_ed_st, g_ed.prompt | g_ed.dirty << 1 | (atomic_load(&g_set_grid) ? OVE_GRID_ON : 0) | (hid ? OVE_SEL_HIDDEN : 0) | (g_ed.held + 1) << 4 | (g_ed.sel + 1) << 8); dm_overlay_redraw(); }
static void ed_open(void) {
    for (int l = 0; l < 2; l++) for (int i = 0; i < OV_N; i++) for (int k = 0; k < CUST_K; k++) atomic_store(&g_cust[1][l][i][k], atomic_load(&g_cust[0][l][i][k]));
    g_ed.ptr = g_ed.ptr2 = g_ed.sel = g_ed.held = g_ed.bdown = -1; g_ed.prompt = g_ed.dirty = g_ed.bar = 0; g_menu_ptr = -1;
    atomic_store(&g_edit, 1); atomic_fetch_add(&g_cust_ver, 1); ed_publish();
    LOG("[overlay] layout editor opened (%s layout)", LAYOUT_NAMES[dm_overlay_layout()]);
}
static void ed_close(int save) {
    if (save) for (int l = 0; l < 2; l++) for (int i = 0; i < OV_N; i++) for (int k = 0; k < CUST_K; k++) atomic_store(&g_cust[0][l][i][k], atomic_load(&g_cust[1][l][i][k]));
    atomic_store(&g_edit, 0); atomic_fetch_add(&g_cust_ver, 1); g_menu_ptr = -1; atomic_store(&g_menu_hi, -1);
    if (save) settings_save();
    dm_haptics_android_tap(); dm_overlay_redraw();
    LOG("[overlay] layout editor closed: %s", save ? "the layout is saved" : g_ed.dirty ? "the changes are thrown away" : "no changes");
}
static void ed_cancel(void) { if (g_ed.dirty && !g_ed.prompt) { g_ed.prompt = 1; g_ed.held = -1; dm_haptics_android_tap(); ed_publish(); } else if (!g_ed.prompt) ed_close(0); }
static float ed_size(int l, int i) { int z = atomic_load(&g_cust[1][l][i][2]); return z ? z / 1000.0f : 1; }
static void ed_resize(int l, int i, float s) { s = fminf(fmaxf(s, OVE_SCALE_MIN), OVE_SCALE_MAX); atomic_store(&g_cust[1][l][i][2], (int)lroundf(s * 1000)); g_ed.dirty = 1; }
static void ed_do(int b) {
    int l = dm_overlay_layout();
    if (b == OVE_SMALLER || b == OVE_BIGGER) { float f = b == OVE_BIGGER ? 1.1f : 1 / 1.1f; for (int i = 0; i < OV_N; i++) if (g_ed.sel < 0 || i == g_ed.sel) ed_resize(l, i, ed_size(l, i) * f);
        atomic_fetch_add(&g_cust_ver, 1); }
    else if (b == OVE_RESET) { for (int i = 0; i < OV_N; i++) for (int k = 0; k < CUST_K; k++) atomic_store(&g_cust[1][l][i][k], 0); g_ed.sel = -1; g_ed.dirty = 1; atomic_fetch_add(&g_cust_ver, 1); }
    else if (b == OVE_HIDE) {   // the chosen control hidden in play, or shown again (the menu button stays: it opens this editor)
        if (g_ed.sel < 0 || OV_LAYOUTS[l][g_ed.sel].kind == OV_MENU) { ed_publish(); return; }
        _Atomic int *h = &g_cust[1][l][g_ed.sel][3]; atomic_store(h, !atomic_load(h)); g_ed.dirty = 1; atomic_fetch_add(&g_cust_ver, 1);
        LOG("[overlay] %s %s", OV_LAYOUTS[l][g_ed.sel].name, atomic_load(h) ? "hidden" : "shown again");
    }
    else if (b == OVE_GRID) { atomic_store(&g_set_grid, !atomic_load(&g_set_grid)); settings_save(); }   // a setting, saved now: Cancel does not undo it
    else if (b == OVE_CANCEL) { ed_cancel(); return; }
    else if (b == OVE_SAVE) { ed_close(1); return; }
    else if (b == OVE_DISCARD) { ed_close(0); return; }
    else if (b == OVE_KEEP) g_ed.prompt = 0;
    dm_haptics_android_tap(); ed_publish();
}
static void ed_key(uint64_t bit) {
    if (g_ed.prompt) { if (bit == DM_BTN_A) ed_do(OVE_DISCARD); else if (bit == DM_BTN_B) ed_do(OVE_KEEP); return; }
    if (bit == DM_BTN_PLUS) ed_do(OVE_SAVE); else if (bit == DM_BTN_B) ed_do(OVE_CANCEL);
}
static void ed_drag(float x, float y, float W, float H) {   // the chosen control's centre follows the finger, kept on the screen
    int l = dm_overlay_layout(); float b[4]; ov_box(&OV_LAYOUTS[l][g_ed.sel], W, H, b);
    float step = atomic_load(&g_set_grid) ? H / OVE_GRID_ROWS : 0;
    float px = ov_drag_pos(x + g_ed.gx, b[2], W, step), py = ov_drag_pos(y + g_ed.gy, b[3], H, step);
    cust_place(1, l, g_ed.sel, px / W, py / H); g_ed.dirty = 1;
}
static void ed_touch(AInputEvent *e, float W, float H) {   // the prompt's buttons, else the toolbar's, else the controls
    int act = AMotionEvent_getAction(e) & AMOTION_EVENT_ACTION_MASK, ai = AMotionEvent_getAction(e) >> AMOTION_EVENT_ACTION_POINTER_INDEX_SHIFT;
    int i = act == AMOTION_EVENT_ACTION_DOWN || act == AMOTION_EVENT_ACTION_UP ? 0 : ai, id = AMotionEvent_getPointerId(e, (size_t)i) & 31, l = dm_overlay_layout();
    float x = AMotionEvent_getX(e, (size_t)i), y = AMotionEvent_getY(e, (size_t)i), tb[4], pr[4]; ov_ed_rect(W, H, tb); ov_ed_prompt_rect(W, H, pr);
    int at = g_ed.prompt ? ov_ed_button_at(x, y, pr, OVE_DISCARD, 2) : ov_ed_button_at(x, y, tb, 0, OVE_N);   // a button under the finger
    int on_bar = !g_ed.prompt && x >= tb[0] && x < tb[0] + tb[2] && y >= tb[1] && y < tb[1] + tb[3];
    if (act == AMOTION_EVENT_ACTION_DOWN || act == AMOTION_EVENT_ACTION_POINTER_DOWN) {
        if (g_ed.prompt || on_bar) {   // a button: done when the finger lifts on it
            if (g_ed.ptr < 0) { g_ed.bdown = at; g_ed.ptr = id; g_ed.held = at; g_ed.bar = 1; ed_publish(); }   // (the chosen control stays chosen for - and +)
            return;
        }
        if (g_ed.ptr >= 0 && !g_ed.bar && g_ed.sel >= 0 && g_ed.ptr2 < 0) {   // a second finger: the pinch, from the two fingers' distance now
            for (size_t k = 0; k < AMotionEvent_getPointerCount(e); k++) if ((AMotionEvent_getPointerId(e, k) & 31) == g_ed.ptr)
                g_ed.d0 = hypotf(x - AMotionEvent_getX(e, k), y - AMotionEvent_getY(e, k));
            g_ed.ptr2 = id; g_ed.s0 = ed_size(l, g_ed.sel); return;
        }
        if (g_ed.ptr >= 0) return;
        int c = ctl_at(x, y, W, H); g_ed.sel = c; g_ed.ptr = c >= 0 ? id : -1; g_ed.bdown = -1; g_ed.bar = 0;   // a control is chosen (nothing: none is)
        if (c >= 0) { float b[4]; ov_box(&OV_LAYOUTS[l][c], W, H, b); g_ed.gx = b[0] - x; g_ed.gy = b[1] - y; dm_haptics_android_tap(); }
        ed_publish();
    } else if (act == AMOTION_EVENT_ACTION_MOVE) {
        float px[2] = {0, 0}, py[2] = {0, 0}; int got = 0;
        for (size_t k = 0; k < AMotionEvent_getPointerCount(e); k++) {
            int pid = AMotionEvent_getPointerId(e, k) & 31, w = pid == g_ed.ptr ? 0 : pid == g_ed.ptr2 ? 1 : -1; if (w < 0) continue;
            px[w] = AMotionEvent_getX(e, k); py[w] = AMotionEvent_getY(e, k); got |= 1 << w;
        }
        if (g_ed.ptr >= 0 && g_ed.bar) {   // on a button: it stays lit while the finger is on it
            if (got & 1) { int a = g_ed.prompt ? ov_ed_button_at(px[0], py[0], pr, OVE_DISCARD, 2) : ov_ed_button_at(px[0], py[0], tb, 0, OVE_N), h = a == g_ed.bdown ? a : -1;
                if (h != g_ed.held) { g_ed.held = h; ed_publish(); } }
            return;
        }
        if (g_ed.sel < 0 || g_ed.ptr < 0) return;
        if (g_ed.ptr2 >= 0 && got == 3 && g_ed.d0 > 1) ed_resize(l, g_ed.sel, g_ed.s0 * hypotf(px[0] - px[1], py[0] - py[1]) / g_ed.d0);
        else if (got & 1) ed_drag(px[0], py[0], W, H);
        else return;
        ed_publish();
    } else if (act == AMOTION_EVENT_ACTION_UP || act == AMOTION_EVENT_ACTION_POINTER_UP || act == AMOTION_EVENT_ACTION_CANCEL) {
        if (id == g_ed.ptr2 && act != AMOTION_EVENT_ACTION_CANCEL) { g_ed.ptr2 = -1; atomic_fetch_add(&g_cust_ver, 1); return; }
        if (id != g_ed.ptr && act != AMOTION_EVENT_ACTION_CANCEL) return;
        int b = g_ed.bar ? g_ed.bdown : -1; if (!g_ed.bar) atomic_fetch_add(&g_cust_ver, 1);
        g_ed.ptr = g_ed.ptr2 = g_ed.bdown = g_ed.held = -1; g_ed.bar = 0;
        if (b >= 0 && b == at && act != AMOTION_EVENT_ACTION_CANCEL) ed_do(b); else ed_publish();
    }
}

// ---- text entry: the game's software keyboard (nn::swkbd, hle_swkbd.c → front_text_input), as an on-screen keyboard of the overlay ----
// The game thread that asks waits here while the player types; the game is held as by the menu and the last frame is shown with the entry's panel and keys
// (overlay_layout.h ov_kb_*, drawn by gfx_vk_present.c ov_text_panel from dm_text_entry). Touch: tap the keys. Controller: the d-pad moves the highlight,
// A presses it, B erases, + accepts, − cancels. A physical keyboard types (letters, digits, space, punctuation as ASCII), Del erases,
// Enter accepts. Back cancels (the game keeps its initial text). Android's own keyboard is not used: a code-less NativeActivity's content view is no text
// editor, so the system does not show one for it, and asking for one routed the key events to it
static pthread_mutex_t g_tx_mu = PTHREAD_MUTEX_INITIALIZER; static pthread_cond_t g_tx_cv = PTHREAD_COND_INITIALIZER;
static struct { char title[128], text[256]; int n, cap, max, done, ok, sel, shift, ptr; } g_tx;   // n, cap: bytes of UTF-8; max: UTF-16 units, the game's limit
int dm_text_entry(char *title, int tcap, char *text, int xcap, int *sel, int *shift) {   // the panel's content (the present thread); 0 when no entry is open
    if (!atomic_load(&g_tx_on)) return 0;
    pthread_mutex_lock(&g_tx_mu);
    snprintf(title, (size_t)tcap, "%s", g_tx.title); snprintf(text, (size_t)xcap, "%s", g_tx.text); *sel = g_tx.sel; *shift = g_tx.shift;
    pthread_mutex_unlock(&g_tx_mu);
    return 1;
}
static void tx_changed(void) { atomic_fetch_add(&g_tx_ver, 1); dm_overlay_redraw(); }
static void tx_add(int c) {   // the caller holds g_tx_mu; a capital ends the shift, as on a phone
    if (!c || g_tx.n >= g_tx.cap || pf_units(g_tx.text) >= g_tx.max) return;
    g_tx.text[g_tx.n++] = (char)c; g_tx.text[g_tx.n] = 0;
    if (c >= 'A' && c <= 'Z') g_tx.shift = 0;
}
static void tx_finish(int ok);
static void tx_press(int i) {   // an on-screen key
    if (i == OVK_OK) { tx_finish(1); return; }
    pthread_mutex_lock(&g_tx_mu);
    if (i == OVK_SHIFT) g_tx.shift = !g_tx.shift;
    else if (i == OVK_SPACE) tx_add(' ');
    else if (i == OVK_DEL) { g_tx.n = (int)pf_chop(g_tx.text, (size_t)g_tx.n); }
    else tx_add(ov_kb_char(i, g_tx.shift));
    pthread_mutex_unlock(&g_tx_mu);
    dm_haptics_android_tap(); tx_changed();
}
static int tx_char(int32_t k, int32_t meta) {   // the ASCII character of a key, 0 for none
    int sh = (meta & AMETA_SHIFT_ON) != 0;
    if (k >= AKEYCODE_A && k <= AKEYCODE_Z) return (sh ^ ((meta & AMETA_CAPS_LOCK_ON) != 0) ? 'A' : 'a') + (k - AKEYCODE_A);
    if (k >= AKEYCODE_0 && k <= AKEYCODE_9) return sh ? ")!@#$%^&*("[k - AKEYCODE_0] : '0' + (k - AKEYCODE_0);
    switch (k) {
    case AKEYCODE_SPACE: return ' ';
    case AKEYCODE_COMMA: return sh ? '<' : ','; case AKEYCODE_PERIOD: return sh ? '>' : '.'; case AKEYCODE_MINUS: return sh ? '_' : '-';
    case AKEYCODE_EQUALS: return sh ? '+' : '='; case AKEYCODE_SLASH: return sh ? '?' : '/'; case AKEYCODE_APOSTROPHE: return sh ? '"' : '\'';
    case AKEYCODE_SEMICOLON: return sh ? ':' : ';'; case AKEYCODE_AT: return '@'; case AKEYCODE_PLUS: return '+'; case AKEYCODE_STAR: return '*'; case AKEYCODE_POUND: return '#';
    case AKEYCODE_LEFT_BRACKET: return sh ? '{' : '['; case AKEYCODE_RIGHT_BRACKET: return sh ? '}' : ']';
    default: return 0;
    }
}
static void tx_finish(int ok) {
    pthread_mutex_lock(&g_tx_mu); g_tx.ok = ok; g_tx.done = 1; pthread_cond_broadcast(&g_tx_cv); pthread_mutex_unlock(&g_tx_mu);
}
static int tx_key(AInputEvent *e) {   // every key goes to the entry while it is open
    int32_t k = AKeyEvent_getKeyCode(e);
    if (AKeyEvent_getAction(e) != AKEY_EVENT_ACTION_DOWN) return 1;
    uint64_t bit = key_bit(k);
    if (k == AKEYCODE_ENTER || k == AKEYCODE_NUMPAD_ENTER || bit == DM_BTN_PLUS) { tx_finish(1); return 1; }
    if (k == AKEYCODE_BACK || k == AKEYCODE_ESCAPE || bit == DM_BTN_MINUS) { tx_finish(0); return 1; }
    if (bit == DM_BTN_A) { int s; pthread_mutex_lock(&g_tx_mu); s = g_tx.sel; pthread_mutex_unlock(&g_tx_mu); tx_press(s); return 1; }
    pthread_mutex_lock(&g_tx_mu);
    int dx = bit == DM_BTN_LEFT ? -1 : bit == DM_BTN_RIGHT ? 1 : 0, dy = bit == DM_BTN_UP ? -1 : bit == DM_BTN_DOWN ? 1 : 0;
    if (dx || dy) g_tx.sel = ov_kb_move(g_tx.sel, dx, dy);
    else if (k == AKEYCODE_DEL || bit == DM_BTN_B) { g_tx.n = (int)pf_chop(g_tx.text, (size_t)g_tx.n); }
    else if (bit == DM_BTN_Y) g_tx.shift = !g_tx.shift;
    else tx_add(tx_char(k, AKeyEvent_getMetaState(e)));
    pthread_mutex_unlock(&g_tx_mu);
    tx_changed();
    return 1;
}
static void tx_touch(AInputEvent *e, float W, float H) {   // a key is pressed when the finger that went down on it lifts there
    int act = AMotionEvent_getAction(e) & AMOTION_EVENT_ACTION_MASK, ai = AMotionEvent_getAction(e) >> AMOTION_EVENT_ACTION_POINTER_INDEX_SHIFT;
    int i = act == AMOTION_EVENT_ACTION_DOWN || act == AMOTION_EVENT_ACTION_UP ? 0 : ai;
    float p[4]; ov_menu_rect(W, H, p);
    int key = ov_kb_at(AMotionEvent_getX(e, (size_t)i), AMotionEvent_getY(e, (size_t)i), p);
    if (act == AMOTION_EVENT_ACTION_DOWN || act == AMOTION_EVENT_ACTION_POINTER_DOWN) {
        pthread_mutex_lock(&g_tx_mu); g_tx.ptr = key; if (key >= 0) g_tx.sel = key; pthread_mutex_unlock(&g_tx_mu);
        if (key >= 0) tx_changed();
    } else if (act == AMOTION_EVENT_ACTION_UP || act == AMOTION_EVENT_ACTION_POINTER_UP) {
        int down; pthread_mutex_lock(&g_tx_mu); down = g_tx.ptr; g_tx.ptr = -1; pthread_mutex_unlock(&g_tx_mu);
        if (key >= 0 && key == down) tx_press(key);
    }
}
#include "keyboard_android.inc"
static pthread_mutex_t g_keyboard_mu = PTHREAD_MUTEX_INITIALIZER;
static int overlay_text_input(const char *title, const char *sub, const char *guide, const char *initial, int max_chars, char *out, int out_cap) {
    if (!out || out_cap <= 0) return -1;
    if (!g_act || !dm_front_window()) return -1;   // the adb shell executable, or no window yet: nothing is shown
    pthread_mutex_lock(&g_tx_mu);
    const char *t = title && *title ? title : guide && *guide ? guide : sub && *sub ? sub : "Enter text";
    snprintf(g_tx.title, sizeof g_tx.title, "%s", t); pf_trim(g_tx.title);
    g_tx.cap = out_cap - 1 < (int)sizeof g_tx.text - 1 ? out_cap - 1 : (int)sizeof g_tx.text - 1;
    g_tx.n = 0; g_tx.text[0] = 0;
    for (const char *s = initial ? initial : ""; *s;) {   // the initial text without its control characters, whole characters only
        const char *c0 = s; uint32_t c = pf_next(&s); int k = (int)(s - c0);
        if (c < 32 || c == 127) continue;
        if (g_tx.n + k > g_tx.cap) break;
        memcpy(g_tx.text + g_tx.n, c0, (size_t)k); g_tx.n += k;
    }
    g_tx.text[g_tx.n] = 0;
    g_tx.max = max_chars > 0 && max_chars < (int)sizeof g_tx.text - 1 ? max_chars : (int)sizeof g_tx.text - 1;
    g_tx.done = g_tx.ok = 0; g_tx.sel = 11; g_tx.shift = g_tx.n == 0; g_tx.ptr = -1;   // the highlight on q; a capital first, as on a phone
    pthread_mutex_unlock(&g_tx_mu);
    dm_input_set(0, 0, 0, 0, 0); dm_touch_set(NULL, 0); atomic_store(&g_touch_btn, 0);   // nothing stays pressed in the game
    atomic_store(&g_tx_on, 1); hold_sync(); tx_changed();
    LOG("[text] entry open: \"%s\" (at most %d characters)", g_tx.title, g_tx.max);
    pthread_mutex_lock(&g_tx_mu);
    while (!g_tx.done) pthread_cond_wait(&g_tx_cv, &g_tx_mu);
    int ok = g_tx.ok; if (ok) snprintf(out, (size_t)out_cap, "%s", g_tx.text);
    pthread_mutex_unlock(&g_tx_mu);
    atomic_store(&g_tx_on, 0); hold_sync(); dm_overlay_redraw();
    LOG("[text] entry %s", ok ? "accepted" : "cancelled");
    return ok;
}

int front_text_input(const char *title, const char *sub, const char *guide, const char *initial, int max_chars, char *out, int out_cap) {
    if (pthread_mutex_trylock(&g_keyboard_mu)) { LOG("[text] request cancelled: another keyboard request is active"); return 0; }
    int result = android_text_input(title, sub, guide, initial, max_chars, out, out_cap);
    if (result == -2) result = overlay_text_input(title, sub, guide, initial, max_chars, out, out_cap);
    pthread_mutex_unlock(&g_keyboard_mu);
    return result;
}

__attribute__((weak)) void dm_picture_rect(float r[4]);   // gfx_vk_present.c
static void touch_event(AInputEvent *e) {
    ANativeWindow *w = phone_window(); float W = w ? (float)ANativeWindow_getWidth(w) : 1, H = w ? (float)ANativeWindow_getHeight(w) : 1;
    if (atomic_load(&g_tx_on)) { tx_touch(e, W, H); return; }   // the text entry's keys
    if (atomic_load(&g_menu)) { if (atomic_load(&g_edit)) ed_touch(e, W, H); else if (atomic_load(&g_sp_on)) sp_touch(e, W, H); else menu_touch(e, W, H); return; }
    int act = AMotionEvent_getAction(e) & AMOTION_EVENT_ACTION_MASK, ai = AMotionEvent_getAction(e) >> AMOTION_EVENT_ACTION_POINTER_INDEX_SHIFT;
    size_t pc = AMotionEvent_getPointerCount(e);
    if (act == AMOTION_EVENT_ACTION_DOWN || act == AMOTION_EVENT_ACTION_POINTER_DOWN) {
        int i = act == AMOTION_EVENT_ACTION_DOWN ? 0 : ai, id = AMotionEvent_getPointerId(e, (size_t)i) & 31;
        int c = ctl_at(AMotionEvent_getX(e, (size_t)i), AMotionEvent_getY(e, (size_t)i), W, H);
        g_ptr_ctl[id] = c + 1;
        if (c >= 0 && OV_CTLS[c].kind == OV_MENU) { atomic_store(&g_menu_t, mono_ns()); dm_haptics_android_tap(); menu_grab(id, &OV_CTLS[c], AMotionEvent_getX(e, (size_t)i), AMotionEvent_getY(e, (size_t)i), W, H); }
        if (!atomic_exchange(&g_overlay, 1) && !atomic_load(&g_pads)) dm_haptics_android_device(-1);   // keep vibration on a used controller that is still attached
    }
    uint64_t btn = 0; int32_t lx = 0, ly = 0, rx = 0, ry = 0; float kn[2][2] = {{0, 0}, {0, 0}}; int open = 0, held[2] = {0, 0};
    dm_touch t[16]; int n = 0;
    float pic[4] = {0, 0, 1, 1}; if (dm_picture_rect) dm_picture_rect(pic); if (pic[2] <= 0 || pic[3] <= 0) pic[2] = pic[3] = 1;   // gfx_vk_present.c: the picture inside the window
    for (size_t i = 0; i < pc; i++) {
        int id = AMotionEvent_getPointerId(e, i) & 31;
        int lifted = act == AMOTION_EVENT_ACTION_UP || act == AMOTION_EVENT_ACTION_CANCEL || (act == AMOTION_EVENT_ACTION_POINTER_UP && (int)i == ai);
        float x = AMotionEvent_getX(e, i), y = AMotionEvent_getY(e, i);
        int c = g_ptr_ctl[id] - 1;
        if (lifted) {
            if (c >= 0 && OV_CTLS[c].kind == OV_MENU) {   // a tap opens the menu; a drag leaves the button where it is
                atomic_store(&g_menu_t, 0);
                if (id == g_drag.id && g_drag.moved) { settings_save(); float mx = 0, my = 0, ms; dm_overlay_custom(dm_overlay_layout(), c, &mx, &my, &ms); LOG("[overlay] menu button moved to %.3f, %.3f", mx, my); }
                else open = act != AMOTION_EVENT_ACTION_CANCEL;
                if (id == g_drag.id) g_drag.id = -1;
            }
            g_ptr_ctl[id] = 0; continue;
        }
        if (c < 0) { if (n < 16) { t[n].id = (uint32_t)id & 15; t[n].x = (x / W - pic[0]) / pic[2]; t[n].y = (y / H - pic[1]) / pic[3]; n++; } continue; }   // relative to the game's picture
        const ov_ctl *k = &OV_CTLS[c];
        if (k->kind == OV_MENU) { if (id == g_drag.id) menu_drag(k, x, y, W, H); continue; }
        float b[4]; ov_box(k, W, H, b);
        float dx = (x - b[0]) / b[2], dy = (y - b[1]) / b[2], m = sqrtf(dx * dx + dy * dy);   // in the control's half widths (a stick's or the d-pad's radius)
        if (k->kind == OV_BUTTON) btn |= ov_button_bits(OV_CTLS, k, x, y, W, H);
        else if (k->kind == OV_DPAD) {
            if (m > 0.25f) { float a = atan2f(dy, dx); int s = (int)lroundf(a / (3.14159265f / 4)) & 7;   // 8 directions, 0 = right, clockwise (y grows downward)
                static const uint64_t dir[8] = {DM_BTN_RIGHT, DM_BTN_RIGHT | DM_BTN_DOWN, DM_BTN_DOWN, DM_BTN_DOWN | DM_BTN_LEFT, DM_BTN_LEFT, DM_BTN_LEFT | DM_BTN_UP, DM_BTN_UP, DM_BTN_UP | DM_BTN_RIGHT};
                btn |= dir[s]; }
        } else {
            if (m > 1) { dx /= m; dy /= m; }
            kn[k->kind == OV_STICK_R][0] = dx; kn[k->kind == OV_STICK_R][1] = dy; held[k->kind == OV_STICK_R] = 1;   // the knob follows the finger (to the ring)
            int32_t sx = (int32_t)(dx * DM_STICK_MAX), sy = (int32_t)(-dy * DM_STICK_MAX);
            if (k->kind == OV_STICK_L) { lx = sx; ly = sy; } else { rx = sx; ry = sy; }
        }
    }
    uint64_t previous = atomic_exchange(&g_touch_btn, btn);
    uint64_t down = btn & ~previous, up = previous & ~btn; int64_t now = mono_ns();
    for (int b = 0; b < 16; b++) { if (down >> b & 1) atomic_store(&g_press_t[b], now); else if (up >> b & 1) atomic_store(&g_press_t[b], 0); }
    if (down) dm_haptics_android_tap();   // a tick for each button (or d-pad direction) the finger reaches
    for (int s = 0; s < 2; s++) for (int a = 0; a < 2; a++) atomic_store(&g_knob[s][a], (int)(kn[s][a] * 10000));
    for (int s = 0; s < 2; s++) if (!held[s]) atomic_store(&g_stick_t[s], 0); else if (!atomic_load(&g_stick_t[s])) atomic_store(&g_stick_t[s], now);
    dm_input_set(btn, lx, ly, rx, ry);
    dm_touch_set(t, n);
    if (open) menu_open(-1);
}
static void pad_mark_used(int32_t device) {
    atomic_store(&g_overlay, 0);
    g_pad_used = 1;
    atomic_store(&g_pad_in, device);
    dm_haptics_android_device(device);
}
// ---- a controller's directions in the menu and the text entry: the d-pad (keys or a hat) and the left stick (past NAV_DEAD) move once when pressed, then
// again after NAV_DELAY and every NAV_EVERY while held; a held left or right on a slider speeds up after NAV_FAST_AFTER. Our own repeat (Android's key
// repeats are ignored), timed by the input thread's poll (nav_timeout_ms, nav_tick). Input thread only ----
#define NAV_DEAD 0.55f
#define NAV_DELAY_NS 350000000ll
#define NAV_EVERY_NS 120000000ll
#define NAV_FAST_NS 50000000ll
#define NAV_FAST_AFTER_NS 1000000000ll
#define NAV_DIRS (DM_BTN_UP | DM_BTN_DOWN | DM_BTN_LEFT | DM_BTN_RIGHT)
static struct { uint64_t keys, hat, stick, bit; int64_t start, next; } g_nav;
static void nav_fire(uint64_t bit) {
    if (atomic_load(&g_tx_on)) {
        int dx = bit == DM_BTN_LEFT ? -1 : bit == DM_BTN_RIGHT ? 1 : 0, dy = bit == DM_BTN_UP ? -1 : bit == DM_BTN_DOWN ? 1 : 0;
        pthread_mutex_lock(&g_tx_mu); g_tx.sel = ov_kb_move(g_tx.sel, dx, dy); pthread_mutex_unlock(&g_tx_mu); tx_changed();
    } else if (atomic_load(&g_menu)) menu_key(bit);
}
static uint64_t nav_dir(uint64_t held) {   // the one direction held: the newest source's, up/down before left/right
    for (int i = 0; i < 4; i++) { static const uint64_t o[4] = {DM_BTN_UP, DM_BTN_DOWN, DM_BTN_LEFT, DM_BTN_RIGHT}; if (held & o[i]) return o[i]; }
    return 0;
}
static void nav_update(void) {   // after a source changed: a new direction moves now
    uint64_t want = nav_dir(g_nav.keys | g_nav.hat | g_nav.stick);
    if (want == g_nav.bit) return;
    g_nav.bit = want; if (!want) return;
    g_nav.start = mono_ns(); g_nav.next = g_nav.start + NAV_DELAY_NS; nav_fire(want);
}
static void nav_tick(void) {
    if (!atomic_load(&g_menu) && !atomic_load(&g_tx_on)) { if (g_nav.bit | g_nav.keys | g_nav.hat | g_nav.stick) memset(&g_nav, 0, sizeof g_nav); return; }
    if (!g_nav.bit) return;
    int64_t now = mono_ns(); if (now < g_nav.next) return;
    int lr = g_nav.bit == DM_BTN_LEFT || g_nav.bit == DM_BTN_RIGHT, slider = lr && !atomic_load(&g_tx_on) && !atomic_load(&g_sp_on) && !atomic_load(&g_edit) &&
        ov_menu_kind(ov_menu_item(atomic_load(&g_menu_tab), atomic_load(&g_menu_hi))) == OVR_SLIDER;
    g_nav.next = now + (slider && now - g_nav.start > NAV_FAST_AFTER_NS ? NAV_FAST_NS : NAV_EVERY_NS);
    nav_fire(g_nav.bit);
}
static int nav_timeout_ms(void) { if (!g_nav.bit) return -1; int64_t d = g_nav.next - mono_ns(); return d <= 0 ? 0 : (int)(d / 1000000) + 1; }
static void nav_axes(AInputEvent *e) {   // a motion event's hat and left stick
    float hx = AMotionEvent_getAxisValue(e, AMOTION_EVENT_AXIS_HAT_X, 0), hy = AMotionEvent_getAxisValue(e, AMOTION_EVENT_AXIS_HAT_Y, 0);
    float sx = AMotionEvent_getAxisValue(e, AMOTION_EVENT_AXIS_X, 0), sy = AMotionEvent_getAxisValue(e, AMOTION_EVENT_AXIS_Y, 0);
    g_nav.hat = (hy < -0.5f ? DM_BTN_UP : hy > 0.5f ? DM_BTN_DOWN : 0) | (hx < -0.5f ? DM_BTN_LEFT : hx > 0.5f ? DM_BTN_RIGHT : 0);
    uint64_t st = 0;   // the stick: its stronger axis only (a diagonal does not move two ways)
    if (fabsf(sy) >= fabsf(sx)) st = sy < -NAV_DEAD ? DM_BTN_UP : sy > NAV_DEAD ? DM_BTN_DOWN : 0;
    else st = sx < -NAV_DEAD ? DM_BTN_LEFT : sx > NAV_DEAD ? DM_BTN_RIGHT : 0;
    g_nav.stick = st; nav_update();
}
static int handle_event(AInputEvent *e) {
    int32_t src = AInputEvent_getSource(e), type = AInputEvent_getType(e);
    if (type == AINPUT_EVENT_TYPE_KEY) {
        { uint64_t d = key_bit(AKeyEvent_getKeyCode(e)) & NAV_DIRS; int32_t a = AKeyEvent_getAction(e);   // a direction in the menu or the text entry: the repeater's
          if (d && (atomic_load(&g_tx_on) || atomic_load(&g_menu)) && !atomic_load(&g_edit)) {
              if (a == AKEY_EVENT_ACTION_DOWN && AKeyEvent_getRepeatCount(e) == 0) g_nav.keys |= d; else if (a == AKEY_EVENT_ACTION_UP) g_nav.keys &= ~d;
              nav_update(); return 1; } }
        if (atomic_load(&g_tx_on)) return tx_key(e);   // the text entry takes every key while it is open
        int32_t k = AKeyEvent_getKeyCode(e), down = AKeyEvent_getAction(e) == AKEY_EVENT_ACTION_DOWN;
        if (k == AKEYCODE_BACK || k == AKEYCODE_BUTTON_MODE || k == AKEYCODE_MENU) {   // the menu, both ways (Back would otherwise finish the activity, and the game with it)
            if (down && AKeyEvent_getRepeatCount(e) == 0) { if (atomic_load(&g_sp_on)) sp_close_all(); else if (atomic_load(&g_menu)) menu_close(); else menu_open(k == AKEYCODE_BACK ? -1 : 0); }
            return 1;
        }
        uint64_t bit = key_bit(k);
        if (!bit) {   // the rest: the system's. A controller's key nothing maps is logged once, so a new controller's menu key can be found
            static uint8_t seen[64];
            if ((src & AINPUT_SOURCE_GAMEPAD) == AINPUT_SOURCE_GAMEPAD && down && k >= 0 && k < 512 && !(seen[k >> 3] >> (k & 7) & 1)) {
                seen[k >> 3] |= (uint8_t)(1 << (k & 7)); LOG("[input] controller key %d (device %d): not mapped", k, AInputEvent_getDeviceId(e)); }
            return 0;
        }
        pad_mark_used(AInputEvent_getDeviceId(e));
        static uint64_t swallow;   // keys the menu took: the game does not get them until they are up again (+ or B closing the menu: its repeats, its release)
        if (atomic_load(&g_menu)) { if (down) swallow |= bit; if (down && AKeyEvent_getRepeatCount(e) == 0) menu_key(bit); return 1; }
        if (swallow & bit) { if (AKeyEvent_getAction(e) == AKEY_EVENT_ACTION_UP) swallow &= ~bit; return 1; }
        if (down) g_btn |= bit; else if (AKeyEvent_getAction(e) == AKEY_EVENT_ACTION_UP) g_btn &= ~bit;
        dm_pad_players(1); pad_publish(); return 1;
    }
    if (atomic_load(&g_tx_on) && ((src & AINPUT_SOURCE_JOYSTICK) == AINPUT_SOURCE_JOYSTICK || (src & AINPUT_SOURCE_GAMEPAD) == AINPUT_SOURCE_GAMEPAD)) {
        // the d-pad (reported as a hat on Retroid and many Android handhelds) or the left stick moves the text-entry highlight, held ones repeating (nav_*); touch taps the keys (tx_touch)
        nav_axes(e); return 1;
    }
    if ((src & AINPUT_SOURCE_JOYSTICK) == AINPUT_SOURCE_JOYSTICK || (src & AINPUT_SOURCE_GAMEPAD) == AINPUT_SOURCE_GAMEPAD) {
        float axes[8] = {
            AMotionEvent_getAxisValue(e, AMOTION_EVENT_AXIS_X, 0), AMotionEvent_getAxisValue(e, AMOTION_EVENT_AXIS_Y, 0),
            AMotionEvent_getAxisValue(e, AMOTION_EVENT_AXIS_Z, 0), AMotionEvent_getAxisValue(e, AMOTION_EVENT_AXIS_RZ, 0),
            AMotionEvent_getAxisValue(e, AMOTION_EVENT_AXIS_LTRIGGER, 0), AMotionEvent_getAxisValue(e, AMOTION_EVENT_AXIS_RTRIGGER, 0),
            AMotionEvent_getAxisValue(e, AMOTION_EVENT_AXIS_HAT_X, 0), AMotionEvent_getAxisValue(e, AMOTION_EVENT_AXIS_HAT_Y, 0),
        };
        if (axes[4] == 0) axes[4] = AMotionEvent_getAxisValue(e, AMOTION_EVENT_AXIS_BRAKE, 0);
        if (axes[5] == 0) axes[5] = AMotionEvent_getAxisValue(e, AMOTION_EVENT_AXIS_GAS, 0);
        // Idle reports still release game inputs, but only a mapped axis outside its dead zone takes over from touch.
        if (axis(axes[0]) || axis(axes[1]) || axis(axes[2]) || axis(axes[3]) || pad_axis_buttons(axes))
            pad_mark_used(AInputEvent_getDeviceId(e));
        if (atomic_load(&g_menu)) {   // the hat and the left stick move the selection (held: repeating); nothing reaches the game
            if (atomic_load(&g_edit)) {   // the layout editor: the hat's edges, as before
                static float hx, hy; float x = AMotionEvent_getAxisValue(e, AMOTION_EVENT_AXIS_HAT_X, 0), y = AMotionEvent_getAxisValue(e, AMOTION_EVENT_AXIS_HAT_Y, 0);
                if (y < -0.5f && hy >= -0.5f) menu_key(DM_BTN_UP); else if (y > 0.5f && hy <= 0.5f) menu_key(DM_BTN_DOWN);
                if (x < -0.5f && hx >= -0.5f) menu_key(DM_BTN_LEFT); else if (x > 0.5f && hx <= 0.5f) menu_key(DM_BTN_RIGHT);
                hx = x; hy = y; return 1;
            }
            nav_axes(e); return 1;
        }
        memcpy(g_ax, axes, sizeof g_ax);
        dm_pad_players(1); pad_publish(); return 1;
    }
    if ((src & AINPUT_SOURCE_TOUCHSCREEN) == AINPUT_SOURCE_TOUCHSCREEN) { touch_event(e); return 1; }
    return 0;
}
static int on_input(int fd, int events, void *data) {
    (void)fd; (void)events; (void)data;
    AInputEvent *e;
    while (g_queue && AInputQueue_getEvent(g_queue, &e) >= 0) {
        // no preDispatch: the app shows no soft keyboard (the text entry draws its own), and the IME's pre-dispatch swallowed a keyboard's letter keys
        AInputQueue_finishEvent(g_queue, e, handle_event(e));
    }
    return 1;
}
static void ext_touch_drain(void) {
    pthread_mutex_lock(&g_ext_touch_mu);
    ext_touch *item = g_ext_touch_head; g_ext_touch_head = g_ext_touch_tail = NULL;
    pthread_mutex_unlock(&g_ext_touch_mu);
    while (item) { ext_touch *next = item->next; if (dm_external_active() && item->epoch == atomic_load(&g_external_epoch)) handle_event((AInputEvent *)item->event); g_input_release(item->event); free(item); item = next; }
}
// ---- shaking the device: the controller is shaken (DM_BTN_SHAKE: its six-axis sensor, hle_hid.c; Super Mario Odyssey's cap throw) ----
// The accelerometer at 60 Hz on the input thread's looper (no permission needed): gravity is followed by a slow low-pass (about 0.8 s), and a sample whose acceleration beyond it
// exceeds DM_SHAKE_G (1.5 G by default; env.txt overrides) is a hit. The controller is shaken until 120 ms after the last hit: a flick is one shake (the host finishes it as a whole
// cycle) and shaking on keeps it going. It goes through player 1's controller path, so it adds to the on-screen gamepad and a controller alike. Off while the app is in the background
// ---- the device's motion: player 1's six-axis sensor turns as the device does (dm_motion_set, hle_hid.c), as the Joy-Cons turn with the console in handheld mode (aiming
// by motion: The Legend of Zelda: Tears of the Kingdom's bow). The gyroscope at 200 Hz (the Switch sensor's rate) with the latest accelerometer sample, in the screen's frame
// (x right, y to the top edge, z out of the screen: Android gives them in the device's natural orientation, so they turn with the display's rotation, read through JNI every
// 250 ms because a 180-degree flip in sensorLandscape sends no configuration change), angular velocity in revolutions / s and gravity's direction in G (the opposite sign of
// Android's). The same mapping as yuzu's Android front end (EmulationActivity: rotation 270 = (y, -x, z), 90 = (-y, x, z); yuzu divides by 6 for 2 pi). DM_MOTION=0 turns it off
// (the sensor at rest, as before); DM_MOTION_LOG=1 logs the mapped values once a second. Off in the background with the accelerometer
static ASensorEventQueue *g_sq; static const ASensor *g_accel, *g_gyro;
static void shake_sensor(int on) {
    if (!g_sq || !g_accel) return;
    // the rate goes with the registration: enableSensor + setEventRate left it at the default 200 ms on the Y700 (dumpsys sensorservice), too slow for a flick.
    // 200 Hz with the gyroscope (its samples carry the latest acceleration), 60 Hz for shaking alone
    if (on) ASensorEventQueue_registerSensor(g_sq, g_accel, g_gyro ? 5000 : 16667, 0); else ASensorEventQueue_disableSensor(g_sq, g_accel);
    if (g_gyro) { if (on) ASensorEventQueue_registerSensor(g_sq, g_gyro, 5000, 0); else ASensorEventQueue_disableSensor(g_sq, g_gyro); }
}
static JNIEnv *g_ienv; static jobject g_display; static jmethodID g_mRotation;   // the input thread's JNI: Display.getRotation
static int g_rot;   // the display's rotation, Surface.ROTATION_0..270 = 0..3
static void rotation_start(void) {
    JavaVM *vm = g_act ? g_act->vm : NULL; JNIEnv *E;
    if (!vm || (*vm)->AttachCurrentThread(vm, &E, NULL) != JNI_OK) { LOG("[RED] front end: the input thread cannot attach to the Java VM: motion keeps rotation 0"); return; }
    jclass cA = (*E)->GetObjectClass(E, g_act->clazz); jmethodID mWm = (*E)->GetMethodID(E, cA, "getWindowManager", "()Landroid/view/WindowManager;");
    jobject wm = mWm ? (*E)->CallObjectMethod(E, g_act->clazz, mWm) : NULL; if (jx(E) || !wm) { LOG("[RED] front end: no WindowManager: motion keeps rotation 0"); return; }
    jclass cWm = (*E)->FindClass(E, "android/view/WindowManager"), cD = (*E)->FindClass(E, "android/view/Display");
    jmethodID mDisp = cWm ? (*E)->GetMethodID(E, cWm, "getDefaultDisplay", "()Landroid/view/Display;") : NULL; if (jx(E) || !mDisp || !cD) return;
    jobject d = (*E)->CallObjectMethod(E, wm, mDisp); g_mRotation = (*E)->GetMethodID(E, cD, "getRotation", "()I"); if (jx(E) || !d || !g_mRotation) return;
    g_display = (*E)->NewGlobalRef(E, d); g_ienv = E;
}
static void rotation_poll(int64_t now) {
    static int64_t next; static int first = 1;
    if (!g_display || now < next) return;
    next = now + 250000000;
    int r = (*g_ienv)->CallIntMethod(g_ienv, g_display, g_mRotation); if (jx(g_ienv)) return;
    if ((r & 3) != g_rot || first) LOG("front end: display rotation %d degrees: motion axes follow it", (r & 3) * 90);
    g_rot = r & 3; first = 0;
}
static void motion_feed(const ASensorEvent *e, const double acc[3]) {   // a gyroscope sample with the latest acceleration (device frame, m/s^2) → the host
    static int log = -1; static int64_t logged;
    if (log < 0) { const char *v = getenv("DM_MOTION_LOG"); log = v && atoi(v) > 0; }
    rotation_poll(e->timestamp);
    double g[3] = {e->vector.x, e->vector.y, e->vector.z}, a[3] = {acc[0], acc[1], acc[2]};
    // device (natural orientation) → screen: 90 = (-y, x), 180 = (-x, -y), 270 = (y, -x); z stays
    for (double *v = g; v; v = v == g ? a : NULL) { double x = v[0], y = v[1];
        switch (g_rot) { case 1: v[0] = -y; v[1] = x; break; case 2: v[0] = -x; v[1] = -y; break; case 3: v[0] = y; v[1] = -x; break; default: break; } }
    float fa[3], fg[3];
    for (int i = 0; i < 3; i++) { fg[i] = (float)(g[i] / (2 * M_PI)); fa[i] = (float)(-a[i] / ASENSOR_STANDARD_GRAVITY); }
    dm_motion_set_device(fa, fg, e->timestamp);
    if (log && e->timestamp - logged >= 1000000000) { logged = e->timestamp;
        LOG("front end: motion (rotation %d) acc %+.2f %+.2f %+.2f G, gyro %+.3f %+.3f %+.3f rev/s", g_rot * 90, fa[0], fa[1], fa[2], fg[0], fg[1], fg[2]); }
}
// Opt in only after checking the game's raw-motion gestures on the target device.
// The default keeps the verified automatic Shake path, including on gyro devices.
static int motion_raw(void) {
    static int on = -1;
    if (on < 0) { const char *e = getenv("DM_MOTION_RAW"); on = e && atoi(e) > 0; }
    return on;
}
static int on_sensor(int fd, int events, void *data) {
    (void)fd; (void)events; (void)data;
    static double thr, gx, gy, gz, peak, acc[3] = {0, 0, ASENSOR_STANDARD_GRAVITY}; static int64_t last = -1, prev_t; static int init;
    if (!thr) { const char *e = getenv("DM_SHAKE_G"); thr = e && atof(e) > 0 ? atof(e) : 1.5; }
    ASensorEvent ev[16]; ssize_t n;
    while ((n = ASensorEventQueue_getEvents(g_sq, ev, 16)) > 0) for (ssize_t i = 0; i < n; i++) {
        if (ev[i].type == ASENSOR_TYPE_GYROSCOPE) { motion_feed(&ev[i], acc); continue; }
        if (ev[i].type != ASENSOR_TYPE_ACCELEROMETER) continue;
        acc[0] = ev[i].acceleration.x; acc[1] = ev[i].acceleration.y; acc[2] = ev[i].acceleration.z;
        // Opt-in raw motion keeps a flick's measured direction instead of the fixed
        // DM_BTN_SHAKE waveform. Without a gyroscope the shake fallback stays active.
        if (g_gyro && motion_raw()) continue;
        double x = ev[i].acceleration.x / ASENSOR_STANDARD_GRAVITY, y = ev[i].acceleration.y / ASENSOR_STANDARD_GRAVITY, z = ev[i].acceleration.z / ASENSOR_STANDARD_GRAVITY;
        if (!init) { gx = x; gy = y; gz = z; init = 1; prev_t = ev[i].timestamp; }
        double dx = x - gx, dy = y - gy, dz = z - gz, g = sqrt(dx * dx + dy * dy + dz * dz);
        // gravity's low-pass: 0.02 a sample at 60 Hz, about 0.8 s, whatever the rate (200 Hz with the gyroscope)
        double dt = (double)(ev[i].timestamp - prev_t) / 1e9, k = 1 - exp(-(dt > 0 && dt < 0.5 ? dt : 1 / 60.0) / 0.825); prev_t = ev[i].timestamp;
        gx += (x - gx) * k; gy += (y - gy) * k; gz += (z - gz) * k;
        if (g > thr) { if (last < 0) peak = 0; last = ev[i].timestamp; peak = fmax(peak, g); }
        int on = last >= 0 && ev[i].timestamp - last < 120000000;
        if (!on && last >= 0) { LOG("front end: device shaken (peak %.1f G)", peak); last = -1; }
        uint64_t want = on ? DM_BTN_SHAKE : 0;
        if (want != g_shake_btn) { g_shake_btn = want; pad_publish(); }
    }
    return 1;
}
static void shake_start(void) {
    ASensorManager *m = ASensorManager_getInstanceForPackage(NULL);
    g_accel = m ? ASensorManager_getDefaultSensor(m, ASENSOR_TYPE_ACCELEROMETER) : NULL;
    if (!g_accel) { LOG("front end: no accelerometer, shaking the device does nothing"); return; }
    g_sq = ASensorManager_createEventQueue(m, g_ilooper, 2, on_sensor, NULL);
    const char *e = getenv("DM_SHAKE_G"), *mo = getenv("DM_MOTION");
    if (!(mo && !atoi(mo))) {
        g_gyro = ASensorManager_getDefaultSensor(m, ASENSOR_TYPE_GYROSCOPE);
        if (g_gyro) { rotation_start(); LOG("front end: the device's gyroscope (%s) moves the controller's six-axis sensor", ASensor_getName(g_gyro)); }
        else LOG("front end: no gyroscope: using accelerometer-only shake");
    } else LOG("front end: DM_MOTION=0: using accelerometer-only shake");
    if (g_gyro && motion_raw()) LOG("front end: DM_MOTION_RAW=1: raw device motion without automatic shake override");
    else LOG("front end: shaking the device shakes the controller (threshold %.2f G)", e && atof(e) > 0 ? atof(e) : 1.5);
    shake_sensor(!atomic_load(&g_bg));
}
static void *input_thread(void *a) {
    (void)a; g_ilooper = ALooper_prepare(0);
    shake_start();
    uint64_t prev = 0, latched = 0, sample = 0; int64_t since = 0;
    for (;;) {
        int nt = nav_timeout_ms(); ALooper_pollOnce(latched ? (nt >= 0 && nt < 10 ? nt : 10) : nt, NULL, NULL, NULL);
        nav_tick(); ext_touch_drain();
        if (!dm_external_active()) { prev = latched = 0; atomic_store(&g_ext_presses, 0); atomic_store(&g_ext_touch_enabled, 0); continue; }
        if (atomic_load(&g_ext_touch_enabled)) continue;
        int toggle = atomic_exchange(&g_ext_menu, 0);
        if (toggle & 1) { if (atomic_load(&g_menu)) menu_close(); else menu_open(0); }
        uint64_t b = atomic_load(&g_ext_buttons), presses = atomic_exchange(&g_ext_presses, 0);
        if (atomic_load(&g_menu)) { uint64_t down = (b & ~prev) | presses; latched = 0; for (int i = 0; i < 16; i++) if (down & (1ull << i)) menu_key(1ull << i); dm_input_set(0, 0, 0, 0, 0); }
        else {
            int64_t elapsed = mono_ns() - since;
            if (latched && ((elapsed >= 20000000 && dm_input_samples() != sample) || elapsed >= 100000000)) latched = 0;
            if (presses) { latched |= presses; sample = dm_input_samples(); since = mono_ns(); }
            dm_input_set(b | latched, atomic_load(&g_ext_lx), atomic_load(&g_ext_ly), atomic_load(&g_ext_rx), atomic_load(&g_ext_ry));
        }
        prev = b;
    }
    return NULL;
}
static void on_queue_created(ANativeActivity *a, AInputQueue *q) { (void)a; g_queue = q; while (!g_ilooper) usleep(1000); AInputQueue_attachLooper(q, g_ilooper, 1, on_input, NULL); }
static void on_queue_destroyed(ANativeActivity *a, AInputQueue *q) { (void)a; AInputQueue_detachLooper(q); g_queue = NULL; }

// ---- the host ----
int main(int argc, char **argv);   // dm/host/main.c
static void *host_thread(void *a) {
    (void)a;
    char mods[600]; snprintf(mods, sizeof mods, "%s/mods", g_root);
    char *argv[] = {"dmhost", mods, NULL};
    atomic_store(&g_running, 1);
    int rc = main(2, argv);
    LOG("host returned %d", rc);
    _exit(rc);
}
// GPU clock floor (every game by default, floor_level; the pause menu's GPU clock floor row chooses another level or none; env.txt DM_GPU_MIN_FREQ=<Hz>, a profile env row, sets another, 0 none): Adreno's governor parks at 660 MHz under a steady 30 fps load on the AYN Odin 3 (available
// frequencies up to 1.1 GHz), and the game's frame-rate controller then steps its present interval down; the same floor (967 MHz) in the emulator took its cave from 21 to 29 fps,
// and here p95 frame time 47 → 36 ms. Pokémon Legends: Arceus at 60 fps needs 734 MHz or more: at the 160 MHz floor the governor lowers the clock while the game holds 30,
// and every trial back to 60 fails (1.2-2.0 vsyncs a frame). The sysfs node is root's. Written first through the firmware's PServerBinder service (AYN's Odin 3 and other
// Android handhelds), which runs a short shell command as root for any app, without a rooted device; then through su (Magisk grants the app once); the line says which worked.
// The technique is the one PULSE (keiretrogaming/pulse, GPL-2.0) and ClusterTune (AurelioB/ClusterTune, MIT) use; this is our own code: ServiceManager.checkService
// ("PServerBinder"), transaction 0 with a parcel of an int32 count and that many strings, here the command (at most 255 bytes) and "1" for its output, which comes back as a
// byte array. The previous floor is restored when the game goes to the background (and set again when it comes back) and when the activity is destroyed; a force-stop
// destroys nothing, so the first PServer write also leaves a root shell that waits for this process to end and then writes the previous floor back. All of it runs on its own
// thread: the first build that called the service from onCreate sat on the splash screen on the Odin 3 (2026-11-11), and a game must start whatever the service does
static char g_gpu_floor_prev[32]; static const char *GPU_MIN_FREQ = "/sys/class/kgsl/kgsl-3d0/devfreq/min_freq";
static long g_gpu_floor_hz; static int g_gpu_floor_via, g_gpu_floor_on;   // the floor asked for (0 = none); 1 PServer, 2 su (0 = neither worked); whether it is in force now
static int su_run(const char *cmd, char *out, size_t n) {   // su -c <cmd>; out: its first line (optional); 0 on success
    char full[1024]; snprintf(full, sizeof full, "su -c '%s' 2>/dev/null", cmd);
    FILE *f = popen(full, "r"); if (!f) return -1;
    if (out && n) { out[0] = 0; if (fgets(out, (int)n, f)) out[strcspn(out, "\n")] = 0; }
    int rc = pclose(f); return rc == 0 ? 0 : -1;
}
// The command through PServerBinder, on a thread attached to the VM (the floor thread, never the main thread); out: its output's first line (optional).
// 0 on success, -1 when the service is not there, -2 when the call failed
static int pserver_run(JNIEnv *E, const char *cmd, char *out, size_t n) {
    if (out && n) out[0] = 0;
    if (!E || strlen(cmd) > 255) return -2;
    int rc = -1; jobject binder = NULL, data = NULL, reply = NULL;
    if ((*E)->PushLocalFrame(E, 16) != JNI_OK) { (*E)->ExceptionClear(E); return -2; }
    jclass sm = (*E)->FindClass(E, "android/os/ServiceManager");
    jmethodID check = sm && !(*E)->ExceptionCheck(E) ? (*E)->GetStaticMethodID(E, sm, "checkService", "(Ljava/lang/String;)Landroid/os/IBinder;") : NULL;
    if (check && !(*E)->ExceptionCheck(E)) binder = (*E)->CallStaticObjectMethod(E, sm, check, (*E)->NewStringUTF(E, "PServerBinder"));
    if ((*E)->ExceptionCheck(E)) { (*E)->ExceptionClear(E); binder = NULL; }
    if (binder) {
        rc = -2;
        jclass pc = (*E)->FindClass(E, "android/os/Parcel"), bc = (*E)->FindClass(E, "android/os/IBinder");
        jmethodID obtain = pc ? (*E)->GetStaticMethodID(E, pc, "obtain", "()Landroid/os/Parcel;") : NULL, wi = pc ? (*E)->GetMethodID(E, pc, "writeInt", "(I)V") : NULL,
                  ws = pc ? (*E)->GetMethodID(E, pc, "writeString", "(Ljava/lang/String;)V") : NULL, cb = pc ? (*E)->GetMethodID(E, pc, "createByteArray", "()[B") : NULL,
                  rec = pc ? (*E)->GetMethodID(E, pc, "recycle", "()V") : NULL, tr = bc ? (*E)->GetMethodID(E, bc, "transact", "(ILandroid/os/Parcel;Landroid/os/Parcel;I)Z") : NULL;
        if (!(*E)->ExceptionCheck(E) && obtain && wi && ws && cb && rec && tr) {
            data = (*E)->CallStaticObjectMethod(E, pc, obtain); reply = (*E)->CallStaticObjectMethod(E, pc, obtain);
            if (data && reply && !(*E)->ExceptionCheck(E)) {
                (*E)->CallVoidMethod(E, data, wi, (jint)2);
                (*E)->CallVoidMethod(E, data, ws, (*E)->NewStringUTF(E, cmd));
                (*E)->CallVoidMethod(E, data, ws, (*E)->NewStringUTF(E, "1"));
                jboolean ok = !(*E)->ExceptionCheck(E) && (*E)->CallBooleanMethod(E, binder, tr, (jint)0, data, reply, (jint)0);
                if (ok && !(*E)->ExceptionCheck(E)) {
                    rc = 0;
                    jbyteArray b = (jbyteArray)(*E)->CallObjectMethod(E, reply, cb);
                    if (b && !(*E)->ExceptionCheck(E) && out && n) {
                        jsize len = (*E)->GetArrayLength(E, b); if ((size_t)len >= n) len = (jsize)n - 1;
                        (*E)->GetByteArrayRegion(E, b, 0, len, (jbyte *)out); out[len] = 0; out[strcspn(out, "\n")] = 0;
                        if (!strcmp(out, "null")) out[0] = 0;
                    }
                }
            }
            if ((*E)->ExceptionCheck(E)) { (*E)->ExceptionClear(E); rc = -2; }
            if (data) (*E)->CallVoidMethod(E, data, rec);
            if (reply) (*E)->CallVoidMethod(E, reply, rec);
        }
        if ((*E)->ExceptionCheck(E)) { (*E)->ExceptionClear(E); rc = -2; }
    }
    (*E)->PopLocalFrame(E, NULL);
    return rc;
}
static JNIEnv *g_fl_env;   // the floor thread's (gpu_floor_thread): no PServer call runs on the main thread, so a slow or stuck service never holds up start-up or the lifecycle
static int floor_run(int via, const char *cmd, char *out, size_t n) { return via == 1 ? pserver_run(g_fl_env, cmd, out, n) : su_run(cmd, out, n); }
static long floor_read(void) { FILE *f = fopen(GPU_MIN_FREQ, "r"); long v = -1; if (f) { if (fscanf(f, "%ld", &v) != 1) v = -1; fclose(f); } return v; }   // -1 when the app may not read it
// Sets the floor through `via`, the previous value into g_gpu_floor_prev; 0 when the node reads the floor afterwards (or cannot be read by the app but the write ran)
static int floor_set(int via) {
    char cmd[256], prev[32] = "";
    snprintf(cmd, sizeof cmd, "cat %s && echo %ld > %s", GPU_MIN_FREQ, g_gpu_floor_hz, GPU_MIN_FREQ);
    if (floor_run(via, cmd, prev, sizeof prev) || !atol(prev)) return -1;
    long now = floor_read(); if (now >= 0 && now != g_gpu_floor_hz) return -1;
    if (!g_gpu_floor_prev[0]) snprintf(g_gpu_floor_prev, sizeof g_gpu_floor_prev, "%ld", atol(prev));   // the floor before the game, kept across trips to the background
    return 0;
}
// The device's clocks (available_frequencies, through PServerBinder), read once by the floor thread; none: no service or no list, and the menu's row says so
static long g_fl_clock[32]; static _Atomic int g_fl_nclock;
static void floor_clocks(void) {
    char cmd[160], out[512]; snprintf(cmd, sizeof cmd, "cat %.*s/available_frequencies", (int)(strrchr(GPU_MIN_FREQ, '/') - GPU_MIN_FREQ), GPU_MIN_FREQ);
    if (pserver_run(g_fl_env, cmd, out, sizeof out)) return;
    int n = 0; char *p = out, *e;
    for (long v; n < 32 && ((v = strtol(p, &e, 10)), e != p); p = e) if (v > 0) g_fl_clock[n++] = v;
    atomic_store(&g_fl_nclock, n);
}
// A level's floor: the lowest clock at or above pct % of the highest. 75 is the default (AYN Odin 3: 832 MHz of 1.1 GHz, the floor that held Legends: Arceus at 60 and
// the one Kirby and the Forgotten Land lacked when the governor sat at 342 MHz, 85 % busy, with a 15.6 ms GPU frame); 0 without the list
static long floor_level(int pct) {
    long top = 0, pick = 0; int n = atomic_load(&g_fl_nclock);
    for (int i = 0; i < n; i++) if (g_fl_clock[i] > top) top = g_fl_clock[i];
    for (int i = 0; i < n; i++) if (g_fl_clock[i] >= top / 100 * pct && (!pick || g_fl_clock[i] < pick)) pick = g_fl_clock[i];
    return pick;
}
// The floor wanted now: the menu's choice (settings.txt gpufloor=), else the profile's (DM_GPU_MIN_FREQ=<Hz>; 0: none), else the default level. *by_default: no profile
// row asked for it, so it goes through PServerBinder only (never a su prompt nobody asked for)
static long floor_wanted(int *by_default) {
    int s = atomic_load(&g_set_floor); const char *e = getenv("DM_GPU_MIN_FREQ");
    *by_default = s >= 0 || !e;
    if (s >= 0) return OVM_FLOORS[s] ? floor_level(OVM_FLOORS[s]) : 0;
    return e ? (atol(e) > 0 ? atol(e) : 0) : floor_level(OVM_FLOORS[OVM_FLOOR_DEFAULT]);
}
static int gpu_floor_choice(void) {   // the menu's row: the level chosen; before a choice the default level, or Custom for a profile's own clock
    if (!atomic_load(&g_fl_nclock)) return OVM_NA;
    int s = atomic_load(&g_set_floor); const char *e = getenv("DM_GPU_MIN_FREQ");
    return s >= 0 ? s : !e ? OVM_FLOOR_DEFAULT : atol(e) > 0 ? OVM_NFLOORS : 0;
}
static void floor_back(void) {   // the previous floor again
    if (!g_gpu_floor_on || !g_gpu_floor_prev[0]) return;
    char cmd[128]; snprintf(cmd, sizeof cmd, "echo %s > %s", g_gpu_floor_prev, GPU_MIN_FREQ);
    if (floor_run(g_gpu_floor_via, cmd, NULL, 0)) LOG("[RED] GPU clock floor: restoring %s Hz failed", g_gpu_floor_prev);
    g_gpu_floor_on = 0;
}
static void floor_change(int first) {   // the floor thread, at its start and after a choice in the menu: the floor wanted, set or taken away
    int by_default; long hz = floor_wanted(&by_default);
    if (!hz) {
        if (first && by_default && atomic_load(&g_set_floor) < 0) LOG("GPU clock floor: none by default on this device (no PServerBinder service or no clock list); DM_GPU_MIN_FREQ=<Hz> in a profile asks for one through su");
        else if (g_gpu_floor_on) LOG("GPU clock floor off: %s Hz again", g_gpu_floor_prev);
        floor_back(); g_gpu_floor_hz = 0; return;
    }
    if (hz == g_gpu_floor_hz && g_gpu_floor_on) return;
    g_gpu_floor_hz = hz; int had = g_gpu_floor_via;
    if (had) { if (floor_set(had)) { LOG("[RED] GPU clock floor %ld Hz: setting it failed", hz); return; } }
    else for (int via = 1; via <= (by_default ? 1 : 2) && !g_gpu_floor_via; via++) {
        LOG("GPU clock floor %ld Hz: trying %s", g_gpu_floor_hz, via == 1 ? "PServerBinder" : "su");
        if (!floor_set(via)) g_gpu_floor_via = via;
    }
    if (!g_gpu_floor_via && by_default) { LOG("GPU clock floor %ld Hz (the default): PServerBinder refused; none set", g_gpu_floor_hz); return; }
    if (!g_gpu_floor_via) { LOG("[RED] GPU clock floor %ld Hz: PServerBinder not there or refused, and su refused (grant the app root in Magisk, or leave DM_GPU_MIN_FREQ out)", g_gpu_floor_hz); return; }
    g_gpu_floor_on = 1;
    LOG("GPU clock floor %ld Hz through %s (was %s; restored in the background and when the game ends)", g_gpu_floor_hz, g_gpu_floor_via == 1 ? "PServerBinder (no root)" : "su", g_gpu_floor_prev);
    if (g_gpu_floor_via == 1 && !had) {   // the watcher for a force-stop (no onDestroy): a root shell, detached from the service's output, that writes the previous floor back once this process is gone
        char cmd[256]; snprintf(cmd, sizeof cmd, "(while [ -d /proc/%d ]; do sleep 2; done; echo %s > %s) </dev/null >/dev/null 2>&1 &", (int)getpid(), g_gpu_floor_prev, GPU_MIN_FREQ);
        if (pserver_run(g_fl_env, cmd, NULL, 0)) LOG("[RED] GPU clock floor: the watcher that restores %s Hz after a force-stop did not start", g_gpu_floor_prev);
        else LOG("GPU clock floor: a root watcher restores %s Hz if the app is killed", g_gpu_floor_prev);
    }
}
// The main thread only posts what it wants (1 the floor, 0 the previous one; a choice in the menu: g_fl_pick counts them); the floor thread does it
static pthread_mutex_t g_fl_mu = PTHREAD_MUTEX_INITIALIZER; static pthread_cond_t g_fl_cv = PTHREAD_COND_INITIALIZER;
static int g_fl_want = 1, g_fl_have = -1, g_fl_started, g_fl_pick, g_fl_picked;   // asked for; done (-1 = not yet); the menu's choices, and those taken
static void *gpu_floor_thread(void *vm) {
    JavaVM *V = vm; if ((*V)->AttachCurrentThread(V, &g_fl_env, NULL) != JNI_OK) g_fl_env = NULL;
    floor_clocks(); floor_change(1);
    pthread_mutex_lock(&g_fl_mu); g_fl_have = 1; pthread_cond_broadcast(&g_fl_cv);
    for (;;) {
        while (g_fl_want == g_fl_have && g_fl_pick == g_fl_picked) pthread_cond_wait(&g_fl_cv, &g_fl_mu);
        int want = g_fl_want, pick = g_fl_pick, chose = pick != g_fl_picked; pthread_mutex_unlock(&g_fl_mu);
        if (chose && want) floor_change(0);
        if (g_gpu_floor_via && g_gpu_floor_hz && want && !g_gpu_floor_on) { if (floor_set(g_gpu_floor_via)) LOG("[RED] GPU clock floor %ld Hz: setting it again after the background failed", g_gpu_floor_hz); else g_gpu_floor_on = 1; }
        else if (!want) floor_back();
        pthread_mutex_lock(&g_fl_mu); g_fl_have = want; g_fl_picked = pick; pthread_cond_broadcast(&g_fl_cv);
    }
    return NULL;
}
static void gpu_floor_apply(void) {   // onCreate, after settings.txt is read: the floor thread, which sets the floor wanted (floor_wanted) and takes the menu's choices
    if (!g_act) return;
    pthread_t t; if (pthread_create(&t, NULL, gpu_floor_thread, g_act->vm)) { LOG("[RED] GPU clock floor: no thread for it"); return; }
    pthread_detach(t); g_fl_started = 1;
}
static void gpu_floor_pick(void) { if (!g_fl_started) return; pthread_mutex_lock(&g_fl_mu); g_fl_pick++; pthread_cond_broadcast(&g_fl_cv); pthread_mutex_unlock(&g_fl_mu); }   // the menu chose
static void floor_post(int want, int wait_ms) {
    if (!g_fl_started) return;
    pthread_mutex_lock(&g_fl_mu); g_fl_want = want; pthread_cond_broadcast(&g_fl_cv);
    if (wait_ms > 0) {   // onDestroy: a short wait for the previous floor, then the process ends whatever the service does
        struct timespec ts; clock_gettime(CLOCK_REALTIME, &ts); ts.tv_nsec += (long)wait_ms * 1000000; ts.tv_sec += ts.tv_nsec / 1000000000; ts.tv_nsec %= 1000000000;
        while (g_fl_have != want) if (pthread_cond_timedwait(&g_fl_cv, &g_fl_mu, &ts)) break;
    }
    pthread_mutex_unlock(&g_fl_mu);
}
static void gpu_floor_restore(void) { floor_post(0, 0); }   // the background: the previous floor back
static void gpu_floor_resume(void) { floor_post(1, 0); }    // back in front: the floor again, through the path that worked
static void gpu_floor_end(void) { floor_post(0, 500); }     // onDestroy
// The previous start's dumps kept beside the new ones (droidplay-core e6eb551de): the host opens DM_NVN_TSV and DM_AUDIO_DUMP "w", so a start after a kill (lmkd,
// a crash) erased the killed run's. Each goes to <name>.prev<extensions> (bench.tsv → bench.prev.tsv), over the one before; log.txt → log.prev.txt is done above
static int keep_prev(const char *path) {
    struct stat st; char prev[1024];
    if (!path || !*path || stat(path, &st) || !S_ISREG(st.st_mode)) return 0;
    const char *base = strrchr(path, '/'); base = base ? base + 1 : path;
    const char *dot = strchr(base, '.'); size_t at = dot ? (size_t)(dot - path) : strlen(path);
    int k = snprintf(prev, sizeof prev, "%.*s.prev%s", (int)at, path, path + at);
    return k > 0 && (size_t)k < sizeof prev && rename(path, prev) == 0;
}
static void env_file(const char *path) {   // KEY=VALUE per line (# comments): switches without rebuilding, as the iOS app's Documents/env.txt
    FILE *f = fopen(path, "r"); char line[1024];
    if (!f) return;
    while (fgets(line, sizeof line, f)) {
        char *e = line + strlen(line); while (e > line && (e[-1] == '\n' || e[-1] == '\r' || e[-1] == ' ')) *--e = 0;
        if (!line[0] || line[0] == '#') continue;
        char *eq = strchr(line, '='); if (!eq) continue; *eq = 0; setenv(line, eq + 1, 1);
    }
    fclose(f);
}
// ---- a RomFS outside the app's files folder (env.txt's DM_ROMFS, written by porter install --romfs-at: a game dump another app also reads, a folder on the SD card). The app
// reads it only with All files access (the appop MANAGE_EXTERNAL_STORAGE, Android 11 on) or, on Android 10, READ_EXTERNAL_STORAGE in the legacy storage view: mkapk.py asks
// for both and porter grants them over adb. Without it scoped storage gives the app none of the files another app made there, and the game would fail somewhere in its
// boot with nothing that says why: the app says so in the log, opens Android's page for the switch and ends (not yet seen on a device: the check is from Android's rules) ----
static void open_access_page(ANativeActivity *act, int sdk) {   // the app's All files access switch (Android 10: its app info page, where its permissions are)
    JNIEnv *E = act->env; jobject a = act->clazz; char uri[300];
    do {
        jclass cA = (*E)->GetObjectClass(E, a); jmethodID mPkg = (*E)->GetMethodID(E, cA, "getPackageName", "()Ljava/lang/String;");
        jstring pkg = mPkg ? (*E)->CallObjectMethod(E, a, mPkg) : NULL; if (jx(E) || !pkg) break;
        const char *pc = (*E)->GetStringUTFChars(E, pkg, NULL); snprintf(uri, sizeof uri, "package:%s", pc ? pc : ""); if (pc) (*E)->ReleaseStringUTFChars(E, pkg, pc);
        jclass cU = (*E)->FindClass(E, "android/net/Uri"); jmethodID mParse = cU ? (*E)->GetStaticMethodID(E, cU, "parse", "(Ljava/lang/String;)Landroid/net/Uri;") : NULL; if (jx(E) || !mParse) break;
        jobject u = (*E)->CallStaticObjectMethod(E, cU, mParse, (*E)->NewStringUTF(E, uri)); if (jx(E) || !u) break;
        jclass cI = (*E)->FindClass(E, "android/content/Intent"); jmethodID mNew = cI ? (*E)->GetMethodID(E, cI, "<init>", "(Ljava/lang/String;Landroid/net/Uri;)V") : NULL; if (jx(E) || !mNew) break;
        jmethodID mStart = (*E)->GetMethodID(E, cA, "startActivity", "(Landroid/content/Intent;)V"); if (jx(E) || !mStart) break;
        const char *pages[] = {sdk >= 30 ? "android.settings.MANAGE_APP_ALL_FILES_ACCESS_PERMISSION" : "android.settings.APPLICATION_DETAILS_SETTINGS", "android.settings.APPLICATION_DETAILS_SETTINGS"};
        for (int i = 0; i < 2; i++) {   // a device without the first page (ActivityNotFoundException) gets the app's info page
            jobject it = (*E)->NewObject(E, cI, mNew, (*E)->NewStringUTF(E, pages[i]), u);
            if (!jx(E) && it) { (*E)->CallVoidMethod(E, a, mStart, it); if (!jx(E)) { LOG("RomFS: opened Android's page %s for the app", pages[i]); return; } }
        }
    } while (0);
    LOG("[RED] RomFS: Android's All files access page could not be opened: open it by hand (Settings > Apps > the game)");
}
static int romfs_readable(ANativeActivity *act, int sdk) {   // 0: the game cannot read its RomFS, and the app ends
    const char *r = getenv("DM_ROMFS"), *own[] = {g_root, act->internalDataPath}; char rr[4096], orr[4096];
    const char *rp = r && realpath(r, rr) ? rr : r;   // canonical paths: /sdcard/… is the /storage/emulated/0/… Android names the files folder by
    for (int i = 0; i < 2; i++) { const char *o = own[i] && realpath(own[i], orr) ? orr : own[i]; size_t n = o ? strlen(o) : 0;
        if (!rp || (n && !strncmp(rp, o, n) && (rp[n] == '/' || !rp[n]))) return 1; }   // the app's own folders need no permission
    JNIEnv *E = act->env; jclass c; jmethodID m; int ok = -1;   // -1: not known
    if (sdk >= 30) { c = (*E)->FindClass(E, "android/os/Environment"); m = c ? (*E)->GetStaticMethodID(E, c, "isExternalStorageManager", "()Z") : NULL; if (m) ok = (*E)->CallStaticBooleanMethod(E, c, m) ? 1 : 0; }
    else { jstring s = (*E)->NewStringUTF(E, "android.permission.READ_EXTERNAL_STORAGE"); c = (*E)->GetObjectClass(E, act->clazz);
           m = s ? (*E)->GetMethodID(E, c, "checkSelfPermission", "(Ljava/lang/String;)I") : NULL; if (m) ok = (*E)->CallIntMethod(E, act->clazz, m, s) == 0; }   // PERMISSION_GRANTED
    if (jx(E)) ok = -1;
    const char *what = sdk >= 30 ? "All files access" : "the storage permission (READ_EXTERNAL_STORAGE)";
    if (!ok) {
        LOG("[RED] RomFS: %s is outside the app's files folder, and the app does not have %s, so the game cannot read its files there: the game is not started. "
            "Allow it on the device (%s), or install again with porter install, which grants it over adb; the app opens that page and ends",
            r, what, sdk >= 30 ? "Settings > Apps > the game > All files access" : "Settings > Apps > the game > Permissions > Files and media");
        open_access_page(act, sdk); return 0;
    }
    if (ok < 0) LOG("[RED] RomFS: %s is outside the app's files folder, and whether the app has %s is not known (a Java call failed): starting the game anyway", r, what);
    struct stat st; int e = stat(r, &st) ? errno : S_ISDIR(st.st_mode) ? 0 : ENOTDIR;
    if (e) { LOG("[RED] RomFS: %s is not there (errno=%d): an SD card that is not in, or a folder moved or renamed; porter install --romfs-at <folder> points the app at it "
                 "(--romfs-at app: back into the files folder). The game is not started", r, e); return 0; }
    LOG("RomFS: %s, outside the app's files folder (%s granted)", r, what);
    return 1;
}
__attribute__((visibility("default"))) void ANativeActivity_onCreate(ANativeActivity *act, void *saved, size_t saved_size) {
    (void)saved; (void)saved_size;
    g_act = act;
    int sdk; { AConfiguration *cfg = AConfiguration_new(); AConfiguration_fromAssetManager(cfg, act->assetManager); g_dpi = AConfiguration_getDensity(cfg); sdk = AConfiguration_getSdkVersion(cfg); AConfiguration_delete(cfg); }
    act->callbacks->onNativeWindowCreated = on_window_created; act->callbacks->onNativeWindowDestroyed = on_window_destroyed;
    act->callbacks->onInputQueueCreated = on_queue_created; act->callbacks->onInputQueueDestroyed = on_queue_destroyed;
    act->callbacks->onPause = on_pause; act->callbacks->onResume = on_resume; act->callbacks->onWindowFocusChanged = on_window_focus; act->callbacks->onDestroy = on_destroy;
    ANativeActivity_setWindowFlags(act, AWINDOW_FLAG_KEEP_SCREEN_ON | AWINDOW_FLAG_FULLSCREEN, 0);
    static int started; if (started++) return;   // a second onCreate (configuration change) keeps the running game
    snprintf(g_root, sizeof g_root, "%s", act->externalDataPath ? act->externalDataPath : act->internalDataPath);
    char p[700];
    snprintf(p, sizeof p, "%s/log.txt", g_root); char prev[720]; snprintf(prev, sizeof prev, "%s/log.prev.txt", g_root); rename(p, prev);
    if (!freopen(p, "w", stderr)) __android_log_print(ANDROID_LOG_ERROR, "dmhost", "cannot write %s", p);
    setvbuf(stderr, NULL, _IOLBF, 0);
    setenv("DM_ROOT", g_root, 1);
    snprintf(p, sizeof p, "%s/RomFS", g_root); setenv("DM_ROMFS", p, 0);
    snprintf(p, sizeof p, "%s/save", act->internalDataPath); mkdir(p, 0700); setenv("DM_SAVE", p, 0);
    snprintf(p, sizeof p, "%s/temp", act->internalDataPath); mkdir(p, 0700); setenv("DM_TEMP", p, 0);
    snprintf(p, sizeof p, "%s/shaders.dmsv", g_root); setenv("DM_VK_SHADERS", p, 0);
    snprintf(p, sizeof p, "%s/env.txt", g_root); env_file(p);
    { int kept = keep_prev(getenv("DM_NVN_TSV")) + keep_prev(getenv("DM_AUDIO_DUMP"));   // before the host opens them
      if (kept) LOG("the previous start's dumps: %d kept beside the new ones, with .prev before the extension", kept); }
#ifdef DM_RELEASE
    unsetenv("DM_CTL");   // a release build takes no remote key presses (the control file is a development tool)
#endif
    // the save manager's requests (saves_android.c), before the host mounts the save: an import chosen in the menu or pushed by porter, an export porter asks for
    if (dm_saves_start(g_root)) { LOG("[saves] the request asked the app to end: the game is not started"); g_quit = 1; return; }
    if (!romfs_readable(act, sdk)) { g_quit = 1; return; }   // after the save manager's requests, which need no RomFS
    settings_load(); dm_frame_cap_set(OVM_CAPS[atomic_load(&g_set_cap)]); dm_res_set(atomic_load(&g_set_res)); dm_haptics_android_start(act, atomic_load(&g_set_rumble)); dm_lan_android_start(act); app_label(act);
    gpu_floor_apply();   // (after settings.txt: the menu's floor, when the player chose one)
    { pthread_t w; if (!pthread_create(&w, NULL, pad_watch, act->vm)) pthread_detach(w); }
    { pthread_t w; if (!pthread_create(&w, NULL, temp_watch, act->vm)) pthread_detach(w); }
    LOG("NX2A host on Android: files %s, RomFS %s, saves %s", g_root, getenv("DM_ROMFS"), act->internalDataPath);
    keep_alive(act);
    pthread_t t; pthread_create(&t, NULL, input_thread, NULL); pthread_detach(t);
    // The game's main thread: a 16 MB stack mapped here and handed over whole (pthread_attr_setstack), so the range pthread_getattr_np reports, which
    // svcQueryMemory gives the game as its main thread stack (hle_svc.c), is exactly mapped memory: IL2CPP's garbage collector scans that range from
    // its bottom, and with bionic's own stack the bottom of the reported range was not readable (SIGSEGV at main+0x2a59b4). A guard page sits below it.
    size_t ssz = 16 << 20, guard = 64 << 10;
    uint8_t *stk = mmap(NULL, ssz + guard, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    pthread_attr_t at; pthread_attr_init(&at);
    if (stk != MAP_FAILED) { mprotect(stk, guard, PROT_NONE); pthread_attr_setstack(&at, stk + guard, ssz); } else pthread_attr_setstacksize(&at, ssz);
    if (pthread_create(&t, &at, host_thread, NULL)) LOG("[RED] cannot start the host thread"); else pthread_detach(t);
    pthread_attr_destroy(&at);
}

// The controller applet's prompt (front.h, ctrl_prompt.h; hle_hid.c ctrl_support): the game wants min..max players and sees fewer. No prompt on Android yet: -1, and the host answers
// as the player closing it (ResultControllerSupportCanceled), as it did before the upsync. TODO(upsync): an overlay prompt while controllers join, as DmGamepad.m's
int front_ctrl_support(int min, int max) { (void)min; (void)max; return -1; }
