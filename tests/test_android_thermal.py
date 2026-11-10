"""perf_android.c's dm_thermal_now: AThermal's status on the host's 0..3 thermal scale, read off the frame's threads, and DM_THERMAL_CAP=0 leaving it at -1."""
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import host_cc   # noqa: E402

SOURCE = Path(__file__).resolve().parents[1] / 'nxnative/dm/android/perf_android.c'
HARNESS = r'''
#define _GNU_SOURCE
#include <assert.h>
#include <dlfcn.h>
#include <errno.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#ifdef __APPLE__   // macOS names only the calling thread (pthread_setname_np(name)); Android and Linux name any thread
#define pthread_setname_np(thread, name) ((void)(thread), (void)(name), 0)
#endif
static int logs, opens, acquires, statuses, status = 3; static pthread_t main_thread, status_thread; static char last[512];
#define LOG(...) do { snprintf(last, sizeof last, __VA_ARGS__); __atomic_fetch_add(&logs, 1, __ATOMIC_RELAXED); } while (0)
static int fake_manager;
static void *fake_acquire(void) { __atomic_fetch_add(&acquires, 1, __ATOMIC_RELAXED); return &fake_manager; }
static int fake_status(void *m) { assert(m == &fake_manager); status_thread = pthread_self(); __atomic_fetch_add(&statuses, 1, __ATOMIC_RELAXED); return __atomic_load_n(&status, __ATOMIC_RELAXED); }
static void *fake_dlopen(const char *name, int flags) { (void)flags; assert(!strcmp(name, "libandroid.so")); opens++; return &opens; }
static void *fake_dlsym(void *lib, const char *name) {
    (void)lib;
    if (!strcmp(name, "AThermal_acquireManager")) return getenv("NO_ATHERMAL") ? NULL : (void *)fake_acquire;
    if (!strcmp(name, "AThermal_getCurrentThermalStatus")) return (void *)fake_status;
    return NULL;
}
#define dlopen fake_dlopen
#define dlsym fake_dlsym
BLOCK
static int wait_for(int want) { for (int i = 0; i < 300; i++) { if (dm_thermal_now() == want) return 1; usleep(10000); } return 0; }
int main(int argc, char **argv) {
    (void)argc; main_thread = pthread_self(); int mode = atoi(argv[1]);
    if (mode == 0) {   // the scale: ERROR -1, NONE 0, LIGHT 1, MODERATE 1, SEVERE 2, CRITICAL / EMERGENCY / SHUTDOWN 3
        int want[] = {-1, 0, 1, 1, 2, 3, 3, 3};
        for (int s = -1; s <= 6; s++) assert(thermal_scale(s) == want[s + 1]);
        T.status = (int (*)(AThermalManager *))fake_status; T.mgr = (AThermalManager *)&fake_manager;
        status = 0; thermal_poll(); assert(logs == 1 && strstr(last, "state 0"));
        thermal_poll(); assert(logs == 1);   // no change, no line
        status = 4; thermal_poll(); assert(logs == 2 && strstr(last, "thermal status 4 (AThermal): state 3"));
        assert(__atomic_load_n(&g_thermal, __ATOMIC_RELAXED) == 3);
    } else if (mode == 1) {   // read on a thread of its own, then the value every caller sees
        status = 3; assert(wait_for(2)); assert(!pthread_equal(status_thread, main_thread) && opens == 1 && acquires == 1);
        status = 1; assert(wait_for(1)); assert(acquires == 1);
    } else if (mode == 2) {   // DM_THERMAL_CAP=0: -1 as before, nothing opened
        assert(dm_thermal_now() == -1); usleep(50000); assert(dm_thermal_now() == -1 && !opens && !statuses && strstr(last, "DM_THERMAL_CAP=0"));
    } else {   // no AThermal: -1, said once
        assert(dm_thermal_now() == -1); usleep(100000); assert(dm_thermal_now() == -1 && !statuses && logs == 1 && strstr(last, "no AThermal"));
    }
    puts("thermal ok"); return 0;
}
'''


@unittest.skipUnless(shutil.which('cc') and host_cc.posix(), 'a POSIX host C compiler is required')
class AndroidThermal(unittest.TestCase):
    def test_thermal_state(self):
        src = SOURCE.read_text(encoding='utf-8')
        start = src.index('// ---- thermal status (AThermal')
        end = src.index('\n}\n', src.index('int dm_thermal_now(void)')) + 3
        with tempfile.TemporaryDirectory(prefix='nx2a-thermal-') as tmp:
            c, exe = Path(tmp) / 'check.c', Path(tmp) / 'check'
            c.write_text(HARNESS.replace('BLOCK', src[start:end]), encoding='utf-8')
            subprocess.run(['cc', '-std=gnu11', '-Wall', '-Wextra', '-Werror', '-pthread', str(c), '-o', str(exe)], check=True)
            for mode, env in (('0', {}), ('1', {}), ('2', {'DM_THERMAL_CAP': '0'}), ('3', {'NO_ATHERMAL': '1'})):
                with self.subTest(mode=mode):
                    e = {k: v for k, v in __import__('os').environ.items() if k not in ('DM_THERMAL_CAP', 'NO_ATHERMAL')}
                    out = subprocess.run([str(exe), mode], capture_output=True, text=True, env={**e, **env}, timeout=60)
                    self.assertEqual(out.returncode, 0, out.stderr)
                    self.assertIn('thermal ok', out.stdout)


if __name__ == '__main__':
    unittest.main()
