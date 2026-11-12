"""Pool read marks (DM_VK_POOL_MARK): the shader rewrite that stores the texture IDs a shader reads, and the host fold of a slot's marks."""
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile
import unittest
import sys; sys.path.insert(0, str(Path(__file__).resolve().parent))
import host_cc   # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def function(source, name):
    start = source.index(name)
    return source[start:source.index("\n}\n", start) + 3]


def op(code, *words):
    return [(len(words) + 1) << 16 | code, *words]


def string(s):
    b = s.encode() + b"\0"; b += b"\0" * (-len(b) % 4)
    return list(struct.unpack(f"<{len(b) // 4}I", b))


def module(clamped=True, version=0x00010300):
    """A fragment shader as the translator writes a texture pool read: the handle's ID, UMin-clamped to the table's last index, as int, into the pool array (ID 16)."""
    w = [0x07230203, version, 0, 40, 0]
    w += op(17, 1) + op(11, 1, *string("GLSL.std.450")) + op(14, 0, 1) + op(15, 4, 27, *string("main"), 21, 23) + op(16, 27, 7)
    w += op(71, 21, 14) + op(71, 21, 30, 0) + op(71, 23, 30, 0) + op(71, 16, 34, 1) + op(71, 16, 33, 26) + op(71, 19, 34, 1) + op(71, 19, 33, 27)
    w += op(19, 2) + op(33, 3, 2) + op(21, 4, 32, 0) + op(21, 5, 32, 1) + op(22, 6, 32) + op(23, 7, 6, 4) + op(23, 8, 6, 2)
    w += op(25, 9, 6, 1, 0, 0, 0, 1, 0) + op(27, 10, 9) + op(43, 4, 11, 65536) + op(43, 4, 12, 65535) + op(28, 13, 9, 11)
    w += op(32, 14, 0, 13) + op(32, 15, 0, 9) + op(59, 14, 16, 0) + op(26, 17) + op(32, 18, 0, 17) + op(59, 18, 19, 0)
    w += op(32, 20, 1, 4) + op(59, 20, 21, 1) + op(32, 22, 3, 7) + op(59, 22, 23, 3) + op(43, 5, 24, 0xFFFFF) + op(43, 6, 25, 0x3F000000) + op(44, 8, 26, 25, 25)
    w += op(54, 2, 27, 0, 3) + op(248, 28) + op(61, 4, 29, 21) + op(124, 5, 30, 29) + op(199, 5, 31, 30, 24) + op(124, 4, 32, 31)
    w += op(12, 4, 33, 1, 38, 32, 12) + op(124, 5, 34, 33 if clamped else 32) + op(65, 15, 35, 16, 34) + op(61, 9, 36, 35) + op(61, 17, 37, 19)
    w += op(86, 10, 38, 36, 37) + op(87, 7, 39, 38, 26) + op(62, 23, 39) + op(253) + op(56)
    return struct.pack(f"<{len(w)}I", *w)


HARNESS = r"""
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
enum { POOL_MARK_B = 8 };
SHADER
WMEM
MARKS
static uint32_t *load(const char *p, uint32_t *nw) {
    FILE *f = fopen(p, "rb"); assert(f); fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    uint32_t *w = malloc((size_t)n); assert(fread(w, 1, (size_t)n, f) == (size_t)n); fclose(f); *nw = (uint32_t)(n / 4); return w;
}
static void scan_test(void) {
    enum { NP = 64 }; uint32_t m[NP] = {0}, last[NP] = {0}, touched[NP] = {0}; int warm[NP], touch[NP], nw, nt;
    m[5] = 1; m[6] = 1; m[63] = 1;   // read in the submission folded at present 10 (now 11)
    assert(pool_mark_scan(m, NP, 11, last, touched, warm, &nw, touch, &nt) == 3);
    assert(nw == 3 && warm[0] == 5 && warm[1] == 6 && warm[2] == 63 && nt == 3);   // read for the first time: warm, and their images counted as drawn
    for (int k = 0; k < NP; k++) assert(!m[k]);                                      // cleared for the slot's next submission
    assert(last[5] == 11 && last[6] == 11 && last[63] == 11 && !last[4]);
    m[5] = 1; assert(pool_mark_scan(m, NP, 12, last, touched, warm, &nw, touch, &nt) == 1 && nw == 0 && nt == 0 && last[5] == 12);   // read again soon: neither
    m[5] = 1; assert(pool_mark_scan(m, NP, 12 + MARK_TOUCH, last, touched, warm, &nw, touch, &nt) == 1 && nw == 0 && nt == 1 && touch[0] == 5);   // its image counted again
    m[6] = 1; assert(pool_mark_scan(m, NP, 11 + MARK_RECENT + 1, last, touched, warm, &nw, touch, &nt) == 1 && nw == 1 && warm[0] == 6);       // unread too long: warm again
    m[63] = 1; assert(pool_mark_scan(m, NP, 11 + MARK_RECENT, last, touched, warm, &nw, touch, &nt) == 1 && nw == 0);                         // (not yet)
    uint32_t odd[3] = {0, 0, 1}, ol[3] = {0}, ot[3] = {0}; assert(pool_mark_scan(odd, 3, 1, ol, ot, warm, &nw, touch, &nt) == 1 && warm[0] == 2 && !odd[2]);
}
int main(int argc, char **argv) {
    assert(argc >= 2); if (!strcmp(argv[1], "scan")) { scan_test(); puts("ok"); return 0; }
    assert(argc == 5);
    uint32_t nw, *w = load(argv[1], &nw), pool = (uint32_t)atoi(argv[3]), np = (uint32_t)atoi(argv[4]), before = nw, *copy = malloc(nw * 4);
    memcpy(copy, w, nw * 4);
    uint32_t var = 0; const char *why = NULL; int r = spv_pool_mark(&w, &nw, &pool, 1, np, &var, &why);
    if (r <= 0) { assert(nw == before && !memcmp(w, copy, nw * 4) && !var); printf("%d %s\n", r, why ? why : "-"); free(w); free(copy); return 0; }   // unchanged when nothing is marked
    assert(var && why == NULL && w[3] > copy[3] && spv_wmem(w, nw, var) == 0 && spv_wmem(w, nw, 0) == 1 && spv_wmem(copy, before, 0) == 0);
    FILE *f = fopen(argv[2], "wb"); fwrite(w, 4, nw, f); fclose(f);
    printf("%d %u\n", r, var); free(w); free(copy); return 0;
}
"""


@unittest.skipUnless(shutil.which("cc"), "a host C compiler is required")
@unittest.skipUnless(host_cc.posix() and host_cc.sanitizers(), "a POSIX host C compiler with ASan and UBSan is required")
class PoolMarks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        shader = (ROOT / "nxnative/dm/android/gfx_vk_shader.c").read_text(encoding="utf-8")
        exec_ = (ROOT / "nxnative/dm/android/gfx_vk_exec.c").read_text(encoding="utf-8")
        enum = exec_[exec_.index("enum { MARK_RECENT"):exec_.index("\n", exec_.index("enum { MARK_RECENT")) + 1]
        code = HARNESS.replace("SHADER", function(shader, "static int spv_pool_mark(")).replace("WMEM", function(exec_, "static int spv_wmem(")).replace(
            "MARKS", enum + function(exec_, "static int pool_mark_scan("))
        target = ROOT / "target/tests"; target.mkdir(parents=True, exist_ok=True)
        cls.dir = tempfile.TemporaryDirectory(prefix="pool-mark-", dir=target); d = Path(cls.dir.name)
        (d / "h.c").write_text(code, encoding="utf-8"); cls.exe = d / "h"
        subprocess.run(["cc", "-std=gnu11", "-Wall", "-Wextra", "-Werror", "-fsanitize=address,undefined", str(d / "h.c"), "-o", str(cls.exe)], check=True)

    @classmethod
    def tearDownClass(cls):
        cls.dir.cleanup()

    def mark(self, spv, np=65536):
        d = Path(self.dir.name); (d / "in.spv").write_bytes(spv)
        run = subprocess.run([str(self.exe), str(d / "in.spv"), str(d / "out.spv"), "16", str(np)], capture_output=True, text=True, timeout=30)
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
        out = run.stdout.split(" ", 1)
        return int(out[0]), out[1].strip(), d / "out.spv"

    def test_fold(self):
        self.assertEqual(subprocess.run([str(self.exe), "scan"], check=True, capture_output=True, text=True, timeout=30).stdout.strip(), "ok")

    def test_clamped_read_is_marked(self):
        n, var, out = self.mark(module())
        self.assertEqual(n, 1)   # one store after the one access chain; the harness checks spv_wmem leaves the marks' buffer out
        w = struct.unpack(f"<{len(out.read_bytes()) // 4}I", out.read_bytes())
        self.assertIn((4 << 16 | 71, int(var), 33, 8), [tuple(w[i:i + 4]) for i in range(5, len(w) - 3)])   # Binding POOL_MARK_B
        if shutil.which("spirv-val"):
            r = subprocess.run(["spirv-val", "--target-env", "vulkan1.1", str(out)], capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)

    def test_unprovable_index_or_module_is_refused(self):
        self.assertEqual(self.mark(module(clamped=False))[0], -1)    # an index not clamped to the table: robust buffer access is off, so no store
        self.assertEqual(self.mark(module(), np=4096)[0], -1)         # the clamp is past this table's length
        self.assertEqual(self.mark(module(version=0x00010400))[0], -1)   # 1.4: the entry points would have to list the buffer
        n, why, _ = self.mark(module()[:20] + module()[20:].replace(struct.pack("<I", 5 << 16 | 65), struct.pack("<I", 5 << 16 | 66)))   # (InBoundsAccessChain: the same)
        self.assertEqual(n, 1, why)

    def test_switch_gates_every_change(self):
        shader = (ROOT / "nxnative/dm/android/gfx_vk_shader.c").read_text(encoding="utf-8")
        exec_ = (ROOT / "nxnative/dm/android/gfx_vk_exec.c").read_text(encoding="utf-8")
        q = function(shader, "static vkshader_t *vk_shader_q(uint64_t key, int stage, int quiet) {")
        self.assertIn("if (!pfail && ntv && vk_pool_mark_on())", q)   # the rewrite only with the switch
        self.assertIn("if (vk_pool_mark_on()) { lb[nb]", function(shader, "VkDescriptorSetLayout vk_tbl_dsl("))
        init = function(exec_, "static void pool_mark_init(")
        self.assertIn("if (!vk_pool_mark_on()) { __atomic_store_n(&g_mk.on, 0", init)
        for name in ("static int pool_mark_warm(", "static int pool_mark_seen(", "static int pool_mark_write("):
            self.assertIn("if (!pool_mark_ready()", function(exec_, name))   # each answers as before without the marks
        self.assertIn("== 1 ? __atomic_load_n(&g_mk.seq", function(exec_, "static uint64_t pool_mark_gen("))
        # no dynamic descriptor in set 0: Adreno then copies the whole set into the command stream at each bind (pm2: 4.5 GB of command memory)
        self.assertIn("VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,", function(shader, "VkDescriptorSetLayout vk_tbl_dsl("))
        self.assertIn("&tbl, 0, NULL);", function(exec_, "static int prepare_draw("))


if __name__ == "__main__":
    unittest.main()
