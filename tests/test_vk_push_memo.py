"""Graphics push writes invalidate the actual descriptor memo dependency."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class PushMemo(unittest.TestCase):
    @unittest.skipUnless(shutil.which("cc"), "a host C compiler is required")
    def test_changed_push_bytes_and_cached_shader_dependency(self):
        dm = ROOT / "nxnative/dm"
        source = (dm / "android/gfx_vk_exec.c").read_text()
        header = (dm / "android/gfx_vk.h").read_text()
        obj = (dm / "gfx/nvn/nvn_obj.h").read_text()
        abi = (dm / "gfx/nvn/dmsh_abi.h").read_text()
        declarations = []
        start = header.index("enum { RK_SUPPORT")
        declarations.append(header[start:header.index("};", start) + 2])
        declarations.append(next(x for x in header.splitlines() if x.startswith("typedef struct ") and x.endswith(" dmsv_bind;")))
        declarations.append(next(x.split("   //")[0] for x in obj.splitlines() if "} p_push_const;" in x))
        declarations.append(next(x for x in abi.splitlines() if x.startswith("#define DMSH_PUSH_BANK ")))
        helpers = []
        for name in ("static void push_constants(", "static uint32_t storage_dynamic_offset(", "static int set_memo_get("):
            start = source.index(name)
            helpers.append(source[start:source.index("\n}\n", start) + 3])
        memo = next(x for x in source.splitlines() if x.startswith("typedef struct ") and x.endswith(" set_memo_t;"))
        start = source.index("        case OP_PUSH_CONST:")
        dispatch = source[start:source.index("        case OP_MULTI_DRAW:", start)]
        code = r"""
#include <vulkan/vulkan_core.h>
#include <assert.h>
#include <stdint.h>
#include <string.h>
DECLARATIONS
typedef struct { int nbind, ndyn, setmemo; const dmsv_bind *b; uint8_t dyn[96]; } vkshader_t;
enum { MEMO_CB = 16, OP_PUSH_CONST = 17 };
MEMO_TYPE
static struct { uint8_t push[2][256]; vkshader_t *sh[3]; } S;
static set_memo_t g_smemo[3];
static uint64_t g_set_epoch=9, g_bind_gen[6], g_ssbo_gen[6], g_ubo_gen[6], g_tex_gen, g_samp_gen, g_ubd_gen, g_num_um, g_num_im;
static uint32_t g_draw_binst; static int g_buffer_memo=1;
static struct { struct { VkPhysicalDeviceLimits limits; } props; } V;
static VkDescriptorBufferInfo g_draw_claim;
static VkBuffer vk_zero_buffer(void) { return (VkBuffer)(uintptr_t)1; }
static uint32_t vk_pool_gen(void) { return 0; }
static void cbuf_desc(int st, vkshader_t *s, const dmsv_bind *b, VkDescriptorBufferInfo *d) {
    (void)st; (void)s; (void)b; (void)d; assert(0 && "fixture has no cbuf");
}
static void sbuf_desc(int st, const dmsv_bind *b, VkDescriptorBufferInfo *d) { (void)st; (void)b; (void)d; assert(0 && "fixture has no sbuf"); }
static void drv_shared(int st, int bank, VkBuffer *buf, VkDeviceSize *off, int dyn) { (void)st; (void)bank; (void)buf; (void)off; (void)dyn; assert(0 && "fixture has no driver bank"); }
HELPERS
static void replay_push(const void *a) { switch(OP_PUSH_CONST) {
DISPATCH
} }
static void cache(int st, vkshader_t *s) { g_smemo[st]=(set_memo_t){.s=s,.epoch=g_set_epoch}; }
static int hit(int st, vkshader_t *s) { uint32_t offsets[96]; return set_memo_get(st,s,offsets); }
int main(void) {
    const dmsv_bind push_binding={.kind=RK_DRV,.reg=DMSH_PUSH_BANK};
    const dmsv_bind other_bank={.kind=RK_DRV,.reg=0};
    const dmsv_bind other_kind={.kind=RK_C1,.reg=DMSH_PUSH_BANK};
    vkshader_t push={.nbind=1,.b=&push_binding}, plain={.nbind=1,.b=&other_bank}, wrong={.nbind=1,.b=&other_kind};
    _Alignas(p_push_const) unsigned char bytes[sizeof(p_push_const)+256]={0};
    p_push_const *r=(p_push_const *)bytes;
    cache(0,&push); cache(1,&plain); cache(2,&wrong);
    assert(hit(0,&push)&&hit(1,&plain)&&hit(2,&wrong));
    S.sh[0]=&plain;  // Changed current shader must not hide the old memo's dependency.
    r->compute=0; r->offset=17; r->size=3;
    r->data[0]=0x91; r->data[1]=0x42; r->data[2]=0xe3;
    replay_push(r);
    assert(!hit(0,&push)&&hit(1,&plain)&&hit(2,&wrong));
    assert(!S.push[0][16]&&!S.push[0][20]&&!memcmp(S.push[0]+17,r->data,3));
    cache(0,&push);
    replay_push(r);  // Same bytes preserve the memo, including a partial write.
    assert(hit(0,&push));
    r->compute=1; r->data[1]=0x22; replay_push(r);
    assert(hit(0,&push)&&hit(1,&plain)&&hit(2,&wrong));
    assert(S.push[0][18]==0x42&&S.push[1][18]==0x22);
    r->compute=0; r->offset=256; r->size=0; replay_push(r);
    assert(hit(0,&push));
    unsigned char saved[sizeof S.push]; memcpy(saved,S.push,sizeof saved);
    r->compute=2; r->offset=0; r->size=1; replay_push(r);
    r->compute=0; r->offset=255; r->size=2; replay_push(r);
    r->offset=1; r->size=UINT32_MAX; replay_push(r); // Old offset+size check wrapped.
    r->offset=UINT16_MAX; r->size=1; replay_push(r);
    assert(hit(0,&push)&&!memcmp(saved,S.push,sizeof saved));
    cache(0,&push); cache(1,&push); cache(2,&push);
    r->compute=0; r->offset=255; r->size=1; r->data[0]=0x7d; replay_push(r);
    assert(S.push[0][255]==0x7d&&!hit(0,&push)&&!hit(1,&push)&&!hit(2,&push));
    cache(0,&plain); S.sh[0]=&push;
    r->offset=1; r->data[0]=0x61; replay_push(r);
    assert(hit(0,&plain)&&!hit(0,&push)); // Switching back must collect the new bank.
    return 0;
}
"""
        for name, value in (("DECLARATIONS", "\n".join(declarations)), ("MEMO_TYPE", memo),
                            ("HELPERS", "\n".join(helpers)), ("DISPATCH", dispatch)):
            code = code.replace(name, value)
        target = ROOT / "target/tests"
        target.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="push-memo-", dir=target) as folder:
            src, exe = Path(folder) / "check.c", Path(folder) / "check"
            src.write_text(code)
            subprocess.run(["cc", "-std=c11", "-Wall", "-Wextra", "-Werror",
                            "-fsanitize=address,undefined",
                            "-I" + str(dm / "third_party/vulkan-headers/include"),
                            str(src), "-o", str(exe)], check=True)
            subprocess.run([str(exe)], check=True, timeout=15)


if __name__ == "__main__":
    unittest.main()
