"""Run texture registration and historical lookup from the Android source."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
import sys; sys.path.insert(0, str(Path(__file__).resolve().parent))
import host_cc   # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def function(source, declaration):
    start = source.index(declaration)
    pos = source.index('{', start) + 1
    depth = 1
    while depth:
        depth += (source[pos] == '{') - (source[pos] == '}')
        pos += 1
    return source[start:pos]


@unittest.skipUnless(shutil.which("cc"), "host C compiler required")
class TextureRegistration(unittest.TestCase):
    @unittest.skipUnless(host_cc.sanitizers(), 'a host C compiler with ASan and UBSan is required (MinGW has neither)')
    def test_registry_listing_is_bounded_metadata_copy(self):
        source = (ROOT / "nxnative/dm/android/gfx_vk_tex.c").read_text(encoding='utf-8')
        common = (ROOT / "nxnative/dm/android/gfx_common.c").read_text(encoding='utf-8')
        actual = function(source, "void vk_registry_texlist(")
        route = function(common, "void dm_gfx_texlist(")
        code = r"""
#define _POSIX_C_SOURCE 200809L
#include <assert.h>
#include <pthread.h>
#include <stdint.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "nvn_obj.h"
typedef struct {tex_t t;view_t v;uint8_t ok,has_v;uint64_t seq;} treg_ver;
typedef struct {treg_ver cur;} treg_t;
static treg_t *g_treg;static int g_ntreg;
static pthread_mutex_t g_rmu=PTHREAD_MUTEX_INITIALIZER;
static char logs[300000];static size_t used;static int mutate;
static void logger(const char *format,...){
 assert(pthread_mutex_trylock(&g_rmu)==0); // no logging under the registry mutex
 if(mutate){mutate=0;for(int i=1;i<g_ntreg;i++)g_treg[i].cur.t.serial=999999;}
 pthread_mutex_unlock(&g_rmu);
 va_list a;va_start(a,format);int n=vsnprintf(logs+used,sizeof logs-used,format,a);va_end(a);
 assert(n>=0&&(size_t)n<sizeof logs-used);used+=(size_t)n;logs[used++]='\n';logs[used]=0;
}
#define VKLOG(...) logger(__VA_ARGS__)
ACTUAL
ROUTE
int main(void){
 g_ntreg=400;g_treg=calloc((size_t)g_ntreg,sizeof *g_treg);assert(g_treg);
 g_treg[1].cur.ok=1;g_treg[1].cur.t.d.target=1;g_treg[1].cur.t.serial=11;
 treg_ver *v=&g_treg[2].cur;v->ok=v->has_v=1;v->seq=123;v->t.serial=22;
 v->t.d.target=4;v->t.d.width=80;v->t.d.height=40;v->t.d.depth=3;v->t.d.levels=2;v->t.d.format=0x32;
 v->t.L.layers=3;v->t.L.size=4096;v->t.L.layer_size=1024;v->t.off=0x2400;
 v->t.pool=(pool_t *)(uintptr_t)0x12340000;v->t.mem=(uint8_t *)(uintptr_t)0x23450000; // invalid to dereference: metadata only
 v->v.set_layers=1;v->v.min_layer=1;v->v.nlayers=2;v->v.set_format=1;v->v.format=0x33;
 g_treg[3].cur.t.d.target=4;g_treg[3].cur.t.serial=33; // unregistered excluded
 mutate=1;dm_gfx_texlist(4);
 assert(strstr(logs,"target 4: 1 entries")&&strstr(logs,"tid=2 ser=22 seq=123")&&!strstr(logs,"ser=999999")&&!strstr(logs,"tid=1")&&!strstr(logs,"tid=3"));
 assert(strstr(logs,"80x40x3 format 0x32 levels 2 layers 3")&&strstr(logs,"pool 0x12340000+0x2400"));
 assert(strstr(logs,"view target 0/0 format 1/0x33")&&strstr(logs,"layers 1/1+2"));
 used=0;logs[0]=0;
 for(int i=1;i<g_ntreg;i++){g_treg[i].cur.ok=1;g_treg[i].cur.has_v=0;g_treg[i].cur.t.d.target=4;g_treg[i].cur.t.serial=(uint64_t)i;}
 dm_gfx_texlist(4);assert(strstr(logs,"256 entries, truncated at 256")&&strstr(logs,"tid=256 ")&&!strstr(logs,"tid=257 "));
 used=0;logs[0]=0;dm_gfx_texlist(2);assert(strstr(logs,"target 2: 0 entries"));
 used=0;logs[0]=0;dm_gfx_texlist(-1);assert(strstr(logs,"unsupported NVN target -1"));
 free(g_treg);return 0;
}
"""
        code = code.replace("ACTUAL", actual).replace("ROUTE", route)
        target=ROOT/"target/tests";target.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="texture-list-",dir=target) as tmp:
            c,exe=Path(tmp)/"test.c",Path(tmp)/"test";c.write_text(code, encoding='utf-8')
            subprocess.run(["cc","-std=c11","-O1","-g","-Wall","-Wextra","-Werror","-Wno-address",
                            "-fsanitize=address,undefined","-pthread",
                            "-I"+str(ROOT/"nxnative/dm/gfx/nvn"),"-I"+str(ROOT/"nxnative/dm/gfx/nvn/include"),
                            str(c),"-o",str(exe)],check=True)
            subprocess.run([str(exe)],check=True,timeout=10)

    @unittest.skipUnless(host_cc.sanitizers(), 'a host C compiler with ASan and UBSan is required (MinGW has neither)')
    def test_identical_registration_preserves_generation_and_history(self):
        source = (ROOT / "nxnative/dm/android/gfx_vk_tex.c").read_text(encoding='utf-8')
        start = source.index("enum { TREG_OLD")
        declarations = source[start:source.index("int vk_tex_id_max", start)]
        history = source[source.index("static struct { uint64_t id, refs, sampler_seq, texture_seq; } *g_registry_jobs;"):source.index("void mtl_reg_tex(")]
        helper = history + "\n" + function(source, "void mtl_reg_tex(") + "\n" + function(source, "vkimg_t *vk_tex_by_id(")
        code = r"""
#define _POSIX_C_SOURCE 200809L
#include <assert.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "nvn_obj.h"
typedef struct {int unused;} vkimg_t;
DECLARATIONS
static uint64_t g_tex_gen=1,queued,job,g_sregs;
uint64_t g_tex_gen_why[4];
static pthread_mutex_t g_rmu=PTHREAD_MUTEX_INITIALIZER;
static __thread treg_ver t_treg;
static uint64_t nvn_jobs_queued(void){return queued;}
static uint64_t nvn_replaying_job(void){return job;}
static uint64_t pin_refs;
void mtl_tex_snapshot(tex_t *copy,const tex_t *source){*copy=*source;pin_refs++;}
void mtl_tex_unpin(uint64_t serial){(void)serial;assert(pin_refs);pin_refs--;}
#define die(...) do {fprintf(stderr,__VA_ARGS__);abort();} while(0)
#define vk_tex_diag_registration(...) ((void)0)
HELPERS
static tex_t texture(uint64_t serial){
 tex_t t={0};t.magic=MAGIC('T','E','X',0);t.serial=serial;t.d.width=640;t.d.height=360;
 t.d.depth=t.d.levels=1;t.d.target=1;t.d.format=0x25;t.L.size=640*360*4;t.mem=(uint8_t *)(uintptr_t)0x100000;
 return t;
}
static uint64_t serial_at(int id,uint64_t at){
 job=at;const tex_t *t=NULL;const view_t *v=NULL;
 return vk_tex_by_id(id,&v,&t)?t->serial:0;
}
int main(void){
 (void)g_sid_max;
 tex_t a=texture(10),b=texture(20);view_t v={0};v.set_levels=1;v.base_level=1;v.nlevels=2;
 uint64_t tx0=0,sx0=0;assert(vk_registry_snapshot(&tx0,&sx0)&&!tx0&&!sx0);
 uint64_t gen=g_tex_gen;mtl_reg_tex(-1,&a,NULL);assert(g_tex_gen==gen&&!g_treg);
 queued=1;mtl_reg_tex(7,&a,&v);assert(g_tex_gen==++gen&&g_tid_max==7);
 assert(g_treg[7].cur.seq==1&&!g_treg[7].old);
 uint64_t tx1=0,sx1=0;assert(vk_registry_snapshot(&tx1,&sx1)&&tx1==tx0+1);
 for(queued=2;queued<100;queued++)mtl_reg_tex(7,&a,&v);
 assert(g_tex_gen==gen&&g_treg[7].cur.seq==1&&!g_treg[7].old);
 assert(vk_registry_snapshot(&tx0,&sx0)&&tx0==tx1);
 job=99;assert(!vk_registry_snapshot(&tx0,&sx0));job=0;
 // A meaningful old version must survive arbitrarily many identical registrations.
 queued=100;mtl_reg_tex(7,&b,&v);assert(g_tex_gen==++gen);
 assert(g_treg[7].old&&g_treg[7].old[0].t.serial==10);
 for(queued=101;queued<200;queued++)mtl_reg_tex(7,&b,&v);
 assert(g_tex_gen==gen&&g_treg[7].cur.seq==100&&g_treg[7].old[0].t.serial==10);
 assert(serial_at(7,100)==10&&serial_at(7,101)==20&&serial_at(7,0)==20);
 // Every byte difference is conservatively a change, including creation/view flags,
 // serial, dimensions, pool/address/layout, swizzle and otherwise unused padding.
 queued=200;mtl_reg_tex(8,&a,&v);
 for(size_t i=0;i<sizeof a;i++){
  tex_t changed=a;((unsigned char *)&changed)[i]^=1;gen=g_tex_gen;
  mtl_reg_tex(8,&changed,&v);assert(g_tex_gen==gen+1);
  mtl_reg_tex(8,&a,&v);assert(g_tex_gen==gen+2);
 }
 for(size_t i=0;i<sizeof v;i++){
  view_t changed=v;((unsigned char *)&changed)[i]^=1;gen=g_tex_gen;
  mtl_reg_tex(8,&a,&changed);assert(g_tex_gen==gen+1);
  mtl_reg_tex(8,&a,&v);assert(g_tex_gen==gen+2);
 }
 gen=g_tex_gen;mtl_reg_tex(8,&a,NULL);assert(g_tex_gen==gen+1&&!g_treg[8].cur.has_v);
 mtl_reg_tex(8,&a,NULL);assert(g_tex_gen==gen+1);
 // Unregister/change behavior and queued historical lookup are unchanged.
 queued=201;mtl_reg_tex(7,NULL,NULL);assert(!g_treg[7].cur.ok);
 assert(serial_at(7,101)==20&&!serial_at(7,202));
 queued=202;mtl_reg_tex(7,&b,&v);assert(serial_at(7,203)==20);
 // Separate IDs and a registry growth cannot conflate equal snapshots.
 gen=g_tex_gen;mtl_reg_tex(3000,&a,&v);assert(g_tex_gen==gen+1&&g_tid_max==3000);
 assert(serial_at(3000,0)==10&&serial_at(8,0)==10);
 // Logical jobs retain every visible interval, independently of execution order.
 queued=300;mtl_reg_tex(4000,&a,&v);
 uint64_t before301=g_texture_registrations;
 g_sregs=7;vk_registry_job_pin(301);queued=301;mtl_reg_tex(4000,&b,&v);
 job=301;assert(vk_registry_snapshot(&tx0,&sx0)&&tx0==before301&&sx0==7);
 job=0;assert(vk_registry_snapshot(&tx1,&sx1)&&tx1==before301+1&&sx1==7);
 // Multiple changed registrations without another enqueue still have distinct identities.
 mtl_reg_tex(4000,&a,&v);mtl_reg_tex(4000,&b,&v);
 assert(vk_registry_snapshot(&tx1,&sx1)&&tx1==before301+3);
 job=301;assert(vk_registry_snapshot(&tx0,&sx0)&&tx0==before301);job=0;
 g_sregs=9;vk_registry_job_pin(302);queued=302;tex_t c=texture(30);mtl_reg_tex(4000,&c,&v);
 for(uint64_t j=303;j<323;j++){vk_registry_job_pin(j);queued=j;tex_t t=texture(j);mtl_reg_tex(4000,&t,&v);}
 assert(serial_at(4000,301)==10&&serial_at(4000,302)==20&&serial_at(4000,303)==30);
 assert(serial_at(4000,317)==316&&serial_at(4000,322)==321&&serial_at(4000,0)==322);
 assert(pin_refs==22);
 uint64_t samp;job=301;assert(registry_job_sampler_seq(&samp)&&samp==7);
 job=302;assert(registry_job_sampler_seq(&samp)&&samp==9);
 job=0;assert(!registry_job_sampler_seq(&samp));
 vk_registry_job_pin(301);vk_registry_job_unpin(301);assert(serial_at(4000,301)==10&&pin_refs==22);
 vk_registry_job_unpin(317);assert(pin_refs==21&&serial_at(4000,301)==10&&serial_at(4000,302)==20);
 vk_registry_job_unpin(301);assert(pin_refs==20);
 for(uint64_t j=302;j<323;j++)if(j!=317)vk_registry_job_unpin(j);
 assert(!pin_refs&&!g_registry_history&&!g_registry_njobs&&!g_treg[4000].history);
 // IDs absent at enqueue, and unregister intervals, stay absent rather than seeing future data.
 g_sregs=0;vk_registry_job_pin(400);job=400;assert(registry_job_sampler_seq(&samp)&&samp==0);queued=400;mtl_reg_tex(5000,&a,NULL);
 assert(!serial_at(5000,400));
 vk_registry_job_pin(401);queued=401;mtl_reg_tex(5000,NULL,NULL);
 assert(serial_at(5000,401)==10);
 vk_registry_job_pin(402);queued=402;mtl_reg_tex(5000,&b,NULL);
 assert(!serial_at(5000,402)&&serial_at(5000,403)==20);
 vk_registry_job_unpin(401);assert(!serial_at(5000,400)&&!serial_at(5000,402));
 vk_registry_job_unpin(402);vk_registry_job_unpin(400);
 assert(!pin_refs&&!g_registry_history&&!g_registry_njobs);
 for(int i=0;i<g_ntreg;i++)free(g_treg[i].old);free(g_treg);free(g_registry_jobs);
 puts("texture registrations: identical snapshots and out-of-order retained versions passed");
}
"""
        code = code.replace("DECLARATIONS", declarations).replace("HELPERS", helper)
        target = ROOT / "target/tests"
        target.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="texture-registration-", dir=target) as tmp:
            src = Path(tmp) / "test.c"
            exe = Path(tmp) / "test"
            src.write_text(code, encoding='utf-8')
            subprocess.run(["cc", "-std=c11", "-O1", "-g", "-Wall", "-Wextra", "-Werror",
                            "-fsanitize=address,undefined", "-Wno-misleading-indentation",
                            "-I" + str(ROOT / "nxnative/dm/gfx/nvn"),
                            "-I" + str(ROOT / "nxnative/dm/gfx/nvn/include"),
                            str(src), "-o", str(exe), "-lpthread"], check=True)
            subprocess.run([str(exe)], check=True)


if __name__ == "__main__":
    unittest.main()
