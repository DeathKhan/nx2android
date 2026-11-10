"""Native image retention uses actual snapshot, lookup, pool removal and release helpers."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
import sys; sys.path.insert(0, str(Path(__file__).resolve().parent))
import host_cc   # noqa: E402
try:
    from tests.test_vk_texture_registration import function
except ModuleNotFoundError:
    from test_vk_texture_registration import function

ROOT=Path(__file__).resolve().parents[1]


@unittest.skipUnless(shutil.which("cc"), "host C compiler required")
class TexturePins(unittest.TestCase):
    @unittest.skipUnless(host_cc.sanitizers(), 'a host C compiler with ASan and UBSan is required (MinGW has neither)')
    def test_native_images_survive_parked_snapshots_and_key_reuse(self):
        s=(ROOT/"nxnative/dm/android/gfx_vk_tex.c").read_text(encoding='utf-8')
        declarations=s[s.index("typedef struct { uint64_t pool;"):s.index("// The sRGB / UNORM twin")]
        start=s.index("typedef struct tex_pin_image")
        pins=s[start:s.index("static vkimg_t *vk_tex_(const tex_t *t);",start)]
        reclaimed=next(x for x in s.splitlines() if x.startswith("static nvn_hmap g_reclaimed"))   # evict_overlaps' memory of taken-over targets (vk_tex_pool_ok reads it)
        helpers="\n".join(function(s,d) for d in (
            "static void img_drop(", "void vk_tex_drop_pool(", "int vk_tex_pool_ok(",
            "size_t vk_img_list(", "void vk_img_evict("))
        code=r"""
#define _POSIX_C_SOURCE 200809L
#include <assert.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "nvn_obj.h"
#define vk_tex_diag_native(...) ((void)0)
#include "nvn_hmap.h"
typedef struct vkimg {
 uint32_t pins;int orphan,packaged,pkg_released,looked;uint32_t pool_ck;
 void *lin_mem,*pkg_mem,*sub,*img,*mem;size_t lin_len,pkg_len;
 uint64_t vkbytes;int64_t size;uint64_t pool;int64_t off;uint64_t used;
} vkimg_t;
typedef struct {int depth,stencil;} vkfmt_t;
typedef struct {nvn_hmap *h;uint32_t i;vkimg_t *im;} vk_img_ref;
static uint64_t g_img_n,g_img_bytes,g_img_vkbytes,g_img_evicted,g_tex_gen=1;
uint64_t g_tex_gen_why[4];   // gfx_vk_tex.c: generation bumps by cause (img_drop counts orphans)
static unsigned freed,view_freed,alias_unlinked,mincores,holding_calls,dirty_released;
static int storage_live=1;
static vkfmt_t f;
static const vkfmt_t *vk_fmt(int fmt){f.depth=fmt==0x35;f.stencil=0;return &f;}
static void vk_lrz_forget(vkimg_t *im){assert(!im->pins);}
static void nvn_dirty_release(void *p,size_t n){assert(p&&n);dirty_released++;}
static void alias_unlink(vkimg_t *im){(void)im;alias_unlinked++;}
static void vk_tex_view_drop_all(vkimg_t *im){assert(!im->pins);view_freed++;}
static void vk_defer_free_image(void *im,void *mem){(void)im;(void)mem;freed++;}
static void vk_defer_free_image_sub(void *im,void *sub){(void)im;(void)sub;freed++;}
static void *vk_pool_holding(uint64_t a,uint64_t n){(void)a;(void)n;holding_calls++;return storage_live?(void *)1:NULL;}
static uint32_t vk_pool_gen(void){return 1;}
static int range_mapped(const uint8_t *p,size_t n){(void)p;(void)n;mincores++;return storage_live;}
#define die(...) do{fprintf(stderr,__VA_ARGS__);abort();}while(0)
DECLARATIONS
RECLAIMED
static void img_drop(vkimg_t *im);
PINS
HELPERS
static tex_t texture(uint64_t serial,int64_t offset,int flags){
 tex_t t={0};t.magic=MAGIC('T','E','X',0);t.serial=serial;t.pool=(pool_t *)(uintptr_t)0x1000;
 t.off=offset;t.mem=(uint8_t *)(uintptr_t)(0x20000+offset);t.L.size=64;
 t.d.target=NVN_TEXTURE_TARGET_2D;t.d.format=0x25;t.d.flags=flags;t.d.width=t.d.height=4;
 t.d.depth=t.d.levels=1;return t;
}
static akey key(const tex_t *t){akey k;tex_storage_key(&k,t,vk_fmt(t->d.format));return k;}
static vkimg_t *new_image(const tex_t *t){
 vkimg_t *im=calloc(1,sizeof *im);assert(im);im->size=64;im->vkbytes=256;im->pool=(uintptr_t)t->pool;
 im->off=t->off;im->looked=1;im->img=im;g_img_n++;g_img_bytes+=64;g_img_vkbytes+=256;
 akey k=key(t);if(t->pool)nvn_hm_put(&g_by_key,&k,im);else nvn_hm_put(&g_by_serial,&t->serial,im);
 return im;
}
static vkimg_t *find(const tex_t *t,tex_pin_image **out){
 akey k=key(t);pthread_mutex_lock(&g_img_mu);vkimg_t *im=tex_image_find(t,&k,out);pthread_mutex_unlock(&g_img_mu);return im;
}
static void cleanup_map(nvn_hmap *h){free(h->keys);free(h->vals);}
int main(void){
 tex_t a=texture(10,0,0),copy;vkimg_t *old=new_image(&a);
 mtl_tex_snapshot(&copy,&a);assert(!memcmp(&copy,&a,sizeof a)&&old->pins==1);
 mtl_tex_snapshot(&copy,&copy);mtl_tex_pin(a.serial);assert(old->pins==1); // 3 owners, one native alias
 mtl_tex_unpin(a.serial);assert(old->pins==1&&!freed);
 vk_img_ref *list=NULL;size_t cap=0;assert(vk_img_list(&list,&cap)==0);free(list);
 old->lin_mem=(void *)1;old->lin_len=64;old->packaged=1;old->pkg_mem=(void *)2;old->pkg_len=64;
 unsigned before=freed;vk_tex_drop_pool(a.pool);
 assert(dirty_released==2&&!old->lin_mem&&!old->pkg_mem&&old->pkg_released);
 assert(!g_by_key.n&&old->orphan&&old->pins==1&&freed==before&&!view_freed&&g_tex_gen_why[0]==1);
 // A new texture at the same address gets a distinct image; queued old metadata sees old contents.
 tex_t b=a;b.serial=20;vkimg_t *next=new_image(&b);assert(find(&a,NULL)==old&&find(&b,NULL)==next);
 storage_live=0;unsigned checked=holding_calls;
 assert(vk_tex_pool_ok(&a)==2&&holding_calls==checked&&!mincores);
 assert(vk_tex_pool_ok(&b)==0&&holding_calls==checked+1);
 mtl_tex_unpin(a.serial);assert(freed==before&&old->pins==1);
 mtl_tex_unpin(a.serial);assert(freed==before+1&&view_freed==1&&dirty_released==2); // final release uses deferred destruction
 assert(find(&b,NULL)==next);vk_tex_drop_pool(b.pool);assert(freed==before+2);
 storage_live=1;
 // Snapshot before native image creation protects the image when it is first resolved.
 tex_t c=texture(30,64,0);mtl_tex_snapshot(&copy,&c);tex_pin_image *pin=NULL;
 assert(!find(&c,&pin)&&pin&&!pin->im);
 vkimg_t *late=new_image(&c);pthread_mutex_lock(&g_img_mu);tex_image_pin_resolved(pin,late);tex_image_pin_resolved(pin,late);pthread_mutex_unlock(&g_img_mu);
 assert(late->pins==1);vk_tex_drop_pool(c.pool);before=freed;mtl_tex_unpin(c.serial);assert(freed==before+1);
 // IMAGE usage intent shares the same storage and rendered contents. Actual storage
 // preflight upgrades that backing; registration must not create an empty second image.
 tex_t x=texture(40,128,0),y=x;y.d.flags|=NVN_TEXTURE_FLAGS_IMAGE;
 akey xkey=key(&x),ykey=key(&y);assert(!memcmp(&xkey,&ykey,sizeof xkey));
 vkimg_t *xi=new_image(&x);mtl_tex_snapshot(&copy,&x);mtl_tex_snapshot(&copy,&y);
 assert(xi->pins==1&&find(&x,NULL)==xi&&find(&y,NULL)==xi);
 vk_tex_drop_pool(x.pool);before=freed;mtl_tex_unpin(x.serial);assert(freed==before);
 mtl_tex_unpin(y.serial);assert(freed==before+1);
 // Real layout differences remain distinct and every pinned alias survives.
 x=texture(41,128,0);y=x;y.d.flags|=NVN_TEXTURE_FLAGS_LINEAR;
 xi=new_image(&x);vkimg_t *yi=new_image(&y);mtl_tex_snapshot(&copy,&x);mtl_tex_snapshot(&copy,&y);
 assert(xi->pins==1&&yi->pins==1&&find(&x,NULL)==xi&&find(&y,NULL)==yi);
 vk_tex_drop_pool(x.pool);before=freed;mtl_tex_unpin(x.serial);assert(freed==before);
 mtl_tex_unpin(y.serial);assert(freed==before+2);
 // Separate serials sharing one native image hold independent references.
 x=texture(50,256,0);y=x;y.serial=51;xi=new_image(&x);
 mtl_tex_snapshot(&copy,&x);mtl_tex_snapshot(&copy,&y);assert(xi->pins==2);
 vk_tex_drop_pool(x.pool);before=freed;mtl_tex_unpin(x.serial);assert(freed==before&&xi->pins==1);
 mtl_tex_unpin(y.serial);assert(freed==before+1);
 // Non-pool images use their serial mapping, and zero serials create no retention entry.
 x=texture(60,0,0);x.pool=NULL;xi=new_image(&x);mtl_tex_snapshot(&copy,&x);
 assert(find(&x,NULL)==xi&&xi->pins==1);nvn_hm_del(&g_by_serial,&x.serial);img_drop(xi);
 before=freed;mtl_tex_unpin(x.serial);assert(freed==before+1);
 x.serial=0;mtl_tex_snapshot(&copy,&x);mtl_tex_pin(0);mtl_tex_unpin(0);
 assert(!g_tex_pins.n&&!g_by_key.n&&!g_by_serial.n&&!g_img_n&&!g_img_bytes&&!g_img_vkbytes);
 assert(!g_reclaimed.n&&!g_img_reclaimed);   // nothing here takes over a render target
 cleanup_map(&g_tex_pins);cleanup_map(&g_by_key);cleanup_map(&g_by_serial);
 puts("native texture pins: exact aliases, queued lifetime, pool detach, key reuse and final release passed");
}
"""
        for k,v in (("DECLARATIONS",declarations),("RECLAIMED",reclaimed),("PINS",pins),("HELPERS",helpers)):
            code=code.replace(k,v)
        target=ROOT/"target/tests";target.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="texture-pins-",dir=target) as tmp:
            src=Path(tmp)/"test.c";exe=Path(tmp)/"test";src.write_text(code, encoding='utf-8')
            subprocess.run(["cc","-std=c11","-O1","-g","-Wall","-Wextra","-Werror",
                            "-Wno-misleading-indentation","-fsanitize=address,undefined",
                            "-I"+str(ROOT/"nxnative/dm/gfx/nvn"),
                            "-I"+str(ROOT/"nxnative/dm/gfx/nvn/include"),
                            str(src),"-o",str(exe),"-lpthread"],check=True)
            subprocess.run([str(exe)],check=True)


if __name__=="__main__":
    unittest.main()
