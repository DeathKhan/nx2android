"""Run actual codec publication/copy/write helpers, including concurrent generation changes."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
ROOT=Path(__file__).resolve().parents[1]
def function(s,decl):
    a=s.index(decl);b=s.index("{",a)+1;depth=1
    while depth:
        depth+=(s[b]=="{")-(s[b]=="}");b+=1
    return s[a:b]
class Nv12Snapshot(unittest.TestCase):
    @unittest.skipUnless(shutil.which("cc"),"host compiler required")
    def test_generation_bytes_lifetime_and_lock(self):
        s=(ROOT/"nxnative/dm/host/hle_codec_video.c").read_text()
        methods="\n".join(function(s,x) for x in ("static uint32_t native_stride(","static uint64_t native_size(uint32_t w, uint32_t h) {",
            "static void last_nv12_set(","static void last_nv12_forget(","int dm_codec_last_nv12(","int dm_codec_copy_last_nv12(","static void pool_native_write("))
        code=r"""
#define _POSIX_C_SOURCE 200809L
#include <assert.h>
#include <errno.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <stdatomic.h>
#include <sched.h>
#define CV_NATIVE_ALIGN 0x4000ull
#define CV_FMT_NV12 3
#define DM_CODEC_CS_BT709 1
#define DM_CODEC_CS_FULL 2
enum {VPX_CS_UNKNOWN,VPX_CS_BT_709,VPX_CR_FULL_RANGE=1};
typedef struct {int cs,range;uint8_t value;} vpx_image_t;
typedef struct {uint16_t w,h;uint32_t fmt;uint8_t *mem;uint64_t size;_Atomic int native;} cv_pool;
static struct {uint64_t addr,gen;uint32_t w,h;int cs;} g_last_nv12;
static pthread_mutex_t g_last_nv12_mu=PTHREAD_MUTEX_INITIALIZER;
static unsigned copies,writes,rgba_writes;
static void lock_held(void){int r=pthread_mutex_trylock(&g_last_nv12_mu);assert(r==EBUSY);}
static void *checked_copy(void *d,const void *s,size_t n){if(n==24){lock_held();copies++;}return memcpy(d,s,n);}
static void md_nv12(const vpx_image_t *im,uint8_t *d,uint32_t w,uint32_t h){lock_held();writes++;memset(d,im->value,(size_t)w*h*3/2);}
static void md_rgba(const vpx_image_t *im,uint8_t *d,uint32_t w,uint32_t h){(void)im;rgba_writes++;memset(d,0xee,(size_t)w*h*4);}
#define memcpy checked_copy
#define LOG(...) ((void)0)
METHODS
#undef memcpy
static cv_pool pool;static atomic_int finished;
static void *writer(void *arg){(void)arg;for(unsigned i=1;i<=2000;i++){vpx_image_t im={.value=(uint8_t)i};pool_native_write(&pool,0,&im);if(!(i%8))sched_yield();}atomic_store(&finished,1);return NULL;}
int main(void){
 uint8_t mem[32768]={0},out[32];pool=(cv_pool){.w=4,.h=4,.fmt=CV_FMT_NV12,.mem=mem,.size=sizeof mem};
 vpx_image_t im={.cs=VPX_CS_BT_709,.range=VPX_CR_FULL_RANGE,.value=0x11};
 assert(!dm_codec_copy_last_nv12(1,out,sizeof out));
 pool_native_write(&pool,0,&im);uint64_t addr,gen;uint32_t w,h;int cs;
 assert(dm_codec_last_nv12(&addr,&w,&h,&gen,&cs)&&addr==(uintptr_t)mem&&w==4&&h==4&&gen==1&&cs==3);
 memset(out,0xa5,sizeof out);assert(!dm_codec_copy_last_nv12(gen,out,23)&&out[0]==0xa5&&!copies);
 assert(!dm_codec_copy_last_nv12(gen+1,out,sizeof out)&&!dm_codec_copy_last_nv12(0,out,sizeof out));
 assert(dm_codec_copy_last_nv12(gen,out,24)&&copies==1);for(int i=0;i<24;i++)assert(out[i]==0x11);assert(out[24]==0xa5);
 im.value=0x22;pool_native_write(&pool,0,&im);assert(!dm_codec_copy_last_nv12(gen,out,sizeof out));
 assert(dm_codec_last_nv12(&addr,&w,&h,&gen,&cs)&&gen==2);assert(dm_codec_copy_last_nv12(gen,out,sizeof out)&&out[0]==0x22);
 // A native-image request makes the slot RGBA; it must not remain advertised as NV12.
 pool.native=1;pool_native_write(&pool,0,&im);assert(rgba_writes==1&&!dm_codec_last_nv12(&addr,&w,&h,&gen,&cs));
 assert(!dm_codec_copy_last_nv12(2,out,sizeof out));pool.native=0;
 pool_native_write(&pool,0,&im);assert(dm_codec_last_nv12(&addr,&w,&h,&gen,&cs));last_nv12_forget(&pool);
 assert(!dm_codec_copy_last_nv12(gen,out,sizeof out)&&!dm_codec_last_nv12(&addr,&w,&h,&gen,&cs));
 // Generation metadata and bytes remain one snapshot while another thread repeatedly overwrites the same slot.
 g_last_nv12=(typeof(g_last_nv12)){0};pthread_t t;assert(!pthread_create(&t,NULL,writer,NULL));unsigned accepted=0;
 do {if(dm_codec_last_nv12(&addr,&w,&h,&gen,&cs)&&dm_codec_copy_last_nv12(gen,out,sizeof out)){
   for(int i=0;i<24;i++){assert(out[i]==(uint8_t)gen);}accepted++;
 }}while(!atomic_load(&finished));
 assert(!pthread_join(t,NULL));assert(dm_codec_last_nv12(&addr,&w,&h,&gen,&cs)&&gen==2000);
 assert(dm_codec_copy_last_nv12(gen,out,sizeof out));for(int i=0;i<24;i++)assert(out[i]==(uint8_t)2000);accepted++;
 assert(accepted&&writes>=2000);return 0;
}
"""
        code=code.replace("METHODS",methods)
        dest=ROOT/"target/tests";dest.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="codec-nv12-",dir=dest) as folder:
            c=Path(folder)/"check.c";exe=Path(folder)/"check";c.write_text(code)
            subprocess.run(["cc","-std=gnu11","-Wall","-Wextra","-Werror","-fsanitize=address,undefined","-pthread",str(c),"-o",str(exe)],check=True)
            subprocess.run([str(exe)],check=True,timeout=15)
if __name__=="__main__":unittest.main()
