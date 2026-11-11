"""Counted draws preserve cold and already-queued pipeline work."""
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
    brace = source.index('{', start)
    depth, pos = 1, brace + 1
    while depth:
        depth += (source[pos] == '{') - (source[pos] == '}')
        pos += 1
    return source[start:pos]


class AsyncQueries(unittest.TestCase):
    @unittest.skipUnless(shutil.which('cc'), 'a host C compiler is required')
    @unittest.skipUnless(host_cc.sanitizers(), 'a host C compiler with ASan and UBSan is required (MinGW has neither)')
    def test_counted_pipeline_lifecycle(self):
        source = (ROOT / 'nxnative/dm/android/gfx_vk_exec.c').read_text(encoding='utf-8')
        core = (ROOT / 'nxnative/dm/android/gfx_vk_core.c').read_text(encoding='utf-8')
        worker = function(source, 'static void *async_pipe_thread(')
        self.assertIn('async_pipe_complete(&k, p);', worker)
        code = r'''
#include <assert.h>
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>
#include <string.h>
#include <pthread.h>
#include <stdatomic.h>
#include <sched.h>
#include <errno.h>
typedef void *VkPipeline;
typedef int VkPipelineLayout;
#define VK_NULL_HANDLE NULL
#define PIPE_PENDING ((VkPipeline)(uintptr_t)2)
enum { RD_SSBO=1, RD_STORAGE=2, RD_STORAGE_TBUF=3, NVN_TEXTURE_FLAGS_LINEAR_RENDER_TARGET=4, APQ=256 };
typedef struct { int flags,desc; } dmsv_bind;
typedef struct { int xfb,nbind; dmsv_bind *b; } vkshader_t;
typedef struct { uint64_t key[3]; uint32_t alt; int raster_discard; } pkey;
typedef struct { uint64_t rt_frames; int w,h,nvn_format; } vkimg_t;
typedef struct { VkPipeline values[32]; } nvn_hmap;
static nvn_hmap g_pipes;
static pthread_mutex_t g_pipe_mu=PTHREAD_MUTEX_INITIALIZER,g_apq_mu=PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t g_pipe_ready=PTHREAD_COND_INITIALIZER,g_apq_cv=PTHREAD_COND_INITIALIZER;
static pkey g_last_k,g_apq[APQ]; static VkPipeline g_last_p;
static int g_apq_h,g_apq_t,g_apq_on,g_async_pipe=1,g_prof_on,g_occ_counting;
static uint64_t g_prof_ns[1];
static struct { vkshader_t *sh[5]; int ncol,rt_has_col[8]; struct { struct { int flags; } d; } rt_col[8]; } S;
static struct { vkimg_t *im; } P_att[9];
static struct { int pipes_built; } g_vkst;
static struct { int dev; } V;
static int builds,records,destroys,creates;
static _Atomic int waits,returned;
GETTER
static VkPipeline nvn_hm_get(nvn_hmap *m,const pkey *k){assert(k->key[0]<32);return m->values[k->key[0]];}
static void nvn_hm_put(nvn_hmap *m,const pkey *k,VkPipeline p){assert(k->key[0]<32);m->values[k->key[0]]=p;}
static void pipe_record(const pkey *k){(void)k;records++;}
static VkPipeline build_pipeline(const pkey *k,VkPipelineLayout l,vkshader_t *const stg[3]){(void)l;(void)stg;builds++;return (VkPipeline)(uintptr_t)(100+k->key[0]);}
static void vkDestroyPipeline(int dev,VkPipeline p,const void *a){(void)dev;(void)p;(void)a;destroys++;}
static uint64_t tick(void){return 0;}
static void log_ignore(const char *fmt,...){(void)fmt;}
#define VKLOG(...) log_ignore(__VA_ARGS__)
static void *async_pipe_thread(void *arg){(void)arg;abort();}
static int fail_create(pthread_t *t,const pthread_attr_t *a,void *(*f)(void *),void *arg){(void)t;(void)a;(void)f;(void)arg;creates++;return EAGAIN;}
static uint64_t g_tex_frame,g_sync_pipes,g_sync_pipe_ns;
static int occ_reset_next(void){return 0;} // no reset is proven ahead: a counted draw waits
static int shader_wmem(vkshader_t *s){(void)s;return 0;}
static void pipe_sibling(const pkey *k){(void)k;}
COMPLETE
SHADER
#define pthread_create fail_create
ASYNC
#undef pthread_create
static int observe_wait(pthread_cond_t *cv,pthread_mutex_t *mu){assert(cv==&g_pipe_ready);atomic_fetch_add(&waits,1);return pthread_cond_wait(cv,mu);}
#define pthread_cond_wait observe_wait
GRAPHICS
#undef pthread_cond_wait
static VkPipeline result;
static void *get_counted(void *arg){result=graphics_pipeline(arg,1);atomic_store(&returned,1);return NULL;}
static void wait_for(int n){while(atomic_load(&waits)<n)sched_yield();assert(!atomic_load(&returned));}
int main(void){
 vkimg_t target={.rt_frames=3}; P_att[0].im=&target;
 pkey cold={.key={1}};g_occ_counting=1;
 assert(graphics_pipeline(&cold,1)==(VkPipeline)(uintptr_t)101);
 assert(builds==1&&!g_apq_h&&!creates&&!atomic_load(&waits));
 assert(graphics_pipeline(&cold,1)==(VkPipeline)(uintptr_t)101);
 assert(graphics_pipeline(&cold,1)==(VkPipeline)(uintptr_t)101);
 assert(builds==1&&!atomic_load(&waits)); // Warm cache and last-key path remain immediate.

 pkey queued={.key={2}};g_occ_counting=0;g_apq_on=1;
 assert(graphics_pipeline(&queued,1)==NULL&&g_apq_h==1&&g_pipes.values[2]==PIPE_PENDING);
 assert(graphics_pipeline(&queued,1)==NULL&&g_apq_h==1&&!atomic_load(&waits));
 g_occ_counting=1;pthread_t tid;assert(!pthread_create(&tid,NULL,get_counted,&queued));wait_for(1);
 pthread_mutex_lock(&g_pipe_mu);pthread_cond_broadcast(&g_pipe_ready);pthread_mutex_unlock(&g_pipe_mu);
 wait_for(2); // Spurious/other-key completion cannot leak PIPE_PENDING or omit the draw.
 async_pipe_complete(&queued,(VkPipeline)(uintptr_t)202);
 assert(!pthread_join(tid,NULL)&&atomic_load(&returned)&&result==(VkPipeline)(uintptr_t)202&&builds==1);
 assert(graphics_pipeline(&queued,1)==(VkPipeline)(uintptr_t)202&&atomic_load(&waits)==2);

 pkey failed={.key={3}};g_occ_counting=0;
 assert(graphics_pipeline(&failed,1)==NULL&&g_pipes.values[3]==PIPE_PENDING);
 g_occ_counting=1;atomic_store(&returned,0);assert(!pthread_create(&tid,NULL,get_counted,&failed));wait_for(3);
 async_pipe_complete(&failed,(VkPipeline)1); // Failure must wake waiters, too.
 assert(!pthread_join(tid,NULL)&&atomic_load(&returned)&&result==NULL);
 assert(graphics_pipeline(&failed,1)==NULL&&atomic_load(&waits)==3);

 pkey no_worker={.key={4}};g_occ_counting=0;g_apq_on=0;int queued_before=g_apq_h;
 assert(graphics_pipeline(&no_worker,1)==(VkPipeline)(uintptr_t)104);
 assert(creates==2&&g_apq_h==queued_before&&g_pipes.values[4]!=PIPE_PENDING&&builds==2);
 pkey full={.key={5}};g_apq_on=1;g_apq_h=g_apq_t+APQ;
 assert(graphics_pipeline(&full,1)==(VkPipeline)(uintptr_t)105&&builds==3); // A full queue also compiles synchronously.
 assert(!destroys&&records==4);
 return 0;
}
'''
        pieces = {
            'GETTER': function(core, 'int vk_occ_counting('),
            'COMPLETE': function(source, 'static void async_pipe_complete('),
            'SHADER': function(source, 'static int async_shader_writes('),
            'ASYNC': function(source, 'static int pipe_must_draw(') + '\n' + function(source, 'static int apq_push(') + '\n' + function(source, 'static int async_pipe('),
            'GRAPHICS': function(source, 'static VkPipeline graphics_pipeline('),
        }
        for marker, text in pieces.items():
            code = code.replace('\n'+marker+'\n', '\n'+text+'\n')
        target = ROOT / 'target/tests'; target.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='vk-async-queries-', dir=target) as folder:
            src, exe = Path(folder)/'check.c', Path(folder)/'check'
            src.write_text(code, encoding='utf-8')
            subprocess.run(['cc','-std=c11','-Wall','-Wextra','-Werror','-pthread','-fsanitize=address,undefined',str(src),'-o',str(exe)],check=True)
            subprocess.run([str(exe)],check=True,timeout=15)


if __name__ == '__main__':
    unittest.main()
