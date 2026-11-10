"""Queued binds finish reading CPU program objects before lifecycle mutation."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


def function(source, declaration):
    start = source.index(declaration)
    brace = source.index('{', start)
    depth, pos = 1, brace + 1
    while depth:
        depth += (source[pos] == '{') - (source[pos] == '}')
        pos += 1
    return source[start:pos]


class ProgramLifetime(unittest.TestCase):
    @unittest.skipUnless(shutil.which('cc'), 'a host C compiler is required')
    def test_queued_finalize_and_replacement(self):
        source = (ROOT / 'nxnative/dm/gfx/nvn/hle_nvn.c').read_text(encoding='utf-8')
        parts = {
            'WAIT': function(source, 'static int job_wait('),
            'DRAIN': function(source, 'static void nvn_drain(void) {'),
            'FINALIZE': function(source, 'NVN(nvnProgramFinalize,'),
            'SET': function(source, 'NVN(nvnProgramSetShaders,'),
            'KEY': function(source, 'int prog_key('),
            'FLAT': function(source, 'static size_t sub_flat('),
            'COPY': function(source, 'static uint64_t sub_copy('),
        }
        # prog_key tells a live program by nvnProgramInitialize's magic (upstream nx2android e69473622): MAGIC comes from nvn_obj.h
        obj = (ROOT / 'nxnative/dm/gfx/nvn/nvn_obj.h').read_text(encoding='utf-8')
        parts['MAGICDEF'] = next(l for l in obj.splitlines() if l.startswith('#define MAGIC('))
        code = r'''
#include <assert.h>
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdarg.h>
#include <pthread.h>
#include <stdatomic.h>
#include <sched.h>
#include <time.h>
#include <errno.h>
#define NVN(name,ret,...) static ret name(__VA_ARGS__)
MAGICDEF
#define PROG_MAGIC MAGIC('P','R','O','G') /* what nvnProgramInitialize writes */
typedef int NVNboolean;
typedef struct { uint64_t data; const void *control; } NVNshaderData;
typedef struct { uint32_t magic,nstage; NVNshaderData sd[6]; uint64_t serial; } prog_t;
typedef prog_t NVNprogram;
typedef struct { uint8_t key[6][8],ok; } prog_keys_t;
typedef struct { const prog_t *prog; int32_t stages; } p_bind_program;
typedef struct { uint32_t op,size; } op_hdr;
enum { OP_END,OP_JUMP,OP_CALL,OP_BIND_PROGRAM };
static pthread_mutex_t g_sub_mu=PTHREAD_MUTEX_INITIALIZER,g_job_mu=PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t g_jobdone_cv=PTHREAD_COND_INITIALIZER;
static uint64_t g_job_enq,g_job_done,g_flat_bytes,serial;
static int async_on=1,mapped=0,forget_count,release_count;
static _Atomic int waits,finished;
static prog_keys_t cache[64];
static prog_keys_t *pk_get(uint64_t s){assert(s<64);return cache+s;}
static uint64_t nvn_new_serial(void){return ++serial;}
static int nvn_async(void){return async_on;}
static void log_ignore(const char *s,...){(void)s;}
#define LOG(...) log_ignore(__VA_ARGS__)
#define OLOG(...) log_ignore(__VA_ARGS__)
#define die(...) abort()
static void dm_backtrace(uint64_t p){(void)p;}
static void nvn_ctl_lens(const void *c,uint32_t *ctl,uint32_t *code){assert(c);*ctl=16;*code=8;}
static int keys_hash(const NVNshaderData *sd,int require_mapping,uint8_t out[8]){
 if(require_mapping&&!mapped)return 0;
 const unsigned char *ctl=sd->control,*code=(const void *)(uintptr_t)sd->data;
 for(int i=0;i<8;i++)out[i]=ctl[i]^code[i];return 1;
}
static void (*g_nvn_on_shader)(const uint8_t[8],const void *,uint64_t,uint32_t);
static void assert_locked(void){int r=pthread_mutex_trylock(&g_sub_mu);assert(r==EBUSY);}
static void mtl_prog_forget(const prog_t *p){assert_locked();assert(p->magic==PROG_MAGIC);forget_count++;}
static void mtl_prog_release(prog_t *p){if(p->serial)assert_locked();release_count++;}
static void mtl_prog_prefetch(const prog_t *p){(void)p;}
static int observe_wait(pthread_cond_t *cv,pthread_mutex_t *mu,const struct timespec *ts){atomic_fetch_add(&waits,1);return pthread_cond_timedwait(cv,mu,ts);}
#define pthread_cond_timedwait observe_wait
WAIT
#undef pthread_cond_timedwait
DRAIN
FINALIZE
SET
KEY
FLAT
COPY
static prog_t *program;
static uint64_t copied;
static NVNshaderData replacement;
static void wait_blocked(void){while(!atomic_load(&waits))sched_yield();assert(!atomic_load(&finished));}
static void *finalize_thread(void *unused){(void)unused;nvnProgramFinalize(program);atomic_store(&finished,1);return NULL;}
static void *replace_thread(void *unused){(void)unused;assert(nvnProgramSetShaders(program,1,&replacement));atomic_store(&finished,1);return NULL;}
static void queue_bind(void){
 struct { op_hdr h; p_bind_program p; op_hdr end; } record={{OP_BIND_PROGRAM,sizeof(op_hdr)+sizeof(p_bind_program)},{program,3},{OP_END,8}};
 copied=sub_copy((uintptr_t)&record);
 p_bind_program *p=(void *)(uintptr_t)(copied+8);assert(p->prog==program); // submitted copies keep the external pointer
 pthread_mutex_lock(&g_job_mu);g_job_enq++;pthread_mutex_unlock(&g_job_mu);
}
static void replay_one(uint64_t expected_serial){
 pthread_mutex_lock(&g_sub_mu);
 p_bind_program *b=(void *)(uintptr_t)(copied+8);const prog_t *p=b->prog;
 assert(p->magic==PROG_MAGIC&&p->serial==expected_serial&&p->nstage==1);
 assert(*(const unsigned char *)p->sd[0].control==0x11);
 assert(*(const unsigned char *)(uintptr_t)p->sd[0].data==0x22);
 uint8_t key[8];assert(prog_key(p,0,key));for(int i=0;i<8;i++)assert(key[i]==0x33);
 pthread_mutex_unlock(&g_sub_mu);
 free((void *)(uintptr_t)copied);copied=0;
 pthread_mutex_lock(&g_job_mu);g_job_done++;pthread_cond_broadcast(&g_jobdone_cv);pthread_mutex_unlock(&g_job_mu);
}
static void make_program(void){
 program=calloc(1,sizeof *program);program->magic=PROG_MAGIC;
 unsigned char *ctl=malloc(16),*code=malloc(8);memset(ctl,0x11,16);memset(code,0x22,8);
 NVNshaderData sd={(uintptr_t)code,ctl};assert(nvnProgramSetShaders(program,1,&sd));
 assert(program->sd[0].control!=ctl);free(ctl); // SetShaders preserved control when keying must wait for a mapped code area
}
static void reset_wait(void){atomic_store(&waits,0);atomic_store(&finished,0);}
int main(void){
 unsetenv("DM_NVN_SHDUMP");unsetenv("DM_NVN_PROGTRACE");pthread_t thread;
 make_program();uint64_t s=program->serial;queue_bind();reset_wait();
 assert(!pthread_create(&thread,NULL,finalize_thread,NULL));wait_blocked();
 assert(!forget_count&&program->magic==PROG_MAGIC);replay_one(s);pthread_join(thread,NULL);
 assert(atomic_load(&finished)&&forget_count==1&&!program->magic);
 assert(cache[s].ok==1); // the live program's replay kept its key in its record
 // a bind replayed after Finalize still keys the program, but makes no record nothing would drop (upstream e69473622)
 {memset(&cache[s],0,sizeof cache[s]);uint8_t key[8];assert(prog_key(program,0,key));for(int i=0;i<8;i++)assert(key[i]==0x33);assert(!cache[s].ok);}
 free((void *)program->sd[0].control);free((void *)(uintptr_t)program->sd[0].data);free(program);
 make_program();s=program->serial;queue_bind();reset_wait();
 const void *old_ctl=program->sd[0].control;uint64_t old_code=program->sd[0].data;
 unsigned char new_ctl[16],new_code[8];memset(new_ctl,0x44,16);memset(new_code,0x55,8);replacement=(NVNshaderData){(uintptr_t)new_code,new_ctl};
 assert(!pthread_create(&thread,NULL,replace_thread,NULL));wait_blocked();
 assert(program->serial==s&&program->sd[0].control==old_ctl);replay_one(s);pthread_join(thread,NULL);
 assert(atomic_load(&finished)&&program->serial!=s&&*(const unsigned char *)program->sd[0].control==0x44);
 free((void *)old_ctl);free((void *)(uintptr_t)old_code);free((void *)program->sd[0].control);free(program);
 // Unrelated queued work cannot stall first-time shader setup or invalid input.
 pthread_mutex_lock(&g_job_mu);g_job_enq++;pthread_mutex_unlock(&g_job_mu);
 reset_wait();make_program();assert(!atomic_load(&waits));assert(!nvnProgramSetShaders(program,7,&replacement));assert(!atomic_load(&waits));
 free((void *)program->sd[0].control);free((void *)(uintptr_t)program->sd[0].data);free(program);
 pthread_mutex_lock(&g_job_mu);g_job_done=g_job_enq;pthread_mutex_unlock(&g_job_mu);
 // Synchronous finalization keeps its existing lock and does not create a wait.
 async_on=0;make_program();reset_wait();nvnProgramFinalize(program);assert(!atomic_load(&waits));
 free((void *)program->sd[0].control);free((void *)(uintptr_t)program->sd[0].data);free(program);
 return 0;
}
'''
        for key, text in parts.items():
            code = code.replace('\n' + key + '\n', '\n' + text + '\n')
        target = ROOT / 'target/tests'
        target.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='nvn-program-lifetime-', dir=target) as folder:
            src, exe = Path(folder) / 'test.c', Path(folder) / 'test'
            src.write_text(code, encoding='utf-8')
            subprocess.run(['cc', '-std=gnu11', '-g', '-O1', '-Wall', '-Wextra', '-Werror', '-Wno-misleading-indentation', '-pthread', '-fsanitize=address,undefined', str(src), '-o', str(exe)], check=True)
            subprocess.run([str(exe)], check=True, timeout=15)

    def test_lifecycle_lock_order(self):
        source = (ROOT / 'nxnative/dm/gfx/nvn/hle_nvn.c').read_text(encoding='utf-8')
        final = function(source, 'NVN(nvnProgramFinalize,')
        self.assertLess(final.index('nvn_drain();'), final.index('pthread_mutex_lock(&g_sub_mu)'))
        self.assertLess(final.index('mtl_prog_release('), final.index('pthread_mutex_unlock(&g_sub_mu)'))
        pool = function(source, 'NVN(nvnMemoryPoolFinalize,')
        self.assertLess(pool.index('nvn_drain();'), pool.index('mtl_pool_del('))


if __name__ == '__main__':
    unittest.main()
