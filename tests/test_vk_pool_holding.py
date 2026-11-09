"""Exercise the actual CPU pool-containment lookup and ordered-index mutations."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


def function(source, declaration):
    start = source.index(declaration)
    pos = source.index('{', start) + 1
    depth = 1
    while depth:
        depth += (source[pos] == '{') - (source[pos] == '}')
        pos += 1
    return source[start:pos]


class PoolHolding(unittest.TestCase):
    @unittest.skipUnless(shutil.which('cc'), 'a host C compiler is required')
    def test_actual_indexed_lookup_against_linear_oracle(self):
        source = (ROOT / 'nxnative/dm/android/gfx_vk_core.c').read_text()
        helpers = '\n'.join(function(source, declaration) for declaration in (
            'static void pool_ord_insert(', 'static void pool_ord_remove(',
            'vkpool_t *vk_pool_holding('))
        code = r'''
#define _POSIX_C_SOURCE 200809L
#include <assert.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <pthread.h>
#include <stdatomic.h>
#define MAXPOOL 8192
typedef struct { uint64_t lo,hi,base,end; } vkpool_t;
static vkpool_t pools[MAXPOOL];
static int g_npool;
static uint16_t g_pool_ord[MAXPOOL];
static pthread_rwlock_t g_pool_mu=PTHREAD_RWLOCK_INITIALIZER;
static _Thread_local int locked;
static _Atomic uint64_t accesses;
static int rdlock(pthread_rwlock_t *m){int r=pthread_rwlock_rdlock(m);assert(!r&&!locked);locked=1;return r;}
static int wrlock(pthread_rwlock_t *m){int r=pthread_rwlock_wrlock(m);assert(!r&&!locked);locked=1;return r;}
static int unlock(pthread_rwlock_t *m){assert(locked);locked=0;return pthread_rwlock_unlock(m);}
static vkpool_t *pool_array(void){assert(locked);atomic_fetch_add(&accesses,1);return pools;}
#define pthread_rwlock_rdlock rdlock
#define pthread_rwlock_wrlock wrlock
#define pthread_rwlock_unlock unlock
#define g_pool (pool_array())
HELPERS
static void add(uint64_t lo,uint64_t hi,uint64_t base,uint64_t end){
 assert(lo<hi&&base<=lo&&hi<=end);pthread_rwlock_wrlock(&g_pool_mu);
 assert(g_npool<MAXPOOL);pools[g_npool]=(vkpool_t){lo,hi,base,end};
 pool_ord_insert(g_npool++);pthread_rwlock_unlock(&g_pool_mu);
}
static void remove_at(int i){
 pthread_rwlock_wrlock(&g_pool_mu);assert(i>=0&&i<g_npool);
 pools[i]=pools[--g_npool];pool_ord_remove(i);pthread_rwlock_unlock(&g_pool_mu);
}
static void clear(void){pthread_rwlock_wrlock(&g_pool_mu);g_npool=0;pthread_rwlock_unlock(&g_pool_mu);}
static int contains(const vkpool_t *p,uint64_t a,uint64_t n){
 return n<=UINT64_MAX-a&&a>=p->lo&&a<p->hi&&a+n<=p->end;
}
static void check(uint64_t a,uint64_t n){
 int expected=0;for(int i=0;i<g_npool;i++)expected|=contains(&pools[i],a,n);
 vkpool_t *p=vk_pool_holding(a,n);assert(!!p==expected);
 if(p)assert(p>=pools&&p<pools+g_npool&&contains(p,a,n));
 assert(!locked);
}
static uint64_t seed=0x12349876;
static uint64_t rng(void){seed^=seed<<13;seed^=seed>>7;seed^=seed<<17;return seed;}
static _Atomic int writer_done;
static void *writer(void *unused){
 (void)unused;
 for(int i=0;i<6000;i++){add(0x8000,0x9000,0x8000,0x9000);remove_at(1);}
 atomic_store(&writer_done,1);return NULL;
}
static void *reader(void *unused){
 (void)unused;unsigned n=0;
 do{assert(vk_pool_holding(0x1100,0x100)!=NULL);assert(!locked);n++;}
 while(!atomic_load(&writer_done)||n<6000);
 return NULL;
}
int main(void){
 check(0,0);check(42,1);check(UINT64_MAX,1);
 // Cross-chunk CPU reads must use the logical pool end, while the starting byte
 // must still lie inside a live chunk. Gaps are not accepted.
 add(0x1000,0x2000,0x1000,0x5000);add(0x2000,0x3000,0x1000,0x5000);
 check(0x1800,0x3000);check(0x2800,0x2800);check(0x2800,0x2801);
 check(0x3000,0);check(0x1000,0);check(0x0fff,1);
 assert(vk_pool_holding(0x1800,0x3000));assert(!vk_pool_holding(0x3000,0));
 // A short overlapping predecessor must not hide an earlier containing pool.
 add(0x1900,0x1a00,0x1900,0x1a00);check(0x1980,0x1000);
 assert(vk_pool_holding(0x1980,0x1000));
 // Equal starts, deletion's swap-last and address reuse all keep the index valid.
 add(0x1000,0x1100,0x1000,0x1100);check(0x1080,0x3000);
 remove_at(0);check(0x1080,0x3000);check(0x2800,0x100);
 remove_at(0);check(0x2800,0x100);add(0x2000,0x4000,0x2000,0x4000);
 check(0x2800,0x100);
 // Overflow cannot wrap a very long range into an apparent valid pool.
 clear();add(UINT64_MAX-100,UINT64_MAX,UINT64_MAX-100,UINT64_MAX);
 check(UINT64_MAX-50,50);check(UINT64_MAX-50,51);check(UINT64_MAX,0);
 assert(!vk_pool_holding(UINT64_MAX-50,51));
 // Positive lookups across thousands of disjoint pools are logarithmic.
 clear();for(unsigned i=0;i<4096;i++)add(0x100000ull+i*0x4000,0x101000ull+i*0x4000,0x100000ull+i*0x4000,0x101000ull+i*0x4000);
 atomic_store(&accesses,0);assert(vk_pool_holding(0x100100ull+4095*0x4000,16));
 uint64_t cost=atomic_load(&accesses);assert(cost<32);
 printf("4096-pool lookup: %llu guarded array accesses\n",(unsigned long long)cost);
 // Random overlapping logical pools and chunk seams, compared to a separate
 // overflow-safe linear oracle after every mutation.
 clear();
 for(int step=0;step<1200;step++){
  if(g_npool&&((rng()&3)==0))remove_at((int)(rng()%(unsigned)g_npool));
  else{uint64_t a=(rng()%4096)*16+0x1000,sz=(rng()%256+1)*16;
   add(a,a+sz,a,a+sz+(rng()%256)*16);}
  for(int j=0;j<20;j++){uint64_t a=rng()%0x14000,n=rng()%0x6000;check(a,n);}
 }
 // The same real rwlock protects lookup and mutation. No generation shortcut
 // or cached pointer permits reading the array while the writer swaps entries.
 clear();add(0x1000,0x2000,0x1000,0x2000);
 pthread_t a,b;assert(!pthread_create(&a,NULL,writer,NULL));assert(!pthread_create(&b,NULL,reader,NULL));
 pthread_join(a,NULL);pthread_join(b,NULL);assert(g_npool==1);
 return 0;
}
'''
        code = code.replace('HELPERS', helpers)
        target = ROOT / 'target/tests'
        target.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='pool-holding-', dir=target) as folder:
            src, exe = Path(folder) / 'test.c', Path(folder) / 'test'
            src.write_text(code)
            subprocess.run(['cc', '-std=c11', '-O1', '-g', '-Wall', '-Wextra', '-Werror',
                            '-fsanitize=address,undefined', '-fno-omit-frame-pointer',
                            '-pthread', str(src), '-o', str(exe)], check=True)
            subprocess.run([str(exe)], check=True, timeout=30)


if __name__ == '__main__':
    unittest.main()
