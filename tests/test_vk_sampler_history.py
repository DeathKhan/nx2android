"""Sampler registration snapshots survive parked jobs and saved queue bindings."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]


def function(source,declaration):
    start=source.index(declaration);pos=source.index("{",start)+1;depth=1
    while depth:
        depth+=(source[pos]=="{")-(source[pos]=="}");pos+=1
    return source[start:pos]


@unittest.skipUnless(shutil.which("cc"),"a host C compiler is required")
class SamplerHistory(unittest.TestCase):
    def test_actual_registration_lookup_and_native_parameter_cache(self):
        source=(ROOT/"nxnative/dm/android/gfx_vk_tex.c").read_text()
        obj=(ROOT/"nxnative/dm/gfx/nvn/nvn_obj.h").read_text()
        exe_source=(ROOT/"nxnative/dm/android/gfx_vk_exec.c").read_text()
        par=next(line.split("//")[0] for line in obj.splitlines() if "} samp_par;" in line)
        state=source[source.index("static samp_par *g_sreg;"):source.index("uint64_t mtl_samp_seq(void)")]
        parts={
            "PAR":par,
            "STATE":state,
            "SEQUENCE":function(source,"uint64_t mtl_samp_seq(void) {"),
            "REGISTER":function(source,"void mtl_reg_samp(int sid, const samp_par *sp) {"),
            "LOOKUP":function(source,"static const samp_par *samp_at(int sid, uint64_t seq) {"),
            "JOBLOOKUP":function(source,"static const samp_par *samp_job_at(int sid) {"),
            "BYID":function(source,"VkSampler vk_samp_by_id(int sid) {"),
            "NEAREST":function(source,"VkSampler vk_samp_by_id_nearest(int sid) {"),
            "FASTCACHE":function(exe_source,"static VkSampler samp_cached(int sid) {"),
            "DYNAMIC":function(exe_source,"static VkSampler samp_dynamic_slot(int st, int slot, int dynslots) {"),
            "DESCRIBE":function(source,"void vk_samp_describe_at(int sid, uint64_t seq, char *out, size_t n) {"),
            "SAMPLER":function(source,"VkSampler vk_samp_at(int sid, uint64_t seq, int nearest) {"),
        }
        self.assertIn("di->sampler = samp_dynamic_slot(st, k, dynslots);",exe_source)
        code=r"""
#include <assert.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
typedef void *VkSampler;
#define VK_NULL_HANDLE NULL
#define LOG(...) do { if(0) fprintf(stderr,__VA_ARGS__); } while(0)
#define gettid() 0
uint64_t g_tex_frame;
PAR
static pthread_mutex_t g_rmu=PTHREAD_MUTEX_INITIALIZER;
static int logical,g_sid_max,g_vk_samp_log;
static uint64_t g_samp_gen=1;
static int nvn_logical_queues_on(void){return logical;}
static uint64_t mock_job,mock_seq;
uint64_t nvn_replaying_job(void){return mock_job;}
static int registry_job_sampler_seq(uint64_t *out){if(!logical||!mock_job)return 0;*out=mock_seq;return 1;}
static void die(const char *m){(void)m;abort();}
STATE
SEQUENCE
REGISTER
LOOKUP
JOBLOOKUP
DESCRIBE
typedef struct {size_t klen;} nvn_hmap;
static nvn_hmap g_samp_cache={sizeof(samp_par)};
static samp_par native_params[16],cache_params[16];static void *cache_handles[16];
static int native_count,cache_count;
static void *make_sampler(const samp_par *p){assert(native_count<16);native_params[native_count++]=*p;return (void*)(uintptr_t)native_count;}
static void *nvn_hm_get(nvn_hmap *h,const samp_par *p){assert(h->klen==sizeof *p);for(int i=0;i<cache_count;i++)if(!memcmp(p,&cache_params[i],sizeof *p))return cache_handles[i];return NULL;}
static void nvn_hm_put(nvn_hmap *h,const samp_par *p,void *s){assert(h->klen==sizeof *p&&cache_count<16);cache_params[cache_count]=*p;cache_handles[cache_count++]=s;}
SAMPLER
BYID
NEAREST
FASTCACHE
static int g_samp_snap=1;
static struct {uint64_t samp[6][128],tex[6][256],samp_sseq[6][128],tex_sseq[6][256];} S;
DYNAMIC
static void clear(void){
 if(g_shistory){for(int i=0;i<g_nsreg;i++)free(g_shistory[i].v);free(g_shistory);}
 free(g_sreg);free(g_sreg_ok);free(g_samp);free(g_sseq);free(g_sold);
 g_shistory=NULL;g_sreg=NULL;g_sreg_ok=NULL;g_samp=NULL;g_sseq=NULL;g_sold=NULL;
 g_nsreg=g_sid_max=0;g_sregs=0;g_samp_gen=1;native_count=cache_count=0;
}
int main(void){
 samp_par p={.minf=5,.magf=1,.lod_min=0,.lod_max=0,.aniso=4,.reduction=2};
 // Mode0 retains the legacy3-old-version path and never allocates full history.
 logical=0;mtl_reg_samp(1,&p);uint64_t first=mtl_samp_seq();
 for(int i=1;i<9;i++){p.lod_min=p.lod_max=(float)i;mtl_reg_samp(1,&p);}
 assert(!g_shistory&&samp_at(1,first)->lod_min==8&&samp_at(1,mtl_samp_seq()-1)->lod_min==7);
 clear();logical=1;
 // A recording's global sequence may precede this ID's first registration.
 mtl_reg_samp(2,&p);uint64_t before=mtl_samp_seq();p.lod_min=p.lod_max=0;mtl_reg_samp(1,&p);
 uint64_t seq[2048];seq[0]=mtl_samp_seq();uint64_t saved_queue_seq=seq[0];
 for(int i=1;i<2048;i++){
  p.lod_min=p.lod_max=(float)i;mtl_reg_samp(1,&p);seq[i]=mtl_samp_seq();
  // Unrelated IDs create gaps; registration sequences are not job numbers.
  p.border[0]=(float)i;mtl_reg_samp(2,&p);
 }
 assert(g_shistory[1].n==2047);
 // A parked job and then a saved binding can still resolve the oldest version.
 assert(samp_at(1,saved_queue_seq)->lod_min==0);
 for(int i=0;i<2048;i++){
  assert(samp_at(1,seq[i])->lod_min==(float)i);
  if(i+1<2048&&seq[i]+1<seq[i+1])assert(samp_at(1,seq[i]+1)->lod_min==(float)i);
 }
 assert(samp_at(1,0)->lod_min==2047); // existing explicit latest sentinel
 assert(!samp_at(1,before)&&!samp_at(0,0)&&!samp_at(-1,0)&&!samp_at(99999,0));
 char desc[512];vk_samp_describe_at(1,before,desc,sizeof desc);assert(strstr(desc,"not registered"));
 vk_samp_describe_at(1,saved_queue_seq,desc,sizeof desc);assert(strstr(desc,"lod 0..0"));
 // Identical re-registration allocates nothing and cannot shift a recorded cutoff.
 p=g_sreg[1];uint64_t stamp=mtl_samp_seq(),gen=g_samp_gen;size_t n=g_shistory[1].n;
 for(int i=0;i<1000;i++)mtl_reg_samp(1,&p);
 assert(mtl_samp_seq()==stamp&&g_samp_gen==gen&&g_shistory[1].n==n);
 // Growing the ID table preserves every old vector, including sparse IDs.
 mtl_reg_samp(4097,&p);assert(samp_at(1,saved_queue_seq)->lod_min==0&&g_shistory[4097].n==0);
 // Real resolution still caches native samplers by historical parameters.
 VkSampler old=vk_samp_at(1,saved_queue_seq,0);assert(old&&native_count==1&&native_params[0].lod_min==0&&native_params[0].minf==5);
 assert(vk_samp_at(1,saved_queue_seq,0)==old&&native_count==1);
 VkSampler nearest=vk_samp_at(1,saved_queue_seq,1);assert(nearest!=old&&native_count==2);
 assert(native_params[1].minf==2&&!native_params[1].magf&&native_params[1].aniso==1&&!native_params[1].reduction);
 assert(samp_at(1,saved_queue_seq)->minf==5&&samp_at(1,saved_queue_seq)->aniso==4);
 assert(!vk_samp_at(1,before,0));
 // Pool-table/implicit lookups use an immutable enqueue cutoff, including captured0.
 VkSampler latest=vk_samp_by_id(1);assert(latest&&latest!=old&&g_samp[1]==latest);
 uint64_t same_generation=g_samp_gen;
 mock_job=101;mock_seq=saved_queue_seq;
 assert(vk_samp_by_id(1)==old&&samp_cached(1)==old&&g_samp[1]==latest);
 assert(vk_samp_by_id_nearest(1)==nearest);
 mock_job=102;mock_seq=seq[1000];
 VkSampler middle=samp_cached(1);assert(middle&&middle!=old&&native_params[(uintptr_t)middle-1].lod_min==1000);
 assert(g_samp_gen==same_generation); // job identity, not global generation, invalidates fast cache
 assert(vk_samp_at(1,saved_queue_seq,0)==old); // an explicit recorded bind overrides the newer job epoch
 S.samp[1][3]=1ull<<20;S.samp_sseq[1][3]=saved_queue_seq;
 assert(samp_dynamic_slot(1,3,1)==old&&!samp_dynamic_slot(1,3,0));
 S.samp_sseq[1][3]=0;assert(!samp_dynamic_slot(1,3,1)); // explicitly recorded before any registration
 S.samp[1][3]=0;S.tex[1][3]=1ull<<20;S.tex_sseq[1][3]=saved_queue_seq;
 assert(samp_dynamic_slot(1,3,1)==old); // combined texture handle fallback keeps its own sequence
 g_samp_snap=0;assert(samp_dynamic_slot(1,3,1)==middle);g_samp_snap=1;
 logical=0;assert(samp_dynamic_slot(1,3,1)==latest);logical=1; // legacy dynamic arrays remain latest

 mock_job=103;mock_seq=0;
 assert(!samp_cached(1)&&!vk_samp_by_id(1)&&!vk_samp_by_id_nearest(1));
 vk_samp_describe_at(1,0,desc,sizeof desc);assert(strstr(desc,"not registered"));
 mock_job=0;assert(samp_cached(1)==latest&&g_samp[1]==latest);
 clear();return 0;
}
"""
        for marker,value in parts.items():code=code.replace("\n"+marker+"\n","\n"+value+"\n")
        target=ROOT/"target/tests";target.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=target,prefix="sampler-history-") as temp:
            src,exe=Path(temp)/"test.c",Path(temp)/"test";src.write_text(code)
            subprocess.run(["cc","-std=gnu11","-O1","-g","-Wall","-Wextra","-Werror",
                            "-fsanitize=address,undefined","-fno-omit-frame-pointer",str(src),"-o",str(exe),"-lpthread"],check=True)
            subprocess.run([str(exe)],check=True)


if __name__=="__main__":unittest.main()
