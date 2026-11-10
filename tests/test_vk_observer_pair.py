"""Actual paired-loader publication: no one-sided module, immutable cache, bounded failures."""
from pathlib import Path
import shutil, subprocess, tempfile, unittest
from test_vk_compute_trace import extract
ROOT=Path(__file__).resolve().parents[1]
@unittest.skipUnless(shutil.which("cc"),"host compiler required")
class ObserverPair(unittest.TestCase):
 def test_atomic_loader(self):
  source=(ROOT/"nxnative/dm/android/gfx_vk_shader.c").read_text(encoding='utf-8')
  actual=extract(source,"int vk_shader_observer_pair(")
  body=r"""
#define _POSIX_C_SOURCE 200809L
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <assert.h>
typedef uintptr_t VkShaderModule;typedef uint32_t CC_LONG;
enum{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO=1,VK_SUCCESS=0,CC_SHA256_DIGEST_LENGTH=32};
typedef struct{int sType;void*pNext;int flags;size_t codeSize;const uint32_t*pCode;}VkShaderModuleCreateInfo;
typedef struct{uint64_t key;uint32_t*spv,nspv;}vkshader_t;
static struct{int dev;}V;
static pthread_mutex_t g_sh_mu=PTHREAD_MUTEX_INITIALIZER;
static int creates,destroys,contracts,scenario;
#define VKLOG(...) do{if(0)fprintf(stderr,__VA_ARGS__);}while(0)
static void CC_SHA256(const void*p,CC_LONG n,unsigned char*h){(void)p;(void)n;memset(h,1,32);}
static int spv_observer_contract(const uint32_t*b,uint32_t bn,const uint32_t*a,uint32_t an,int f){
 assert(b&&bn==6&&an==6);contracts++;return a[5]==(uint32_t)(100+f);
}
static int vkCreateShaderModule(int d,const VkShaderModuleCreateInfo*c,void*p,VkShaderModule*out){
 (void)d;(void)p;assert(contracts==2&&c->codeSize==24);creates++;
 if(scenario==3&&creates==2)return -1;*out=(uintptr_t)(1000+creates);return 0;
}
static void vkDestroyShaderModule(int d,VkShaderModule m,void*p){(void)d;(void)p;assert(m==1001);destroys++;}
"""+actual+r"""
static uint32_t base[6]={0x07230203,0x10300,0,9,0,0};
static vkshader_t vs={0xc8ab0e758c17df4aull,base,6},fs={0x96dbc0392691b437ull,base,6};
static void *thread(void*p){(void)p;VkShaderModule mods[2]={0};assert(vk_shader_observer_pair(&vs,&fs,mods));assert(mods[0]==1001&&mods[1]==1002);return NULL;}
int main(int argc,char**argv){
 assert(argc==3);scenario=atoi(argv[1]);setenv("DM_VK_FLAT_OBSERVER_DIR",argv[2],1);setenv("DM_VK_FLAT_OBSERVER",scenario==0?"0":"1",1);
 char path[1024];uint64_t keys[2]={vs.key,fs.key};
 for(unsigned i=0;i<2;i++){
  snprintf(path,sizeof path,"%s/%016llx.spv",argv[2],(unsigned long long)keys[i]);
  if(scenario==1&&i==1)continue;
  FILE*f=fopen(path,"wb");assert(f);uint32_t words[6]={0x07230203,0x10300,0,10,0,100+i};
  if(scenario==2&&i==1)words[5]=999;
  assert(fwrite(words,4,6,f)==6);fclose(f);
 }
 VkShaderModule mods[2]={77,88};
 if(scenario<=3){
  assert(!vk_shader_observer_pair(&vs,&fs,mods));assert(mods[0]==77&&mods[1]==88);
  if(scenario==3){assert(creates==2&&destroys==1);}else assert(!creates&&!destroys);
  assert(!vk_shader_observer_pair(&vs,&fs,mods)); // failed pair does not partially retry/publish
 }else{
  pthread_t t[12];for(unsigned i=0;i<12;i++)assert(!pthread_create(&t[i],NULL,thread,NULL));
  for(unsigned i=0;i<12;i++)assert(!pthread_join(t[i],NULL));
  assert(creates==2&&!destroys&&contracts==2);
  assert(vk_shader_observer_pair(&vs,&fs,mods)&&mods[0]==1001&&mods[1]==1002);
  vkshader_t other=vs;mods[0]=77;mods[1]=88;
  assert(!vk_shader_observer_pair(&other,&fs,mods)&&mods[0]==77&&mods[1]==88);
  assert(creates==2); // a different normalized base cannot reuse the module pair
 }
 return 0;
}
"""
  target=ROOT/"target/tests";target.mkdir(parents=True,exist_ok=True)
  with tempfile.TemporaryDirectory(prefix="observer-pair-",dir=target) as tmp:
   tmp=Path(tmp);c=tmp/"pair.c";e=tmp/"pair";c.write_text(body, encoding='utf-8')
   subprocess.run(["cc","-std=c11","-O1","-g","-Wall","-Wextra","-Werror","-Wno-misleading-indentation","-fsanitize=address,undefined","-pthread",str(c),"-o",str(e)],check=True)
   for scenario in range(5):
    d=tmp/str(scenario);d.mkdir();subprocess.run([str(e),str(scenario),str(d)],check=True,timeout=10)
 def test_integration_identity(self):
  s=(ROOT/"nxnative/dm/android/gfx_vk_shader.c").read_text(encoding='utf-8')
  e=(ROOT/"nxnative/dm/android/gfx_vk_exec.c").read_text(encoding='utf-8')
  alt=extract(s,"static VkShaderModule alt_module(")
  self.assertIn('return s->mod; // the observer pair',alt)
  self.assertIn('k->alt == 2',extract(e,"static int async_pipe("))
  self.assertIn('k->alt || k->flat',extract(e,"static void pipe_record(const pkey *k) {"))
  build=extract(e,"static VkPipeline build_pipeline(const pkey *k, VkPipelineLayout layout, vkshader_t *const stg_in")
  self.assertIn('k->alt==2&&!vk_shader_observer_pair',build)
  self.assertIn('(k->alt==2&&i<2)?observer[i]',build)
  self.assertIn('k->alt==1 &&',build)
if __name__=="__main__":unittest.main()
