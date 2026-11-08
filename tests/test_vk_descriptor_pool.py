"""Exercise actual descriptor allocation/reset against a per-type budget-checking Vulkan mock."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(shutil.which("cc"), "host C compiler required")
class DescriptorPool(unittest.TestCase):
    def test_allocation_budgets_failures_and_pool_lifetimes(self):
        source = (ROOT / "nxnative/dm/android/gfx_vk_core.c").read_text()
        start = source.index("enum { DP_TYPES")
        declarations = source[start:source.index("struct vksub {", start)]
        start = source.index("enum { DP_SETS")
        capacities = source[start:source.index("static nvn_hmap g_dsl_need", start)]
        start = source.index("static VkDescriptorSet descriptor_alloc_set(")
        allocate = source[start:source.index("VkDescriptorSet vk_alloc_set(", start)]
        start = source.index("static void descriptor_pools_reset(")
        reset = source[start:source.index("static void slot_recycle(", start)]
        code = r"""
#define VK_NO_PROTOTYPES
#include <vulkan/vulkan_core.h>
#include <assert.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <pthread.h>
#define H(type, value) ((type)(uintptr_t)(value))
DECLARATIONS
CAPACITIES
typedef struct {
 VkDescriptorPool *dp; int ndp,capdp,dp_cur;
 uint32_t dp_used[DP_TYPES],dp_sets; int dp_counted;
} vksub_t;
static struct {VkDevice dev;int dyn_ssbo;} V;
static int g_dsl_need;
static pthread_mutex_t g_dsl_mu=PTHREAD_MUTEX_INITIALIZER;
static uint32_t needs[128][DP_TYPES],lookups,pool_count,next_set,fail_count;
static VkResult failure;
static struct {uint32_t sets,used[DP_TYPES],capacity[DP_TYPES],generation;} pools[128];
static const uint32_t *nvn_hm_get(void *map,const void *key) {
 (void)map;lookups++;uintptr_t id=(uintptr_t)*(const VkDescriptorSetLayout *)key;
 assert(id<128);return id==127?NULL:needs[id];
}
#define die(...) do {fprintf(stderr,__VA_ARGS__);abort();} while(0)
#define VKCHK(expr) assert((expr)==VK_SUCCESS)
static VkResult vkCreateDescriptorPool(VkDevice device,const VkDescriptorPoolCreateInfo *info,const VkAllocationCallbacks *cb,VkDescriptorPool *out) {
 (void)device;(void)cb;assert(info->maxSets==DP_SETS);assert(pool_count+1<128);
 *out=H(VkDescriptorPool,++pool_count);pools[pool_count].generation=1;
 for(uint32_t i=0;i<info->poolSizeCount;i++){assert((uint32_t)info->pPoolSizes[i].type<DP_TYPES);pools[pool_count].capacity[info->pPoolSizes[i].type]=info->pPoolSizes[i].descriptorCount;}return VK_SUCCESS;
}
static VkResult vkResetDescriptorPool(VkDevice device,VkDescriptorPool pool,VkDescriptorPoolResetFlags flags) {
 (void)device;(void)flags;uintptr_t id=(uintptr_t)pool;assert(id&&id<=pool_count);
 pools[id].sets=0;memset(pools[id].used,0,sizeof pools[id].used);pools[id].generation++;return VK_SUCCESS;
}
static VkResult vkAllocateDescriptorSets(VkDevice device,const VkDescriptorSetAllocateInfo *info,VkDescriptorSet *out) {
 (void)device;assert(info->descriptorSetCount==1);uintptr_t p=(uintptr_t)info->descriptorPool;assert(p&&p<=pool_count);
 if(fail_count){fail_count--;*out=VK_NULL_HANDLE;return failure;}
 assert(pools[p].sets+1<=DP_SETS);uintptr_t l=(uintptr_t)info->pSetLayouts[0];assert(l<128);
 for(int t=0;t<DP_TYPES;t++){assert(pools[p].used[t]+needs[l][t]<=pools[p].capacity[t]);pools[p].used[t]+=needs[l][t];}
 pools[p].sets++;*out=H(VkDescriptorSet,++next_set);return VK_SUCCESS;
}
RESET
ALLOCATE
static void fresh(vksub_t *s) {
 free(s->dp);memset(s,0,sizeof *s);s->dp_counted=-1;
 memset(needs,0,sizeof needs);memset(pools,0,sizeof pools);
 lookups=pool_count=next_set=fail_count=0;V.dyn_ssbo=1;
 g_desc_alloc_calls=g_desc_alloc_sets=0;
}
static VkDescriptorSet take(vksub_t *s,uint32_t layout){return descriptor_alloc_set(s,H(VkDescriptorSetLayout,layout));}
int main(void) {
 vksub_t s={0};fresh(&s);
 needs[1][VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC]=2;needs[1][VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER]=4;
 for(int i=0;i<32;i++)assert((uintptr_t)take(&s,1)==(uintptr_t)i+1);
 assert(g_desc_alloc_calls==32&&g_desc_alloc_sets==32&&lookups==32);
 assert(s.dp_used[VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC]==64&&s.dp_sets==32);
 // Every descriptor type, including dynamic storage enum9, must stop allocation before actual pool exhaustion.
 for(int type=0;type<DP_TYPES;type++){
  fresh(&s);needs[1][type]=DP_CAP[type]/2+1;take(&s,1);take(&s,1);
  assert(pool_count==2&&s.dp_sets==1&&s.dp_used[type]==needs[1][type]&&g_desc_alloc_sets==2);
 }
 // The default-off dynamic-storage path reserves no additional driver capacity.
 fresh(&s);V.dyn_ssbo=0;take(&s,1);assert(!pools[1].capacity[VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC]);
 // maxSets is an independent limit, even for an empty layout.
 fresh(&s);take(&s,1);s.dp_sets=pools[1].sets=DP_SETS-1;take(&s,1);assert(s.dp_sets==DP_SETS);
 take(&s,1);assert(pool_count==2&&s.dp_sets==1);
 // Reset rewinds both real driver capacity and host accounting before reusing the pool.
 fresh(&s);take(&s,1);descriptor_pools_reset(&s);assert(s.dp_cur==0&&s.dp_counted==-1&&pools[1].generation==2);
 assert((uintptr_t)take(&s,1)==2&&pool_count==1&&s.dp_sets==1);
 // Failed allocations cannot consume host counts or publish a handle.
 for(int e=0;e<2;e++){
  fresh(&s);fail_count=1;failure=e?VK_ERROR_FRAGMENTED_POOL:VK_ERROR_OUT_OF_POOL_MEMORY;
  assert((uintptr_t)take(&s,1)==1&&pool_count==2&&g_desc_alloc_calls==2&&g_desc_alloc_sets==1&&s.dp_sets==1);
 }
 // An unknown layout retains single-set behavior and no speculative per-type reservation.
 fresh(&s);take(&s,127);assert(g_desc_alloc_sets==1&&s.dp_sets==1);
 // Every in-flight slot owns separate pools. Recycling one cannot rewind another.
 fresh(&s);take(&s,1);vksub_t other={.dp_counted=-1};take(&other,1);
 assert(pool_count==2);descriptor_pools_reset(&other);
 assert(pools[1].generation==1&&pools[2].generation==2&&pools[1].sets==1);
 take(&s,1);assert(pools[1].sets==2);free(other.dp);free(s.dp);
 puts("descriptor pools: all type budgets, maxSets, failures, reset and slot isolation passed");
}
"""
        for name, value in (("DECLARATIONS", declarations), ("CAPACITIES", capacities),
                            ("RESET", reset), ("ALLOCATE", allocate)):
            code = code.replace(name, value)
        target = ROOT / "target/tests"; target.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="descriptor-pool-", dir=target) as tmp:
            src = Path(tmp) / "test.c"; exe = Path(tmp) / "test"
            src.write_text(code)
            subprocess.run(["cc", "-std=c11", "-O1", "-g", "-Wall", "-Wextra", "-Werror",
                            "-fsanitize=address,undefined", "-I" + str(ROOT / "nxnative/dm/third_party/vulkan-headers/include"),
                            str(src), "-o", str(exe), "-lpthread"], check=True)
            subprocess.run([str(exe)], check=True)

    def test_registered_need_pointer_survives_hashmap_growth(self):
        source = (ROOT / "nxnative/dm/android/gfx_vk_core.c").read_text()
        start = source.index("static nvn_hmap g_dsl_need")
        register = source[start:source.index("// Allocate exactly one set", start)]
        register = next(line for line in source.splitlines() if line.startswith("enum { DP_TYPES")) + "\n" + register
        code = r"""
#define VK_NO_PROTOTYPES
#include <vulkan/vulkan_core.h>
#include <assert.h>
#include <pthread.h>
#include "nvn_hmap.h"
REGISTER
static void *grow_registry(void *unused) {
 (void)unused;
 for (uintptr_t i=2;i<=4096;i++) {
  VkDescriptorSetLayoutBinding b={.binding=0,.descriptorType=VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,.descriptorCount=(uint32_t)(i%32+1)};
  vk_dsl_register((VkDescriptorSetLayout)i,&b,1);
 }
 return NULL;
}
int main(void) {
 VkDescriptorSetLayout l=(VkDescriptorSetLayout)(uintptr_t)1;
 VkDescriptorSetLayoutBinding b[2]={
  {.binding=0,.descriptorType=VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,.descriptorCount=3},
  {.binding=1,.descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,.descriptorCount=4}};
 vk_dsl_register(l,b,2);
 pthread_mutex_lock(&g_dsl_mu);
 const uint32_t *cached=nvn_hm_get(&g_dsl_need,&l);
 uint32_t initial_cap=g_dsl_need.cap;
 pthread_mutex_unlock(&g_dsl_mu);
 pthread_t t; assert(!pthread_create(&t,NULL,grow_registry,NULL));
 // Exactly the unlocked cached-pointee access used by descriptor_alloc_set while background registration grows the map.
 for(int i=0;i<100000;i++)assert(cached[VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC]==3&&cached[VK_DESCRIPTOR_TYPE_STORAGE_BUFFER]==4);
 assert(!pthread_join(t,NULL));assert(g_dsl_need.cap>initial_cap);
 assert(cached==nvn_hm_get(&g_dsl_need,&l));
 assert(cached[VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC]==3&&cached[VK_DESCRIPTOR_TYPE_STORAGE_BUFFER]==4);
 nvn_hm_clear(&g_dsl_need,free);free(g_dsl_need.keys);free(g_dsl_need.vals);
}
"""
        code = code.replace("REGISTER", register)
        target = ROOT / "target/tests"; target.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="descriptor-registry-", dir=target) as tmp:
            src = Path(tmp) / "test.c"; exe = Path(tmp) / "test"
            src.write_text(code)
            subprocess.run(["cc", "-std=c11", "-O1", "-g", "-Wall", "-Wextra", "-Werror",
                            "-fsanitize=address,undefined",
                            "-I" + str(ROOT / "nxnative/dm/third_party/vulkan-headers/include"),
                            "-I" + str(ROOT / "nxnative/dm/gfx/nvn"),
                            str(src), "-o", str(exe), "-lpthread"], check=True)
            subprocess.run([str(exe)], check=True)


if __name__ == "__main__":
    unittest.main()
