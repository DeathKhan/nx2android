"""Exercise actual descriptor allocation/reset against a per-type budget-checking Vulkan mock."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
import sys; sys.path.insert(0, str(Path(__file__).resolve().parent))
import host_cc   # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(shutil.which("cc"), "host C compiler required")
class DescriptorPool(unittest.TestCase):
    @unittest.skipUnless(host_cc.posix() and host_cc.sanitizers(), 'a POSIX host C compiler with ASan and UBSan is required (MinGW has no setenv, libasan or libubsan)')
    def test_allocation_budgets_failures_and_pool_lifetimes(self):
        source = (ROOT / "nxnative/dm/android/gfx_vk_core.c").read_text(encoding='utf-8')
        start = source.index("enum { DP_TYPES")
        declarations = source[start:source.index("struct vksub {", start)]
        start = source.index("enum { DP_SETS")
        capacities = source[start:source.index("static nvn_hmap g_dsl_need", start)]
        start = source.index("static VkDescriptorSet descriptor_alloc_set(")
        allocate = source[start:source.index("VkDescriptorSet vk_alloc_set(", start)]
        start = source.index("static int descriptor_large_reuse(")
        reset = source[start:source.index("static void slot_recycle(", start)]
        lease = source[source.index("uint64_t vk_descriptor_lease("):source.index("// ---------------- submissions: begin, CPU-visible effects, commit")]
        oversized = "\n".join(line for line in source.splitlines()
                              if line.startswith("enum { DPX_KEEP") or line.startswith("typedef struct { VkDescriptorPool pool;") or line.startswith("static uint64_t g_descriptor_lease"))
        # The allocator's eligibility reset is behind the real slot completion wait.
        recycle = source[source.index("static void slot_recycle("):source.index("static vksub_t *cur(")]
        self.assertLess(recycle.index("mtl_wait(s->fence"), recycle.index("descriptor_pools_reset(s)"))
        self.assertLess(recycle.index("wait_fence(s->fence"), recycle.index("descriptor_pools_reset(s)"))
        code = r"""
#define _POSIX_C_SOURCE 200809L
#define VK_NO_PROTOTYPES
#include <vulkan/vulkan_core.h>
#include <assert.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <pthread.h>
#define H(type, value) ((type)(uintptr_t)(value))
static struct {VkDevice dev;int dyn_ssbo;} V;
static int vk_pool_mark_on(void){return 0;}   // (DM_VK_POOL_MARK off: dynamic storage in the shared pools only with DM_VK_DYN_SSBO)
enum { DP_KEEP = 16 };
static int g_uab;static uint64_t g_dp_made,g_dp_freed,g_set_new,destroyed;
static uint64_t kgsl_mapped(void){return 0;}
static void vklog(const char *f,...){(void)f;}
#define VKLOG(...) vklog(__VA_ARGS__)
DECLARATIONS
OVERSIZED
CAPACITIES
typedef struct {
 VkDescriptorPool *dp; int ndp,capdp,dp_cur;
 uint32_t dp_used[DP_TYPES],dp_sets; int dp_counted;
 dpx_entry *dpx; int ndpx,capdpx;
} vksub_t;
static void vkDestroyDescriptorPool(VkDevice device,VkDescriptorPool pool,const VkAllocationCallbacks *cb){(void)device;(void)cb;assert(pool);destroyed++;}
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
 (void)device;(void)cb;assert(info->maxSets==DP_SETS||info->maxSets==1);assert(pool_count+1<128);
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
static vksub_t *active_slot;
static vksub_t *cur(void){return active_slot;}
RESET
ALLOCATE
LEASE
static void fresh(vksub_t *s) {
 free(s->dp);free(s->dpx);memset(s,0,sizeof *s);s->dp_counted=-1;
 memset(needs,0,sizeof needs);memset(pools,0,sizeof pools);
 lookups=pool_count=next_set=fail_count=0;V.dyn_ssbo=1;
 g_desc_alloc_calls=g_desc_alloc_sets=0;
}
static VkDescriptorSet take(vksub_t *s,uint32_t layout){return descriptor_alloc_set(s,H(VkDescriptorSetLayout,layout));}
int main(void) {
 unsetenv("DM_VK_DPX_REUSE");
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
 // A layout no shared pool can hold gets a pool of its own (not a new shared pool per try), destroyed at the recycle; pools past DP_KEEP go too.
 s.dp=NULL;fresh(&s);needs[2][VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER]=DP_CAP[VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER]+1;
 take(&s,1);take(&s,2);take(&s,2);assert(s.ndpx==2&&s.ndp==1&&pool_count==3&&g_desc_alloc_sets==3&&s.dp_sets==1);
 s.ndp=DP_KEEP+3;s.dp=realloc(s.dp,(size_t)s.ndp*sizeof *s.dp);for(int i=1;i<s.ndp;i++)s.dp[i]=s.dp[0];
 destroyed=0;descriptor_pools_reset(&s);assert(destroyed==5&&s.ndpx==0&&s.ndp==DP_KEEP&&g_dp_freed==3&&s.dp_cur==0);
 // Oversized sets remain unique throughout an open submission. Completion makes
 // only exact-layout matches reusable, without resetting/reallocating their pool.
 fresh(&s);setenv("DM_VK_DPX_REUSE","1",1);
 needs[2][VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE]=131074;
 needs[3][VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE]=131074; // same counts, different layout
 VkDescriptorSet a=take(&s,2),b=take(&s,2);assert(a!=b);
 assert(pool_count==2&&g_desc_alloc_calls==2);
 descriptor_pools_reset(&s);assert(s.ndpx==2&&!s.dpx[0].used&&!s.dpx[1].used);
 assert(take(&s,2)==a&&take(&s,2)==b);
 VkDescriptorSet c=take(&s,2);assert(c!=a&&c!=b&&pool_count==3);
 VkDescriptorSet d=take(&s,3);assert(d!=a&&d!=b&&d!=c&&pool_count==4);
 for(unsigned i=1;i<=pool_count;i++)assert(pools[i].generation==1&&pools[i].sets==1);
 assert(g_desc_alloc_calls==4&&g_desc_alloc_sets==4); // reused sets are not driver allocations
 vksub_t independent={.dp_counted=-1};VkDescriptorSet x=take(&independent,2);
 assert(x!=a&&x!=b&&x!=c&&x!=d);descriptor_pools_reset(&independent);
 assert(take(&independent,2)==x&&take(&s,2)!=x);free(independent.dpx);independent.dpx=NULL;independent.ndpx=0;
 // Content leases are exact-slot and one-use. Normal rewrite allocation changes the token.
 active_slot=&s;uint64_t token=vk_descriptor_lease(a,0);assert(token);
 assert(!vk_descriptor_claim(a,token)); // still in flight/currently owned
 descriptor_pools_reset(&s);assert(vk_descriptor_claim(a,token));assert(!vk_descriptor_claim(a,token));
 assert(vk_descriptor_lease(a,0)==token);
 uint64_t revised=vk_descriptor_lease(a,1);assert(revised&&revised!=token);
 descriptor_pools_reset(&s);assert(!vk_descriptor_claim(a,token));assert(vk_descriptor_claim(a,revised));
 descriptor_pools_reset(&s);assert(take(&s,2)==a);assert(vk_descriptor_lease(a,0)!=revised);
 active_slot=&independent;assert(!vk_descriptor_claim(a,revised));active_slot=&s;
 // Retention is bounded even if a loading submission allocates many huge sets.
 fresh(&s);needs[2][VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE]=131074;
 for(int i=0;i<DPX_KEEP+7;i++)take(&s,2);
 destroyed=0;descriptor_pools_reset(&s);
 assert(s.ndpx==DPX_KEEP&&destroyed==7);
 for(int i=0;i<DPX_KEEP;i++)assert(!s.dpx[i].used);
 // Unused retained sets expire at the very next recycle, not after a long idle timeout.
 destroyed=0;descriptor_pools_reset(&s);assert(!s.ndpx&&destroyed==DPX_KEEP);
 // The aggregate descriptor budget bounds retention independently of entry count.
 fresh(&s);needs[2][VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE]=DPX_DESCRIPTORS/2+1;
 for(int i=0;i<3;i++)take(&s,2);
 destroyed=0;descriptor_pools_reset(&s);assert(s.ndpx==1&&destroyed==2);
 assert(take(&s,2)==H(VkDescriptorSet,1)); // the surviving immutable allocation
 // Only F1-leased contents get bounded idle retention. Eight unused completed-slot
 // cycles are allowed; the ninth expires it. Original DPX-only expiry remains above.
 fresh(&s);needs[2][VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE]=131074;active_slot=&s;
 setenv("DM_VK_POOL_PERSIST","1",1);a=take(&s,2);token=vk_descriptor_lease(a,0);
 descriptor_pools_reset(&s);
 for(int idle=1;idle<=8;idle++){descriptor_pools_reset(&s);assert(s.ndpx==1&&s.dpx[0].idle==(unsigned)idle);}
 assert(vk_descriptor_claim(a,token));descriptor_pools_reset(&s);assert(s.dpx[0].idle==0);
 for(int idle=0;idle<9;idle++)descriptor_pools_reset(&s);
 assert(!s.ndpx);
 // Pool destruction/driver handle reuse cannot revive a stale content token.
 fresh(&s);needs[2][VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE]=131074;a=take(&s,2);assert(a==H(VkDescriptorSet,1));
 descriptor_pools_reset(&s);assert(!vk_descriptor_claim(a,token));
 // F1 idle retention cannot increase either existing cap.
 fresh(&s);needs[2][VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE]=DPX_DESCRIPTORS/2+1;
 for(int i=0;i<3;i++){a=take(&s,2);assert(vk_descriptor_lease(a,0));}
 descriptor_pools_reset(&s);assert(s.ndpx==1);
 descriptor_pools_reset(&s);assert(s.ndpx==1);
 unsetenv("DM_VK_POOL_PERSIST");
 fresh(&s);needs[2][VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE]=DPX_DESCRIPTORS/2+1;
 take(&s,2);descriptor_pools_reset(&s);assert(take(&s,2)==H(VkDescriptorSet,1));
 // Disable live: no retained set is handed out, and all old pools disappear safely at recycle.
 setenv("DM_VK_DPX_REUSE","0",1);VkDescriptorSet z=take(&s,2);
 assert(z!=H(VkDescriptorSet,1));destroyed=0;descriptor_pools_reset(&s);
 assert(!s.ndpx&&destroyed==2);
 // A single layout beyond the entire retention budget is never kept.
 fresh(&s);setenv("DM_VK_DPX_REUSE","1",1);
 needs[2][VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE]=DPX_DESCRIPTORS+1;take(&s,2);
 destroyed=0;descriptor_pools_reset(&s);assert(!s.ndpx&&destroyed==1);
 unsetenv("DM_VK_DPX_REUSE");free(s.dpx);free(s.dp);
 puts("descriptor pools: budgets, failure, completed-slot reuse, bounds and layout/slot isolation passed");
}
"""
        for name, value in (("DECLARATIONS", declarations), ("OVERSIZED", oversized), ("CAPACITIES", capacities),
                            ("RESET", reset), ("ALLOCATE", allocate), ("LEASE", lease)):
            code = code.replace(name, value)
        target = ROOT / "target/tests"; target.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="descriptor-pool-", dir=target) as tmp:
            src = Path(tmp) / "test.c"; exe = Path(tmp) / "test"
            src.write_text(code, encoding='utf-8')
            subprocess.run(["cc", "-std=c11", "-O1", "-g", "-Wall", "-Wextra", "-Werror",
                            "-fsanitize=address,undefined", "-I" + str(ROOT / "nxnative/dm/third_party/vulkan-headers/include"),
                            str(src), "-o", str(exe), "-lpthread"], check=True)
            subprocess.run([str(exe)], check=True)

    @unittest.skipUnless(host_cc.sanitizers(), 'a host C compiler with ASan and UBSan is required (MinGW has neither)')
    def test_registered_need_pointer_survives_hashmap_growth(self):
        source = (ROOT / "nxnative/dm/android/gfx_vk_core.c").read_text(encoding='utf-8')
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
            src.write_text(code, encoding='utf-8')
            subprocess.run(["cc", "-std=c11", "-O1", "-g", "-Wall", "-Wextra", "-Werror",
                            "-fsanitize=address,undefined",
                            "-I" + str(ROOT / "nxnative/dm/third_party/vulkan-headers/include"),
                            "-I" + str(ROOT / "nxnative/dm/gfx/nvn"),
                            str(src), "-o", str(exe), "-lpthread"], check=True)
            subprocess.run([str(exe)], check=True)


if __name__ == "__main__":
    unittest.main()
