"""Actual table construction: pool bases, rebased samplers, kinds and immutable sets."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = "nxnative/dm/android/gfx_vk_exec.c"


def function(source, name):
    start = source.index(name)
    return source[start:source.index("\n}\n", start) + 3]


def fixture(source):
    header = (ROOT / "nxnative/dm/android/gfx_vk.h").read_text()
    defs = header[header.index("enum { RK_SUPPORT"):header.index("_Static_assert(sizeof(dmsv_bind)")]
    start = source.index("static struct { uint64_t h, epoch, tex, samp;")
    cache = source[start:source.index("static VkDescriptorSet tbl_set", start)]
    kinds = function(source, "static unsigned tbl_kinds(") if "static unsigned tbl_kinds(" in source else ""
    draw = function(source, "static int prepare_draw(")
    start = draw.index("    vkshader_t *ts = NULL;")
    selection = draw[start:draw.index("    uint64_t tdesc =", start)]
    return r"""
#include <vulkan/vulkan_core.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>
#include "nvn_hmap.h"
DEFINITIONS
enum { TBL_SLOTS=3, POOL_TEX_SLOTS=4, POOL_B0=3, POOL_SAMP_B=7, SET0_BINDINGS=8 };
#define H(type,n) ((type)(uintptr_t)(n))
#define VKLOG(...) mock_log(__VA_ARGS__)
static unsigned trace_notes;
static void mock_log(const char *format, ...){if(strstr(format,"  table set "))trace_notes++;}
#define VKRED_ONCE(...) ((void)0)
typedef struct { uint64_t addr,size; } p_bind_buf;
typedef struct { uint64_t end; } vkpool_t;
typedef struct { int set_target,target; } view_t;
typedef struct { struct { int target,width,height,depth; } d; struct {uint64_t size;} L; } tex_t;
typedef struct {uint64_t last_use;} vkimg_t;
typedef struct {uint64_t key;int ntbl,npool,tbl_reg,tbl_off,nbind,set;uint8_t tbl[96];const dmsv_bind*b;} vkshader_t;
static struct { p_bind_buf ubo[6][16]; vkshader_t *sh[6]; } S;
static struct {VkDevice dev;} V;
static uint64_t g_set_epoch=1,g_tex_gen=1,g_samp_gen=1,clock_,allocs;
static int g_tracing,g_x_nosetc,g_pool_tex_base,g_pool_samp_base,g_var_now;
static uint64_t g_num_um,g_num_im;
static uint32_t pool_generation=1, writes[256], resolves[4], sampler_resolves;
static VkDescriptorImageInfo captured[256][8][8];
static uint32_t input_words[16]={1,0,2,0};
static tex_t texture={.d={1,16,16,1},.L={4}};
static uint64_t current_texture_seq=1,current_sampler_seq=1,refreshes,dirty_uploads;
static int snapshot_valid=1,content_dirty,refresh_change,refresh_fail,pool_ok=2,current_slot;
static uint64_t leases[256],lease_clock;static int owners[256],used[256];
static vkimg_t refreshed;
static int vk_registry_snapshot(uint64_t *t,uint64_t *s){*t=current_texture_seq;*s=current_sampler_seq;return snapshot_valid;}
static uint64_t vk_descriptor_lease(VkDescriptorSet set,int rewrite){
 uintptr_t i=(uintptr_t)set;if(!used[i]||owners[i]!=current_slot)return 0;
 if(rewrite)leases[i]=++lease_clock;
 return leases[i];
}
static int vk_descriptor_claim(VkDescriptorSet set,uint64_t lease){
 uintptr_t i=(uintptr_t)set;
 if(!getenv("DM_VK_DPX_REUSE")||used[i]||owners[i]!=current_slot||leases[i]!=lease)return 0;
 used[i]=1;return 1;
}
static void completed_slot(int slot){current_slot=slot;g_set_epoch++;for(int i=1;i<256;i++)if(owners[i]==slot)used[i]=0;}
static vkimg_t *vk_tex(const tex_t *t){
 (void)t;refreshes++;if(content_dirty){dirty_uploads++;content_dirty=0;}
 if(refresh_change){g_tex_gen++;refresh_change=0;}return refresh_fail?NULL:&refreshed;
}
static uint64_t mtl_last_fence(void){return 90;}
static void vk_texdump_note(vkimg_t *im){assert(im);}
static uint64_t tick(void){return ++clock_;}
static int vk_tbl_n(void){return 2;} static int vk_pool_n(void){return 8;} static int vk_pool_samp_n(void){return 4;}
static uint32_t vk_pool_gen(void){return pool_generation;}
static vkpool_t *vk_pool_peek(uint64_t a,uint64_t n){static vkpool_t p={999999};(void)a;(void)n;return &p;}
static void vk_ubd_read(uint64_t a,uint64_t n,uint8_t *d){(void)a;memcpy(d,input_words,n);}
static VkDescriptorSetLayout vk_tbl_dsl(void){return H(VkDescriptorSetLayout,1);}
static VkDescriptorSet vk_alloc_set(VkDescriptorSetLayout l){(void)l;assert(allocs<255);++allocs;leases[allocs]=++lease_clock;owners[allocs]=current_slot;used[allocs]=1;return H(VkDescriptorSet,allocs);}
static VkImageView vk_dummy_view(int a,int b,int c,int d,int e){(void)a;(void)b;(void)c;(void)d;(void)e;return H(VkImageView,999);}
static VkSampler vk_dummy_sampler(void){return H(VkSampler,999);}
static void vk_tex_pool_fill(int on){(void)on;}
static int vk_tex_id_max(void){return g_pool_tex_base+3;}
static int vk_samp_id_max(void){return g_pool_samp_base+3;}
static int vk_tex_by_id(int id,const view_t **v,const tex_t **t){(void)id;*v=NULL;*t=&texture;return 1;}
static int view_num(const tex_t *t,const view_t *v){(void)t;(void)v;return 0;}
static int vk_tex_pool_ok(const tex_t *t){(void)t;return pool_ok;}
static VkSampler vk_samp_by_id(int id){sampler_resolves++;return H(VkSampler,id+1000);}
static void tex_desc_for(int st,const dmsv_bind*b,uint64_t h,VkDescriptorImageInfo*d,int x){
 (void)st;(void)x;unsigned kind=b->dim==3?2:b->dim==2?3:b->flags&1?1:0;resolves[kind]++;
 *d=(VkDescriptorImageInfo){H(VkSampler,h>>20),H(VkImageView,(h&0xfffff)+10000+100000*(b->flags&1)),VK_IMAGE_LAYOUT_GENERAL};
}
static void mock_update(VkDevice dev,uint32_t n,const VkWriteDescriptorSet *w,uint32_t nc,const VkCopyDescriptorSet *copy){
 (void)dev;(void)nc;(void)copy;
 for(unsigned j=0;j<n;j++){uintptr_t set=(uintptr_t)w[j].dstSet;assert(w[j].descriptorCount<=8&&set<256);
  assert(!(writes[set]&(1u<<w[j].dstBinding))); // Sets are written once, never changed after use.
  writes[set]|=1u<<w[j].dstBinding;
  memcpy(captured[set][w[j].dstBinding],w[j].pImageInfo,w[j].descriptorCount*sizeof(VkDescriptorImageInfo));
 }
}
#define vkUpdateDescriptorSets mock_update
HELPERS
static VkDescriptorSet stage_set(int st,vkshader_t *shader,uint32_t *offsets){(void)st;(void)shader;(void)offsets;return VK_NULL_HANDLE;}
static __attribute__((unused)) VkDescriptorSet draw_tables(int discard) {
    struct {int raster_discard;uint64_t um[3],im[3];} k={.raster_discard=discard};
    VkDescriptorSet sets[6]={0},tbl=VK_NULL_HANDLE;uint32_t dofs[6][96];
SELECTION
    (void)sets;return tbl;
}
static VkImageView view(VkDescriptorSet s,int binding,int element){return captured[(uintptr_t)s][binding][element].imageView;}
int main(int argc,char **argv){
 assert(argc==2);int mode=atoi(argv[1]);unsetenv("DM_VK_POOL_PERSIST");unsetenv("DM_VK_DPX_REUSE");
 if(mode==0){
  VkDescriptorSet a=tbl_set(0,NULL,17);assert(view(a,3,1)==H(VkImageView,10001));
  assert(tbl_set(0,NULL,17)==a);
  g_pool_tex_base=64;VkDescriptorSet b=tbl_set(0,NULL,17);
  assert(b!=a&&view(b,3,1)==H(VkImageView,10065));assert(view(a,3,1)==H(VkImageView,10001));
  g_pool_samp_base=128;VkDescriptorSet c=tbl_set(0,NULL,17);assert(c!=b);
  assert(captured[(uintptr_t)c][7][1].sampler==H(VkSampler,1129));
  g_pool_tex_base=0;g_pool_samp_base=0;assert(view(tbl_set(0,NULL,17),3,1)==H(VkImageView,10001));
 }else if(mode==1){
  g_pool_samp_base=4096;VkDescriptorSet a=tbl_set(0,NULL,16);
  assert(captured[(uintptr_t)a][7][1].sampler==H(VkSampler,5097));
  assert(captured[(uintptr_t)a][7][0].sampler==vk_dummy_sampler());
 }else if(mode==2){
  dmsv_bind b[2]={{.kind=RK_TEX_UBO,.desc=RD_SAMPLED,.dim=1,.samp=RS_TEX},{.kind=RK_TEX_UBO,.desc=RD_SAMPLED,.dim=1,.flags=1,.samp=RS_TEX}};
  S.ubo[0][0]=(p_bind_buf){1,16};
  vkshader_t s={.ntbl=1,.tbl_reg=0,.tbl_off=0,.nbind=1,.tbl={1},.b=b};
  VkDescriptorSet a=tbl_set(0,&s,0);assert(view(a,1,0)==vk_dummy_view(0,0,0,0,0));
  s.tbl[0]=2;s.b=&b[1];VkDescriptorSet c=tbl_set(0,&s,0);
  assert(c!=a&&view(c,1,0)==H(VkImageView,110001));assert(view(a,1,0)==H(VkImageView,999));
 }else if(mode==5){
  g_tracing=1;
  VkDescriptorSet a=tbl_set(0,NULL,19);uint64_t built=g_tbl_builds;
  unsigned r0=resolves[0],r1=resolves[1];
  for(int i=0;i<1024;i++)assert(tbl_set(0,NULL,19)==a);
  assert(g_tbl_builds==built&&allocs==1&&trace_notes==1025);
  assert(resolves[0]==r0&&resolves[1]==r1);
  assert(writes[(uintptr_t)a]==((1u<<3)|(1u<<4)|(1u<<7)));
  // Tracing preserves every normal invalidation and never rewrites old sets.
  g_tex_gen++;VkDescriptorSet x=tbl_set(0,NULL,19);assert(x!=a);a=x;
  g_samp_gen++;x=tbl_set(0,NULL,19);assert(x!=a);a=x;
  pool_generation++;x=tbl_set(0,NULL,19);assert(x!=a);a=x;
  g_set_epoch++;x=tbl_set(0,NULL,19);assert(x!=a);a=x;
  g_pool_tex_base++;x=tbl_set(0,NULL,19);assert(x!=a);a=x;
  g_pool_samp_base++;x=tbl_set(0,NULL,19);assert(x!=a);a=x;
  x=tbl_set(0,NULL,17);assert(x!=a);
  g_x_nosetc=1;a=tbl_set(0,NULL,17);assert(tbl_set(0,NULL,17)!=a);
  g_x_nosetc=0;
  dmsv_bind bind={.kind=RK_TEX_UBO,.desc=RD_SAMPLED,.dim=1,.samp=RS_TEX};
  vkshader_t shader={.ntbl=1,.tbl_reg=0,.nbind=1,.tbl={1},.b=&bind};
  S.ubo[0][0]=(p_bind_buf){1,16};a=tbl_set(0,&shader,0);
  assert(tbl_set(0,&shader,0)==a);input_words[0]=3;
  assert(tbl_set(0,&shader,0)!=a);
 }else if(mode==4){
  VkDescriptorSet a=tbl_set(0,NULL,17),b=tbl_set(0,NULL,19);assert(a!=b);
  uint64_t built=g_tbl_builds;
  for(int i=0;i<100;i++){
   assert(tbl_set(0,NULL,17)==a);assert(tbl_set(0,NULL,19)==b);
  }
  assert(g_tbl_builds==built); // Masks differing in high hash bits must not evict each other.
  assert(writes[(uintptr_t)a]==((1u<<3)|(1u<<7)));
  assert(writes[(uintptr_t)b]==((1u<<3)|(1u<<4)|(1u<<7)));
 }else{
 #if HAS_KINDS
  dmsv_bind b[3]={{.kind=RK_TEXP},{.kind=RK_TEXP},{.kind=RK_SAMPP}};
  vkshader_t vs={.nbind=1,.npool=1,.tbl={4},.b=b};
  vkshader_t fs={.nbind=2,.npool=2,.tbl={5,8},.b=b};
  vkshader_t gs={.nbind=1,.npool=1,.tbl={7},.b=b};
  S.sh[0]=&vs;S.sh[1]=&fs;S.sh[2]=&gs;
  unsigned poolmask=0;for(int st=0;st<3;st++)poolmask|=tbl_kinds(S.sh[st])>>TBL_SLOTS;
  assert(poolmask==27);assert(!tbl_kinds(NULL));
  VkDescriptorSet a=draw_tables(0);
  assert(writes[(uintptr_t)a]==((1u<<3)|(1u<<4)|(1u<<6)|(1u<<7)));
  assert(resolves[0]==3&&resolves[1]==3&&!resolves[2]&&!resolves[3]); // Test fixture contains2D textures.
  assert(sampler_resolves==3);
  VkDescriptorSet discarded=draw_tables(1);
  assert(writes[(uintptr_t)discarded]==((1u<<3)|(1u<<6))&&sampler_resolves==3);
  VkDescriptorSet narrow=tbl_set(0,NULL,1);assert(narrow!=a&&writes[(uintptr_t)narrow]==1u<<3);
  assert(writes[(uintptr_t)tbl_set(0,NULL,poolmask)]==writes[(uintptr_t)a]);
  g_tex_gen++;assert(tbl_set(0,NULL,poolmask)!=a);
  VkDescriptorSet old=tbl_set(0,NULL,poolmask);g_samp_gen++;assert(tbl_set(0,NULL,poolmask)!=old);
  old=tbl_set(0,NULL,poolmask);pool_generation++;assert(tbl_set(0,NULL,poolmask)!=old);
  old=tbl_set(0,NULL,poolmask);g_set_epoch++;assert(tbl_set(0,NULL,poolmask)!=old);
  old=tbl_set(0,NULL,poolmask);g_tracing=1;assert(tbl_set(0,NULL,poolmask)==old);
 #else
  assert(0&&"required-kind optimization is absent");
 #endif
 }
 if(mode>=6){
  setenv("DM_VK_POOL_PERSIST","1",1);setenv("DM_VK_DPX_REUSE","1",1);
  VkDescriptorSet a=tbl_set(0,NULL,19);uint64_t builds=g_tbl_builds;
  assert(g_tblp_bytes==sizeof(tbl_payload)+sizeof(tex_t)); // identical aliases and both pool kinds deduplicate the full source
  if(mode==6){
   completed_slot(0);content_dirty=1;
   assert(tbl_set(0,NULL,19)==a&&g_tbl_builds==builds&&g_tblp_hits==1);
   assert(refreshes==1&&dirty_uploads==1&&refreshed.last_use==91);
   assert(tbl_set(0,NULL,19)==a&&refreshes==1); // submission hit does not re-claim or rewrite
   completed_slot(1);assert(tbl_set(0,NULL,19)!=a); // another slot never gets its in-flight set
   completed_slot(0);assert(tbl_set(0,NULL,19)==a&&g_tblp_hits==2);
  }else if(mode==7){
   // Global native-view/pool/sampler changes and semantic snapshots independently refuse reuse.
   uint64_t *deps[]={&g_tex_gen,&g_samp_gen,&current_texture_seq,&current_sampler_seq};
   for(unsigned i=0;i<sizeof deps/sizeof *deps;i++){completed_slot(0);(*deps[i])++;
    VkDescriptorSet b=tbl_set(0,NULL,19);assert(b!=a);a=b;}
   completed_slot(0);pool_generation++;assert(tbl_set(0,NULL,19)!=a);
   completed_slot(0);snapshot_valid=0;builds=g_tbl_builds;tbl_set(0,NULL,19);assert(g_tbl_builds==builds+1);
  }else if(mode==8){
   // Refresh may upload CPU data and discover a changed native allocation/view. Rewrite only
   // after obtaining exclusive completed-slot ownership; old lease metadata is invalidated.
   completed_slot(0);content_dirty=1;refresh_change=1;writes[(uintptr_t)a]=0;
   uint64_t oldlease=leases[(uintptr_t)a];assert(tbl_set(0,NULL,19)==a);
   assert(g_tbl_builds==builds+1&&leases[(uintptr_t)a]!=oldlease&&g_tblp_fallbacks==1&&dirty_uploads==1);
   completed_slot(0);refresh_fail=1;writes[(uintptr_t)a]=0;
   assert(tbl_set(0,NULL,19)==a&&g_tblp_fallbacks==2);
  }else if(mode==9){
   // Exact lease prevents driver handle reuse or an intervening ordinary rewrite from
   // resurrecting the cached content. Cache eviction never changes driver ownership.
   completed_slot(0);leases[(uintptr_t)a]=++lease_clock;assert(tbl_set(0,NULL,19)!=a);
   uint64_t old=leases[(uintptr_t)a];tbl_persist_drop(&g_tblp[0]);assert(leases[(uintptr_t)a]==old);
   completed_slot(0);g_pool_tex_base=64;VkDescriptorSet b=tbl_set(0,NULL,19);assert(b!=a);
   completed_slot(0);g_pool_samp_base=128;assert(tbl_set(0,NULL,19)!=b);
  }else if(mode==10){
   // Missing contents must remain retryable even if CPU writes advance no registry generation.
   completed_slot(0);g_tex_gen++;pool_ok=0;VkDescriptorSet b=tbl_set(0,NULL,19);
   completed_slot(0);pool_ok=2;assert(tbl_set(0,NULL,19)!=b);
   completed_slot(0);g_var_now=3;builds=g_tbl_builds;tbl_set(0,NULL,19);assert(g_tbl_builds==builds+1);
  }else if(mode==11){
   // Full source identity includes content eligibility. Serial/layout fields in the real tex_t
   // correspond here to differing source bytes: one image must not hide another upload source.
   tbl_sources b={.seen={.klen=sizeof(tex_t)},.valid=1};tex_t t=texture;
   tbl_sources_add(&b,&t);tbl_sources_add(&b,&t);t.L.size++;tbl_sources_add(&b,&t);
   assert(b.n==2);b.cap=TBL_PERSIST_ENTRY_BYTES/sizeof(tex_t);b.n=b.cap;t.L.size++;
   tbl_sources_add(&b,&t);assert(!b.valid);tbl_sources_free(&b);
  }else if(mode==13){
   // Six completed submission slots, each visiting both table masks. The small fixture
   // budget holds six duplicate lists, so per-set copies thrash before the owner returns.
   for(int round=0;round<2;round++){
    for(int slot=0;slot<6;slot++){completed_slot(slot);tbl_set(0,NULL,17);tbl_set(0,NULL,19);}
    if(!round)g_tblp_hits=0;
   }
   assert(g_tblp_hits==12);
  }else if(mode==12){
   // CPU metadata eviction is bounded and does not release, rewrite, or invalidate driver sets.
   tbl_sources b={.valid=1};b.n=TBL_PERSIST_ENTRY_BYTES/sizeof(tex_t);b.v=calloc(b.n,sizeof(tex_t));assert(b.v);
   tbl_persist_key key;assert(tbl_persist_identity(&key,19));
   uint64_t token=leases[(uintptr_t)a];
   for(int i=0;i<6;i++){b.v[0].L.size=(uint64_t)i;b.hash=(uint64_t)i;VkDescriptorSet set=vk_alloc_set(vk_tbl_dsl());tbl_persist_save(&key,set,&b);assert(g_tblp_bytes<=TBL_PERSIST_BYTES);}
   assert(leases[(uintptr_t)a]==token&&used[(uintptr_t)a]);tbl_sources_free(&b);
  }
  vk_table_report();
 }
 for(unsigned i=0;i<TBL_PERSIST_ENTRIES;i++)tbl_persist_drop(&g_tblp[i]);
 return 0;
}
""".replace("DEFINITIONS", defs).replace("HELPERS", "\n".join((
        function(source, "static uint64_t mem_handle("), kinds, cache,
        function(source, "static VkDescriptorSet tbl_set(")))).replace("SELECTION", selection).replace("HAS_KINDS", str(int(bool(kinds))))


class PoolTables(unittest.TestCase):
    @unittest.skipUnless(shutil.which("cc"), "a host C compiler is required")
    def test_real_table_construction_and_previous_regressions(self):
        source = (ROOT / SOURCE).read_text()
        target = ROOT / "target/tests"
        target.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="pool-tables-", dir=target) as folder:
            folder = Path(folder)
            def compile_one(text, name):
                c, exe = folder / (name + ".c"), folder / name
                c.write_text(fixture(text))
                subprocess.run(["cc", "-std=gnu11", "-Wall", "-Wextra", "-Werror",
                                "-fsanitize=address,undefined",
                                "-I" + str(ROOT / "nxnative/dm/third_party/vulkan-headers/include"),
                                "-I" + str(ROOT / "nxnative/dm/gfx/nvn"),
                                str(c), "-o", str(exe)], check=True)
                return exe
            exe = compile_one(source, "candidate")
            for mode in range(13):
                subprocess.run([str(exe), str(mode)], check=True, timeout=15)
            bounded = source.replace("TBL_PERSIST_BYTES = 32 << 20", "TBL_PERSIST_BYTES = 6 * sizeof(tex_t)")
            exe = compile_one(bounded, "six-slots")
            subprocess.run([str(exe), "13"], check=True, timeout=15)
            # Restoring per-set payload copies reproduces the zero-hit bounded working set.
            unshared = bounded.replace("if (candidate && candidate->hash", "if (0 && candidate && candidate->hash", 1)
            self.assertNotEqual(unshared, bounded)
            exe = compile_one(unshared, "per-set-payloads")
            result = subprocess.run([str(exe), "13"], capture_output=True, timeout=15)
            self.assertNotEqual(result.returncode, 0)

    def test_all_active_raster_stages_and_mesh_contract(self):
        source = (ROOT / SOURCE).read_text()
        draw = function(source, "static int prepare_draw(")
        self.assertIn("poolmask |= tbl_kinds(S.sh[st]) >> TBL_SLOTS", draw)
        self.assertIn("st < 3", draw)
        self.assertIn("!(st == 1 && k.raster_discard)", draw)
        self.assertIn("tbl_set(ts ? tst : 0, ts, poolmask)", draw)
        mesh = function(source, "static void mesh_draw(const p_draw *d) {")
        self.assertIn("prepare_draw(prim, 1)", mesh)
        shader = (ROOT / "nxnative/dm/android/gfx_vk_shader.c").read_text()
        self.assertIn('stage > 4) pfail = "a compute kernel has no pool table set"', shader)


if __name__ == "__main__":
    unittest.main()
