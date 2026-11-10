"""GPU reports use calibrated event times, independent of CPU completion delay."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
import sys; sys.path.insert(0, str(Path(__file__).resolve().parent))
import host_cc   # noqa: E402

ROOT=Path(__file__).resolve().parents[1]


def function(source,declaration):
    start=source.index(declaration);brace=source.index('{',start);depth=1;pos=brace+1
    while depth:
        depth+=(source[pos]=='{')-(source[pos]=='}');pos+=1
    return source[start:pos]


class CounterClock(unittest.TestCase):
    @unittest.skipUnless(shutil.which('cc'),'a host C compiler is required')
    @unittest.skipUnless(host_cc.sanitizers(), 'a host C compiler with ASan and UBSan is required (MinGW has neither)')
    def test_actual_calibration_and_projection(self):
        dm=ROOT/'nxnative/dm';core=(dm/'android/gfx_vk_core.c').read_text(encoding='utf-8');os=(dm/'host/hle_os.c').read_text(encoding='utf-8')
        helper=core[core.index('static PFN_vkGetCalibratedTimestampsKHR g_gpu_calibrated;'):core.index('static void completion_start(void);')]
        project=function(os,'uint64_t guest_ns_at_host_time(')
        code=r'''
#define VK_NO_PROTOTYPES
#include <vulkan/vulkan_core.h>
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <stdatomic.h>
#include <stdarg.h>
#include <math.h>
#include <time.h>
#include "nvn_timestamp.h"
static struct { VkInstance inst;VkPhysicalDevice pd;VkDevice dev;struct { struct { float timestampPeriod; } limits; } props; } V;
static VkTimeDomainKHR offered[3];static uint32_t offered_n;
static int absent,enumeration_fail,cal_fail,cal_calls,clock_n,clock_at;
static uint64_t clock_ns[4],sample_values[2],sample_deviation;
static clockid_t clock_ids[4];
static int mock_clock_gettime(clockid_t id,struct timespec *t){assert(clock_at<clock_n);assert(clock_ids[clock_at]==id);uint64_t n=clock_ns[clock_at++];*t=(struct timespec){n/1000000000,n%1000000000};return 0;}
static VkResult VKAPI_CALL domains(VkPhysicalDevice p,uint32_t *n,VkTimeDomainKHR *v){(void)p;if(enumeration_fail)return VK_ERROR_UNKNOWN;if(v)memcpy(v,offered,offered_n*sizeof *v);*n=offered_n;return VK_SUCCESS;}
static VkResult VKAPI_CALL calibrated(VkDevice d,uint32_t n,const VkCalibratedTimestampInfoKHR *i,uint64_t *t,uint64_t *dev){(void)d;assert(n==2&&i[0].timeDomain==VK_TIME_DOMAIN_DEVICE_KHR);assert(i[1].timeDomain!=i[0].timeDomain);cal_calls++;if(cal_fail)return VK_ERROR_UNKNOWN;memcpy(t,sample_values,sizeof sample_values);*dev=sample_deviation;return VK_SUCCESS;}
static PFN_vkVoidFunction VKAPI_CALL instance_proc(VkInstance i,const char *n){(void)i;if(absent)return NULL;assert(!strcmp(n,"vkGetPhysicalDeviceCalibrateableTimeDomainsKHR")||!strcmp(n,"vkGetPhysicalDeviceCalibrateableTimeDomainsEXT"));return (PFN_vkVoidFunction)domains;}
static PFN_vkVoidFunction VKAPI_CALL device_proc(VkDevice d,const char *n){(void)d;if(absent)return NULL;assert(!strcmp(n,"vkGetCalibratedTimestampsKHR")||!strcmp(n,"vkGetCalibratedTimestampsEXT"));return (PFN_vkVoidFunction)calibrated;}
static PFN_vkGetInstanceProcAddr p_vkGetInstanceProcAddr=instance_proc;
static PFN_vkGetDeviceProcAddr vkGetDeviceProcAddr=device_proc;
static void log_ignore(const char *s,...){(void)s;}
#define VKLOG(...) log_ignore(__VA_ARGS__)
#define clock_gettime mock_clock_gettime
HELPER
#undef clock_gettime
static _Atomic int g_gdet_on,g_gfz;
static _Atomic uint64_t g_gdet,g_gfrozen,g_glag;
PROJECT
static void mono_clocks(uint64_t before,uint64_t after){clock_at=0;clock_n=2;clock_ids[0]=clock_ids[1]=CLOCK_MONOTONIC;clock_ns[0]=before;clock_ns[1]=after;}
static void raw_clocks(uint64_t before,uint64_t ra,uint64_t rb,uint64_t after){clock_at=0;clock_n=4;clock_ids[0]=clock_ids[3]=CLOCK_MONOTONIC;clock_ids[1]=clock_ids[2]=CLOCK_MONOTONIC_RAW;clock_ns[0]=before;clock_ns[1]=ra;clock_ns[2]=rb;clock_ns[3]=after;}
static void reset_init(void){g_gpu_calibrated=NULL;g_gpu_timestamp_bits=48;offered_n=3;offered[0]=VK_TIME_DOMAIN_DEVICE_KHR;offered[1]=VK_TIME_DOMAIN_CLOCK_MONOTONIC_RAW_KHR;offered[2]=VK_TIME_DOMAIN_CLOCK_MONOTONIC_KHR;}
int main(void){
 reset_init();gpu_clock_init("VK_KHR_calibrated_timestamps");assert(g_gpu_calibrated&&g_gpu_host_domain==VK_TIME_DOMAIN_CLOCK_MONOTONIC_KHR);
 reset_init();offered_n=2;gpu_clock_init("VK_EXT_calibrated_timestamps");assert(g_gpu_calibrated&&g_gpu_host_domain==VK_TIME_DOMAIN_CLOCK_MONOTONIC_RAW_KHR);
 reset_init();absent=1;gpu_clock_init("VK_KHR_calibrated_timestamps");assert(!g_gpu_calibrated);absent=0;
 reset_init();enumeration_fail=1;gpu_clock_init("VK_KHR_calibrated_timestamps");assert(!g_gpu_calibrated);enumeration_fail=0;
 reset_init();offered_n=1;gpu_clock_init("VK_KHR_calibrated_timestamps");assert(!g_gpu_calibrated);
 reset_init();g_gpu_timestamp_bits=0;gpu_clock_init("VK_KHR_calibrated_timestamps");assert(!g_gpu_calibrated);
 reset_init();gpu_clock_init(NULL);assert(!g_gpu_calibrated);
 reset_init();gpu_clock_init("VK_KHR_calibrated_timestamps");
 V.props.limits.timestampPeriod=1;sample_deviation=100;sample_values[0]=30000000;sample_values[1]=1000000000;
 mono_clocks(999999990,1000000010);gpu_clock_pair p=gpu_clock_sample();assert(p.valid&&p.host==1000000000&&p.device==30000000&&cal_calls==1);
 uint64_t host;assert(gpu_clock_host_ns(&p,22000000,&host)&&host==992000000);
 uint64_t expected=nvn_timestamp_from_ns(guest_ns_at_host_time(host));
 // Moving the paired sample20ms later leaves the actual event time unchanged.
 p.device+=20000000;p.host+=20000000;assert(gpu_clock_host_ns(&p,22000000,&host));assert(nvn_timestamp_from_ns(guest_ns_at_host_time(host))==expected);
 // The old completion-now anchor would have inflated the event by20ms.
 assert(nvn_timestamp_from_ns(1012000000)-expected==nvn_timestamp_from_ns(20000000));
 //48-bit wrap and unused high result bits both preserve a nearby device delta.
 uint64_t mask=(1ull<<48)-1;p=(gpu_clock_pair){5,1000,100,1};
 assert(gpu_clock_host_ns(&p,mask-4,&host)&&host==990);
 assert(gpu_clock_host_ns(&p,(mask-4)|(0xabcdull<<48),&host)&&host==990);
 g_gpu_timestamp_bits=64;p.device=3;assert(gpu_clock_host_ns(&p,UINT64_MAX-6,&host)&&host==990);
 g_gpu_timestamp_bits=8;assert(!gpu_clock_host_ns(&p,2,&host)); // half-wrap shorter than accepted age window
 g_gpu_timestamp_bits=48;p.device=100;p.host=1000;
 assert(gpu_clock_host_ns(&p,102,&host)&&host==1002); // within maxDeviation
 assert(!gpu_clock_host_ns(&p,201,&host));
 p.device=3000000000ull;p.host=5000000000ull;assert(!gpu_clock_host_ns(&p,1,&host)); // stale beyond two seconds
 p.device=200;p.host=10;assert(!gpu_clock_host_ns(&p,1,&host)); // host underflow
 p.device=100;p.host=UINT64_MAX;assert(!gpu_clock_host_ns(&p,101,&host)); // host overflow
 p.device=100;p.host=1000;V.props.limits.timestampPeriod=52.083332061767578f;assert(gpu_clock_host_ns(&p,90,&host)&&host==480);
 V.props.limits.timestampPeriod=NAN;assert(!gpu_clock_host_ns(&p,90,&host));V.props.limits.timestampPeriod=0;assert(!gpu_clock_host_ns(&p,90,&host));V.props.limits.timestampPeriod=1;
 p.valid=0;assert(!gpu_clock_host_ns(&p,90,&host));p.valid=1;p.deviation=0;assert(!gpu_clock_host_ns(&p,90,&host));p.deviation=100001;assert(!gpu_clock_host_ns(&p,90,&host));
 sample_values[1]=1000000000;sample_deviation=100001;mono_clocks(999999990,1000000010);assert(!gpu_clock_sample().valid);
 sample_deviation=0;mono_clocks(999999990,1000000010);assert(!gpu_clock_sample().valid);
 sample_deviation=100;cal_fail=1;mono_clocks(999999990,1000000010);assert(!gpu_clock_sample().valid);cal_fail=0;
 mono_clocks(1000001000,1000001010);assert(!gpu_clock_sample().valid); // stale paired CPU timestamp
 mono_clocks(999999990,999999999);assert(gpu_clock_sample().valid); // within calibration error
 // RAW may have an unrelated epoch. Only the tightly bounded MONOTONIC bracket is used.
 g_gpu_host_domain=VK_TIME_DOMAIN_CLOCK_MONOTONIC_RAW_KHR;sample_values[1]=70000000000ull;
 raw_clocks(1000000000,69999999990ull,70000000010ull,1000000200);p=gpu_clock_sample();assert(p.valid&&p.host==1000000100&&p.deviation==200);
 raw_clocks(1000000000,69999999990ull,70000000010ull,1000200001);assert(!gpu_clock_sample().valid);
 raw_clocks(1000000000,70000001000ull,70000001010ull,1000000200);assert(!gpu_clock_sample().valid);
 // Projection happens after reconstructing host event time, preserving freeze and deterministic semantics.
 atomic_store(&g_glag,100);assert(guest_ns_at_host_time(990)==890);assert(guest_ns_at_host_time(50)==0);
 atomic_store(&g_gfz,1);atomic_store(&g_gfrozen,333);assert(guest_ns_at_host_time(990)==333&&guest_ns_at_host_time(1)==333);
 atomic_store(&g_gdet_on,1);atomic_store(&g_gdet,777);assert(guest_ns_at_host_time(990)==777);atomic_store(&g_gdet,888);assert(guest_ns_at_host_time(990)==888);
 return 0;
}
'''.replace('HELPER',helper).replace('PROJECT',project)
        target=ROOT/'target/tests';target.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='vk-counter-clock-',dir=target) as folder:
            src,exe=Path(folder)/'test.c',Path(folder)/'test';src.write_text(code, encoding='utf-8')
            subprocess.run(['cc','-std=gnu11','-g','-O1','-Wall','-Wextra','-Werror','-Wno-misleading-indentation','-fsanitize=address,undefined','-I'+str(dm/'third_party/vulkan-headers/include'),'-I'+str(dm/'gfx/nvn'),str(src),'-o',str(exe)],check=True)
            subprocess.run([str(exe)],check=True,timeout=15)

    @unittest.skipUnless(shutil.which('cc'),'a host C compiler is required')
    @unittest.skipUnless(host_cc.sanitizers(), 'a host C compiler with ASan and UBSan is required (MinGW has neither)')
    def test_timestamp_recording_stages(self):
        dm=ROOT/'nxnative/dm';core=(dm/'android/gfx_vk_core.c').read_text(encoding='utf-8')
        nvn=(dm/'gfx/nvn/include/nvn.h').read_text(encoding='utf-8')
        at=nvn.index('NVN_COUNTER_TYPE_TIMESTAMP =')
        start=nvn.rfind('typedef enum {',0,at)
        counter_enum=nvn[start:nvn.index('} NVNcounterType;',at)+len('} NVNcounterType;')]
        metal=(dm/'gfx/nvn/nvn_mtl.m').read_text(encoding='utf-8')
        self.assertIn('int top = o->type == 0x10;',metal)
        self.assertIn('ns = top ? c0 : c1;',metal)
        code=r"""
#define VK_NO_PROTOTYPES
#include <vulkan/vulkan_core.h>
#include <assert.h>
#include <stdint.h>
COUNTERS
enum { CQ=64 };
typedef struct { int kind;volatile uint32_t *val;uint32_t mode,v;uint64_t addr;uint32_t type;int q; } post_op;
typedef struct { int ncq; } vksub_t;
static vksub_t slot;static int g_cur=2,ends,writes,npost;
static VkQueryPool g_cq=(VkQueryPool)(uintptr_t)7;
static VkPipelineStageFlagBits stages[4];static uint32_t queries[4];static post_op reports[8];
static vksub_t *cur(void){return &slot;}
static VkCommandBuffer vk_cmd(void){return (VkCommandBuffer)(uintptr_t)9;}
static void vk_occ_end(void){ends++;}
static void post(post_op p){assert(npost<8);reports[npost++]=p;}
static void vkCmdWriteTimestamp(VkCommandBuffer cb,VkPipelineStageFlagBits stage,VkQueryPool pool,uint32_t query){assert(cb==vk_cmd()&&pool==g_cq&&writes<4);stages[writes]=stage;queries[writes++]=query;}
REPORT
int main(void){
 assert(NVN_COUNTER_TYPE_TIMESTAMP==0&&NVN_COUNTER_TYPE_TIMESTAMP_TOP==0x10);
 mtl_sub_counter(100,NVN_COUNTER_TYPE_TIMESTAMP_TOP);
 mtl_sub_counter(116,NVN_COUNTER_TYPE_TIMESTAMP);
 assert(writes==2&&!ends&&slot.ncq==2);
 assert(stages[0]==VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT&&stages[1]==VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);
 assert(queries[0]==2*CQ&&queries[1]==2*CQ+1);
 assert(reports[0].addr==100&&reports[0].q==2*CQ&&reports[1].q==2*CQ+1);
 mtl_sub_counter(132,NVN_COUNTER_TYPE_SAMPLES_PASSED);assert(ends==1&&writes==2&&reports[2].q==-1);
 mtl_sub_counter(148,NVN_COUNTER_TYPE_INPUT_VERTICES);assert(ends==1&&writes==2&&reports[3].q==-1);
 slot.ncq=CQ;mtl_sub_counter(164,NVN_COUNTER_TYPE_TIMESTAMP_TOP);assert(writes==2&&reports[4].q==-1);
 g_cq=VK_NULL_HANDLE;slot.ncq=0;mtl_sub_counter(180,NVN_COUNTER_TYPE_TIMESTAMP);assert(writes==2&&reports[5].q==-1);
 return 0;
}
""".replace('COUNTERS',counter_enum).replace('REPORT',function(core,'void mtl_sub_counter('))
        target=ROOT/'target/tests';target.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='vk-counter-record-',dir=target) as folder:
            src,exe=Path(folder)/'test.c',Path(folder)/'test';src.write_text(code, encoding='utf-8')
            subprocess.run(['cc','-std=c11','-Wall','-Wextra','-Werror','-fsanitize=address,undefined','-I'+str(dm/'third_party/vulkan-headers/include'),str(src),'-o',str(exe)],check=True)
            subprocess.run([str(exe)],check=True,timeout=15)

    def test_completion_integration(self):
        core=(ROOT/'nxnative/dm/android/gfx_vk_core.c').read_text(encoding='utf-8')
        completion=function(core,'static void *completion_main(')
        self.assertEqual(completion.count('gpu_clock_sample()'),1)
        self.assertIn('ts = nvn_timestamp_from_ns(guest_ns_at_host_time(host))',completion)
        self.assertIn('else if (tend && tq <= tend)',completion)
        self.assertIn('if (back < now) ts = now - back;',completion)
        self.assertIn('c->slot >= 0 && g_cq',completion)
        self.assertIn('g_gpu_timestamp_bits = qp[V.qf].timestampValidBits',core)


if __name__=='__main__':unittest.main()
