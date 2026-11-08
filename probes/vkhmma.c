// One-shot device validation of the actual translated shared/shuffle HMMA helpers.
// Generate compare.spv with hmma_compare_spv.py, then run: vkhmma compare.spv
// No app or persistent device state is changed.
#include <vulkan/vulkan_core.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define CHECK(x) do { VkResult r=(x); if(r) { fprintf(stderr,"%s failed: %d\n",#x,r); return 2; } } while(0)
enum { GROUPS=128, THREADS=128, WORDS=14, LANES=GROUPS*THREADS };
static uint32_t random_word(uint32_t *s) { *s^=*s<<13;*s^=*s>>17;*s^=*s<<5;return *s; }
static uint32_t finite_half(uint32_t x) { x&=65535;return (x&0x7c00)==0x7c00?x&~0x7c00u:x; }
static int half_nan(uint32_t x) { return (x&0x7c00)==0x7c00 && (x&1023); }
static int nan_only_difference(uint32_t a,uint32_t b) {
    for(int k=0;k<2;k++) { uint32_t x=a&65535,y=b&65535;if(x!=y && !(half_nan(x)&&half_nan(y)))return 0;a>>=16;b>>=16; } return 1;
}
int main(int argc,char **argv) {
    if(argc!=2) { fprintf(stderr,"usage: vkhmma compare.spv\n");return 2; }
    FILE *f=fopen(argv[1],"rb");if(!f) { perror(argv[1]);return 2; }
    fseek(f,0,SEEK_END);long n=ftell(f);rewind(f);
    uint32_t *code=malloc((size_t)n);if(n<20 || (n&3) || fread(code,1,(size_t)n,f)!=(size_t)n)return 2;fclose(f);
    VkApplicationInfo ai={VK_STRUCTURE_TYPE_APPLICATION_INFO,NULL,"hmma-equivalence",1,NULL,0,VK_API_VERSION_1_2};
    VkInstanceCreateInfo ici={VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,NULL,0,&ai};
    VkInstance in;CHECK(vkCreateInstance(&ici,NULL,&in));
    uint32_t np=1;VkPhysicalDevice pd;CHECK(vkEnumeratePhysicalDevices(in,&np,&pd));if(!np)return 2;
    VkPhysicalDeviceSubgroupProperties sg={VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES};
    VkPhysicalDeviceProperties2 props={VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,&sg};vkGetPhysicalDeviceProperties2(pd,&props);
    printf("device %s advertised subgroup %u; local %u; groups %u\n",props.properties.deviceName,sg.subgroupSize,THREADS,GROUPS);
    if(!(sg.supportedStages&VK_SHADER_STAGE_COMPUTE_BIT) || !(sg.supportedOperations&VK_SUBGROUP_FEATURE_SHUFFLE_BIT))return 2;
    uint32_t nq=16;VkQueueFamilyProperties qp[16];vkGetPhysicalDeviceQueueFamilyProperties(pd,&nq,qp);
    uint32_t qf=0;while(qf<nq && !(qp[qf].queueFlags&VK_QUEUE_COMPUTE_BIT))qf++;if(qf==nq)return 2;
    float priority=1;VkDeviceQueueCreateInfo qi={VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,NULL,0,qf,1,&priority};
    VkPhysicalDeviceFeatures feats;vkGetPhysicalDeviceFeatures(pd,&feats);
    VkDeviceCreateInfo di={VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,NULL,0,1,&qi,0,NULL,0,NULL,&feats};
    VkDevice dev;CHECK(vkCreateDevice(pd,&di,NULL,&dev));
    VkQueue queue;vkGetDeviceQueue(dev,qf,0,&queue);
    VkDeviceSize bytes=(VkDeviceSize)LANES*WORDS*4;
    VkBufferCreateInfo bi={VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,NULL,0,bytes,VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,VK_SHARING_MODE_EXCLUSIVE};
    VkBuffer buf;CHECK(vkCreateBuffer(dev,&bi,NULL,&buf));
    VkMemoryRequirements mr;vkGetBufferMemoryRequirements(dev,buf,&mr);
    VkPhysicalDeviceMemoryProperties mp;vkGetPhysicalDeviceMemoryProperties(pd,&mp);
    uint32_t mt=0;for(;mt<mp.memoryTypeCount;mt++)if((mr.memoryTypeBits&(1u<<mt)) && (mp.memoryTypes[mt].propertyFlags&(VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT))==(VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT))break;
    if(mt==mp.memoryTypeCount)return 2;
    VkMemoryAllocateInfo ma={VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,NULL,mr.size,mt};
    VkDeviceMemory mem;CHECK(vkAllocateMemory(dev,&ma,NULL,&mem));CHECK(vkBindBufferMemory(dev,buf,mem,0));
    uint32_t *mapped;CHECK(vkMapMemory(dev,mem,0,VK_WHOLE_SIZE,0,(void**)&mapped));
    static const uint32_t edge[]={0x7c00,0xfc00,0x7e01,0xfe7f,0x7c01,0xfc01,0,0x8000};
    uint32_t seed=3187;
    for(uint32_t i=0;i<LANES;i++)for(uint32_t k=0;k<WORDS;k++) {
        uint32_t lo=finite_half(random_word(&seed)),hi=finite_half(random_word(&seed));
        uint32_t value=lo|(hi<<16),group=i/THREADS;
        if(group==0)value=(i&1)?0x80008000u:0;
        if(group==1)value=edge[(i+k)%8]|(edge[(i+k+3)%8]<<16);
        if(group==2)value=(i&1)?0xfbfffbffu:0x7bff7bffu;
        mapped[i*WORDS+k]=k<8?value:0xccccccccu;
    }
    VkDescriptorSetLayoutBinding lb={0,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,VK_SHADER_STAGE_COMPUTE_BIT,NULL};
    VkDescriptorSetLayoutCreateInfo li={VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,NULL,0,1,&lb};
    VkDescriptorSetLayout layout;CHECK(vkCreateDescriptorSetLayout(dev,&li,NULL,&layout));
    VkDescriptorPoolSize ps={VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1};
    VkDescriptorPoolCreateInfo pi={VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,NULL,0,1,1,&ps};
    VkDescriptorPool pool;CHECK(vkCreateDescriptorPool(dev,&pi,NULL,&pool));
    VkDescriptorSetAllocateInfo sa={VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,NULL,pool,1,&layout};
    VkDescriptorSet set;CHECK(vkAllocateDescriptorSets(dev,&sa,&set));
    VkDescriptorBufferInfo db={buf,0,bytes};
    VkWriteDescriptorSet wr={VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,NULL,set,0,0,1,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,NULL,&db,NULL};
    vkUpdateDescriptorSets(dev,1,&wr,0,NULL);
    VkPipelineLayoutCreateInfo pli={VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,NULL,0,1,&layout};
    VkPipelineLayout pl;CHECK(vkCreatePipelineLayout(dev,&pli,NULL,&pl));
    VkShaderModuleCreateInfo smi={VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,NULL,0,(size_t)n,code};
    VkShaderModule sm;CHECK(vkCreateShaderModule(dev,&smi,NULL,&sm));
    VkComputePipelineCreateInfo cpi={VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,NULL,0,{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,NULL,0,VK_SHADER_STAGE_COMPUTE_BIT,sm,"main",NULL},pl,VK_NULL_HANDLE,-1};
    VkPipeline pipeline;CHECK(vkCreateComputePipelines(dev,VK_NULL_HANDLE,1,&cpi,NULL,&pipeline));
    VkCommandPoolCreateInfo pci={VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,NULL,0,qf};
    VkCommandPool cp;CHECK(vkCreateCommandPool(dev,&pci,NULL,&cp));
    VkCommandBufferAllocateInfo cai={VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,NULL,cp,VK_COMMAND_BUFFER_LEVEL_PRIMARY,1};
    VkCommandBuffer cb;CHECK(vkAllocateCommandBuffers(dev,&cai,&cb));
    VkCommandBufferBeginInfo cbi={VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};CHECK(vkBeginCommandBuffer(cb,&cbi));
    vkCmdBindPipeline(cb,VK_PIPELINE_BIND_POINT_COMPUTE,pipeline);
    vkCmdBindDescriptorSets(cb,VK_PIPELINE_BIND_POINT_COMPUTE,pl,0,1,&set,0,NULL);
    vkCmdDispatch(cb,GROUPS,1,1);
    VkMemoryBarrier mb={VK_STRUCTURE_TYPE_MEMORY_BARRIER,NULL,VK_ACCESS_SHADER_WRITE_BIT,VK_ACCESS_HOST_READ_BIT};
    vkCmdPipelineBarrier(cb,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_HOST_BIT,0,1,&mb,0,NULL,0,NULL);
    CHECK(vkEndCommandBuffer(cb));
    VkSubmitInfo si={VK_STRUCTURE_TYPE_SUBMIT_INFO,NULL,0,NULL,NULL,1,&cb,0,NULL};CHECK(vkQueueSubmit(queue,1,&si,VK_NULL_HANDLE));CHECK(vkQueueWaitIdle(queue));
    unsigned bad=0,nanbad=0,invalid=0,minsg=~0u,maxsg=0;
    for(unsigned i=0;i<LANES;i++) {
        uint32_t *w=&mapped[i*WORDS],width=w[12];
        if(width<minsg)minsg=width;
        if(width>maxsg)maxsg=width;
        invalid+=width!=sg.subgroupSize || w[13]>=width;
        for(unsigned k=0;k<2;k++)if(w[8+k]!=w[10+k]) {
            if(bad<8)printf("mismatch group=%u local=%u word=%u shared=%08x shuffle=%08x\n",i/THREADS,i%THREADS,k,w[8+k],w[10+k]);
            bad++;nanbad+=nan_only_difference(w[8+k],w[10+k]);
        }
    }
    printf("RESULT packed outputs=%u bit mismatches=%u (NaN-payload-only=%u); actual subgroup=%u..%u metadata errors=%u\n",LANES*2,bad,nanbad,minsg,maxsg,invalid);
    vkDestroyCommandPool(dev,cp,NULL);vkDestroyPipeline(dev,pipeline,NULL);vkDestroyShaderModule(dev,sm,NULL);
    vkDestroyPipelineLayout(dev,pl,NULL);vkDestroyDescriptorPool(dev,pool,NULL);vkDestroyDescriptorSetLayout(dev,layout,NULL);
    vkUnmapMemory(dev,mem);vkDestroyBuffer(dev,buf,NULL);vkFreeMemory(dev,mem,NULL);vkDestroyDevice(dev,NULL);vkDestroyInstance(in,NULL);free(code);
    return bad||invalid?1:0;
}

