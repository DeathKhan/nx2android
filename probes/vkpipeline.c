// Compile-only executable statistics for audited Bananza14a8 bindings. No dispatch.
// vkpipeline module.spv [second.spv ...]; compiler IR is saved beside each module.
#include <vulkan/vulkan_core.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define CK(x) do { VkResult r=(x);if(r) { fprintf(stderr,"%s: %d\n",#x,r);return 2; } }while(0)
int main(int argc,char **argv) {
    if(argc<2)return 2;
    VkApplicationInfo ai={VK_STRUCTURE_TYPE_APPLICATION_INFO,NULL,"pipeline-statistics",1,NULL,0,VK_API_VERSION_1_2};
    VkInstanceCreateInfo ici={VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,NULL,0,&ai};VkInstance in;CK(vkCreateInstance(&ici,NULL,&in));
    uint32_t n=1;VkPhysicalDevice pd;CK(vkEnumeratePhysicalDevices(in,&n,&pd));if(!n)return 2;
    VkPhysicalDeviceProperties prop;vkGetPhysicalDeviceProperties(pd,&prop);printf("device %s\n",prop.deviceName);
    const char *ext=VK_KHR_PIPELINE_EXECUTABLE_PROPERTIES_EXTENSION_NAME;
    uint32_t ne=0;CK(vkEnumerateDeviceExtensionProperties(pd,NULL,&ne,NULL));VkExtensionProperties *ep=calloc(ne,sizeof *ep);CK(vkEnumerateDeviceExtensionProperties(pd,NULL,&ne,ep));
    int have=0;for(uint32_t i=0;i<ne;i++)have|=!strcmp(ext,ep[i].extensionName);free(ep);if(!have){puts("pipeline executable extension absent");return 3;}
    VkPhysicalDevicePipelineExecutablePropertiesFeaturesKHR pe={VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_EXECUTABLE_PROPERTIES_FEATURES_KHR};
    VkPhysicalDeviceFeatures2 feats={VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,&pe};vkGetPhysicalDeviceFeatures2(pd,&feats);
    if(!pe.pipelineExecutableInfo){puts("pipeline executable feature absent");return 3;}
    uint32_t nq=16;VkQueueFamilyProperties qp[16];vkGetPhysicalDeviceQueueFamilyProperties(pd,&nq,qp);uint32_t qf=0;
    while(qf<nq && !(qp[qf].queueFlags&VK_QUEUE_COMPUTE_BIT))qf++;if(qf==nq)return 2;
    float priority=1;VkDeviceQueueCreateInfo qi={VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,NULL,0,qf,1,&priority};
    VkDeviceCreateInfo di={VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,&pe,0,1,&qi,0,NULL,1,&ext,&feats.features};VkDevice dev;CK(vkCreateDevice(pd,&di,NULL,&dev));
    PFN_vkGetPipelineExecutablePropertiesKHR props=(PFN_vkGetPipelineExecutablePropertiesKHR)vkGetDeviceProcAddr(dev,"vkGetPipelineExecutablePropertiesKHR");
    PFN_vkGetPipelineExecutableStatisticsKHR stats=(PFN_vkGetPipelineExecutableStatisticsKHR)vkGetDeviceProcAddr(dev,"vkGetPipelineExecutableStatisticsKHR");
    PFN_vkGetPipelineExecutableInternalRepresentationsKHR ir=(PFN_vkGetPipelineExecutableInternalRepresentationsKHR)vkGetDeviceProcAddr(dev,"vkGetPipelineExecutableInternalRepresentationsKHR");
    if(!props||!stats||!ir)return 3;
    VkDescriptorType types[]={VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER};
    VkDescriptorSetLayoutBinding binding[6];for(int i=0;i<6;i++)binding[i]=(VkDescriptorSetLayoutBinding){(uint32_t)i,types[i],1,VK_SHADER_STAGE_COMPUTE_BIT,NULL};
    VkDescriptorSetLayoutCreateInfo li={VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,NULL,0,6,binding};VkDescriptorSetLayout layout;CK(vkCreateDescriptorSetLayout(dev,&li,NULL,&layout));
    VkPipelineLayoutCreateInfo pli={VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,NULL,0,1,&layout};VkPipelineLayout pl;CK(vkCreatePipelineLayout(dev,&pli,NULL,&pl));
    for(int module=1;module<argc;module++) {
        FILE *file=fopen(argv[module],"rb");if(!file){perror(argv[module]);return 2;}fseek(file,0,SEEK_END);long size=ftell(file);rewind(file);
        if(size<20 || (size&3))return 2;uint32_t *code=malloc((size_t)size);if(fread(code,1,(size_t)size,file)!=(size_t)size)return 2;fclose(file);
        printf("MODULE %s bytes %ld\n",argv[module],size);
        VkShaderModuleCreateInfo smi={VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,NULL,0,(size_t)size,code};VkShaderModule sm;CK(vkCreateShaderModule(dev,&smi,NULL,&sm));free(code);
        VkComputePipelineCreateInfo ci={VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,NULL,VK_PIPELINE_CREATE_CAPTURE_STATISTICS_BIT_KHR|VK_PIPELINE_CREATE_CAPTURE_INTERNAL_REPRESENTATIONS_BIT_KHR,{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,NULL,0,VK_SHADER_STAGE_COMPUTE_BIT,sm,"main",NULL},pl,VK_NULL_HANDLE,-1};
        VkPipeline pipeline;CK(vkCreateComputePipelines(dev,VK_NULL_HANDLE,1,&ci,NULL,&pipeline));
        VkPipelineInfoKHR pi={VK_STRUCTURE_TYPE_PIPELINE_INFO_KHR,NULL,pipeline};uint32_t nx=0;CK(props(dev,&pi,&nx,NULL));
        VkPipelineExecutablePropertiesKHR *xp=calloc(nx,sizeof *xp);for(uint32_t i=0;i<nx;i++)xp[i].sType=VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_PROPERTIES_KHR;CK(props(dev,&pi,&nx,xp));
        for(uint32_t x=0;x<nx;x++) {
            printf("EXEC %u %s subgroup %u: %s\n",x,xp[x].name,xp[x].subgroupSize,xp[x].description);
            VkPipelineExecutableInfoKHR ei={VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_INFO_KHR,NULL,pipeline,x};uint32_t ns=0;CK(stats(dev,&ei,&ns,NULL));
            VkPipelineExecutableStatisticKHR *sp=calloc(ns,sizeof *sp);for(uint32_t i=0;i<ns;i++)sp[i].sType=VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_STATISTIC_KHR;CK(stats(dev,&ei,&ns,sp));
            for(uint32_t i=0;i<ns;i++) {
                printf("STAT %s = ",sp[i].name);
                switch(sp[i].format) {
                case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_BOOL32_KHR:printf("%u",sp[i].value.b32);break;
                case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_INT64_KHR:printf("%lld",(long long)sp[i].value.i64);break;
                case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_UINT64_KHR:printf("%llu",(unsigned long long)sp[i].value.u64);break;
                case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_FLOAT64_KHR:printf("%.17g",sp[i].value.f64);break;
                default:printf("unknown format %u",sp[i].format);
                }
                printf(" (%s)\n",sp[i].description);
            }free(sp);
            uint32_t ni=0;CK(ir(dev,&ei,&ni,NULL));VkPipelineExecutableInternalRepresentationKHR *reps=calloc(ni,sizeof *reps);
            for(uint32_t i=0;i<ni;i++)reps[i].sType=VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_INTERNAL_REPRESENTATION_KHR;
            CK(ir(dev,&ei,&ni,reps));
            for(uint32_t i=0;i<ni;i++){if(reps[i].dataSize>(256u<<20))return 2;reps[i].pData=calloc(1,reps[i].dataSize);}
            CK(ir(dev,&ei,&ni,reps));
            for(uint32_t i=0;i<ni;i++) {
                char path[1024];snprintf(path,sizeof path,"%s.exec%u.ir%u.%s",argv[module],x,i,reps[i].isText?"txt":"bin");
                file=fopen(path,"wb");if(!file){perror(path);return 2;}fwrite(reps[i].pData,1,reps[i].dataSize,file);fclose(file);
                printf("IR %s bytes %zu text %u saved %s (%s)\n",reps[i].name,reps[i].dataSize,reps[i].isText,path,reps[i].description);free(reps[i].pData);
            }free(reps);
        }free(xp);vkDestroyPipeline(dev,pipeline,NULL);vkDestroyShaderModule(dev,sm,NULL);
    }
    vkDestroyPipelineLayout(dev,pl,NULL);vkDestroyDescriptorSetLayout(dev,layout,NULL);vkDestroyDevice(dev,NULL);vkDestroyInstance(in,NULL);return 0;
}

