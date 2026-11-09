// Headless samples-query availability regression, with no app state or persistent device changes.
// Build shaders: glslangValidator -V probes/occlusion.{vert,frag} -o <separate files in dist>.
// Build: aarch64-linux-android31-clang -O2 probes/vkocclusion.c -o dist/vkocclusion -lvulkan
// Run: vkocclusion occlusion.vert.spv occlusion.frag.spv [iterations=1000]
// Reuses two 256-query pools, split across four submissions; resets are in upload CBs submitted
// before the main CB where a block first becomes active. Only the final submission signals completion (alternating fence/timeline waits).
// Alternates one render pass per part versus per query, precise versus approximate queries, and
// empty scopes / zero direct counts / zero indirect counts / raster discard / shader discard /
// empty scissors / visible triangles. Checks availability after completion WITHOUT WAIT or PARTIAL.
// Counts are reported, not asserted: sample counting is not a final-color-visibility test, and
// implementations may differ even with PRECISE (notably fragment-shader discard/early tests).
#include <vulkan/vulkan_core.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#define CHECK(x) do { VkResult result=(x); if(result) { fprintf(stderr,"%s: %d\n",#x,result); exit(2); } } while(0)
enum { N=387, B=256, W=16, PARTS=4 };
static VkDevice dev;
static VkPhysicalDeviceMemoryProperties memory;
static uint32_t type_for(uint32_t bits,VkMemoryPropertyFlags flags) {
    for(uint32_t i=0;i<memory.memoryTypeCount;i++)
        if((bits&(1u<<i)) && (memory.memoryTypes[i].propertyFlags&flags)==flags)return i;
    fprintf(stderr,"no memory type for %#x/%#x\n",bits,flags);exit(2);
}
static VkDeviceMemory alloc(VkMemoryRequirements r,VkMemoryPropertyFlags flags) {
    VkDeviceMemory m;VkMemoryAllocateInfo a={.sType=VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,.allocationSize=r.size,.memoryTypeIndex=type_for(r.memoryTypeBits,flags)};
    CHECK(vkAllocateMemory(dev,&a,NULL,&m));return m;
}
static VkShaderModule shader(const char *path) {
    FILE *f=fopen(path,"rb");if(!f){perror(path);exit(2);}
    fseek(f,0,SEEK_END);long n=ftell(f);rewind(f);if(n<20||(n&3)){fprintf(stderr,"invalid SPIR-V\n");exit(2);}
    uint32_t *words=malloc((size_t)n);if(!words||fread(words,1,(size_t)n,f)!=(size_t)n)exit(2);fclose(f);
    VkShaderModuleCreateInfo ci={.sType=VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,.codeSize=(size_t)n,.pCode=words};
    VkShaderModule m;CHECK(vkCreateShaderModule(dev,&ci,NULL,&m));free(words);return m;
}
static const char *names[]={"empty","direct-zero-vertices","direct-zero-instances","indirect-zero-indices","indirect-zero-instances","raster-discard","shader-discard","empty-scissor","visible-direct","visible-indirect"};
int main(int argc,char **argv) {
    if(argc<3||argc>4){fprintf(stderr,"usage: vkocclusion vertex.spv fragment.spv [iterations]\n");return 2;}
    char *end=NULL;long loops=argc==4?strtol(argv[3],&end,10):1000;
    if(loops<1||loops>100000||(argc==4&&(!*argv[3]||*end))){fprintf(stderr,"bad iteration count\n");return 2;}
    setvbuf(stdout,NULL,_IOLBF,0);
    VkApplicationInfo ai={.sType=VK_STRUCTURE_TYPE_APPLICATION_INFO,.pApplicationName="occlusion-availability",.apiVersion=VK_API_VERSION_1_2};
    VkInstanceCreateInfo ii={.sType=VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,.pApplicationInfo=&ai};
    VkInstance in;CHECK(vkCreateInstance(&ii,NULL,&in));uint32_t nd=1;VkPhysicalDevice pd;CHECK(vkEnumeratePhysicalDevices(in,&nd,&pd));if(!nd)return 2;
    VkPhysicalDeviceProperties props;vkGetPhysicalDeviceProperties(pd,&props);vkGetPhysicalDeviceMemoryProperties(pd,&memory);
    VkPhysicalDeviceTimelineSemaphoreFeatures timeline={.sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES};
    VkPhysicalDeviceFeatures2 all={.sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,.pNext=&timeline};vkGetPhysicalDeviceFeatures2(pd,&all);
    VkPhysicalDeviceFeatures features=all.features;
    printf("device %s driver %#x precise %u timeline %u; %ld iterations x %d queries\n",props.deviceName,props.driverVersion,features.occlusionQueryPrecise,timeline.timelineSemaphore,loops,N);
    uint32_t nq=32;VkQueueFamilyProperties qprops[32];vkGetPhysicalDeviceQueueFamilyProperties(pd,&nq,qprops);
    uint32_t family=0;while(family<nq&&!(qprops[family].queueFlags&VK_QUEUE_GRAPHICS_BIT))family++;if(family==nq)return 2;
    float priority=1;VkDeviceQueueCreateInfo qi={.sType=VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,.queueFamilyIndex=family,.queueCount=1,.pQueuePriorities=&priority};
    VkPhysicalDeviceFeatures enabled={.occlusionQueryPrecise=features.occlusionQueryPrecise};
    VkDeviceCreateInfo di={.sType=VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,.pNext=&timeline,.queueCreateInfoCount=1,.pQueueCreateInfos=&qi,.pEnabledFeatures=&enabled};
    CHECK(vkCreateDevice(pd,&di,NULL,&dev));VkQueue queue;vkGetDeviceQueue(dev,family,0,&queue);
    VkImageCreateInfo ici={.sType=VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,.imageType=VK_IMAGE_TYPE_2D,.format=VK_FORMAT_R8G8B8A8_UNORM,.extent={W,W,1},.mipLevels=1,.arrayLayers=1,.samples=VK_SAMPLE_COUNT_1_BIT,.tiling=VK_IMAGE_TILING_OPTIMAL,.usage=VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT};
    VkImage image;CHECK(vkCreateImage(dev,&ici,NULL,&image));VkMemoryRequirements mr;vkGetImageMemoryRequirements(dev,image,&mr);VkDeviceMemory imem=alloc(mr,0);CHECK(vkBindImageMemory(dev,image,imem,0));
    VkImageViewCreateInfo vi={.sType=VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,.image=image,.viewType=VK_IMAGE_VIEW_TYPE_2D,.format=ici.format,.subresourceRange={VK_IMAGE_ASPECT_COLOR_BIT,0,1,0,1}};
    VkImageView view;CHECK(vkCreateImageView(dev,&vi,NULL,&view));
    VkAttachmentDescription att={.format=ici.format,.samples=VK_SAMPLE_COUNT_1_BIT,.loadOp=VK_ATTACHMENT_LOAD_OP_DONT_CARE,.storeOp=VK_ATTACHMENT_STORE_OP_DONT_CARE,.stencilLoadOp=VK_ATTACHMENT_LOAD_OP_DONT_CARE,.stencilStoreOp=VK_ATTACHMENT_STORE_OP_DONT_CARE,.initialLayout=VK_IMAGE_LAYOUT_UNDEFINED,.finalLayout=VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    VkAttachmentReference ref={0,VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    VkSubpassDescription sub={.pipelineBindPoint=VK_PIPELINE_BIND_POINT_GRAPHICS,.colorAttachmentCount=1,.pColorAttachments=&ref};
    VkSubpassDependency deps[2]={
      {VK_SUBPASS_EXTERNAL,0,VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,VK_DEPENDENCY_BY_REGION_BIT},
      {0,VK_SUBPASS_EXTERNAL,VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,VK_DEPENDENCY_BY_REGION_BIT}};
    VkRenderPassCreateInfo ri={.sType=VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,.attachmentCount=1,.pAttachments=&att,.subpassCount=1,.pSubpasses=&sub,.dependencyCount=2,.pDependencies=deps};
    VkRenderPass rp;CHECK(vkCreateRenderPass(dev,&ri,NULL,&rp));
    VkFramebufferCreateInfo fi={.sType=VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,.renderPass=rp,.attachmentCount=1,.pAttachments=&view,.width=W,.height=W,.layers=1};
    VkFramebuffer fb;CHECK(vkCreateFramebuffer(dev,&fi,NULL,&fb));
    VkBufferCreateInfo bi={.sType=VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,.size=256,.usage=VK_BUFFER_USAGE_INDEX_BUFFER_BIT|VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT};
    VkBuffer buf;CHECK(vkCreateBuffer(dev,&bi,NULL,&buf));vkGetBufferMemoryRequirements(dev,buf,&mr);
    VkDeviceMemory bmem=alloc(mr,VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);CHECK(vkBindBufferMemory(dev,buf,bmem,0));
    unsigned char *mapped;CHECK(vkMapMemory(dev,bmem,0,VK_WHOLE_SIZE,0,(void**)&mapped));memset(mapped,0,256);
    uint16_t idx[3]={0,1,2};memcpy(mapped,idx,sizeof idx);
    VkDrawIndexedIndirectCommand commands[3]={{0,1,0,0,0},{3,0,0,0,0},{3,1,0,0,0}};memcpy(mapped+16,commands,sizeof commands);
    VkPipelineLayoutCreateInfo pli={.sType=VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};VkPipelineLayout pl;CHECK(vkCreatePipelineLayout(dev,&pli,NULL,&pl));
    VkShaderModule vs=shader(argv[1]),fs=shader(argv[2]);
    VkSpecializationMapEntry map={0,0,sizeof(VkBool32)};VkBool32 kill=0;VkSpecializationInfo spec={1,&map,sizeof kill,&kill};
    VkPipelineShaderStageCreateInfo stages[2]={
      {.sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,.stage=VK_SHADER_STAGE_VERTEX_BIT,.module=vs,.pName="main"},
      {.sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,.stage=VK_SHADER_STAGE_FRAGMENT_BIT,.module=fs,.pName="main",.pSpecializationInfo=&spec}};
    VkPipelineVertexInputStateCreateInfo vertex={.sType=VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    VkPipelineInputAssemblyStateCreateInfo assembly={.sType=VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,.topology=VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST};
    VkViewport vp={0,0,W,W,0,1};VkRect2D scissor={{0,0},{W,W}};
    VkPipelineViewportStateCreateInfo viewport={.sType=VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,.viewportCount=1,.pViewports=&vp,.scissorCount=1,.pScissors=&scissor};
    VkPipelineRasterizationStateCreateInfo raster={.sType=VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,.polygonMode=VK_POLYGON_MODE_FILL,.lineWidth=1};
    VkPipelineMultisampleStateCreateInfo ms={.sType=VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,.rasterizationSamples=VK_SAMPLE_COUNT_1_BIT};
    VkPipelineColorBlendAttachmentState ba={.colorWriteMask=15};
    VkPipelineColorBlendStateCreateInfo blend={.sType=VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,.attachmentCount=1,.pAttachments=&ba};
    VkDynamicState dynstate=VK_DYNAMIC_STATE_SCISSOR;VkPipelineDynamicStateCreateInfo dyn={.sType=VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,.dynamicStateCount=1,.pDynamicStates=&dynstate};
    VkGraphicsPipelineCreateInfo gp={.sType=VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,.stageCount=2,.pStages=stages,.pVertexInputState=&vertex,.pInputAssemblyState=&assembly,.pViewportState=&viewport,.pRasterizationState=&raster,.pMultisampleState=&ms,.pColorBlendState=&blend,.pDynamicState=&dyn,.layout=pl,.renderPass=rp,.basePipelineIndex=-1};
    VkPipeline pipeline[3];for(int k=0;k<3;k++){raster.rasterizerDiscardEnable=k==1;kill=k==2;CHECK(vkCreateGraphicsPipelines(dev,VK_NULL_HANDLE,1,&gp,NULL,&pipeline[k]));}
    VkQueryPoolCreateInfo qpi={.sType=VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,.queryType=VK_QUERY_TYPE_OCCLUSION,.queryCount=B};
    VkQueryPool pool[2];CHECK(vkCreateQueryPool(dev,&qpi,NULL,&pool[0]));CHECK(vkCreateQueryPool(dev,&qpi,NULL,&pool[1]));
    VkCommandPoolCreateInfo pci={.sType=VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,.queueFamilyIndex=family};
    VkCommandPool cp;CHECK(vkCreateCommandPool(dev,&pci,NULL,&cp));VkCommandBuffer cb[PARTS*2];
    VkCommandBufferAllocateInfo cai={.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,.commandPool=cp,.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY,.commandBufferCount=PARTS*2};
    CHECK(vkAllocateCommandBuffers(dev,&cai,cb));VkCommandBufferBeginInfo cbi={.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,.flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
    VkFenceCreateInfo fci={.sType=VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};VkFence fence;CHECK(vkCreateFence(dev,&fci,NULL,&fence));
    VkSemaphore sem=VK_NULL_HANDLE;
    if(timeline.timelineSemaphore){VkSemaphoreTypeCreateInfo sti={.sType=VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,.semaphoreType=VK_SEMAPHORE_TYPE_TIMELINE};
      VkSemaphoreCreateInfo sci={.sType=VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,.pNext=&sti};CHECK(vkCreateSemaphore(dev,&sci,NULL,&sem));}
    VkRenderPassBeginInfo rbi={.sType=VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,.renderPass=rp,.framebuffer=fb,.renderArea={{0,0},{W,W}}};
    const int bounds[PARTS+1]={0,1,169,357,N};
    uint64_t checked=0,low[2][10],high[2][10]={{0}};memset(low,255,sizeof low);
    for(long iteration=0;iteration<loops;iteration++){
      int per_query=(int)(iteration&1);int precise=features.occlusionQueryPrecise&&((iteration>>1)&1);int timed=sem&&((iteration>>2)&1);uint64_t serial=(uint64_t)iteration+1;
      CHECK(vkResetCommandPool(dev,cp,0));CHECK(vkResetFences(dev,1,&fence));
      for(int part=0;part<PARTS;part++){
        VkCommandBuffer up=cb[2*part],main=cb[2*part+1];CHECK(vkBeginCommandBuffer(up,&cbi));CHECK(vkBeginCommandBuffer(main,&cbi));
        if(!per_query)vkCmdBeginRenderPass(main,&rbi,VK_SUBPASS_CONTENTS_INLINE);
        for(int q=bounds[part];q<bounds[part+1];q++){
          if(!(q%B))vkCmdResetQueryPool(up,pool[q/B],0,B);
          if(per_query)vkCmdBeginRenderPass(main,&rbi,VK_SUBPASS_CONTENTS_INLINE);
          int mode=(q+(int)(iteration%10))%10;scissor.extent.width=mode==7?0:W;
          vkCmdBindPipeline(main,VK_PIPELINE_BIND_POINT_GRAPHICS,pipeline[mode==5?1:mode==6?2:0]);
          vkCmdSetScissor(main,0,1,&scissor);vkCmdBindIndexBuffer(main,buf,0,VK_INDEX_TYPE_UINT16);
          vkCmdBeginQuery(main,pool[q/B],(uint32_t)(q%B),precise?VK_QUERY_CONTROL_PRECISE_BIT:0);
          if(mode==3||mode==4||mode==9)vkCmdDrawIndexedIndirect(main,buf,16+(mode==9?2:mode-3)*sizeof commands[0],1,sizeof commands[0]);
          else if(mode!=0)vkCmdDraw(main,mode==1?0:3,mode==2?0:1,0,0);
          vkCmdEndQuery(main,pool[q/B],(uint32_t)(q%B));
          if(per_query)vkCmdEndRenderPass(main);
        }
        if(!per_query)vkCmdEndRenderPass(main);CHECK(vkEndCommandBuffer(up));CHECK(vkEndCommandBuffer(main));
        VkTimelineSemaphoreSubmitInfo ts={.sType=VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO,.signalSemaphoreValueCount=1,.pSignalSemaphoreValues=&serial};
        VkCommandBuffer submit[2]={up,main};VkSubmitInfo si={.sType=VK_STRUCTURE_TYPE_SUBMIT_INFO,.commandBufferCount=2,.pCommandBuffers=submit};
        if(timed&&part==PARTS-1){si.pNext=&ts;si.signalSemaphoreCount=1;si.pSignalSemaphores=&sem;}
        CHECK(vkQueueSubmit(queue,1,&si,!timed&&part==PARTS-1?fence:VK_NULL_HANDLE));
      }
      if(timed){VkSemaphoreWaitInfo wi={.sType=VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,.semaphoreCount=1,.pSemaphores=&sem,.pValues=&serial};CHECK(vkWaitSemaphores(dev,&wi,UINT64_MAX));}
      else CHECK(vkWaitForFences(dev,1,&fence,VK_TRUE,UINT64_MAX));
      for(int q=0;q<N;q++){
        uint64_t value=UINT64_MAX;VkResult r=vkGetQueryPoolResults(dev,pool[q/B],(uint32_t)(q%B),1,sizeof value,&value,sizeof value,VK_QUERY_RESULT_64_BIT);
        int mode=(q+(int)(iteration%10))%10;
        if(r){uint64_t avail[2]={UINT64_MAX,UINT64_MAX};VkResult r2=vkGetQueryPoolResults(dev,pool[q/B],(uint32_t)(q%B),1,sizeof avail,avail,sizeof avail,VK_QUERY_RESULT_64_BIT|VK_QUERY_RESULT_WITH_AVAILABILITY_BIT);
          printf("FAIL iteration=%ld query=%d mode=%s per-query-pass=%d precise=%d timeline=%d result=%d second=%d samples=%llu availability=%llu\n",iteration,q,names[mode],per_query,precise,timed,r,r2,(unsigned long long)avail[0],(unsigned long long)avail[1]);return 1;}
        if(value<low[precise][mode])low[precise][mode]=value;
        if(value>high[precise][mode])high[precise][mode]=value;
        checked++;
      }
      if(!(iteration%100))printf("checked %llu queries after %ld iterations\n",(unsigned long long)checked,iteration+1);
    }
    for(int p=0;p<2;p++)for(int m=0;m<10;m++)if(low[p][m]!=UINT64_MAX)printf("count precise=%d mode=%s range=%llu..%llu\n",p,names[m],(unsigned long long)low[p][m],(unsigned long long)high[p][m]);
    printf("PASS %llu available query results; all ten modes, split reset/main CBs, pool reuse, per-query/per-part passes, precise/approximate, fence/timeline\n",(unsigned long long)checked);
    if(sem)vkDestroySemaphore(dev,sem,NULL);
    vkDestroyFence(dev,fence,NULL);vkDestroyCommandPool(dev,cp,NULL);for(int i=0;i<2;i++)vkDestroyQueryPool(dev,pool[i],NULL);
    for(int i=0;i<3;i++)vkDestroyPipeline(dev,pipeline[i],NULL);vkDestroyShaderModule(dev,vs,NULL);vkDestroyShaderModule(dev,fs,NULL);vkDestroyPipelineLayout(dev,pl,NULL);
    vkUnmapMemory(dev,bmem);vkDestroyBuffer(dev,buf,NULL);vkFreeMemory(dev,bmem,NULL);vkDestroyFramebuffer(dev,fb,NULL);vkDestroyRenderPass(dev,rp,NULL);vkDestroyImageView(dev,view,NULL);vkDestroyImage(dev,image,NULL);vkFreeMemory(dev,imem,NULL);vkDestroyDevice(dev,NULL);vkDestroyInstance(in,NULL);return 0;
}
