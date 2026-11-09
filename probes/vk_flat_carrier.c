// Headless Flat-float vs Flat-uint transport, based on vkocclusion.c's graphics setup.
// Runtime storage-buffer word4 -> VS -> flat varying -> FS -> exact RGBA32_UINT readback.
// No game files or persistent settings. Usage: vk_flat_carrier vertex.spv fragment.spv
#include <vulkan/vulkan_core.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define CHECK(call) do { VkResult rc_=(call); if(rc_!=VK_SUCCESS) { fprintf(stderr,"%s: %d\n",#call,rc_); goto cleanup; } } while(0)
static const uint32_t payloads[]={
 0,1,2,3,0x7f,0xff,0x100,0x7fff,0x8000,0xffff,
 0x001fffff,0x007fffff,0x00800000,0x80000000,0x80000001,0x807fffff,
 0x102014c2,0x3f000000,0x3f800000,0x40000000,0xbf800000,
 0x7f800000,0xff800000,0x7fc00000
};
enum { COUNT=sizeof payloads/sizeof *payloads, STRIDE=21 };
static uint32_t memory_type(VkPhysicalDevice physical,uint32_t bits,VkMemoryPropertyFlags flags) {
 VkPhysicalDeviceMemoryProperties m;vkGetPhysicalDeviceMemoryProperties(physical,&m);
 for(uint32_t i=0;i<m.memoryTypeCount;i++)
  if((bits&(1u<<i))&&(m.memoryTypes[i].propertyFlags&flags)==flags)return i;
 return UINT32_MAX;
}
static int make_shader(VkDevice device,const char *path,VkShaderModule *out) {
 FILE *f=fopen(path,"rb");if(!f){perror(path);return 0;}
 if(fseek(f,0,SEEK_END)){fclose(f);return 0;}long size=ftell(f);
 if(size<20||size>(1<<20)||(size&3)||fseek(f,0,SEEK_SET)){fclose(f);return 0;}
 uint32_t *w=malloc((size_t)size);if(!w){fclose(f);return 0;}
 int ok=fread(w,1,(size_t)size,f)==(size_t)size&&w[0]==0x07230203u;fclose(f);
 if(ok){VkShaderModuleCreateInfo ci={.sType=VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,.codeSize=(size_t)size,.pCode=w};
  ok=vkCreateShaderModule(device,&ci,NULL,out)==VK_SUCCESS;}
 free(w);return ok;
}
int main(int argc,char **argv) {
 if(argc!=3){fprintf(stderr,"usage: vk_flat_carrier vertex.spv fragment.spv\n");return 2;}
 int result=2,submitted=0;VkInstance instance=VK_NULL_HANDLE;VkPhysicalDevice physical=VK_NULL_HANDLE;
 VkDevice device=VK_NULL_HANDLE;VkQueue queue=VK_NULL_HANDLE;VkBuffer buffer=VK_NULL_HANDLE;
 VkDeviceMemory bmem=VK_NULL_HANDLE,imem=VK_NULL_HANDLE;void *mapped=NULL;
 VkImage image=VK_NULL_HANDLE;VkImageView view=VK_NULL_HANDLE;VkRenderPass rp=VK_NULL_HANDLE;
 VkFramebuffer fb=VK_NULL_HANDLE;VkDescriptorSetLayout dsl=VK_NULL_HANDLE;VkDescriptorPool pool=VK_NULL_HANDLE;
 VkPipelineLayout layout=VK_NULL_HANDLE;VkPipeline pipeline=VK_NULL_HANDLE;VkShaderModule vs=VK_NULL_HANDLE,fs=VK_NULL_HANDLE;
 VkCommandPool cp=VK_NULL_HANDLE;VkFence fence=VK_NULL_HANDLE;
 VkApplicationInfo app={.sType=VK_STRUCTURE_TYPE_APPLICATION_INFO,.pApplicationName="flat-carrier",
                       .applicationVersion=1,.apiVersion=VK_API_VERSION_1_1};
 VkInstanceCreateInfo ici={.sType=VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,.pApplicationInfo=&app};
 CHECK(vkCreateInstance(&ici,NULL,&instance));
 uint32_t n=1;VkResult er=vkEnumeratePhysicalDevices(instance,&n,&physical);
 if((er!=VK_SUCCESS&&er!=VK_INCOMPLETE)||!n){fprintf(stderr,"no physical device: %d\n",er);goto cleanup;}
 VkPhysicalDeviceProperties props;vkGetPhysicalDeviceProperties(physical,&props);
 printf("device=%s vendor=%#x deviceID=%#x driver=%#x api=%u.%u.%u payloads=%u stride=%u inputword=4\n",
  props.deviceName,props.vendorID,props.deviceID,props.driverVersion,VK_VERSION_MAJOR(props.apiVersion),
  VK_VERSION_MINOR(props.apiVersion),VK_VERSION_PATCH(props.apiVersion),(unsigned)COUNT,(unsigned)STRIDE);
 VkFormatProperties fp;vkGetPhysicalDeviceFormatProperties(physical,VK_FORMAT_R32G32B32A32_UINT,&fp);
 if((fp.optimalTilingFeatures&(VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT|VK_FORMAT_FEATURE_TRANSFER_SRC_BIT))!=
    (VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT|VK_FORMAT_FEATURE_TRANSFER_SRC_BIT)){fprintf(stderr,"RGBA32_UINT color/copy unavailable\n");goto cleanup;}
 uint32_t nq=0;vkGetPhysicalDeviceQueueFamilyProperties(physical,&nq,NULL);
 if(!nq||nq>256)goto cleanup;VkQueueFamilyProperties *families=calloc(nq,sizeof *families);if(!families)goto cleanup;
 vkGetPhysicalDeviceQueueFamilyProperties(physical,&nq,families);
 uint32_t family=0;while(family<nq&&!(families[family].queueFlags&VK_QUEUE_GRAPHICS_BIT))family++;
 free(families);if(family==nq)goto cleanup;
 float priority=1;VkDeviceQueueCreateInfo qi={.sType=VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
  .queueFamilyIndex=family,.queueCount=1,.pQueuePriorities=&priority};
 VkDeviceCreateInfo di={.sType=VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,.queueCreateInfoCount=1,.pQueueCreateInfos=&qi};
 CHECK(vkCreateDevice(physical,&di,NULL,&device));vkGetDeviceQueue(device,family,0,&queue);
 const VkDeviceSize input_bytes=COUNT*STRIDE*4u,output_offset=(input_bytes+255u)&~255ull,output_bytes=COUNT*16u;
 VkBufferCreateInfo bi={.sType=VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,.size=output_offset+output_bytes,
  .usage=VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT,.sharingMode=VK_SHARING_MODE_EXCLUSIVE};
 CHECK(vkCreateBuffer(device,&bi,NULL,&buffer));VkMemoryRequirements req;vkGetBufferMemoryRequirements(device,buffer,&req);
 uint32_t mt=memory_type(physical,req.memoryTypeBits,VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
 if(mt==UINT32_MAX){fprintf(stderr,"no coherent host buffer memory\n");goto cleanup;}
 VkMemoryAllocateInfo ai={.sType=VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,.allocationSize=req.size,.memoryTypeIndex=mt};
 CHECK(vkAllocateMemory(device,&ai,NULL,&bmem));CHECK(vkBindBufferMemory(device,buffer,bmem,0));
 CHECK(vkMapMemory(device,bmem,0,VK_WHOLE_SIZE,0,&mapped));memset(mapped,0xa5,(size_t)bi.size);
 for(unsigned i=0;i<COUNT;i++)((uint32_t*)mapped)[i*STRIDE+4]=payloads[i];
 VkImageCreateInfo imageci={.sType=VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,.imageType=VK_IMAGE_TYPE_2D,
  .format=VK_FORMAT_R32G32B32A32_UINT,.extent={COUNT,1,1},.mipLevels=1,.arrayLayers=1,
  .samples=VK_SAMPLE_COUNT_1_BIT,.tiling=VK_IMAGE_TILING_OPTIMAL,
  .usage=VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT|VK_IMAGE_USAGE_TRANSFER_SRC_BIT,.sharingMode=VK_SHARING_MODE_EXCLUSIVE};
 CHECK(vkCreateImage(device,&imageci,NULL,&image));vkGetImageMemoryRequirements(device,image,&req);
 mt=memory_type(physical,req.memoryTypeBits,0);if(mt==UINT32_MAX)goto cleanup;
 ai.allocationSize=req.size;ai.memoryTypeIndex=mt;CHECK(vkAllocateMemory(device,&ai,NULL,&imem));
 CHECK(vkBindImageMemory(device,image,imem,0));
 VkImageViewCreateInfo vi={.sType=VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,.image=image,.viewType=VK_IMAGE_VIEW_TYPE_2D,
  .format=imageci.format,.subresourceRange={VK_IMAGE_ASPECT_COLOR_BIT,0,1,0,1}};
 CHECK(vkCreateImageView(device,&vi,NULL,&view));
 VkAttachmentDescription att={.format=imageci.format,.samples=VK_SAMPLE_COUNT_1_BIT,.loadOp=VK_ATTACHMENT_LOAD_OP_CLEAR,
  .storeOp=VK_ATTACHMENT_STORE_OP_STORE,.stencilLoadOp=VK_ATTACHMENT_LOAD_OP_DONT_CARE,
  .stencilStoreOp=VK_ATTACHMENT_STORE_OP_DONT_CARE,.initialLayout=VK_IMAGE_LAYOUT_UNDEFINED,
  .finalLayout=VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
 VkAttachmentReference ar={0,VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
 VkSubpassDescription sub={.pipelineBindPoint=VK_PIPELINE_BIND_POINT_GRAPHICS,.colorAttachmentCount=1,.pColorAttachments=&ar};
 VkRenderPassCreateInfo ri={.sType=VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,.attachmentCount=1,.pAttachments=&att,
  .subpassCount=1,.pSubpasses=&sub};CHECK(vkCreateRenderPass(device,&ri,NULL,&rp));
 VkFramebufferCreateInfo fi={.sType=VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,.renderPass=rp,.attachmentCount=1,
  .pAttachments=&view,.width=COUNT,.height=1,.layers=1};CHECK(vkCreateFramebuffer(device,&fi,NULL,&fb));
 VkDescriptorSetLayoutBinding binding={.binding=0,.descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
  .descriptorCount=1,.stageFlags=VK_SHADER_STAGE_VERTEX_BIT};
 VkDescriptorSetLayoutCreateInfo sli={.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,.bindingCount=1,.pBindings=&binding};
 CHECK(vkCreateDescriptorSetLayout(device,&sli,NULL,&dsl));
 VkDescriptorPoolSize ps={VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1};
 VkDescriptorPoolCreateInfo dpi={.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,.maxSets=1,.poolSizeCount=1,.pPoolSizes=&ps};
 CHECK(vkCreateDescriptorPool(device,&dpi,NULL,&pool));VkDescriptorSet set;
 VkDescriptorSetAllocateInfo sai={.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,.descriptorPool=pool,.descriptorSetCount=1,.pSetLayouts=&dsl};
 CHECK(vkAllocateDescriptorSets(device,&sai,&set));VkDescriptorBufferInfo dbi={buffer,0,input_bytes};
 VkWriteDescriptorSet write={.sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,.dstSet=set,.dstBinding=0,.descriptorCount=1,
  .descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,.pBufferInfo=&dbi};vkUpdateDescriptorSets(device,1,&write,0,NULL);
 VkPushConstantRange push={VK_SHADER_STAGE_VERTEX_BIT,0,4};
 VkPipelineLayoutCreateInfo pli={.sType=VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,.setLayoutCount=1,.pSetLayouts=&dsl,
  .pushConstantRangeCount=1,.pPushConstantRanges=&push};CHECK(vkCreatePipelineLayout(device,&pli,NULL,&layout));
 if(!make_shader(device,argv[1],&vs)||!make_shader(device,argv[2],&fs))goto cleanup;
 VkPipelineShaderStageCreateInfo stages[2]={
  {.sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,.stage=VK_SHADER_STAGE_VERTEX_BIT,.module=vs,.pName="main"},
  {.sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,.stage=VK_SHADER_STAGE_FRAGMENT_BIT,.module=fs,.pName="main"}};
 VkPipelineVertexInputStateCreateInfo vin={.sType=VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
 VkPipelineInputAssemblyStateCreateInfo ia={.sType=VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,.topology=VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST};
 VkViewport viewport={0,0,COUNT,1,0,1};VkRect2D scissor={{0,0},{COUNT,1}};
 VkPipelineViewportStateCreateInfo vpi={.sType=VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,.viewportCount=1,
  .pViewports=&viewport,.scissorCount=1,.pScissors=&scissor};
 VkPipelineRasterizationStateCreateInfo raster={.sType=VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
  .polygonMode=VK_POLYGON_MODE_FILL,.lineWidth=1};
 VkPipelineMultisampleStateCreateInfo multi={.sType=VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,.rasterizationSamples=VK_SAMPLE_COUNT_1_BIT};
 VkPipelineColorBlendAttachmentState ba={.colorWriteMask=15};
 VkPipelineColorBlendStateCreateInfo blend={.sType=VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,.attachmentCount=1,.pAttachments=&ba};
 VkGraphicsPipelineCreateInfo gp={.sType=VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,.stageCount=2,.pStages=stages,
  .pVertexInputState=&vin,.pInputAssemblyState=&ia,.pViewportState=&vpi,.pRasterizationState=&raster,
  .pMultisampleState=&multi,.pColorBlendState=&blend,.layout=layout,.renderPass=rp,.basePipelineIndex=-1};
 CHECK(vkCreateGraphicsPipelines(device,VK_NULL_HANDLE,1,&gp,NULL,&pipeline));
 VkCommandPoolCreateInfo cpi={.sType=VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,.queueFamilyIndex=family};
 CHECK(vkCreateCommandPool(device,&cpi,NULL,&cp));VkCommandBuffer cb;
 VkCommandBufferAllocateInfo cai={.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,.commandPool=cp,
  .level=VK_COMMAND_BUFFER_LEVEL_PRIMARY,.commandBufferCount=1};CHECK(vkAllocateCommandBuffers(device,&cai,&cb));
 VkCommandBufferBeginInfo cbi={.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,.flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
 CHECK(vkBeginCommandBuffer(cb,&cbi));VkClearValue clear={.color={.uint32={0xdeadbeef,0xdeadbeef,0xdeadbeef,0xdeadbeef}}};
 VkRenderPassBeginInfo rbi={.sType=VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,.renderPass=rp,.framebuffer=fb,
  .renderArea={{0,0},{COUNT,1}},.clearValueCount=1,.pClearValues=&clear};
 vkCmdBeginRenderPass(cb,&rbi,VK_SUBPASS_CONTENTS_INLINE);vkCmdBindPipeline(cb,VK_PIPELINE_BIND_POINT_GRAPHICS,pipeline);
 vkCmdBindDescriptorSets(cb,VK_PIPELINE_BIND_POINT_GRAPHICS,layout,0,1,&set,0,NULL);
 uint32_t count=COUNT;vkCmdPushConstants(cb,layout,VK_SHADER_STAGE_VERTEX_BIT,0,4,&count);vkCmdDraw(cb,6,COUNT,0,0);
 vkCmdEndRenderPass(cb);
 VkImageMemoryBarrier imb={.sType=VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,.srcAccessMask=VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
  .dstAccessMask=VK_ACCESS_TRANSFER_READ_BIT,.oldLayout=VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,.newLayout=VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
  .srcQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED,.dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED,.image=image,
  .subresourceRange={VK_IMAGE_ASPECT_COLOR_BIT,0,1,0,1}};
 vkCmdPipelineBarrier(cb,VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,VK_PIPELINE_STAGE_TRANSFER_BIT,0,0,NULL,0,NULL,1,&imb);
 VkBufferImageCopy copy={.bufferOffset=output_offset,.imageSubresource={VK_IMAGE_ASPECT_COLOR_BIT,0,0,1},.imageExtent={COUNT,1,1}};
 vkCmdCopyImageToBuffer(cb,image,VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,buffer,1,&copy);
 VkBufferMemoryBarrier bmb={.sType=VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,.srcAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT,
  .dstAccessMask=VK_ACCESS_HOST_READ_BIT,.srcQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED,.dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED,
  .buffer=buffer,.offset=output_offset,.size=output_bytes};
 vkCmdPipelineBarrier(cb,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_HOST_BIT,0,0,NULL,1,&bmb,0,NULL);
 CHECK(vkEndCommandBuffer(cb));VkFenceCreateInfo fci={.sType=VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
 CHECK(vkCreateFence(device,&fci,NULL,&fence));
 VkSubmitInfo si={.sType=VK_STRUCTURE_TYPE_SUBMIT_INFO,.commandBufferCount=1,.pCommandBuffers=&cb};
 CHECK(vkQueueSubmit(queue,1,&si,fence));submitted=1;CHECK(vkWaitForFences(device,1,&fence,VK_TRUE,10000000000ull));
 submitted=0;const uint32_t *got=(const uint32_t*)((const unsigned char*)mapped+output_offset);
 unsigned fbad=0,ubad=0,metadata=0;
 for(unsigned i=0;i<COUNT;i++){
  fbad+=got[4*i]!=payloads[i];ubad+=got[4*i+1]!=payloads[i];
  metadata+=got[4*i+2]!=i||got[4*i+3]!=0xc0decafeu;
  printf("case=%u input=%08x flat_float=%08x flat_uint=%08x tag=%u marker=%08x\n",
   i,payloads[i],got[4*i],got[4*i+1],got[4*i+2],got[4*i+3]);
 }
 printf("pipeline_executed=%u float_bit_mismatches=%u uint_bit_mismatches=%u metadata_errors=%u\n",
        metadata?0u:1u,fbad,ubad,metadata);
 result=(fbad||ubad||metadata)?1:0;
cleanup:
 if(device&&submitted)vkDeviceWaitIdle(device);
 if(device){
  if(fence)vkDestroyFence(device,fence,NULL);
  if(cp)vkDestroyCommandPool(device,cp,NULL);
  if(pipeline)vkDestroyPipeline(device,pipeline,NULL);
  if(vs)vkDestroyShaderModule(device,vs,NULL);if(fs)vkDestroyShaderModule(device,fs,NULL);
  if(layout)vkDestroyPipelineLayout(device,layout,NULL);if(pool)vkDestroyDescriptorPool(device,pool,NULL);
  if(dsl)vkDestroyDescriptorSetLayout(device,dsl,NULL);if(fb)vkDestroyFramebuffer(device,fb,NULL);
  if(rp)vkDestroyRenderPass(device,rp,NULL);if(view)vkDestroyImageView(device,view,NULL);
  if(image)vkDestroyImage(device,image,NULL);if(imem)vkFreeMemory(device,imem,NULL);
  if(mapped)vkUnmapMemory(device,bmem);if(buffer)vkDestroyBuffer(device,buffer,NULL);
  if(bmem)vkFreeMemory(device,bmem,NULL);vkDestroyDevice(device,NULL);
 }
 if(instance)vkDestroyInstance(instance,NULL);return result;
}
