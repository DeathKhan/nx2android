// One-shot sampling of exact BC6H/BC7 blocks using a 2D-array linear/clamp sampler.
// Usage: vk_bc_sample input.bcs output.rgba32f bc_sample.spv
// Standalone Vulkan only; no game files, drivers, or persistent device settings are changed.
#include <vulkan/vulkan_core.h>
#include <math.h>
#include "bc_sample.h"

typedef struct { VkBuffer buffer; VkDeviceMemory memory; void *mapped; } host_buffer;
static VkDevice device;
static VkPhysicalDeviceMemoryProperties memory_properties;
static uint32_t memory_type(uint32_t bits, VkMemoryPropertyFlags flags) {
    for (uint32_t i = 0; i < memory_properties.memoryTypeCount; i++)
        if ((bits & (1u << i)) && (memory_properties.memoryTypes[i].propertyFlags & flags) == flags) return i;
    return UINT32_MAX;
}
static int make_host_buffer(VkDeviceSize bytes, VkBufferUsageFlags usage, host_buffer *out) {
    VkBufferCreateInfo ci = {.sType=VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size=bytes,
        .usage=usage, .sharingMode=VK_SHARING_MODE_EXCLUSIVE};
    VkResult r = vkCreateBuffer(device, &ci, NULL, &out->buffer);
    if (r) { fprintf(stderr, "create buffer failed: %d\n", r); return 0; }
    VkMemoryRequirements req; vkGetBufferMemoryRequirements(device, out->buffer, &req);
    uint32_t mt = memory_type(req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (mt == UINT32_MAX) { fprintf(stderr, "no coherent host memory type\n"); return 0; }
    VkMemoryAllocateInfo ai = {.sType=VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize=req.size, .memoryTypeIndex=mt};
    r = vkAllocateMemory(device, &ai, NULL, &out->memory);
    if (!r) r = vkBindBufferMemory(device, out->buffer, out->memory, 0);
    if (!r) r = vkMapMemory(device, out->memory, 0, VK_WHOLE_SIZE, 0, &out->mapped);
    if (r) { fprintf(stderr, "bind/map host buffer failed: %d\n", r); return 0; }
    return 1;
}
static void free_host_buffer(host_buffer *b) {
    if (b->mapped) vkUnmapMemory(device, b->memory);
    if (b->buffer) vkDestroyBuffer(device, b->buffer, NULL);
    if (b->memory) vkFreeMemory(device, b->memory, NULL);
}
static uint32_t *read_spirv(const char *path, size_t *size) {
    FILE *f = fopen(path, "rb"); if (!f) { perror(path); return NULL; }
    if (fseek(f, 0, SEEK_END)) { fclose(f); return NULL; }
    long n = ftell(f);
    if (n < 20 || n > (4 << 20) || (n & 3) || fseek(f, 0, SEEK_SET)) { fclose(f); return NULL; }
    uint32_t *words = malloc((size_t)n);
    if (!words) { fclose(f); return NULL; }
    int ok = fread(words, 1, (size_t)n, f) == (size_t)n && words[0] == 0x07230203u;
    fclose(f); if (!ok) { free(words); return NULL; } *size = (size_t)n; return words;
}
#define CHECK(call) do { VkResult rc_=(call); if(rc_) { fprintf(stderr,"%s failed: %d\n",#call,rc_); goto cleanup; } } while(0)
int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: vk_bc_sample input.bcs output.rgba32f shader.spv\n"); return 2; }
    int result = 2, submitted = 0, completed = 0;
    bc_sample_input input = {0}; uint32_t *code = NULL; size_t code_size = 0;
    VkInstance instance = VK_NULL_HANDLE; VkPhysicalDevice physical = VK_NULL_HANDLE;
    VkImage image = VK_NULL_HANDLE; VkDeviceMemory image_memory = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE; VkSampler sampler = VK_NULL_HANDLE;
    VkDescriptorSetLayout set_layout = VK_NULL_HANDLE; VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
    VkPipelineLayout pipeline_layout = VK_NULL_HANDLE; VkShaderModule module = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE; VkCommandPool command_pool = VK_NULL_HANDLE; VkFence fence = VK_NULL_HANDLE;
    host_buffer staging = {0}, points = {0}, output = {0};
    if (!bc_sample_read(argv[1], &input)) { fprintf(stderr, "invalid BCSAMP01 input\n"); goto cleanup; }
    for (uint32_t i = 0; i < input.h.point_count; i++) {
        const float *p = input.points + 4 * i;
        if (!isfinite(p[0]) || !isfinite(p[1]) || !isfinite(p[2]) ||
            p[2] < 0 || p[2] >= input.h.layers || floorf(p[2]) != p[2]) {
            fprintf(stderr, "invalid sample coordinate/layer at %u\n", i); goto cleanup;
        }
    }
    code = read_spirv(argv[3], &code_size);
    if (!code) { fprintf(stderr, "invalid SPIR-V file\n"); goto cleanup; }
    VkApplicationInfo app = {.sType=VK_STRUCTURE_TYPE_APPLICATION_INFO, .pApplicationName="bc-sample",
        .applicationVersion=1, .apiVersion=VK_API_VERSION_1_1};
    VkInstanceCreateInfo ici = {.sType=VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pApplicationInfo=&app};
    CHECK(vkCreateInstance(&ici, NULL, &instance));
    uint32_t count = 1; VkResult enumerate = vkEnumeratePhysicalDevices(instance, &count, &physical);
    if ((enumerate != VK_SUCCESS && enumerate != VK_INCOMPLETE) || !count) {
        fprintf(stderr, "no Vulkan physical device: %d\n", enumerate); goto cleanup;
    }
    VkPhysicalDeviceProperties props; vkGetPhysicalDeviceProperties(physical, &props);
    VkPhysicalDeviceFeatures available; vkGetPhysicalDeviceFeatures(physical, &available);
    VkFormat format = (VkFormat)input.h.vk_format;
    VkFormatProperties fp; vkGetPhysicalDeviceFormatProperties(physical, format, &fp);
    VkFormatFeatureFlags needed = VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT |
        VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT | VK_FORMAT_FEATURE_TRANSFER_DST_BIT | VK_FORMAT_FEATURE_TRANSFER_SRC_BIT;
    printf("device=%s api=%u.%u.%u driver=%u format=%u optimal_features=%#x BC=%u input=%ux%ux%u points=%u\n",
        props.deviceName, VK_VERSION_MAJOR(props.apiVersion), VK_VERSION_MINOR(props.apiVersion),
        VK_VERSION_PATCH(props.apiVersion), props.driverVersion, input.h.vk_format, fp.optimalTilingFeatures,
        available.textureCompressionBC, input.h.width, input.h.height, input.h.layers, input.h.point_count);
    if (!available.textureCompressionBC || (fp.optimalTilingFeatures & needed) != needed) {
        fprintf(stderr, "native compressed linear sampling unsupported\n"); goto cleanup;
    }
    VkImageFormatProperties ifp;
    CHECK(vkGetPhysicalDeviceImageFormatProperties(physical, format, VK_IMAGE_TYPE_2D, VK_IMAGE_TILING_OPTIMAL,
        VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, 0, &ifp));
    if (input.h.width > ifp.maxExtent.width || input.h.height > ifp.maxExtent.height ||
        input.h.layers > ifp.maxArrayLayers || input.h.point_bytes > props.limits.maxStorageBufferRange ||
        props.limits.maxComputeWorkGroupInvocations < 64 || props.limits.maxComputeWorkGroupSize[0] < 64 ||
        (input.h.point_count + 63) / 64 > props.limits.maxComputeWorkGroupCount[0]) {
        fprintf(stderr, "input exceeds physical device limits\n"); goto cleanup;
    }
    uint32_t nq = 32; VkQueueFamilyProperties families[32];
    vkGetPhysicalDeviceQueueFamilyProperties(physical, &nq, families);
    uint32_t qf = 0; while (qf < nq && !(families[qf].queueFlags & VK_QUEUE_COMPUTE_BIT)) qf++;
    if (qf == nq) { fprintf(stderr, "no compute queue\n"); goto cleanup; }
    float priority = 1;
    VkDeviceQueueCreateInfo qi = {.sType=VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex=qf, .queueCount=1, .pQueuePriorities=&priority};
    VkPhysicalDeviceFeatures enabled = {.textureCompressionBC=VK_TRUE};
    VkDeviceCreateInfo di = {.sType=VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .queueCreateInfoCount=1, .pQueueCreateInfos=&qi, .pEnabledFeatures=&enabled};
    CHECK(vkCreateDevice(physical, &di, NULL, &device));
    VkQueue queue; vkGetDeviceQueue(device, qf, 0, &queue);
    vkGetPhysicalDeviceMemoryProperties(physical, &memory_properties);
    if (!make_host_buffer(input.h.block_bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, &staging) ||
        !make_host_buffer(input.h.point_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &points) ||
        !make_host_buffer(input.h.point_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &output)) goto cleanup;
    memcpy(staging.mapped, input.blocks, (size_t)input.h.block_bytes);
    memcpy(points.mapped, input.points, (size_t)input.h.point_bytes);
    memset(output.mapped, 0xff, (size_t)input.h.point_bytes); // poison: any unwritten sample is a NaN
    VkImageCreateInfo image_ci = {.sType=VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType=VK_IMAGE_TYPE_2D,
        .format=format, .extent={input.h.width, input.h.height, 1}, .mipLevels=1, .arrayLayers=input.h.layers,
        .samples=VK_SAMPLE_COUNT_1_BIT, .tiling=VK_IMAGE_TILING_OPTIMAL,
        .usage=VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, .sharingMode=VK_SHARING_MODE_EXCLUSIVE,
        .initialLayout=VK_IMAGE_LAYOUT_UNDEFINED};
    CHECK(vkCreateImage(device, &image_ci, NULL, &image));
    VkMemoryRequirements req; vkGetImageMemoryRequirements(device, image, &req);
    uint32_t mt = memory_type(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (mt == UINT32_MAX) mt = memory_type(req.memoryTypeBits, 0);
    if (mt == UINT32_MAX) goto cleanup;
    VkMemoryAllocateInfo ma = {.sType=VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize=req.size, .memoryTypeIndex=mt};
    CHECK(vkAllocateMemory(device, &ma, NULL, &image_memory));
    CHECK(vkBindImageMemory(device, image, image_memory, 0));
    VkImageViewCreateInfo vi = {.sType=VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image=image,
        .viewType=VK_IMAGE_VIEW_TYPE_2D_ARRAY, .format=format,
        .components={VK_COMPONENT_SWIZZLE_IDENTITY,VK_COMPONENT_SWIZZLE_IDENTITY,VK_COMPONENT_SWIZZLE_IDENTITY,VK_COMPONENT_SWIZZLE_IDENTITY},
        .subresourceRange={VK_IMAGE_ASPECT_COLOR_BIT,0,1,0,input.h.layers}};
    CHECK(vkCreateImageView(device, &vi, NULL, &view));
    VkSamplerCreateInfo sampler_ci = {.sType=VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO, .magFilter=VK_FILTER_LINEAR,
        .minFilter=VK_FILTER_LINEAR, .mipmapMode=VK_SAMPLER_MIPMAP_MODE_NEAREST,
        .addressModeU=VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, .addressModeV=VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        .addressModeW=VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, .minLod=0, .maxLod=0,
        .borderColor=VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK};
    CHECK(vkCreateSampler(device, &sampler_ci, NULL, &sampler));
    VkDescriptorSetLayoutBinding bindings[] = {
        {0,VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,1,VK_SHADER_STAGE_COMPUTE_BIT,NULL},
        {1,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,VK_SHADER_STAGE_COMPUTE_BIT,NULL},
        {2,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,VK_SHADER_STAGE_COMPUTE_BIT,NULL}};
    VkDescriptorSetLayoutCreateInfo li = {.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, .bindingCount=3, .pBindings=bindings};
    CHECK(vkCreateDescriptorSetLayout(device, &li, NULL, &set_layout));
    VkDescriptorPoolSize sizes[] = {{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,1},{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,2}};
    VkDescriptorPoolCreateInfo pool_ci = {.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, .maxSets=1, .poolSizeCount=2, .pPoolSizes=sizes};
    CHECK(vkCreateDescriptorPool(device, &pool_ci, NULL, &descriptor_pool));
    VkDescriptorSetAllocateInfo sa = {.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, .descriptorPool=descriptor_pool,
        .descriptorSetCount=1, .pSetLayouts=&set_layout};
    VkDescriptorSet set; CHECK(vkAllocateDescriptorSets(device, &sa, &set));
    VkDescriptorImageInfo ii = {sampler,view,VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    VkDescriptorBufferInfo buffer_info[] = {{points.buffer,0,input.h.point_bytes},{output.buffer,0,input.h.point_bytes}};
    VkWriteDescriptorSet writes[] = {
        {.sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,.dstSet=set,.dstBinding=0,.descriptorCount=1,.descriptorType=VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,.pImageInfo=&ii},
        {.sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,.dstSet=set,.dstBinding=1,.descriptorCount=1,.descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,.pBufferInfo=&buffer_info[0]},
        {.sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,.dstSet=set,.dstBinding=2,.descriptorCount=1,.descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,.pBufferInfo=&buffer_info[1]}};
    vkUpdateDescriptorSets(device, 3, writes, 0, NULL);
    VkPushConstantRange push = {VK_SHADER_STAGE_COMPUTE_BIT,0,4};
    VkPipelineLayoutCreateInfo pli = {.sType=VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount=1,.pSetLayouts=&set_layout,.pushConstantRangeCount=1,.pPushConstantRanges=&push};
    CHECK(vkCreatePipelineLayout(device, &pli, NULL, &pipeline_layout));
    VkShaderModuleCreateInfo smi = {.sType=VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,.codeSize=code_size,.pCode=code};
    CHECK(vkCreateShaderModule(device, &smi, NULL, &module));
    VkComputePipelineCreateInfo cpi = {.sType=VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage={.sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,.stage=VK_SHADER_STAGE_COMPUTE_BIT,.module=module,.pName="main"},
        .layout=pipeline_layout,.basePipelineIndex=-1};
    CHECK(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cpi, NULL, &pipeline));
    VkCommandPoolCreateInfo pci = {.sType=VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,.queueFamilyIndex=qf};
    CHECK(vkCreateCommandPool(device, &pci, NULL, &command_pool));
    VkCommandBufferAllocateInfo cai = {.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool=command_pool,.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY,.commandBufferCount=1};
    VkCommandBuffer cb; CHECK(vkAllocateCommandBuffers(device, &cai, &cb));
    VkCommandBufferBeginInfo begin = {.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,.flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
    CHECK(vkBeginCommandBuffer(cb, &begin));
    VkMemoryBarrier host = {.sType=VK_STRUCTURE_TYPE_MEMORY_BARRIER,.srcAccessMask=VK_ACCESS_HOST_WRITE_BIT,
        .dstAccessMask=VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT};
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0, 1, &host, 0, NULL, 0, NULL);
    VkImageMemoryBarrier upload = {.sType=VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,.dstAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT,
        .oldLayout=VK_IMAGE_LAYOUT_UNDEFINED,.newLayout=VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        .srcQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED,.dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED,
        .image=image,.subresourceRange={VK_IMAGE_ASPECT_COLOR_BIT,0,1,0,input.h.layers}};
    vkCmdPipelineBarrier(cb,VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,VK_PIPELINE_STAGE_TRANSFER_BIT,0,0,NULL,0,NULL,1,&upload);
    VkBufferImageCopy region = {.imageSubresource={VK_IMAGE_ASPECT_COLOR_BIT,0,0,input.h.layers},
        .imageExtent={input.h.width,input.h.height,1}};
    vkCmdCopyBufferToImage(cb,staging.buffer,image,VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,1,&region);
    upload.srcAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT;upload.dstAccessMask=VK_ACCESS_SHADER_READ_BIT;
    upload.oldLayout=VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;upload.newLayout=VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    vkCmdPipelineBarrier(cb,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,0,0,NULL,0,NULL,1,&upload);
    vkCmdBindPipeline(cb,VK_PIPELINE_BIND_POINT_COMPUTE,pipeline);
    vkCmdBindDescriptorSets(cb,VK_PIPELINE_BIND_POINT_COMPUTE,pipeline_layout,0,1,&set,0,NULL);
    vkCmdPushConstants(cb,pipeline_layout,VK_SHADER_STAGE_COMPUTE_BIT,0,4,&input.h.point_count);
    vkCmdDispatch(cb,(input.h.point_count+63)/64,1,1);
    VkBufferMemoryBarrier readback = {.sType=VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
        .srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT,.dstAccessMask=VK_ACCESS_HOST_READ_BIT,
        .srcQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED,.dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED,
        .buffer=output.buffer,.offset=0,.size=input.h.point_bytes};
    vkCmdPipelineBarrier(cb,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_HOST_BIT,0,0,NULL,1,&readback,0,NULL);
    // Independently verify that every uploaded compressed block survives the image roundtrip.
    // The staging buffer is reused only after its initial transfer reads have completed.
    upload.srcAccessMask=VK_ACCESS_SHADER_READ_BIT;upload.dstAccessMask=VK_ACCESS_TRANSFER_READ_BIT;
    upload.oldLayout=VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;upload.newLayout=VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    VkBufferMemoryBarrier stage_reuse={.sType=VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
        .srcAccessMask=VK_ACCESS_TRANSFER_READ_BIT,.dstAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT,
        .srcQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED,.dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED,
        .buffer=staging.buffer,.offset=0,.size=input.h.block_bytes};
    vkCmdPipelineBarrier(cb,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT|VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT,0,0,NULL,1,&stage_reuse,1,&upload);
    vkCmdCopyImageToBuffer(cb,image,VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,staging.buffer,1,&region);
    stage_reuse.srcAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT;stage_reuse.dstAccessMask=VK_ACCESS_HOST_READ_BIT;
    vkCmdPipelineBarrier(cb,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_HOST_BIT,0,0,NULL,1,&stage_reuse,0,NULL);
    CHECK(vkEndCommandBuffer(cb));
    VkFenceCreateInfo fi={.sType=VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};CHECK(vkCreateFence(device,&fi,NULL,&fence));
    VkSubmitInfo submit={.sType=VK_STRUCTURE_TYPE_SUBMIT_INFO,.commandBufferCount=1,.pCommandBuffers=&cb};
    CHECK(vkQueueSubmit(queue,1,&submit,fence));submitted=1;
    CHECK(vkWaitForFences(device,1,&fence,VK_TRUE,10000000000ull));completed=1;
    // Coherent mapped memory plus HOST_READ visibility and a signaled fence: no invalidate needed.
    if(memcmp(input.blocks,staging.mapped,(size_t)input.h.block_bytes)) {
        const uint8_t *actual=staging.mapped;size_t first=0,wrong=0;
        for(size_t i=0;i<(size_t)input.h.block_bytes;i++)if(actual[i]!=input.blocks[i]){if(!wrong)first=i;wrong++;}
        fprintf(stderr,"compressed upload roundtrip FAILED: %zu differing bytes, first offset %zu expected=%02x actual=%02x\n",
            wrong,first,input.blocks[first],actual[first]);result=1;goto cleanup;
    }
    printf("compressed upload roundtrip PASS: %llu exact bytes\n",(unsigned long long)input.h.block_bytes);
    if (!bc_sample_write(argv[2],output.mapped,(size_t)input.h.point_bytes)) goto cleanup;
    unsigned nonfinite=0;const float *samples=output.mapped;
    for(uint32_t i=0;i<input.h.point_count*4;i++)nonfinite+=!isfinite(samples[i]);
    printf("wrote %llu bytes (%u float4 samples), nonfinite components=%u -> %s\n",
        (unsigned long long)input.h.point_bytes,input.h.point_count,nonfinite,argv[2]);
    result=0;
cleanup:
    // If a driver did not complete within the bounded wait, process exit owns teardown.
    // Never destroy resources while their queue may still use them.
    if(submitted&&!completed) { fprintf(stderr,"submission incomplete; leaving Vulkan teardown to process exit\n"); return result; }
    if(device) {
        if(fence)vkDestroyFence(device,fence,NULL);
        if(command_pool)vkDestroyCommandPool(device,command_pool,NULL);
        if(pipeline)vkDestroyPipeline(device,pipeline,NULL);
        if(module)vkDestroyShaderModule(device,module,NULL);
        if(pipeline_layout)vkDestroyPipelineLayout(device,pipeline_layout,NULL);
        if(descriptor_pool)vkDestroyDescriptorPool(device,descriptor_pool,NULL);
        if(set_layout)vkDestroyDescriptorSetLayout(device,set_layout,NULL);
        if(sampler)vkDestroySampler(device,sampler,NULL);
        if(view)vkDestroyImageView(device,view,NULL);
        if(image)vkDestroyImage(device,image,NULL);
        if(image_memory)vkFreeMemory(device,image_memory,NULL);
        free_host_buffer(&output);free_host_buffer(&points);free_host_buffer(&staging);
        vkDestroyDevice(device,NULL);
    }
    if(instance)vkDestroyInstance(instance,NULL);
    free(code);free(input.blocks);free(input.points);return result;
}
