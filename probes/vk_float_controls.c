// Read-only physical-device inventory. No VkDevice, queues, shaders, or GPU work.
// Usage: vk_float_controls > float-controls.txt
#include <vulkan/vulkan_core.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char *independence(VkShaderFloatControlsIndependence x) {
    switch (x) {
    case VK_SHADER_FLOAT_CONTROLS_INDEPENDENCE_32_BIT_ONLY: return "32_BIT_ONLY";
    case VK_SHADER_FLOAT_CONTROLS_INDEPENDENCE_ALL: return "ALL";
    case VK_SHADER_FLOAT_CONTROLS_INDEPENDENCE_NONE: return "NONE";
    default: return "UNKNOWN";
    }
}
static int properties(VkPhysicalDevice physical, unsigned index) {
    VkPhysicalDeviceProperties base;
    vkGetPhysicalDeviceProperties(physical, &base);
    uint32_t n = 0;
    VkResult r = vkEnumerateDeviceExtensionProperties(physical, NULL, &n, NULL);
    if (r != VK_SUCCESS || n > 4096) return 0;
    VkExtensionProperties *ext = n ? calloc(n, sizeof *ext) : NULL;
    if (n && !ext) return 0;
    if (n) r = vkEnumerateDeviceExtensionProperties(physical, NULL, &n, ext);
    if (r != VK_SUCCESS) { free(ext); return 0; }
    int fc_ext = 0, dp_ext = 0;
    for (uint32_t i = 0; i < n; i++) {
        if (!strcmp(ext[i].extensionName, VK_KHR_SHADER_FLOAT_CONTROLS_EXTENSION_NAME)) fc_ext = 1;
        if (!strcmp(ext[i].extensionName, VK_KHR_DRIVER_PROPERTIES_EXTENSION_NAME)) dp_ext = 1;
    }
    free(ext);
    int core12 = base.apiVersion >= VK_API_VERSION_1_2;
    int have_fc = core12 || fc_ext, have_driver = core12 || dp_ext;
    printf("device[%u].name=%s\napi=%u.%u.%u vendor=%#x device=%#x driverVersion=%#x type=%u\n",
           index, base.deviceName, VK_VERSION_MAJOR(base.apiVersion), VK_VERSION_MINOR(base.apiVersion),
           VK_VERSION_PATCH(base.apiVersion), base.vendorID, base.deviceID, base.driverVersion, base.deviceType);
    printf("KHR_shader_float_controls=%d KHR_driver_properties=%d core12=%d\n", fc_ext, dp_ext, core12);
    if (base.apiVersion < VK_API_VERSION_1_1) {
        puts("query=unsupported_physical_device_api_below_1_1"); return 1;
    }
    VkPhysicalDeviceIDProperties id = {.sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES};
    VkPhysicalDeviceDriverProperties driver = {.sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES};
    VkPhysicalDeviceFloatControlsProperties fc = {.sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FLOAT_CONTROLS_PROPERTIES};
    VkPhysicalDeviceProperties2 p = {.sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, .pNext=&id};
    void **tail = &id.pNext;
    if (have_driver) { *tail = &driver; tail = &driver.pNext; }
    if (have_fc) { *tail = &fc; tail = &fc.pNext; }
    *tail = NULL;
    vkGetPhysicalDeviceProperties2(physical, &p);
    printf("deviceUUID="); for (unsigned i=0;i<VK_UUID_SIZE;i++) printf("%02x",id.deviceUUID[i]);
    printf("\ndriverUUID="); for (unsigned i=0;i<VK_UUID_SIZE;i++) printf("%02x",id.driverUUID[i]); puts("");
    if (have_driver) printf("driverID=%u driverName=%s\ndriverInfo=%s\nconformance=%u.%u.%u.%u\n",
        driver.driverID, driver.driverName, driver.driverInfo, driver.conformanceVersion.major,
        driver.conformanceVersion.minor, driver.conformanceVersion.subminor, driver.conformanceVersion.patch);
    if (!have_fc) { puts("floatControls=unsupported_not_queried"); return 1; }
    printf("denormBehaviorIndependence=%u(%s)\nroundingModeIndependence=%u(%s)\n",
           fc.denormBehaviorIndependence, independence(fc.denormBehaviorIndependence),
           fc.roundingModeIndependence, independence(fc.roundingModeIndependence));
#define FIELD(field) printf(#field "=%u\n", fc.field)
#define WIDTH(bits) do { FIELD(shaderSignedZeroInfNanPreserveFloat##bits); \
    FIELD(shaderDenormPreserveFloat##bits); FIELD(shaderDenormFlushToZeroFloat##bits); \
    FIELD(shaderRoundingModeRTEFloat##bits); FIELD(shaderRoundingModeRTZFloat##bits); } while (0)
    WIDTH(16); WIDTH(32); WIDTH(64);
#undef WIDTH
#undef FIELD
    puts("query=complete"); return 1;
}
int main(void) {
    VkApplicationInfo app = {.sType=VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName="float-controls-inventory", .applicationVersion=1, .apiVersion=VK_API_VERSION_1_1};
    VkInstanceCreateInfo ci = {.sType=VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pApplicationInfo=&app};
    VkInstance instance;
    VkResult r = vkCreateInstance(&ci, NULL, &instance);
    if (r != VK_SUCCESS) { fprintf(stderr,"vkCreateInstance failed: %d\n",r); return 1; }
    uint32_t n = 0; VkPhysicalDevice *physical = NULL; int result = 1;
    r = vkEnumeratePhysicalDevices(instance, &n, NULL);
    if (r != VK_SUCCESS || !n || n > 256) goto done;
    physical = calloc(n, sizeof *physical);
    if (!physical) goto done;
    r = vkEnumeratePhysicalDevices(instance, &n, physical);
    if (r != VK_SUCCESS) goto done;
    result = 0;
    for (uint32_t i=0;i<n;i++) if (!properties(physical[i], i)) { result=1; break; }
done:
    if (result) fprintf(stderr,"physical-device inventory incomplete (last enumeration result %d)\n",r);
    free(physical); vkDestroyInstance(instance, NULL); return result;
}
