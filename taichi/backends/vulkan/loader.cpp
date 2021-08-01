#pragma once

#define VOLK_IMPLEMENTATION
#include <volk.h>

#include "taichi/backends/vulkan/loader.h"


namespace taichi {
namespace lang {
namespace vulkan {


VulkanLoader::VulkanLoader(){
    initialized = false;
}

bool VulkanLoader::init(){
    if(initialized){
        return true;
    }
    VkResult result = volkInitialize();
    initialized = result == VK_SUCCESS;
    return initialized;
}

void VulkanLoader::load_instance(VkInstance instance){
    vulkan_instance = instance;
    volkLoadInstance(instance);
}
void VulkanLoader::load_device(VkDevice device){
    vulkan_device = device;
    volkLoadDevice(device);
}

}  // namespace vulkan
}  // namespace lang
}  // namespace taichi

