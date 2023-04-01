# ./taichi/rhi/vulkan/CMakeLists.txt

set(VULKAN_RHI vulkan_rhi)
add_library(${VULKAN_RHI})
target_sources(${VULKAN_RHI}
  PRIVATE
    vulkan_api.cpp
    vulkan_device.cpp
    vulkan_device_creator.cpp
    vulkan_loader.cpp
    vulkan_memory_allocator.cpp
    ${PROJECT_SOURCE_DIR}/external/SPIRV-Reflect/spirv_reflect.c

    vulkan_api.h
    vulkan_device.h
    vulkan_device_creator.h
    vulkan_loader.h
  )

#TODO 4832, some dependencies here should not be required as they
# are build requirements of other targets.
# public dirs here are required by backends/device.cpp
target_include_directories(${VULKAN_RHI}
  PRIVATE
    ${PROJECT_SOURCE_DIR}
  )

# By specifying SYSTEM, we suppressed the warnings from third-party headers.
# This is used to bypass unused variables in the header vk_mem_alloc.h
target_include_directories(${VULKAN_RHI} SYSTEM
  PUBLIC
    ${PROJECT_SOURCE_DIR}/external/volk
    ${PROJECT_SOURCE_DIR}/external/Vulkan-Headers/include
    ${PROJECT_SOURCE_DIR}/external/VulkanMemoryAllocator/include
  PRIVATE
    ${PROJECT_SOURCE_DIR}/external/SPIRV-Reflect
    ${PROJECT_SOURCE_DIR}/external/spdlog/include
  )
