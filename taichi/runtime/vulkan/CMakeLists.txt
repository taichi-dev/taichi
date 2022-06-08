# ./taichi/runtime/gfx/CMakeLists.txt

add_library(vulkan_runtime)
target_sources(vulkan_runtime
  PRIVATE
    vulkan_api.cpp
    vulkan_device.cpp
    vulkan_device_creator.cpp
    vulkan_loader.cpp
    vulkan_memory_allocator.cpp
    vulkan_program.cpp
  )

#TODO 4832, some dependencies here should not be required as they
# are build requirements of other targets.
# public dirs here are required by backends/device.cpp
target_include_directories(vulkan_runtime
  PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/external/SPIRV-Tools/include
    ${PROJECT_SOURCE_DIR}/external/eigen
    ${PROJECT_SOURCE_DIR}/external/FP16/include
    ${PROJECT_SOURCE_DIR}/external/SPIRV-Reflect
  PRIVATE
    ${PROJECT_SOURCE_DIR}/external/spdlog/include
        ${LLVM_INCLUDE_DIRS}
  PUBLIC
    ${PROJECT_SOURCE_DIR}/external/volk
    ${PROJECT_SOURCE_DIR}/external/Vulkan-Headers/include
  )

# By specifying SYSTEM, we suppressed the warnings from third-party headers.
# This is used to bypass unused variables in the header vk_mem_alloc.h
target_include_directories(vulkan_runtime SYSTEM
  PUBLIC
    ${PROJECT_SOURCE_DIR}/external/VulkanMemoryAllocator/include
  )

target_link_libraries(${CORE_LIBRARY_NAME} PRIVATE gfx_runtime)
