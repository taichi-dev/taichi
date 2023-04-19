# ./taichi/rhi/common/CMakeLists.txt

set(COMMON_RHI common_rhi)
add_library(${COMMON_RHI})
target_sources(${COMMON_RHI}
  PRIVATE
    host_memory_pool.cpp
    unified_allocator.cpp
    window_system.cpp
  )

target_include_directories(${COMMON_RHI}
  PRIVATE
    ${PROJECT_SOURCE_DIR}
  )

if (TI_WITH_OPENGL OR TI_WITH_VULKAN AND NOT ANDROID)
  target_compile_definitions(${COMMON_RHI} PUBLIC TI_WITH_GLFW)

  target_include_directories(${COMMON_RHI}
    PRIVATE
      ${PROJECT_SOURCE_DIR}/external/glfw/include
    )
endif()

if(TI_WITH_VULKAN)
    target_link_libraries(${COMMON_RHI} PUBLIC vulkan_rhi)

    target_include_directories(${COMMON_RHI} SYSTEM
      PUBLIC
        ${PROJECT_SOURCE_DIR}/external/volk
        ${PROJECT_SOURCE_DIR}/external/Vulkan-Headers/include
        ${PROJECT_SOURCE_DIR}/external/VulkanMemoryAllocator/include
      PRIVATE
        ${PROJECT_SOURCE_DIR}/external/SPIRV-Reflect
        ${PROJECT_SOURCE_DIR}/external/spdlog/include
      )
endif()

target_link_libraries(${COMMON_RHI} PRIVATE taichi_util)
