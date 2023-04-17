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

if(TI_WITH_VULKAN)

  target_include_directories(${COMMON_RHI} SYSTEM
  PUBLIC
    ${PROJECT_SOURCE_DIR}/external/volk
  )

endif()

target_link_libraries(${COMMON_RHI} PRIVATE taichi_util)
