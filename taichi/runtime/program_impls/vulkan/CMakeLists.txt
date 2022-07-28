# ./taichi/runtime/program_impls/vulkan/CMakeLists.txt

add_library(vulkan_program_impl)
target_sources(vulkan_program_impl
  PRIVATE
    vulkan_program.cpp
  )

target_include_directories(vulkan_program_impl
  PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/external/eigen
    ${PROJECT_SOURCE_DIR}/external/spdlog/include
    ${PROJECT_SOURCE_DIR}/external/SPIRV-Tools/include
    ${LLVM_INCLUDE_DIRS}
  )

target_link_libraries(vulkan_program_impl PRIVATE gfx_runtime)
target_link_libraries(vulkan_program_impl PRIVATE vulkan_rhi)
