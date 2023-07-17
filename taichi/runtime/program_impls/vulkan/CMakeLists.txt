# ./taichi/runtime/program_impls/vulkan/CMakeLists.txt

add_library(vulkan_program_impl)
target_sources(vulkan_program_impl
  PRIVATE
    vulkan_program.cpp
    vulkan_program.h
  )

target_include_directories(vulkan_program_impl
  PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/external/eigen
    ${PROJECT_SOURCE_DIR}/external/spdlog/include
    ${PROJECT_SOURCE_DIR}/external/SPIRV-Tools/include
    ${PROJECT_SOURCE_DIR}/external/glfw/include
    ${LLVM_INCLUDE_DIRS}
  )

target_link_libraries(vulkan_program_impl PRIVATE spirv_codegen)
target_link_libraries(vulkan_program_impl PRIVATE gfx_runtime)
target_link_libraries(vulkan_program_impl PRIVATE gfx_program_impl)
target_link_libraries(vulkan_program_impl PRIVATE compilation_manager)
target_link_libraries(vulkan_program_impl PRIVATE ti_device_api)
target_link_libraries(vulkan_program_impl PRIVATE vulkan_rhi)
