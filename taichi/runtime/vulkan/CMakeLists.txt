# ./taichi/runtime/vulkan/CMakeLists.txt

add_library(vulkan_runtime)
target_sources(vulkan_runtime
  PRIVATE
    runtime.cpp
  )
target_include_directories(vulkan_runtime
  PRIVATE
    ${PROJECT_SOURCE_DIR}/external/SPIRV-Tools/include
    ${PROJECT_SOURCE_DIR}/external/eigen
    ${PROJECT_SOURCE_DIR}/external/FP16/include
  )
