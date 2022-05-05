# ./taichi/runtime/vulkan/CMakeLists.txt

add_library(vulkan_runtime)
target_sources(vulkan_runtime
  PRIVATE
    runtime.cpp
  )

#TODO 4832, some dependencies here should not be required as they
# are build requirements of other targets.
target_include_directories(vulkan_runtime
  PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/external/SPIRV-Tools/include
    ${PROJECT_SOURCE_DIR}/external/eigen
    ${PROJECT_SOURCE_DIR}/external/FP16/include
    ${PROJECT_SOURCE_DIR}/external/spdlog/include
    ${LLVM_INCLUDE_DIRS}
  )
