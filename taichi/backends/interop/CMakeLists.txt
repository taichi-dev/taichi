# ./taichi/backends/interop/CMakeLists.txt

set(INTEROP_RHI interop_rhi)
add_library(${INTEROP_RHI})
target_sources(${INTEROP_RHI}
  PRIVATE
    vulkan_cpu_interop.cpp
    vulkan_cuda_interop.cpp
  )

target_include_directories(${INTEROP_RHI}
  PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/external/eigen
    ${PROJECT_SOURCE_DIR}/external/spdlog/include
    ${PROJECT_SOURCE_DIR}/external/glfw/include
    ${LLVM_INCLUDE_DIRS}
  )
