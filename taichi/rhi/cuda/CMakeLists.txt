# ./taichi/rhi/cuda/CMakeLists.txt

set(CUDA_RHI cuda_rhi)
add_library(${CUDA_RHI})
target_sources(${CUDA_RHI}
  PRIVATE
    cuda_device.cpp
    cuda_context.cpp
    cuda_driver.cpp
    cuda_profiler.cpp
    cupti_toolkit.cpp
  )

target_include_directories(${CUDA_RHI}
  PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/external/eigen
    ${PROJECT_SOURCE_DIR}/external/spdlog/include
    ${PROJECT_SOURCE_DIR}/external/glfw/include
  )

target_link_libraries(${CUDA_RHI} PRIVATE interop_rhi)
