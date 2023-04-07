# ./taichi/rhi/cpu/CMakeLists.txt

set(CPU_RHI cpu_rhi)
add_library(${CPU_RHI})
target_sources(${CPU_RHI}
  PRIVATE
    cpu_device.cpp
  )

target_include_directories(${CPU_RHI}
  PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/external/eigen
    ${PROJECT_SOURCE_DIR}/external/spdlog/include
    ${PROJECT_SOURCE_DIR}/external/glfw/include
  )
