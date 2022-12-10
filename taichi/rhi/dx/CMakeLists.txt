# ./taichi/rhi/dx/CMakeLists.txt

set(DX_RHI dx_rhi)
add_library(${DX_RHI})
target_sources(${DX_RHI}
  PRIVATE
    dx_api.cpp
    dx_device.cpp
    dx_info_queue.cpp
  )

target_include_directories(${DX_RHI}
  PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/external/spdlog/include
    ${PROJECT_SOURCE_DIR}/external/SPIRV-Cross
  )

target_link_libraries(${DX_RHI} PRIVATE spirv-cross-hlsl spirv-cross-core)
