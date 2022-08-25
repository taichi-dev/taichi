# ./taichi/rhi/dx12/CMakeLists.txt

set(DX12_RHI dx12_rhi)
add_library(${DX12_RHI})
target_sources(${DX12_RHI}
  PRIVATE
    dx12_api.cpp
  )

target_include_directories(${DX12_RHI}
  PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/external/eigen
    ${PROJECT_SOURCE_DIR}/external/spdlog/include
    ${LLVM_INCLUDE_DIRS}
  )

target_link_libraries(${DX12_RHI} PRIVATE )
