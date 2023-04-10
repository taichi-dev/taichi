# ./taichi/rhi/llvm/CMakeLists.txt

set(LLVM_RHI llvm_rhi)
add_library(${LLVM_RHI})
target_sources(${LLVM_RHI}
  PRIVATE
    llvm_device.cpp
    allocator.cpp
    unified_allocator.cpp
    memory_pool.cpp
  )

target_include_directories(${LLVM_RHI}
  PRIVATE
    ${PROJECT_SOURCE_DIR}
  )

target_link_libraries(${LLVM_RHI} PRIVATE taichi_util)
