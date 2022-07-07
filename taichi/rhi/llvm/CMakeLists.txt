# ./taichi/rhi/llvm/CMakeLists.txt

set(LLVM_RHI llvm_rhi)
add_library(${LLVM_RHI})
target_sources(${LLVM_RHI}
  PRIVATE
    llvm_device.cpp
  )

#TODO 4832, some dependencies here should not be required as they
# are build requirements of other targets.
target_include_directories(${LLVM_RHI}
  PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/external/spdlog/include
    ${PROJECT_SOURCE_DIR}/external/glfw/include
    ${LLVM_INCLUDE_DIRS}
  )
