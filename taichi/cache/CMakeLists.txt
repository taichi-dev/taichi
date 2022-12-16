# ./taichi/cache/CMakeLists.txt

add_library(kernel_compilation_manager)
target_sources(kernel_compilation_manager
  PRIVATE
    kernel_compilation_manager.cpp
  )

target_include_directories(kernel_compilation_manager
  PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/external/spdlog/include
    ${PROJECT_SOURCE_DIR}/external/eigen
    ${LLVM_INCLUDE_DIRS} # For "llvm/ADT/SmallVector.h" included in ir.h
  )
