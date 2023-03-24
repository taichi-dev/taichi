# ./taichi/codegen/llvm/CMakeLists.txt

add_library(llvm_codegen)
target_sources(llvm_codegen
  PRIVATE
    codegen_llvm.cpp
    codegen_llvm_quant.cpp
    llvm_codegen_utils.cpp
    struct_llvm.cpp
    compiled_kernel_data.cpp
    kernel_compiler.cpp
  )

target_include_directories(llvm_codegen
  PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/external/eigen
    ${LLVM_INCLUDE_DIRS}
  )

target_link_libraries(llvm_codegen PRIVATE taichi_util)
target_link_libraries(llvm_codegen PRIVATE llvm_runtime)
