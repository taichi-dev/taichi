# ./taichi/runtime/program_impls/llvm/CMakeLists.txt

add_library(llvm_program_impl)
target_sources(llvm_program_impl
  PRIVATE
    llvm_program.cpp
  )

target_include_directories(llvm_program_impl
  PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/external/eigen
    ${PROJECT_SOURCE_DIR}/external/spdlog/include
    ${LLVM_INCLUDE_DIRS}
  )

target_link_libraries(llvm_program_impl PRIVATE llvm_codegen)

target_link_libraries(llvm_program_impl PRIVATE cpu_runtime)
if (TI_WITH_CUDA)
    target_link_libraries(llvm_program_impl PRIVATE cuda_runtime)
endif()
