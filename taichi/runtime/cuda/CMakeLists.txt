# ./taichi/runtime/cuda/CMakeLists.txt

add_library(cuda_runtime)
target_sources(cuda_runtime
  PRIVATE
    aot_module_builder_impl.cpp
    aot_module_loader_impl.cpp
    jit_cuda.cpp
    runtime.cpp
  )

#TODO #4832, some path here should not be included as they are
# dependencies of other targets.
target_include_directories(cuda_runtime
  PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/external/eigen
    ${PROJECT_SOURCE_DIR}/external/spdlog/include
    ${LLVM_INCLUDE_DIRS}
  )

target_link_libraries(cuda_runtime PRIVATE cuda_rhi)
