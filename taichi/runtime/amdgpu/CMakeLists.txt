# ./taichi/runtime/amdgpu/CMakeLists.txt

add_library(amdgpu_runtime)
target_sources(amdgpu_runtime
  PRIVATE
    jit_amdgpu.cpp
    kernel_launcher.cpp
  )

target_include_directories(amdgpu_runtime
  PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/external/eigen
    ${PROJECT_SOURCE_DIR}/external/spdlog/include
    ${LLVM_INCLUDE_DIRS}
  )
