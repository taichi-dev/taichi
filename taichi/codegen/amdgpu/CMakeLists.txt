# ./taichi/codegen/amdgpu/CMakeLists.txt

add_library(amdgpu_codegen)
target_sources(amdgpu_codegen
  PRIVATE
    codegen_amdgpu.cpp
  )

target_include_directories(amdgpu_codegen
  PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/external/eigen
    ${LLVM_INCLUDE_DIRS}
  )

target_link_libraries(amdgpu_codegen PRIVATE taichi_util)
target_link_libraries(amdgpu_codegen PRIVATE amdgpu_runtime)
