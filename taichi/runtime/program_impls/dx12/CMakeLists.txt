# ./taichi/runtime/program_impls/dx12/CMakeLists.txt

add_library(dx12_program_impl)
target_sources(dx12_program_impl
  PRIVATE
    dx12_program.cpp
  )

target_include_directories(dx12_program_impl
  PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/external/eigen
    ${PROJECT_SOURCE_DIR}/external/spdlog/include

    ${PROJECT_SOURCE_DIR}/external/DirectX-Headers/include
    ${LLVM_INCLUDE_DIRS}
  )
