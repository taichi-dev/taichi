# ./taichi/codegen/dx12/CMakeLists.txt

add_library(dx12_codegen)
target_sources(dx12_codegen
  PRIVATE
    codegen_dx12.cpp
  )

target_include_directories(dx12_codegen
  PRIVATE
    ${PROJECT_SOURCE_DIR}/external/eigen
    ${PROJECT_SOURCE_DIR}/external/spdlog/include
    ${PROJECT_SOURCE_DIR}
    ${LLVM_INCLUDE_DIRS}
  )

target_link_libraries(dx12_codegen PRIVATE taichi_util)
target_link_libraries(dx12_codegen PRIVATE dx12_runtime)
