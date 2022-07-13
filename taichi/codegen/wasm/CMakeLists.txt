# ./taichi/codegen/wasm/CMakeLists.txt

add_library(wasm_codegen)
target_sources(wasm_codegen
  PRIVATE
    codegen_wasm.cpp
  )

target_include_directories(wasm_codegen
  PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/external/eigen
  )

target_link_libraries(wasm_codegen PRIVATE taichi_util)
