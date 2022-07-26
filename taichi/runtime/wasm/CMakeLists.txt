# ./taichi/runtime/wasm/CMakeLists.txt

add_library(wasm_runtime)
target_sources(wasm_runtime
  PRIVATE
    aot_module_builder_impl.cpp
  )

target_include_directories(wasm_runtime
  PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/external/spdlog/include
    ${PROJECT_SOURCE_DIR}/external/eigen
    ${LLVM_INCLUDE_DIRS}
  )
