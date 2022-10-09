# ./taichi/runtime/dx12/CMakeLists.txt

add_library(dx12_runtime)
target_sources(dx12_runtime
  PRIVATE
    aot_module_builder_impl.cpp
  )

target_include_directories(dx12_runtime
  PRIVATE
    ${PROJECT_SOURCE_DIR}/external/eigen
    ${PROJECT_SOURCE_DIR}/external/spdlog/include
    ${PROJECT_SOURCE_DIR}
    ${LLVM_INCLUDE_DIRS}
  )

target_link_libraries(dx12_runtime PRIVATE dx12_rhi)
