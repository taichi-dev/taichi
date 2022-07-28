# ./taichi/runtime/program_impls/metal/CMakeLists.txt

add_library(metal_program_impl)
target_sources(metal_program_impl
  PRIVATE
    metal_program.cpp
  )

target_include_directories(metal_program_impl
  PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${LLVM_INCLUDE_DIRS}
  )

target_link_libraries(metal_program_impl PUBLIC metal_codegen)
target_link_libraries(metal_program_impl PUBLIC metal_runtime)
