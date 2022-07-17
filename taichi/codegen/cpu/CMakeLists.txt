# ./taichi/codegen/cpu/CMakeLists.txt

add_library(cpu_codegen)
target_sources(cpu_codegen
  PRIVATE
    codegen_cpu.cpp
  )

target_include_directories(cpu_codegen
  PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/external/eigen
    ${LLVM_INCLUDE_DIRS}
  )

target_link_libraries(cpu_codegen PRIVATE taichi_util)
target_link_libraries(cpu_codegen PRIVATE cpu_runtime)
