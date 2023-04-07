# ./taichi/runtime/cpu/CMakeLists.txt

add_library(cpu_runtime)
target_sources(cpu_runtime
  PRIVATE
    jit_cpu.cpp
    kernel_launcher.cpp
  )

#TODO #4832, some path here should not be included as they are
# dependencies of other targets.
target_include_directories(cpu_runtime
  PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/external/eigen
    ${PROJECT_SOURCE_DIR}/external/spdlog/include
    ${LLVM_INCLUDE_DIRS}
  )
