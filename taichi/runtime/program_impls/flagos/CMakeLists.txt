# ./taichi/runtime/program_impls/flagos/CMakeLists.txt

set(FLAGOS_PROGRAM flagos_program)
add_library(${FLAGOS_PROGRAM})

target_sources(${FLAGOS_PROGRAM}
  PRIVATE
    flagos_program.cpp
    flagos_kernel_compiler.cpp
    flagos_kernel_launcher.cpp
  )

target_include_directories(${FLAGOS_PROGRAM}
  PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/external/spdlog/include
    ${LLVM_INCLUDE_DIRS}
  )

target_link_libraries(${FLAGOS_PROGRAM}
  PRIVATE
    taichi_util
    taichi_ir
    llvm_program_impl
    flagos_rhi
    flagos_codegen
  )

if(TI_WITH_FLAGOS)
  target_compile_definitions(${FLAGOS_PROGRAM} PRIVATE TI_WITH_FLAGOS)
endif()
