# ./taichi/codegen/flagos/CMakeLists.txt

set(FLAGOS_CODEGEN flagos_codegen)
add_library(${FLAGOS_CODEGEN})

target_sources(${FLAGOS_CODEGEN}
  PRIVATE
    codegen_flagos.cpp
  )

target_include_directories(${FLAGOS_CODEGEN}
  PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/external/spdlog/include
    ${LLVM_INCLUDE_DIRS}
  )

target_link_libraries(${FLAGOS_CODEGEN}
  PRIVATE
    taichi_util
    taichi_ir
    llvm_codegen
  )

if(TI_WITH_FLAGOS)
  target_compile_definitions(${FLAGOS_CODEGEN} PRIVATE TI_WITH_FLAGOS)
endif()
