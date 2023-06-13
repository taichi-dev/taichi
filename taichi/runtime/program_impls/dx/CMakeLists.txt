# ./taichi/runtime/program_impls/dx/CMakeLists.txt

add_library(dx_program_impl)
target_sources(dx_program_impl
  PRIVATE
    dx_program.cpp
  )

target_include_directories(dx_program_impl
  PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/external/eigen
    ${PROJECT_SOURCE_DIR}/external/spdlog/include
    ${PROJECT_SOURCE_DIR}/external/SPIRV-Tools/include
    ${LLVM_INCLUDE_DIRS}
  )

target_link_libraries(dx_program_impl PRIVATE compilation_manager)
target_link_libraries(dx_program_impl PRIVATE gfx_program_impl)
