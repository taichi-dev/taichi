# ./taichi/runtime/program_impls/opengl/CMakeLists.txt

add_library(opengl_program_impl)
target_sources(opengl_program_impl
  PRIVATE
    opengl_program.cpp
  )

target_include_directories(opengl_program_impl
  PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/external/eigen
    ${PROJECT_SOURCE_DIR}/external/spdlog/include
    ${PROJECT_SOURCE_DIR}/external/SPIRV-Tools/include
    ${LLVM_INCLUDE_DIRS}
  )

target_link_libraries(opengl_program_impl PRIVATE gfx_runtime)
target_link_libraries(opengl_program_impl PRIVATE opengl_rhi)
