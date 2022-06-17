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
  )
