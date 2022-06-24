# ./taichi/codegen/opengl/CMakeLists.txt

add_library(opengl_codegen)
target_sources(opengl_codegen
  PRIVATE
    codegen_opengl.cpp
    struct_opengl.cpp
  )

target_include_directories(opengl_codegen
  PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/external/spdlog/include
    ${PROJECT_SOURCE_DIR}/external/eigen
    ${LLVM_INCLUDE_DIRS}
  )

target_link_libraries(opengl_codegen PRIVATE opengl_runtime)
