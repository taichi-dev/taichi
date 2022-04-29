# ./taichi/runtime/opengl/CMakeLists.txt

add_library(opengl_runtime)
target_sources(opengl_runtime
  PRIVATE
    opengl_api.cpp
  )
target_include_directories(opengl_runtime
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders
    ${PROJECT_SOURCE_DIR}/external/SPIRV-Tools/include
    ${PROJECT_SOURCE_DIR}/external/glad/include
    ${PROJECT_SOURCE_DIR}/external/eigen
  )
