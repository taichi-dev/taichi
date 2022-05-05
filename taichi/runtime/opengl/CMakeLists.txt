# ./taichi/runtime/opengl/CMakeLists.txt

add_library(opengl_runtime)
target_sources(opengl_runtime
  PRIVATE
    opengl_api.cpp
  )

#TODO #4832, some path here should not be included as they are
# dependencies of other targets.
target_include_directories(opengl_runtime
  PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders
    ${PROJECT_SOURCE_DIR}/external/SPIRV-Tools/include
    ${PROJECT_SOURCE_DIR}/external/glad/include
    ${PROJECT_SOURCE_DIR}/external/eigen
    ${PROJECT_SOURCE_DIR}/external/glfw/include
  PRIVATE
    ${PROJECT_SOURCE_DIR}/external/spdlog/include
    ${LLVM_INCLUDE_DIRS}
  )
