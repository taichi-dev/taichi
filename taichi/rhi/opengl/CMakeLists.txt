# ./taichi/rhi/opengl/CMakeLists.txt

set(OPENGL_RHI opengl_rhi)
add_library(${OPENGL_RHI})
target_sources(${OPENGL_RHI}
  PRIVATE
    opengl_device.cpp
    opengl_api.cpp
    ${PROJECT_SOURCE_DIR}/external/glad/src/egl.c
    ${PROJECT_SOURCE_DIR}/external/glad/src/gl.c)

#TODO 4832, some dependencies here should not be required as they
# are build requirements of other targets.
target_include_directories(${OPENGL_RHI}
  PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/external/spdlog/include
    ${PROJECT_SOURCE_DIR}/external/glad/include
    ${PROJECT_SOURCE_DIR}/external/glfw/include
    ${PROJECT_SOURCE_DIR}/external/SPIRV-Cross
  )

if(NOT ANDROID)
target_link_libraries(opengl_rhi PRIVATE glfw)
endif()
target_link_libraries(opengl_rhi PRIVATE spirv-cross-glsl spirv-cross-core)
target_link_libraries(opengl_rhi PRIVATE common_rhi)
