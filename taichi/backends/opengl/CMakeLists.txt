# ./taichi/backends/vulkan/CMakeLists.txt

set(OPENGL_RHI opengl_rhi)
add_library(${OPENGL_RHI})
target_sources(${OPENGL_RHI}
  PRIVATE
    opengl_device.cpp
  )

#TODO 4832, some dependencies here should not be required as they
# are build requirements of other targets.
# public dirs here are required by backends/device.cpp
target_include_directories(${OPENGL_RHI}
  PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/external/spdlog/include
    ${PROJECT_SOURCE_DIR}/external/glad/include
  )
