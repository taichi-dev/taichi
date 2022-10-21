# ./taichi/rhi/metal/CMakeLists.txt

set(METAL_RHI metal_rhi)
add_library(${METAL_RHI})
target_sources(${METAL_RHI}
  PRIVATE
    api.cpp
    device.cpp
  )

# TODO(#4832) some dependencies here should not be required as they
# are build requirements of other targets.
target_include_directories(${METAL_RHI}
  PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/external/spdlog/include
    ${PROJECT_SOURCE_DIR}/external/glad/include
    ${PROJECT_SOURCE_DIR}/external/glfw/include
    ${LLVM_INCLUDE_DIRS}
  )
