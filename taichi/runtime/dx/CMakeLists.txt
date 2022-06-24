# ./taichi/runtime/dx/CMakeLists.txt

add_library(dx_runtime)
target_sources(dx_runtime
  PRIVATE
    dx_api.cpp
  )

target_include_directories(dx_runtime
  PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/external/glad/include
    ${PROJECT_SOURCE_DIR}/external/spdlog/include
  )
