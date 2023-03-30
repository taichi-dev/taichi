# ./taichi/common/CMakeLists.txt

add_library(taichi_common)
target_sources(taichi_common
  PRIVATE
    cleanup.cpp
    core.cpp
    json.cpp
    logging.cpp
    symbol_version.cpp
    virtual_dir.cpp
    zip.cpp
    miniz.c
    dynamic_loader.cpp
  )

target_include_directories(taichi_common
  PUBLIC
    ${PROJECT_SOURCE_DIR}/external/spdlog/include
  PRIVATE
    ${PROJECT_SOURCE_DIR}
  )
