cmake_minimum_required(VERSION 3.0)

set(TAICHI_EXPORT_CORE_NAME taichi_export_core)

add_library(${TAICHI_EXPORT_CORE_NAME} SHARED)
target_link_libraries(${TAICHI_EXPORT_CORE_NAME} PRIVATE taichi_isolated_core)
set_target_properties(${TAICHI_EXPORT_CORE_NAME} PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/build"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/build")
