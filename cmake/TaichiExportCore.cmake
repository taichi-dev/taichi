cmake_minimum_required(VERSION 3.0)

set(TAICHI_EXPORT_CORE_NAME taichi_export_core)

add_library(${TAICHI_EXPORT_CORE_NAME} SHARED)
target_link_libraries(${TAICHI_EXPORT_CORE_NAME} taichi_isolated_core)
install(TARGETS ${TAICHI_EXPORT_CORE_NAME} DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/build)
