cmake_minimum_required(VERSION 3.0)

set(TAICHI_EXPORT_CORE_NAME taichi_export_core)

add_library(${TAICHI_EXPORT_CORE_NAME} SHARED)
target_link_libraries(${TAICHI_EXPORT_CORE_NAME} taichi_isolated_core)

set(TAICHI_C_API_LIB_NAME taichi_c_api)

file(GLOB_RECURSE C_API_SOURCE
    "c_api/src/*.cpp"
)
add_library(${TAICHI_C_API_LIB_NAME} SHARED ${C_API_SOURCE})

target_link_libraries(${TAICHI_C_API_LIB_NAME} taichi_isolated_core)
