cmake_minimum_required(VERSION 3.5)

project(c_api_demo)

set(CMAKE_CXX_STANDARD 11)

set(TAICHI_REPO_DIR ${CMAKE_SOURCE_DIR}/../..)
message("TAICHI_REPO_DIR=${TAICHI_REPO_DIR}")
find_library(TAICHI_C_API libtaichi_c_api.so PATHS ${TAICHI_REPO_DIR}/build/)
message("TAICHI_C_API=${TAICHI_C_API}")
add_executable(aot_demo src/main.c)
target_include_directories(aot_demo
    PUBLIC
    ${TAICHI_REPO_DIR}
)
target_link_libraries(aot_demo ${TAICHI_C_API})
