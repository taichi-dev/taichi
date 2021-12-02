cmake_minimum_required(VERSION 3.12)

set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -fsized-deallocation -Wall -DTI_ARCH_x64 -DTI_INCLUDED -O3 -DNDEBUG")
set(CHI_EXAMPLES chi_examples)

if (NOT DEFINED ENV{TAICHI_REPO_DIR})
    set(ENV{TAICHI_REPO_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/..)
endif()

project(chi_examples)

add_executable(${CHI_EXAMPLES} main.cpp)
target_include_directories(${CHI_EXAMPLES}
    PUBLIC $ENV{TAICHI_REPO_DIR}
    PUBLIC $ENV{TAICHI_REPO_DIR}/external/eigen
    PUBLIC $ENV{TAICHI_REPO_DIR}/external/include
    PUBLIC $ENV{TAICHI_REPO_DIR}/external/spdlog/include
    PUBLIC $ENV{TAICHI_REPO_DIR}/external/glad/include
    PUBLIC $ENV{TAICHI_REPO_DIR}/external/glfw/include
)
target_link_libraries(${CHI_EXAMPLES} $ENV{TAICHI_REPO_DIR}/build/libtaichi_export_core.so)
