cmake_minimum_required(VERSION 2.8)

project(pakua)

set(CMAKE_BUILD_TYPE Release)

set(TAICHI_INCLUDE_DIR "$ENV{TAICHI_ROOT_DIR}/taichi/include")
set(TAICHI_CMAKE_DIR "$ENV{TAICHI_ROOT_DIR}/taichi/cmake")
message("Taichi include: ${TAICHI_INCLUDE_DIR}")

if (WIN32)
    list(APPEND CMAKE_MODULE_PATH "${TAICHI_CMAKE_DIR}")
    list(APPEND CMAKE_MODULE_PATH "${CMAKE_ROOT}/Modules")
else ()
    list(APPEND CMAKE_MODULE_PATH "${CMAKE_ROOT}/Modules")
    list(APPEND CMAKE_MODULE_PATH "${TAICHI_CMAKE_DIR}")
    list(APPEND CMAKE_MODULE_PATH "${TAICHI_ROOT_DIR}")
endif ()

include(${TAICHI_CMAKE_DIR}/TaichiCXXFlags.cmake)
message(${CMAKE_CXX_FLAGS})

include_directories(${TAICHI_INCLUDE_DIR})
include_directories(external/include)
include_directories(include)

