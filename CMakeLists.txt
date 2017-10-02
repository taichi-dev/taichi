#*********************************************************************
# Taichi - Physically based Computer Graphics Library
#
# All rights reserved. Use of this source code is governed by
# the MIT license as written in the LICENSE file.
#*********************************************************************

cmake_minimum_required(VERSION 2.8)

project(taichi)

set(CMAKE_BUILD_TYPE Release)

set(TAICHI_CMAKE_DIR "${CMAKE_CURRENT_LIST_DIR}/cmake/")

if (WIN32)
    list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake/")
    list(APPEND CMAKE_MODULE_PATH "${CMAKE_ROOT}/Modules")
else ()
    list(APPEND CMAKE_MODULE_PATH "${CMAKE_ROOT}/Modules")
    list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake/")
endif ()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/build")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/build")

include(cmake/TaichiCXXFlags.cmake)
include(cmake/TaichiCore.cmake)
include(cmake/TaichiPakua.cmake)
include(cmake/TaichiMain.cmake)

set(TAICHI_PROJECT_LIBRARIES "")
if (EXISTS "${CMAKE_CURRENT_LIST_DIR}/projects")
    add_subdirectory(projects)
endif()

message("Project libraries: ${TAICHI_PROJECT_LIBRARIES}")
file(WRITE modules.txt "${TAICHI_PROJECT_LIBRARIES}")
