#*********************************************************************
# The Taichi Programming Language
#*********************************************************************

cmake_minimum_required(VERSION 3.12)

project(taichi)

SET(TI_VERSION_MAJOR ${TI_VERSION_MAJOR})
SET(TI_VERSION_MINOR ${TI_VERSION_MINOR})
SET(TI_VERSION_PATCH ${TI_VERSION_PATCH})

set(CMAKE_CXX_STANDARD 17)

execute_process(
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  COMMAND git rev-parse --short HEAD
  RESULT_VARIABLE SHORT_HASH_RESULT
  OUTPUT_VARIABLE TI_COMMIT_SHORT_HASH)
execute_process(
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  COMMAND git rev-parse HEAD
  RESULT_VARIABLE SHORT_HASH_RESULT
  OUTPUT_VARIABLE TI_COMMIT_HASH)
string(STRIP ${TI_COMMIT_HASH} TI_COMMIT_HASH)
string(STRIP ${TI_COMMIT_SHORT_HASH} TI_COMMIT_SHORT_HASH)

message("Taichi Version ${TI_VERSION_MAJOR}.${TI_VERSION_MINOR}.${TI_VERSION_PATCH}")
message("       Commit ${TI_COMMIT_HASH}")

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING
            "Choose the type of build, options are: Debug Release
RelWithDebInfo MinSizeRel."
            FORCE)
endif(NOT CMAKE_BUILD_TYPE)

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

include(cmake/PythonNumpyPybind11.cmake)
include(cmake/TaichiCXXFlags.cmake)
include(cmake/TaichiCore.cmake)
include(cmake/TaichiMain.cmake)

option(TI_BUILD_TESTS "Build the CPP tests" OFF)

if (TI_BUILD_TESTS)
  add_subdirectory(external/googletest EXCLUDE_FROM_ALL)
  include(cmake/TaichiTests.cmake)
endif()

include_directories(${PROJECT_SOURCE_DIR}/external/eigen)

message("C++ Flags: ${CMAKE_CXX_FLAGS}")
message("Build type: ${CMAKE_BUILD_TYPE}")

if (NOT TI_WITH_CUDA)
    set(CUDA_VERSION "0.0")
    set(CUDA_TOOLKIT_ROOT_DIR "")
endif()

message("python=${PYTHON_EXECUTABLE}")

add_custom_target(
    generate_commit_hash
    COMMAND ${PYTHON_EXECUTABLE} "${CMAKE_CURRENT_LIST_DIR}/misc/generate_commit_hash.py"
)
add_dependencies(${CORE_LIBRARY_NAME} generate_commit_hash)

if (TI_WITH_CUDA)
    set(CUDA_ARCH "cuda")
endif()

find_program(CLANG_EXECUTABLE NAMES clang clang-7 clang-8 clang-9 clang-10)
if (NOT CLANG_EXECUTABLE)
    message(FATAL_ERROR "Cannot find any clang executable.")
endif()

find_program(LLVM_AS_EXECUTABLE NAMES llvm-as)
if (NOT LLVM_AS_EXECUTABLE)
    message(FATAL_ERROR "Cannot find llvm-as executable.")
endif()

# Build llvm-runtime for host arch and cuda (if available)
foreach(arch IN LISTS HOST_ARCH CUDA_ARCH)
  add_custom_target(
      "generate_llvm_runtime_${arch}"
      COMMAND ${CLANG_EXECUTABLE} -S runtime.cpp -o "runtime_${arch}.ll" -fno-exceptions -emit-llvm -std=c++17 -D "ARCH_${arch}" -I ${PROJECT_SOURCE_DIR};
      COMMAND ${LLVM_AS_EXECUTABLE} "runtime_${arch}.ll" -o "runtime_${arch}.bc"
      WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}/taichi/runtime/llvm"
  )
  add_dependencies(${CORE_LIBRARY_NAME} "generate_llvm_runtime_${arch}")
endforeach()


FILE(WRITE ${CMAKE_CURRENT_LIST_DIR}/taichi/common/version.h
        "#pragma once\n"
        "#define TI_VERSION_MAJOR \"${TI_VERSION_MAJOR}\"\n"
        "#define TI_VERSION_MINOR \"${TI_VERSION_MINOR}\"\n"
        "#define TI_VERSION_PATCH \"${TI_VERSION_PATCH}\"\n"
        "#define TI_CUDAVERSION \"${CUDA_VERSION}\"\n"
        )

option(TI_EXPORT_CORE "export taichi core" OFF)

if (TI_EXPORT_CORE)
  include(cmake/TaichiExportCore.cmake)
endif()
