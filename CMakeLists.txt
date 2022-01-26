#*********************************************************************
# The Taichi Programming Language
#*********************************************************************

cmake_minimum_required(VERSION 3.12)

option(TI_EMSCRIPTENED "Build using emscripten" OFF)

if(TI_EMSCRIPTENED)
  if(DEFINED ENV{EMSCRIPTEN_DIR})
      set(EMSCRIPTEN_DIR $ENV{EMSCRIPTEN_DIR})
      # Tool chain file must be specified before the `project()`
      set(CMAKE_TOOLCHAIN_FILE "${EMSCRIPTEN_DIR}/upstream/emscripten/cmake/Modules/Platform/Emscripten.cmake")
      set(CMAKE_CROSSCOMPILING_EMULATOR "${EMSCRIPTEN_DIR}/node/14.15.5_64bit/bin/node.exe")
  else()
      message(FATAL_ERROR "To build taichi with emscripten, Please set EMSCRIPTEN_DIR environment variable.")
  endif()
endif()

project(taichi)

if (NOT DEFINED TI_VERSION_MAJOR)
    message(WARNING "It seems that you are running cmake manually, which may cause issues. Please use setup.py to build taichi from source, see https://docs.taichi.graphics/lang/articles/contribution/dev_install for more details.")
    set(TI_VERSION_MAJOR 0)
    set(TI_VERSION_MINOR 0)
    set(TI_VERSION_PATCH 0)
endif()

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

find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
    set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
endif()

# No support of Python for Android build
if (NOT ANDROID)
    include(cmake/PythonNumpyPybind11.cmake)
endif()
include(cmake/TaichiCXXFlags.cmake)
include(cmake/TaichiCore.cmake)

option(TI_BUILD_TESTS "Build the CPP tests" OFF)

if (TI_BUILD_TESTS)
  add_subdirectory(external/googletest EXCLUDE_FROM_ALL)
  include(cmake/TaichiTests.cmake)
endif()

option(TI_BUILD_EXAMPLES "Build the CPP examples" ON)

if (TI_BUILD_EXAMPLES)
  include(cmake/TaichiExamples.cmake)
endif()

include_directories(${PROJECT_SOURCE_DIR}/external/eigen)

message("C++ Flags: ${CMAKE_CXX_FLAGS}")
message("Build type: ${CMAKE_BUILD_TYPE}")

if (NOT TI_WITH_CUDA)
    set(CUDA_VERSION "0.0")
    set(CUDA_TOOLKIT_ROOT_DIR "")
endif()

if (TI_WITH_CUDA)
    set(CUDA_ARCH "cuda")
endif()

if (CLANG_EXECUTABLE)
  message("Clang executable ${CLANG_EXECUTABLE}")
elseif ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
  set (CLANG_EXECUTABLE ${CMAKE_CXX_COMPILER})
else()
  find_program(CLANG_EXECUTABLE NAMES clang clang-10 clang-11 clang-9 clang-8 clang-7)
endif()

if (NOT CLANG_EXECUTABLE)
  message(FATAL_ERROR "Cannot find any clang executable.")
endif()

execute_process(COMMAND ${CLANG_EXECUTABLE} --version OUTPUT_VARIABLE CLANG_VERSION_OUTPUT)
string(REGEX MATCH "([0-9]+)\\.[0-9]+(\\.[0-9]+)?" CLANG_VERSION "${CLANG_VERSION_OUTPUT}")
message("clang --version: ${CLANG_VERSION}")

set(CLANG_VERSION_MAJOR "${CMAKE_MATCH_1}")

if (APPLE)
  set(CLANG_OSX_FLAGS "-isysroot${CMAKE_OSX_SYSROOT}")
  set(CLANG_HIGHEST_VERSION "12")
else()
  set(CLANG_HIGHEST_VERSION "11")
endif()

if (${CLANG_VERSION_MAJOR} VERSION_GREATER ${CLANG_HIGHEST_VERSION})
  message(FATAL_ERROR "${CLANG_EXECUTABLE} version: ${CLANG_VERSION}, required: <=${CLANG_HIGHEST_VERSION}. Condider passing -DCLANG_PATH=/path/to/clang to cmake to use a specific clang.")
endif()

# Build llvm-runtime for host arch and cuda (if available)
foreach(arch IN LISTS HOST_ARCH CUDA_ARCH)
  add_custom_target(
      "generate_llvm_runtime_${arch}"
      COMMAND ${CLANG_EXECUTABLE} ${CLANG_OSX_FLAGS} -c runtime.cpp -o "runtime_${arch}.bc" -fno-exceptions -emit-llvm -std=c++17 -D "ARCH_${arch}" -I ${PROJECT_SOURCE_DIR};
      WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}/taichi/runtime/llvm"
  )
  add_dependencies(${CORE_LIBRARY_NAME} "generate_llvm_runtime_${arch}")
endforeach()

configure_file(taichi/common/version.h.in ${CMAKE_SOURCE_DIR}/taichi/common/version.h)
configure_file(taichi/common/commit_hash.h.in ${CMAKE_SOURCE_DIR}/taichi/common/commit_hash.h)

option(TI_EXPORT_CORE "export taichi core" OFF)

if (TI_EXPORT_CORE)
  include(cmake/TaichiExportCore.cmake)
endif()
