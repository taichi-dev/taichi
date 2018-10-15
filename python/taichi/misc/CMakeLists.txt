cmake_minimum_required(VERSION 2.8)
project(taichi_unit)

if (WIN32)
    link_directories($ENV{TAICHI_REPO_DIR}/external/lib)
endif ()

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/build")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -DGL_DO_NOT_WARN_IF_MULTI_GL_VERSION_HEADERS_INCLUDED")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DTC_PASS_EXCEPTION_TO_PYTHON")

if (WIN32)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP /Z7 /arch:AVX")
else ()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
endif ()

set(CMAKE_BUILD_TYPE Release)

include_directories("$ENV{TAICHI_REPO_DIR}/include")
include_directories("$ENV{TAICHI_REPO_DIR}/external/include")

set(UNIT_SOURCE "unit.cpp")

set(CORE_LIBRARY_NAME unit)

add_library(${CORE_LIBRARY_NAME} SHARED ${UNIT_SOURCE})
target_link_libraries(${CORE_LIBRARY_NAME} $ENV{TAICHI_REPO_DIR}/build/libtaichi_core.dylib)

if (MSVC)
    set_property(TARGET ${CORE_LIBRARY_NAME} APPEND PROPERTY LINK_FLAGS /DEBUG)
endif ()

if (WIN32)
    set_target_properties(${CORE_LIBRARY_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/runtimes")
endif ()
