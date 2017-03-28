cmake_minimum_required(VERSION 2.8)
project(taichi)

if (WIN32)
    link_directories(${CMAKE_CURRENT_SOURCE_DIR}/external/lib)
endif ()

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/build")

message("Using c++ compiler: " ${CMAKE_CXX_COMPILER})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -DGL_DO_NOT_WARN_IF_MULTI_GL_VERSION_HEADERS_INCLUDED")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DTC_PASS_EXCEPTION_TO_PYTHON")

if (USE_OPENGL)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DTC_USE_OPENGL")
    find_package(OpenGL REQUIRED)
    find_package(GLFW3 REQUIRED)
    include_directories(${GLFW_INCLUDE_DIRS})
    find_package(GLEW REQUIRED)
    include_directories(${GLEW_INCLUDE_DIRS})
endif ()

if (WIN32)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP /Z7 /D \"_CRT_SECURE_NO_WARNINGS\"  /arch:AVX")
else ()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
endif ()

if (WIN32)
    list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake/")
    list(APPEND CMAKE_MODULE_PATH "${CMAKE_ROOT}/Modules")
else ()
    list(APPEND CMAKE_MODULE_PATH "${CMAKE_ROOT}/Modules")
    list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake/")
endif ()

set(CMAKE_BUILD_TYPE Release)

# Python and numpy

if (PYTHON_EXECUTABLE)
    message("Using ${PYTHON_EXECUTABLE} as python executable.")
else ()
    message("Using 'python' as python executable.")
    set(PYTHON_EXECUTABLE python)
endif ()


if (WIN32)
    find_package(PythonLibs 2.7 REQUIRED)
else ()
    execute_process(COMMAND ${PYTHON_EXECUTABLE} --version)
    execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "import sys; from distutils import sysconfig; sys.stdout.write(sysconfig.get_config_var('INCLUDEDIR') + '/python2.7/')"
            OUTPUT_VARIABLE PYTHON_INCLUDE_DIRS)
    execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "import sys; from distutils import sysconfig; sys.stdout.write(sysconfig.get_config_var('LIBDIR'))"
            OUTPUT_VARIABLE PYTHON_LIBRARY_DIR)

    find_library(PYTHON_LIBRARY NAMES python2.7 PATHS ${PYTHON_LIBRARY_DIR} NO_DEFAULT_PATH NO_SYSTEM_ENVIRONMENT_PATH PATH_SUFFIXES x86_64-linux-gnu)
    set(PYTHON_LIBRARIES ${PYTHON_LIBRARY})
endif ()

include_directories(${PYTHON_INCLUDE_DIRS})
message("    include: ${PYTHON_INCLUDE_DIRS}")
message("    library: ${PYTHON_LIBRARIES}")

execute_process(COMMAND python -c "import numpy.distutils, sys; sys.stdout.write(':'.join(numpy.distutils.misc_util.get_numpy_include_dirs()))"
        OUTPUT_VARIABLE PYTHON_NUMPY_INCLUDE_DIR)

message("    numpy include: ${PYTHON_NUMPY_INCLUDE_DIR}")
include_directories(${PYTHON_NUMPY_INCLUDE_DIR})

execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "import sys; import pybind11; sys.stdout.write(pybind11.get_include())"
        OUTPUT_VARIABLE PYBIND11_INCLUDE_DIR
        RESULT_VARIABLE PYBIND11_IMPORT_RET)
if (NOT PYBIND11_IMPORT_RET)
    message("    pybind11 include: " ${PYBIND11_INCLUDE_DIR})
else()
    message(FATAL_ERROR "Can not import pybind11. Please install. ([sudo] pip install pybind11)")
endif()

include_directories(${PYBIND11_INCLUDE_DIR})

find_package(embree REQUIRED)
include_directories(${EMBREE_INCLUDE_PATH})
message("Embree")
message("    include: " ${EMBREE_INCLUDE_PATH})
message("    library: " ${EMBREE_LIBRARY})

include_directories(include)
include_directories(external/include)

file(GLOB TAICHI_SOURCE
        "src/*/*/*/*.cpp" "src/*/*/*.cpp" "src/*/*.cpp" "src/*.cpp" "src/*/*/*/*.h" "src/*/*/*.h" "src/*/*.h" "src/*.h"
        "include/taichi/*/*/*/*.cpp" "include/taichi/*/*/*.cpp" "include/taichi/*/*.cpp" "include/taichi/*.cpp"
        "include/taichi/*/*/*/*.h" "include/taichi/*/*/*.h" "include/taichi/*/*.h" "include/taichi/*.h")

set(CORE_LIBRARY_NAME taichi_core)
add_library(${CORE_LIBRARY_NAME} SHARED ${TAICHI_SOURCE})

if (NOT WIN32)
    target_link_libraries(${CORE_LIBRARY_NAME} pthread)
endif ()

# Optional dependencies

if (USE_OPENGL)
    target_link_libraries(${CORE_LIBRARY_NAME} ${GLEW_LIBRARY})
    target_link_libraries(${CORE_LIBRARY_NAME} ${GLEW_LIBRARIES})
    target_link_libraries(${CORE_LIBRARY_NAME} ${OPENGL_LIBRARIES})
    target_link_libraries(${CORE_LIBRARY_NAME} ${GLFW3_LIBRARY})
    if (APPLE)
        target_link_libraries(${CORE_LIBRARY_NAME} glfw3)
    endif ()
endif ()

# Required dependencies

target_link_libraries(${CORE_LIBRARY_NAME} ${EMBREE_LIBRARY})
target_link_libraries(${CORE_LIBRARY_NAME} ${PYTHON_LIBRARIES})

foreach (source IN LISTS TAICHI_SOURCE)
    get_filename_component(source_path "${source}" PATH)
    string(REPLACE "/" "\\" source_path_msvc "${source_path}")
    source_group("${source_path_msvc}" FILES "${source}")
endforeach ()

if (MSVC)
    set_property(TARGET ${CORE_LIBRARY_NAME} APPEND PROPERTY LINK_FLAGS /DEBUG)
endif ()

if (WIN32)
    set_target_properties(${CORE_LIBRARY_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/runtimes")
endif ()
