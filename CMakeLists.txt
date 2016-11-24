cmake_minimum_required(VERSION 2.8)
message("Module path:" ${CMAKE_MODULE_PATH})
project(taichi)

include_directories(src)

if (WIN32)
	set(BOOST_ROOT "C:/boost_1_62_0")
	link_directories(external/lib)
endif()

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/output")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -DGL_DO_NOT_WARN_IF_MULTI_GL_VERSION_HEADERS_INCLUDED")

if (WIN32)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP")
else()
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
endif()


list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake/")
list(APPEND CMAKE_MODULE_PATH "${CMAKE_ROOT}/Modules")

message("Module path:" ${CMAKE_MODULE_PATH})

set(CMAKE_BUILD_TYPE Release)

find_package(OpenGL REQUIRED)
find_package(assimp REQUIRED)
include_directories(${ASSIMP_INCLUDE_DIRS})

include_directories("external/include/glm")
# find_library(OPENCL_LIBRARIES OpenCL)

find_package(GLFW3 REQUIRED)
include_directories(${GLFW_INCLUDE_DIRS})
find_package(GLEW REQUIRED)
include_directories(${GLEW_INCLUDE_DIRS})

find_package(Boost  COMPONENTS
        system
        thread
        python REQUIRED)
if (WIN32)
	include_directories(${BOOST_ROOT})
else()
	include_directories(${BOOST_INCLUDE_DIRS})
endif()
find_package(PythonLibs REQUIRED)
if(APPLE)
    set(PYTHON_INCLUDE_DIRS "/usr/include/python2.7/")
    set(PYTHON_LIBRARIES "/Users/Iterator/homebrew/Cellar/python/2.7.8_1/Frameworks/Python.framework/Versions/2.7/lib/libpython2.7.dylib")
endif()
message("Python include ${PYTHON_INCLUDE_DIRS}")
include_directories(${PYTHON_INCLUDE_DIRS})
message("Python lib ${PYTHON_LIBRARIES}")

find_package(NumPy REQUIRED)
include_directories(${PYTHON_NUMPY_INCLUDE_DIR})
message(${PYTHON_NUMPY_INCLUDE_DIR})

find_package(embree REQUIRED)
include_directories(${EMBREE_INCLUDE_PATH})
message("Embree lib ${EMBREE_LIBRARY}")

find_package(ANN REQUIRED)

include_directories(external/include)

add_subdirectory(src)

message("Taichi Source: ${TAICHI_SOURCE}")

add_executable(run_taichi src/run_taichi.cpp)
if (NOT WIN32)
	target_link_libraries(run_taichi dl)
endif()
if (WIN32)
	set_target_properties(taichi_dll PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/runtimes")
	set_target_properties(run_taichi PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/runtimes")
else()
	set_target_properties(run_taichi PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/output")
endif()

