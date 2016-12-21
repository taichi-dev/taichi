cmake_minimum_required(VERSION 2.8)
message("Module path:" ${CMAKE_MODULE_PATH})
project(taichi)

include_directories(src)

if (WIN32)
    set(BOOST_ROOT "C:/boost_1_62_0")
    link_directories(external/lib)
endif ()

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/output")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -DGL_DO_NOT_WARN_IF_MULTI_GL_VERSION_HEADERS_INCLUDED")
if (USE_OPENGL)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DTC_USE_OPENGL")
    find_package(OpenGL REQUIRED)
    find_package(GLFW3 REQUIRED)
    include_directories(${GLFW_INCLUDE_DIRS})
    find_package(GLEW REQUIRED)
    include_directories(${GLEW_INCLUDE_DIRS})
endif()

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

message("Module path:" ${CMAKE_MODULE_PATH})

set(CMAKE_BUILD_TYPE Release)

include_directories("external/include/glm")
# find_library(OPENCL_LIBRARIES OpenCL)

find_package(Boost COMPONENTS
        system
        thread
        python REQUIRED)
if (WIN32)
    include_directories(${BOOST_ROOT})
else ()
    include_directories(${BOOST_INCLUDE_DIRS})
endif ()
find_package(PythonLibs REQUIRED)
if (APPLE)
    set(PYTHON_INCLUDE_DIRS "/usr/include/python2.7/")
    set(PYTHON_LIBRARIES "/Users/Iterator/homebrew/Cellar/python/2.7.8_1/Frameworks/Python.framework/Versions/2.7/lib/libpython2.7.dylib")
endif ()
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

include_directories(include)
include_directories(external/include)

file(GLOB TAICHI_SOURCE
        "src/*/*/*/*.cpp" "src/*/*/*.cpp" "src/*/*.cpp" "src/*.cpp" "src/*/*/*/*.h" "src/*/*/*.h" "src/*/*.h" "src/*.h"
        "include/taichi/*/*/*/*.cpp" "include/taichi/*/*/*.cpp" "include/taichi/*/*.cpp" "include/taichi/*.cpp"
        "include/taichi/*/*/*/*.h" "include/taichi/*/*/*.h" "include/taichi/*/*.h" "include/taichi/*.h"
        )

set(CORE_LIBRARY_NAME taichi_core)
add_library(${CORE_LIBRARY_NAME} SHARED ${TAICHI_SOURCE})

if (NOT WIN32)
    target_link_libraries(${CORE_LIBRARY_NAME} pthread)
endif()

if (USE_OPENGL)
    target_link_libraries(${CORE_LIBRARY_NAME} ${GLEW_LIBRARY})
    target_link_libraries(${CORE_LIBRARY_NAME} ${GLEW_LIBRARIES})
    target_link_libraries(${CORE_LIBRARY_NAME} ${OPENGL_LIBRARIES})
    target_link_libraries(${CORE_LIBRARY_NAME} ${GLFW3_LIBRARY})
if (APPLE)
    target_link_libraries(${CORE_LIBRARY_NAME} glfw3)
endif()
endif()
target_link_libraries(${CORE_LIBRARY_NAME} ${EMBREE_LIBRARY})
target_link_libraries(${CORE_LIBRARY_NAME} ${PYTHON_LIBRARIES})
target_link_libraries(${CORE_LIBRARY_NAME} ${ANN_LIBRARIES})
target_link_libraries(${CORE_LIBRARY_NAME} ${Boost_LIBRARIES})

if (UNIX)
    target_link_libraries(${CORE_LIBRARY_NAME} boost_system)
    target_link_libraries(${CORE_LIBRARY_NAME} boost_python)
endif()

foreach(source IN LISTS TAICHI_SOURCE)
    get_filename_component(source_path "${source}" PATH)
    string(REPLACE "/" "\\" source_path_msvc "${source_path}")
    source_group("${source_path_msvc}" FILES "${source}")
endforeach()

if(MSVC)
    set_property(TARGET ${CORE_LIBRARY_NAME} APPEND PROPERTY LINK_FLAGS /DEBUG)
endif()

if (WIN32)
    set_target_properties(${CORE_LIBRARY_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/runtimes")
endif ()
