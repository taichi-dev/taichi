project(vdb_baseline)

cmake_minimum_required(VERSION 3.8)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DTC_ISE_AVX2 -march=native -DGL_DO_NOT_WARN_IF_MULTI_GL_VERSION_HEADERS_INCLUDED -Wall -DTC_PASS_EXCEPTION_TO_PYTHON -g")

add_executable(vdb_baseline test.cpp)
include_directories(${CMAKE_CURRENT_LIST_DIR})
set_target_properties(vdb_baseline
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
        )
target_link_libraries(vdb_baseline tbb ${CMAKE_CURRENT_LIST_DIR}/libopenvdb.so boost_iostreams Half pthread X11)
