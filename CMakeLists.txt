set(TAICHI_PROJECT_NAME "lang")
file(GLOB_RECURSE PROJECT_SOURCES "src/*.cpp" "src/*.h" "headers/*.h" "external/xxhash/*.c")
include_directories(include)
set(SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src")
set(TAICHI_PROJECT_LIBRARIES ${TAICHI_PROJECT_LIBRARIES} ${TAICHI_PROJECT_NAME} PARENT_SCOPE)

option(TLANG_WITH_VDB "Use VDB" OFF)

add_subdirectory(external/partio)
include_directories(external/openvdb/)
include_directories(external/partio/include)
include_directories(external/xxhash)

if (TLANG_WITH_VDB)
    list(APPEND  PROJECT_SOURCES "baselines/vdb/benchmark_vdb.cpp")
endif()


add_library(taichi_lang SHARED ${PROJECT_SOURCES})
target_link_libraries(taichi_lang ${CORE_LIBRARY_NAME} partio)

if (TLANG_WITH_VDB)
    target_link_libraries(taichi_lang ${CMAKE_CURRENT_LIST_DIR}/external/openvdb/openvdb/libopenvdb.so Half log4cplus boost_iostreams)
endif()

find_package(CUDA 8.0)
if (CUDA_FOUND)
    set(CUDA_ARCH 61)
    message("Found CUDA. Arch = ${CUDA_ARCH}")
    include_directories(/usr/local/cuda/include)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DCUDA_FOUND")
    target_link_libraries(taichi_lang /usr/local/cuda/lib64/libcudart.so )
else()
    message("CUDA Not Found.")
endif()

# find_package(MPI 3.1 REQUIRED)
# include_directories(SYSTEM ${MPI_INCLUDE_PATH})
# target_link_libraries(taichi_pangu ${MPI_C_LIBRARIES})
