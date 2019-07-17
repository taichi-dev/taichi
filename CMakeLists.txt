set(TAICHI_PROJECT_NAME "lang_core")
file(GLOB_RECURSE PROJECT_SOURCES "src/*.cpp" "src/*.h" "headers/*.h" "external/xxhash/*.c" "test/cpp/*.cpp" "examples/*.cpp")
include_directories(include)
set(SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src")
set(TAICHI_PROJECT_LIBRARIES ${TAICHI_PROJECT_LIBRARIES} ${TAICHI_PROJECT_NAME} PARENT_SCOPE)

option(TLANG_WITH_VDB "Use VDB" OFF)
option(TLANG_WITH_FEM "Use FEM" OFF)

# add_subdirectory(external/partio)
include_directories(external/openvdb/)
# include_directories(external/partio/include)
include_directories(external/xxhash)

if (TLANG_WITH_VDB)
    list(APPEND  PROJECT_SOURCES "baselines/vdb/benchmark_vdb.cpp")
    list(APPEND  PROJECT_SOURCES "baselines/vdb/convert_vdb.cpp")
endif()

if (NOT TLANG_WITH_FEM)
    set(SPGridSource "")
    list(REMOVE_ITEM PROJECT_SOURCES ${CMAKE_CURRENT_LIST_DIR}/examples/fem/fem.cpp)
else()
    include_directories(external/)
    file(GLOB SPGridSource "external/SPGrid/*/*.cpp")
endif()

set(LIBRARY_NAME taichi_lang_core)

add_library(${LIBRARY_NAME} SHARED ${PROJECT_SOURCES} ${SPGridSource})
target_link_libraries(${LIBRARY_NAME} ${CORE_LIBRARY_NAME})

if (TLANG_WITH_VDB)
    target_link_libraries(${LIBRARY_NAME} ${CMAKE_CURRENT_LIST_DIR}/external/openvdb/openvdb/libopenvdb.so Half log4cplus boost_iostreams)
endif()

if (TLANG_WITH_FEM)
    target_link_libraries(${LIBRARY_NAME} /home/yuanming/repos/topo_opt_private/solver/libSPGridCPUSolver.so )
    target_link_libraries(${LIBRARY_NAME} /opt/intel/compilers_and_libraries_2019/linux/mkl/lib/intel64_lin/libmkl_rt.so)
endif()

find_package(CUDA 8.0)
if (CUDA_FOUND)
    set(CUDA_ARCH 61)
    message("Found CUDA. Arch = ${CUDA_ARCH}")
    include_directories(/usr/local/cuda/include)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DCUDA_FOUND")
    target_link_libraries(${LIBRARY_NAME} /usr/local/cuda/lib64/libcudart.so )
else()
    message("CUDA not found.")
endif()

# http://llvm.org/docs/CMake.html#embedding-llvm-in-your-project
find_package(LLVM CONFIG 8.0)
if (LLVM_FOUND)
    message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
    message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
    include_directories(${LLVM_INCLUDE_DIRS})
    add_definitions(${LLVM_DEFINITIONS})
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DTLANG_WITH_LLVM")
    llvm_map_components_to_libnames(llvm_libs
            Analysis
            Core
            ExecutionEngine
            InstCombine
            Object
            OrcJIT
            RuntimeDyld
            ScalarOpts
            Support
            native
            )
    target_link_libraries(${LIBRARY_NAME} ${llvm_libs})
else()
    message("LLVM not found.")
endif()

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DOPENMP_FOUND")
    message("Found OpenMP.")
else()
    message("OpenMP not found.")
endif()
