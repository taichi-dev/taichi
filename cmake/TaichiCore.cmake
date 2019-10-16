set(CORE_LIBRARY_NAME taichi_core)

include(cmake/PythonNumpyPybind11.cmake)

file(GLOB TAICHI_CORE_SOURCE
        "src/*/*/*/*.cpp" "src/*/*/*.cpp" "src/*/*.cpp" "src/*.cpp"
        "src/*/*/*/*.h" "src/*/*/*.h" "src/*/*.h" "src/*.h"
        "include/taichi/*/*/*/*.cpp" "include/taichi/*/*/*.cpp" "include/taichi/*/*.cpp"
        "include/taichi/*/*/*/*.h" "include/taichi/*/*/*.h" "include/taichi/*/*.h")

file(GLOB_RECURSE PROJECT_SOURCES "lang/src/*.cpp" "lang/src/*.h" "lang/headers/*.h" "external/xxhash/*.c" "lang/test/cpp/*.cpp" "lang/cpp_examples/*.cpp")
include_directories(lang/include)

add_library(${CORE_LIBRARY_NAME} SHARED ${TAICHI_CORE_SOURCE} ${PROJECT_SOURCES} ${SPGridSource})

option(USE_STDCPP "Use -stdlib=libc++" OFF)
option(TLANG_WITH_VDB "Use VDB" OFF)
option(TLANG_WITH_FEM "Use FEM" OFF)
option(TLANG_WITH_CUDA "Build with GPU support" ON)

# include_directories(external/openvdb/)
include_directories(external/xxhash)

if (TLANG_WITH_VDB)
    list(APPEND  PROJECT_SOURCES "baselines/vdb/benchmark_vdb.cpp")
    list(APPEND  PROJECT_SOURCES "baselines/vdb/convert_vdb.cpp")
endif()

if (NOT TLANG_WITH_FEM)
    set(SPGridSource "" src/kernel.cpp src/kernel.h)
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DTLANG_WITH_FEM")
    include_directories(external/)
    file(GLOB SPGridSource "external/SPGrid/*/*.cpp")
endif()

set(LIBRARY_NAME ${CORE_LIBRARY_NAME})

if (TLANG_WITH_VDB)
    target_link_libraries(${LIBRARY_NAME} ${CMAKE_CURRENT_LIST_DIR}/external/openvdb/openvdb/libopenvdb.so Half log4cplus boost_iostreams)
endif()

if (TLANG_WITH_FEM)
    # Change the path to your own libSPGridCPUSolver.so
    target_link_libraries(${LIBRARY_NAME} /home/user/repos/topo_opt_private/solver/libSPGridCPUSolver.so )
    target_link_libraries(${LIBRARY_NAME} /opt/intel/compilers_and_libraries_2019/linux/mkl/lib/intel64_lin/libmkl_rt.so)
endif()

if (TLANG_WITH_CUDA)
    find_package(CUDA 10.0)
    if (CUDA_FOUND)
        set(CUDA_ARCH 61)
        message("Found CUDA. Arch = ${CUDA_ARCH}")
        include_directories(/usr/local/cuda/include)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DCUDA_FOUND -DTLANG_WITH_CUDA")
        target_link_libraries(${LIBRARY_NAME} /usr/local/cuda/lib64/libcudart.so cuda)
    else()
        message("CUDA not found.")
    endif()
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
            Core
            ExecutionEngine
            InstCombine
            OrcJIT
            RuntimeDyld
            TransformUtils
            BitReader
            BitWriter
            Object
            ScalarOpts
            Support
            native
            NVPTX
            Linker
            )
    target_link_libraries(${LIBRARY_NAME} ${llvm_libs})
else()
    message("LLVM not found.")
endif()

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DOPENMP_FOUND -DTLANG_WITH_OPENMP")
    message("Found OpenMP.")
else()
    message("OpenMP not found.")
endif()

# add_executable(runtime runtime/runtime.cpp)

# Optional dependencies

if (APPLE)
    target_link_libraries(${CORE_LIBRARY_NAME} "-framework Cocoa")
endif ()

if (NOT WIN32)
    target_link_libraries(${CORE_LIBRARY_NAME} pthread stdc++)
    if (APPLE)
        # OS X
    else()
        # Linux
        target_link_libraries(${CORE_LIBRARY_NAME} stdc++fs X11)
    endif()
endif ()
message("PYTHON_LIBRARIES" ${PYTHON_LIBRARIES})
target_link_libraries(${CORE_LIBRARY_NAME} ${PYTHON_LIBRARIES})

if (NOT APPLE)
    target_link_libraries(${CORE_LIBRARY_NAME} -static-libgcc -static-libstdc++)
endif()

foreach (source IN LISTS TAICHI_CORE_SOURCE)
    file(RELATIVE_PATH source_rel ${CMAKE_CURRENT_LIST_DIR} ${source})
    get_filename_component(source_path "${source_rel}" PATH)
    string(REPLACE "/" "\\" source_path_msvc "${source_path}")
    source_group("${source_path_msvc}" FILES "${source}")
endforeach ()

if (MSVC)
    set_property(TARGET ${CORE_LIBRARY_NAME} APPEND PROPERTY LINK_FLAGS /DEBUG)
endif ()

if (WIN32)
    set_target_properties(${CORE_LIBRARY_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY
            "${CMAKE_CURRENT_SOURCE_DIR}/runtimes")
endif ()

include_directories(include)
include_directories(external/include)

#add_custom_target(
#        clangformat
#        COMMAND clang-format-6.0
#        -style=file
#        -i
#        ${TAICHI_CORE_SOURCE} ${TAICHI_PROJECT_SOURCE}
#)
#
#add_custom_target(
#        yapfformat
#        COMMAND yapf
#        -irp
#        ${CMAKE_CURRENT_LIST_DIR}/../
#)
