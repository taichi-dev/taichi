set(CORE_LIBRARY_NAME taichi_core)

include(cmake/PythonNumpyPybind11.cmake)

file(GLOB TAICHI_CORE_SOURCE
        "examples/cpp/*.cpp"
        "taichi/*/*/*/*.cpp" "taichi/*/*/*.cpp" "taichi/*/*.cpp" "taichi/*.cpp"
        "taichi/*/*/*/*.h" "taichi/*/*/*.h" "taichi/*/*.h" "taichi/*.h")

file(GLOB_RECURSE PROJECT_SOURCES "lang/headers/*.h" "external/xxhash/*.c" "tests/cpp/*.cpp" "lang/cpp_examples/*.cpp")

add_library(${CORE_LIBRARY_NAME} SHARED ${TAICHI_CORE_SOURCE} ${PROJECT_SOURCES} ${SPGridSource})

option(USE_STDCPP "Use -stdlib=libc++" OFF)
option(TLANG_WITH_CUDA "Build with GPU support" ON)

include_directories(${CMAKE_SOURCE_DIR})
include_directories(external/xxhash)
include_directories(external/include)

set(LIBRARY_NAME ${CORE_LIBRARY_NAME})

if (TLANG_WITH_CUDA)
    if(NOT CUDA_VERSION)
        set(CUDA_VERSION 10.0)
    endif()
    find_package(CUDA ${CUDA_VERSION})
    if (CUDA_FOUND)
        message("Building with CUDA ${CUDA_VERSION}")
        set(CUDA_ARCH 61)
        message("Found CUDA. Arch = ${CUDA_ARCH}")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DCUDA_FOUND -DTLANG_WITH_CUDA -D TLANG_CUDA_VERSION='\"${CUDA_VERSION}\"'")
        include_directories(/usr/local/cuda-${CUDA_VERSION}/include)
        target_link_libraries(${LIBRARY_NAME} /usr/local/cuda-${CUDA_VERSION}/lib64/libcudart.so cuda)
    else()
        message("CUDA not found.")
    endif()
endif()

# http://llvm.org/docs/CMake.html#embedding-llvm-in-your-project
find_package(LLVM REQUIRED CONFIG 8.0)
message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
include_directories(${LLVM_INCLUDE_DIRS})
    message("llvm include dirs ${LLVM_INCLUDE_DIRS}")
add_definitions(${LLVM_DEFINITIONS})

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
        Target
        MC
        Passes
        ipo
        Analysis
        )
target_link_libraries(${LIBRARY_NAME} ${llvm_libs})

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
