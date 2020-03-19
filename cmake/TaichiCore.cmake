set(CORE_LIBRARY_NAME taichi_core)

include(cmake/PythonNumpyPybind11.cmake)

file(GLOB TAICHI_CORE_SOURCE
        "taichi/*/*/*/*.cpp" "taichi/*/*/*.cpp" "taichi/*/*.cpp" "taichi/*.cpp"
        "taichi/*/*/*/*.h" "taichi/*/*/*.h" "taichi/*/*.h" "taichi/*.h" "external/xxhash/*.c" "tests/cpp/*.cpp")

file(GLOB TAICHI_BACKEND_SOURCE "taichi/backends/**/*.cpp" "taichi/backends/**/*.h")

file(GLOB TAICHI_CUDA_SOURCE "taichi/backends/cuda/*.cpp" "taichi/backends/cuda/*.h")

list(REMOVE_ITEM TAICHI_CORE_SOURCE ${TAICHI_BACKEND_SOURCE})

if (TI_WITH_CUDA)
    list(APPEND TAICHI_CORE_SOURCE ${TAICHI_CUDA_SOURCE})
endif()

option(BUILD_CPP_EXAMPLES "Build legacy C++ examples" OFF)

if (BUILD_CPP_EXAMPLES)
    file(GLOB_RECURSE CPP_EXAMPLES "examples/cpp/*.cpp")
else()
    set(CPP_EXAMPLES "")
endif()

add_library(${CORE_LIBRARY_NAME} SHARED ${TAICHI_CORE_SOURCE} ${PROJECT_SOURCES} ${CPP_EXAMPLES})

if (APPLE)
# Ask OS X to minic Linux dynamic linking behavior
target_link_libraries(${CORE_LIBRARY_NAME} "-undefined dynamic_lookup")
endif()


option(USE_STDCPP "Use -stdlib=libc++" OFF)
option(TI_WITH_CUDA "Build with the CUDA backend" OFF)
option(TI_WITH_OPENGL "Build with the OpenGL backend" ON)
option(GLEW_USE_STATIC_LIBS OFF)

include_directories(${CMAKE_SOURCE_DIR})
include_directories(external/xxhash)
include_directories(external/include)
include_directories(external/spdlog/include)

set(LIBRARY_NAME ${CORE_LIBRARY_NAME})

if (TI_WITH_CUDA)
    if(NOT CUDA_VERSION)
        set(CUDA_VERSION 10.0)
    endif()
    find_package(CUDA ${CUDA_VERSION})
    if (CUDA_FOUND)
        message("Building with CUDA ${CUDA_VERSION}")
        set(CUDA_ARCH 61)
        message("Found CUDA. Arch = ${CUDA_ARCH}")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DTI_WITH_CUDA")
        if (MSVC)
            include_directories(${CUDA_TOOLKIT_ROOT_DIR}/include)
            target_link_libraries(${LIBRARY_NAME} ${CUDA_TOOLKIT_ROOT_DIR}/lib/x64/cudart.lib ${CUDA_TOOLKIT_ROOT_DIR}/lib/x64/cuda.lib)
        else()
            include_directories(/usr/local/cuda-${CUDA_VERSION}/include)
            target_link_libraries(${LIBRARY_NAME} /usr/local/cuda-${CUDA_VERSION}/lib64/libcudart.so cuda)
        endif()
    else()
        message(FATAL_ERROR "CUDA not found.")
    endif()
endif()

if (TI_WITH_OPENGL)
  if(NOT GLEW_VERSION)
    set(GLEW_VERSION 2.0.0)
  endif()
  find_package(GLEW ${GLEW_VERSION})
  if (GLEW_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DTI_WITH_OPENGL")
    message("Building with GLEW ${GLEW_VERSION}")
    message("Using GLEW: ${GLEW_LIBRARIES}")
    target_include_directories(${LIBRARY_NAME} PUBLIC ${GLEW_INCLUDE_DIRS})
    target_link_libraries(${LIBRARY_NAME} ${GLEW_LIBRARIES} GLEW)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DGLEW_STATIC")
    find_package(glfw3 REQUIRED)
    if (NOT glfw3_FOUND)
      message(FATAL_ERROR "glfw3 not found.")
    endif()
    message("Building with glfw ${glfw3_VERSION}")
    target_link_libraries(${LIBRARY_NAME} glfw)
  else()
    message(WARNING "GLEW not found, ignoring TI_WITH_OPENGL.")
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
        Linker
        Target
        MC
        Passes
        ipo
        Analysis
        )
target_link_libraries(${LIBRARY_NAME} ${llvm_libs})

if (TI_WITH_CUDA)
    llvm_map_components_to_libnames(llvm_ptx_libs NVPTX)
    target_link_libraries(${LIBRARY_NAME} ${llvm_ptx_libs})
endif()

# add_executable(runtime runtime/runtime.cpp)

# Optional dependencies

if (APPLE)
    target_link_libraries(${CORE_LIBRARY_NAME} "-framework Cocoa -framework Metal")
endif ()

if (NOT WIN32)
    target_link_libraries(${CORE_LIBRARY_NAME} pthread stdc++)
    if (APPLE)
        # OS X
    else()
        # Linux
        target_link_libraries(${CORE_LIBRARY_NAME} stdc++fs X11)
        target_link_libraries(${CORE_LIBRARY_NAME} -static-libgcc -static-libstdc++)
    endif()
endif ()
message("PYTHON_LIBRARIES" ${PYTHON_LIBRARIES})

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
