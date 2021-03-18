option(USE_STDCPP "Use -stdlib=libc++" OFF)
option(TI_WITH_CUDA "Build with the CUDA backend" ON)
option(TI_WITH_OPENGL "Build with the OpenGL backend" ON)
option(TI_WITH_CC "Build with the C backend" ON)

if (APPLE)
    if (TI_WITH_CUDA)
        set(TI_WITH_CUDA OFF)
        message(WARNING "CUDA backend not supported on OS X. Setting TI_WITH_CUDA to OFF.")
    endif()
    if (TI_WITH_OPENGL)
        set(TI_WITH_OPENGL OFF)
        message(WARNING "OpenGL backend not supported on OS X. Setting TI_WITH_OPENGL to OFF.")
    endif()
    if (TI_WITH_CC)
        set(TI_WITH_CC OFF)
        message(WARNING "C backend not supported on OS X. Setting TI_WITH_CC to OFF.")
    endif()
endif()

if (WIN32)
    if (TI_WITH_CC)
        set(TI_WITH_CC OFF)
        message(WARNING "C backend not supported on Windows. Setting TI_WITH_CC to OFF.")
    endif()
endif()

if (NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/external/glad/src/glad.c")
    set(TI_WITH_OPENGL OFF)
    message(WARNING "external/glad submodule not detected. Settings TI_WITH_OPENGL to OFF.")
endif()

file(GLOB TAICHI_CORE_SOURCE
        "taichi/*/*/*/*.cpp" "taichi/*/*/*.cpp" "taichi/*/*.cpp" "taichi/*.cpp"
        "taichi/*/*/*/*.h" "taichi/*/*/*.h" "taichi/*/*.h" "taichi/*.h" "tests/cpp/*.cpp")

file(GLOB TAICHI_BACKEND_SOURCE "taichi/backends/**/*.cpp" "taichi/backends/**/*.h")

file(GLOB TAICHI_CPU_SOURCE "taichi/backends/cpu/*.cpp" "taichi/backends/cpu/*.h")
file(GLOB TAICHI_CUDA_SOURCE "taichi/backends/cuda/*.cpp" "taichi/backends/cuda/*.h")
file(GLOB TAICHI_METAL_SOURCE "taichi/backends/metal/*.h" "taichi/backends/metal/*.cpp" "taichi/backends/metal/shaders/*")
file(GLOB TAICHI_OPENGL_SOURCE "taichi/backends/opengl/*.h" "taichi/backends/opengl/*.cpp" "taichi/backends/opengl/shaders/*")
file(GLOB TAICHI_CC_SOURCE "taichi/backends/cc/*.h" "taichi/backends/cc/*.cpp")

list(REMOVE_ITEM TAICHI_CORE_SOURCE ${TAICHI_BACKEND_SOURCE})

list(APPEND TAICHI_CORE_SOURCE ${TAICHI_CPU_SOURCE})

if (TI_WITH_CUDA)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DTI_WITH_CUDA")
  list(APPEND TAICHI_CORE_SOURCE ${TAICHI_CUDA_SOURCE})
endif()

if(NOT CUDA_VERSION)
    set(CUDA_VERSION 10.0)
endif()

# TODO(#529) include Metal source only on Apple MacOS, and OpenGL only when TI_WITH_OPENGL is ON
list(APPEND TAICHI_CORE_SOURCE ${TAICHI_METAL_SOURCE})
list(APPEND TAICHI_CORE_SOURCE ${TAICHI_OPENGL_SOURCE})

if (TI_WITH_OPENGL)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DTI_WITH_OPENGL")
  # Q: Why not external/glad/src/*.c?
  # A: To ensure glad submodule exists when TI_WITH_OPENGL is ON.
  file(GLOB TAICHI_GLAD_SOURCE "external/glad/src/glad.c")
  list(APPEND TAICHI_CORE_SOURCE ${TAICHI_GLAD_SOURCE})
endif()

if (TI_WITH_CC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DTI_WITH_CC")
  list(APPEND TAICHI_CORE_SOURCE ${TAICHI_CC_SOURCE})
endif()

# This compiles all the libraries with -fPIC, which is critical to link a static
# library into a shared lib.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# The short-term goal is to have a sub-library, "taichi_isolated_core", that is
# mostly Taichi-focused, free from the "application" layer such as pybind11 or
# GUI. At a minimum, we must decouple from pybind11/python-environment. Then we
# can 1) unit test a major part of Taichi, and 2) integrate a new frontend lang
# with "taichi_isolated_core".
#
# TODO(#2198): Long-term speaking, we should create a separate library for each
# sub-module. This way we can guarantee that the lib dependencies form a DAG.
file(GLOB TAICHI_PYBIND_SOURCE
      "taichi/python/*.cpp"
      "taichi/python/*.h"
)
list(REMOVE_ITEM TAICHI_CORE_SOURCE ${TAICHI_PYBIND_SOURCE})

# TODO(#2196): Rename these CMAKE variables:
# CORE_LIBRARY_NAME --> TAICHI_ISOLATED_CORE_LIB_NAME
# CORE_WITH_PYBIND_LIBRARY_NAME --> TAICHI_CORE_LIB_NAME
#
# However, the better strategy is probably to rename the actual library:
#
# taichi_core --> taichi_pylib (this requires python-side refactoring...)
# taichi_isolated_core --> taichi_core
#
# But this requires more efforts, because taichi_core is already referenced
# everywhere in python.
set(CORE_LIBRARY_NAME taichi_isolated_core)
add_library(${CORE_LIBRARY_NAME} OBJECT ${TAICHI_CORE_SOURCE})

if (APPLE)
    # Ask OS X to minic Linux dynamic linking behavior
    target_link_libraries(${CORE_LIBRARY_NAME} "-undefined dynamic_lookup")
endif()

include_directories(${CMAKE_SOURCE_DIR})
include_directories(external/include)
include_directories(external/spdlog/include)
if (TI_WITH_OPENGL)
  include_directories(external/glad/include)
endif()

set(LIBRARY_NAME ${CORE_LIBRARY_NAME})

if (TI_WITH_OPENGL)
  set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
  set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)

  message("Building with GLFW")
  add_subdirectory(external/glfw)
  target_link_libraries(${LIBRARY_NAME} glfw)
endif()

# http://llvm.org/docs/CMake.html#embedding-llvm-in-your-project
find_package(LLVM REQUIRED CONFIG)
message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
if(${LLVM_PACKAGE_VERSION} VERSION_LESS "10.0")
    message(FATAL_ERROR "LLVM version < 10 is not supported")
endif()
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
include_directories(${LLVM_INCLUDE_DIRS})
message("LLVM include dirs ${LLVM_INCLUDE_DIRS}")
message("LLVM library dirs ${LLVM_LIBRARY_DIRS}")
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

if (APPLE AND "${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "arm64")
    llvm_map_components_to_libnames(llvm_aarch64_libs AArch64)
    target_link_libraries(${LIBRARY_NAME} ${llvm_aarch64_libs})
endif()

if (TI_WITH_CUDA)
    llvm_map_components_to_libnames(llvm_ptx_libs NVPTX)
    target_link_libraries(${LIBRARY_NAME} ${llvm_ptx_libs})
endif()

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
        target_link_libraries(${CORE_LIBRARY_NAME} -Wl,--version-script,${CMAKE_CURRENT_SOURCE_DIR}/misc/linker.map)
        target_link_libraries(${CORE_LIBRARY_NAME} -Wl,--wrap=log2f) # Avoid glibc dependencies
    endif()
else()
    # windows
    target_link_libraries(${CORE_LIBRARY_NAME} Winmm)
endif ()

foreach (source IN LISTS TAICHI_CORE_SOURCE)
    file(RELATIVE_PATH source_rel ${CMAKE_CURRENT_LIST_DIR} ${source})
    get_filename_component(source_path "${source_rel}" PATH)
    string(REPLACE "/" "\\" source_path_msvc "${source_path}")
    source_group("${source_path_msvc}" FILES "${source}")
endforeach ()

message("PYTHON_LIBRARIES: " ${PYTHON_LIBRARIES})

set(CORE_WITH_PYBIND_LIBRARY_NAME taichi_core)
add_library(${CORE_WITH_PYBIND_LIBRARY_NAME} SHARED ${TAICHI_PYBIND_SOURCE})
# It is actually possible to link with an OBJECT library
# https://cmake.org/cmake/help/v3.13/command/target_link_libraries.html?highlight=target_link_libraries#linking-object-libraries
target_link_libraries(${CORE_WITH_PYBIND_LIBRARY_NAME} PUBLIC ${CORE_LIBRARY_NAME})

# These commands should apply to the DLL that is loaded from python, not the OBJECT library.
if (MSVC)
    set_property(TARGET ${CORE_WITH_PYBIND_LIBRARY_NAME} APPEND PROPERTY LINK_FLAGS /DEBUG)
endif ()

if (WIN32)
    set_target_properties(${CORE_WITH_PYBIND_LIBRARY_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY
            "${CMAKE_CURRENT_SOURCE_DIR}/runtimes")
endif ()
