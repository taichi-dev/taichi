set(CORE_LIBRARY_NAME taichi_core)

option(USE_STDCPP "Use -stdlib=libc++" OFF)
option(TI_WITH_CUDA "Build with the CUDA backend" ON)
option(TI_WITH_OPENGL "Build with the OpenGL backend" ON)

if (APPLE)
    if (TI_WITH_CUDA)
        set(TI_WITH_CUDA OFF)
        message(WARNING "CUDA not supported on OS X. Setting TI_WITH_CUDA to OFF.")
    endif()
    if (TI_WITH_OPENGL)
        set(TI_WITH_OPENGL OFF)
        message(WARNING "OpenGL not supported on OS X. Setting TI_WITH_OPENGL to OFF.")
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

add_library(${CORE_LIBRARY_NAME} SHARED ${TAICHI_CORE_SOURCE} ${PROJECT_SOURCES})

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
if(${LLVM_PACKAGE_VERSION} VERSION_LESS "8.0")
    message(FATAL_ERROR "LLVM less than 8.0 is not supported")
endif()
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
