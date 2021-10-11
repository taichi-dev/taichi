option(USE_STDCPP "Use -stdlib=libc++" OFF)
option(TI_WITH_CUDA "Build with the CUDA backend" ON)
option(TI_WITH_CUDA_TOOLKIT "Build with the CUDA toolkit" OFF)
option(TI_WITH_OPENGL "Build with the OpenGL backend" ON)
option(TI_WITH_CC "Build with the C backend" ON)
option(TI_WITH_VULKAN "Build with the Vulkan backend" OFF)

if(UNIX AND NOT APPLE)
    # Handy helper for Linux
    # https://stackoverflow.com/a/32259072/12003165
    set(LINUX TRUE)
endif()

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

set(TI_WITH_GGUI OFF)
if(TI_WITH_CUDA AND TI_WITH_VULKAN)
    set(TI_WITH_GGUI ON)
endif()


if (NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/external/glad/src/glad.c")
    set(TI_WITH_OPENGL OFF)
    message(WARNING "external/glad submodule not detected. Settings TI_WITH_OPENGL to OFF.")
endif()



file(GLOB TAICHI_CORE_SOURCE
        "taichi/*/*/*/*.cpp" "taichi/*/*/*.cpp" "taichi/*/*.cpp" "taichi/*.cpp"
        "taichi/*/*/*/*.h" "taichi/*/*/*.h" "taichi/*/*.h" "taichi/*.h" "tests/cpp/task/*.cpp")

file(GLOB TAICHI_BACKEND_SOURCE "taichi/backends/**/*.cpp" "taichi/backends/**/*.h")

file(GLOB TAICHI_CPU_SOURCE "taichi/backends/cpu/*.cpp" "taichi/backends/cpu/*.h")
file(GLOB TAICHI_WASM_SOURCE "taichi/backends/wasm/*.cpp" "taichi/backends/wasm/*.h")
file(GLOB TAICHI_CUDA_SOURCE "taichi/backends/cuda/*.cpp" "taichi/backends/cuda/*.h")
file(GLOB TAICHI_METAL_SOURCE "taichi/backends/metal/*.h" "taichi/backends/metal/*.cpp" "taichi/backends/metal/shaders/*")
file(GLOB TAICHI_OPENGL_SOURCE "taichi/backends/opengl/*.h" "taichi/backends/opengl/*.cpp" "taichi/backends/opengl/shaders/*")
file(GLOB TAICHI_CC_SOURCE "taichi/backends/cc/*.h" "taichi/backends/cc/*.cpp")
file(GLOB TAICHI_VULKAN_SOURCE "taichi/backends/vulkan/*.h" "taichi/backends/vulkan/*.cpp" "taichi/backends/vulkan/shaders/*" "external/SPIRV-Reflect/spirv_reflect.c")
file(GLOB TAICHI_INTEROP_SOURCE "taichi/backends/interop/*.cpp" "taichi/backends/interop/*.h")


file(GLOB TAICHI_GGUI_SOURCE
    "taichi/ui/*.cpp"  "taichi/ui/*/*.cpp" "taichi/ui/*/*/*.cpp"  "taichi/ui/*/*/*/*.cpp" "taichi/ui/*/*/*/*/*.cpp"
    "taichi/ui/*.h"  "taichi/ui/*/*.h" "taichi/ui/*/*/*.h"  "taichi/ui/*/*/*/*.h" "taichi/ui/*/*/*/*/*.h"
)
list(REMOVE_ITEM TAICHI_CORE_SOURCE ${TAICHI_GGUI_SOURCE})


if(TI_WITH_GGUI)
    add_definitions(-DTI_WITH_GGUI)

    list(APPEND TAICHI_CORE_SOURCE ${TAICHI_GGUI_SOURCE})

    include_directories(SYSTEM external/glm)

endif()

# These files are compiled into .bc and loaded as LLVM module dynamically. They should not be compiled into libtaichi. So they're removed here
file(GLOB BYTECODE_SOURCE "taichi/runtime/llvm/runtime.cpp")
list(REMOVE_ITEM TAICHI_CORE_SOURCE ${BYTECODE_SOURCE})


# These are required, regardless of whether Vulkan is enabled or not
# TODO(#2298): Clean up the Vulkan code structure, all Vulkan API related things should be
# guarded by TI_WITH_VULKAN macro at the source code level.
file(GLOB TAICHI_OPENGL_REQUIRED_SOURCE
  "taichi/backends/opengl/opengl_program.*"
  "taichi/backends/opengl/opengl_api.*"
  "taichi/backends/opengl/codegen_opengl.*"
  "taichi/backends/opengl/struct_opengl.*"
)
file(GLOB TAICHI_VULKAN_REQUIRED_SOURCE "taichi/backends/vulkan/runtime.h" "taichi/backends/vulkan/runtime.cpp")

list(REMOVE_ITEM TAICHI_CORE_SOURCE ${TAICHI_BACKEND_SOURCE})

list(APPEND TAICHI_CORE_SOURCE ${TAICHI_CPU_SOURCE})
list(APPEND TAICHI_CORE_SOURCE ${TAICHI_WASM_SOURCE})
list(APPEND TAICHI_CORE_SOURCE ${TAICHI_INTEROP_SOURCE})


if (TI_WITH_CUDA)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DTI_WITH_CUDA")
  list(APPEND TAICHI_CORE_SOURCE ${TAICHI_CUDA_SOURCE})
endif()

if(NOT CUDA_VERSION)
    set(CUDA_VERSION 10.0)
endif()

# TODO(#529) include Metal source only on Apple MacOS, and OpenGL only when TI_WITH_OPENGL is ON
list(APPEND TAICHI_CORE_SOURCE ${TAICHI_METAL_SOURCE})

if (TI_WITH_OPENGL)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DTI_WITH_OPENGL")
  # Q: Why not external/glad/src/*.c?
  # A: To ensure glad submodule exists when TI_WITH_OPENGL is ON.
  file(GLOB TAICHI_GLAD_SOURCE "external/glad/src/glad.c")
  list(APPEND TAICHI_CORE_SOURCE ${TAICHI_GLAD_SOURCE})
  list(APPEND TAICHI_CORE_SOURCE ${TAICHI_OPENGL_SOURCE})
endif()
list(APPEND TAICHI_CORE_SOURCE ${TAICHI_OPENGL_REQUIRED_SOURCE})

if (TI_WITH_CC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DTI_WITH_CC")
  list(APPEND TAICHI_CORE_SOURCE ${TAICHI_CC_SOURCE})
endif()


if (TI_WITH_VULKAN)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DTI_WITH_VULKAN")
    list(APPEND TAICHI_CORE_SOURCE ${TAICHI_VULKAN_SOURCE})
endif()
list(APPEND TAICHI_CORE_SOURCE ${TAICHI_VULKAN_REQUIRED_SOURCE})

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

if(DEFINED ENV{LLVM_DIR})
    set(LLVM_DIR $ENV{LLVM_DIR})
    message("Getting LLVM_DIR=${LLVM_DIR} from the environment variable")
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

if (TI_WITH_CUDA_TOOLKIT)
    if("$ENV{CUDA_TOOLKIT_ROOT_DIR}" STREQUAL "")
        message(FATAL_ERROR "TI_WITH_CUDA_TOOLKIT is ON but CUDA_TOOLKIT_ROOT_DIR not found")
    else()
        message(STATUS "TI_WITH_CUDA_TOOLKIT = ON")
        message(STATUS "CUDA_TOOLKIT_ROOT_DIR=$ENV{CUDA_TOOLKIT_ROOT_DIR}")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DTI_WITH_CUDA_TOOLKIT")
        include_directories($ENV{CUDA_TOOLKIT_ROOT_DIR}/include)
        link_directories($ENV{CUDA_TOOLKIT_ROOT_DIR}/lib64)
        #libraries for cuda kernel profiler CuptiToolkit
        target_link_libraries(${CORE_LIBRARY_NAME} cupti nvperf_host)
    endif()
else()
    message(STATUS "TI_WITH_CUDA_TOOLKIT = OFF")
endif()

if (TI_WITH_VULKAN)
    # Vulkan libs
    # https://cmake.org/cmake/help/latest/module/FindVulkan.html
    # https://github.com/PacktPublishing/Learning-Vulkan/blob/master/Chapter%2003/HandShake/CMakeLists.txt
    find_package(Vulkan REQUIRED)

    if(NOT Vulkan_FOUND)
        message(FATAL_ERROR "TI_WITH_VULKAN is ON but Vulkan could not be found")
    endif()

    message(STATUS "Vulkan_INCLUDE_DIR=${Vulkan_INCLUDE_DIR}")
    message(STATUS "Vulkan_LIBRARY=${Vulkan_LIBRARY}")

    include_directories(external/SPIRV-Headers/include)

    set(SPIRV_SKIP_EXECUTABLES true)
    set(SPIRV-Headers_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/external/SPIRV-Headers)
    add_subdirectory(external/SPIRV-Tools)
    # NOTE: SPIRV-Tools-opt must come before SPIRV-Tools
    # https://github.com/KhronosGroup/SPIRV-Tools/issues/1569#issuecomment-390250792
    target_link_libraries(${CORE_LIBRARY_NAME} SPIRV-Tools-opt ${SPIRV_TOOLS})

    # No longer link against vulkan, using volk instead
    #target_link_libraries(${CORE_LIBRARY_NAME} ${Vulkan_LIBRARY})
    include_directories(${Vulkan_INCLUDE_DIR})
    include_directories(external/volk)

    # Is this the best way to include the SPIRV-Headers?
    target_include_directories(${CORE_LIBRARY_NAME} PRIVATE external/SPIRV-Headers/include)
    target_include_directories(${CORE_LIBRARY_NAME} PRIVATE external/SPIRV-Reflect)
    target_include_directories(${CORE_LIBRARY_NAME} PRIVATE external/VulkanMemoryAllocator/include)

    if (LINUX)
        # shaderc requires pthread
        set(THREADS_PREFER_PTHREAD_FLAG ON)
        find_package(Threads REQUIRED)
        target_link_libraries(${CORE_LIBRARY_NAME} Threads::Threads)
    endif()
endif ()

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
        if (NOT TI_EXPORT_CORE) # expose api for CHI IR Builder
            target_link_libraries(${CORE_LIBRARY_NAME} -Wl,--version-script,${CMAKE_CURRENT_SOURCE_DIR}/misc/linker.map)
        endif ()
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


if(TI_WITH_GGUI)

    # Dear ImGui
    add_definitions(-DIMGUI_IMPL_VULKAN_NO_PROTOTYPES)
    set(IMGUI_DIR external/imgui)
    include_directories(external/glfw/include)
    include_directories(SYSTEM ${IMGUI_DIR} ${IMGUI_DIR}/backends ..)
    add_library(imgui  ${IMGUI_DIR}/backends/imgui_impl_glfw.cpp ${IMGUI_DIR}/backends/imgui_impl_vulkan.cpp ${IMGUI_DIR}/imgui.cpp ${IMGUI_DIR}/imgui_draw.cpp  ${IMGUI_DIR}/imgui_tables.cpp ${IMGUI_DIR}/imgui_widgets.cpp)
    target_link_libraries(${CORE_LIBRARY_NAME} imgui)

endif()
