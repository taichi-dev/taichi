option(USE_STDCPP "Use -stdlib=libc++" OFF)
option(TI_WITH_LLVM "Build with LLVM backends" ON)
option(TI_WITH_METAL "Build with the Metal backend" ON)
option(TI_WITH_CUDA "Build with the CUDA backend" ON)
option(TI_WITH_CUDA_TOOLKIT "Build with the CUDA toolkit" OFF)
option(TI_WITH_OPENGL "Build with the OpenGL backend" ON)
option(TI_WITH_CC "Build with the C backend" ON)
option(TI_WITH_VULKAN "Build with the Vulkan backend" OFF)
option(TI_WITH_DX11 "Build with the DX11 backend" OFF)
option(TI_EMSCRIPTENED "Build using emscripten" OFF)

# Force symbols to be 'hidden' by default so nothing is exported from the Taichi
# library including the third-party dependencies.
# As Taichi can be used by external projects, some of the internal dependencies
# such as Vulkan, ImGui, etc. could be in conflict with the dependencies of those
# projects.
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)
# Suppress warnings from submodules introduced by the above symbol visibility change
set(CMAKE_POLICY_DEFAULT_CMP0063 NEW)
set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)

if(ANDROID)
    set(TI_WITH_VULKAN ON)
    set(TI_EXPORT_CORE ON)
    set(TI_WITH_LLVM OFF)
    set(TI_WITH_METAL OFF)
    set(TI_WITH_CUDA OFF)
    set(TI_WITH_OPENGL OFF)
    set(TI_WITH_CC OFF)
    set(TI_WITH_DX11 OFF)
endif()

if(TI_EMSCRIPTENED)
    set(TI_WITH_LLVM OFF)
    set(TI_WITH_METAL OFF)
    set(TI_WITH_CUDA OFF)
    set(TI_WITH_OPENGL OFF)
    set(TI_WITH_CC OFF)
    set(TI_WITH_DX11 OFF)

    set(TI_WITH_VULKAN ON)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DTI_EMSCRIPTENED")
endif()

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
if(TI_WITH_VULKAN AND NOT TI_EMSCRIPTENED)
    set(TI_WITH_GGUI ON)
endif()


if (NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/external/glad/src/gl.c")
    set(TI_WITH_OPENGL OFF)
    message(WARNING "external/glad submodule not detected. Settings TI_WITH_OPENGL to OFF.")
endif()

if(NOT TI_WITH_LLVM)
    set(TI_WITH_CUDA OFF)
    set(TI_WITH_CUDA_TOOLKIT OFF)
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
file(GLOB TAICHI_DX11_SOURCE "taichi/backends/dx/*.h" "taichi/backends/dx/*.cpp")
file(GLOB TAICHI_CC_SOURCE "taichi/backends/cc/*.h" "taichi/backends/cc/*.cpp")
file(GLOB TAICHI_VULKAN_SOURCE "taichi/backends/vulkan/*.h" "taichi/backends/vulkan/*.cpp" "external/SPIRV-Reflect/spirv_reflect.c")
file(GLOB TAICHI_INTEROP_SOURCE "taichi/backends/interop/*.cpp" "taichi/backends/interop/*.h")


file(GLOB TAICHI_GGUI_SOURCE
    "taichi/ui/*.cpp"  "taichi/ui/*/*.cpp" "taichi/ui/*/*/*.cpp"  "taichi/ui/*/*/*/*.cpp" "taichi/ui/*/*/*/*/*.cpp"
    "taichi/ui/*.h"  "taichi/ui/*/*.h" "taichi/ui/*/*/*.h"  "taichi/ui/*/*/*/*.h" "taichi/ui/*/*/*/*/*.h"
)
file(GLOB TAICHI_GGUI_GLFW_SOURCE
  "taichi/ui/common/window_base.cpp"
  "taichi/ui/backends/vulkan/window.cpp"
)
list(REMOVE_ITEM TAICHI_CORE_SOURCE ${TAICHI_GGUI_SOURCE})


if(TI_WITH_GGUI)
    add_definitions(-DTI_WITH_GGUI)

    # Remove GLFW dependencies from the build for Android
    if(ANDROID)
        list(REMOVE_ITEM TAICHI_GGUI_SOURCE ${TAICHI_GGUI_GLFW_SOURCE})
    endif()

    list(APPEND TAICHI_CORE_SOURCE ${TAICHI_GGUI_SOURCE})
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
file(GLOB TAICHI_VULKAN_REQUIRED_SOURCE
  "taichi/backends/vulkan/runtime.h"
  "taichi/backends/vulkan/runtime.cpp"
)

list(REMOVE_ITEM TAICHI_CORE_SOURCE ${TAICHI_BACKEND_SOURCE})

if(TI_WITH_LLVM)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DTI_WITH_LLVM")
    list(APPEND TAICHI_CORE_SOURCE ${TAICHI_CPU_SOURCE})
    list(APPEND TAICHI_CORE_SOURCE ${TAICHI_WASM_SOURCE})
else()
    file(GLOB TAICHI_LLVM_SOURCE "taichi/llvm/*.cpp" "taichi/llvm/*.h")
    list(REMOVE_ITEM TAICHI_CORE_SOURCE ${TAICHI_LLVM_SOURCE})
endif()

list(APPEND TAICHI_CORE_SOURCE ${TAICHI_INTEROP_SOURCE})


if (TI_WITH_CUDA)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DTI_WITH_CUDA")
  list(APPEND TAICHI_CORE_SOURCE ${TAICHI_CUDA_SOURCE})
endif()

if(NOT CUDA_VERSION)
    set(CUDA_VERSION 10.0)
endif()


# By default, TI_WITH_METAL is ON for all platforms.
# As of right now, on non-macOS platforms, the metal backend won't work at all.
# We have future plans to allow metal AOT to run on non-macOS devices.
if (TI_WITH_METAL)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DTI_WITH_METAL")
    list(APPEND TAICHI_CORE_SOURCE ${TAICHI_METAL_SOURCE})
endif()


if (TI_WITH_OPENGL)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DTI_WITH_OPENGL")
  # Q: Why not external/glad/src/*.c?
  # A: To ensure glad submodule exists when TI_WITH_OPENGL is ON.
  file(GLOB TAICHI_GLAD_SOURCE "external/glad/src/gl.c" "external/glad/src/egl.c")
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

if (TI_WITH_DX11)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DTI_WITH_DX11")
    list(APPEND TAICHI_CORE_SOURCE ${TAICHI_DX11_SOURCE})
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

file(GLOB TAICHI_EMBIND_SOURCE
      "taichi/javascript/*.cpp"
      "taichi/javascript/*.h"
)
if (TAICHI_EMBIND_SOURCE)
  list(REMOVE_ITEM TAICHI_CORE_SOURCE ${TAICHI_EMBIND_SOURCE})
endif()

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
include_directories(external/PicoSHA2)
if (TI_WITH_OPENGL)
    target_include_directories(${CORE_LIBRARY_NAME} PRIVATE external/glad/include)
endif()
    target_include_directories(${CORE_LIBRARY_NAME} PRIVATE external/FP16/include)

set(LIBRARY_NAME ${CORE_LIBRARY_NAME})

# GLFW not available on Android
if (TI_WITH_OPENGL OR TI_WITH_VULKAN AND NOT ANDROID AND NOT TI_EMSCRIPTENED)
  set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
  set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)

  if (APPLE)
    set(GLFW_VULKAN_STATIC ON CACHE BOOL "" FORCE)
  endif()

  message("Building with GLFW")
  add_subdirectory(external/glfw)
  target_link_libraries(${LIBRARY_NAME} glfw)
  target_include_directories(${CORE_LIBRARY_NAME} PRIVATE external/glfw/include)
endif()

if(DEFINED ENV{LLVM_DIR})
    set(LLVM_DIR $ENV{LLVM_DIR})
    message("Getting LLVM_DIR=${LLVM_DIR} from the environment variable")
endif()

if(TI_WITH_LLVM)
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

if (TI_WITH_OPENGL)
    set(SPIRV_CROSS_CLI false)
    add_subdirectory(external/SPIRV-Cross)
    target_include_directories(${CORE_LIBRARY_NAME} PRIVATE external/SPIRV-Cross)
    target_link_libraries(${CORE_LIBRARY_NAME} spirv-cross-glsl spirv-cross-core)
endif()

if (TI_WITH_DX11)
    set(SPIRV_CROSS_CLI false)
    #target_include_directories(${CORE_LIBRARY_NAME} PRIVATE external/SPIRV-Cross)
    target_link_libraries(${CORE_LIBRARY_NAME} spirv-cross-hlsl spirv-cross-core)
endif()

# SPIR-V codegen is always there, regardless of Vulkan
set(SPIRV_SKIP_EXECUTABLES true)
set(SPIRV-Headers_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/external/SPIRV-Headers)
add_subdirectory(external/SPIRV-Tools)
# NOTE: SPIRV-Tools-opt must come before SPIRV-Tools
# https://github.com/KhronosGroup/SPIRV-Tools/issues/1569#issuecomment-390250792
target_link_libraries(${CORE_LIBRARY_NAME} SPIRV-Tools-opt ${SPIRV_TOOLS})

if (TI_WITH_VULKAN)
    include_directories(SYSTEM external/Vulkan-Headers/include)

    include_directories(SYSTEM external/volk)

    target_include_directories(${CORE_LIBRARY_NAME} PRIVATE external/SPIRV-Headers/include)
    target_include_directories(${CORE_LIBRARY_NAME} PRIVATE external/SPIRV-Reflect)
    target_include_directories(${CORE_LIBRARY_NAME} PRIVATE external/VulkanMemoryAllocator/include)

    if (LINUX)
        # shaderc requires pthread
        set(THREADS_PREFER_PTHREAD_FLAG ON)
        find_package(Threads REQUIRED)
        target_link_libraries(${CORE_LIBRARY_NAME} Threads::Threads)
    endif()

    if (APPLE)
        find_library(MOLTEN_VK libMoltenVK.dylib PATHS $HOMEBREW_CELLAR/molten-vk $VULKAN_SDK REQUIRED)
        configure_file(${MOLTEN_VK} ${CMAKE_BINARY_DIR}/libMoltenVK.dylib COPYONLY)
        message(STATUS "MoltenVK library ${MOLTEN_VK}")
    endif()
endif ()

# Optional dependencies

if (APPLE)
    target_link_libraries(${CORE_LIBRARY_NAME} "-framework Cocoa -framework Metal")
endif ()

if (NOT WIN32)
    # Android has a custom toolchain so pthread is not available and should
    # link against other libraries as well for logcat and internal features.
    if (ANDROID)
        target_link_libraries(${CORE_LIBRARY_NAME} android log)
    else()
        target_link_libraries(${CORE_LIBRARY_NAME} pthread stdc++)
    endif()

    if (UNIX AND NOT ${CMAKE_SYSTEM_NAME} MATCHES "Linux")
	# OS X or BSD
    else()
        # Linux
        target_link_libraries(${CORE_LIBRARY_NAME} stdc++fs X11)
        target_link_libraries(${CORE_LIBRARY_NAME} -static-libgcc -static-libstdc++)
        if (${CMAKE_HOST_SYSTEM_PROCESSOR} STREQUAL "x86_64")
            # Avoid glibc dependencies
            if (TI_WITH_VULKAN)
                target_link_libraries(${CORE_LIBRARY_NAME} -Wl,--wrap=log2f)
            else()
                # Enforce compatibility with manylinux2014
                target_link_libraries(${CORE_LIBRARY_NAME} -Wl,--wrap=log2f -Wl,--wrap=exp2 -Wl,--wrap=log2 -Wl,--wrap=logf -Wl,--wrap=powf -Wl,--wrap=exp -Wl,--wrap=log -Wl,--wrap=pow)
            endif()
        endif()
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

if(NOT TI_EMSCRIPTENED)
    set(CORE_WITH_PYBIND_LIBRARY_NAME taichi_core)
    # Cannot compile Python source code with Android, but TI_EXPORT_CORE should be set and
    # Android should only use the isolated library ignoring those source code.
    if (NOT ANDROID)
        add_library(${CORE_WITH_PYBIND_LIBRARY_NAME} SHARED ${TAICHI_PYBIND_SOURCE})
    else()
        add_library(${CORE_WITH_PYBIND_LIBRARY_NAME} SHARED)
    endif ()

    # Remove symbols from static libs: https://stackoverflow.com/a/14863432/12003165
    if (LINUX)
        target_link_options(${CORE_WITH_PYBIND_LIBRARY_NAME} PUBLIC -Wl,--exclude-libs=ALL)
    endif()
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
endif()

if(TI_EMSCRIPTENED)
    set(CORE_WITH_EMBIND_LIBRARY_NAME taichi)
    add_executable(${CORE_WITH_EMBIND_LIBRARY_NAME} ${TAICHI_EMBIND_SOURCE})
    target_link_libraries(${CORE_WITH_EMBIND_LIBRARY_NAME} PUBLIC ${CORE_LIBRARY_NAME})
    target_compile_options(${CORE_WITH_EMBIND_LIBRARY_NAME} PRIVATE "-Oz")
    # target_compile_options(${CORE_LIBRARY_NAME} PRIVATE "-Oz")
    set_target_properties(${CORE_LIBRARY_NAME} PROPERTIES LINK_FLAGS "-s ERROR_ON_UNDEFINED_SYMBOLS=0 -s ASSERTIONS=1")
    set_target_properties(${CORE_WITH_EMBIND_LIBRARY_NAME} PROPERTIES LINK_FLAGS "--bind -s MODULARIZE=1 -s EXPORT_NAME=createTaichiModule -s WASM=0  --memory-init-file 0 -Oz --closure 1 -s ERROR_ON_UNDEFINED_SYMBOLS=0 -s ASSERTIONS=1 -s NO_DISABLE_EXCEPTION_CATCHING")
endif()

if(TI_WITH_GGUI)
    include_directories(SYSTEM PRIVATE external/glm)

    # Dear ImGui
    add_definitions(-DIMGUI_IMPL_VULKAN_NO_PROTOTYPES)
    set(IMGUI_DIR external/imgui)
    include_directories(SYSTEM ${IMGUI_DIR} ${IMGUI_DIR}/backends ..)
if(ANDROID)
    add_library(imgui  ${IMGUI_DIR}/backends/imgui_impl_android.cpp ${IMGUI_DIR}/backends/imgui_impl_vulkan.cpp ${IMGUI_DIR}/imgui.cpp ${IMGUI_DIR}/imgui_draw.cpp  ${IMGUI_DIR}/imgui_tables.cpp ${IMGUI_DIR}/imgui_widgets.cpp)
else()
    include_directories(external/glfw/include)
    add_library(imgui  ${IMGUI_DIR}/backends/imgui_impl_glfw.cpp ${IMGUI_DIR}/backends/imgui_impl_vulkan.cpp ${IMGUI_DIR}/imgui.cpp ${IMGUI_DIR}/imgui_draw.cpp  ${IMGUI_DIR}/imgui_tables.cpp ${IMGUI_DIR}/imgui_widgets.cpp)
endif()
    target_link_libraries(${CORE_LIBRARY_NAME} imgui)

endif()
