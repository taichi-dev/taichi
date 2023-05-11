#*********************************************************************
# The Taichi Programming Language
#*********************************************************************

cmake_minimum_required(VERSION 3.17)

project(taichi)

include("cmake/utils.cmake")

if (NOT DEFINED TI_VERSION_MAJOR)
    message(WARNING "It seems that you are running cmake manually, which may cause issues. Please use setup.py to build taichi from source, see https://docs.taichi-lang.org/docs/dev_install for more details.")
    file(READ "${CMAKE_CURRENT_LIST_DIR}/version.txt" TI_VERSION_LITERAL)
    string(REGEX MATCH "v([0-9]+)\\.([0-9]+)\\.([0-9]+)" TI_VERSION_LITERAL ${TI_VERSION_LITERAL})
    set(TI_VERSION_MAJOR ${CMAKE_MATCH_1})
    set(TI_VERSION_MINOR ${CMAKE_MATCH_2})
    set(TI_VERSION_PATCH ${CMAKE_MATCH_3})
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_EXPORT_COMPILECOMMANDS ON)

execute_process(
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  COMMAND git rev-parse --short HEAD
  RESULT_VARIABLE SHORT_HASH_RESULT
  OUTPUT_VARIABLE TI_COMMIT_SHORT_HASH)
execute_process(
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  COMMAND git rev-parse HEAD
  RESULT_VARIABLE SHORT_HASH_RESULT
  OUTPUT_VARIABLE TI_COMMIT_HASH)
string(STRIP ${TI_COMMIT_HASH} TI_COMMIT_HASH)
string(STRIP ${TI_COMMIT_SHORT_HASH} TI_COMMIT_SHORT_HASH)

message("Taichi Version ${TI_VERSION_MAJOR}.${TI_VERSION_MINOR}.${TI_VERSION_PATCH}")
message("       Commit ${TI_COMMIT_HASH}")

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING
            "Choose the type of build, options are: Debug Release
RelWithDebInfo MinSizeRel."
            FORCE)
endif(NOT CMAKE_BUILD_TYPE)

set(TAICHI_CMAKE_DIR "${CMAKE_CURRENT_LIST_DIR}/cmake/")

if (WIN32)
    list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake/")
    list(APPEND CMAKE_MODULE_PATH "${CMAKE_ROOT}/Modules")
else ()
    list(APPEND CMAKE_MODULE_PATH "${CMAKE_ROOT}/Modules")
    list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake/")
endif ()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/build")

find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
    set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
endif()

# This compiles all the libraries with -fPIC, which is critical to link a static
# library into a shared lib.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)


option(USE_LLD "Use lld (from llvm) linker" OFF)
option(USE_MOLD "Use mold (A Modern Linker)" OFF)
option(TI_WITH_BACKTRACE "Use backward-cpp to print out C++ stack trace upon failure" OFF)  # wheel-tag: bt
option(TI_GENERATE_PDB "Generate Program Database (PDB) files (will make compilation uncacheable)" OFF)

if(LINUX OR APPLE)
    if (NOT IOS)
        # (penguinliong) Not compatible with -fembed-bitcode. Not sure where it
        # comes from; probably a XCode preset?
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ffunction-sections -fdata-sections")
    endif()
endif()

if (USE_LLD)
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -fuse-ld=lld")
    set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} -fuse-ld=lld")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fuse-ld=lld")

    if (WIN32)
        if (CMAKE_BUILD_TYPE EQUAL "RelWithDebInfo")
            # -debug for LLD generates PDB files
            set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,-debug")
            set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} -Wl,-debug")
            set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-debug")
        endif()
    endif()
endif()

if (USE_MOLD)
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -fuse-ld=mold")
    set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} -fuse-ld=mold")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fuse-ld=mold")
endif()

if (WIN32)
  # For `Debug` configs MSVC links to a debuggable runtime by default which has
  # symbol conflicts with the prebuilt LLVM in `Release`. We should be providing
  # prebuilt LLVMs for both `Debug` and `Release` but LLVM 10 cannot be built by
  # MSVC in `Debug` config because MSVC would try to fill uninitialize memory
  # with `0xCC` but it too breaks `LLVMTableGen` which is depended on by almost
  # every component in LLVM.
  message("CMAKE_MSVC_RUNTIME_LIBRARY: ${CMAKE_MSVC_RUNTIME_LIBRARY}")
endif()

# No support of Python for Android build; or in any case taichi is integrated
# in another project as submodule.
option(TI_WITH_PYTHON "Build with Python language binding" ON)
if (TI_WITH_PYTHON AND NOT ANDROID)
    include(cmake/PythonNumpyPybind11.cmake)
endif()

if (TI_WITH_BACKTRACE)
  add_subdirectory(external/backward_cpp)
endif()

if (TI_DISTRIBUTED_COMPILE)
    include(cmake/Distributed.cmake)
endif()

include(cmake/TaichiCXXFlags.cmake)
include(cmake/TaichiCore.cmake)

option(TI_BUILD_TESTS "Build the CPP tests" OFF)

if (TI_BUILD_TESTS)
  add_subdirectory(external/googletest EXCLUDE_FROM_ALL)
  include(cmake/TaichiTests.cmake)
endif()

option(TI_BUILD_EXAMPLES "Build the CPP examples" ON)
option(TI_BUILD_RHI_EXAMPLES "Build the Unified Device API examples" OFF)

if(NOT TI_WITH_LLVM OR NOT TI_WITH_METAL)
    set(TI_BUILD_EXAMPLES OFF)
endif()

message("C++ Flags: ${CMAKE_CXX_FLAGS}")
message("Build type: ${CMAKE_BUILD_TYPE}")

if (NOT TI_WITH_CUDA)
    set(CUDA_VERSION "0.0")
    set(CUDA_TOOLKIT_ROOT_DIR "")
endif()

if (TI_WITH_CUDA)
    set(CUDA_ARCH "cuda")
endif()

if (TI_WITH_AMDGPU)
    set(AMDGPU_ARCH "amdgpu")
endif()

if (TI_WITH_DX12)
    set(DX12_ARCH "dx12")
endif()

if (TI_WITH_LLVM)
    # Setup CLANG_EXECUTABLE
    if (CLANG_EXECUTABLE)
      message("CLANG_EXECUTABLE defined: ${CLANG_EXECUTABLE}")
    elseif ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
      set (CLANG_EXECUTABLE ${CMAKE_CXX_COMPILER})
      message("Clang executable using host compiler ${CLANG_EXECUTABLE}")
    else()
      find_program(CLANG_EXECUTABLE NAMES clang-15 clang-14 clang-13 clang-12 clang-11 clang-10 clang-9 clang-8 clang-7 clang)
      message("Clang executable found at ${CLANG_EXECUTABLE}")
    endif()

    if (NOT CLANG_EXECUTABLE)
      message(FATAL_ERROR "Cannot find any clang executable.")
    endif()

    macro(check_clang_version)
      execute_process(COMMAND ${CLANG_EXECUTABLE} --version OUTPUT_VARIABLE CLANG_VERSION_OUTPUT)
      string(REGEX MATCH "([0-9]+)\\.[0-9]+(\\.[0-9]+)?" CLANG_VERSION "${CLANG_VERSION_OUTPUT}")

      message("${CLANG_EXECUTABLE} --version: ${CLANG_VERSION}")

      set(CLANG_VERSION_MAJOR "${CMAKE_MATCH_1}")
    endmacro()

    if (APPLE)
      set(CLANG_OSX_FLAGS "-isysroot${CMAKE_OSX_SYSROOT}")
    endif()

    # Highest clang version that we've tested
    set(CLANG_HIGHEST_VERSION "15")

    check_clang_version()

    if (${CLANG_VERSION_MAJOR} VERSION_GREATER ${CLANG_HIGHEST_VERSION})
      message(FATAL_ERROR "${CLANG_EXECUTABLE} version: ${CLANG_VERSION}, required: <=${CLANG_HIGHEST_VERSION}. Consider passing -DCLANG_EXECUTABLE=/path/to/clang to cmake to use a specific clang.")
    endif()

    add_subdirectory(taichi/runtime/llvm/runtime_module)
endif()

configure_file(taichi/common/version.h.in ${CMAKE_SOURCE_DIR}/taichi/common/version.h)
configure_file(taichi/common/commit_hash.h.in ${CMAKE_SOURCE_DIR}/taichi/common/commit_hash.h)

option(TI_WITH_C_API "build taichi runtime c-api library" ON)  # wheel-tag: aot
option(TI_WITH_STATIC_C_API "build static taichi runtime c-api library" OFF)  # wheel-tag: static_aot

if(TI_WITH_STATIC_C_API)
    set(TI_WITH_C_API ${TI_WITH_STATIC_C_API})
    if(NOT APPLE)
        message(FATAL_ERROR "TI_WITH_STATIC_C_API requires Apple compilation sdk, thus only supported on MacOS")
    endif()
endif()

if (TI_WITH_C_API)
  include(cmake/TaichiCAPI.cmake)
  if (TI_BUILD_TESTS)
    include(cmake/TaichiCAPITests.cmake)
  endif()
endif()

if (TI_BUILD_EXAMPLES)
  include(cmake/TaichiExamples.cmake)
endif()

if (TI_BUILD_RHI_EXAMPLES)
  add_subdirectory(cpp_examples/rhi_examples)
endif()


option(TI_WITH_GRAPHVIZ "generate dependency graphs between targets" OFF)  # wheel-tag: viz
if (TI_WITH_GRAPHVIZ)
  set(GRAPHVIZ_GRAPH_NAME "ti_targets")
  add_custom_target(graphviz ALL
      COMMAND cp ${CMAKE_SOURCE_DIR}/cmake/CMakeGraphVizOptions.cmake .
      COMMAND ${CMAKE_COMMAND} "--graphviz=${GRAPHVIZ_GRAPH_NAME}.dot" .
      COMMAND dot -Tpng ${GRAPHVIZ_GRAPH_NAME}.dot -o ${GRAPHVIZ_GRAPH_NAME}.png
      COMMAND cp ti_targets.png ${CMAKE_SOURCE_DIR}/build
      WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
  )
endif()
