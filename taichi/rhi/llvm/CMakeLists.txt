# ./taichi/rhi/llvm/CMakeLists.txt

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

if (APPLE AND "${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "arm64")
llvm_map_components_to_libnames(llvm_aarch64_libs AArch64)
endif()



set(LLVM_RHI llvm_rhi)
add_library(${LLVM_RHI})
target_sources(${LLVM_RHI}
  PRIVATE
    jit_session.cpp
    llvm_device.cpp
  )

target_include_directories(${LLVM_RHI}
  PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${LLVM_INCLUDE_DIRS}
  )

target_link_libraries(${LLVM_RHI} PRIVATE taichi_util)
