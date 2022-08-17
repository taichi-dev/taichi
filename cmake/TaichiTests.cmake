cmake_minimum_required(VERSION 3.0)

set(TESTS_NAME taichi_cpp_tests)
if (WIN32)
    # Prevent overriding the parent project's compiler/linker
    # settings on Windows
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
endif()

# TODO(#2195):
# 1. "cpp" -> "cpp_legacy", "cpp_new" -> "cpp"
# 2. Re-implement the legacy CPP tests using googletest
file(GLOB_RECURSE TAICHI_TESTS_SOURCE
        "tests/cpp/analysis/*.cpp"
        "tests/cpp/aot/llvm/*.cpp"
        "tests/cpp/backends/*.cpp"
        "tests/cpp/codegen/*.cpp"
        "tests/cpp/common/*.cpp"
        "tests/cpp/ir/*.cpp"
        "tests/cpp/llvm/*.cpp"
        "tests/cpp/program/*.cpp"
        "tests/cpp/struct/*.cpp"
        "tests/cpp/transforms/*.cpp")

if (TI_WITH_OPENGL OR TI_WITH_VULKAN)
    file(GLOB TAICHI_TESTS_GFX_UTILS_SOURCE
        "tests/cpp/aot/gfx_utils.cpp")
    list(APPEND TAICHI_TESTS_SOURCE ${TAICHI_TESTS_GFX_UTILS_SOURCE})
endif()

if(TI_WITH_VULKAN)
  file(GLOB TAICHI_TESTS_VULKAN_SOURCE "tests/cpp/aot/vulkan/*.cpp")
  list(APPEND TAICHI_TESTS_SOURCE ${TAICHI_TESTS_VULKAN_SOURCE})
endif()

if(TI_WITH_OPENGL)
  file(GLOB TAICHI_TESTS_OPENGL_SOURCE "tests/cpp/aot/opengl/*.cpp")
  list(APPEND TAICHI_TESTS_SOURCE ${TAICHI_TESTS_OPENGL_SOURCE})
endif()

add_executable(${TESTS_NAME} ${TAICHI_TESTS_SOURCE})
if (WIN32)
    # Output the executable to bin/ instead of build/Debug/...
    set(TESTS_OUTPUT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/bin")
    set_target_properties(${TESTS_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${TESTS_OUTPUT_DIR})
    set_target_properties(${TESTS_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY_DEBUG ${TESTS_OUTPUT_DIR})
    set_target_properties(${TESTS_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY_RELEASE ${TESTS_OUTPUT_DIR})
    set_target_properties(${TESTS_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL ${TESTS_OUTPUT_DIR})
    set_target_properties(${TESTS_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO ${TESTS_OUTPUT_DIR})
endif()
target_link_libraries(${TESTS_NAME} PRIVATE taichi_core)
target_link_libraries(${TESTS_NAME} PRIVATE gtest_main)

if (TI_WITH_OPENGL OR TI_WITH_VULKAN)
  target_link_libraries(${TESTS_NAME} PRIVATE gfx_runtime)
endif()

if (TI_WITH_VULKAN)
  target_link_libraries(${TESTS_NAME} PRIVATE vulkan_rhi)
endif()

if (TI_WITH_OPENGL)
  target_link_libraries(${TESTS_NAME} PRIVATE opengl_rhi)
endif()

target_include_directories(${TESTS_NAME}
  PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/external/spdlog/include
    ${PROJECT_SOURCE_DIR}/external/include
    ${PROJECT_SOURCE_DIR}/external/eigen
    ${PROJECT_SOURCE_DIR}/external/volk
    ${PROJECT_SOURCE_DIR}/external/glad/include
    ${PROJECT_SOURCE_DIR}/external/SPIRV-Tools/include
    ${PROJECT_SOURCE_DIR}/external/Vulkan-Headers/include
  )

target_include_directories(${TESTS_NAME} SYSTEM
  PRIVATE
    ${PROJECT_SOURCE_DIR}/external/VulkanMemoryAllocator/include
  )

if (NOT ANDROID)
  target_include_directories(${TESTS_NAME}
  PRIVATE
    external/glfw/include
  )
endif ()

add_test(NAME ${TESTS_NAME} COMMAND ${TESTS_NAME})
