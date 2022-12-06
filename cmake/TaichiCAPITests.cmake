cmake_minimum_required(VERSION 3.0)

set(C_API_TESTS_NAME taichi_c_api_tests)
if (WIN32)
    # Prevent overriding the parent project's compiler/linker
    # settings on Windows
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
endif()

# TODO(#2195):
# 1. "cpp" -> "cpp_legacy", "cpp_new" -> "cpp"
# 2. Re-implement the legacy CPP tests using googletest
file(GLOB_RECURSE TAICHI_C_API_TESTS_SOURCE
        "c_api/tests/*.cpp")

add_executable(${C_API_TESTS_NAME} ${TAICHI_C_API_TESTS_SOURCE})
if (WIN32)
    # Output the executable to build/ instead of build/Debug/...
    set(C_API_TESTS_OUTPUT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/build")
    set_target_properties(${C_API_TESTS_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${C_API_TESTS_OUTPUT_DIR})
    set_target_properties(${C_API_TESTS_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY_DEBUG ${C_API_TESTS_OUTPUT_DIR})
    set_target_properties(${C_API_TESTS_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY_RELEASE ${C_API_TESTS_OUTPUT_DIR})
    set_target_properties(${C_API_TESTS_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL ${C_API_TESTS_OUTPUT_DIR})
    set_target_properties(${C_API_TESTS_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO ${C_API_TESTS_OUTPUT_DIR})
endif()
target_link_libraries(${C_API_TESTS_NAME} PRIVATE taichi_c_api)
target_link_libraries(${C_API_TESTS_NAME} PRIVATE taichi_common)
target_link_libraries(${C_API_TESTS_NAME} PRIVATE gtest_main)

if (TI_WITH_BACKTRACE)
    target_link_libraries(${C_API_TESTS_NAME} PRIVATE ${BACKWARD_ENABLE})
endif()

target_include_directories(${C_API_TESTS_NAME}
  PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/c_api/include
    ${PROJECT_SOURCE_DIR}/c_api/src
    ${PROJECT_SOURCE_DIR}/external/Vulkan-Headers/include
    ${CMAKE_CURRENT_SOURCE_DIR}/external/glad/include
    ${CMAKE_CURRENT_SOURCE_DIR}/external/spdlog/include
  )

add_test(NAME ${C_API_TESTS_NAME} COMMAND ${C_API_TESTS_NAME})

if(NOT USE_MOLD)
    target_link_options(${C_API_TESTS_NAME} PRIVATE -Wl,-fuse-ld=lld)
endif()
