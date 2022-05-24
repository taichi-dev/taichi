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
        "tests/cpp/aot/*.cpp"
        "tests/cpp/backends/*.cpp"
        "tests/cpp/backends/llvm/*.cpp"
        "tests/cpp/codegen/*.cpp"
        "tests/cpp/common/*.cpp"
        "tests/cpp/ir/*.cpp"
        "tests/cpp/llvm/*.cpp"
        "tests/cpp/program/*.cpp"
        "tests/cpp/struct/*.cpp"
        "tests/cpp/transforms/*.cpp")

include_directories(
    ${PROJECT_SOURCE_DIR},
)

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
target_link_libraries(${TESTS_NAME} PRIVATE taichi_isolated_core)
target_link_libraries(${TESTS_NAME} PRIVATE gtest_main)

add_test(NAME ${TESTS_NAME} COMMAND ${TESTS_NAME})
