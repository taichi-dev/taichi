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
file(GLOB_RECURSE TAICHI_TESTS_SOURCE "tests/cpp/common/*.cpp" "tests/cpp/ir/*.cpp")

include_directories(
    ${PROJECT_SOURCE_DIR},
)

add_executable(${TESTS_NAME} ${TAICHI_TESTS_SOURCE})
target_link_libraries(${TESTS_NAME} taichi_isolated_core)
target_link_libraries(${TESTS_NAME} gtest_main)

add_test(NAME ${TESTS_NAME} COMMAND ${TESTS_NAME})
