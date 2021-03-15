cmake_minimum_required(VERSION 3.0)

set(TESTS_NAME taichi_cpp_tests)
if (WIN32)
  message(WARNING "TODO(#2195): Confirm that googletest works on Windows")
  return()
endif()

# TODO(#2195):
# 1. "cpp" -> "cpp_legacy", "cpp_new" -> "cpp"
# 2. Re-implement the legacy CPP tests using googletest
file(GLOB_RECURSE TAICHI_TESTS_SOURCE "tests/cpp_new/*.cpp")

include_directories(
    ${PROJECT_SOURCE_DIR},
)

add_executable(${TESTS_NAME} ${TAICHI_TESTS_SOURCE})
target_link_libraries(${TESTS_NAME} taichi_testable_lib)
target_link_libraries(${TESTS_NAME} gtest_main)

add_test(NAME ${TESTS_NAME} COMMAND ${TESTS_NAME})
