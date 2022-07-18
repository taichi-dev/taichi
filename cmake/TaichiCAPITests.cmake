cmake_minimum_required(VERSION 3.0)

# Note that `taichi_c_api_tests` does not directly link with `taichi_isolated_core` - it links with `libtaichi_c_api.so` instead.
# This is trying to stay synchronized with the customer's build structure, so that test writers aren't allowed to use internal Taichi functions.
# If you have a strong feeling that certain function is neccessary for writing CAPI test cases,
# consider exposing it to the CAPI or reimplement it as test utils.

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

target_link_libraries(${C_API_TESTS_NAME} PRIVATE taichi_c_api)
target_link_libraries(${C_API_TESTS_NAME} PRIVATE gtest_main)

target_include_directories(${C_API_TESTS_NAME}
  PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/c_api/include
    ${PROJECT_SOURCE_DIR}/c_api/tests
  )

add_test(NAME ${C_API_TESTS_NAME} COMMAND ${C_API_TESTS_NAME})
