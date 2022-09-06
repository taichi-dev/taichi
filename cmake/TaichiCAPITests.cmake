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

target_link_libraries(${C_API_TESTS_NAME} PRIVATE taichi_c_api)
target_link_libraries(${C_API_TESTS_NAME} PRIVATE gtest_main)

target_include_directories(${C_API_TESTS_NAME}
  PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/c_api/include
    ${PROJECT_SOURCE_DIR}/c_api/src
    ${PROJECT_SOURCE_DIR}/external/Vulkan-Headers/include
    ${CMAKE_CURRENT_SOURCE_DIR}/external/glad/include
  )

add_test(NAME ${C_API_TESTS_NAME} COMMAND ${C_API_TESTS_NAME})
