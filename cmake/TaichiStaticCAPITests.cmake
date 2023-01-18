cmake_minimum_required(VERSION 3.0)

set(C_STATIC_API_TESTS_NAME taichi_static_c_api_tests)

# TODO(#2195):
# 1. "cpp" -> "cpp_legacy", "cpp_new" -> "cpp"
# 2. Re-implement the legacy CPP tests using googletest
file(GLOB_RECURSE TAICHI_STATIC_C_API_TESTS_SOURCE
        "c_api/tests/*.cpp")

add_executable(${C_STATIC_API_TESTS_NAME} ${TAICHI_STATIC_C_API_TESTS_SOURCE})
add_dependencies(${C_STATIC_API_TESTS_NAME} taichi_static_c_api)

target_link_libraries(${C_STATIC_API_TESTS_NAME} PRIVATE $<TARGET_FILE:taichi_static_c_api>)
target_link_libraries(${C_STATIC_API_TESTS_NAME} PRIVATE taichi_common)
target_link_libraries(${C_STATIC_API_TESTS_NAME} PRIVATE gtest_main)

target_link_libraries(${C_STATIC_API_TESTS_NAME} PRIVATE glfw)
target_link_options(${C_STATIC_API_TESTS_NAME} PRIVATE -Wl,-dead_strip)

if (TI_WITH_BACKTRACE)
    target_link_libraries(${C_STATIC_API_TESTS_NAME} PRIVATE ${BACKWARD_ENABLE})
endif()

target_include_directories(${C_STATIC_API_TESTS_NAME}
  PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/c_api/include
    ${PROJECT_SOURCE_DIR}/c_api/src
    ${PROJECT_SOURCE_DIR}/external/Vulkan-Headers/include
    ${CMAKE_CURRENT_SOURCE_DIR}/external/glad/include
    ${CMAKE_CURRENT_SOURCE_DIR}/external/spdlog/include
  )

add_test(NAME ${C_STATIC_API_TESTS_NAME} COMMAND ${C_STATIC_API_TESTS_NAME})
