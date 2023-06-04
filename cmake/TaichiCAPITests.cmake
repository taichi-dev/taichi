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
    if (MSVC AND TI_GENERATE_PDB)
        target_compile_options(${C_API_TESTS_NAME} PRIVATE "$<$<CONFIG:Release>:/Zi>")
        target_link_options(${C_API_TESTS_NAME} PRIVATE "$<$<CONFIG:Release>:/DEBUG>")
        target_link_options(${C_API_TESTS_NAME} PRIVATE "$<$<CONFIG:Release>:/OPT:REF>")
        target_link_options(${C_API_TESTS_NAME} PRIVATE "$<$<CONFIG:Release>:/OPT:ICF>")
    endif()
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

if(LINUX)
    target_link_options(${C_API_TESTS_NAME} PUBLIC -static-libgcc -static-libstdc++)
endif()

if(TI_WITH_STATIC_C_API)
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

    find_package(ZLIB REQUIRED)
    find_library(LIBZSTD_LIBRARY zstd REQUIRED)

    target_link_libraries(${C_STATIC_API_TESTS_NAME} PRIVATE "-framework Cocoa" "-framework IOKit" "-framework CoreFoundation")
    target_link_libraries(${C_STATIC_API_TESTS_NAME} PRIVATE "-framework Metal")
    target_link_libraries(${C_STATIC_API_TESTS_NAME} PRIVATE "${LIBZSTD_LIBRARY}")
    target_link_libraries(${C_STATIC_API_TESTS_NAME} PRIVATE ZLIB::ZLIB)
    target_link_options(${C_STATIC_API_TESTS_NAME} PRIVATE -Wl,-dead_strip)

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
endif()
