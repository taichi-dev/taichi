cmake_minimum_required(VERSION 3.0)

if(NOT TI_EMSCRIPTENED)

set(EXAMPLES_NAME taichi_cpp_examples)

file(GLOB_RECURSE TAICHI_EXAMPLES_SOURCE
"cpp_examples/main.cpp"
"cpp_examples/run_snode.cpp"
"cpp_examples/autograd.cpp"
"cpp_examples/aot_save.cpp"
)

include_directories(
    ${PROJECT_SOURCE_DIR},
)

add_executable(${EXAMPLES_NAME} ${TAICHI_EXAMPLES_SOURCE})
if (WIN32)
    # Output the executable to bin/ instead of build/Debug/...
    set(EXAMPLES_OUTPUT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/bin")
    set_target_properties(${EXAMPLES_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${EXAMPLES_OUTPUT_DIR})
    set_target_properties(${EXAMPLES_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY_DEBUG ${EXAMPLES_OUTPUT_DIR})
    set_target_properties(${EXAMPLES_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY_RELEASE ${EXAMPLES_OUTPUT_DIR})
    set_target_properties(${EXAMPLES_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL ${EXAMPLES_OUTPUT_DIR})
    set_target_properties(${EXAMPLES_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO ${EXAMPLES_OUTPUT_DIR})
endif()
target_link_libraries(${EXAMPLES_NAME} PRIVATE taichi_isolated_core)

endif()
