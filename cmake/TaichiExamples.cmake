cmake_minimum_required(VERSION 3.17)

function(add_taichi_example NAME)
set(TARGET_NAME "cpp_examples_${NAME}")
set(SOURCE_FILE "cpp_examples/${NAME}.cpp")
  add_executable(${TARGET_NAME} ${SOURCE_FILE})
  if (WIN32)
    # Output the executable to build/ instead of build/Debug/...
    set(TARGET_OUTPUT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/build")
    set_target_properties(${TARGET_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${TARGET_OUTPUT_DIR})
    set_target_properties(${TARGET_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY_DEBUG ${TARGET_OUTPUT_DIR})
    set_target_properties(${TARGET_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY_RELEASE ${TARGET_OUTPUT_DIR})
    set_target_properties(${TARGET_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL ${TARGET_OUTPUT_DIR})
    set_target_properties(${TARGET_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO ${TARGET_OUTPUT_DIR})
  endif()
  target_link_libraries(${TARGET_NAME} PRIVATE taichi_core)
  target_include_directories(${TARGET_NAME}
    PRIVATE
      ${PROJECT_SOURCE_DIR}
      ${PROJECT_SOURCE_DIR}/external/spdlog/include
      ${PROJECT_SOURCE_DIR}/external/eigen
  )
  if (TI_WITH_METAL)
  target_link_libraries(${TARGET_NAME} PRIVATE
    metal_program_impl
  )
  endif()

  if (TI_WITH_VULKAN OR TI_WITH_OPENGL OR TI_WITH_METAL)
    target_link_libraries(${TARGET_NAME} PRIVATE gfx_runtime)
  endif()

  if (TI_WITH_VULKAN)
    target_link_libraries(${TARGET_NAME} PRIVATE vulkan_rhi)
  endif()

  if (TI_WITH_OPENGL)
    target_link_libraries(${TARGET_NAME} PRIVATE opengl_rhi)
  endif()

  if (TI_WITH_METAL)
    target_link_libraries(${TARGET_NAME} PRIVATE metal_rhi)
  endif()
endfunction()

add_taichi_example(run_snode)
add_taichi_example(autograd)
add_taichi_example(aot_save)
