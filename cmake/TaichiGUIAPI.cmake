cmake_minimum_required(VERSION 3.0)

set(TAICHI_GUI_API_NAME taichi_gui_api)

file(GLOB_RECURSE GUI_API_SOURCE "c_api/src/gui_utils/*.cpp")

add_library(${TAICHI_GUI_API_NAME} SHARED ${GUI_API_SOURCE})

# TODO(jim19930609): link with "gui" target instead of "taichi_core" target
target_link_libraries(${TAICHI_GUI_API_NAME} PRIVATE taichi_core)
target_link_libraries(${TAICHI_GUI_API_NAME} PRIVATE taichi_c_api)

set(GUI_API_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/build")
set_target_properties(${TAICHI_GUI_API_NAME} PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${GUI_API_OUTPUT_DIRECTORY}
    ARCHIVE_OUTPUT_DIRECTORY ${GUI_API_OUTPUT_DIRECTORY})

target_include_directories(${TAICHI_GUI_API_NAME}
    PRIVATE
        ${PROJECT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/c_api/include
        ${CMAKE_CURRENT_SOURCE_DIR}/external/spdlog/include
        ${CMAKE_CURRENT_SOURCE_DIR}/external/Vulkan-Headers/include
        ${CMAKE_CURRENT_SOURCE_DIR}/external/VulkanMemoryAllocator/include
        ${CMAKE_CURRENT_SOURCE_DIR}/external/SPIRV-Tools/include
        ${CMAKE_CURRENT_SOURCE_DIR}/external/volk
        ${CMAKE_CURRENT_SOURCE_DIR}/external/glfw/include
        ${CMAKE_CURRENT_SOURCE_DIR}/external/glm
        ${CMAKE_CURRENT_SOURCE_DIR}/external/eigen
        ${CMAKE_CURRENT_SOURCE_DIR}/external/imgui
        ${CMAKE_CURRENT_SOURCE_DIR}/external/imgui/backends
    )
