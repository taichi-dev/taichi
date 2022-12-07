# ./taichi/ui/backends/vulkan/CMakeLists.txt

add_library(taichi_ui_vulkan)
target_sources(taichi_ui_vulkan
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/app_context.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/canvas.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/gui.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/renderable.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/renderer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/swap_chain.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/vertex.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/renderables/circles.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/renderables/lines.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/renderables/mesh.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/renderables/particles.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/renderables/scene_lines.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/renderables/set_image.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/renderables/triangles.cpp
  )

if(NOT ANDROID)
  target_sources(taichi_ui_vulkan PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/window.cpp)
endif()

target_include_directories(taichi_ui_vulkan
  PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/external/glm
    ${PROJECT_SOURCE_DIR}/external/eigen
    ${PROJECT_SOURCE_DIR}/external/volk
    ${PROJECT_SOURCE_DIR}/external/VulkanMemoryAllocator/include
    ${PROJECT_SOURCE_DIR}/external/imgui
    ${PROJECT_SOURCE_DIR}/external/imgui/backends
    ${PROJECT_SOURCE_DIR}/external/Vulkan-Headers/include
    ${LLVM_INCLUDE_DIRS}
  PUBLIC
    ${PROJECT_SOURCE_DIR}/external/glfw/include
  )

target_link_libraries(taichi_ui_vulkan PRIVATE taichi_core)
target_link_libraries(taichi_ui_vulkan PRIVATE taichi_common)
target_link_libraries(taichi_ui_vulkan PRIVATE imgui)

if(TI_WITH_VULKAN)
  target_link_libraries(taichi_ui_vulkan PUBLIC vulkan_rhi)
endif()
